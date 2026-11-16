"""以只追加的 SQLite 档案保存完整对话、事件和可重建的历史摘要。"""

import json
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any

KINDS = frozenset({"message", "tool_result", "event", "summary"})


class Archive:
    """按任务线程和用户隔离对话档案，不保存或恢复 Agent 任务状态。"""

    def __init__(self, path: Path) -> None:
        """打开档案数据库，首次使用时建立只追加的记录表。

        Args:
            path: SQLite 文件路径；父目录须已存在。

        Raises:
            sqlite3.Error: 数据库无法打开或初始化。
        """
        self.path = Path(path)
        self._lock = RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS archive_entries (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    turn INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    UNIQUE (thread_id, owner_id, event_id)
                );
                CREATE INDEX IF NOT EXISTS archive_scope
                    ON archive_entries(thread_id, owner_id, sequence);
                CREATE TRIGGER IF NOT EXISTS archive_no_update
                    BEFORE UPDATE ON archive_entries
                    BEGIN SELECT RAISE(ABORT, 'archive is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS archive_no_delete
                    BEFORE DELETE ON archive_entries
                    BEGIN SELECT RAISE(ABORT, 'archive is append-only'); END;
                """
            )

    def __enter__(self) -> "Archive":
        """返回当前档案，供 with 语句管理连接。"""
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """关闭数据库连接，不抑制调用方异常。"""
        self.close()

    def close(self) -> None:
        """释放档案数据库连接。"""
        with self._lock:
            self._connection.close()

    def append(
        self,
        thread_id: str,
        owner_id: str,
        event_id: str,
        kind: str,
        payload: dict,
        *,
        turn: int = 0,
    ) -> int:
        """追加完整记录；节点重放使用同一事件标识时返回已有序号。

        Args:
            thread_id: 跨聊天延续的业务任务线程标识。
            owner_id: 当前已验证的业务用户标识。
            event_id: 作用域内稳定且唯一的事件标识。
            kind: message、tool_result、event 或 summary，不接受任务检查点。
            payload: 可序列化为 JSON 的原始记录；消息可包含当时的任务快照，
                仅供回溯，不能作为另一份可修改的任务状态或恢复来源。
            turn: 全线程的用户轮次，0 用于尚未进入对话的事件。

        Returns:
            新记录或完全相同的已有记录的全局递增序号。

        Raises:
            ValueError: 标识、类型或轮次无效，或同标识携带了不同内容。
            TypeError: payload 不是字典或包含非 JSON 值。
            sqlite3.Error: 数据库无法写入。
        """
        with self._lock, self._connection:
            return self._append_record(thread_id, owner_id, event_id, kind, payload, turn)

    def _append_record(
        self, thread_id: str, owner_id: str, event_id: str, kind: str, payload: dict, turn: int
    ) -> int:
        """在调用方已持有的事务中校验并追加记录，不自行提交。

        Args:
            thread_id: 本次记录的任务线程标识。
            owner_id: 当前已验证用户标识。
            event_id: 作用域内稳定的事件标识。
            kind: 支持的档案记录类型。
            payload: 要完整保存的 JSON 字典。
            turn: 记录所属的全线程用户轮次。

        Returns:
            新记录或相同已有记录的序号。

        Raises:
            ValueError: 元数据或内容无效，或同标识已有不同记录。
            TypeError: payload 不是字典或包含非 JSON 值。
            sqlite3.Error: 数据库无法写入。
        """
        _scope(thread_id, owner_id)
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("档案事件标识不能为空")
        if kind not in KINDS:
            raise ValueError("档案只接受消息、工具结果、事件或摘要")
        if type(turn) is not int or turn < 0:
            raise ValueError("档案轮次必须为非负整数")
        if not isinstance(payload, dict):
            raise TypeError("档案内容必须是字典")
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        self._connection.execute(
            "INSERT INTO archive_entries "
            "(thread_id, owner_id, event_id, kind, turn, payload) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(thread_id, owner_id, event_id) DO NOTHING",
            (thread_id, owner_id, event_id, kind, turn, encoded),
        )
        row = self._connection.execute(
            "SELECT sequence, kind, turn, payload FROM archive_entries "
            "WHERE thread_id = ? AND owner_id = ? AND event_id = ?",
            (thread_id, owner_id, event_id),
        ).fetchone()
        if (row["kind"], row["turn"], row["payload"]) != (kind, turn, encoded):
            raise ValueError("同一档案事件标识不能对应不同内容")
        return row["sequence"]

    def append_message(
        self, thread_id: str, owner_id: str, event_id: str, message: dict, *, turn: int
    ) -> int:
        """完整保存一条模型格式的消息，不截断正文或工具调用。

        Args:
            thread_id: 跨聊天延续的业务任务线程标识。
            owner_id: 当前已验证的业务用户标识。
            event_id: 该消息在当前作用域内的稳定标识。
            message: 含 role 的 Chat Completions 消息字典。
            turn: 全线程的用户轮次。

        Returns:
            本消息对应的档案序号。

        Raises:
            ValueError: 消息角色无效或同事件标识的内容不同。
        """
        return self.append_messages(thread_id, owner_id, [(event_id, message)], turn=turn)[0]

    def append_messages(
        self,
        thread_id: str,
        owner_id: str,
        messages: list[tuple[str, dict]],
        *,
        turn: int,
    ) -> list[int]:
        """在同一事务内追加整批消息，工具调用及其结果一起保存或一起回滚。

        Args:
            thread_id: 跨聊天延续的业务任务线程标识。
            owner_id: 当前已验证业务用户标识。
            messages: 按原始顺序排列的事件标识与完整消息二元组列表。
                每条消息须包含有效 role，相同标识只允许完全相同的内容。
            turn: 全批消息所属的用户轮次。

        Returns:
            与输入顺序对应的档案序号；空批次返回空列表。

        Raises:
            ValueError: 作用域、轮次、消息无效或任一标识存在内容冲突；整批回滚。
            TypeError: 消息不能编码为 JSON 字典；整批回滚。
            sqlite3.Error: 任一记录无法写入；整批回滚。
        """
        _scope(thread_id, owner_id)
        if type(turn) is not int or turn < 0:
            raise ValueError("档案轮次必须为非负整数")
        sequences = []
        with self._lock, self._connection:
            for event_id, message in messages:
                if not isinstance(message, dict):
                    raise TypeError("档案消息必须是字典")
                if message.get("role") not in {"user", "assistant", "tool", "system"}:
                    raise ValueError("档案消息须包含有效角色")
                sequences.append(
                    self._append_record(thread_id, owner_id, event_id, "message", message, turn)
                )
        return sequences

    def entries(self, thread_id: str, owner_id: str, *, kind: str | None = None) -> list[dict]:
        """按追加顺序读取指定线程和用户的完整记录。

        Args:
            thread_id: 要读取的业务任务线程标识。
            owner_id: 要读取的已验证业务用户标识。
            kind: 只读取此类型；None 表示读取全部档案记录。

        Returns:
            包含 sequence、event_id、kind、turn、payload 的独立字典列表。
            不存在的作用域返回空列表，其他用户的记录不可见。

        Raises:
            ValueError: 作用域或类型无效。
            sqlite3.Error: 数据库无法读取。
        """
        _scope(thread_id, owner_id)
        query = (
            "SELECT sequence, event_id, kind, turn, payload FROM archive_entries "
            "WHERE thread_id = ? AND owner_id = ?"
        )
        parameters = [thread_id, owner_id]
        if kind is not None:
            if kind not in KINDS:
                raise ValueError("未知档案类型")
            query += " AND kind = ?"
            parameters.append(kind)
        with self._lock:
            rows = self._connection.execute(query + " ORDER BY sequence", parameters).fetchall()
        return [_entry(row) for row in rows]

    def messages(self, thread_id: str, owner_id: str) -> list[dict]:
        """读取指定用户与线程的完整消息档案。

        Args:
            thread_id: 要读取的业务任务线程标识。
            owner_id: 要读取的已验证业务用户标识。

        Returns:
            按追加顺序排列的 message 记录，正文位于各项的 payload。
        """
        return self.entries(thread_id, owner_id, kind="message")

    def get_event(self, thread_id: str, owner_id: str, event_id: str) -> dict | None:
        """按引用读取完整消息、工具结果或其他事件。

        Args:
            thread_id: 引用所属的业务任务线程标识。
            owner_id: 引用所属的已验证业务用户标识。
            event_id: 引用中的事件标识。

        Returns:
            完整档案记录；该用户和线程下不存在此引用时返回 None。

        Raises:
            ValueError: 没有提供有效线程或用户作用域。
            sqlite3.Error: 数据库无法读取。
        """
        _scope(thread_id, owner_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT sequence, event_id, kind, turn, payload FROM archive_entries "
                "WHERE thread_id = ? AND owner_id = ? AND event_id = ?",
                (thread_id, owner_id, event_id),
            ).fetchone()
        return _entry(row) if row is not None else None


def _scope(thread_id: str, owner_id: str) -> None:
    """要求每次档案访问明确给出线程与用户作用域。"""
    if any(not isinstance(value, str) or not value.strip() for value in (thread_id, owner_id)):
        raise ValueError("档案访问必须指定线程和用户")


def _entry(row: sqlite3.Row) -> dict:
    """把数据库行转为包含独立 JSON 正文的档案记录。"""
    result = dict(row)
    result["payload"] = json.loads(result["payload"])
    return result

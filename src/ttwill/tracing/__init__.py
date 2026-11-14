"""会话轨迹、模型原始记录和费用账本。"""

import json
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any
from uuid import uuid4

__all__ = ["Span", "Trace", "render_tree"]

_write_lock = RLock()


def _validate_id(value: str) -> None:
    """校验用于文件名的记录标识，不允许路径分隔符或以点开头。"""
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", value):
        raise ValueError("记录标识只能包含英文字母、数字、下划线、点和连字符，且不能以点开头")


def _json_line(value: dict[str, Any]) -> str:
    """生成带换行的 UTF-8 友好 JSON 文本，拒绝 NaN 和无穷值。"""
    return json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"


def _append(path: Path, line: str) -> None:
    """在进程内写锁保护下追加一条记录，结束后关闭文件。"""
    # 多个会话共用费用账本；一条记录的写入不能被另一线程拆开。
    with _write_lock, path.open("a", encoding="utf-8") as stream:
        stream.write(line)


@dataclass
class Span:
    """一段操作的轨迹状态，由 Trace.span 创建并维护起止时间。

    调用方可更新 attrs 或把 status 设为 blocked；上下文异常退出时状态改为
    error。时间戳使用 UTC，duration_ms 为单调时钟测得的毫秒耗时。
    """

    span_id: str
    parent_id: str | None
    kind: str
    name: str
    turn: int | None
    attrs: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    start: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    end: str | None = None
    duration_ms: float | None = None


class Trace:
    """记录一个会话的 span 树，独立保存模型原文并向运行级账本追加用量。

    父子关系由当前执行上下文维护；不同 Trace 对象互不共享当前 span。
    """

    def __init__(
        self,
        run_dir: Path,
        session_id: str,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
        repeat: int | None = None,
        attempt_id: str | None = None,
    ) -> None:
        """创建新的会话轨迹文件，拒绝覆盖已有记录。

        Args:
            run_dir: 运行产物目录，模型记录与费用账本也保存在这里。
            session_id: 本段聊天的唯一标识，用作会话文件名。
            run_id: 可选的评测运行标识。
            task_id: 可选的评测任务标识，用于关联同一任务的多段聊天。
            repeat: 可选的评测重复序号。
            attempt_id: 可选的本次运行尝试标识，用于区分补跑。

        Raises:
            ValueError: session_id 不是合法文件标识。
            FileExistsError: 该会话的轨迹文件已经存在。
            OSError: 无法创建目录或文件。
        """
        _validate_id(session_id)
        self.run_dir = Path(run_dir)
        self.session_id = session_id
        self.path = self.run_dir / "traces" / f"{session_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 已有会话产物属于一次历史运行，不能被新实例覆盖或拼接。
        with self.path.open("x", encoding="utf-8"):
            pass
        self._association = {
            "session_id": session_id,
            "run_id": run_id,
            "task_id": task_id,
            "repeat": repeat,
            "attempt_id": attempt_id,
        }
        self._current: ContextVar[Span | None] = ContextVar(
            f"ttwill_span_{session_id}", default=None
        )
        self._subscribers: list[Callable[[dict[str, Any]], None]] = []
        self._event_lock = RLock()

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """订阅后续事件；回调同步执行，失败会向调用者抛出。"""
        with self._event_lock:
            self._subscribers.append(callback)

    def _emit(self, event: str, span: Span) -> None:
        """先写入轨迹，再向每位订阅者同步发送独立的事件快照。"""
        line = _json_line(
            {
                **self._association,
                "event": event,
                "span_id": span.span_id,
                "parent_id": span.parent_id,
                "kind": span.kind,
                "name": span.name,
                "turn": span.turn,
                "start": span.start,
                "end": span.end,
                "duration_ms": span.duration_ms,
                "status": "running" if event == "start" else span.status,
                "attrs": span.attrs,
            }
        )
        with self._event_lock:
            _append(self.path, line)
            # 每位订阅者收到独立快照，避免修改事件影响其他订阅者或历史记录。
            for callback in tuple(self._subscribers):
                callback(json.loads(line))

    @contextmanager
    def span(
        self,
        kind: str,
        name: str,
        *,
        turn: int | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        """在上下文进入和退出时记录事件，自动挂到当前父 span 下。

        Args:
            kind: 操作类别，如 session、llm 或 tool。
            name: 操作名称，如模型档位或业务工具名。
            turn: 所属用户轮次；省略时继承父 span 的轮次。
            attrs: 初始属性，须可序列化为 JSON；在结束前可继续更新。

        Yields:
            当前 Span，可补充属性或将状态标记为 blocked。

        操作异常或取消时记录 error 和异常类型，再向上传播；退出时恢复父上下文。
        写入和订阅回调失败同样向上传播，不作静默处理。
        """
        parent = self._current.get()
        span = Span(
            span_id=uuid4().hex,
            parent_id=parent.span_id if parent else None,
            kind=kind,
            name=name,
            turn=turn if turn is not None else parent.turn if parent else None,
            attrs=dict(attrs) if attrs is not None else {},
        )
        token = self._current.set(span)
        started = perf_counter()
        completed = False
        try:
            self._emit("start", span)
            yield span
            completed = True
        finally:
            error = sys.exception() if not completed else None
            self._current.reset(token)
            span.end = datetime.now(UTC).isoformat()
            span.duration_ms = (perf_counter() - started) * 1000
            if error is not None:
                span.status = "error"
                # SDK 异常文本可能带请求头或密钥，只记录异常类型。
                span.attrs["error_type"] = type(error).__name__
            self._emit("end", span)

    def write_model_record(self, call_id: str, record: dict[str, Any]) -> str:
        """保存完整模型输入输出，返回相对于运行目录的文件引用。

        call_id 是本次尝试的唯一标识，可包含重试序号。record 须可序列化为 JSON。
        标识不合法时抛 ValueError，文件已存在时抛 FileExistsError，不覆盖历史记录。
        """
        _validate_id(call_id)
        line = _json_line(record)
        relative = Path("models") / f"{call_id}.json"
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(line)
        return relative.as_posix()

    def record_usage(self, record: dict[str, Any]) -> None:
        """将一次尝试的用量追加到运行级账本，补入当前会话及评测关联字段。

        未知费用和用量可保留为 None；不合并或覆盖其他会话、重试的记录。
        """
        _append(self.run_dir / "usage.jsonl", _json_line({**record, **self._association}))


def _details(span: dict[str, Any]) -> str:
    """格式化模型、用量、费用、耗时及记录引用，供轨迹树每行展示。"""
    attrs = span["attrs"]
    details = []
    for key in ("model", "tier"):
        if key in attrs:
            details.append(f"{key}={attrs[key]}")
    if "input_tokens" in attrs or "output_tokens" in attrs:
        details.append(f"tokens={attrs.get('input_tokens', '?')}/{attrs.get('output_tokens', '?')}")
    if "cost_usd" in attrs:
        cost = attrs["cost_usd"]
        details.append("cost=unknown" if cost is None else f"cost=${cost:.8f}")
    latency = attrs.get("latency_ms", span.get("duration_ms"))
    if latency is not None:
        details.append(f"latency={latency:.1f}ms")
    for key in ("error_type", "reason", "record_ref"):
        if key in attrs:
            details.append(f"{key}={attrs[key]}")
    return " " + " ".join(details) if details else ""


def render_tree(path: Path) -> str:
    """读取会话 JSONL，返回带状态及调用摘要的文本树。

    只有开始事件的 span 显示为 running；找不到父节点的 span 显示为根节点。
    文件读取错误、JSON 错误或缺失必要字段直接抛出，不跳过损坏记录。
    """
    spans: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            spans[event["span_id"]] = event
    children: dict[str | None, list[str]] = defaultdict(list)
    for span_id, span in spans.items():
        parent = span["parent_id"]
        children[parent if parent in spans else None].append(span_id)
    lines = []

    def visit(span_id: str, prefix: str, connector: str, continuation: str) -> None:
        """按记录顺序递归追加节点，并保留树分支的缩进与连线。"""
        span = spans[span_id]
        status = span["status"] if span["event"] == "end" else "running"
        lines.append(f"{prefix}{connector}{span['kind']} {span['name']} [{status}]{_details(span)}")
        descendants = children[span_id]
        for index, child_id in enumerate(descendants):
            last = index == len(descendants) - 1
            visit(
                child_id,
                prefix + continuation,
                "└─ " if last else "├─ ",
                "   " if last else "│  ",
            )

    for root_id in children[None]:
        visit(root_id, "", "", "")
    return "\n".join(lines)

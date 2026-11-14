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
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", value):
        raise ValueError("记录标识只能包含英文字母、数字、下划线、点和连字符，且不能以点开头")


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"


def _append(path: Path, line: str) -> None:
    # 多个会话共用费用账本；一条记录的写入不能被另一线程拆开。
    with _write_lock, path.open("a", encoding="utf-8") as stream:
        stream.write(line)


@dataclass
class Span:
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
        """单独保存完整模型输入输出，返回相对于运行目录的引用。"""
        _validate_id(call_id)
        line = _json_line(record)
        relative = Path("models") / f"{call_id}.json"
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(line)
        return relative.as_posix()

    def record_usage(self, record: dict[str, Any]) -> None:
        """保留每次尝试的费用，不覆盖其他会话或重试记录。"""
        _append(self.run_dir / "usage.jsonl", _json_line({**record, **self._association}))


def _details(span: dict[str, Any]) -> str:
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
    """读取开始 / 结束事件，展示已完成与仍运行中的 span。"""
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

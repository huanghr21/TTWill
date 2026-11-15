"""验证轨迹层级、异常记录、并发隔离和模型用量落盘。"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ttwill.tracing import Trace, render_tree


def read_events(trace: Trace) -> list[dict]:
    """按写入顺序读取轨迹事件。

    Args:
        trace: 提供 JSONL 文件路径的轨迹对象。

    Returns:
        从轨迹文件逐行解析得到的事件列表。
    """
    return [json.loads(line) for line in trace.path.read_text().splitlines()]


def test_spans_publish_snapshots_and_reset_after_error(tmp_path: Path) -> None:
    """事件发布独立快照，异常后恢复父层级，并仅记录异常类型。"""
    trace = Trace(tmp_path, "session-1", run_id="run-1", task_id="task-1", repeat=2)
    observed = []
    trace.subscribe(observed.append)
    trace.subscribe(lambda event: event["attrs"].clear())
    with trace.span("session", "对话") as root:
        with trace.span("turn", "第一轮", turn=1) as turn:
            with pytest.raises(ValueError, match="secret"):
                with trace.span("tool", "查订单", attrs={"input": {"id": 1}}) as tool:
                    tool.attrs["input"]["id"] = 2
                    raise ValueError("secret request headers")
            with trace.span("guard", "权限校验") as guard:
                guard.status = "blocked"
                guard.attrs["reason"] = "身份未验证"
    with trace.span("session", "另一段操作") as after:
        pass

    events = read_events(trace)
    assert observed == events
    assert tool.parent_id == turn.span_id
    assert guard.parent_id == turn.span_id
    assert turn.parent_id == root.span_id
    assert after.parent_id is None
    assert tool.turn == guard.turn == 1
    assert events[2]["attrs"]["input"] == {"id": 1}
    assert events[3]["attrs"]["input"] == {"id": 2}
    assert events[3]["status"] == "error"
    assert events[3]["attrs"]["error_type"] == "ValueError"
    assert "secret" not in trace.path.read_text()
    assert all(event["task_id"] == "task-1" and event["repeat"] == 2 for event in events)
    assert "[blocked]" in render_tree(trace.path)


def test_handled_exception_is_not_reported_for_successful_span(tmp_path: Path) -> None:
    """已被外层捕获的异常不影响成功 span 的状态。"""
    trace = Trace(tmp_path, "handled")
    try:
        raise ValueError("已经捕获")
    except ValueError:
        with trace.span("tool", "成功"):
            pass
    assert read_events(trace)[-1]["status"] == "ok"


def test_async_siblings_and_other_traces_do_not_share_parents(tmp_path: Path) -> None:
    """并发兄弟任务与不同轨迹分别保留自己的父层级。"""
    trace = Trace(tmp_path, "async")
    other = Trace(tmp_path, "other")

    async def run() -> None:
        """驱动当前测试场景中的并发任务与轨迹上下文。"""
        ready = asyncio.Event()
        entered = 0

        async def worker(name: str) -> None:
            """执行当前测试的异步操作，供父任务检查层级或取消行为。"""
            nonlocal entered
            with trace.span("subagent", name):
                entered += 1
                if entered == 2:
                    ready.set()
                await ready.wait()
                await asyncio.sleep(0)
                with trace.span("tool", f"{name}-tool"):
                    pass
                with other.span("tool", f"{name}-isolated"):
                    pass

        with trace.span("session", "并发对话", turn=3):
            await asyncio.gather(worker("a"), worker("b"))

    asyncio.run(run())
    starts = {event["name"]: event for event in read_events(trace) if event["event"] == "start"}
    assert starts["a"]["parent_id"] == starts["b"]["parent_id"] == starts["并发对话"]["span_id"]
    for name in ("a", "b"):
        assert starts[f"{name}-tool"]["parent_id"] == starts[name]["span_id"]
        assert starts[f"{name}-tool"]["turn"] == 3
    assert all(event["parent_id"] is None for event in read_events(other))


def test_cancellation_is_recorded_and_context_restored(tmp_path: Path) -> None:
    """任务取消写入异常类型，并为后续操作恢复轨迹上下文。"""
    trace = Trace(tmp_path, "cancelled")

    async def run() -> None:
        """驱动当前测试场景中的并发任务与轨迹上下文。"""
        entered = asyncio.Event()

        async def worker() -> None:
            """执行当前测试的异步操作，供父任务检查层级或取消行为。"""
            with trace.span("llm", "等待响应"):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(worker())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with trace.span("tool", "后续"):
            pass

    asyncio.run(run())
    events = read_events(trace)
    assert events[1]["status"] == "error"
    assert events[1]["attrs"]["error_type"] == "CancelledError"
    assert events[2]["parent_id"] is None


def test_model_records_and_usage_keep_all_attempts(tmp_path: Path) -> None:
    """原始模型记录禁止覆盖，并发追加的用量记录完整保留。"""
    trace = Trace(tmp_path, "billing", run_id="run", attempt_id="attempt")
    record = {"messages": [{"role": "user", "content": "完整输入" * 1000}], "output": "完整输出"}
    reference = trace.write_model_record("call-1", record)
    assert reference == "models/call-1.json"
    assert json.loads((tmp_path / reference).read_text()) == record
    with pytest.raises(FileExistsError):
        trace.write_model_record("call-1", {"output": "不能覆盖"})

    other = Trace(tmp_path, "billing-other", attempt_id="other")

    def append(index: int) -> None:
        """按索引选择轨迹并追加一条用量记录。"""
        owner = trace if index % 2 == 0 else other
        owner.record_usage({"call_id": f"call-{index}", "cost_usd": index / 1000})

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(append, range(30)))
    usage = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert len(usage) == 30
    assert {entry["call_id"] for entry in usage} == {f"call-{index}" for index in range(30)}
    assert {entry["attempt_id"] for entry in usage} == {"attempt", "other"}


def test_render_tree_shows_running_and_model_details(tmp_path: Path) -> None:
    """树形轨迹呈现运行状态、模型信息、用量和原始记录位置。"""
    trace = Trace(tmp_path, "render")
    with trace.span("session", "测试"):
        with trace.span(
            "llm",
            "回复",
            attrs={
                "model": "fake",
                "tier": "flash",
                "input_tokens": 10,
                "output_tokens": 2,
                "cost_usd": 0.002,
                "record_ref": "models/call-1.json",
            },
        ):
            running = render_tree(trace.path)
    assert "└─ llm 回复 [running]" in running
    result = render_tree(trace.path)
    assert "└─ llm 回复 [ok]" in result
    assert "model=fake" in result and "tier=flash" in result
    assert "tokens=10/2" in result and "cost=$0.00200000" in result
    assert "latency=" in result and "record_ref=models/call-1.json" in result


@pytest.mark.parametrize("unsafe", ["../escape", "a/b", "a\\b", ".", "..", "", "/absolute"])
def test_identifiers_cannot_escape_run_dir(tmp_path: Path, unsafe: str) -> None:
    """会话与调用标识不能构造越出运行目录的文件路径。"""
    with pytest.raises(ValueError):
        Trace(tmp_path, unsafe)
    trace = Trace(tmp_path, "safe")
    with pytest.raises(ValueError):
        trace.write_model_record(unsafe, {})


def test_existing_session_is_not_overwritten(tmp_path: Path) -> None:
    """重复会话标识报错，并保留原轨迹文件内容。"""
    trace = Trace(tmp_path, "existing")
    with trace.span("session", "原记录"):
        pass
    original = trace.path.read_bytes()
    with pytest.raises(FileExistsError):
        Trace(tmp_path, "existing")
    assert trace.path.read_bytes() == original

"""从落盘产物核对完整场景的轨迹、模型账本和业务授权链。"""

import json
from collections import Counter
from pathlib import Path


def assert_trace_integrity(run_dir: Path, turns: list[dict], expected_task_id: str) -> None:
    """验证每段聊天的 span 树，并对账模型费用及展示、确认、提交和核验。

    Args:
        run_dir: 单个验收场景的运行目录，包含 traces、models 与 usage.jsonl。
        turns: 按执行顺序保存的用户轮次，每项含 session_id、全线程 turn、
            user_message 和运行时完整 result；调用前须关闭 session span。
        expected_task_id: Trace 顶层 task_id 的场景标识，不是子任务 ID。
    """
    events_by_session = {}
    ended = {}
    positions = {}
    for path in sorted((run_dir / "traces").glob("*.jsonl")):
        events = _read_jsonl(path)
        assert events, path
        session_id = path.stem
        assert {event["session_id"] for event in events} == {session_id}
        assert {event["task_id"] for event in events} == {expected_task_id}
        for key in ("run_id", "repeat", "attempt_id"):
            assert len({event[key] for event in events}) == 1
            assert events[0][key] is not None
        starts = [event for event in events if event["event"] == "start"]
        ends = [event for event in events if event["event"] == "end"]
        assert len(starts) + len(ends) == len(events)
        counts = Counter(event["span_id"] for event in starts)
        assert counts and set(counts.values()) == {1}
        assert counts == Counter(event["span_id"] for event in ends)
        start_by_id = {event["span_id"]: event for event in starts}
        end_by_id = {event["span_id"]: event for event in ends}
        index = {(event["span_id"], event["event"]): i for i, event in enumerate(events)}
        roots = [event for event in ends if event["parent_id"] is None]
        assert len(roots) == 1 and roots[0]["kind"] == "session"
        for span_id, start in start_by_id.items():
            end = end_by_id[span_id]
            assert index[span_id, "start"] < index[span_id, "end"]
            assert end["end"] >= start["start"]
            assert end["duration_ms"] >= 0
            assert end["status"] in {"ok", "blocked", "error"}
            for key in ("parent_id", "kind", "name", "turn", "start"):
                assert start[key] == end[key]
            parent_id = start["parent_id"]
            if parent_id is not None:
                assert parent_id in start_by_id
                assert index[parent_id, "start"] < index[span_id, "start"]
                assert index[span_id, "end"] < index[parent_id, "end"]
                parent = start_by_id[parent_id]
                if parent["turn"] is not None:
                    assert start["turn"] == parent["turn"]
            if start["kind"] == "dispatch":
                assert start["attrs"]["instruction"]
                assert start["attrs"]["task_id"]
                assert start["attrs"]["revision"] >= 1
        events_by_session[session_id] = ends
        ended.update(end_by_id)
        positions.update({(session_id, *key): value for key, value in index.items()})

    assert set(events_by_session) == {turn["session_id"] for turn in turns}
    associations = {
        (event["run_id"], event["task_id"], event["repeat"], event["attempt_id"])
        for event in ended.values()
    }
    assert len(associations) == 1
    _assert_model_records(run_dir, ended)

    outputs = {(turn["session_id"], turn["turn"]): turn for turn in turns}
    assert len(outputs) == len(turns)
    committed = []
    for session_id, spans in events_by_session.items():
        identities = [
            span
            for span in spans
            if span["kind"] == "tool" and span["name"].startswith("find_user_id_by_")
        ]
        recoveries = [span for span in spans if span["kind"] == "recovery"]
        assert identities and recoveries
        assert (
            positions[session_id, identities[0]["span_id"], "end"]
            < positions[session_id, recoveries[0]["span_id"], "start"]
        )
        for span in spans:
            key = (session_id, span["turn"])
            if span["kind"] == "reply":
                result = outputs[key]["result"]
                assert span["attrs"]["content"] == result["reply"]
                assert span["attrs"]["displayed"] == result["displayed"]
                for shown in result["displayed"]:
                    assert shown["session_id"] == session_id and shown["turn"] == span["turn"]
                    assert shown["content"] in result["reply"]
            if span["kind"] == "state_snapshot":
                assert span["attrs"]["tasks"] == outputs[key]["result"]["tasks"]
            if span["kind"] == "confirmation":
                assert span["attrs"]["user_message"] == outputs[key]["user_message"]
                previous = outputs[session_id, span["turn"] - 1]["result"]["displayed"]
                assert span["attrs"]["displayed"] == [
                    {"operation_id": shown["operation_id"], "content": shown["content"]}
                    for shown in previous
                ]
            if span["kind"] == "tool" and span["attrs"].get("kind") == "commit":
                _assert_commit(span, spans, outputs, positions)
                if span["attrs"]["status"] == "committed":
                    committed.append(span["attrs"]["operation_id"])
    assert len(committed) == len(set(committed)), "同一操作不得重复提交"
    for key in outputs:
        spans = [span for span in events_by_session[key[0]] if span["turn"] == key[1]]
        kinds = Counter(span["kind"] for span in spans)
        assert kinds["turn"] == kinds["coordinator"] == kinds["reply"] == 1
        assert kinds["state_snapshot"] == 1
        assert kinds["context"] >= 1


def _read_jsonl(path: Path) -> list[dict]:
    """读取验收过程中实际落盘的逐行 JSON 记录。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _assert_model_records(run_dir: Path, spans: dict[str, dict]) -> None:
    """核对每次模型尝试与费用账本、原始输入输出文件一一对应。

    Args:
        run_dir: 包含模型原文和费用账本的场景目录。
        spans: 按标识索引的全部已结束 span。
    """
    calls = {span_id: span for span_id, span in spans.items() if span["kind"] == "llm"}
    usage = _read_jsonl(run_dir / "usage.jsonl")
    assert calls
    assert Counter(entry["span_id"] for entry in usage) == Counter(calls.keys())
    references = set()
    for entry in usage:
        span = calls[entry["span_id"]]
        attrs = span["attrs"]
        for key in ("session_id", "run_id", "task_id", "repeat", "attempt_id"):
            assert entry[key] == span[key]
        for key in (
            "call_id",
            "attempt",
            "span_id",
            "tier",
            "requested_model",
            "model",
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "cost_usd",
            "latency_ms",
            "record_ref",
            "error",
        ):
            assert entry[key] == attrs[key]
        assert entry["status"] == span["status"]
        reference = entry["record_ref"]
        assert reference not in references
        references.add(reference)
        record = json.loads((run_dir / reference).read_text(encoding="utf-8"))
        for key in ("call_id", "attempt", "tier", "error"):
            assert record[key] == entry[key]
        assert record["request"]["model"] == entry["requested_model"]
        assert record["request"]["messages"]
        if record["response"] is not None:
            assert record["response"].get("model", entry["requested_model"]) == entry["model"]
        assert all(
            tool["function"]["name"] != "commit" for tool in record["request"].get("tools", [])
        )
        for key in ("input_tokens", "output_tokens", "cached_input_tokens", "cost_usd"):
            assert entry[key] is None or entry[key] >= 0
    assert references == {
        path.relative_to(run_dir).as_posix() for path in (run_dir / "models").glob("*.json")
    }


def _assert_commit(commit: dict, spans: list[dict], outputs: dict, positions: dict) -> None:
    """逐项追溯提交授权、上轮完整展示与随后发生的真实业务核验。

    Args:
        commit: 一次已结束的代码提交 span。
        spans: 同一聊天中的全部已结束 span。
        outputs: 按会话及全线程轮次索引的真实原话与运行结果。
        positions: 按会话、span 标识和事件类别索引的文件内事件位置。
    """
    session_id, turn = commit["session_id"], commit["turn"]
    attrs = commit["attrs"]
    operation_id = attrs["operation_id"]
    previous = outputs[session_id, turn - 1]["result"]["displayed"]
    shown = [item for item in previous if item["operation_id"] == operation_id]
    assert len(shown) == 1
    shown = shown[0]
    assert attrs["args"] == shown["args"]
    assert attrs["task_id"] == shown["task_id"]
    assert attrs["revision"] == shown["revision"]
    confirmations = [
        span
        for span in spans
        if span["kind"] == "confirmation"
        and span["turn"] == turn
        and any(
            item["operation_id"] == operation_id and item["decision"] == "confirm"
            for item in span["attrs"]["results"]
        )
    ]
    assert len(confirmations) == 1
    assert (
        positions[session_id, confirmations[0]["span_id"], "end"]
        < positions[session_id, commit["span_id"], "start"]
    )
    if attrs["status"] != "committed":
        return
    operation = attrs["result"]
    assert operation["args"] == shown["args"]
    assert operation["preview"] == shown["preview"]
    verifications = [
        span
        for span in spans
        if span["kind"] == "verification"
        and span["turn"] == turn
        and span["attrs"]["operation_id"] == operation_id
    ]
    assert len(verifications) == 1
    verification = verifications[0]
    assert (
        positions[session_id, commit["span_id"], "end"]
        < positions[session_id, verification["span_id"], "start"]
    )
    assert verification["attrs"]["task_id"] == shown["task_id"]
    assert verification["attrs"]["revision"] == shown["revision"]
    assert any(
        span["parent_id"] == verification["span_id"]
        and span["attrs"].get("kind") == "verification_read"
        for span in spans
    )
    task = next(
        task
        for task in outputs[session_id, turn]["result"]["tasks"]
        if task["id"] == shown["task_id"]
    )
    assert (task["status"] == "done") == verification["attrs"]["result"]["verified"]

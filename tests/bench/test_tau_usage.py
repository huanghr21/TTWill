"""用离线 LiteLLM 请求验证官方模型调用的轨迹、失败账本和凭据隔离。"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

litellm = pytest.importorskip("litellm", reason="安装 bench 额外依赖后测试官方调用账本")

from openai import OpenAI  # noqa: E402

from ttwill.bench.usage import TauUsageCallback  # noqa: E402
from ttwill.config import ModelConfig  # noqa: E402
from ttwill.llm.types import LLMError  # noqa: E402
from ttwill.tracing import Trace  # noqa: E402


def callback_at(
    path: Path, role: str = "user_simulator", *, priced: bool = True
) -> TauUsageCallback:
    """创建具备完整评测关联字段的回调。

    Args:
        path: 独立测试运行目录。
        role: 官方模型调用的角色。
        priced: 是否配置输入、输出和缓存的每百万 token 价格。

    Returns:
        绑定测试模型及会话轨迹的回调。
    """
    trace = Trace(path, role, run_id="run", task_id="retail-1", repeat=2, attempt_id="sample-a")
    model = ModelConfig(
        model="openai/local-test",
        input_cost_per_million=2 if priced else None,
        output_cost_per_million=4 if priced else None,
        cached_input_cost_per_million=1 if priced else None,
    )
    return TauUsageCallback(trace, role=role, model=model)


def response_body() -> dict:
    """构造保留缓存 token、工具调用和扩展字段的离线完整响应。"""
    return {
        "id": "response-test",
        "object": "chat.completion",
        "created": 1,
        "model": "local-test-v1",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "完整模型输出",
                    "reasoning_content": "推理",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "total_tokens": 25,
            "prompt_tokens_details": {"cached_tokens": 4},
        },
    }


def usage_at(path: Path) -> list[dict]:
    """按落盘顺序读取本次测试的调用账本。"""
    return [json.loads(line) for line in (path / "usage.jsonl").read_text().splitlines()]


def test_real_litellm_callback_keeps_full_records_and_parent_span(tmp_path: Path) -> None:
    """真实 LiteLLM 和 SDK 经 MockTransport 返回，回调完整落盘且不保存凭据。"""
    callback = callback_at(tmp_path)
    requests = []

    def transport(request: httpx.Request) -> httpx.Response:
        """接收 SDK 请求并返回内存响应，不访问网络。"""
        requests.append(request)
        return httpx.Response(200, json=response_body())

    client = OpenAI(
        api_key="secret-must-not-be-logged",
        base_url="https://unit.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    messages = [{"role": "user", "content": "完整输入" * 2000}]
    try:
        with callback.trace.span("session", "评测") as parent:
            with callback.capture():
                litellm.completion(
                    model=callback.model.model,
                    messages=messages,
                    client=client,
                    temperature=0.1,
                    **callback.request_kwargs(),
                )
    finally:
        client.close()
    assert len(requests) == 1
    usage = usage_at(tmp_path)
    assert len(usage) == 1
    entry = usage[0]
    assert entry["role"] == "user_simulator"
    assert (entry["run_id"], entry["task_id"], entry["repeat"], entry["attempt_id"]) == (
        "run",
        "retail-1",
        2,
        "sample-a",
    )
    assert (entry["input_tokens"], entry["output_tokens"], entry["cached_input_tokens"]) == (
        20,
        5,
        4,
    )
    assert entry["cost_usd"] == pytest.approx(0.000056)
    record = json.loads((tmp_path / entry["record_ref"]).read_text())
    assert record["request"]["messages"] == messages
    assert record["request"]["temperature"] == 0.1
    assert record["response"]["choices"][0]["message"]["reasoning_content"] == "推理"
    events = [json.loads(line) for line in callback.trace.path.read_text().splitlines()]
    llm = [event for event in events if event["kind"] == "llm"]
    assert [event["event"] for event in llm] == ["start", "end"]
    assert all(event["parent_id"] == parent.span_id for event in llm)
    assert events[-1]["event"] == "end" and events[-1]["span_id"] == parent.span_id
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert "secret-must-not-be-logged" not in path.read_text()


def test_failure_and_explicit_retry_both_reach_ledger_without_internal_retry(
    tmp_path: Path,
) -> None:
    """关闭内部重试后保留首次限流及外层重试成功，失败不假记零费用。"""
    callback = callback_at(tmp_path, "evaluator")
    requests = []

    def transport(request: httpx.Request) -> httpx.Response:
        """首个请求限流，第二个请求成功，记录实际传输次数。"""
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, json={"error": {"message": "限流", "type": "rate_limit"}})
        return httpx.Response(200, json=response_body())

    client = OpenAI(
        api_key="test-key",
        base_url="https://unit.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    try:
        with pytest.raises(litellm.RateLimitError):
            with callback.capture():
                litellm.completion(
                    model=callback.model.model,
                    messages=[{"role": "user", "content": "判分"}],
                    client=client,
                    **callback.request_kwargs(),
                )
        assert len(requests) == 1
        with callback.capture():
            litellm.completion(
                model=callback.model.model,
                messages=[{"role": "user", "content": "判分"}],
                client=client,
                **callback.request_kwargs(),
            )
    finally:
        client.close()
    usage = usage_at(tmp_path)
    assert len(requests) == len(usage) == 2
    assert [entry["status"] for entry in usage] == ["error", "ok"]
    assert usage[0]["error"]["status_code"] == 429
    assert usage[0]["cost_usd"] is None and usage[0]["input_tokens"] is None
    assert usage[0]["call_id"] != usage[1]["call_id"]
    assert all(entry["role"] == "evaluator" for entry in usage)


def test_multiple_calls_in_one_official_operation_have_distinct_attempts(tmp_path: Path) -> None:
    """官方结构校验重试的每次调用共用关联号并各自保留原文与用量。"""
    callback = callback_at(tmp_path, priced=False)
    with callback.capture():
        for content in ("首次无效回答", "修正后的回答"):
            body = response_body()
            body["choices"][0]["message"]["content"] = content
            litellm.completion(
                model=callback.model.model,
                messages=[{"role": "user", "content": "判分"}],
                mock_response=body,
                **callback.request_kwargs(),
            )
    usage = usage_at(tmp_path)
    assert sorted(entry["attempt"] for entry in usage) == [1, 2]
    assert len({entry["call_id"] for entry in usage}) == 1
    assert all(entry["cost_usd"] is None for entry in usage)
    assert len({entry["record_ref"] for entry in usage}) == 2


def test_background_recording_error_is_raised_to_driver(tmp_path: Path, monkeypatch) -> None:
    """LiteLLM 吞掉后台回调错误时，capture 仍向驱动器报告账本写入失败。"""
    callback = callback_at(tmp_path)

    def fail_usage(record: dict) -> None:
        """模拟费用账本写入失败。"""
        raise OSError("账本不可写")

    monkeypatch.setattr(callback.trace, "record_usage", fail_usage)
    with pytest.raises(OSError, match="账本不可写"):
        with callback.capture():
            litellm.completion(
                model=callback.model.model,
                messages=[{"role": "user", "content": "请求"}],
                mock_response=response_body(),
                **callback.request_kwargs(),
            )
    assert callback not in litellm.input_callback


def test_pre_request_failure_keeps_safe_input_and_exception_type(tmp_path: Path) -> None:
    """输入回调之前失败也保留请求，参数里的密钥和异常原文均不落盘。"""
    callback = callback_at(tmp_path)
    start = datetime.now(UTC)
    kwargs = {
        "litellm_call_id": "validation-error",
        "model": "unknown",
        "messages": [{"role": "user", "content": "真实输入"}],
        "optional_params": {
            "api_key": "secret",
            "extra_headers": {"Authorization": "secret"},
            "temperature": 0.2,
            "extra_body": {"api_key": "secret", "safe": True},
        },
        "exception": ValueError("secret"),
        "response_cost": 0.0,
    }
    with callback.capture():
        callback.log_failure_event(kwargs, None, start, start + timedelta(milliseconds=5))
    entry = usage_at(tmp_path)[0]
    assert entry["latency_ms"] == 5
    assert entry["error"] == {"type": "ValueError"}
    raw = (tmp_path / entry["record_ref"]).read_text()
    assert "secret" not in raw
    record = json.loads(raw)
    assert record["request"]["extra_body"] == {"safe": True}
    assert record["request"]["messages"] == kwargs["messages"]


def test_other_callbacks_and_nested_capture_do_not_pollute_run(tmp_path: Path) -> None:
    """输入回调不接收其他请求，重入报错且原作用域可以正常收尾。"""
    callback = callback_at(tmp_path)
    with callback.capture():
        callback.log_pre_api_call("unrelated", [], {"litellm_params": {"metadata": {}}})
        with pytest.raises(RuntimeError, match="同时捕获"):
            with callback.capture():
                pass
    assert not (tmp_path / "usage.jsonl").exists()
    assert callback not in litellm.input_callback


def test_invalid_usage_preserves_raw_response_and_marks_unknown_cost(tmp_path: Path) -> None:
    """无效用量不能导致一次已发生调用从原文和账本中消失。"""
    callback = callback_at(tmp_path)
    body = response_body()
    body["usage"]["prompt_tokens"] = -1
    with pytest.raises(LLMError, match="非负整数"):
        with callback.capture():
            litellm.completion(
                model=callback.model.model,
                messages=[{"role": "user", "content": "请求"}],
                mock_response=body,
                **callback.request_kwargs(),
            )
    entry = usage_at(tmp_path)[0]
    record = json.loads((tmp_path / entry["record_ref"]).read_text())
    assert record["response"]["usage"]["prompt_tokens"] == -1
    assert entry["status"] == "error" and entry["cost_usd"] is None
    assert entry["error"] == {"type": "LLMError"}

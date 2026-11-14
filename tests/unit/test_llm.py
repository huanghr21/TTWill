import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ttwill.config import AppConfig, LLMConfig, ModelConfig
from ttwill.llm import CompatibleModel, FakeModel, LLMClient, LLMError
from ttwill.tracing import Trace


def make_config(
    tmp_path: Path,
    *,
    max_retries: int = 2,
    timeout_seconds: float = 1,
    model: ModelConfig | None = None,
) -> AppConfig:
    return AppConfig(
        root=tmp_path,
        runs_dir=tmp_path / "runs",
        llm=LLMConfig(timeout_seconds, max_retries, 0),
        models={
            "flash": model
            or ModelConfig(
                model="requested-model",
                base_url="https://model.invalid/v1/",
                api_key_env="TTWILL_TEST_KEY",
                input_cost_per_million=1,
                output_cost_per_million=2,
                cached_input_cost_per_million=0.1,
                parameters={"temperature": 0.2},
            )
        },
    )


def model_response(*, usage: bool = True) -> dict[str, Any]:
    raw = {
        "id": "completion-1",
        "model": "actual-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": "订单尚未发货。"},
                "finish_reason": "stop",
            }
        ],
    }
    if usage:
        raw["usage"] = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 40},
        }
    return raw


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_usage(trace: Trace) -> list[dict[str, Any]]:
    return read_jsonl(trace.run_dir / "usage.jsonl")


def read_model_records(trace: Trace) -> list[dict[str, Any]]:
    return [
        json.loads((trace.run_dir / row["record_ref"]).read_text(encoding="utf-8"))
        for row in read_usage(trace)
    ]


def write_env(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text("TTWILL_TEST_KEY=file-secret\n", encoding="utf-8")
    return path


def test_refusal_is_a_valid_response_and_is_preserved(tmp_path):
    config = make_config(tmp_path)
    trace = Trace(config.runs_dir, "refusal")
    raw = model_response()
    raw["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "refusal": "无法处理这一请求。",
    }
    fake = FakeModel([raw])
    response = asyncio.run(
        LLMClient(config, trace, fake).complete("flash", [{"role": "user", "content": "请求"}])
    )
    assert response.message["refusal"] == "无法处理这一请求。"
    assert response.message["content"] is None
    assert read_usage(trace)[0]["status"] == "ok"
    assert read_model_records(trace)[0]["response"] == raw


def test_compatible_request_and_full_response_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("TTWILL_TEST_KEY", "environment-secret")
    config = make_config(tmp_path)
    trace = Trace(config.runs_dir, "request", run_id="run", task_id="task", repeat=2)
    messages = [{"role": "user", "content": "完整消息不应被截断。" * 1000}]
    tool = {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "查询订单",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
    raw = model_response()
    raw["choices"][0] = {
        "message": {
            "role": "assistant",
            "content": None,
            "reasoning_content": "先查询状态。" * 1000,
            "tool_calls": [
                {
                    "id": "tool-1",
                    "type": "function",
                    "function": {"name": "get_order", "arguments": '{"order_id":"W123"}'},
                }
            ],
        },
        "finish_reason": "tool_calls",
    }
    raw["vendor_extension"] = {"trace": [1, 2, 3]}
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=raw)

    async def run():
        async with CompatibleModel(
            write_env(tmp_path), transport=httpx.MockTransport(handler)
        ) as backend:
            return await LLMClient(config, trace, backend).complete("flash", messages, tools=[tool])

    result = asyncio.run(run())

    assert result.raw == raw
    assert result.message == raw["choices"][0]["message"]
    assert result.finish_reason == "tool_calls"
    assert result.model == "actual-model"
    assert len(requests) == 1
    assert str(requests[0].url) == "https://model.invalid/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer file-secret"
    expected = {
        "model": "requested-model",
        "messages": messages,
        "tools": [tool],
        "temperature": 0.2,
        "stream": False,
    }
    assert json.loads(requests[0].content) == expected
    ledger = read_usage(trace)
    assert len(ledger) == 1
    row = ledger[0]
    assert row["requested_model"] == "requested-model"
    assert row["model"] == "actual-model"
    assert row["usage"] == {"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 40}
    assert row["cost_usd"] == pytest.approx(0.000104)
    assert row["latency_ms"] >= 0
    assert row["session_id"] == "request" and row["task_id"] == "task" and row["repeat"] == 2
    record = read_model_records(trace)[0]
    assert record["request"] == expected
    assert record["response"] == raw
    assert record["error"] is None
    span = read_jsonl(trace.path)[-1]
    assert span["attrs"]["record_ref"] == row["record_ref"]
    for public_path in [trace.path, trace.run_dir / "usage.jsonl"]:
        public_text = public_path.read_text(encoding="utf-8")
        assert messages[0]["content"] not in public_text
        assert "reasoning_content" not in public_text
        assert "tool_calls" not in public_text
    for path in trace.run_dir.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert "file-secret" not in text
            assert "environment-secret" not in text


def test_retryable_http_errors_keep_each_attempt_and_span_parent(tmp_path):
    config = make_config(tmp_path)
    trace = Trace(config.runs_dir, "retries")
    statuses = iter([429, 503, 200])
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        status = next(statuses)
        return httpx.Response(
            status,
            json=model_response() if status == 200 else {"error": "sensitive-service-body"},
        )

    async def run():
        async with CompatibleModel(
            write_env(tmp_path), transport=httpx.MockTransport(handler)
        ) as backend:
            with trace.span("turn", "本轮", turn=3) as parent:
                await LLMClient(config, trace, backend).complete(
                    "flash", [{"role": "user", "content": "查询订单"}]
                )
            return parent.span_id

    parent_id = asyncio.run(run())
    ledger = read_usage(trace)
    assert len(requests) == 3 and requests[0] == requests[1] == requests[2]
    assert [row["attempt"] for row in ledger] == [1, 2, 3]
    assert len({row["call_id"] for row in ledger}) == 1
    assert len({row["record_ref"] for row in ledger}) == 3
    assert [row["status"] for row in ledger] == ["error", "error", "ok"]
    assert [row["error"]["status_code"] for row in ledger[:2]] == [429, 503]
    assert all(row["cost_usd"] is None and row["usage"] is None for row in ledger[:2])
    ends = [
        event
        for event in read_jsonl(trace.path)
        if event["kind"] == "llm" and event["event"] == "end"
    ]
    assert len(ends) == 3
    assert all(event["parent_id"] == parent_id and event["turn"] == 3 for event in ends)
    records = read_model_records(trace)
    assert [record["response"] for record in records[:2]] == [None, None]
    assert "sensitive-service-body" not in json.dumps(records)


def test_unauthorized_does_not_retry_and_next_span_keeps_parent(tmp_path):
    config = make_config(tmp_path)
    trace = Trace(config.runs_dir, "unauthorized")
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="file-secret is invalid")

    async def run():
        async with CompatibleModel(
            write_env(tmp_path), transport=httpx.MockTransport(handler)
        ) as backend:
            with trace.span("turn", "本轮") as parent:
                with pytest.raises(LLMError, match="HTTP 401") as error:
                    await LLMClient(config, trace, backend).complete(
                        "flash", [{"role": "user", "content": "查询订单"}]
                    )
                with trace.span("guard", "失败后继续") as next_span:
                    pass
                assert "file-secret" not in str(error.value)
            with trace.span("session", "下一段") as next_session:
                pass
            return parent, next_span, next_session

    parent, next_span, next_session = asyncio.run(run())
    assert calls == 1
    assert len(read_usage(trace)) == 1
    assert next_span.parent_id == parent.span_id
    assert next_session.parent_id is None


def test_transport_error_is_retried_without_recording_exception_text(tmp_path):
    config = make_config(tmp_path)
    trace = Trace(config.runs_dir, "transport")
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("sensitive-transport-detail", request=request)
        return httpx.Response(200, json=model_response())

    async def run():
        async with CompatibleModel(
            write_env(tmp_path), transport=httpx.MockTransport(handler)
        ) as backend:
            return await LLMClient(config, trace, backend).complete(
                "flash", [{"role": "user", "content": "你好"}]
            )

    assert asyncio.run(run()).message["content"] == "订单尚未发货。"
    assert calls == 2
    assert [row["status"] for row in read_usage(trace)] == ["error", "ok"]
    assert "sensitive-transport-detail" not in json.dumps(read_model_records(trace))


def test_total_attempt_timeout_is_bounded_and_each_retry_is_recorded(tmp_path):
    config = make_config(tmp_path, timeout_seconds=0.01, max_retries=1)
    trace = Trace(config.runs_dir, "timeout")
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()

    async def run():
        async with CompatibleModel(
            write_env(tmp_path), transport=httpx.MockTransport(handler)
        ) as backend:
            with pytest.raises(TimeoutError):
                await LLMClient(config, trace, backend).complete(
                    "flash", [{"role": "user", "content": "你好"}]
                )

    asyncio.run(run())
    assert calls == 2
    ledger = read_usage(trace)
    assert [row["attempt"] for row in ledger] == [1, 2]
    assert all(row["error"] == {"type": "TimeoutError"} for row in ledger)
    assert all(row["status"] == "error" and row["cost_usd"] is None for row in ledger)
    assert len(read_model_records(trace)) == 2


def test_cancellation_stops_retries_and_records_cancelled_attempt(tmp_path):
    config = make_config(tmp_path)
    trace = Trace(config.runs_dir, "cancelled")
    calls = 0

    async def run():
        entered = asyncio.Event()

        async def handler(request):
            nonlocal calls
            calls += 1
            entered.set()
            await asyncio.Event().wait()

        async with CompatibleModel(
            write_env(tmp_path), transport=httpx.MockTransport(handler)
        ) as backend:
            with trace.span("turn", "本轮") as parent:
                task = asyncio.create_task(
                    LLMClient(config, trace, backend).complete(
                        "flash", [{"role": "user", "content": "你好"}]
                    )
                )
                await entered.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                with trace.span("guard", "取消后继续") as next_span:
                    pass
                return parent, next_span

    parent, next_span = asyncio.run(run())
    assert calls == 1
    ledger = read_usage(trace)
    assert len(ledger) == 1
    assert ledger[0]["error"] == {"type": "CancelledError"}
    assert ledger[0]["status"] == "error" and ledger[0]["cost_usd"] is None
    llm_end = next(
        event
        for event in read_jsonl(trace.path)
        if event["kind"] == "llm" and event["event"] == "end"
    )
    assert llm_end["parent_id"] == next_span.parent_id == parent.span_id
    assert llm_end["attrs"]["error_type"] == "CancelledError"


@pytest.mark.parametrize(
    ("has_usage", "prices", "expected_cost"),
    [
        (False, (1, 2, 0.1), None),
        (True, (None, None, None), None),
        (True, (1, 2, None), None),
        (True, (0, 0, 0), 0),
    ],
)
def test_missing_usage_or_prices_stay_unknown(tmp_path, has_usage, prices, expected_cost):
    model = ModelConfig(
        model="fake",
        input_cost_per_million=prices[0],
        output_cost_per_million=prices[1],
        cached_input_cost_per_million=prices[2],
    )
    config = make_config(tmp_path, model=model)
    trace = Trace(config.runs_dir, "unknown")
    raw = model_response(usage=has_usage)
    backend = FakeModel([raw])
    result = asyncio.run(
        LLMClient(config, trace, backend).complete("flash", [{"role": "user", "content": "你好"}])
    )

    row = read_usage(trace)[0]
    assert row["cost_usd"] == expected_cost
    assert (row["usage"] is None) == (not has_usage)
    assert (result.usage is None) == (not has_usage)
    if not has_usage:
        assert row["input_tokens"] is None and row["output_tokens"] is None
        assert row["cached_input_tokens"] is None


def test_malformed_response_keeps_original_and_bills_known_usage(tmp_path):
    config = make_config(tmp_path)
    trace = Trace(config.runs_dir, "invalid")
    raw = model_response()
    raw["choices"] = []
    backend = FakeModel([raw, model_response()])
    client = LLMClient(config, trace, backend)
    with pytest.raises(LLMError, match="choice"):
        asyncio.run(client.complete("flash", [{"role": "user", "content": "你好"}]))

    assert len(backend.requests) == 1
    row = read_usage(trace)[0]
    assert row["status"] == "error"
    assert row["cost_usd"] == pytest.approx(0.000104)
    assert read_model_records(trace)[0]["response"] == raw
    asyncio.run(client.complete("flash", [{"role": "user", "content": "再次查询"}]))
    ledger = read_usage(trace)
    assert len(ledger) == 2
    assert ledger[0] == row
    assert ledger[0]["call_id"] != ledger[1]["call_id"]
    assert sum(entry["cost_usd"] for entry in ledger) == pytest.approx(0.000208)


def test_success_inside_handled_exception_is_not_billed_as_error(tmp_path):
    config = make_config(tmp_path)
    trace = Trace(config.runs_dir, "handled")
    client = LLMClient(config, trace, FakeModel([model_response()]))

    async def run():
        try:
            raise ValueError("已处理的异常")
        except ValueError:
            return await client.complete("flash", [{"role": "user", "content": "你好"}])

    assert asyncio.run(run()).message["content"] == "订单尚未发货。"
    assert read_usage(trace)[0]["status"] == "ok"
    assert read_usage(trace)[0]["error"] is None
    assert read_model_records(trace)[0]["error"] is None


def test_env_is_not_replaced_by_process_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("TTWILL_TEST_KEY", "environment-secret")
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    config = make_config(tmp_path)
    trace = Trace(config.runs_dir, "no-key")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=model_response())

    async def run():
        async with CompatibleModel(env, transport=httpx.MockTransport(handler)) as backend:
            with pytest.raises(LLMError, match=".env"):
                await LLMClient(config, trace, backend).complete(
                    "flash", [{"role": "user", "content": "你好"}]
                )

    asyncio.run(run())
    assert calls == []
    assert len(read_usage(trace)) == 1
    assert "environment-secret" not in json.dumps(read_model_records(trace))

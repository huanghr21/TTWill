"""以离线 HTTP 捕获验证关闭思考的配置抵达本体、官方模拟用户和判分器。"""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

pytest.importorskip("tau2", reason="安装 bench 额外依赖后验证三个评测角色的请求")

import litellm  # noqa: E402
from openai import OpenAI  # noqa: E402
from tau2.data_model.message import AssistantMessage  # noqa: E402
from tau2.user.user_simulator import UserSimulator  # noqa: E402
from tau2.utils import llm_utils  # noqa: E402

from ttwill.bench import runner  # noqa: E402
from ttwill.bench.evaluation import _configured_nl_evaluator  # noqa: E402
from ttwill.config import AppConfig, load_config  # noqa: E402
from ttwill.llm import CompatibleModel, LLMClient  # noqa: E402
from ttwill.tracing import Trace  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def offline_config(tmp_path: Path) -> AppConfig:
    """保留当前模型参数与预算，仅将凭据和产物位置改为测试临时目录。

    Args:
        tmp_path: 接收虚拟凭据和调用产物的独立测试目录。

    Returns:
        来源于项目配置、不会读取真实 .env 的离线测试配置。
    """
    config = load_config(ROOT / "config.toml")
    (tmp_path / ".env").write_text("TTWILL_API_KEY=offline-test-key\n", encoding="utf-8")
    return replace(config, root=tmp_path, runs_dir=tmp_path / "runs")


def response_body(content: str) -> dict:
    """构造包含扩展思考字段的合法响应，以同时检查消息字段保留约定。

    Args:
        content: 模拟服务返回的文本或判分 JSON 字符串。

    Returns:
        OpenAI SDK 和本体均可解析的非流式响应。
    """
    return {
        "id": "offline-thinking-test",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-flash",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": "服务返回的扩展字段仍须保存。",
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def assert_disabled(request: dict) -> None:
    """核对最终 HTTP 请求明确关闭思考，且没有残留的强度或 SDK 包装字段。

    Args:
        request: 模拟 HTTP 传输收到的 JSON 请求体。
    """
    assert request["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in request
    assert "extra_body" not in request


def test_agent_sends_disabled_thinking_and_preserves_returned_fields(offline_config) -> None:
    """当前配置经统一模型层发到 HTTP，并完整保留服务返回的扩展字段。"""
    requests = []
    raw = response_body("请提供订单信息。")

    def transport(request: httpx.Request) -> httpx.Response:
        """捕获本体最终请求体，只返回本地响应。"""
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=raw)

    async def run():
        """通过真实统一调用层执行离线请求，返回校验后的响应。"""
        async with CompatibleModel(
            offline_config.root / ".env", transport=httpx.MockTransport(transport)
        ) as backend:
            return await LLMClient(
                offline_config, Trace(offline_config.runs_dir, "thinking-test"), backend
            ).complete("flash", [{"role": "user", "content": "请查询订单。"}])

    result = asyncio.run(run())

    assert len(requests) == 1
    assert_disabled(requests[0])
    assert result.message == raw["choices"][0]["message"]


@pytest.mark.parametrize("role", ["user_simulator", "evaluator"])
def test_official_roles_send_disabled_thinking(offline_config, monkeypatch, role) -> None:
    """官方模拟用户和自然语言判分均经真实 LiteLLM、SDK 将关闭参数送达 HTTP。"""
    requests = []
    content = (
        "请查询订单。"
        if role == "user_simulator"
        else json.dumps(
            {
                "results": [
                    {
                        "expectedOutcome": "客服回答了订单问题。",
                        "metExpectation": True,
                        "reasoning": "测试对话包含所需回答。",
                    }
                ]
            },
            ensure_ascii=False,
        )
    )

    def transport(request: httpx.Request) -> httpx.Response:
        """捕获 SDK 合并参数后的请求体，阻断外部模型调用。"""
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response_body(content))

    model = offline_config.models["flash"]
    arguments = runner._model_args(offline_config, model)
    with OpenAI(
        api_key="offline-test-key",
        base_url="https://unit.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    ) as client:

        def completion(**kwargs):
            """仅替换最终 HTTP 客户端，保留官方调用及 LiteLLM 参数变换。"""
            return litellm.completion(**kwargs, client=client)

        monkeypatch.setattr(llm_utils, "completion", completion)
        if role == "user_simulator":
            user = UserSimulator(
                llm="openai/" + model.model,
                instructions="向客服询问订单。",
                llm_args={**arguments, "num_retries": 0},
                global_guidelines_override="只作为顾客询问订单，不代替客服执行工具。",
            )
            result, _ = user.generate_next_message(
                AssistantMessage(role="assistant", content="您好，需要什么帮助？"),
                user.get_init_state(),
            )
            assert result.content == content
        else:
            evaluator = _configured_nl_evaluator(
                "openai/" + model.model, {**arguments, "num_retries": 0}
            )
            result = evaluator.evaluate_nl_assertions(
                [AssistantMessage(role="assistant", content="订单尚未发货。")],
                ["客服回答了订单问题。"],
            )
            assert len(result) == 1 and result[0].met

    assert len(requests) == 1
    assert_disabled(requests[0])
    assert model.parameters == {"thinking": {"type": "disabled"}}

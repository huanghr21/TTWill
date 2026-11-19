"""验证网关工具定义经过统一模型调用层后仍可选择和派发。"""

import asyncio
import json
from pathlib import Path

import httpx

from ttwill.backend import RetailBackend
from ttwill.config import AppConfig, LLMConfig, ModelConfig
from ttwill.gateway import CallContext, SessionContext, ToolGateway
from ttwill.gateway.skills import SkillCatalog
from ttwill.llm import CompatibleModel, LLMClient
from ttwill.tracing import Trace


def test_model_selects_gateway_tools_through_compatible_transport(
    tmp_path: Path, backend: RetailBackend, skill_catalog: SkillCatalog
) -> None:
    """模拟兼容服务验证身份、加载取消政策并准备取消订单，实际派发到 SQLite。"""
    trace = Trace(tmp_path / "run", "tools")
    gateway = ToolGateway(backend, trace, skill_catalog)
    config = AppConfig(
        root=tmp_path,
        runs_dir=trace.run_dir,
        llm=LLMConfig(timeout_seconds=1, max_retries=0, retry_delay_seconds=0),
        models={
            "flash": ModelConfig(
                model="test-model",
                base_url="https://model.invalid/v1",
                api_key_env="TTWILL_TEST_KEY",
            )
        },
    )
    env_path = tmp_path / ".env"
    env_path.write_text("TTWILL_TEST_KEY=test-placeholder\n", encoding="utf-8")
    requests = []

    def choose_tool(request: httpx.Request) -> httpx.Response:
        """从兼容工具定义中依次选择身份查询、政策加载和订单取消操作。

        Args:
            request: 包含工具定义及对话消息的模拟兼容接口请求。

        Returns:
            包含单条工具调用的成功响应；三轮依次选择身份查询、政策加载和取消订单。
        """
        payload = json.loads(request.content)
        requests.append(payload)
        assert all(tool["type"] == "function" for tool in payload["tools"])
        available = {tool["function"]["name"]: tool["function"] for tool in payload["tools"]}
        assert "commit" not in available
        if len(requests) == 1:
            selected = available["find_user_id_by_email"]
            args = {"email": "u1@example.com"}
        elif len(requests) == 2:
            selected = available["load_skill"]
            args = {"name": skill_catalog.required_skill("cancel_pending_order")}
        else:
            selected = available["cancel_pending_order"]
            args = {"order_id": "#W0000001", "reason": "no longer needed"}
        assert set(selected["parameters"]["required"]) == set(args)
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call-{len(requests)}",
                                    "type": "function",
                                    "function": {
                                        "name": selected["name"],
                                        "arguments": json.dumps(args),
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    async def select_and_dispatch() -> dict:
        """将模型选择依次交给身份入口和订单服务入口。

        Returns:
            完成身份绑定后，由订单服务创建的取消订单准备记录。
        """
        session = SessionContext("tools")
        contexts = [
            CallContext(session, "coordinator", "coordinator"),
            CallContext(session, "order_service", "instance-1", "task-1"),
            CallContext(session, "order_service", "instance-1", "task-1"),
        ]
        messages = [
            {"role": "user", "content": "我的邮箱是 u1@example.com，请取消订单 #W0000001。"}
        ]
        async with CompatibleModel(
            env_path, transport=httpx.MockTransport(choose_tool)
        ) as transport:
            llm = LLMClient(config, trace, transport)
            results = []
            for context in contexts:
                response = await llm.complete("flash", messages, tools=gateway.schemas(context))
                call = response.message["tool_calls"][0]
                function = call["function"]
                result = gateway.call(context, function["name"], json.loads(function["arguments"]))
                results.append(result)
                messages.extend(
                    [
                        response.message,
                        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)},
                    ]
                )
            assert results[0] == session.user_id == "u1"
            assert results[1]["name"] == skill_catalog.required_skill("cancel_pending_order")
            return results[2]

    prepared = asyncio.run(select_and_dispatch())
    assert len(requests) == 3
    assert requests[1]["tools"] == requests[2]["tools"]
    assert prepared["status"] == "prepared"
    assert prepared["preview"]["status"] == "cancelled"
    assert (
        backend.query("u1", "get_order_details", {"order_id": "#W0000001"})["status"] == "pending"
    )
    assert gateway.records[-1]["result"] == prepared

"""用脚本化模型与真实 SQLite 验证子 Agent 的权限、结构结果和循环终止。"""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ttwill.agents import build_registry
from ttwill.config import AppConfig, LLMConfig, ModelConfig
from ttwill.core import Capability, CapabilityRegistry, DispatchTask, SubAgentRunner
from ttwill.gateway import CallContext, SessionContext, ToolGateway
from ttwill.llm import FakeModel, LLMClient, LLMError
from ttwill.tracing import Trace


def response(*calls: tuple[str, dict | str], content: dict | str | None = None) -> dict:
    """构造单条文本或若干工具调用组成的假模型响应。

    Args:
        calls: 按执行顺序排列的工具名与参数；字符串参数按原文保留。
        content: 文本内容或待序列化的 JSON 结束对象；None 表示只有工具调用。

    Returns:
        符合统一调用层响应结构的完整字典。
    """
    message = {
        "role": "assistant",
        "content": json.dumps(content, ensure_ascii=False)
        if isinstance(content, dict)
        else content,
    }
    if calls:
        message["tool_calls"] = [
            {
                "id": f"call-{index}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False)
                    if isinstance(args, dict)
                    else args,
                },
            }
            for index, (name, args) in enumerate(calls)
        ]
    return {"model": "fake", "choices": [{"message": message, "finish_reason": "stop"}]}


def finish(*, missing: list[str] | None = None, hint: str = "请核对操作详情。") -> dict:
    """构造只报告信息缺口与回复要点的合法结束响应。

    Args:
        missing: 尚缺的信息列表；None 表示没有缺口。
        hint: 模型提供给协调者的回复要点。

    Returns:
        只调用 finish 的完整模型响应。
    """
    return response(("finish", {"missing": missing or [], "message_hint": hint}))


def cancel_steps(order_id: str = "#1") -> list[dict]:
    """构造查询、加载政策、准备取消和结束的脚本。

    Args:
        order_id: 要查询并准备取消的待发货订单。

    Returns:
        四次模型响应组成的独立脚本。
    """
    return [
        response(("get_order_details", {"order_id": order_id})),
        response(("load_skill", {"name": "cancel_pending_order"})),
        response(("cancel_pending_order", {"order_id": order_id, "reason": "no longer needed"})),
        finish(),
    ]


@pytest.fixture
def runtime(tmp_path, backend, skill_catalog):
    """为各测试创建共用真实网关及独立假模型实例的工厂。

    Args:
        tmp_path: 本项测试的临时目录。
        backend: 本项测试独占的 SQLite 业务后端。
        skill_catalog: 仓库中的真实政策和 Skill 目录。

    Returns:
        make：接收响应脚本、步数预算和可选模型配置，返回 runner、fake、gateway。
    """
    trace = Trace(tmp_path / "runs", "session")
    gateway = ToolGateway(backend, trace, skill_catalog)
    registry = build_registry(skill_catalog)

    def make(script, *, max_steps=12, models=None, timeout_seconds=1):
        """绑定指定脚本，复用当前测试的网关与业务库。

        Args:
            script: FakeModel 按序消费的响应或异常列表。
            max_steps: 本实例的步骤预算。
            models: 档位配置；None 表示只有 flash 假模型。
            timeout_seconds: 每次模型请求的总时限，单位为秒。

        Returns:
            依次返回运行器、假模型和共享网关。
        """
        fake = FakeModel(script)
        config = AppConfig(
            tmp_path,
            tmp_path / "runs",
            LLMConfig(timeout_seconds, 0, 0),
            models or {"flash": ModelConfig("fake")},
        )
        llm = LLMClient(config, trace, fake)
        return SubAgentRunner(registry, gateway, llm, max_steps=max_steps), fake, gateway

    return make


def run_task(runner, *, capability="order_service", task=None, session=None, model_tier=None):
    """同步运行测试任务，默认使用已验证用户的订单服务能力。

    Args:
        runner: 待运行的子 Agent 循环。
        capability: 当前任务的预设能力名称。
        task: 分派信息；None 表示一个取消订单写任务。
        session: 会话上下文；None 表示 session 会话内的用户 u1。
        model_tier: 覆盖的模型档位；None 表示使用能力默认值。

    Returns:
        子 Agent 运行时代码组装的结构化结果。
    """
    return asyncio.run(
        runner.run(
            session or SessionContext("session", "u1"),
            capability,
            task or DispatchTask("task", "取消订单 #1", is_write=True),
            model_tier=model_tier,
        )
    )


def trace_events(path: Path) -> list[dict[str, Any]]:
    """读取当前轨迹的 JSONL 事件列表。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_cancel_returns_real_pending_operation_without_committing(runtime, backend):
    """取消任务只准备真实操作，并且结果字段和轨迹均可追溯到工具调用。"""
    initial = backend.export_snapshot()
    runner, fake, gateway = runtime(cancel_steps())
    result = run_task(runner)
    assert result["status"] == "need_confirm"
    assert result["steps"] == 4
    assert result["facts"][0]["result"]["status"] == "pending"
    assert result["facts"][0]["source"]["instance_id"] == result["instance_id"]
    assert len(result["facts"]) == 1
    pending = result["pending_confirmation"]
    assert len(pending) == 1
    operation = backend.get_operation("u1", operation_id=pending[0]["operation_id"])
    assert operation["status"] == "prepared"
    assert pending[0]["args"] == operation["args"]
    assert pending[0]["preview"] == operation["preview"]
    assert result["actions"][0]["result"] == operation
    assert backend.export_snapshot() == initial
    context = CallContext(SessionContext("session", "u1"), "order_service", result["instance_id"])
    assert not gateway.loaded_skills(context)
    first_request = fake.requests[0]
    assert len(first_request["messages"]) == 2
    assert "取消订单 #1" in first_request["messages"][1]["content"]
    assert "已送达订单的退货" not in first_request["messages"][0]["content"]
    events = trace_events(gateway.trace.path)
    starts = [event for event in events if event["event"] == "start"]
    dispatch = next(event for event in starts if event["kind"] == "dispatch")
    subagent = next(event for event in starts if event["kind"] == "subagent")
    assert subagent["parent_id"] == dispatch["span_id"]
    assert all(
        event["parent_id"] == subagent["span_id"]
        for event in starts
        if event["kind"] in {"llm", "tool", "result"}
    )


def test_return_uses_after_sales_skill_and_keeps_database_unchanged(runtime, backend):
    """售后实例通过退货 Skill 准备申请，业务订单仍是已送达。"""
    runner, _, _ = runtime(
        [
            response(("get_order_details", {"order_id": "#2"})),
            response(("load_skill", {"name": "return_delivered_order"})),
            response(
                (
                    "return_delivered_order_items",
                    {"order_id": "#2", "item_ids": ["a", "x"], "payment_method_id": "card"},
                )
            ),
            finish(),
        ]
    )
    result = run_task(
        runner,
        capability="after_sales",
        task=DispatchTask("return", "退回已送达商品", is_write=True),
    )
    assert result["status"] == "need_confirm"
    assert result["pending_confirmation"][0]["preview"]["status"] == "return requested"
    assert backend.query("u1", "get_order_details", {"order_id": "#2"})["status"] == "delivered"


def test_missing_fields_are_the_only_model_provided_state(runtime):
    """缺少订单号时只返回经严格校验的信息缺口，模型没有创建动作或事实。"""
    runner, _, _ = runtime([finish(missing=["订单号"], hint="请提供要取消的订单号。")])
    result = run_task(runner)
    assert result["status"] == "need_info"
    assert result["missing"] == ["订单号"]
    assert result["actions"] == result["facts"] == result["pending_confirmation"] == []


@pytest.mark.parametrize(
    "forged", ["actions", "status", "task_id", "facts", "pending_confirmation"]
)
def test_forged_result_fields_are_rejected_then_model_can_correct(runtime, forged):
    """模型伪造可信字段会触发格式护栏，修正后也不产生虚假动作。"""
    runner, fake, gateway = runtime(
        [
            response(("finish", {"missing": [], "message_hint": "已完成", forged: "伪造"})),
            finish(missing=["订单号"]),
        ]
    )
    result = run_task(runner)
    assert result["status"] == "need_info"
    assert result["task_id"] == "task"
    assert result["actions"] == result["facts"] == []
    assert "invalid_tool_call" in fake.requests[1]["messages"][-1]["content"]
    assert any(event["name"] == "invalid_tool_call" for event in trace_events(gateway.trace.path))


@pytest.mark.parametrize("missing", ["订单号", [1], None])
def test_finish_does_not_coerce_invalid_fields(runtime, missing):
    """结束字段不进行字符串、列表或空值的隐式类型转换。"""
    runner, _, _ = runtime(
        [response(content={"missing": missing, "message_hint": "缺少资料"})], max_steps=1
    )
    result = run_task(runner)
    assert result["status"] == "failed"
    assert result["missing"] == []
    assert "步骤上限 1" in result["reason"]


def test_plain_json_finish_and_read_result_are_supported(runtime):
    """合法文本 JSON 可以结束只读任务，状态依据实际查询结果生成。"""
    runner, _, _ = runtime(
        [
            response(("get_order_details", {"order_id": "#1"})),
            response(content={"missing": [], "message_hint": "订单待发货。"}),
        ]
    )
    result = run_task(runner, task=DispatchTask("read", "查询订单"))
    assert result["status"] == "ready"
    assert result["facts"][0]["result"]["order_id"] == "#1"


def test_thinking_messages_survive_tool_loop_and_invalid_text(runtime):
    """工具循环和无工具的格式修正均回传完整思考字段，原始响应仍可追溯。"""
    script = [
        response(("get_order_details", {"order_id": "#1"})),
        response(("get_order_details", {"order_id": "#2"})),
        response(content="订单信息已查到。"),
        finish(hint="两笔订单已查询。"),
    ]
    for index, item in enumerate(script):
        message = item["choices"][0]["message"]
        message["reasoning_content"] = f"测试思考标记 {index}"
        if message.get("tool_calls"):
            message["tool_calls"][0]["id"] = f"thinking-call-{index}"
    runner, fake, gateway = runtime(script)

    result = run_task(runner, task=DispatchTask("read", "查询两笔订单"))

    assert result["status"] == "ready"
    assert result["steps"] == len(fake.requests) == 4
    for index, request in enumerate(fake.requests):
        assert [message for message in request["messages"] if message["role"] == "assistant"] == [
            item["choices"][0]["message"] for item in script[:index]
        ]
    assert [
        message["tool_call_id"]
        for message in fake.requests[-1]["messages"]
        if message["role"] == "tool"
    ] == ["thinking-call-0", "thinking-call-1"]
    assert any(event["name"] == "invalid_finish" for event in trace_events(gateway.trace.path))
    usage = trace_events(gateway.trace.run_dir / "usage.jsonl")
    records = [
        json.loads((gateway.trace.run_dir / row["record_ref"]).read_text(encoding="utf-8"))
        for row in usage
    ]
    assert [record["response"] for record in records] == script
    assert records[-1]["request"] == fake.requests[-1]


def test_write_without_skill_is_rejected_and_can_be_retried_after_loading(runtime, backend):
    """未加载政策的写调用被网关拦截，加载后同实例才可准备操作。"""
    steps = cancel_steps()
    runner, fake, _ = runtime([steps[2], *steps[1:]])
    result = run_task(runner)
    assert result["status"] == "need_confirm"
    assert result["actions"][0]["status"] == "failed"
    assert result["actions"][1]["status"] == "prepared"
    assert "skill" in fake.requests[1]["messages"][-1]["content"]
    assert backend.export_snapshot()["orders"]["#1"]["status"] == "pending"


def test_invalid_json_unknown_tool_and_business_error_each_consume_steps(runtime):
    """参数格式、越权工具和业务拒绝都有反馈且计入同一个步骤预算。"""
    runner, fake, _ = runtime(
        [
            response(("get_order_details", "{broken")),
            response(("commit", {"operation_id": "invented"})),
            response(("get_order_details", {"order_id": "#3"})),
            finish(missing=["本人订单号"]),
        ],
        max_steps=4,
    )
    result = run_task(runner)
    assert result["status"] == "need_info"
    assert result["steps"] == 4
    assert result["facts"] == []
    assert all("error" in request["messages"][-1]["content"] for request in fake.requests[1:])


def test_batch_tools_cannot_exceed_budget_and_failed_prepare_is_rejected(runtime, backend):
    """单个响应塞入大量调用只能执行剩余预算，已准备操作在失败时释放。"""
    calls = [
        ("load_skill", {"name": "cancel_pending_order"}),
        ("cancel_pending_order", {"order_id": "#1", "reason": "no longer needed"}),
    ] + [("get_order_details", {"order_id": "#4"})] * 100
    runner, fake, gateway = runtime([response(*calls)], max_steps=2)
    result = run_task(runner)
    assert result["status"] == "failed"
    assert result["steps"] == 2
    assert result["facts"] == result["pending_confirmation"] == []
    assert len(fake.requests) == 1
    assert [record["kind"] for record in gateway.records] == ["call", "call", "operation", "reject"]
    operation_id = result["actions"][0]["operation_id"]
    assert backend.get_operation("u1", operation_id=operation_id)["status"] == "rejected"
    assert any(
        event["name"] == "subagent_step_limit" and event.get("status") == "blocked"
        for event in trace_events(gateway.trace.path)
    )
    next_runner, _, _ = runtime(cancel_steps())
    retry = run_task(next_runner, task=DispatchTask("next", "取消订单", is_write=True))
    assert retry["status"] == "need_confirm"


def test_finish_at_exact_budget_succeeds_and_same_batch_finish_is_rejected(runtime):
    """独立 finish 恰好占满预算仍成功，同批混合 finish 必须重新返回。"""
    runner, fake, _ = runtime(
        [
            response(
                ("finish", {"missing": [], "message_hint": "结束"}),
                ("get_order_details", {"order_id": "#1"}),
            ),
            finish(hint="订单待发货"),
        ],
        max_steps=3,
    )
    result = run_task(runner, task=DispatchTask("read", "查询订单"))
    assert result["status"] == "ready"
    assert result["steps"] == 3
    assert "finish 须单独调用" in fake.requests[1]["messages"][-2]["content"]


def test_new_instance_has_no_previous_context_or_skill_unlock(runtime):
    """同一会话的后续实例不会继承聊天或 Skill 加载状态。"""
    runner, _, gateway = runtime(cancel_steps())
    first = run_task(runner)
    second_runner, second_fake, _ = runtime(
        [
            response(("cancel_pending_order", {"order_id": "#4", "reason": "no longer needed"})),
            finish(missing=["本次操作政策"]),
        ]
    )
    second = run_task(second_runner, task=DispatchTask("new", "取消另一订单", is_write=True))
    assert first["instance_id"] != second["instance_id"]
    assert second["status"] == "need_info"
    assert second["actions"][0]["status"] == "failed"
    assert len(second_fake.requests[0]["messages"]) == 2
    assert "取消订单 #1" not in json.dumps(second_fake.requests[0]["messages"], ensure_ascii=False)
    assert len(second["actions"]) == 1
    assert second["pending_confirmation"] == []
    assert any(record["instance_id"] == second["instance_id"] for record in gateway.records)


def test_committed_retry_requires_verification_and_cannot_become_done(runtime, backend):
    """同任务重试找到已提交操作时仅返回待核验状态，不由模型宣布完成。"""
    runner, _, gateway = runtime(cancel_steps())
    first = run_task(runner)
    operation = first["pending_confirmation"][0]
    gateway.commit(
        CallContext(SessionContext("session", "u1"), "order_service", "coordinator"),
        operation["operation_id"],
        operation["args"],
    )
    retry_runner, _, _ = runtime(cancel_steps())
    retry = run_task(retry_runner)
    assert retry["status"] == "needs_verification"
    assert retry["pending_confirmation"] == []
    assert retry["actions"][0]["result"]["status"] == "committed"
    assert backend.export_snapshot()["orders"]["#1"]["status"] == "cancelled"


def test_model_error_after_prepare_rejects_operation_and_releases_skill(runtime, backend):
    """模型失败向上传播，同时作废本实例准备项和 Skill 记录。"""
    runner, _, gateway = runtime([*cancel_steps()[:-1], LLMError("模拟服务失败")])
    with pytest.raises(LLMError, match="模拟服务失败"):
        run_task(runner)
    prepared = next(record for record in gateway.records if record.get("status") == "prepared")
    assert (
        backend.get_operation("u1", operation_id=prepared["operation_id"])["status"] == "rejected"
    )
    context = CallContext(SessionContext("session", "u1"), "order_service", prepared["instance_id"])
    assert not gateway.loaded_skills(context)


@pytest.mark.parametrize("interruption", ["cancel", "timeout"])
def test_interruption_after_prepare_cleans_operation_and_instance(runtime, backend, interruption):
    """请求取消或超时发生在准备之后时，保留异常语义并释放占用。"""
    runner, fake, gateway = runtime(cancel_steps()[:-1], timeout_seconds=0.01)
    original_send = fake.send
    entered = asyncio.Event()

    async def delayed_send(model, request, timeout_seconds):
        """前三次消费正常脚本，结束前暂停以触发取消或超时。"""
        if len(fake.requests) < 3:
            return await original_send(model, request, timeout_seconds)
        entered.set()
        await asyncio.Event().wait()

    fake.send = delayed_send

    async def exercise():
        """在模型暂停后触发指定中断，并等待运行器完成清理。"""
        task = asyncio.create_task(
            runner.run(
                SessionContext("session", "u1"),
                "order_service",
                DispatchTask("task", "取消", is_write=True),
            )
        )
        await entered.wait()
        if interruption == "cancel":
            task.cancel()
        error = asyncio.CancelledError if interruption == "cancel" else TimeoutError
        with pytest.raises(error):
            await task

    asyncio.run(exercise())
    prepared = next(record for record in gateway.records if record.get("status") == "prepared")
    assert (
        backend.get_operation("u1", operation_id=prepared["operation_id"])["status"] == "rejected"
    )
    context = CallContext(SessionContext("session", "u1"), "order_service", prepared["instance_id"])
    assert not gateway.loaded_skills(context)


def test_no_tool_facts_or_model_hint_can_complete_a_write(runtime):
    """只靠模型声称成功不会构成业务事实或写任务完成证据。"""
    runner, _, _ = runtime([finish(hint="已经取消订单并退款。")])
    result = run_task(runner)
    assert result["status"] == "failed"
    assert result["facts"] == result["actions"] == []
    assert "未准备操作" in result["reason"]


def test_multiple_operations_have_separate_pending_entries(runtime):
    """订单取消与用户地址的两个操作保留各自参数和待确认详情。"""
    script = cancel_steps()
    script[-1:-1] = [
        response(("load_skill", {"name": "modify_user_address"})),
        response(
            (
                "modify_user_address",
                {
                    "user_id": "u1",
                    "address1": "新地址",
                    "address2": "",
                    "city": "上海",
                    "state": "上海",
                    "country": "中国",
                    "zip": "200000",
                },
            )
        ),
    ]
    runner, _, _ = runtime(script)
    result = run_task(runner)
    assert result["status"] == "need_confirm"
    pending = result["pending_confirmation"]
    assert len(pending) == 2
    assert {operation["action"] for operation in pending} == {
        "cancel_pending_order",
        "modify_user_address",
    }
    assert len({operation["operation_id"] for operation in pending}) == 2


def test_registry_and_dispatch_validate_identity_revision_and_budget(runtime):
    """注册重复能力和无效分派输入在调用模型之前失败。"""
    registry = CapabilityRegistry()
    capability = Capability("example", "描述", "提示词", (), frozenset())
    registry.register(capability)
    assert registry.get("example") is capability
    with pytest.raises(ValueError, match="已注册"):
        registry.register(capability)
    with pytest.raises(ValueError, match="未知能力"):
        registry.get("missing")
    runner, fake, _ = runtime([])
    with pytest.raises(ValueError, match="验证"):
        run_task(runner, session=SessionContext("session"))
    with pytest.raises(ValueError, match="修订号"):
        run_task(runner, task=DispatchTask("task", "查询", revision=0))
    with pytest.raises(ValueError, match="步数上限"):
        runtime([], max_steps=True)
    assert fake.requests == []


def test_model_tier_override_and_confirmed_facts_enter_only_new_task(runtime):
    """派发可覆盖模型档位，已确认事实通过任务输入传递且保持原对象。"""
    known = [{"tool": "get_order_details", "result": {"order_id": "#1"}}]
    original = deepcopy(known)
    runner, fake, _ = runtime(
        [finish(missing=["取消原因"])],
        models={"flash": ModelConfig("cheap"), "strong": ModelConfig("strong")},
    )
    result = run_task(
        runner,
        task=DispatchTask("task", "确认取消原因", facts=known, is_write=True),
        model_tier="strong",
    )
    assert result["status"] == "need_info"
    assert fake.requests[0]["model"] == "strong"
    assert json.loads(fake.requests[0]["messages"][1]["content"])["facts"] == original
    assert known == original


def test_registered_skill_scope_is_enforced_when_name_is_guessed(runtime):
    """模型猜中未被能力声明的 Skill 名称时，加载与后续写入仍被拒绝。"""
    runner, _, gateway = runtime(cancel_steps()[1:])
    registry = CapabilityRegistry()
    capability = runner.registry.get("order_service")
    registry.register(
        Capability(capability.name, capability.description, capability.prompt, (), capability.tools)
    )
    runner.registry = registry
    result = run_task(runner)
    assert result["status"] == "failed"
    assert not result["pending_confirmation"]
    assert not any(record["action"] == "load_skill" for record in gateway.records)
    assert result["actions"][0]["status"] == "failed"


def test_registry_tools_cannot_be_bypassed_by_guessing_load_skill(runtime):
    """未在能力工具名单内的 load_skill 不会被运行器额外开放。"""
    runner, fake, _ = runtime(cancel_steps()[1:])
    capability = runner.registry.get("order_service")
    registry = CapabilityRegistry()
    registry.register(
        Capability(
            capability.name,
            capability.description,
            capability.prompt,
            capability.skills,
            capability.tools - {"load_skill"},
        )
    )
    runner.registry = registry
    result = run_task(runner)
    assert result["status"] == "failed"
    assert result["pending_confirmation"] == []
    assert "load_skill" not in {tool["function"]["name"] for tool in fake.requests[0]["tools"]}


@pytest.mark.parametrize("kind", ["result", "subagent", "dispatch"])
def test_trace_error_after_prepare_rejects_operation(runtime, backend, kind):
    """最终结果或外层轨迹结束时失败，仍清理已准备操作与实例加载记录。"""
    runner, _, gateway = runtime(cancel_steps())

    def fail_trace(event):
        """在指定轨迹的结束事件模拟写入或订阅异常。"""
        if event["kind"] == kind and event["event"] == "end":
            raise OSError("模拟轨迹失败")

    gateway.trace.subscribe(fail_trace)
    with pytest.raises(OSError, match="模拟轨迹失败"):
        run_task(runner)
    prepared = next(record for record in gateway.records if record.get("status") == "prepared")
    operation = backend.get_operation("u1", operation_id=prepared["operation_id"])
    assert operation["status"] == "rejected"
    context = CallContext(SessionContext("session", "u1"), "order_service", prepared["instance_id"])
    assert not gateway.loaded_skills(context)


def test_concurrent_instances_keep_context_and_results_separate(runtime):
    """同一会话并发派发两笔订单时，Skill、事实、动作和轨迹按实例隔离。"""
    first_runner, first_fake, gateway = runtime(cancel_steps("#1"))
    second_runner, second_fake, _ = runtime(cancel_steps("#4"))

    def interleave(fake):
        """在假模型返回前让出事件循环，使两实例交错执行。"""
        original_send = fake.send

        async def send(model, request, timeout_seconds):
            """保留假模型行为，并在每次响应前交出执行权。"""
            await asyncio.sleep(0)
            return await original_send(model, request, timeout_seconds)

        fake.send = send

    interleave(first_fake)
    interleave(second_fake)

    async def exercise():
        """并发执行两笔互不占用同一订单的取消任务。"""
        session = SessionContext("session", "u1")
        return await asyncio.gather(
            first_runner.run(
                session, "order_service", DispatchTask("first", "取消 #1", is_write=True)
            ),
            second_runner.run(
                session, "order_service", DispatchTask("second", "取消 #4", is_write=True)
            ),
        )

    results = asyncio.run(exercise())
    assert len({result["instance_id"] for result in results}) == 2
    for result, expected_order in zip(results, ("#1", "#4"), strict=True):
        assert result["status"] == "need_confirm"
        assert len(result["actions"]) == len(result["facts"]) == 1
        assert result["pending_confirmation"][0]["args"]["order_id"] == expected_order
        assert result["facts"][0]["result"]["order_id"] == expected_order
        context = CallContext(
            SessionContext("session", "u1"), "order_service", result["instance_id"]
        )
        assert not gateway.loaded_skills(context)


def test_expired_operation_is_refreshed_before_pending_result(runtime, backend):
    """模型结束前到期的操作不会继续作为待确认结果返回。"""
    now = [100.0]
    backend.clock = lambda: now[0]
    backend.ttl_seconds = 10
    runner, fake, gateway = runtime(cancel_steps())
    original_send = fake.send

    async def advance_before_finish(model, request, timeout_seconds):
        """在结束响应前推进业务时钟，让之前的准备项到期。"""
        if len(fake.requests) == 3:
            now[0] = 111.0
        return await original_send(model, request, timeout_seconds)

    fake.send = advance_before_finish
    result = run_task(runner)
    assert result["status"] == "failed"
    assert result["pending_confirmation"] == []
    assert "expired" in result["reason"]
    assert len(result["actions"]) == 1
    operation_record = next(record for record in gateway.records if record["kind"] == "operation")
    assert operation_record["result"]["status"] == "expired"

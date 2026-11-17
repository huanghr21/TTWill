"""用脚本化协调者、真实子 Agent 和 SQLite 验证多轮办理及写操作护栏。"""

import asyncio
import json

import pytest
from sqlalchemy import select, update

from ttwill.agents import build_registry
from ttwill.backend.store import orders
from ttwill.config import AppConfig, LLMConfig, ModelConfig
from ttwill.core.coordinator import STOP_NOTICE, ConversationState, Coordinator
from ttwill.core.subagent import SubAgentRunner
from ttwill.core.tasks import Task
from ttwill.gateway import SessionContext, ToolGateway
from ttwill.llm import FakeModel, LLMClient, LLMError
from ttwill.tracing import Trace

ADDRESS = {
    "address1": "新地址",
    "address2": "",
    "city": "北京",
    "state": "北京",
    "country": "中国",
    "zip": "100000",
}


def tool(name: str, args: dict) -> dict:
    """生成只含一个工具调用的完整假模型响应。

    Args:
        name: 协调者或子 Agent 将调用的工具名。
        args: 本次工具调用的参数。

    Returns:
        统一 LLM 调用层可解析的 Chat Completions 响应。
    """
    return {
        "model": "fake",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args, ensure_ascii=False),
                            },
                        }
                    ],
                },
            }
        ],
    }


def add_task(action: str, order_id: str, instruction: str, **fields) -> dict:
    """构造由模型提出、标识和执行状态仍由代码维护的新增任务。

    Args:
        action: 请求的业务动作名称。
        order_id: 目标订单标识。
        instruction: 子任务的具体办理要求。
        **fields: 附加的合法任务字段，例如自然语言 condition。

    Returns:
        update_tasks 的单项新增请求。
    """
    return {
        "operation": "add",
        "capability": "after_sales" if "delivered" in action else "order_service",
        "action": action,
        "order_id": order_id,
        "instruction": instruction,
        **fields,
    }


def text_response(content: str) -> dict:
    """将模型直接回复的客服文本包装为兼容接口响应。"""
    return {
        "model": "fake",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }


def cancel_task(order_id: str) -> dict:
    """生成指定订单的取消诉求。"""
    return add_task("cancel_pending_order", order_id, f"取消订单 {order_id}，不再需要")


def prepare_steps(action: str, args: dict, *, skill: str | None = None) -> list[dict]:
    """创建加载 Skill、准备真实写操作和结束的子 Agent 脚本。

    Args:
        action: 子 Agent 应准备的业务动作名称。
        args: 交给真实后端校验的写操作参数。
        skill: 关联 Skill 的名称；省略时与 action 相同。

    Returns:
        三次假模型响应，最后的成功措辞不能作为完成依据。
    """
    return [
        tool("load_skill", {"name": skill or action}),
        tool(action, args),
        tool("finish", {"missing": [], "message_hint": "已全部成功办理"}),
    ]


def cancel_steps(order_id: str) -> list[dict]:
    """生成取消订单并保留待确认状态的子 Agent 脚本。"""
    return prepare_steps(
        "cancel_pending_order", {"order_id": order_id, "reason": "no longer needed"}
    )


def reply(show_task_ids: list[str] | None = None) -> dict:
    """请求回复并指定展示范围，默认展示全部待确认操作。"""
    return tool("reply", {"message_hint": "已全部成功办理", "show_task_ids": show_task_ids})


def task_ids(state: ConversationState, *, status: str | None = None) -> list[str]:
    """读取代码创建的任务标识，可按状态筛选。

    Args:
        state: 当前多轮对话状态。
        status: 只选择该状态；None 表示全部任务。

    Returns:
        按任务创建顺序排列的标识列表。
    """
    return [
        task.id for task in state.tasks.tasks.values() if status is None or task.status == status
    ]


def by_order(result: dict) -> dict:
    """将每单仅有一个任务的结果按订单标识索引。"""
    return {task["order_id"]: task for task in result["tasks"]}


@pytest.fixture
def runtime(tmp_path, backend, skill_catalog):
    """为每项测试连接真实业务库、子 Agent 和两个独立假模型。

    Args:
        tmp_path: 测试轨迹和模型原始记录的临时目录。
        backend: 本测试独占的小型 SQLite 后端。
        skill_catalog: 七种写操作所需的静态 Skill 目录。

    Returns:
        make 工厂，接收协调脚本生成器与子 Agent 响应列表，依次返回协调者、
        会话状态、网关、协调假模型和子 Agent 假模型。
    """
    trace = Trace(tmp_path / "runs", "chat")
    gateway = ToolGateway(backend, trace, skill_catalog)
    config = AppConfig(
        tmp_path,
        tmp_path / "runs",
        LLMConfig(1, 0, 0),
        {"flash": ModelConfig("fake")},
    )

    def make(script, child_script, *, max_rounds=12):
        """把含动态任务标识的协调脚本接入统一模型调用层。

        Args:
            script: 接收 state 并产生协调模型响应的生成器函数。
            child_script: 子 Agent 按序消费的响应或模型异常。
            max_rounds: 每个用户轮次允许的协调决策轮数。

        Returns:
            coordinator、state、gateway、coordinator_fake、child_fake。
        """
        state = ConversationState(SessionContext("chat"))
        coordinator_fake = FakeModel(script(state))
        child_fake = FakeModel(child_script)
        llm = LLMClient(config, trace, coordinator_fake)
        runner = SubAgentRunner(
            build_registry(skill_catalog),
            gateway,
            LLMClient(config, trace, child_fake),
            max_steps=8,
        )
        coordinator = Coordinator(gateway, runner, llm, max_rounds=max_rounds)
        return coordinator, state, gateway, coordinator_fake, child_fake

    return make


def turn(coordinator: Coordinator, state: ConversationState, message: str) -> dict:
    """同步运行一个真实用户轮次并取得最终回复及代码任务状态。

    Args:
        coordinator: 本测试的协调者实例。
        state: 跨轮复用的会话与任务状态。
        message: 用户本轮实际输入。

    Returns:
        协调者 handle_turn 的结构化结果。
    """
    return asyncio.run(coordinator.handle_turn(state, message))


def test_write_completes_only_after_next_turn_confirmation_and_verification(runtime, backend):
    """首轮准备不能报完成，下一轮真实确认后提交并由核验代码设为 done。"""

    def script(state):
        """首轮验证身份并派工，第二轮确认上一回复中的唯一操作。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#1")]})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()
        yield tool("complete_plan", {})
        yield reply()

    coordinator, state, gateway, _, _ = runtime(script, cancel_steps("#1"))
    first = turn(coordinator, state, "我是 u1@example.com，请取消订单 #1")
    assert first["tasks"][0]["status"] == "pending_confirm"
    assert len(first["displayed"]) == 1
    assert "已全部成功办理" not in first["reply"]
    assert backend.export_snapshot()["orders"]["#1"]["status"] == "pending"
    second = turn(coordinator, state, "好的")
    assert second["tasks"][0]["status"] == "done"
    assert backend.export_snapshot()["orders"]["#1"]["status"] == "cancelled"
    assert any(
        record["kind"] == "verification" and record["status"] == "verified"
        for record in gateway.records
    )
    assert len(second["confirmation_results"]) == 1


def test_same_order_requests_aggregate_items_and_confirm_actions_one_by_one(runtime, backend):
    """同单商品诉求先合并，不同动作依次准备、展示、确认与核验。"""

    def script(state):
        """先改地址，地址确认后才派出已汇总的商品变更任务。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool(
            "update_tasks",
            {
                "changes": [
                    add_task("modify_pending_order_address", "#1", "收货地址改为北京新地址"),
                    add_task("modify_pending_order_items", "#1", "衬衫从 a 换成 b"),
                    add_task("modify_pending_order_items", "#1", "鞋从 x 换成 y，差价用 gift 支付"),
                ]
            },
        )
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state, status="pending")})
        yield reply()
        yield tool("complete_plan", {})
        yield reply()

    child_script = prepare_steps(
        "modify_pending_order_address", {"order_id": "#1", **ADDRESS}
    ) + prepare_steps(
        "modify_pending_order_items",
        {
            "order_id": "#1",
            "item_ids": ["a", "x"],
            "new_item_ids": ["b", "y"],
            "payment_method_id": "gift",
        },
    )
    coordinator, state, _, _, _ = runtime(script, child_script)
    first = turn(coordinator, state, "先改地址，并把衬衫和鞋换规格")
    assert len(first["tasks"]) == 2
    items_task = next(
        task for task in first["tasks"] if task["action"] == "modify_pending_order_items"
    )
    assert "衬衫" in items_task["instruction"] and "鞋" in items_task["instruction"]
    assert items_task["status"] == "pending"
    assert len(first["displayed"]) == 1
    second = turn(coordinator, state, "确认")
    assert [task["status"] for task in second["tasks"]] == ["done", "pending_confirm"]
    assert len(second["displayed"]) == 1
    snapshot = backend.export_snapshot()
    assert snapshot["orders"]["#1"]["address"] == ADDRESS
    assert [item["item_id"] for item in snapshot["orders"]["#1"]["items"]] == ["a", "x"]
    third = turn(coordinator, state, "确认")
    assert all(task["status"] == "done" for task in third["tasks"])
    assert [item["item_id"] for item in backend.export_snapshot()["orders"]["#1"]["items"]] == [
        "b",
        "y",
    ]


def test_multiple_orders_require_explicit_scope_and_report_partial_commit_failure(runtime, backend):
    """多个操作上的“好的”不授权，全部确认时逐项提交并保留部分失败结果。"""

    def script(state):
        """两单同时准备，连续两轮只完成计划并按代码状态回复。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#1"), cancel_task("#4")]})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()
        yield tool("complete_plan", {})
        yield reply()
        yield tool("complete_plan", {})
        yield reply()

    coordinator, state, _, _, _ = runtime(script, cancel_steps("#1") + cancel_steps("#4"))
    first = turn(coordinator, state, "取消 #1 和 #4")
    assert len(first["displayed"]) == 2
    ambiguous = turn(coordinator, state, "好的")
    assert all(task["status"] == "pending_confirm" for task in ambiguous["tasks"])
    assert all(
        backend.export_snapshot()["orders"][order_id]["status"] == "pending"
        for order_id in ("#1", "#4")
    )
    with backend.engine.begin() as connection:
        order = connection.execute(select(orders.c.data).where(orders.c.id == "#4")).scalar_one()
        order["status"] = "delivered"
        connection.execute(update(orders).where(orders.c.id == "#4").values(data=order))
    confirmed = turn(coordinator, state, "全部确认")
    tasks = by_order(confirmed)
    assert tasks["#1"]["status"] == "done"
    assert tasks["#4"]["status"] == "failed"
    assert tasks["#4"]["reason_internal"] is False
    assert tasks["#4"]["reason"] in confirmed["reply"]
    assert len(confirmed["confirmation_results"]) == 2
    assert backend.export_snapshot()["orders"]["#1"]["status"] == "cancelled"
    assert backend.export_snapshot()["orders"]["#4"]["status"] == "delivered"


def test_changed_request_invalidates_old_operation_before_confirmation(runtime, backend):
    """含同意与改口的当前消息先应用修订，不能提交旧回复展示的过时操作。"""

    def script(state):
        """第二轮将原任务修改到另一订单后，再完成计划和重新准备。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#1")]})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()
        yield tool(
            "update_tasks",
            {
                "changes": [
                    {**cancel_task("#4"), "operation": "update", "task_id": task_ids(state)[0]}
                ]
            },
        )
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()

    coordinator, state, _, _, _ = runtime(script, cancel_steps("#1") + cancel_steps("#4"))
    first = turn(coordinator, state, "取消 #1")
    old_operation = first["tasks"][0]["operation_ids"][0]
    second = turn(coordinator, state, "好的，不过改为取消 #4，#1 保留")
    assert backend.get_operation("u1", operation_id=old_operation)["status"] == "rejected"
    assert second["tasks"][0]["revision"] == 2
    assert second["tasks"][0]["status"] == "pending_confirm"
    assert second["tasks"][0]["order_id"] == "#4"
    assert all(
        backend.export_snapshot()["orders"][order_id]["status"] == "pending"
        for order_id in ("#1", "#4")
    )


def test_operation_omitted_from_latest_reply_cannot_use_later_confirmation(runtime, backend):
    """旧待确认任务保留，但仅最近一次回复展示的操作能获得下一轮授权。"""

    def script(state):
        """第二次回复只展示 #4，第三轮确认后也不重新展示 #1。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#1"), cancel_task("#4")]})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()
        yield tool("complete_plan", {})
        yield reply([task.id for task in state.tasks.tasks.values() if task.order_id == "#4"])
        yield tool("complete_plan", {})
        yield reply([])

    coordinator, state, _, _, _ = runtime(script, cancel_steps("#1") + cancel_steps("#4"))
    turn(coordinator, state, "取消 #1 和 #4")
    second = turn(coordinator, state, "好的")
    assert len(second["displayed"]) == 1
    third = turn(coordinator, state, "好的")
    tasks = by_order(third)
    assert tasks["#1"]["status"] == "pending_confirm"
    assert tasks["#4"]["status"] == "done"
    assert third["displayed"] == []
    assert backend.export_snapshot()["orders"]["#1"]["status"] == "pending"
    assert backend.export_snapshot()["orders"]["#4"]["status"] == "cancelled"


def test_user_rejection_and_cancelled_conditional_fallback_do_not_write(runtime, backend):
    """用户拒绝当前操作并撤销条件备用任务后，不再准备或提交替代动作。"""

    def script(state):
        """先准备换货并等待，再撤销退货备用任务并判定用户拒绝。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool(
            "update_tasks",
            {
                "changes": [
                    add_task("exchange_delivered_order_items", "#2", "衬衫 a 换成 b"),
                    add_task(
                        "return_delivered_order_items",
                        "#2",
                        "无法换货时退货",
                        condition="换货无法办理",
                    ),
                ]
            },
        )
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state, status="pending")})
        yield reply()
        fallback = next(task for task in state.tasks.tasks.values() if task.condition)
        yield tool(
            "update_tasks",
            {
                "changes": [
                    {"operation": "cancel", "task_id": fallback.id, "reason": "用户不再需要售后"}
                ]
            },
        )
        yield tool("complete_plan", {})
        yield reply()

    child_script = prepare_steps(
        "exchange_delivered_order_items",
        {"order_id": "#2", "item_ids": ["a"], "new_item_ids": ["b"], "payment_method_id": "card"},
        skill="exchange_delivered_order",
    )
    coordinator, state, _, _, _ = runtime(script, child_script)
    first = turn(coordinator, state, "换货，不行就退货")
    assert [task["status"] for task in first["tasks"]] == ["pending_confirm", "conditional"]
    second = turn(coordinator, state, "不用了")
    assert all(task["status"] == "cancelled" for task in second["tasks"])
    assert backend.export_snapshot()["orders"]["#2"]["status"] == "delivered"


def test_unverified_dispatch_and_model_owned_completion_are_blocked(runtime, backend):
    """未验证身份不能建任务或派工，模型不能通过状态字段或 commit 工具写入。"""

    def script(state):
        """尝试越权后使用合法工具继续，使护栏拒绝本身不丢失本轮任务。"""
        yield tool("update_tasks", {"changes": [cancel_task("#1")]})
        assert state.tasks.tasks == {}
        yield tool("dispatch_tasks", {"task_ids": ["forged"]})
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [{**cancel_task("#1"), "status": "done"}]})
        assert state.tasks.tasks == {}
        yield tool("commit", {"operation_id": "forged", "expected_args": {}})
        yield tool("update_tasks", {"changes": [cancel_task("#1")]})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()

    coordinator, state, gateway, _, _ = runtime(script, cancel_steps("#1"))
    result = turn(coordinator, state, "取消 #1")
    assert len(result["tasks"]) == 1
    assert result["tasks"][0]["status"] == "pending_confirm"
    assert not any(record["kind"] == "commit" for record in gateway.records)
    assert backend.export_snapshot()["orders"]["#1"]["status"] == "pending"


def test_one_subagent_failure_does_not_abort_another_order(runtime, backend):
    """一单准备后模型异常会清理其操作，另一订单仍可正常准备并展示。"""

    def script(state):
        """同时分派两个订单，在各任务独立结束后按实际状态回复。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#1"), cancel_task("#4")]})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()

    coordinator, state, gateway, _, _ = runtime(
        script, [*cancel_steps("#1")[:-1], LLMError("第一单模型失败"), *cancel_steps("#4")]
    )
    result = turn(coordinator, state, "取消 #1 和 #4")
    tasks = by_order(result)
    assert tasks["#1"]["status"] == "failed"
    assert tasks["#4"]["status"] == "pending_confirm"
    assert len(result["displayed"]) == 1
    prepared = next(record for record in gateway.records if record.get("status") == "prepared")
    assert (
        backend.get_operation("u1", operation_id=prepared["operation_id"])["status"] == "rejected"
    )
    assert all(
        backend.export_snapshot()["orders"][order_id]["status"] == "pending"
        for order_id in ("#1", "#4")
    )


def test_subagent_cannot_prepare_another_order_under_current_task(runtime, backend):
    """子 Agent 即使获准操作另一张本人订单，也不能把它挂到错误任务中等待确认。"""

    def script(state):
        """只分派 #1 请求，子 Agent 的 #4 操作由结果关联检查拒绝。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#1")]})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()

    coordinator, state, gateway, _, _ = runtime(script, cancel_steps("#4"))
    result = turn(coordinator, state, "取消 #1")
    assert result["tasks"][0]["status"] == "failed"
    assert result["displayed"] == []
    prepared = next(record for record in gateway.records if record.get("status") == "prepared")
    assert (
        backend.get_operation("u1", operation_id=prepared["operation_id"])["status"] == "rejected"
    )
    assert backend.export_snapshot()["orders"]["#4"]["status"] == "pending"


def test_unfinished_task_cannot_reply_and_fails_at_round_limit(runtime, backend):
    """仍可推进的任务阻止提前回复，协调决策超限后明确失败并记录护栏。"""

    def script(state):
        """建立任务后跳过分派直接请求回复，以验证未完成检查和预算退出。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#1")]})
        yield tool("complete_plan", {})
        yield reply()

    coordinator, state, gateway, fake, child = runtime(script, [], max_rounds=4)
    result = turn(coordinator, state, "取消 #1")
    assert result["rounds"] == 4
    assert len(fake.requests) == 4
    assert child.requests == []
    assert result["tasks"][0]["status"] == "failed"
    assert "上限" in result["tasks"][0]["reason"]
    assert backend.export_snapshot()["orders"]["#1"]["status"] == "pending"
    events = [json.loads(line) for line in gateway.trace.path.read_text().splitlines()]
    assert any(
        event["kind"] == "guard"
        and event["name"] == "coordinator_limit"
        and event["status"] == "blocked"
        for event in events
    )


def test_removing_one_of_two_displayed_operations_does_not_expand_ambiguous_consent(
    runtime, backend
):
    """已展示两项时，即使本轮撤销一项，“好的”仍不能自动授权剩余项。"""

    def script(state):
        """第二轮先撤销 #1，再用完整的上一回复展示范围判定简短同意。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#1"), cancel_task("#4")]})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()
        first = next(task for task in state.tasks.tasks.values() if task.order_id == "#1")
        yield tool(
            "update_tasks",
            {"changes": [{"operation": "cancel", "task_id": first.id, "reason": "撤销第一项"}]},
        )
        yield tool("complete_plan", {})
        yield reply()

    coordinator, state, _, _, _ = runtime(script, cancel_steps("#1") + cancel_steps("#4"))
    first = turn(coordinator, state, "取消 #1 和 #4")
    assert len(first["displayed"]) == 2
    second = turn(coordinator, state, "好的")
    tasks = by_order(second)
    assert tasks["#1"]["status"] == "cancelled"
    assert tasks["#4"]["status"] == "pending_confirm"
    assert all(
        backend.export_snapshot()["orders"][order_id]["status"] == "pending"
        for order_id in ("#1", "#4")
    )


@pytest.mark.parametrize(
    ("query", "args", "expected_status"),
    [
        ("get_order_details", {"order_id": "#1"}, "done"),
        ("get_order_details", {"order_id": "#4"}, "failed"),
        ("get_user_details", {"user_id": "u1"}, "failed"),
    ],
)
def test_read_task_requires_facts_for_its_own_action_and_order(
    runtime, query, args, expected_status
):
    """查询有结果还不够，工具类型和目标订单必须匹配当前只读任务。"""

    def script(state):
        """分派 #1 订单查询任务，代码检查返回事实是否回答了这个请求。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool(
            "update_tasks",
            {"changes": [add_task("get_order_details", "#1", "查询订单 #1 的状态")]},
        )
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()

    coordinator, state, _, _, _ = runtime(
        script,
        [tool(query, args), tool("finish", {"missing": [], "message_hint": "订单已查询"})],
    )
    result = turn(coordinator, state, "查询 #1")
    assert result["tasks"][0]["status"] == expected_status
    assert result["tasks"][0]["facts"]
    assert result["tasks"][0]["operation_ids"] == []
    assert result["displayed"] == []


@pytest.mark.parametrize("error_type", [asyncio.CancelledError, OSError])
def test_interruption_after_prepare_keeps_failed_task_and_operation_evidence(
    runtime, backend, error_type
):
    """取消或存取异常向外传播，同时作废准备项并保留失败任务的操作证据。"""

    def script(state):
        """建立并分派任务，子 Agent 准备之后的异常终止本轮处理。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#1")]})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})

    coordinator, state, _, _, child = runtime(script, cancel_steps("#1"))
    original_send = child.send

    async def interrupt_after_prepare(model, request, timeout_seconds):
        """加载和准备正常执行，在结束响应前模拟取消或持久化存取异常。"""
        if len(child.requests) == 2:
            raise error_type("模拟准备后的中断")
        return await original_send(model, request, timeout_seconds)

    child.send = interrupt_after_prepare
    with pytest.raises(error_type):
        turn(coordinator, state, "取消 #1")
    task = next(iter(state.tasks.tasks.values()))
    assert task.status == "failed"
    assert task.reason
    assert len(task.operation_ids) == 1
    assert task.actions
    operation = backend.get_operation("u1", operation_id=task.operation_ids[0])
    assert operation["status"] == "rejected"
    assert operation["args"]["order_id"] == "#1"
    assert backend.export_snapshot()["orders"]["#1"]["status"] == "pending"


def test_coordinator_read_failure_after_subagent_finish_preserves_and_cleans_operation(
    runtime, backend, monkeypatch
):
    """子 Agent 正常返回后对账失败，任务仍应退出 running 并保留和清理准备项。"""

    def script(state):
        """分派真实写任务，由协调代码接收结果时发生一次存取异常。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#1")]})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})

    coordinator, state, gateway, _, child = runtime(script, cancel_steps("#1"))
    original_get_operation = gateway.get_operation
    failed = False

    def fail_first_coordinator_read(context, operation_id):
        """只中断协调者的首次操作读取，子 Agent 的结束对账和后续清理照常执行。"""
        nonlocal failed
        if context.capability == "coordinator" and not failed:
            failed = True
            raise OSError("模拟协调者首次读取操作失败")
        return original_get_operation(context, operation_id)

    monkeypatch.setattr(gateway, "get_operation", fail_first_coordinator_read)
    with pytest.raises(OSError, match="首次读取操作失败"):
        turn(coordinator, state, "取消 #1")
    assert failed is True
    assert len(child.requests) == 3
    task = next(iter(state.tasks.tasks.values()))
    assert task.status == "failed"
    prepared = next(record for record in gateway.records if record.get("status") == "prepared")
    assert task.operation_ids == [prepared["operation_id"]]
    assert task.actions
    operation = backend.get_operation("u1", operation_id=task.operation_ids[0])
    assert operation["status"] == "rejected"
    assert backend.export_snapshot()["orders"]["#1"]["status"] == "pending"


def test_round_limit_explains_stop_and_keeps_prepared_operation_for_confirmation(runtime, backend):
    """最后一轮成功准备操作后达到上限，回复说明原因并保留可确认的准备项。"""

    def script(state):
        """首轮预算止于分派，第二个用户轮次仍可确认已经展示的准备项。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#1")]})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield tool("complete_plan", {})
        yield reply()

    coordinator, state, _, _, _ = runtime(script, cancel_steps("#1"), max_rounds=4)
    first = turn(coordinator, state, "取消 #1")
    assert first["rounds"] == 4
    assert first["tasks"][0]["status"] == "pending_confirm"
    assert "上限" not in first["reply"]
    assert "转人工" in first["reply"]
    assert len(first["displayed"]) == 1
    operation_id = first["tasks"][0]["operation_ids"][0]
    assert backend.get_operation("u1", operation_id=operation_id)["status"] == "prepared"
    second = turn(coordinator, state, "好的")
    assert second["tasks"][0]["status"] == "done"
    assert backend.get_operation("u1", operation_id=operation_id)["status"] == "committed"


def test_cancellation_before_subagent_start_does_not_leave_selected_task_running(runtime, backend):
    """选择任务后、子协程启动前取消，未执行的任务也必须退出 running。"""

    def script(state):
        """产生分派响应时安排外层取消，使取消先于新建子协程运行。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#1")]})
        yield tool("complete_plan", {})
        asyncio.get_running_loop().call_soon(asyncio.current_task().cancel)
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})

    coordinator, state, _, _, child = runtime(script, cancel_steps("#1"))
    with pytest.raises(asyncio.CancelledError):
        turn(coordinator, state, "取消 #1")
    task = next(iter(state.tasks.tasks.values()))
    assert child.requests == []
    assert task.status == "failed"
    assert task.reason
    assert task.operation_ids == []
    assert backend.export_snapshot()["orders"]["#1"]["status"] == "pending"


@pytest.mark.parametrize("plain_text", [True, False])
def test_unverified_identity_request_replies_in_one_round(runtime, plain_text):
    """真实运行中的身份询问可直接回复，纯文本和 reply 工具都不触发格式拒绝。"""
    message = "您好，我可以帮您办理换货。请先提供您的邮箱，或姓名和邮编，以便验证身份。"

    def script(state):
        """复现首轮未提供身份时模型生成的合理询问。"""
        yield text_response(message) if plain_text else tool("reply", {"message_hint": message})

    coordinator, state, gateway, fake, child = runtime(script, [])
    result = turn(coordinator, state, "我想换一下收到的商品")
    assert result["reply"] == message
    assert result["rounds"] == 1 and result["tasks"] == []
    assert len(fake.requests) == 1 and child.requests == []
    tools = {item["function"]["name"] for item in fake.requests[0]["tools"]}
    assert "reply" in tools and "complete_plan" not in tools
    assert state.session.user_id is None and gateway.records == []
    events = [json.loads(line) for line in gateway.trace.path.read_text().splitlines()]
    assert not any(event["kind"] == "guard" for event in events)


def test_unverified_complete_plan_cannot_authorize_later_dispatch(runtime):
    """缺少身份时即使模型伪造 complete_plan，也不能提前完成确认阶段。"""

    def script(state):
        """伪造计划结束后再验证身份，直接分派仍须被原有顺序护栏拒绝。"""
        yield tool("complete_plan", {})
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#1")]})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()

    coordinator, state, gateway, _, child = runtime(script, cancel_steps("#1"))
    result = turn(coordinator, state, "邮箱 u1@example.com，请取消 #1")
    assert result["tasks"][0]["status"] == "pending_confirm"
    assert len(child.requests) == 3
    reasons = [
        event["attrs"].get("reason", "")
        for event in map(json.loads, gateway.trace.path.read_text().splitlines())
        if event["kind"] == "guard"
    ]
    assert "请先验证用户身份" in reasons
    assert "先提出所有任务修改并调用 complete_plan，再分派" in reasons


def test_verified_plain_text_requires_complete_plan(runtime):
    """已验证后直接输出文本仍被拒绝，完成计划后才能按同一回复路径返回。"""
    message = "请告诉我您需要办理的业务。"

    def script(state):
        """先违反计划顺序，再按原有协议结束计划并返回相同客服文本。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield text_response(message)
        yield tool("complete_plan", {})
        yield text_response(message)

    coordinator, state, gateway, fake, _ = runtime(script, [])
    result = turn(coordinator, state, "我的邮箱是 u1@example.com")
    assert result["reply"] == message and result["rounds"] == 4
    assert len(fake.requests) == 4 and result["tasks"] == []
    events = [json.loads(line) for line in gateway.trace.path.read_text().splitlines()]
    assert (
        sum(event["name"] == "coordinator_format" and event["event"] == "end" for event in events)
        == 1
    )


@pytest.mark.parametrize("plain_text", [True, False])
def test_actionable_task_cannot_reply_early_and_internal_reason_stays_hidden(runtime, plain_text):
    """纯文本和工具都不能跳过待办任务，超限原因在下一轮仍只保留于状态和轨迹。"""

    def script(state):
        """首轮不分派使任务超限，下一轮只读取其失败状态。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#1")]})
        yield tool("complete_plan", {})
        yield text_response("已经办理完成。") if plain_text else reply()
        yield tool("complete_plan", {})
        yield reply()

    coordinator, state, gateway, _, child = runtime(script, [], max_rounds=4)
    first = turn(coordinator, state, "请取消 #1")
    task = first["tasks"][0]
    assert task["status"] == "failed" and task["reason_internal"] is True
    assert "达到本轮上限 4" in task["reason"]
    second = turn(coordinator, state, "请告诉我办理情况")
    assert second["tasks"][0]["reason"] == task["reason"]
    assert child.requests == []
    for result in (first, second):
        assert STOP_NOTICE in result["reply"]
        assert task["reason"] not in result["reply"]
        assert "已经办理完成" not in result["reply"]
    events = [json.loads(line) for line in gateway.trace.path.read_text().splitlines()]
    assert any(event["attrs"].get("reason") == task["reason"] for event in events)


def test_dispatched_query_receives_verified_identity(runtime):
    """查询子 Agent 收到协调者已验证的用户标识，直接查询后由代码生成真实结果。"""

    def script(state):
        """验证身份后分派一项读取订单的任务。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [add_task("get_order_details", "#1", "查询订单")]})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()

    coordinator, state, _, _, child = runtime(
        script,
        [
            tool("get_order_details", {"order_id": "#1"}),
            tool("finish", {"missing": [], "message_hint": "已查询订单"}),
        ],
    )
    result = turn(coordinator, state, "邮箱 u1@example.com，帮我查询订单 #1")
    instruction = json.loads(child.requests[0]["messages"][1]["content"])["instruction"]
    assert "用户身份已由协调者验证为 u1，不要再要求用户验证身份" in instruction
    assert result["tasks"][0]["status"] == "done"
    assert result["tasks"][0]["missing"] == []


@pytest.mark.parametrize("operation", ["update", "cancel"])
def test_task_change_clears_internal_reason_visibility(runtime, operation):
    """失败任务修改或撤销后清除旧内部标记，新的业务说明正常可见。"""

    def script(state):
        """先使待办超限，后续按用户改口更新或撤销同一任务。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#1")]})
        task_id = task_ids(state)[0]
        change = {"operation": operation, "task_id": task_id}
        if operation == "cancel":
            change["reason"] = "用户决定保留订单"
        else:
            change.update({**cancel_task("#1"), "operation": "update", "task_id": task_id})
        yield tool("update_tasks", {"changes": [change]})
        yield tool("complete_plan", {})
        if operation == "update":
            yield tool("dispatch_tasks", {"task_ids": [task_id]})
        yield reply()

    coordinator, state, _, _, _ = runtime(script, cancel_steps("#1"), max_rounds=2)
    first = turn(coordinator, state, "取消订单 #1")
    assert first["tasks"][0]["reason_internal"] is True
    coordinator.max_rounds = 4
    second = turn(
        coordinator, state, "按刚才说的重新办理" if operation == "update" else "先保留订单"
    )
    assert second["tasks"][0]["reason_internal"] is False
    assert STOP_NOTICE not in second["reply"]
    if operation == "cancel":
        assert "用户决定保留订单" in second["reply"]
    else:
        assert second["tasks"][0]["reason"] is None
        assert len(second["displayed"]) == 1


def test_model_cannot_write_internal_reason_flag(runtime):
    """任务原因可见性属于代码，模型增改提议不能伪造这个字段。"""

    def script(state):
        """向合法新增任务混入代码字段，随后按空任务正常结束。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [{**cancel_task("#1"), "reason_internal": False}]})
        yield tool("complete_plan", {})
        yield reply()

    coordinator, state, gateway, _, _ = runtime(script, [])
    result = turn(coordinator, state, "请取消 #1")
    assert result["tasks"] == []
    events = [json.loads(line) for line in gateway.trace.path.read_text().splitlines()]
    assert any(
        event["name"] == "coordinator_tool"
        and "reason_internal" in event["attrs"].get("reason", "")
        for event in events
    )


def test_older_task_snapshot_defaults_to_visible_business_reason():
    """旧检查点未保存原因可见性字段时，仍能还原任务及原有业务说明。"""
    task = Task(
        **{
            "id": "old-task",
            "capability": "order_service",
            "action": "cancel_pending_order",
            "instruction": "取消订单 #1",
            "order_id": "#1",
            "status": "failed",
            "reason": "订单已发货，不能取消",
        }
    )
    assert task.reason_internal is False
    assert task.reason == "订单已发货，不能取消"


def test_coordinator_replays_tool_reasoning_and_preserves_raw_record(runtime):
    """思考模式工具响应连同工具结果回传，原始响应也完整保存在模型记录中。"""
    raw = tool("find_user_id_by_email", {"email": "u1@example.com"})
    raw["choices"][0]["message"]["reasoning_content"] = "已收到邮箱，先查询并验证身份。"

    def script(state):
        """用带思考字段的身份调用进入已验证回复流程。"""
        yield raw
        yield tool("complete_plan", {})
        yield reply()

    coordinator, state, gateway, fake, _ = runtime(script, [])
    turn(coordinator, state, "邮箱 u1@example.com")
    original = raw["choices"][0]["message"]
    messages = fake.requests[1]["messages"]
    index = messages.index(original)
    assert messages[index + 1]["role"] == "tool"
    assert messages[index + 1]["tool_call_id"] == original["tool_calls"][0]["id"]
    assert json.loads(messages[index + 1]["content"]) == "u1"
    records = [
        json.loads(path.read_text()) for path in (gateway.trace.run_dir / "models").glob("*.json")
    ]
    assert any(record["response"] == raw for record in records)

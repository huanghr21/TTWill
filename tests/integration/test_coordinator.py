"""用脚本化协调者、真实子 Agent 和 SQLite 验证多轮办理及写操作护栏。"""

import asyncio
import json

import pytest
from sqlalchemy import select, update

from ttwill.agents import build_registry
from ttwill.backend.store import orders
from ttwill.config import AppConfig, LLMConfig, ModelConfig
from ttwill.core.confirmation import ConfirmationJudge
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
        tool("finish", {"missing": [], "note": "已全部成功办理"}),
    ]


def cancel_steps(order_id: str) -> list[dict]:
    """生成取消订单并保留待确认状态的子 Agent 脚本。"""
    return prepare_steps(
        "cancel_pending_order", {"order_id": order_id, "reason": "no longer needed"}
    )


def reply(show_task_ids: list[str] | None = None) -> dict:
    """请求回复并指定展示范围，默认展示全部待确认操作。"""
    return tool("reply", {"message": "已全部成功办理", "show_task_ids": show_task_ids})


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
    """为每项测试连接真实业务库、子 Agent 和三个独立假模型。

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

        def judgments():
            """为现有场景的固定原话提供独立判定，未登记输入立即使测试失败。"""
            expected = {
                ("好的", 1): "confirm",
                ("好的", 2): "unrelated",
                ("确认", 1): "confirm",
                ("全部确认", 1): "confirm",
                ("全部确认", 2): "confirm",
                ("不用了", 1): "reject",
                ("不同意", 1): "reject",
            }
            consumed = 0
            while len(judge_fake.requests) > consumed:
                data = json.loads(judge_fake.requests[consumed]["messages"][-1]["content"])
                consumed += 1
                key = data["user_message"], len(data["displayed"])
                assert key in expected, f"未配置的确认场景：{key}"
                assert all(
                    shown["content"] in data["assistant_reply"] for shown in data["displayed"]
                )
                yield text_response(
                    json.dumps(
                        [
                            {
                                "operation_id": shown["operation_id"],
                                "decision": expected[key],
                                "reason": "场景预设的明确授权范围",
                            }
                            for shown in data["displayed"]
                        ],
                        ensure_ascii=False,
                    )
                )

        judge_fake = FakeModel(judgments())
        llm = LLMClient(config, trace, coordinator_fake)
        runner = SubAgentRunner(
            build_registry(skill_catalog),
            gateway,
            LLMClient(config, trace, child_fake),
            max_steps=8,
        )
        coordinator = Coordinator(
            gateway,
            runner,
            llm,
            max_rounds=max_rounds,
            judge=ConfirmationJudge(LLMClient(config, trace, judge_fake)),
        )
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
    """模型正文不构成完成证据，下一轮真实确认后提交并由核验代码设为 done。"""

    def script(state):
        """首轮验证身份并派工，第二轮确认上一回复中的唯一操作。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()
        yield tool("complete_plan", {})
        yield reply()

    coordinator, state, gateway, _, _ = runtime(script, cancel_steps("#W0000001"))
    first = turn(coordinator, state, "我是 u1@example.com，请取消订单 #W0000001")
    assert first["tasks"][0]["status"] == "pending_confirm"
    assert len(first["displayed"]) == 1
    assert first["reply"].startswith("已全部成功办理")
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "pending"
    second = turn(coordinator, state, "好的")
    assert second["tasks"][0]["status"] == "done"
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "cancelled"
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
                    add_task("modify_pending_order_address", "#W0000001", "收货地址改为北京新地址"),
                    add_task("modify_pending_order_items", "#W0000001", "衬衫从 a 换成 b"),
                    add_task(
                        "modify_pending_order_items", "#W0000001", "鞋从 x 换成 y，差价用 gift 支付"
                    ),
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
        "modify_pending_order_address", {"order_id": "#W0000001", **ADDRESS}
    ) + prepare_steps(
        "modify_pending_order_items",
        {
            "order_id": "#W0000001",
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
    assert snapshot["orders"]["#W0000001"]["address"] == ADDRESS
    assert [item["item_id"] for item in snapshot["orders"]["#W0000001"]["items"]] == ["a", "x"]
    third = turn(coordinator, state, "确认")
    assert all(task["status"] == "done" for task in third["tasks"])
    assert [
        item["item_id"] for item in backend.export_snapshot()["orders"]["#W0000001"]["items"]
    ] == [
        "b",
        "y",
    ]


def test_multiple_orders_require_explicit_scope_and_report_partial_commit_failure(runtime, backend):
    """多个操作上的“好的”不授权，全部确认时逐项提交并保留部分失败结果。"""

    def script(state):
        """两单同时准备，连续两轮只完成计划并按代码状态回复。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool(
            "update_tasks", {"changes": [cancel_task("#W0000001"), cancel_task("#W0000004")]}
        )
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()
        yield tool("complete_plan", {})
        yield reply()
        yield tool("complete_plan", {})
        yield reply()

    coordinator, state, _, _, _ = runtime(
        script, cancel_steps("#W0000001") + cancel_steps("#W0000004")
    )
    first = turn(coordinator, state, "取消 #W0000001 和 #W0000004")
    assert len(first["displayed"]) == 2
    ambiguous = turn(coordinator, state, "好的")
    assert all(task["status"] == "pending_confirm" for task in ambiguous["tasks"])
    assert all(
        backend.export_snapshot()["orders"][order_id]["status"] == "pending"
        for order_id in ("#W0000001", "#W0000004")
    )
    with backend.engine.begin() as connection:
        order = connection.execute(
            select(orders.c.data).where(orders.c.id == "#W0000004")
        ).scalar_one()
        order["status"] = "delivered"
        connection.execute(update(orders).where(orders.c.id == "#W0000004").values(data=order))
    confirmed = turn(coordinator, state, "全部确认")
    tasks = by_order(confirmed)
    assert tasks["#W0000001"]["status"] == "done"
    assert tasks["#W0000004"]["status"] == "failed"
    assert tasks["#W0000004"]["reason_internal"] is False
    assert confirmed["reply"] == "已全部成功办理"
    assert len(confirmed["confirmation_results"]) == 2
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "cancelled"
    assert backend.export_snapshot()["orders"]["#W0000004"]["status"] == "delivered"


def test_changed_request_invalidates_old_operation_before_confirmation(runtime, backend):
    """含同意与改口的当前消息先应用修订，不能提交旧回复展示的过时操作。"""

    def script(state):
        """第二轮将原任务修改到另一订单后，再完成计划和重新准备。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()
        yield tool(
            "update_tasks",
            {
                "changes": [
                    {
                        **cancel_task("#W0000004"),
                        "operation": "update",
                        "task_id": task_ids(state)[0],
                    }
                ]
            },
        )
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()

    coordinator, state, _, _, _ = runtime(
        script, cancel_steps("#W0000001") + cancel_steps("#W0000004")
    )
    first = turn(coordinator, state, "取消 #W0000001")
    old_operation = first["tasks"][0]["operation_ids"][0]
    second = turn(coordinator, state, "好的，不过改为取消 #W0000004，#W0000001 保留")
    assert backend.get_operation("u1", operation_id=old_operation)["status"] == "rejected"
    assert second["tasks"][0]["revision"] == 2
    assert second["tasks"][0]["status"] == "pending_confirm"
    assert second["tasks"][0]["order_id"] == "#W0000004"
    assert all(
        backend.export_snapshot()["orders"][order_id]["status"] == "pending"
        for order_id in ("#W0000001", "#W0000004")
    )


def test_operation_omitted_from_latest_reply_cannot_use_later_confirmation(runtime, backend):
    """旧待确认任务保留，但仅最近一次回复展示的操作能获得下一轮授权。"""

    def script(state):
        """第二次回复只展示 #W0000004，第三轮确认后也不重新展示 #W0000001。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool(
            "update_tasks", {"changes": [cancel_task("#W0000001"), cancel_task("#W0000004")]}
        )
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()
        yield tool("complete_plan", {})
        yield reply(
            [task.id for task in state.tasks.tasks.values() if task.order_id == "#W0000004"]
        )
        yield tool("complete_plan", {})
        yield reply([])

    coordinator, state, _, _, _ = runtime(
        script, cancel_steps("#W0000001") + cancel_steps("#W0000004")
    )
    turn(coordinator, state, "取消 #W0000001 和 #W0000004")
    second = turn(coordinator, state, "好的")
    assert len(second["displayed"]) == 1
    third = turn(coordinator, state, "好的")
    tasks = by_order(third)
    assert tasks["#W0000001"]["status"] == "pending_confirm"
    assert tasks["#W0000004"]["status"] == "done"
    assert third["displayed"] == []
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "pending"
    assert backend.export_snapshot()["orders"]["#W0000004"]["status"] == "cancelled"


def test_user_rejection_and_cancelled_conditional_fallback_do_not_write(runtime, backend):
    """用户拒绝当前操作并撤销条件备用任务后，不再准备或提交替代动作。"""

    def script(state):
        """先准备换货并等待，再撤销退货备用任务并判定用户拒绝。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool(
            "update_tasks",
            {
                "changes": [
                    add_task("exchange_delivered_order_items", "#W0000002", "衬衫 a 换成 b"),
                    add_task(
                        "return_delivered_order_items",
                        "#W0000002",
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
        {
            "order_id": "#W0000002",
            "item_ids": ["a"],
            "new_item_ids": ["b"],
            "payment_method_id": "card",
        },
        skill="exchange_delivered_order",
    )
    coordinator, state, _, _, _ = runtime(script, child_script)
    first = turn(coordinator, state, "换货，不行就退货")
    assert [task["status"] for task in first["tasks"]] == ["pending_confirm", "conditional"]
    second = turn(coordinator, state, "不用了")
    assert all(task["status"] == "cancelled" for task in second["tasks"])
    assert backend.export_snapshot()["orders"]["#W0000002"]["status"] == "delivered"


def test_unverified_dispatch_and_model_owned_completion_are_blocked(runtime, backend):
    """未验证身份不能建任务或派工，模型不能通过状态字段或 commit 工具写入。"""

    def script(state):
        """尝试越权后使用合法工具继续，使护栏拒绝本身不丢失本轮任务。"""
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        assert state.tasks.tasks == {}
        yield tool("dispatch_tasks", {"task_ids": ["forged"]})
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [{**cancel_task("#W0000001"), "status": "done"}]})
        assert state.tasks.tasks == {}
        yield tool("commit", {"operation_id": "forged", "expected_args": {}})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()

    coordinator, state, gateway, _, _ = runtime(script, cancel_steps("#W0000001"))
    result = turn(coordinator, state, "取消 #W0000001")
    assert len(result["tasks"]) == 1
    assert result["tasks"][0]["status"] == "pending_confirm"
    assert not any(record["kind"] == "commit" for record in gateway.records)
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "pending"


def test_one_subagent_failure_does_not_abort_another_order(runtime, backend):
    """一单准备后模型异常会清理其操作，另一订单仍可正常准备并展示。"""

    def script(state):
        """同时分派两个订单，在各任务独立结束后按实际状态回复。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool(
            "update_tasks", {"changes": [cancel_task("#W0000001"), cancel_task("#W0000004")]}
        )
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()

    coordinator, state, gateway, _, _ = runtime(
        script,
        [*cancel_steps("#W0000001")[:-1], LLMError("第一单模型失败"), *cancel_steps("#W0000004")],
    )
    result = turn(coordinator, state, "取消 #W0000001 和 #W0000004")
    tasks = by_order(result)
    assert tasks["#W0000001"]["status"] == "failed"
    assert tasks["#W0000004"]["status"] == "pending_confirm"
    assert len(result["displayed"]) == 1
    prepared = next(record for record in gateway.records if record.get("status") == "prepared")
    assert (
        backend.get_operation("u1", operation_id=prepared["operation_id"])["status"] == "rejected"
    )
    assert all(
        backend.export_snapshot()["orders"][order_id]["status"] == "pending"
        for order_id in ("#W0000001", "#W0000004")
    )


def test_subagent_cannot_prepare_another_order_under_current_task(runtime, backend):
    """子 Agent 即使获准操作另一张本人订单，也不能把它挂到错误任务中等待确认。"""

    def script(state):
        """只分派 #W0000001 请求，子 Agent 的 #W0000004 操作由结果关联检查拒绝。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()

    coordinator, state, gateway, _, _ = runtime(script, cancel_steps("#W0000004"))
    result = turn(coordinator, state, "取消 #W0000001")
    assert result["tasks"][0]["status"] == "failed"
    assert result["displayed"] == []
    prepared = next(record for record in gateway.records if record.get("status") == "prepared")
    assert (
        backend.get_operation("u1", operation_id=prepared["operation_id"])["status"] == "rejected"
    )
    assert backend.export_snapshot()["orders"]["#W0000004"]["status"] == "pending"


def test_unfinished_task_cannot_reply_and_fails_at_round_limit(runtime, backend):
    """仍可推进的任务阻止提前回复，协调决策超限后明确失败并记录护栏。"""

    def script(state):
        """建立任务后跳过分派直接请求回复，以验证未完成检查和预算退出。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {})
        yield reply()

    coordinator, state, gateway, fake, child = runtime(script, [], max_rounds=4)
    result = turn(coordinator, state, "取消 #W0000001")
    assert result["rounds"] == 4
    assert len(fake.requests) == 4
    assert child.requests == []
    assert result["tasks"][0]["status"] == "failed"
    assert "上限" in result["tasks"][0]["reason"]
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "pending"
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
        """第二轮先撤销 #W0000001，再用完整的上一回复展示范围判定简短同意。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool(
            "update_tasks", {"changes": [cancel_task("#W0000001"), cancel_task("#W0000004")]}
        )
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()
        first = next(task for task in state.tasks.tasks.values() if task.order_id == "#W0000001")
        yield tool(
            "update_tasks",
            {"changes": [{"operation": "cancel", "task_id": first.id, "reason": "撤销第一项"}]},
        )
        yield tool("complete_plan", {})
        yield reply()

    coordinator, state, _, _, _ = runtime(
        script, cancel_steps("#W0000001") + cancel_steps("#W0000004")
    )
    first = turn(coordinator, state, "取消 #W0000001 和 #W0000004")
    assert len(first["displayed"]) == 2
    second = turn(coordinator, state, "好的")
    tasks = by_order(second)
    assert tasks["#W0000001"]["status"] == "cancelled"
    assert tasks["#W0000004"]["status"] == "pending_confirm"
    assert all(
        backend.export_snapshot()["orders"][order_id]["status"] == "pending"
        for order_id in ("#W0000001", "#W0000004")
    )


@pytest.mark.parametrize(
    ("query", "args", "expected_status"),
    [
        ("get_order_details", {"order_id": "#W0000001"}, "done"),
        ("get_order_details", {"order_id": "#W0000004"}, "failed"),
        ("get_user_details", {"user_id": "u1"}, "failed"),
    ],
)
def test_read_task_requires_facts_for_its_own_action_and_order(
    runtime, query, args, expected_status
):
    """查询有结果还不够，工具类型和目标订单必须匹配当前只读任务。"""

    def script(state):
        """分派 #W0000001 订单查询任务，代码检查返回事实是否回答了这个请求。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool(
            "update_tasks",
            {"changes": [add_task("get_order_details", "#W0000001", "查询订单 #W0000001 的状态")]},
        )
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()

    coordinator, state, _, _, _ = runtime(
        script,
        [tool(query, args), tool("finish", {"missing": [], "note": "订单已查询"})],
    )
    result = turn(coordinator, state, "查询 #W0000001")
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
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})

    coordinator, state, _, _, child = runtime(script, cancel_steps("#W0000001"))
    original_send = child.send

    async def interrupt_after_prepare(model, request, timeout_seconds):
        """加载和准备正常执行，在结束响应前模拟取消或持久化存取异常。"""
        if len(child.requests) == 2:
            raise error_type("模拟准备后的中断")
        return await original_send(model, request, timeout_seconds)

    child.send = interrupt_after_prepare
    with pytest.raises(error_type):
        turn(coordinator, state, "取消 #W0000001")
    task = next(iter(state.tasks.tasks.values()))
    assert task.status == "failed"
    assert task.reason
    assert len(task.operation_ids) == 1
    assert task.actions
    operation = backend.get_operation("u1", operation_id=task.operation_ids[0])
    assert operation["status"] == "rejected"
    assert operation["args"]["order_id"] == "#W0000001"
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "pending"


def test_coordinator_read_failure_after_subagent_finish_preserves_and_cleans_operation(
    runtime, backend, monkeypatch
):
    """子 Agent 正常返回后对账失败，任务仍应退出 running 并保留和清理准备项。"""

    def script(state):
        """分派真实写任务，由协调代码接收结果时发生一次存取异常。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})

    coordinator, state, gateway, _, child = runtime(script, cancel_steps("#W0000001"))
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
        turn(coordinator, state, "取消 #W0000001")
    assert failed is True
    assert len(child.requests) == 3
    task = next(iter(state.tasks.tasks.values()))
    assert task.status == "failed"
    prepared = next(record for record in gateway.records if record.get("status") == "prepared")
    assert task.operation_ids == [prepared["operation_id"]]
    assert task.actions
    operation = backend.get_operation("u1", operation_id=task.operation_ids[0])
    assert operation["status"] == "rejected"
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "pending"


def test_round_limit_explains_stop_and_keeps_prepared_operation_for_confirmation(runtime, backend):
    """最后一轮成功准备操作后达到上限，回复说明原因并保留可确认的准备项。"""

    def script(state):
        """首轮预算止于分派，第二个用户轮次仍可确认已经展示的准备项。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield tool("complete_plan", {})
        yield reply()

    coordinator, state, _, _, _ = runtime(script, cancel_steps("#W0000001"), max_rounds=4)
    first = turn(coordinator, state, "取消 #W0000001")
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
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {})
        asyncio.get_running_loop().call_soon(asyncio.current_task().cancel)
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})

    coordinator, state, _, _, child = runtime(script, cancel_steps("#W0000001"))
    with pytest.raises(asyncio.CancelledError):
        turn(coordinator, state, "取消 #W0000001")
    task = next(iter(state.tasks.tasks.values()))
    assert child.requests == []
    assert task.status == "failed"
    assert task.reason
    assert task.operation_ids == []
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "pending"


@pytest.mark.parametrize("plain_text", [True, False])
def test_unverified_identity_request_replies_in_one_round(runtime, plain_text):
    """真实运行中的身份询问可直接回复，纯文本和 reply 工具都不触发格式拒绝。"""
    message = "您好，我可以帮您办理换货。请先提供您的邮箱，或姓名和邮编，以便验证身份。"

    def script(state):
        """复现首轮未提供身份时模型生成的合理询问。"""
        yield text_response(message) if plain_text else tool("reply", {"message": message})

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
        """伪造计划结束后再验证身份，合法计划仍应能完成，未验证调用不写确认进度。"""
        yield tool("complete_plan", {})
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {"task_ids": task_ids(state)})
        yield reply()

    coordinator, state, gateway, _, child = runtime(script, cancel_steps("#W0000001"))
    result = turn(coordinator, state, "邮箱 u1@example.com，请取消 #W0000001")
    assert result["tasks"][0]["status"] == "pending_confirm"
    assert len(child.requests) == 3
    reasons = [
        event["attrs"].get("reason", "")
        for event in map(json.loads, gateway.trace.path.read_text().splitlines())
        if event["kind"] == "guard"
    ]
    assert "请先验证用户身份" in reasons
    assert "本轮已经判定过用户确认" not in reasons


def test_verified_plain_text_without_pending_confirmation_can_reply(runtime):
    """已验证但无待确认操作和可推进任务时，纯文本直接按回复处理。"""
    message = "请告诉我您需要办理的业务。"

    def script(state):
        """身份验证后直接输出客服文本，不插入空计划调用。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield text_response(message)

    coordinator, state, gateway, fake, _ = runtime(script, [])
    result = turn(coordinator, state, "我的邮箱是 u1@example.com")
    assert result["reply"] == message and result["rounds"] == 2
    assert len(fake.requests) == 2 and result["tasks"] == []
    events = [json.loads(line) for line in gateway.trace.path.read_text().splitlines()]
    assert (
        sum(event["name"] == "coordinator_format" and event["event"] == "end" for event in events)
        == 0
    )


@pytest.mark.parametrize("plain_text", [True, False])
def test_actionable_task_cannot_reply_early_and_internal_reason_stays_hidden(runtime, plain_text):
    """纯文本和工具都不能跳过待办任务，超限原因在下一轮仍只保留于状态和轨迹。"""

    def script(state):
        """首轮不分派使任务超限，下一轮只读取其失败状态。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {})
        yield text_response("已经办理完成。") if plain_text else reply()
        yield tool("complete_plan", {})
        yield reply()

    coordinator, state, gateway, _, child = runtime(script, [], max_rounds=4)
    first = turn(coordinator, state, "请取消 #W0000001")
    task = first["tasks"][0]
    assert task["status"] == "failed" and task["reason_internal"] is True
    assert "达到本轮上限 4" in task["reason"]
    second = turn(coordinator, state, "请告诉我办理情况")
    assert second["tasks"][0]["reason"] == task["reason"]
    assert child.requests == []
    assert first["reply"] == STOP_NOTICE
    assert second["reply"] == "已全部成功办理"
    for result in (first, second):
        assert task["reason"] not in result["reply"]
        assert "已经办理完成" not in result["reply"]
    events = [json.loads(line) for line in gateway.trace.path.read_text().splitlines()]
    assert any(event["attrs"].get("reason") == task["reason"] for event in events)


def test_dispatched_query_receives_verified_identity(runtime):
    """查询子 Agent 收到协调者已验证的用户标识，直接查询后由代码生成真实结果。"""

    def script(state):
        """验证身份后分派一项读取订单的任务。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool(
            "update_tasks", {"changes": [add_task("get_order_details", "#W0000001", "查询订单")]}
        )
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()

    coordinator, state, _, _, child = runtime(
        script,
        [
            tool("get_order_details", {"order_id": "#W0000001"}),
            tool("finish", {"missing": [], "note": "已查询订单"}),
        ],
    )
    result = turn(coordinator, state, "邮箱 u1@example.com，帮我查询订单 #W0000001")
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
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        task_id = task_ids(state)[0]
        change = {"operation": operation, "task_id": task_id}
        if operation == "cancel":
            change["reason"] = "用户决定保留订单"
        else:
            change.update({**cancel_task("#W0000001"), "operation": "update", "task_id": task_id})
            change["instruction"] = "取消订单 #W0000001，取消原因改为重复购买"
        yield tool("update_tasks", {"changes": [change]})
        yield tool("complete_plan", {})
        if operation == "update":
            yield tool("dispatch_tasks", {"task_ids": [task_id]})
        yield reply()

    coordinator, state, _, _, _ = runtime(script, cancel_steps("#W0000001"), max_rounds=2)
    first = turn(coordinator, state, "取消订单 #W0000001")
    assert first["tasks"][0]["reason_internal"] is True
    coordinator.max_rounds = 4
    second = turn(
        coordinator, state, "取消原因改成重复购买" if operation == "update" else "先保留订单"
    )
    assert second["tasks"][0]["reason_internal"] is False
    assert STOP_NOTICE not in second["reply"]
    if operation == "cancel":
        assert second["reply"] == "已全部成功办理"
        assert second["tasks"][0]["reason"] == "用户决定保留订单"
    else:
        assert second["tasks"][0]["reason"] is None
        assert len(second["displayed"]) == 1


def test_model_cannot_write_internal_reason_flag(runtime):
    """任务原因可见性属于代码，模型增改提议不能伪造这个字段。"""

    def script(state):
        """向合法新增任务混入代码字段，随后按空任务正常结束。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool(
            "update_tasks", {"changes": [{**cancel_task("#W0000001"), "reason_internal": False}]}
        )
        yield tool("complete_plan", {})
        yield reply()

    coordinator, state, gateway, _, _ = runtime(script, [])
    result = turn(coordinator, state, "请取消 #W0000001")
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
            "instruction": "取消订单 #W0000001",
            "order_id": "#W0000001",
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


@pytest.mark.parametrize("condition", [None, "用户决定取消这笔订单时"])
def test_complete_plan_can_prepare_selected_task_in_one_decision(runtime, backend, condition):
    """结束计划可在同一次决策分派任务，减少一次调用且首次准备不视为用户授权。"""

    def script(state):
        """创建任务后一次完成计划与派工，再按实际准备结果展示。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool(
            "update_tasks", {"changes": [{**cancel_task("#W0000001"), "condition": condition}]}
        )
        yield tool("complete_plan", {"task_ids": task_ids(state)})
        yield reply()

    coordinator, state, gateway, fake, child = runtime(script, cancel_steps("#W0000001"))
    result = turn(coordinator, state, "邮箱 u1@example.com，确认取消订单 #W0000001")
    assert result["rounds"] == len(fake.requests) == 4
    assert len(child.requests) == 3
    assert result["tasks"][0]["status"] == "pending_confirm"
    assert len(result["displayed"]) == 1 and result["confirmation_results"] == []
    assert not any(record["kind"] == "commit" for record in gateway.records)
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "pending"
    outcome = json.loads(
        [message for message in fake.requests[-1]["messages"] if message["role"] == "tool"][-1][
            "content"
        ]
    )
    assert outcome["confirmation_results"] == [] and outcome["blocked"] == []
    assert "tasks" not in outcome
    assert outcome["dispatched"] == [
        {"task_id": result["tasks"][0]["id"], "status": "pending_confirm", "reason": None}
    ]


@pytest.mark.parametrize("arguments", [{}, {"task_ids": []}])
def test_complete_plan_without_selection_keeps_separate_dispatch(runtime, backend, arguments):
    """省略任务列表或传空列表都只结束计划，旧的独立分派流程保持不变。"""

    def script(state):
        """在完整计划结果之后才显式发起业务子任务。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", arguments)
        assert child.requests == []
        outcome = json.loads(
            [message for message in fake.requests[-1]["messages"] if message["role"] == "tool"][-1][
                "content"
            ]
        )
        assert outcome == {"confirmation_results": [], "dispatched": [], "blocked": []}
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()

    coordinator, state, _, fake, child = runtime(script, cancel_steps("#W0000001"))
    result = turn(coordinator, state, "邮箱 u1@example.com，请取消 #W0000001")
    assert result["rounds"] == len(fake.requests) == 5
    assert result["tasks"][0]["status"] == "pending_confirm"
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "pending"


def test_complete_plan_confirms_previous_action_before_dispatching_same_order(runtime, backend):
    """合并调用先确认并核验前项，释放同单占用后才准备下一动作。"""

    def script(state):
        """同单先准备地址，用户确认后在一次计划调用内准备商品变更。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool(
            "update_tasks",
            {
                "changes": [
                    add_task("modify_pending_order_address", "#W0000001", "收货地址改为北京新地址"),
                    add_task(
                        "modify_pending_order_items",
                        "#W0000001",
                        "衬衫从 a 换成 b，差价用 gift 支付",
                    ),
                ]
            },
        )
        yield tool("complete_plan", {"task_ids": task_ids(state)})
        yield reply()
        yield tool("complete_plan", {"task_ids": task_ids(state, status="pending")})
        yield reply()
        yield tool("complete_plan", {"task_ids": []})
        yield reply()

    children = prepare_steps("modify_pending_order_address", {"order_id": "#W0000001", **ADDRESS})
    children += prepare_steps(
        "modify_pending_order_items",
        {
            "order_id": "#W0000001",
            "item_ids": ["a"],
            "new_item_ids": ["b"],
            "payment_method_id": "gift",
        },
    )
    coordinator, state, gateway, fake, _ = runtime(script, children)
    first = turn(coordinator, state, "先改地址，再换商品规格")
    assert [task["status"] for task in first["tasks"]] == ["pending_confirm", "pending"]
    first_outcome = json.loads(
        [message for message in fake.requests[-1]["messages"] if message["role"] == "tool"][-1][
            "content"
        ]
    )
    assert len(first_outcome["dispatched"]) == len(first_outcome["blocked"]) == 1
    second = turn(coordinator, state, "确认")
    assert second["rounds"] == 2
    assert [task["status"] for task in second["tasks"]] == ["done", "pending_confirm"]
    assert len(second["confirmation_results"]) == len(second["displayed"]) == 1
    assert (
        second["confirmation_results"][0]["operation_id"] == first["displayed"][0]["operation_id"]
    )
    snapshot = backend.export_snapshot()
    assert snapshot["orders"]["#W0000001"]["address"] == ADDRESS
    assert snapshot["orders"]["#W0000001"]["items"][0]["item_id"] == "a"
    records = gateway.records
    committed = next(index for index, record in enumerate(records) if record["kind"] == "commit")
    prepared = next(
        index
        for index, record in enumerate(records)
        if record["action"] == "modify_pending_order_items" and record.get("status") == "prepared"
    )
    assert committed < prepared
    third = turn(coordinator, state, "确认")
    assert all(task["status"] == "done" for task in third["tasks"])
    assert backend.export_snapshot()["orders"]["#W0000001"]["items"][0]["item_id"] == "b"


@pytest.mark.parametrize("invalid_selection", ["string", "null", "number", "object"])
def test_invalid_plan_parameter_structure_precedes_confirmation_side_effects(
    runtime, backend, monkeypatch, invalid_selection
):
    """工具参数结构不合法时整次拒绝，修正结构后同轮仍可独立处理用户确认。"""
    confirmations = []

    def script(state):
        """先展示第一单，次轮新增第二单并先给出一项不合法的合并调用。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {"task_ids": task_ids(state)})
        yield reply()
        yield tool("update_tasks", {"changes": [cancel_task("#W0000004")]})
        selected = task_ids(state, status="pending")[0]
        candidates = {
            "string": selected,
            "null": None,
            "number": [7],
            "object": {"task_id": selected},
        }
        yield tool("complete_plan", {"task_ids": candidates[invalid_selection]})
        assert len(confirmations) == 1 and len(child.requests) == 3
        assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "pending"
        assert not any(record["kind"] == "commit" for record in gateway.records)
        yield tool("complete_plan", {"task_ids": [selected]})
        yield reply()

    coordinator, state, gateway, _, child = runtime(
        script, cancel_steps("#W0000001") + cancel_steps("#W0000004")
    )
    original_confirm = coordinator._confirm

    async def observe_confirmation(state, context, user_message, previous, previous_reply=None):
        """记录确认的真实来源，继续执行原有确认与提交逻辑。

        Args:
            state: 当前任务状态。
            context: 当前协调者的可信调用身份。
            user_message: 本轮入口收到的真实用户原话。
            previous: 最近一次实际回复展示的完整快照。
            previous_reply: 上次实际回复正文和卡片，用于短答消歧。

        Returns:
            原确认器给出的逐项判定与执行结果。
        """
        confirmations.append((user_message, previous))
        return await original_confirm(state, context, user_message, previous, previous_reply)

    monkeypatch.setattr(coordinator, "_confirm", observe_confirmation)
    first = turn(coordinator, state, "取消 #W0000001")
    second = turn(coordinator, state, "确认")
    assert second["rounds"] == 4
    assert confirmations == [("取消 #W0000001", []), ("确认", first["displayed"])]
    assert [task["status"] for task in second["tasks"]] == ["done", "pending_confirm"]
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "cancelled"
    assert backend.export_snapshot()["orders"]["#W0000004"]["status"] == "pending"
    assert sum(record["kind"] == "commit" for record in gateway.records) == 1


def test_combined_plan_keeps_multiple_coordination_calls_blocked(runtime, backend):
    """同一响应混含多个协调工具时整条拒绝，不能执行其中任何准备动作。"""

    def script(state):
        """先产生协调工具批次错误，再用合法合并调用准备并展示操作。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        selection = {"task_ids": task_ids(state)}
        combined = tool("complete_plan", selection)
        combined["choices"][0]["message"]["tool_calls"].extend(
            tool("dispatch_tasks", selection)["choices"][0]["message"]["tool_calls"]
        )
        yield combined
        assert child.requests == []
        yield tool("complete_plan", selection)
        yield reply()

    coordinator, state, gateway, _, child = runtime(script, cancel_steps("#W0000001"))
    result = turn(coordinator, state, "取消 #W0000001")
    assert result["rounds"] == 5 and len(child.requests) == 3
    assert result["tasks"][0]["status"] == "pending_confirm"
    events = [json.loads(line) for line in gateway.trace.path.read_text().splitlines()]
    guards = [
        event["name"]
        for event in events
        if event["kind"] == "guard"
        and event["event"] == "end"
        and event["name"].startswith("coordinator_")
    ]
    assert guards == ["coordinator_format"]
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "pending"


def test_completed_task_in_plan_selection_is_reported_without_blocking_valid_task(runtime, backend):
    """选择已完成任务时返回跳过原因，同次合并调用仍可分派其他合法待办任务。"""

    def script(state):
        """完成首单后混选旧任务和新任务，由代码逐项筛选并准备新单。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {"task_ids": task_ids(state)})
        yield reply()
        yield tool("complete_plan", {})
        yield reply()
        yield tool("update_tasks", {"changes": [cancel_task("#W0000004")]})
        yield tool("complete_plan", {"task_ids": task_ids(state)})
        assert len(child.requests) == 6
        assert [task.status for task in state.tasks.tasks.values()] == ["done", "pending_confirm"]
        yield reply()

    coordinator, state, gateway, _, child = runtime(
        script, cancel_steps("#W0000001") + cancel_steps("#W0000004")
    )
    turn(coordinator, state, "取消 #W0000001")
    turn(coordinator, state, "确认")
    result = turn(coordinator, state, "还要取消 #W0000004")
    assert result["rounds"] == 3
    assert [task["status"] for task in result["tasks"]] == ["done", "pending_confirm"]
    assert len(child.requests) == 6
    assert sum(record["kind"] == "commit" for record in gateway.records) == 1
    assert backend.export_snapshot()["orders"]["#W0000004"]["status"] == "pending"

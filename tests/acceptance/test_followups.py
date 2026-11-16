"""用固定中文开发种子验收改口、补充信息和跨会话续办的完整业务结果。"""

import asyncio
from copy import deepcopy

from .support import add_task, prepare_steps, reply, task_ids, tool

CANCEL = "cancel_pending_order"
FIRST_ORDER = "#W8770097"
SECOND_ORDER = "#W5183325"
THIRD_ORDER = "#W3913498"


def assert_single_cancellation(case, order_id: str, reason: str) -> None:
    """逐字段验证指定订单只退款一次，其他业务实体保持种子中的值。

    Args:
        case: 当前验收场景，提供固定种子和真实 SQLite 快照。
        order_id: 本场景最终允许取消的订单。
        reason: 最终确认的后端取消原因。
    """
    expected = deepcopy(case.seed)
    order = expected["orders"][order_id]
    payment = order["payment_history"][0]
    order["payment_history"].append(
        {
            "transaction_type": "refund",
            "amount": payment["amount"],
            "payment_method_id": payment["payment_method_id"],
        }
    )
    order.update(status="cancelled", cancel_reason=reason)
    assert case.snapshot() == expected


def test_changed_order_rejects_old_operation_and_waits_for_new_confirmation(acceptance_case):
    """同意中的改口使旧操作失效，新订单展示后再确认才取消且只退款一次。"""
    case = acceptance_case

    def script(tasks):
        """先准备办公椅取消，再把同一任务修订为另一订单，最后处理确认。"""
        yield tool(
            "update_tasks",
            {"changes": [add_task(CANCEL, FIRST_ORDER, "办公椅不再需要，请取消订单")]},
        )
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(tasks)})
        yield reply()
        yield tool(
            "update_tasks",
            {
                "changes": [
                    {
                        **add_task(CANCEL, SECOND_ORDER, "保留办公椅，另一订单不再需要并取消"),
                        "operation": "update",
                        "task_id": task_ids(tasks)[0],
                    }
                ]
            },
        )
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(tasks)})
        yield reply()
        yield tool("complete_plan", {})
        yield reply()

    async def scenario():
        """验证改口前后两份展示、操作修订和最终全库状态。"""
        children = prepare_steps(
            CANCEL, {"order_id": FIRST_ORDER, "reason": "no longer needed"}
        ) + prepare_steps(CANCEL, {"order_id": SECOND_ORDER, "reason": "no longer needed"})
        async with case.session(script, children) as runtime:
            first = await case.send(runtime, f"办公椅不再需要，请取消订单 {FIRST_ORDER}")
            original_task = first["tasks"][0]
            original_operation = first["displayed"][0]["operation_id"]
            assert original_task["status"] == "pending_confirm"
            assert case.operation(original_operation)["status"] == "prepared"
            assert case.snapshot() == case.seed

            changed = await case.send(
                runtime,
                f"好的，不过改为取消 {SECOND_ORDER}，这单不再需要；{FIRST_ORDER} 的办公椅保留",
            )
            revised_task = changed["tasks"][0]
            revised_operation = changed["displayed"][0]["operation_id"]
            assert len(changed["tasks"]) == len(changed["displayed"]) == 1
            assert revised_task["id"] == original_task["id"]
            assert revised_task["revision"] == original_task["revision"] + 1
            assert revised_task["order_id"] == SECOND_ORDER
            assert revised_task["status"] == "pending_confirm"
            assert revised_operation != original_operation
            assert case.operation(original_operation)["status"] == "rejected"
            assert case.operation(revised_operation)["status"] == "prepared"
            assert changed["confirmation_results"] == []
            assert case.snapshot() == case.seed
            assert changed["displayed"][0]["args"]["order_id"] == SECOND_ORDER
            assert "等待确认" in changed["reply"]

            confirmed = await case.send(runtime, "确认")
            assert confirmed["tasks"][0]["status"] == "done"
            assert confirmed["displayed"] == []
            assert len(confirmed["confirmation_results"]) == 1
            assert confirmed["confirmation_results"][0]["operation_id"] == revised_operation
            assert confirmed["confirmation_results"][0]["verification"]["verified"]
            assert case.operation(original_operation)["status"] == "rejected"
            assert case.operation(revised_operation)["status"] == "committed"
            assert_single_cancellation(case, SECOND_ORDER, "no longer needed")

    asyncio.run(scenario())


def test_missing_cancel_reason_is_collected_before_prepare_and_confirmation(acceptance_case):
    """缺少取消原因时只询问，补全后准备并展示，下一轮确认才执行退款。"""
    case = acceptance_case

    def script(tasks):
        """把信息缺口交由子 Agent 返回，用户补全后更新原任务再分派。"""
        yield tool(
            "update_tasks",
            {"changes": [add_task(CANCEL, THIRD_ORDER, "请取消订单，尚未提供取消原因")]},
        )
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(tasks)})
        yield reply()
        yield tool(
            "update_tasks",
            {
                "changes": [
                    {
                        **add_task(CANCEL, THIRD_ORDER, "取消订单，原因是误下单", missing=[]),
                        "operation": "update",
                        "task_id": task_ids(tasks)[0],
                    }
                ]
            },
        )
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(tasks)})
        yield reply()
        yield tool("complete_plan", {})
        yield reply()

    async def scenario():
        """逐轮核对缺口、操作生成时间、核验结果和完整业务终态。"""
        children = [
            tool("get_order_details", {"order_id": THIRD_ORDER}),
            tool("load_skill", {"name": CANCEL}),
            tool(
                "finish",
                {
                    "missing": ["取消原因：不再需要或误下单"],
                    "message_hint": "请告知是因为不再需要，还是误下单",
                },
            ),
        ] + prepare_steps(CANCEL, {"order_id": THIRD_ORDER, "reason": "ordered by mistake"})
        async with case.session(script, children) as runtime:
            incomplete = await case.send(runtime, f"请帮我取消订单 {THIRD_ORDER}")
            waiting_task = incomplete["tasks"][0]
            assert waiting_task["status"] == "waiting_info"
            assert waiting_task["missing"] == ["取消原因：不再需要或误下单"]
            assert waiting_task["operation_ids"] == []
            assert waiting_task["actions"] == []
            assert incomplete["displayed"] == []
            assert "取消原因" in incomplete["reply"]
            assert case.snapshot() == case.seed

            prepared = await case.send(runtime, "原因是误下单")
            updated_task = prepared["tasks"][0]
            operation_id = prepared["displayed"][0]["operation_id"]
            assert updated_task["id"] == waiting_task["id"]
            assert updated_task["revision"] == waiting_task["revision"] + 1
            assert updated_task["missing"] == []
            assert updated_task["status"] == "pending_confirm"
            assert prepared["displayed"][0]["args"]["reason"] == "ordered by mistake"
            assert case.operation(operation_id)["status"] == "prepared"
            assert prepared["confirmation_results"] == []
            assert case.snapshot() == case.seed

            confirmed = await case.send(runtime, "确认")
            assert confirmed["tasks"][0]["status"] == "done"
            assert len(confirmed["confirmation_results"]) == 1
            assert confirmed["confirmation_results"][0]["verification"]["verified"]
            assert case.operation(operation_id)["status"] == "committed"
            assert_single_cancellation(case, THIRD_ORDER, "ordered by mistake")

    asyncio.run(scenario())


def test_reopened_chat_requires_redisplay_before_single_refund(acceptance_case):
    """中断后重开同一业务库并重验身份，立即确认只重展，下一轮才退款。"""
    case = acceptance_case

    def initial_script(tasks):
        """在首段聊天准备取消操作并将待确认任务保存到检查点。"""
        yield tool(
            "update_tasks",
            {"changes": [add_task(CANCEL, FIRST_ORDER, "办公椅不再需要，请取消订单")]},
        )
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(tasks)})
        yield reply()

    def resumed_script(tasks):
        """续办第一轮仅重新展示，下一轮才依据新展示接受确认。"""
        yield tool("complete_plan", {})
        yield reply()
        yield tool("complete_plan", {})
        yield reply()

    async def scenario():
        """关闭再打开业务库、档案和检查点，验证任务复用与授权失效。"""
        children = prepare_steps(CANCEL, {"order_id": FIRST_ORDER, "reason": "no longer needed"})
        async with case.session(initial_script, children, session_id="chat1") as runtime:
            first = await case.send(runtime, f"办公椅不再需要，请取消订单 {FIRST_ORDER}")
            previous_task = first["tasks"][0]
            operation_id = first["displayed"][0]["operation_id"]
            assert previous_task["status"] == "pending_confirm"
            assert first["displayed"][0]["session_id"] == "chat1"
            assert case.snapshot() == case.seed

        async with case.session(resumed_script, (), session_id="chat2") as runtime:
            restored = await runtime.snapshot()
            assert restored.last_display == []
            assert restored.tasks.snapshot()[0]["id"] == previous_task["id"]
            assert restored.tasks.snapshot()[0]["revision"] == previous_task["revision"]
            assert restored.session.user_id == case.user["user_id"]
            assert restored.session.session_id == "chat2"

            redisplayed = await case.send(runtime, "确认")
            assert redisplayed["confirmation_results"] == []
            assert redisplayed["tasks"][0]["status"] == "pending_confirm"
            assert redisplayed["displayed"][0]["operation_id"] == operation_id
            assert redisplayed["displayed"][0]["session_id"] == "chat2"
            assert redisplayed["displayed"][0]["turn"] > first["displayed"][0]["turn"]
            assert case.operation(operation_id)["status"] == "prepared"
            assert case.snapshot() == case.seed

            confirmed = await case.send(runtime, "确认")
            assert confirmed["tasks"][0]["status"] == "done"
            assert confirmed["tasks"][0]["operation_ids"] == [operation_id]
            assert confirmed["displayed"] == []
            assert len(confirmed["confirmation_results"]) == 1
            assert confirmed["confirmation_results"][0]["verification"]["verified"]
            assert case.operation(operation_id)["status"] == "committed"
            assert_single_cancellation(case, FIRST_ORDER, "no longer needed")

    asyncio.run(scenario())

"""用真实商品库存和持久任务验收自然语言条件的两条售后分支。"""

import asyncio
from copy import deepcopy

from .support import DELIVERED, PAYMENT, add_task, prepare_steps, reply, task_ids, tool

EXCHANGE = "exchange_delivered_order_items"
RETURN = "return_delivered_order_items"
OLD_ITEM = "5206946487"
AVAILABLE = "8481719475"
UNAVAILABLE = "5753502325"


def conditional_tasks(target_item: str) -> list[dict]:
    """提出优先换货、无库存才退货的两个任务，条件保留为自然语言。

    Args:
        target_item: 用户指定的新规格标识。

    Returns:
        同一订单的换货主任务与条件退货任务，不包含任何执行状态。
    """
    return [
        add_task(EXCHANGE, DELIVERED, f"花园水管 {OLD_ITEM} 换成 {target_item}，差价用 {PAYMENT}"),
        add_task(
            RETURN,
            DELIVERED,
            f"若指定水管规格无库存，退回 {OLD_ITEM}，原路退款",
            condition="指定的新规格无库存，无法换货",
        ),
    ]


def test_exchange_success_cancels_conditional_return(acceptance_case):
    """换货核验成功后撤销备用退货，不把条件未触发记作业务失败。"""
    case = acceptance_case
    expected = deepcopy(case.seed)
    order = expected["orders"][DELIVERED]
    order.update(
        status="exchange requested",
        exchange_items=[OLD_ITEM],
        exchange_new_items=[AVAILABLE],
        exchange_payment_method_id=PAYMENT,
    )

    def script(tasks):
        """只分派主任务，确认核验后撤销不再需要的条件分支。"""
        yield tool("update_tasks", {"changes": conditional_tasks(AVAILABLE)})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(tasks, status="pending")})
        yield reply()
        yield tool("complete_plan", {})
        assert next(task for task in tasks() if task["action"] == EXCHANGE)["status"] == "done"
        yield tool(
            "update_tasks",
            {
                "changes": [
                    {
                        "operation": "cancel",
                        "task_id": task_ids(tasks, action=RETURN)[0],
                        "reason": "指定规格已成功申请换货，不再需要退货备用方案",
                    }
                ]
            },
        )
        yield reply()

    async def scenario():
        """两轮完成准备与确认，逐轮检查条件任务和真实订单。"""
        children = prepare_steps(
            EXCHANGE,
            {
                "order_id": DELIVERED,
                "item_ids": [OLD_ITEM],
                "new_item_ids": [AVAILABLE],
                "payment_method_id": PAYMENT,
            },
        )
        async with case.session(script, children) as runtime:
            first = await case.send(
                runtime,
                f"订单 {DELIVERED} 的水管换成 {AVAILABLE}；"
                "如果这个规格没有库存，就退掉水管，退款回原付款方式。",
            )
            assert [task["status"] for task in first["tasks"]] == ["pending_confirm", "conditional"]
            assert len(first["displayed"]) == 1
            assert case.snapshot() == case.seed
            final = await case.send(runtime, "确认")
            assert [task["status"] for task in final["tasks"]] == ["done", "cancelled"]
            assert "不再需要" in final["tasks"][1]["reason"]
            assert final["tasks"][1]["operation_ids"] == []
            assert final["displayed"] == []
            assert final["tasks"][0]["facts"][-1]["verification"]["verified"] is True
        actual = case.snapshot()
        # 换货申请的差价与实际字段在后端计算，以下独立核对固定种子的预期金额。
        assert actual["orders"][DELIVERED]["exchange_price_difference"] == 24.71
        order["exchange_price_difference"] = 24.71
        assert actual == expected

    asyncio.run(scenario())


def test_unavailable_variant_activates_return_then_requires_confirmation(acceptance_case):
    """真实库存不满足条件时改办退货，条件成立仍不能代替用户确认。"""
    case = acceptance_case
    assert case.seed["products"]["6679515468"]["variants"][UNAVAILABLE]["available"] is False

    def script(tasks):
        """查询库存后由协调者取消换货、解除退货条件并重新分派。"""
        yield tool("update_tasks", {"changes": conditional_tasks(UNAVAILABLE)})
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(tasks, status="pending")})
        exchange = next(task for task in tasks() if task["action"] == EXCHANGE)
        assert exchange["status"] == "waiting_info"
        product = next(
            fact["result"] for fact in exchange["facts"] if fact["tool"] == "get_product_details"
        )
        assert product["variants"][UNAVAILABLE]["available"] is False
        yield tool(
            "update_tasks",
            {
                "changes": [
                    {
                        "operation": "cancel",
                        "task_id": exchange["id"],
                        "reason": "查询确认用户指定规格无库存，按原诉求改办退货",
                    },
                    {
                        **add_task(
                            RETURN, DELIVERED, f"指定规格无库存，退回水管 {OLD_ITEM}，原路退款"
                        ),
                        "operation": "update",
                        "task_id": task_ids(tasks, action=RETURN)[0],
                        "condition": None,
                    },
                ]
            },
        )
        yield tool("dispatch_tasks", {"task_ids": task_ids(tasks, status="pending")})
        yield reply()
        yield tool("complete_plan", {})
        yield reply()

    async def scenario():
        """查询无库存的事实后准备退货，下一轮授权才写入申请。"""
        children = [
            tool("get_order_details", {"order_id": DELIVERED}),
            tool("get_product_details", {"product_id": "6679515468"}),
            tool(
                "finish",
                {
                    "missing": ["指定规格无库存，需采用备用退货方案"],
                    "note": "指定的绿色橡胶水管无库存。",
                },
            ),
            *prepare_steps(
                RETURN,
                {"order_id": DELIVERED, "item_ids": [OLD_ITEM], "payment_method_id": PAYMENT},
            ),
        ]
        async with case.session(script, children) as runtime:
            first = await case.send(
                runtime,
                f"把订单 {DELIVERED} 的水管换成 {UNAVAILABLE}；"
                "该规格没库存就退水管，退款回原付款方式。",
            )
            assert [task["status"] for task in first["tasks"]] == ["cancelled", "pending_confirm"]
            assert first["tasks"][0]["operation_ids"] == []
            assert len(first["displayed"]) == 1
            shown_operation = case.operation(first["displayed"][0]["operation_id"])
            assert shown_operation["action"] == RETURN
            assert shown_operation["args"]["item_ids"] == [OLD_ITEM]
            assert case.snapshot() == case.seed
            final = await case.send(runtime, "确认")
            assert [task["status"] for task in final["tasks"]] == ["cancelled", "done"]
            assert final["tasks"][1]["facts"][-1]["verification"]["verified"] is True
        expected = deepcopy(case.seed)
        expected["orders"][DELIVERED].update(
            status="return requested", return_items=[OLD_ITEM], return_payment_method_id=PAYMENT
        )
        assert case.snapshot() == expected

    asyncio.run(scenario())

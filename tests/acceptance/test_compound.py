"""用中文开发种子验收跨能力复合诉求与同单多动作的完整办理。"""

import asyncio
from copy import deepcopy

import pytest

from .support import add_task, prepare_steps, reply, task_ids, tool

ADDRESS = {
    "address1": "示例路 88 号",
    "address2": "2 栋 601 室",
    "city": "杭州市",
    "state": "浙江省",
    "country": "中国",
    "zip": "310000",
}


def test_cancel_and_return_require_explicit_confirmation_scope(acceptance_case):
    """跨订单取消与退货分别派工，模糊确认不写库，全部确认后核验两项结果。"""
    case = acceptance_case

    def script(tasks):
        """将两类诉求分别交给订单服务与售后，等待用户明确确认两项操作。"""
        yield tool(
            "update_tasks",
            {
                "changes": [
                    add_task(
                        "cancel_pending_order",
                        "#W8770097",
                        "办公椅不再需要，取消订单 #W8770097 并退回原支付方式",
                    ),
                    add_task(
                        "return_delivered_order_items",
                        "#W6893533",
                        "退回订单 #W6893533 的花园水管 5206946487，退款用原 PayPal，耳机保留",
                    ),
                ]
            },
        )
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(tasks)})
        yield reply()
        yield tool("complete_plan", {})
        yield reply()
        yield tool("complete_plan", {})
        yield reply()

    children = prepare_steps(
        "cancel_pending_order",
        {"order_id": "#W8770097", "reason": "no longer needed"},
    ) + prepare_steps(
        "return_delivered_order_items",
        {
            "order_id": "#W6893533",
            "item_ids": ["5206946487"],
            "payment_method_id": "paypal_6151711",
        },
    )

    async def scenario():
        """经过准备、范围澄清和明确确认三轮对话，检查完整业务快照。"""
        async with case.session(script, children) as runtime:
            prepared = await case.send(
                runtime,
                "订单 #W8770097 的办公椅不再需要了，请取消；另外把已收货订单 "
                "#W6893533 的花园水管 5206946487 退掉，耳机保留，退款退回原 PayPal。",
            )
            assert len(prepared["tasks"]) == 2
            assert {task["capability"] for task in prepared["tasks"]} == {
                "order_service",
                "after_sales",
            }
            assert all(task["status"] == "pending_confirm" for task in prepared["tasks"])
            assert len(prepared["displayed"]) == 2
            assert case.snapshot() == case.seed
            operation_ids = [item["operation_id"] for item in prepared["displayed"]]
            assert all(case.operation(key)["status"] == "prepared" for key in operation_ids)

            ambiguous = await case.send(runtime, "好的")
            assert all(task["status"] == "pending_confirm" for task in ambiguous["tasks"])
            assert len(ambiguous["displayed"]) == 2
            assert case.snapshot() == case.seed
            assert all(case.operation(key)["status"] == "prepared" for key in operation_ids)

            confirmed = await case.send(runtime, "全部确认")
            assert "上下文编译无法继续" not in confirmed["reply"]
            assert all(task["status"] == "done" for task in confirmed["tasks"])
            assert confirmed["displayed"] == []
            assert len(confirmed["confirmation_results"]) == 2
            assert all(case.operation(key)["status"] == "committed" for key in operation_ids)

            expected = deepcopy(case.seed)
            cancelled = expected["orders"]["#W8770097"]
            cancelled.update(status="cancelled", cancel_reason="no longer needed")
            cancelled["payment_history"].append(
                {
                    "transaction_type": "refund",
                    "amount": 3810.03,
                    "payment_method_id": "paypal_6151711",
                }
            )
            expected["orders"]["#W6893533"].update(
                status="return requested",
                return_items=["5206946487"],
                return_payment_method_id="paypal_6151711",
            )
            assert case.snapshot() == expected

    asyncio.run(scenario())


def test_same_order_changes_address_before_combined_item_modification(acceptance_case):
    """同单地址先确认，两个规格请求合为一次商品修改，最终地址、商品和差价均正确。"""
    case = acceptance_case

    def script(tasks):
        """先准备地址，等待确认后才准备已汇总的两件商品修改。"""
        yield tool(
            "update_tasks",
            {
                "changes": [
                    add_task(
                        "modify_pending_order_address",
                        "#W5183325",
                        "将订单 #W5183325 收货地址改为浙江省杭州市示例路 88 号 "
                        "2 栋 601 室，中国，邮编 310000",
                    ),
                    add_task(
                        "modify_pending_order_items",
                        "#W5183325",
                        "将 LED 灯泡由 6206533187 改成 7445824652，差价使用 PayPal",
                    ),
                    add_task(
                        "modify_pending_order_items",
                        "#W5183325",
                        "将机械键盘由 1421289881 改成 7706410293，差价使用 paypal_6151711",
                    ),
                ]
            },
        )
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(tasks)})
        yield reply()
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": task_ids(tasks, status="pending")})
        yield reply()
        yield tool("complete_plan", {})
        yield reply()

    children = prepare_steps(
        "modify_pending_order_address", {"order_id": "#W5183325", **ADDRESS}
    ) + prepare_steps(
        "modify_pending_order_items",
        {
            "order_id": "#W5183325",
            "item_ids": ["6206533187", "1421289881"],
            "new_item_ids": ["7445824652", "7706410293"],
            "payment_method_id": "paypal_6151711",
        },
    )

    async def scenario():
        """逐次确认两个动作，检查订单串行、完整商品清单和一次性差价支付。"""
        async with case.session(script, children) as runtime:
            first = await case.send(
                runtime,
                "请先把订单 #W5183325 的收货地址改为浙江省杭州市示例路 88 号 "
                "2 栋 601 室，中国，邮编 310000；再把 LED 灯泡规格 6206533187 改为 "
                "7445824652，机械键盘规格 1421289881 改为 7706410293，"
                "差价使用 paypal_6151711 支付。",
            )
            assert len(first["tasks"]) == 2
            address_task, items_task = first["tasks"]
            assert address_task["status"] == "pending_confirm"
            assert items_task["action"] == "modify_pending_order_items"
            assert "LED 灯泡" in items_task["instruction"]
            assert "机械键盘" in items_task["instruction"]
            assert items_task["status"] == "pending"
            assert items_task["operation_ids"] == []
            assert len(first["displayed"]) == 1
            assert first["displayed"][0]["task_id"] == address_task["id"]
            assert case.snapshot() == case.seed

            second = await case.send(runtime, "确认")
            assert [task["status"] for task in second["tasks"]] == ["done", "pending_confirm"]
            assert len(second["confirmation_results"]) == 1
            assert len(second["displayed"]) == 1
            assert second["displayed"][0]["task_id"] == items_task["id"]
            expected = deepcopy(case.seed)
            expected["orders"]["#W5183325"]["address"] = ADDRESS
            assert case.snapshot() == expected
            items_operation = case.operation(second["displayed"][0]["operation_id"])
            assert items_operation["status"] == "prepared"
            assert items_operation["args"]["item_ids"] == ["6206533187", "1421289881"]
            assert items_operation["args"]["new_item_ids"] == ["7445824652", "7706410293"]

            third = await case.send(runtime, "确认")
            assert "上下文编译无法继续" not in third["reply"]
            assert all(task["status"] == "done" for task in third["tasks"])
            assert len(third["confirmation_results"]) == 1
            assert third["displayed"] == []
            assert all(len(task["operation_ids"]) == 1 for task in third["tasks"])
            assert all(
                case.operation(task["operation_ids"][0])["status"] == "committed"
                for task in third["tasks"]
            )
            order = expected["orders"]["#W5183325"]
            for item, new_id in zip(order["items"], ["7445824652", "7706410293"], strict=True):
                variant = case.seed["products"][item["product_id"]]["variants"][new_id]
                item.update(item_id=new_id, price=variant["price"], options=variant["options"])
            assert [item["price"] for item in order["items"]] == [348.60, 1884.12]
            order["status"] = "pending (item modified)"
            order["payment_history"].append(
                {
                    "transaction_type": "payment",
                    "amount": pytest.approx(16.52),
                    "payment_method_id": "paypal_6151711",
                }
            )
            assert case.snapshot() == expected

    asyncio.run(scenario())

"""验证确认展示覆盖完整请求、保留预期金额且不泄露运行内部字段。"""

import json
from copy import deepcopy

import pytest

from ttwill.backend import request_key
from ttwill.gateway.presentation import action_label, operation_content

ADDRESS = {
    "address1": "新街道 18 号",
    "address2": "2 单元 301",
    "city": "苏州",
    "state": "江苏",
    "country": "中国",
    "zip": "215000",
}


def sections(content: str) -> tuple[dict, dict]:
    """解析展示正文中的办理参数与预期结果，便于逐值核对。"""
    arguments, preview = content.split("办理内容：\n", 1)[1].split("\n\n确认后预期结果：\n")
    return json.loads(arguments), json.loads(preview)


@pytest.mark.parametrize(
    ("action", "args", "label", "expected_fields"),
    [
        (
            "cancel_pending_order",
            {"order_id": "#W0000001", "reason": "no longer needed"},
            "取消订单",
            {"订单状态", "取消原因", "支付与退款明细"},
        ),
        (
            "modify_pending_order_address",
            {"order_id": "#W0000001", **ADDRESS},
            "修改订单收货地址",
            {"订单状态", "地址"},
        ),
        (
            "modify_pending_order_items",
            {
                "order_id": "#W0000001",
                "item_ids": ["a"],
                "new_item_ids": ["b"],
                "payment_method_id": "gift",
            },
            "修改订单商品规格",
            {"订单状态", "本次商品变更", "支付与退款明细"},
        ),
        (
            "modify_pending_order_payment",
            {"order_id": "#W0000001", "payment_method_id": "gift"},
            "修改订单支付方式",
            {"订单状态", "支付与退款明细"},
        ),
        (
            "modify_user_address",
            {"user_id": "u1", **ADDRESS},
            "修改账户默认地址",
            {"地址"},
        ),
        (
            "return_delivered_order_items",
            {"order_id": "#W0000002", "item_ids": ["a"], "payment_method_id": "card"},
            "申请退货",
            {"订单状态", "订单商品", "退货商品清单", "退货退款方式", "支付与退款明细"},
        ),
        (
            "exchange_delivered_order_items",
            {
                "order_id": "#W0000002",
                "item_ids": ["a"],
                "new_item_ids": ["b"],
                "payment_method_id": "gift",
            },
            "申请换货",
            {
                "订单状态",
                "订单商品",
                "换货商品清单",
                "换货新规格",
                "换货差价支付方式",
                "换货差价",
                "支付与退款明细",
            },
        ),
    ],
)
def test_seven_actions_show_complete_request_and_relevant_preview(
    backend, action, args, label, expected_fields
):
    """七种写动作保留全部请求值与相关预期字段，始终等待确认且不改变源记录。"""
    operation = backend.prepare("u1", action, args, request_key("display", 1, action))
    before = deepcopy(operation)
    original_items = (
        backend.query("u1", "get_order_details", {"order_id": args["order_id"]})["items"]
        if action == "modify_pending_order_items"
        else None
    )
    content = operation_content(operation, original_items=original_items)
    shown_args, shown_preview = sections(content)
    assert content.startswith(f"待确认：{label}\n")
    assert set(shown_preview) == expected_fields
    assert len(shown_args) == len(args)
    assert all(not key.isascii() for key in shown_args)
    for key, value in args.items():
        if key == "user_id":
            assert shown_args["办理账户"] == "当前账户"
            assert value not in content
        elif key == "reason":
            assert value in shown_args["取消原因"]
            assert "不再需要" in shown_args["取消原因"]
        else:
            assert value in shown_args.values()
    assert operation["operation_id"] not in content
    assert operation["request_key"] not in content
    assert all(key not in content for key in ("operation_id", "request_key", "user_id", "prepared"))
    assert all(phrase not in content for phrase in ("已完成", "办理成功", "已到账", "退款到账"))
    assert operation == before
    if "payment_history" in operation["preview"] and "支付与退款明细" in shown_preview:
        original = operation["preview"]["payment_history"]
        shown = shown_preview["支付与退款明细"]
        assert [entry["金额"] for entry in shown] == [entry["amount"] for entry in original]
        assert [entry["支付方式"] for entry in shown] == [
            entry["payment_method_id"] for entry in original
        ]
    if "订单商品" in shown_preview:
        original = operation["preview"]["items"]
        shown = shown_preview["订单商品"]
        assert [item["价格"] for item in shown] == [item["price"] for item in original]
        assert [item["规格"] for item in shown] == [item["options"] for item in original]
        assert [item["商品规格编号"] for item in shown] == [item["item_id"] for item in original]


def test_user_address_preview_excludes_private_account_details(backend):
    """默认地址确认只显示地址，不泄露邮箱、支付方式余额和其他订单资料。"""
    action = "modify_user_address"
    operation = backend.prepare(
        "u1", action, {"user_id": "u1", **ADDRESS}, request_key("a", 1, action)
    )
    content = operation_content(operation)
    assert "账户：当前账户" in content
    assert all(
        text not in content
        for text in ("u1@example.com", "1234", "300.0", "#W0000001", "#W0000002")
    )
    assert set(sections(content)[1]) == {"地址"}


def test_payment_preview_preserves_decimal_amounts_and_transaction_order():
    """展示不计算新金额、不添加币种，付款退款次序与来源小数保持一致。"""
    operation = {
        "action": "modify_pending_order_payment",
        "args": {"order_id": "#W0000009", "payment_method_id": "new-card"},
        "preview": {
            "status": "pending",
            "payment_history": [
                {"transaction_type": "payment", "amount": 10245.9, "payment_method_id": "old-card"},
                {"transaction_type": "payment", "amount": 10245.9, "payment_method_id": "new-card"},
                {"transaction_type": "refund", "amount": 10245.9, "payment_method_id": "old-card"},
            ],
        },
    }
    content = operation_content(operation)
    preview = sections(content)[1]
    assert [entry["类型"] for entry in preview["支付与退款明细"]] == ["付款", "付款", "退款"]
    assert all(entry["金额"] == 10245.9 for entry in preview["支付与退款明细"])
    assert "¥" not in content and "元" not in content
    assert "确认后预期结果" in content


def test_read_action_labels_and_unknown_action_fallback():
    """常用查询使用中文名称，未来未知动作保留原名便于识别。"""
    assert action_label("get_order_details") == "查询订单详情"
    assert action_label("get_product_details") == "查询商品详情"
    assert action_label("get_user_details") == "查询账户详情"
    assert action_label("future_action") == "future_action"

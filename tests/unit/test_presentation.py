"""验证确认展示覆盖完整请求、保留预期金额且不泄露运行内部字段。"""

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


def sections(content: str) -> tuple[str, str]:
    """拆出卡片的办理参数与预期结果正文。

    Args:
        content: 包含两段固定标题的操作卡片。

    Returns:
        按办理参数、预期结果顺序返回两段自然语言文本。
    """
    arguments, preview = content.split("办理内容：\n", 1)[1].split("\n\n确认后预期结果：\n")
    return arguments, preview


def field_values(content: str, field: str) -> list[str]:
    """按展示顺序取得指定字段的标量文本。

    Args:
        content: 用项目符号逐行展示的正文。
        field: 要核对的中文字段名称。

    Returns:
        所有同名字段在冒号后的原文，顺序与正文一致。
    """
    prefix = f"· {field}："
    return [
        line.strip()[len(prefix) :]
        for line in content.splitlines()
        if line.strip().startswith(prefix)
    ]


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
    target = f"订单：{args['order_id']}" if "order_id" in args else "账户：当前账户"
    assert content.splitlines()[1] == target
    assert {
        line[2:].split("：", 1)[0] for line in shown_preview.splitlines() if line.startswith("· ")
    } == expected_fields
    argument_fields = [
        line[2:].split("：", 1)[0] for line in shown_args.splitlines() if line.startswith("· ")
    ]
    assert len(argument_fields) == len(args)
    assert all(not key.isascii() for key in argument_fields)
    assert all(
        line.lstrip().startswith("· ") for line in (shown_args + "\n" + shown_preview).splitlines()
    )
    for key, value in args.items():
        if key == "user_id":
            assert field_values(shown_args, "办理账户") == ["当前账户"]
            assert value not in content
        elif key == "reason":
            assert field_values(shown_args, "取消原因") == ["不再需要（no longer needed）"]
        elif isinstance(value, list):
            field = {"item_ids": "商品清单", "new_item_ids": "替换规格"}[key]
            expected = f"· {field}：\n" + "\n".join(f"  · {item}" for item in value)
            assert expected in shown_args
        else:
            assert any(line.endswith(f"：{value}") for line in shown_args.splitlines())
    assert operation["operation_id"] not in content
    assert operation["request_key"] not in content
    assert all(key not in content for key in ("operation_id", "request_key", "user_id", "prepared"))
    assert all(phrase not in content for phrase in ("已完成", "办理成功", "已到账", "退款到账"))
    assert operation == before
    if "payment_history" in operation["preview"] and "支付与退款明细" in shown_preview:
        original = operation["preview"]["payment_history"]
        assert field_values(shown_preview, "金额") == [str(entry["amount"]) for entry in original]
        assert field_values(shown_preview, "支付方式") == [
            entry["payment_method_id"] for entry in original
        ]
    if "订单商品" in shown_preview:
        original = operation["preview"]["items"]
        assert field_values(shown_preview, "价格") == [str(item["price"]) for item in original]
        assert field_values(shown_preview, "商品规格编号") == [item["item_id"] for item in original]
        for item in original:
            for field, value in item["options"].items():
                assert f"      · {field}：{value}" in shown_preview


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
    assert sections(content)[1] == (
        "· 地址：\n"
        "  · 地址第一行：新街道 18 号\n"
        "  · 地址第二行：2 单元 301\n"
        "  · 城市：苏州\n"
        "  · 省或州：江苏\n"
        "  · 国家：中国\n"
        "  · 邮政编码：215000"
    )


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
    assert field_values(preview, "类型") == ["付款", "付款", "退款"]
    assert field_values(preview, "金额") == ["10245.9", "10245.9", "10245.9"]
    assert field_values(preview, "支付方式") == ["old-card", "new-card", "old-card"]
    assert "· 支付与退款明细：\n  · 第 1 项：\n    · 类型：付款" in preview
    assert "¥" not in content and "元" not in content
    assert "确认后预期结果" in content


def test_address_preserves_quotes_braces_and_multiline_business_text():
    """地址中的引号和花括号保留为业务文字，多行地址归属于同一字段。"""
    address = {**ADDRESS, "address1": '新街道 "东门" {A} 18 号', "address2": "2 单元\n301 室"}
    operation = {
        "action": "modify_user_address",
        "args": {"user_id": "private-user", **address},
        "preview": {"address": address},
    }
    content = operation_content(operation)
    arguments, preview = sections(content)
    assert '· 地址第一行：新街道 "东门" {A} 18 号' in arguments
    assert '  · 地址第一行：新街道 "东门" {A} 18 号' in preview
    assert "· 地址第二行：2 单元\n  301 室" in arguments
    assert "  · 地址第二行：2 单元\n    301 室" in preview
    assert "private-user" not in content
    assert '\\"' not in content and "\\n" not in content


def test_nested_options_and_empty_values_use_indented_lines():
    """商品嵌套规格和空集合逐层展开，不把业务引号或列表序列化为 JSON。"""
    operation = {
        "action": "return_delivered_order_items",
        "args": {"order_id": "#W1", "item_ids": ["a"], "payment_method_id": "card"},
        "preview": {
            "status": "return requested",
            "items": [
                {
                    "name": "定制衬衫",
                    "options": {"定制": {"图案": '文字 "A"', "附加": ["袋", "卡"]}, "备注": ""},
                }
            ],
            "return_items": ["a"],
            "return_payment_method_id": "card",
            "payment_history": [],
        },
    }
    preview = sections(operation_content(operation))[1]
    assert (
        "· 订单商品：\n"
        "  · 第 1 项：\n"
        "    · 商品名称：定制衬衫\n"
        "    · 规格：\n"
        "      · 定制：\n"
        '        · 图案：文字 "A"\n'
        "        · 附加：\n"
        "          · 袋\n"
        "          · 卡\n"
        "      · 备注：（空）"
    ) in preview
    assert preview.endswith("· 支付与退款明细：\n  · 无")


def test_read_action_labels_and_unknown_action_fallback():
    """常用查询使用中文名称，未来未知动作保留原名便于识别。"""
    assert action_label("get_order_details") == "查询订单详情"
    assert action_label("get_product_details") == "查询商品详情"
    assert action_label("get_user_details") == "查询账户详情"
    assert action_label("future_action") == "future_action"

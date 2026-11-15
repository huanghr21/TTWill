"""验证零售写操作的业务限制、逐件替换和资金变化。"""

from copy import deepcopy

import pytest

from ttwill.backend.rules import WRITE_ACTIONS, BusinessError, apply_action


@pytest.fixture
def entities() -> tuple[dict, dict, dict]:
    """构造含两类商品、两张礼品卡及两种普通支付方式的小型业务数据。

    Returns:
        按顺序包含以下对象的三元组：
            user: 含地址与支付方式的用户数据。
            order: 包含两类商品的待发货订单数据。
            products: 按商品 ID 索引的商品与规格映射。
    """
    address = dict(
        address1="原地址", address2="", city="上海", state="上海", country="中国", zip="200000"
    )
    user = {
        "user_id": "u1",
        "address": address,
        "payment_methods": {
            "card": {"id": "card", "source": "credit_card"},
            "paypal": {"id": "paypal", "source": "paypal"},
            "gift": {"id": "gift", "source": "gift_card", "balance": 300.0},
            "gift2": {"id": "gift2", "source": "gift_card", "balance": 0.0},
        },
    }
    products = {
        "p1": {
            "product_id": "p1",
            "variants": {
                "a": {"item_id": "a", "price": 20.0, "available": True, "options": {"颜色": "蓝"}},
                "b": {"item_id": "b", "price": 30.0, "available": True, "options": {"颜色": "红"}},
                "c": {"item_id": "c", "price": 10.0, "available": False, "options": {"颜色": "绿"}},
            },
        },
        "p2": {
            "product_id": "p2",
            "variants": {
                "x": {"item_id": "x", "price": 40.0, "available": True, "options": {"尺码": "小"}},
                "y": {"item_id": "y", "price": 60.0, "available": True, "options": {"尺码": "大"}},
            },
        },
    }
    order = {
        "order_id": "#1",
        "user_id": "u1",
        "address": deepcopy(address),
        "status": "pending",
        "items": [
            {"product_id": "p1", "item_id": "a", "price": 20.0, "options": {"颜色": "蓝"}},
            {"product_id": "p2", "item_id": "x", "price": 40.0, "options": {"尺码": "小"}},
        ],
        "payment_history": [
            {"transaction_type": "payment", "amount": 60.0, "payment_method_id": "card"}
        ],
    }
    return user, order, products


def test_seven_write_actions() -> None:
    """业务入口仅包含当前阶段的七个写动作。"""
    assert len(WRITE_ACTIONS) == 7
    assert "modify_user_address" in WRITE_ACTIONS


@pytest.mark.parametrize("action", ["modify_user_address", "modify_pending_order_address"])
def test_address_updates_only_requested_entity(entities: tuple, action: str) -> None:
    """用户默认地址与订单收件地址独立更新。"""
    user, order, products = entities
    address = {**user["address"], "address1": "新地址"}
    result = apply_action(action, address, user, order, products)
    assert result["address"] == address
    unchanged = order if action == "modify_user_address" else user
    assert unchanged["address"]["address1"] == "原地址"


def test_cancel_refunds_gift_card_immediately(entities: tuple) -> None:
    """取消订单登记退款，并立即增加对应礼品卡余额。"""
    user, order, products = entities
    order["payment_history"][0]["payment_method_id"] = "gift"
    apply_action("cancel_pending_order", {"reason": "no longer needed"}, user, order, products)
    assert order["status"] == "cancelled"
    assert order["cancel_reason"] == "no longer needed"
    assert order["payment_history"][-1] == {
        "transaction_type": "refund",
        "amount": 60.0,
        "payment_method_id": "gift",
    }
    assert user["payment_methods"]["gift"]["balance"] == 360.0


def test_cancel_after_payment_change_refunds_only_outstanding_payment(entities: tuple) -> None:
    """改支付后取消只退净实付，避免再次退还已经退款的旧付款。"""
    user, order, products = entities
    order["payment_history"][0]["payment_method_id"] = "gift"
    apply_action(
        "modify_pending_order_payment", {"payment_method_id": "card"}, user, order, products
    )
    apply_action("cancel_pending_order", {"reason": "ordered by mistake"}, user, order, products)
    assert user["payment_methods"]["gift"]["balance"] == 360.0
    assert len(order["payment_history"]) == 4
    assert order["payment_history"][-1] == {
        "transaction_type": "refund",
        "amount": 60.0,
        "payment_method_id": "card",
    }


def test_invalid_cancel_reason_leaves_entities_unchanged(entities: tuple) -> None:
    """不支持的取消原因不能改动订单或余额。"""
    before = deepcopy(entities)
    user, order, products = entities
    with pytest.raises(BusinessError, match="取消原因"):
        apply_action("cancel_pending_order", {"reason": "other"}, user, order, products)
    assert entities == before


def test_change_payment_moves_funds_between_gift_cards(entities: tuple) -> None:
    """更改支付同时记录新付款与原路退款。"""
    user, order, products = entities
    order["payment_history"][0]["payment_method_id"] = "gift2"
    apply_action(
        "modify_pending_order_payment", {"payment_method_id": "gift"}, user, order, products
    )
    assert user["payment_methods"]["gift"]["balance"] == 240.0
    assert user["payment_methods"]["gift2"]["balance"] == 60.0
    assert [entry["transaction_type"] for entry in order["payment_history"]] == [
        "payment",
        "payment",
        "refund",
    ]
    assert order["status"] == "pending"


@pytest.mark.parametrize(
    ("payment_id", "code"),
    [
        ("card", "unchanged_payment"),
        ("other", "payment_method_not_found"),
        ("gift2", "insufficient_balance"),
    ],
)
def test_change_payment_rejects_invalid_method(entities: tuple, payment_id: str, code: str) -> None:
    """相同、未归属及余额不足的支付方式均不能替换原付款。"""
    user, order, products = entities
    before = deepcopy(entities)
    with pytest.raises(BusinessError) as caught:
        apply_action(
            "modify_pending_order_payment", {"payment_method_id": payment_id}, user, order, products
        )
    assert caught.value.code == code
    assert entities == before


def test_payment_can_only_be_changed_once(entities: tuple) -> None:
    """已产生付款与退款组合的订单不能再次更改支付方式。"""
    user, order, products = entities
    apply_action(
        "modify_pending_order_payment", {"payment_method_id": "paypal"}, user, order, products
    )
    with pytest.raises(BusinessError) as caught:
        apply_action(
            "modify_pending_order_payment", {"payment_method_id": "gift"}, user, order, products
        )
    assert caught.value.code == "payment_history_conflict"


def test_modify_items_uses_each_new_variant(entities: tuple) -> None:
    """批量改商品逐项写入对应的新价格和规格，并一次结算总差价。"""
    user, order, products = entities
    args = {"item_ids": ["a", "x"], "new_item_ids": ["b", "y"], "payment_method_id": "gift"}
    apply_action("modify_pending_order_items", args, user, order, products)
    assert [(item["item_id"], item["price"], item["options"]) for item in order["items"]] == [
        ("b", 30.0, {"颜色": "红"}),
        ("y", 60.0, {"尺码": "大"}),
    ]
    assert order["status"] == "pending (item modified)"
    assert order["payment_history"][-1]["amount"] == 30.0
    assert user["payment_methods"]["gift"]["balance"] == 270.0
    order["items"][0]["options"]["颜色"] = "改订单副本"
    assert products["p1"]["variants"]["b"]["options"] == {"颜色": "红"}


def test_modify_items_uses_original_positions_during_swap(entities: tuple) -> None:
    """同类规格互换按修改前的订单定位，避免新 ID 命中已替换的条目。"""
    user, order, products = entities
    order["items"][1] = {
        "product_id": "p1",
        "item_id": "b",
        "price": 30.0,
        "options": {"颜色": "红"},
    }
    args = {"item_ids": ["a", "b"], "new_item_ids": ["b", "a"], "payment_method_id": "gift"}
    apply_action("modify_pending_order_items", args, user, order, products)
    assert [item["item_id"] for item in order["items"]] == ["b", "a"]
    assert order["payment_history"][-1] == {
        "transaction_type": "refund",
        "amount": 0.0,
        "payment_method_id": "gift",
    }
    assert user["payment_methods"]["gift"]["balance"] == 300.0


def test_duplicate_items_are_modified_once_per_occurrence(entities: tuple) -> None:
    """同一规格买了两件时，两次修改分别作用于两个原始条目。"""
    user, order, products = entities
    order["items"][1] = deepcopy(order["items"][0])
    args = {"item_ids": ["a", "a"], "new_item_ids": ["b", "b"], "payment_method_id": "gift"}
    apply_action("modify_pending_order_items", args, user, order, products)
    assert [item["item_id"] for item in order["items"]] == ["b", "b"]
    assert order["payment_history"][-1]["amount"] == 20.0


@pytest.mark.parametrize(
    ("item_ids", "new_item_ids", "code"),
    [
        (["a", "a"], ["b", "b"], "item_not_found"),
        (["a"], [], "item_count_mismatch"),
        (["a"], ["a"], "unchanged_item"),
        (["a"], ["x"], "variant_unavailable"),
        (["a"], ["c"], "variant_unavailable"),
        (["a"], ["missing"], "variant_unavailable"),
    ],
)
def test_invalid_replacements_preserve_entities(
    entities: tuple, item_ids: list[str], new_item_ids: list[str], code: str
) -> None:
    """商品数量、同类规格、库存或变更要求不满足时，业务实体保持原样。"""
    user, order, products = entities
    before = deepcopy(entities)
    args = {"item_ids": item_ids, "new_item_ids": new_item_ids, "payment_method_id": "gift"}
    with pytest.raises(BusinessError) as caught:
        apply_action("modify_pending_order_items", args, user, order, products)
    assert caught.value.code == code
    assert entities == before


@pytest.mark.parametrize("action", ["modify_pending_order_items", "exchange_delivered_order_items"])
def test_replacement_needs_enough_gift_card_balance(entities: tuple, action: str) -> None:
    """修改商品和换货都要在受理前检查差价对应的礼品卡余额。"""
    user, order, products = entities
    if action == "exchange_delivered_order_items":
        order["status"] = "delivered"
    before = deepcopy(entities)
    args = {"item_ids": ["a"], "new_item_ids": ["b"], "payment_method_id": "gift2"}
    with pytest.raises(BusinessError) as caught:
        apply_action(action, args, user, order, products)
    assert caught.value.code == "insufficient_balance"
    assert entities == before


def test_cheaper_replacement_refunds_difference(entities: tuple) -> None:
    """新商品更便宜时将差价退回指定礼品卡。"""
    user, order, products = entities
    products["p1"]["variants"]["b"]["price"] = 15.0
    args = {"item_ids": ["a"], "new_item_ids": ["b"], "payment_method_id": "gift"}
    apply_action("modify_pending_order_items", args, user, order, products)
    assert user["payment_methods"]["gift"]["balance"] == 305.0
    assert order["payment_history"][-1]["transaction_type"] == "refund"
    assert order["payment_history"][-1]["amount"] == 5.0


@pytest.mark.parametrize(
    "action",
    [
        "cancel_pending_order",
        "modify_pending_order_address",
        "modify_pending_order_payment",
        "modify_pending_order_items",
    ],
)
def test_modified_order_rejects_further_pending_operations(entities: tuple, action: str) -> None:
    """修改商品后的订单不得继续修改地址、支付、商品或取消。"""
    user, order, products = entities
    order["status"] = "pending (item modified)"
    with pytest.raises(BusinessError) as caught:
        apply_action(action, {}, user, order, products)
    assert caught.value.code == "invalid_status"


def test_exchange_records_request_without_charging(entities: tuple) -> None:
    """换货申请保存排序后的商品清单与差价，不立即改商品或扣款。"""
    user, order, products = entities
    order["status"] = "delivered"
    original_items = deepcopy(order["items"])
    args = {"item_ids": ["x", "a"], "new_item_ids": ["y", "b"], "payment_method_id": "gift"}
    apply_action("exchange_delivered_order_items", args, user, order, products)
    assert order["status"] == "exchange requested"
    assert order["exchange_items"] == ["a", "x"]
    assert order["exchange_new_items"] == ["b", "y"]
    assert order["exchange_price_difference"] == 30.0
    assert order["items"] == original_items
    assert len(order["payment_history"]) == 1
    assert user["payment_methods"]["gift"]["balance"] == 300.0


def test_exchange_requires_a_different_variant(entities: tuple) -> None:
    """换货应换为不同规格，不能将原商品再次登记为替代商品。"""
    user, order, products = entities
    order["status"] = "delivered"
    before = deepcopy(entities)
    args = {"item_ids": ["a"], "new_item_ids": ["a"], "payment_method_id": "gift"}
    with pytest.raises(BusinessError) as caught:
        apply_action("exchange_delivered_order_items", args, user, order, products)
    assert caught.value.code == "unchanged_item"
    assert entities == before


@pytest.mark.parametrize("payment_id", ["card", "gift"])
def test_return_accepts_original_method_or_gift_card(entities: tuple, payment_id: str) -> None:
    """退货登记接受原支付方式或已有礼品卡，不立即退款。"""
    user, order, products = entities
    order["status"] = "delivered"
    args = {"item_ids": ["x", "a"], "payment_method_id": payment_id}
    apply_action("return_delivered_order_items", args, user, order, products)
    assert order["status"] == "return requested"
    assert order["return_items"] == ["a", "x"]
    assert order["return_payment_method_id"] == payment_id
    assert len(order["payment_history"]) == 1
    assert user["payment_methods"]["gift"]["balance"] == 300.0


def test_return_rejects_other_normal_payment_method(entities: tuple) -> None:
    """即使属于当前用户，非原付款的普通支付方式也不能接收退货退款。"""
    user, order, products = entities
    order["status"] = "delivered"
    with pytest.raises(BusinessError) as caught:
        apply_action(
            "return_delivered_order_items",
            {"item_ids": ["a"], "payment_method_id": "paypal"},
            user,
            order,
            products,
        )
    assert caught.value.code == "invalid_refund_method"


@pytest.mark.parametrize(
    "action", ["return_delivered_order_items", "exchange_delivered_order_items"]
)
@pytest.mark.parametrize(
    "status", ["pending", "processed", "return requested", "exchange requested"]
)
def test_after_sales_requires_delivered_status(entities: tuple, action: str, status: str) -> None:
    """只有已送达且未申请过售后的订单能够办理退换货。"""
    user, order, products = entities
    order["status"] = status
    with pytest.raises(BusinessError) as caught:
        apply_action(action, {}, user, order, products)
    assert caught.value.code == "invalid_status"


def test_unknown_action_has_explicit_error_code(entities: tuple) -> None:
    """未注册动作返回稳定错误码，便于上层记录拒绝原因。"""
    user, order, products = entities
    with pytest.raises(BusinessError) as caught:
        apply_action("delete_user", {}, user, order, products)
    assert caught.value.code == "unknown_action"

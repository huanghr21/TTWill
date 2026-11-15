"""基于实体副本校验并应用订单与售后业务规则。"""

from collections import defaultdict, deque
from copy import deepcopy

WRITE_ACTIONS = frozenset(
    {
        "cancel_pending_order",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
        "exchange_delivered_order_items",
    }
)
ADDRESS_FIELDS = ("address1", "address2", "city", "state", "country", "zip")


class BusinessError(ValueError):
    """表示业务规则未满足，并提供稳定的错误码。

    Attributes:
        code: 供调用方识别和记录的业务错误码。
    """

    def __init__(self, message: str, code: str = "business_rule") -> None:
        """保存错误原因与业务错误码。

        Args:
            message: 向调用方说明的错误原因。
            code: 用于区分错误类型的稳定标识，默认表示一般业务规则错误。
        """
        super().__init__(message)
        self.code = code


def apply_action(
    action: str,
    args: dict,
    user: dict,
    order: dict | None,
    products: dict[str, dict],
) -> dict:
    """校验业务规则并在传入的实体副本上应用一个写动作。

    参数结构与实体归属由后端入口校验。订单付款变化也会修改用户礼品卡余额，
    调用方须将相关实体放在同一事务中保存。

    Args:
        action: WRITE_ACTIONS 中的写操作名称。
        args: 已完成结构校验的操作参数。
        user: 当前用户的实体副本；默认地址和礼品卡余额可能被原地修改。
        order: 目标订单的实体副本；修改用户默认地址时可为 None。
        products: 订单涉及的商品数据，以商品 ID 为键，包含可替换的规格信息。

    Returns:
        修改后的订单字典；修改用户默认地址时返回用户字典。返回值与传入的
        对应实体为同一对象。

    Raises:
        BusinessError: 写操作未知、所需订单不存在，或订单状态、商品、支付等
            业务规则不满足。
    """
    if action not in WRITE_ACTIONS:
        raise BusinessError("未知写操作", "unknown_action")
    if action == "modify_user_address":
        user["address"] = {field: args[field] for field in ADDRESS_FIELDS}
        return user
    if order is None:
        raise BusinessError("订单不存在", "not_found")

    expected_status = "delivered" if action.endswith("delivered_order_items") else "pending"
    if order["status"] != expected_status:
        raise BusinessError(f"订单状态必须为 {expected_status}", "invalid_status")

    if action == "modify_pending_order_address":
        order["address"] = {field: args[field] for field in ADDRESS_FIELDS}
    elif action == "cancel_pending_order":
        _cancel_order(args, user, order)
    elif action == "modify_pending_order_payment":
        _modify_payment(args, user, order)
    elif action == "return_delivered_order_items":
        _return_items(args, user, order)
    else:
        _replace_items(action, args, user, order, products)
    return order


def _payment_method(user: dict, payment_method_id: str) -> dict:
    """取得用户自己的支付方式。"""
    payment = user["payment_methods"].get(payment_method_id)
    if payment is None:
        raise BusinessError("支付方式不属于当前用户", "payment_method_not_found")
    return payment


def _check_balance(payment: dict, amount: float) -> None:
    """使用礼品卡补款时校验可用余额。"""
    if payment["source"] == "gift_card" and payment["balance"] < amount:
        raise BusinessError("礼品卡余额不足", "insufficient_balance")


def _change_balance(payment: dict, amount: float) -> None:
    """以正数入账、负数扣款更新礼品卡余额并保留两位小数。"""
    if payment["source"] == "gift_card":
        payment["balance"] = round(payment["balance"] + amount, 2)


def _select_items(order: dict, item_ids: list[str]) -> list[dict]:
    """按原订单逐件匹配请求，重复商品不能超过已购数量。

    Args:
        order: 包含已购商品清单的订单。
        item_ids: 待处理的规格 ID，重复购买的商品按件重复列出。

    Returns:
        按 item_ids 顺序匹配的订单商品列表；各项引用原订单中的商品字典。

    Raises:
        BusinessError: 请求中的商品不存在，或请求数量超过订单内的数量。
    """
    by_id: dict[str, deque[dict]] = defaultdict(deque)
    for item in order["items"]:
        by_id[item["item_id"]].append(item)
    selected = []
    for item_id in item_ids:
        if not by_id[item_id]:
            raise BusinessError(f"订单内商品 {item_id} 的数量不足", "item_not_found")
        selected.append(by_id[item_id].popleft())
    return selected


def _replacement_pairs(
    args: dict, order: dict, products: dict[str, dict]
) -> list[tuple[dict, dict]]:
    """按原始商品定位全部替换对，校验同类规格及库存。

    Args:
        args: 包含 item_ids 与 new_item_ids 的参数，两份清单按位置一一对应。
        order: 包含待替换商品的订单。
        products: 以商品 ID 为键的商品字典，包含各商品的规格与可用状态。

    Returns:
        与请求顺序一致的替换对列表，每对为按顺序包含以下对象的二元组：
            item: 原订单中的商品字典引用。
            variant: 商品字典中的目标规格引用。

    Raises:
        BusinessError: 新旧清单长度不同、原商品数量不足、商品不存在，或目标
            规格与原规格相同、不属于同一商品、没有库存。
    """
    if len(args["item_ids"]) != len(args["new_item_ids"]):
        raise BusinessError("新旧商品数量必须一致", "item_count_mismatch")
    selected = _select_items(order, args["item_ids"])
    pairs = []
    for item, new_item_id in zip(selected, args["new_item_ids"], strict=True):
        if item["item_id"] == new_item_id:
            raise BusinessError("新规格必须不同于原规格", "unchanged_item")
        product = products.get(item["product_id"])
        if product is None:
            raise BusinessError("商品不存在", "product_not_found")
        variant = product["variants"].get(new_item_id)
        if variant is None or not variant["available"]:
            raise BusinessError("新规格必须属于同一商品且有库存", "variant_unavailable")
        pairs.append((item, variant))
    return pairs


def _replace_items(
    action: str, args: dict, user: dict, order: dict, products: dict[str, dict]
) -> None:
    """修改待发货商品并结算差价，或登记已送达商品的换货申请。

    Args:
        action: modify_pending_order_items 或 exchange_delivered_order_items。
        args: 包含原规格清单、新规格清单与支付方式 ID 的参数。
        user: 当前用户副本；修改待发货商品时会原地更新相关礼品卡余额。
        order: 订单副本，原地更新商品、付款记录或换货申请与订单状态。
        products: 以商品 ID 为键的商品字典，用于校验目标规格及计算差价。

    Raises:
        BusinessError: 商品替换条件不满足、支付方式不属于当前用户，或礼品卡
            余额不足以支付差价。
    """
    pairs = _replacement_pairs(args, order, products)
    difference = sum(variant["price"] - item["price"] for item, variant in pairs)
    if action == "exchange_delivered_order_items":
        difference = round(difference, 2)
    payment_id = args["payment_method_id"]
    payment = _payment_method(user, payment_id)
    _check_balance(payment, difference)

    if action == "exchange_delivered_order_items":
        order.update(
            status="exchange requested",
            exchange_items=sorted(args["item_ids"]),
            exchange_new_items=sorted(args["new_item_ids"]),
            exchange_payment_method_id=payment_id,
            exchange_price_difference=difference,
        )
        return

    order["payment_history"].append(
        {
            "transaction_type": "payment" if difference > 0 else "refund",
            "amount": abs(difference),
            "payment_method_id": payment_id,
        }
    )
    _change_balance(payment, -difference)
    for item, variant in pairs:
        item.update(
            item_id=variant["item_id"],
            price=variant["price"],
            options=deepcopy(variant["options"]),
        )
    order["status"] = "pending (item modified)"


def _modify_payment(args: dict, user: dict, order: dict) -> None:
    """将唯一一次原付款退款，并以新支付方式支付相同金额。

    Args:
        args: 包含新 payment_method_id 的参数。
        user: 当前用户副本，原地更新涉及的礼品卡余额。
        order: 订单副本，原地追加新付款和原付款退款的记录。

    Raises:
        BusinessError: 订单不只有一笔原付款、新旧支付方式相同、支付方式不属于
            当前用户，或新支付方式的礼品卡余额不足。
    """
    history = order["payment_history"]
    if len(history) != 1 or history[0]["transaction_type"] != "payment":
        raise BusinessError("更改支付方式前必须只有一笔付款", "payment_history_conflict")
    old_payment = history[0]
    payment_id = args["payment_method_id"]
    if payment_id == old_payment["payment_method_id"]:
        raise BusinessError("新支付方式必须与原支付方式不同", "unchanged_payment")
    payment = _payment_method(user, payment_id)
    previous = _payment_method(user, old_payment["payment_method_id"])
    amount = old_payment["amount"]
    _check_balance(payment, amount)
    history.extend(
        [
            {"transaction_type": "payment", "amount": amount, "payment_method_id": payment_id},
            {
                "transaction_type": "refund",
                "amount": amount,
                "payment_method_id": old_payment["payment_method_id"],
            },
        ]
    )
    _change_balance(payment, -amount)
    _change_balance(previous, amount)


def _return_items(args: dict, user: dict, order: dict) -> None:
    """向原支付方式或用户已有礼品卡登记退货申请。

    Args:
        args: 包含退货商品清单 item_ids 与退款方式 payment_method_id 的参数。
        user: 当前用户，用于确认退款方式归属与类型。
        order: 订单副本，原地保存退货商品、退款方式及 return requested 状态。

    Raises:
        BusinessError: 退款方式不属于当前用户、不满足退款规则，或退货商品数量
            超过订单内的数量。
    """
    payment_id = args["payment_method_id"]
    payment = _payment_method(user, payment_id)
    if (
        payment["source"] != "gift_card"
        and payment_id != order["payment_history"][0]["payment_method_id"]
    ):
        raise BusinessError("退款必须退回原支付方式或已有礼品卡", "invalid_refund_method")
    _select_items(order, args["item_ids"])
    order.update(
        status="return requested",
        return_items=sorted(args["item_ids"]),
        return_payment_method_id=payment_id,
    )


def _cancel_order(args: dict, user: dict, order: dict) -> None:
    """取消待发货订单，将各支付方式尚未退款的实付金额退回。

    Args:
        args: 包含取消原因 reason 的参数，只接受 no longer needed 或
            ordered by mistake。
        user: 当前用户副本，原地退回涉及的礼品卡余额。
        order: 订单副本，原地追加退款记录并保存取消原因和 cancelled 状态。

    Raises:
        BusinessError: 取消原因不在允许范围内，或需退款的支付方式不属于当前用户。
    """
    reason = args["reason"]
    if reason not in {"no longer needed", "ordered by mistake"}:
        raise BusinessError("取消原因必须为不再需要或误下单", "invalid_cancel_reason")
    balances: dict[str, float] = defaultdict(float)
    for transaction in order["payment_history"]:
        sign = 1 if transaction["transaction_type"] == "payment" else -1
        balances[transaction["payment_method_id"]] += sign * transaction["amount"]
    refunds = []
    for payment_id, amount in balances.items():
        if amount > 0:
            payment = _payment_method(user, payment_id)
            _change_balance(payment, amount)
            refunds.append(
                {"transaction_type": "refund", "amount": amount, "payment_method_id": payment_id}
            )
    order["payment_history"].extend(refunds)
    order.update(status="cancelled", cancel_reason=reason)

"""将已核对的业务操作转换成用户可确认的中文详情。"""

import json
from typing import Any

_ACTION_LABELS = {
    "find_user_id_by_email": "通过邮箱验证身份",
    "find_user_id_by_name_zip": "通过姓名与邮编验证身份",
    "get_order_details": "查询订单详情",
    "get_product_details": "查询商品详情",
    "get_item_details": "查询商品规格",
    "get_user_details": "查询账户详情",
    "list_all_product_types": "查询商品目录",
    "cancel_pending_order": "取消订单",
    "modify_pending_order_address": "修改订单收货地址",
    "modify_pending_order_items": "修改订单商品规格",
    "modify_pending_order_payment": "修改订单支付方式",
    "modify_user_address": "修改账户默认地址",
    "return_delivered_order_items": "申请退货",
    "exchange_delivered_order_items": "申请换货",
}
_FIELD_LABELS = {
    "order_id": "订单编号",
    "user_id": "办理账户",
    "reason": "取消原因",
    "cancel_reason": "取消原因",
    "item_ids": "商品清单",
    "new_item_ids": "替换规格",
    "payment_method_id": "支付方式",
    "address": "地址",
    "address1": "地址第一行",
    "address2": "地址第二行",
    "city": "城市",
    "state": "省或州",
    "country": "国家",
    "zip": "邮政编码",
    "status": "订单状态",
    "items": "订单商品",
    "name": "商品名称",
    "product_id": "商品编号",
    "item_id": "商品规格编号",
    "price": "价格",
    "options": "规格",
    "payment_history": "支付与退款明细",
    "transaction_type": "类型",
    "amount": "金额",
    "return_items": "退货商品清单",
    "return_payment_method_id": "退货退款方式",
    "exchange_items": "换货商品清单",
    "exchange_new_items": "换货新规格",
    "exchange_payment_method_id": "换货差价支付方式",
    "exchange_price_difference": "换货差价",
}
_STATUS_LABELS = {
    "pending": "待发货",
    "pending (item modified)": "待发货（商品修改后）",
    "delivered": "已送达",
    "cancelled": "取消",
    "return requested": "退货申请中",
    "exchange requested": "换货申请中",
}
_REASON_LABELS = {
    "no longer needed": "不再需要（no longer needed）",
    "ordered by mistake": "误下单（ordered by mistake）",
}
_PREVIEW_FIELDS = {
    "cancel_pending_order": ("status", "cancel_reason", "payment_history"),
    "modify_pending_order_address": ("status", "address"),
    "modify_pending_order_items": ("status", "items", "payment_history"),
    "modify_pending_order_payment": ("status", "payment_history"),
    "modify_user_address": ("address",),
    "return_delivered_order_items": (
        "status",
        "items",
        "return_items",
        "return_payment_method_id",
        "payment_history",
    ),
    "exchange_delivered_order_items": (
        "status",
        "items",
        "exchange_items",
        "exchange_new_items",
        "exchange_payment_method_id",
        "exchange_price_difference",
        "payment_history",
    ),
}


def action_label(action: str) -> str:
    """取得业务动作的中文名称，未知动作保留原名。

    Args:
        action: 业务工具名称。

    Returns:
        固定中文动作名称；未登记时返回 action 原文。
    """
    return _ACTION_LABELS.get(action, action)


def operation_content(operation: dict) -> str:
    """展示完整办理参数与对应预期结果，不将准备预览表述为已经完成。

    Args:
        operation: 已核对身份、参数和状态的后端准备记录，包含 action、args
            与 preview。金额、商品及支付方式标识保留原值；用户身份参数以
            当前账户表达，不展示内部用户标识或准备记录的关联标识。

    Returns:
        包含中文动作、办理对象、完整参数与动作相关预期结果的展示文本。
        用户地址操作仅展示地址预期结果，其他账户资料不进入正文。
    """
    action = operation["action"]
    arguments = operation["args"]
    preview = operation["preview"]
    target = f"订单：{arguments['order_id']}" if "order_id" in arguments else "账户：当前账户"
    fields = _PREVIEW_FIELDS.get(action, ())
    expected = {key: preview[key] for key in fields if key in preview}
    return (
        f"待确认：{action_label(action)}\n{target}\n\n"
        f"办理内容：\n{json.dumps(_translate(arguments), ensure_ascii=False, indent=2)}\n\n"
        f"确认后预期结果：\n{json.dumps(_translate(expected), ensure_ascii=False, indent=2)}"
    )


def _translate(value: Any, field: str | None = None) -> Any:
    """翻译已选展示字段，递归复制结构并保留金额、清单顺序与业务标识。

    Args:
        value: 要展示的业务字段值，包含字典、列表或 JSON 标量。
        field: 当前值对应的原字段名，供状态、原因与交易类型翻译使用。

    Returns:
        中文字段组成的独立结构；未知字段和标量按来源值保留，用户标识替换
        为当前账户说明。
    """
    if field == "user_id":
        return "当前账户"
    if isinstance(value, dict):
        return {_FIELD_LABELS.get(key, key): _translate(item, key) for key, item in value.items()}
    if isinstance(value, list):
        return [_translate(item) for item in value]
    if field == "status":
        return _STATUS_LABELS.get(value, value)
    if field in {"reason", "cancel_reason"}:
        return _REASON_LABELS.get(value, value)
    if field == "transaction_type":
        return {"payment": "付款", "refund": "退款"}.get(value, value)
    return value

"""在模型业务入口规范标识符，并用已验证账户的权威数据限定取值。"""

import json
import re
from copy import deepcopy

from ttwill.backend import BusinessError, RetailBackend


def require_identifier(value: str, allowed: list[str], field: str) -> str:
    """校验一个精确标识符，并在拒绝时列出同范围的合法选择。

    Args:
        value: 已完成确定性格式转换的模型输入。
        allowed: 从当前用户权威数据取得的合法标识符。
        field: 用于错误说明的参数字段名。

    Returns:
        通过白名单校验的原值。

    Raises:
        BusinessError: 输入不在白名单中。
    """
    if value not in allowed:
        choices = json.dumps(sorted(set(allowed)), ensure_ascii=False)
        raise BusinessError(f"{field} 不合法；合法值：{choices}", code="invalid_identifier")
    return value


def normalize_order_id(value: str, allowed: list[str]) -> str:
    """规范订单号格式并检查其属于已验证用户的订单列表。

    Args:
        value: 模型给出的订单号，只允许去首尾空白、转大写和补开头的 #。
        allowed: 当前用户资料中的权威订单标识符列表。

    Returns:
        满足 #W 加七位 ASCII 数字且属于白名单的订单号。

    Raises:
        BusinessError: 格式错误或订单不在白名单中，异常包含合法订单列表。
    """
    value = value.strip().upper()
    if not value.startswith("#"):
        value = f"#{value}"
    if re.fullmatch(r"#W[0-9]{7}", value) is None:
        choices = json.dumps(sorted(set(allowed)), ensure_ascii=False)
        raise BusinessError(
            f"order_id 须为 #W 加七位数字；合法值：{choices}", code="invalid_identifier"
        )
    return require_identifier(value, allowed, "order_id")


def normalize_arguments(
    backend: RetailBackend, user_id: str, args: dict, *, request_key: str | None = None
) -> dict:
    """以业务库为准校验模型给出的订单、商品规格和支付方式。

    Args:
        backend: 当前评测任务或业务会话使用的真实业务库。
        user_id: 代码保存的已验证用户身份。
        args: 已通过工具参数结构校验的参数字典。
        request_key: 写操作的稳定请求标识；已有操作参数完全相同时按操作记录
            重放，避免已提交的规格修改导致旧商品不再存在。

    Returns:
        与输入隔离、全部标识符采用规范值的参数字典。替换规格逐项限定于
        对应原商品；无订单字段的单规格查询限定于用户订单涉及的商品规格。

    Raises:
        BusinessError: 标识符不属于对应权威白名单，或新旧商品清单长度不同。
    """
    result = deepcopy(args)
    user = backend.query(user_id, "get_user_details", {"user_id": user_id})
    for field in ("user_id", "product_id", "item_id", "payment_method_id"):
        if field in result:
            result[field] = result[field].strip()
    if "user_id" in result:
        require_identifier(result["user_id"], [user_id], "user_id")
    if "order_id" in result:
        result["order_id"] = normalize_order_id(result["order_id"], user["orders"])
    for field in ("item_ids", "new_item_ids"):
        if field in result:
            result[field] = [value.strip() for value in result[field]]
    if request_key is not None:
        try:
            operation = backend.get_operation(user_id, request_key=request_key)
        except BusinessError as exc:
            if exc.code != "not_found":
                raise
        else:
            if result == operation["args"]:
                return result
    if "payment_method_id" in result:
        require_identifier(
            result["payment_method_id"], list(user["payment_methods"]), "payment_method_id"
        )
    if "item_ids" in result:
        order = backend.query(user_id, "get_order_details", {"order_id": result["order_id"]})
        items = {item["item_id"]: item for item in order["items"]}
        result["item_ids"] = [
            require_identifier(value, list(items), "item_ids") for value in result["item_ids"]
        ]
        if "new_item_ids" in result:
            if len(result["item_ids"]) != len(result["new_item_ids"]):
                raise BusinessError("新旧商品数量必须一致", code="item_count_mismatch")
            result["new_item_ids"] = [
                require_identifier(
                    value,
                    list(
                        backend.query(
                            user_id,
                            "get_product_details",
                            {"product_id": items[old]["product_id"]},
                        )["variants"]
                    ),
                    f"new_item_ids[{index}]（原规格 {old}）",
                )
                for index, (old, value) in enumerate(
                    zip(result["item_ids"], result["new_item_ids"], strict=True)
                )
            ]
    if "item_id" in result:
        product_ids = {
            item["product_id"]
            for order_id in user["orders"]
            for item in backend.query(user_id, "get_order_details", {"order_id": order_id})["items"]
        }
        variants = {
            item_id
            for product_id in product_ids
            for item_id in backend.query(
                user_id, "get_product_details", {"product_id": product_id}
            )["variants"]
        }
        require_identifier(result["item_id"], sorted(variants), "item_id")
    return result

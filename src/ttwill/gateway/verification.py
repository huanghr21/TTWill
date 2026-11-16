"""按写操作选择业务字段，将实际重读结果与提交参数和记录逐项核对。"""

from collections import Counter, defaultdict
from copy import deepcopy
from math import isclose

from ttwill.backend import BusinessError
from ttwill.backend.rules import ADDRESS_FIELDS, WRITE_ACTIONS


def verification_query(operation: dict) -> tuple[str, dict]:
    """按已提交操作选择需要重新读取的业务实体。

    Args:
        operation: 经过身份归属检查的后端操作记录。

    Returns:
        查询工具名及其参数；默认地址读取用户，其他动作读取订单。

    Raises:
        BusinessError: 操作的动作类型不在已支持的七种写操作中。
    """
    action = operation["action"]
    if action not in WRITE_ACTIONS:
        raise BusinessError("该写操作没有对应的结果核验规则", code="unknown_action")
    if action == "modify_user_address":
        return "get_user_details", {"user_id": operation["args"]["user_id"]}
    return "get_order_details", {"order_id": operation["args"]["order_id"]}


def _expected_fields(operation: dict) -> dict:
    """根据动作参数构造目标状态，并从提交结果取得商品与金额明细。

    Args:
        operation: 含有效 action、args、user_id 和 result 的已提交操作。

    Returns:
        本动作需要核对的字段；未涉及的用户订单列表、履约记录等不参与比较。
    """
    action, args, result = operation["action"], operation["args"], operation["result"]
    if action == "modify_user_address":
        return {
            "user_id": args["user_id"],
            "address": {field: args[field] for field in ADDRESS_FIELDS},
        }

    expected = {"order_id": args["order_id"], "user_id": operation["user_id"]}
    if action == "modify_pending_order_address":
        expected.update(status="pending", address={field: args[field] for field in ADDRESS_FIELDS})
    elif action == "cancel_pending_order":
        expected.update(
            status="cancelled",
            cancel_reason=args["reason"],
            payment_history=result["payment_history"],
        )
    elif action == "modify_pending_order_payment":
        expected.update(status="pending", payment_history=result["payment_history"])
    elif action == "modify_pending_order_items":
        expected.update(
            status="pending (item modified)",
            items=result["items"],
            payment_history=result["payment_history"],
        )
    elif action == "return_delivered_order_items":
        expected.update(
            status="return requested",
            return_items=sorted(args["item_ids"]),
            return_payment_method_id=args["payment_method_id"],
            items=result["items"],
        )
    elif action == "exchange_delivered_order_items":
        expected.update(
            status="exchange requested",
            exchange_items=sorted(args["item_ids"]),
            exchange_new_items=sorted(args["new_item_ids"]),
            exchange_payment_method_id=args["payment_method_id"],
            exchange_price_difference=result["exchange_price_difference"],
            items=result["items"],
            payment_history=result["payment_history"],
        )
    return deepcopy(expected)


def _argument_evidence(action: str, args: dict, actual: dict) -> str | None:
    """检查商品数量与收付款语义确实支持已经确认的请求。

    Args:
        action: 已支持的写动作名称。
        args: 用户确认时的准备参数。
        actual: 从业务库重新读取的当前订单或用户实体。

    Returns:
        不一致的原因；相关业务明细满足请求时为 None。
    """
    if action == "cancel_pending_order":
        balances = defaultdict(float)
        for entry in actual["payment_history"]:
            sign = 1 if entry["transaction_type"] == "payment" else -1
            balances[entry["payment_method_id"]] += sign * entry["amount"]
        if any(not isclose(amount, 0, abs_tol=1e-8) for amount in balances.values()):
            return "取消后仍有未退回的订单实付金额"
    elif action == "modify_pending_order_payment":
        history = actual["payment_history"]
        if len(history) != 3:
            return "更换支付方式后的付款和退款记录数量不符"
        original, payment, refund = history
        if (
            original["transaction_type"] != "payment"
            or payment
            != {
                "transaction_type": "payment",
                "amount": original["amount"],
                "payment_method_id": args["payment_method_id"],
            }
            or refund != {**original, "transaction_type": "refund"}
        ):
            return "新付款或原付款退款与确认的支付方式、金额不符"
    elif action == "modify_pending_order_items":
        actual_ids = Counter(item["item_id"] for item in actual["items"])
        if not Counter(args["new_item_ids"]) <= actual_ids:
            return "修改后的商品规格或数量未包含全部确认项"
        history = actual["payment_history"]
        if not history or history[-1]["payment_method_id"] != args["payment_method_id"]:
            return "商品差价的收付款方式与确认项不符"
    elif action in {"return_delivered_order_items", "exchange_delivered_order_items"}:
        actual_ids = Counter(item["item_id"] for item in actual["items"])
        if not Counter(args["item_ids"]) <= actual_ids:
            return "退换货申请中的商品规格或数量超出订单"
    return None


def verify_business_result(operation: dict, actual: dict) -> dict:
    """核对实际业务字段、提交结果和已确认参数，生成代码事实。

    当前核验针对提交后立即读取的实体。订单金额核验覆盖商品价格、付款及
    退款明细或换货差额；操作记录没有账户余额快照，因此不声明核验礼品卡余额。

    Args:
        operation: 状态为 committed、归属已校验的后端操作记录。
        actual: 从业务库重新读取的对应订单或用户实体。

    Returns:
        包含 operation_id、action、args、verified、expected、actual、reason 和
        checked_fields 的字典。actual 与 expected 只保留本次核验字段；reason
        在通过时为 None，不通过时说明字段或请求证据的差异。

    Raises:
        BusinessError: 动作未知，或操作尚未提交。
    """
    verification_query(operation)
    if operation["status"] != "committed":
        raise BusinessError("只能核验已提交操作", code="operation_not_committed")
    expected = _expected_fields(operation)
    observed = {field: deepcopy(actual.get(field)) for field in expected}
    recorded = {field: operation["result"].get(field) for field in expected}
    differences = [field for field in expected if observed[field] != expected[field]]
    recorded_differences = [field for field in expected if recorded[field] != expected[field]]
    reason = None
    if recorded_differences:
        reason = f"提交结果与确认参数不符：{', '.join(recorded_differences)}"
    elif differences:
        reason = f"业务数据与提交结果不符：{', '.join(differences)}"
    else:
        reason = _argument_evidence(operation["action"], operation["args"], actual)
    return {
        "operation_id": operation["operation_id"],
        "action": operation["action"],
        "args": deepcopy(operation["args"]),
        "verified": reason is None,
        "expected": expected,
        "actual": observed,
        "checked_fields": list(expected),
        "reason": reason,
    }

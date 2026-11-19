"""从网关事实生成业务数据增量，并按读取轮次限制参考数据的大小。"""

import json
from copy import deepcopy

_READ_ENTITIES = {
    "get_user_details": ("user", "user_id"),
    "get_order_details": ("order", "order_id"),
    "get_product_details": ("product", "product_id"),
    "get_item_details": ("item", "item_id"),
    "list_all_product_types": ("catalog", None),
}
_WRITE_ACTIONS = {
    "cancel_pending_order",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
}


def business_updates(
    records: list[dict], *, user_id: str, session_id: str, turn: int
) -> dict[str, dict]:
    """按调用顺序提取当前会话的真实业务读取，并使提交涉及的实体失效。

    Args:
        records: 按发生顺序排列的网关记录，只取成功的 call、verification_read
            和 commit；身份查询、Skill、准备预览及失败记录不提供缓存实体。
        user_id: 当前已验证用户，仅接受该用户的调用和所属实体。
        session_id: 当前聊天标识，其他聊天的记录不进入本次增量。
        turn: 这些记录所在的用户轮次，写入 read_turn 作为读取或失效轮次。

    Returns:
        以 user、order、product、item 前缀和业务标识组成键的最新增量，目录
        固定为 catalog:all。有效值包含 type、id、read_turn、result；失效值
        改含 needs_refresh 和 reason，不保留旧结果。订单提交同时使用户资料
        失效，以免继续引用可能已变化的支付余额；随后真实重读可覆盖失效标记。
        结果与输入不共享嵌套对象，可直接合并到调用方持有的业务数据区。
    """
    updates = {}
    for record in records:
        if record.get("user_id") != user_id or record.get("session_id") != session_id:
            continue
        kind = record.get("kind")
        action = record.get("action")
        if kind == "commit" and record.get("status") == "committed":
            operation = record["result"]
            if action not in _WRITE_ACTIONS or operation.get("user_id") != user_id:
                continue
            targets = [("user", user_id)]
            if action != "modify_user_address":
                targets.append(("order", operation["args"]["order_id"]))
            for entity_type, entity_id in targets:
                key = f"{entity_type}:{entity_id}"
                updates.pop(key, None)
                updates[key] = {
                    "type": entity_type,
                    "id": entity_id,
                    "read_turn": turn,
                    "needs_refresh": True,
                    "reason": "已变更，需重新查询",
                }
            continue
        if (
            kind not in {"call", "verification_read"}
            or record.get("status") != "executed"
            or record.get("read_write") != "read"
            or action not in _READ_ENTITIES
        ):
            continue
        entity_type, id_field = _READ_ENTITIES[action]
        result = record["result"]
        if entity_type == "item":
            # 后端原样返回规格，规格自身不必重复携带映射中的 item_id。
            entity_id = record["args"]["item_id"]
        else:
            entity_id = result[id_field] if id_field else "all"
        if entity_type in {"user", "order"} and result["user_id"] != user_id:
            continue
        key = f"{entity_type}:{entity_id}"
        updates.pop(key, None)
        updates[key] = {
            "type": entity_type,
            "id": entity_id,
            "read_turn": turn,
            "result": deepcopy(result),
        }
    return updates


def limit_business_data(data: dict, max_chars: int) -> None:
    """原地移出最久未读的实体结果，必要时再移出最旧提示以满足字符预算。

    Args:
        data: business_updates 形成并由调用方合并的实体映射；按 read_turn
            从旧到新裁剪，同轮按映射现有顺序处理。
        max_chars: list(data.values()) 用 ensure_ascii=False、紧凑 JSON 序列化
            后的字符上限，至少为 2，包含所有需重查提示。

    Raises:
        ValueError: max_chars 不是整数，或不足以容纳空列表的两个字符。
    """
    if type(max_chars) is not int or max_chars < 2:
        raise ValueError("业务数据预算至少为 2 个字符，才能容纳空列表")
    oldest = sorted(data, key=lambda key: data[key]["read_turn"])
    for key in oldest:
        if _data_chars(data) <= max_chars:
            return
        entry = data[key]
        if "result" in entry:
            data[key] = {
                "type": entry["type"],
                "id": entry["id"],
                "read_turn": entry["read_turn"],
                "needs_refresh": True,
                "reason": "已移出，需要时重新查询",
            }
    for key in oldest:
        if _data_chars(data) <= max_chars:
            return
        del data[key]


def _data_chars(data: dict) -> int:
    """统计完整数据条目及提示的紧凑 JSON 字符数。"""
    return len(json.dumps(list(data.values()), ensure_ascii=False, separators=(",", ":")))

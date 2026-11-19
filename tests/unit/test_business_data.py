"""验证业务缓存只接收真实读取、写后失效及包含提示的字符预算。"""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from ttwill.backend import RetailBackend, request_key
from ttwill.core.business_data import business_updates, limit_business_data
from ttwill.gateway import CallContext, SessionContext, ToolGateway
from ttwill.gateway.skills import SkillCatalog
from ttwill.tracing import Trace


@pytest.fixture
def gateway(backend: RetailBackend, tmp_path: Path, skill_catalog: SkillCatalog) -> ToolGateway:
    """创建记录真实业务查询和提交的测试网关。

    Args:
        backend: 含两位用户及订单、商品的独立业务库。
        tmp_path: 该用例的轨迹保存目录。
        skill_catalog: 项目使用的静态能力目录。

    Returns:
        绑定本项业务库和轨迹的网关，调用上下文由用例提供。
    """
    return ToolGateway(backend, Trace(tmp_path / "run", "chat"), skill_catalog)


def record(action: str, result: dict, *, kind: str = "call", **overrides: object) -> dict:
    """构造与网关格式一致的读取记录，供来源过滤和顺序边界测试使用。

    Args:
        action: 业务工具名称。
        result: 该次读取的真实结果形状。
        kind: 记录来源类别，默认是普通业务调用。
        **overrides: 覆盖状态、归属或会话等字段的测试边界值。

    Returns:
        当前用户和聊天的成功只读记录，应用指定的字段覆盖。
    """
    return {
        "kind": kind,
        "action": action,
        "user_id": "u1",
        "session_id": "chat",
        "read_write": "read",
        "status": "executed",
        "result": result,
        **overrides,
    }


def updates(records: list[dict], turn: int = 3) -> dict[str, dict]:
    """取得固定测试用户和当前聊天的业务增量。

    Args:
        records: 按发生顺序排列的网关记录。
        turn: 本批记录所属用户轮次。

    Returns:
        由公开业务缓存函数产生的独立数据映射。
    """
    return business_updates(records, user_id="u1", session_id="chat", turn=turn)


def data_chars(data: dict) -> int:
    """按上下文约定统计条目及提示的紧凑 JSON 字符数。"""
    return len(json.dumps(list(data.values()), ensure_ascii=False, separators=(",", ":")))


@pytest.mark.parametrize(
    ("action", "args", "key", "entity_type", "entity_id"),
    [
        ("get_user_details", {"user_id": "u1"}, "user:u1", "user", "u1"),
        ("get_order_details", {"order_id": "#W0000001"}, "order:#W0000001", "order", "#W0000001"),
        ("get_product_details", {"product_id": "p1"}, "product:p1", "product", "p1"),
        ("get_item_details", {"item_id": "a"}, "item:a", "item", "a"),
        ("list_all_product_types", {}, "catalog:all", "catalog", "all"),
        ("find_user_id_by_email", {"email": "u1@example.com"}, None, None, None),
        (
            "find_user_id_by_name_zip",
            {"first_name": "小明", "last_name": "王", "zip": "200000"},
            None,
            None,
            None,
        ),
    ],
)
def test_seven_read_tools_map_entities_and_exclude_identity(
    gateway: ToolGateway,
    action: str,
    args: dict,
    key: str | None,
    entity_type: str | None,
    entity_id: str | None,
) -> None:
    """七个非写工具各有明确处理，身份查询不能冒充完整账户资料。"""
    context = CallContext(SessionContext("chat", "u1"), "coordinator", "main")
    result = gateway.call(context, action, args)
    before = deepcopy(gateway.records)
    data = updates(gateway.records)
    if key is None:
        assert data == {}
    else:
        assert data == {
            key: {"type": entity_type, "id": entity_id, "read_turn": 3, "result": result}
        }
        data[key]["result"]["test_only"] = "缓存修改不会污染来源"
    assert gateway.records == before


def test_item_result_cannot_replace_product_and_newest_read_wins(gateway: ToolGateway) -> None:
    """规格缓存独立于完整商品，重复读取同一实体仅保留最后一份。"""
    context = CallContext(SessionContext("chat", "u1"), "coordinator", "main")
    product = gateway.call(context, "get_product_details", {"product_id": "p1"})
    item = gateway.call(context, "get_item_details", {"item_id": "a"})
    latest = {**deepcopy(product), "name": "最新商品名称"}
    records = [*gateway.records, record("get_product_details", latest)]
    data = updates(records)
    assert set(data) == {"product:p1", "item:a"}
    assert data["product:p1"]["result"] == latest
    assert data["item:a"]["result"] == item
    assert data["product:p1"]["result"]["variants"] == product["variants"]
    assert list(data) == ["item:a", "product:p1"]


def test_item_without_embedded_id_uses_validated_query_args(
    tmp_path: Path, seed_snapshot: dict, skill_catalog: SkillCatalog
) -> None:
    """真实后端返回未重复存储标识的规格时，缓存用查询参数建键并保留原结果。"""
    variant = seed_snapshot["products"]["p1"]["variants"]["a"]
    del variant["item_id"]
    backend = RetailBackend(tmp_path / "item-without-id.sqlite")
    backend.initialize(seed_snapshot)
    try:
        gateway = ToolGateway(backend, Trace(tmp_path / "item-run", "chat"), skill_catalog)
        context = CallContext(SessionContext("chat", "u1"), "coordinator", "main")
        product = gateway.call(context, "get_product_details", {"product_id": "p1"})
        result = gateway.call(context, "get_item_details", {"item_id": "a"})
        data = updates(gateway.records)
        assert result == variant
        assert "item_id" not in result
        assert data["item:a"] == {"type": "item", "id": "a", "read_turn": 3, "result": result}
        assert data["product:p1"]["result"] == product
        assert data["item:a"]["result"] is not result
        assert gateway.records[-1]["result"] == variant
    finally:
        backend.close()


@pytest.mark.parametrize(
    "override",
    [
        {"status": "failed"},
        {"status": "not_found"},
        {"user_id": "u2"},
        {"session_id": "old-chat"},
        {"read_write": "write"},
        {"kind": "presentation_read"},
        {"kind": "operation"},
        {"kind": "verification"},
        {"action": "load_skill"},
        {"action": "modify_pending_order_items", "status": "prepared"},
    ],
)
def test_failed_private_or_nonquery_records_are_excluded(override: dict) -> None:
    """失败、跨用户会话、Skill、展示读取和操作预览都不能进入业务区。"""
    source = record("get_order_details", {"order_id": "#W1", "user_id": "u1"})
    source.update(override)
    assert updates([source]) == {}


@pytest.mark.parametrize("action", ["get_user_details", "get_order_details"])
def test_entity_owner_must_match_even_when_record_context_matches(action: str) -> None:
    """实体归属与当前账户不一致时，不能仅凭记录外层的用户字段缓存。"""
    source = record(action, {"user_id": "u2", "order_id": "#W3"})
    assert updates([source]) == {}


@pytest.mark.parametrize(
    ("action", "args"),
    [
        ("cancel_pending_order", {"order_id": "#W0000001", "reason": "no longer needed"}),
        (
            "modify_pending_order_address",
            {
                "order_id": "#W0000001",
                "address1": "新街道",
                "address2": "",
                "city": "上海",
                "state": "上海",
                "country": "中国",
                "zip": "200001",
            },
        ),
        (
            "modify_pending_order_items",
            {
                "order_id": "#W0000001",
                "item_ids": ["a"],
                "new_item_ids": ["b"],
                "payment_method_id": "gift",
            },
        ),
        ("modify_pending_order_payment", {"order_id": "#W0000001", "payment_method_id": "gift"}),
        (
            "modify_user_address",
            {
                "user_id": "u1",
                "address1": "新街道",
                "address2": "",
                "city": "上海",
                "state": "上海",
                "country": "中国",
                "zip": "200001",
            },
        ),
        (
            "return_delivered_order_items",
            {"order_id": "#W0000002", "item_ids": ["a"], "payment_method_id": "card"},
        ),
        (
            "exchange_delivered_order_items",
            {
                "order_id": "#W0000002",
                "item_ids": ["a"],
                "new_item_ids": ["b"],
                "payment_method_id": "gift",
            },
        ),
    ],
)
def test_every_commit_invalidates_entities_until_actual_verification_read(
    gateway: ToolGateway, action: str, args: dict
) -> None:
    """七种提交均使用户资料及对应订单失效，核验读取只刷新实际读到的实体。"""
    context = CallContext(SessionContext("chat", "u1"), "coordinator", "main", "task")
    gateway.call(context, "get_user_details", {"user_id": "u1"})
    if "order_id" in args:
        gateway.call(context, "get_order_details", {"order_id": args["order_id"]})
    operation = gateway.backend.prepare("u1", action, args, request_key("task", 1, action))
    gateway.commit(context, operation["operation_id"], args)

    stale = updates(gateway.records)

    keys = {"user:u1"}
    if "order_id" in args:
        keys.add(f"order:{args['order_id']}")
    assert set(stale) == keys
    assert all(entry["needs_refresh"] is True and "result" not in entry for entry in stale.values())
    assert all(entry["reason"] == "已变更，需重新查询" for entry in stale.values())
    assert gateway.verify_operation(context, operation["operation_id"])["verified"] is True

    refreshed = updates(gateway.records)

    entity_type, entity_id = ("order", args["order_id"]) if "order_id" in args else ("user", "u1")
    key = f"{entity_type}:{entity_id}"
    snapshot = gateway.backend.export_snapshot()
    assert refreshed[key]["result"] == snapshot[f"{entity_type}s"][entity_id]
    assert "needs_refresh" not in refreshed[key]
    if entity_type == "order":
        assert refreshed["user:u1"]["needs_refresh"] is True
        assert "result" not in refreshed["user:u1"]


def test_input_order_prevents_earlier_read_from_overriding_later_commit() -> None:
    """同批中先读后提交必须失效，只有提交之后真实重读才能恢复可用值。"""
    read = record("get_order_details", {"order_id": "#W1", "user_id": "u1", "status": "pending"})
    commit = record(
        "cancel_pending_order",
        {"user_id": "u1", "args": {"order_id": "#W1"}},
        kind="commit",
        status="committed",
        read_write="write",
    )
    fresh = record(
        "get_order_details",
        {"order_id": "#W1", "user_id": "u1", "status": "cancelled"},
        kind="verification_read",
    )
    assert updates([read, fresh, commit])["order:#W1"]["needs_refresh"] is True
    assert updates([read, commit, fresh])["order:#W1"]["result"]["status"] == "cancelled"
    assert (
        updates([read, {**commit, "status": "failed"}])["order:#W1"]["result"]["status"]
        == "pending"
    )
    assert (
        updates([read, {**commit, "kind": "reject", "status": "rejected"}])["order:#W1"]["result"][
            "status"
        ]
        == "pending"
    )


def test_budget_drops_oldest_result_keeps_marker_and_recent_entity() -> None:
    """超预算先移出最旧结果，提示也计入预算，较新的真实数据保留。"""
    old = updates(
        [record("get_product_details", {"product_id": "old", "description": "旧数据" * 1000})],
        turn=1,
    )
    recent = updates([record("get_product_details", {"product_id": "new", "price": 10.25})], turn=2)
    data = {**recent, **old}
    before = deepcopy(data)

    limit_business_data(data, 300)

    assert data_chars(data) <= 300
    assert data["product:old"] == {
        "type": "product",
        "id": "old",
        "read_turn": 1,
        "needs_refresh": True,
        "reason": "已移出，需要时重新查询",
    }
    assert data["product:new"] == recent["product:new"]
    assert before["product:old"]["result"]["description"] == "旧数据" * 1000


def test_exact_budget_preserves_entries_and_same_turn_order_is_stable() -> None:
    """刚好容纳时不裁剪，同轮超限时按映射顺序先移出较早条目。"""
    data = updates(
        [
            record("get_product_details", {"product_id": "first", "name": "前" * 300}),
            record("get_product_details", {"product_id": "second", "name": "后" * 300}),
        ]
    )
    before = deepcopy(data)
    limit_business_data(data, data_chars(data))
    assert data == before
    limit_business_data(data, 500)
    assert "result" not in data["product:first"]
    assert data["product:second"] == before["product:second"]
    assert data_chars(data) <= 500


def test_budget_removes_oldest_markers_when_markers_alone_overflow() -> None:
    """移出提示本身也超限时继续移出最旧提示，不留下超预算的元数据。"""
    data = {
        f"product:{number}": {
            "type": "product",
            "id": str(number),
            "read_turn": number,
            "needs_refresh": True,
            "reason": "已移出，需要时重新查询",
        }
        for number in range(3)
    }
    newest = deepcopy(data["product:2"])
    budget = data_chars({"product:2": newest})
    limit_business_data(data, budget)
    assert data == {"product:2": newest}
    assert data_chars(data) == budget
    limit_business_data(data, 2)
    assert data == {} and data_chars(data) == 2


def test_evicted_result_is_restored_by_later_read() -> None:
    """移出后的同一实体可被下一轮真实读取覆盖，不保留旧提示。"""
    source = record("get_product_details", {"product_id": "p1", "name": "长商品" * 300})
    data = updates([source], turn=1)
    limit_business_data(data, 200)
    assert data["product:p1"]["needs_refresh"] is True
    data.update(
        updates([record("get_product_details", {"product_id": "p1", "name": "新商品"})], turn=2)
    )
    assert data["product:p1"] == {
        "type": "product",
        "id": "p1",
        "read_turn": 2,
        "result": {"product_id": "p1", "name": "新商品"},
    }


@pytest.mark.parametrize("budget", [True, 1, 0, -1, 1.5])
def test_budget_must_fit_at_least_empty_list(budget: object) -> None:
    """空列表至少占两个 JSON 字符，无法容纳时明确拒绝而非假称满足预算。"""
    with pytest.raises(ValueError, match="至少为 2"):
        limit_business_data({}, budget)

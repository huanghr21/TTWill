"""通过真实 SQLite 写入、重读与篡改验证七种业务操作的写后核验。"""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from sqlalchemy import select, update

from ttwill.backend import BusinessError, RetailBackend, request_key
from ttwill.backend.store import operations, orders, users
from ttwill.gateway import CallContext, SessionContext, ToolGateway
from ttwill.gateway.skills import SkillCatalog
from ttwill.tracing import Trace

ADDRESS = {
    "address1": "新地址",
    "address2": "二单元",
    "city": "北京",
    "state": "北京",
    "country": "中国",
    "zip": "100000",
}
WRITE_CASES = [
    ("cancel_pending_order", {"order_id": "#W0000001", "reason": "no longer needed"}),
    ("modify_pending_order_address", {"order_id": "#W0000001", **ADDRESS}),
    (
        "modify_pending_order_items",
        {
            "order_id": "#W0000001",
            "item_ids": ["a", "x"],
            "new_item_ids": ["b", "y"],
            "payment_method_id": "gift",
        },
    ),
    ("modify_pending_order_payment", {"order_id": "#W0000001", "payment_method_id": "gift"}),
    ("modify_user_address", {"user_id": "u1", **ADDRESS}),
    (
        "return_delivered_order_items",
        {"order_id": "#W0000002", "item_ids": ["a"], "payment_method_id": "card"},
    ),
    (
        "exchange_delivered_order_items",
        {
            "order_id": "#W0000002",
            "item_ids": ["a", "x"],
            "new_item_ids": ["b", "y"],
            "payment_method_id": "card",
        },
    ),
]


@pytest.fixture
def gateway(backend: RetailBackend, tmp_path: Path, skill_catalog: SkillCatalog) -> ToolGateway:
    """构造使用独立数据库与轨迹的核验入口。

    Args:
        backend: 本项测试独立的小型业务库。
        tmp_path: 本项测试的临时目录。
        skill_catalog: 覆盖全部写工具的静态 Skill 目录。

    Returns:
        连接真实业务库和测试轨迹的网关。
    """
    return ToolGateway(backend, Trace(tmp_path / "run", "chat"), skill_catalog)


@pytest.fixture
def context(gateway: ToolGateway) -> CallContext:
    """经邮箱查询绑定用户，并返回代码核验使用的协调者上下文。

    Args:
        gateway: 执行真实身份查询的网关。

    Returns:
        绑定用户 u1 的协调者上下文。
    """
    result = CallContext(SessionContext("chat"), "coordinator", "coordinator", "task-1")
    gateway.call(result, "find_user_id_by_email", {"email": "u1@example.com"})
    return result


def commit_operation(gateway: ToolGateway, context: CallContext, action: str, args: dict) -> dict:
    """为测试准备一项操作，并从代码入口提交。

    Args:
        gateway: 连接独立业务库的网关。
        context: 已验证为 u1 的代码调用上下文。
        action: 本次测试的写动作名称。
        args: 本次测试使用的有效写操作参数。

    Returns:
        状态为 committed 的操作记录。
    """
    operation = gateway.backend.prepare("u1", action, args, request_key("task-1", 1, action))
    return gateway.commit(context, operation["operation_id"], args)


def change_entity(backend: RetailBackend, args: dict, path: tuple, value: object) -> None:
    """直接篡改一项业务字段，保留已提交操作记录不变。

    Args:
        backend: 本项测试的业务后端。
        args: 原写操作参数，含用户或订单 ID。
        path: 从实体根开始的键或数组索引序列。
        value: 写入叶子字段的新值。
    """
    table = orders if "order_id" in args else users
    entity_id = args.get("order_id", args.get("user_id"))
    with backend.engine.begin() as connection:
        entity = connection.execute(
            select(table.c.data).where(table.c.id == entity_id)
        ).scalar_one()
        parent = entity
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = value
        connection.execute(update(table).where(table.c.id == entity_id).values(data=entity))


@pytest.mark.parametrize(("action", "args"), WRITE_CASES)
def test_all_seven_committed_actions_verify_real_business_state(
    gateway: ToolGateway, context: CallContext, action: str, args: dict
) -> None:
    """七种写动作均在重新读取业务实体并核对参数后通过。"""
    operation = commit_operation(gateway, context, action, args)
    result = gateway.verify_operation(context, operation["operation_id"])
    assert result["verified"] is True
    assert result["action"] == action
    assert result["args"] == args
    assert result["expected"] == result["actual"]
    assert result["checked_fields"] == list(result["expected"])
    assert result["reason"] is None
    assert gateway.records[-2]["kind"] == "verification_read"
    assert gateway.records[-2]["action"] == (
        "get_user_details" if action == "modify_user_address" else "get_order_details"
    )
    assert gateway.records[-1]["result"] == result


@pytest.mark.parametrize(
    ("case", "path", "value"),
    [
        (0, ("status",), "pending"),
        (0, ("payment_history", 1, "amount"), 1.0),
        (1, ("address", "city"), "错误城市"),
        (2, ("items", 0, "item_id"), "a"),
        (2, ("items", 0, "price"), 999.0),
        (2, ("items", 0, "options", "颜色"), "绿"),
        (2, ("payment_history", 1, "amount"), 999.0),
        (3, ("payment_history", 1, "payment_method_id"), "card"),
        (3, ("payment_history", 2, "amount"), 1.0),
        (4, ("address", "address1"), "错误地址"),
        (5, ("return_items",), ["x"]),
        (5, ("return_payment_method_id",), "gift"),
        (6, ("exchange_new_items",), ["y"]),
        (6, ("exchange_price_difference",), 999.0),
    ],
)
def test_committed_record_does_not_hide_changed_business_fields(
    gateway: ToolGateway, context: CallContext, case: int, path: tuple, value: object
) -> None:
    """即使操作记录仍是 committed，实际字段不符也不能判为核验通过。"""
    action, args = WRITE_CASES[case]
    operation = commit_operation(gateway, context, action, args)
    change_entity(gateway.backend, args, path, value)
    result = gateway.verify_operation(context, operation["operation_id"])
    assert result["verified"] is False
    assert path[0] in result["reason"]
    assert (
        gateway.backend.get_operation("u1", operation_id=operation["operation_id"])["status"]
        == "committed"
    )
    events = [json.loads(line) for line in gateway.trace.path.read_text().splitlines()]
    guards = [
        event
        for event in events
        if event["name"] == "verification_matches" and event["event"] == "end"
    ]
    assert guards[-1]["status"] == "blocked"
    assert guards[-1]["attrs"]["allowed"] is False


def test_user_address_verification_ignores_unrelated_orders(
    gateway: ToolGateway, context: CallContext
) -> None:
    """用户新增订单不会使已正确更新的默认地址被误判失败。"""
    action, args = WRITE_CASES[4]
    operation = commit_operation(gateway, context, action, args)
    change_entity(
        gateway.backend, args, ("orders",), ["#W0000001", "#W0000002", "#W0000004", "#new"]
    )
    assert gateway.verify_operation(context, operation["operation_id"])["verified"] is True


def test_committed_result_must_agree_with_confirmed_parameters(
    gateway: ToolGateway, context: CallContext
) -> None:
    """重读结果与提交记录相同时，仍须符合确认参数中的地址。"""
    action, args = WRITE_CASES[1]
    operation = commit_operation(gateway, context, action, args)
    changed_args = {**args, "city": "另一城市"}
    with gateway.backend.engine.begin() as connection:
        connection.execute(
            update(operations)
            .where(operations.c.operation_id == operation["operation_id"])
            .values(args=changed_args)
        )
    result = gateway.verify_operation(context, operation["operation_id"])
    assert result["verified"] is False
    assert "确认参数" in result["reason"]


def test_prepared_operation_cannot_verify(gateway: ToolGateway, context: CallContext) -> None:
    """准备成功不能代替提交与实际业务结果核验。"""
    action, args = WRITE_CASES[0]
    operation = gateway.backend.prepare("u1", action, args, "prepare-only")
    with pytest.raises(BusinessError) as error:
        gateway.verify_operation(context, operation["operation_id"])
    assert error.value.code == "operation_not_committed"
    assert not any(record["kind"] == "verification_read" for record in gateway.records)


def test_verification_requires_operation_owner(gateway: ToolGateway, context: CallContext) -> None:
    """核验入口继续执行操作归属检查，不允许跨用户读取。"""
    action, args = WRITE_CASES[0]
    operation = commit_operation(gateway, context, action, args)
    other = CallContext(SessionContext("other", "u2"), "coordinator", "other")
    with pytest.raises(BusinessError) as error:
        gateway.verify_operation(other, operation["operation_id"])
    assert error.value.code == "forbidden"


def test_unverified_session_cannot_verify(gateway: ToolGateway, context: CallContext) -> None:
    """未验证身份时不允许访问已提交操作或业务实体。"""
    action, args = WRITE_CASES[0]
    operation = commit_operation(gateway, context, action, args)
    guest = CallContext(SessionContext("guest"), "coordinator", "guest")
    with pytest.raises(BusinessError) as error:
        gateway.verify_operation(guest, operation["operation_id"])
    assert error.value.code == "identity_required"


def test_unknown_write_action_has_no_implicit_verification(
    gateway: ToolGateway, context: CallContext
) -> None:
    """尚无核验规则的写动作不能静默通过。"""
    action, args = WRITE_CASES[0]
    operation = commit_operation(gateway, context, action, args)
    with gateway.backend.engine.begin() as connection:
        connection.execute(
            update(operations)
            .where(operations.c.operation_id == operation["operation_id"])
            .values(action="future_action")
        )
    with pytest.raises(BusinessError) as error:
        gateway.verify_operation(context, operation["operation_id"])
    assert error.value.code == "unknown_action"


def test_verification_is_not_a_model_tool(gateway: ToolGateway, context: CallContext) -> None:
    """所有能力的模型工具均不暴露核验入口，伪造工具调用也被拦截。"""
    for capability in ("coordinator", "order_service", "after_sales"):
        caller = CallContext(context.session, capability, capability)
        assert "verify_operation" not in {
            tool["function"]["name"] for tool in gateway.schemas(caller)
        }
        with pytest.raises(BusinessError) as error:
            gateway.call(caller, "verify_operation", {"operation_id": "forged"})
        assert error.value.code == "unknown_tool"


def test_verification_trace_and_records_preserve_actual_read(
    gateway: ToolGateway, context: CallContext
) -> None:
    """核验轨迹包含真实业务查询，返回字典的后续变化不污染记录。"""
    action, args = WRITE_CASES[2]
    operation = commit_operation(gateway, context, action, args)
    result = gateway.verify_operation(context, operation["operation_id"])
    expected_record = deepcopy(result)
    result["actual"]["items"][0]["price"] = -1
    assert gateway.records[-1]["result"] == expected_record
    events = [json.loads(line) for line in gateway.trace.path.read_text().splitlines()]
    verification = next(
        event for event in events if event["kind"] == "verification" and event["event"] == "end"
    )
    query = next(
        event
        for event in events
        if event["kind"] == "tool"
        and event["attrs"]["kind"] == "verification_read"
        and event["event"] == "end"
    )
    assert query["parent_id"] == verification["span_id"]
    assert query["attrs"]["result"] == operation["result"]
    assert verification["attrs"]["result"] == expected_record

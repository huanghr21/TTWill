"""通过小型 SQLite 业务库验证工具网关的身份、权限和调用事实。"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ttwill.backend import BusinessError, RetailBackend
from ttwill.backend.arguments import ARGUMENT_MODELS, validate_arguments
from ttwill.gateway import CallContext, SessionContext, ToolGateway
from ttwill.gateway.skills import SkillCatalog
from ttwill.tracing import Trace


@pytest.fixture
def gateway(backend: RetailBackend, tmp_path: Path, skill_catalog: SkillCatalog) -> ToolGateway:
    """将真实业务库接到独立的会话轨迹。

    Args:
        backend: 本项测试的真实业务后端。
        tmp_path: 保存运行轨迹的临时目录。
        skill_catalog: 覆盖全部业务写工具的测试 Skill 目录。

    Returns:
        使用独立测试会话轨迹的工具网关。
    """
    return ToolGateway(backend, Trace(tmp_path / "run", "chat"), skill_catalog)


@pytest.fixture
def pending_account(seed_snapshot: dict) -> tuple[dict, dict]:
    """挑选一个待发货订单及其所属用户。

    Args:
        seed_snapshot: 含待发货订单的业务快照。

    Returns:
        按顺序包含以下对象的二元组：
            user: 订单所属的用户数据。
            order: 待发货订单数据。
    """
    order = next(
        order for order in seed_snapshot["orders"].values() if order["status"] == "pending"
    )
    return seed_snapshot["users"][order["user_id"]], order


def verified_context(gateway: ToolGateway, user: dict) -> CallContext:
    """通过邮箱完成身份验证，再构造订单服务任务上下文。

    Args:
        gateway: 执行身份查询并记录调用事实的网关。
        user: 含 email 和 user_id 的测试用户数据。

    Returns:
        已绑定用户身份、订单服务能力、实例 ID 和任务 ID 的调用上下文。
    """
    session = SessionContext("chat")
    coordinator = CallContext(session, "coordinator", "coordinator")
    assert (
        gateway.call(coordinator, "find_user_id_by_email", {"email": user["email"]})
        == user["user_id"]
    )
    return CallContext(session, "order_service", "instance-1", "task-1")


def cancellation(order: dict) -> dict:
    """构造一次取消待发货订单的参数。

    Args:
        order: 含 order_id 的目标订单数据。

    Returns:
        包含订单 ID 和固定取消原因 no longer needed 的参数字典。
    """
    return {"order_id": order["order_id"], "reason": "no longer needed"}


def load_for(gateway: ToolGateway, context: CallContext, tool: str) -> None:
    """通过网关为当前实例加载某项写工具的政策。

    Args:
        gateway: 保存实例加载记录的工具网关。
        context: 具有对应业务能力且已验证身份的上下文。
        tool: 即将调用的写工具名称。
    """
    gateway.call(context, "load_skill", {"name": gateway.skills.required_skill(tool)})


def test_schemas_follow_identity_and_capability(
    gateway: ToolGateway, pending_account: tuple
) -> None:
    """模型可见工具随身份与能力变化，提交和拒绝只留在代码接口。"""
    guest = CallContext(SessionContext("chat"), "coordinator", "coordinator")
    assert {tool["function"]["name"] for tool in gateway.schemas(guest)} == {
        "find_user_id_by_email",
        "find_user_id_by_name_zip",
    }
    assert gateway.schemas(replace(guest, capability="order_service")) == []
    context = verified_context(gateway, pending_account[0])
    names = {tool["function"]["name"] for tool in gateway.schemas(context)}
    assert "cancel_pending_order" in names
    assert "return_delivered_order_items" not in names
    assert "find_user_id_by_email" not in names
    assert "commit" not in names and "reject" not in names
    assert "get_operation" not in names
    assert len(ARGUMENT_MODELS) == 14
    for tool in gateway.schemas(context):
        assert tool["type"] == "function"
        assert tool["function"]["description"]
        assert tool["function"]["parameters"]["additionalProperties"] is False


def test_database_path_identifies_business_store_without_allowing_rebind(
    gateway: ToolGateway, tmp_path: Path
) -> None:
    """运行时取得真实业务库绝对路径，属性赋值不能切换网关后端。"""
    assert gateway.database_path == (tmp_path / "business.sqlite").resolve()
    assert gateway.database_path.is_file()
    with pytest.raises(AttributeError):
        gateway.database_path = tmp_path / "other.sqlite"


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("get_order_details", {"order_id": 1}),
        ("find_user_id_by_email", {"email": "x@example.com", "user_id": "forged"}),
        ("list_all_product_types", {"user_id": "forged"}),
        ("cancel_pending_order", {"order_id": "#1", "reason": "other"}),
        (
            "modify_pending_order_items",
            {"order_id": "#1", "item_ids": ("1",), "new_item_ids": ["2"], "payment_method_id": "p"},
        ),
    ],
)
def test_argument_schemas_preserve_types_and_fields(name: str, args: dict) -> None:
    """参数结构保留字段类型并拒绝额外字段与非法枚举。"""
    with pytest.raises(ValueError):
        validate_arguments(name, args)


def test_unverified_reads_are_blocked_and_traced(
    gateway: ToolGateway, pending_account: tuple
) -> None:
    """读取业务数据前必须完成身份验证，失败原因写入护栏轨迹。"""
    context = CallContext(SessionContext("chat"), "order_service", "instance-1")
    with pytest.raises(BusinessError) as error:
        gateway.call(context, "get_order_details", {"order_id": pending_account[1]["order_id"]})
    assert error.value.code == "identity_required"
    assert gateway.records[-1]["status"] == "failed"
    events = [json.loads(line) for line in gateway.trace.path.read_text().splitlines()]
    blocked = [
        event for event in events if event["kind"] == "guard" and event["status"] == "blocked"
    ]
    assert blocked[0]["attrs"]["reason"] == "请先验证用户身份"


def test_identity_is_bound_by_both_lookup_methods_and_cannot_switch(
    gateway: ToolGateway, seed_snapshot: dict, pending_account: tuple
) -> None:
    """两种查找都绑定代码身份，同一会话查到另一用户时保持原身份。"""
    user, _ = pending_account
    context = CallContext(SessionContext("chat"), "coordinator", "coordinator")
    with pytest.raises(BusinessError):
        gateway.call(context, "find_user_id_by_email", {"email": "missing@example.invalid"})
    assert context.session.user_id is None
    assert (
        gateway.call(
            context,
            "find_user_id_by_name_zip",
            {**user["name"], "zip": user["address"]["zip"]},
        )
        == user["user_id"]
    )
    assert (
        gateway.call(context, "find_user_id_by_email", {"email": user["email"]}) == user["user_id"]
    )
    other = next(other for other in seed_snapshot["users"].values() if other != user)
    with pytest.raises(BusinessError) as error:
        gateway.call(context, "find_user_id_by_email", {"email": other["email"]})
    assert error.value.code == "identity_locked"
    assert context.session.user_id == user["user_id"]


@pytest.mark.parametrize(
    ("capability", "name", "code"),
    [
        ("coordinator", "cancel_pending_order", "capability_denied"),
        ("order_service", "return_delivered_order_items", "capability_denied"),
        ("after_sales", "modify_user_address", "capability_denied"),
        ("order_service", "commit", "unknown_tool"),
    ],
)
def test_tool_allowlists_are_enforced(
    gateway: ToolGateway, pending_account: tuple, capability: str, name: str, code: str
) -> None:
    """调用入口校验能力白名单，模型不能直接调用提交接口。"""
    context = replace(verified_context(gateway, pending_account[0]), capability=capability)
    with pytest.raises(BusinessError) as error:
        gateway.call(context, name, {})
    assert error.value.code == code


def test_foreign_user_data_and_invalid_arguments_are_rejected(
    gateway: ToolGateway, seed_snapshot: dict, pending_account: tuple
) -> None:
    """业务后端复核数据归属，网关记录参数错误。"""
    user, _ = pending_account
    context = verified_context(gateway, user)
    other_id = next(user_id for user_id in seed_snapshot["users"] if user_id != user["user_id"])
    with pytest.raises(BusinessError):
        gateway.call(context, "get_user_details", {"user_id": other_id})
    with pytest.raises(BusinessError) as error:
        gateway.call(context, "get_order_details", {"order_id": 123})
    assert error.value.code == "invalid_arguments"
    assert gateway.records[-1]["status"] == "failed"


def test_prepare_does_not_write_business_data_and_retries_share_operation(
    gateway: ToolGateway, backend: RetailBackend, pending_account: tuple
) -> None:
    """同一任务跨实例重试复用准备记录，业务实体在提交前保持原样。"""
    user, order = pending_account
    context = verified_context(gateway, user)
    load_for(gateway, context, "cancel_pending_order")
    before = backend.export_snapshot()
    operation = gateway.call(context, "cancel_pending_order", cancellation(order))
    assert operation["status"] == "prepared"
    assert operation["preview"]["status"] == "cancelled"
    assert backend.export_snapshot() == before
    original_record = gateway.records[-1]
    retry_context = replace(context, instance_id="instance-2")
    load_for(gateway, retry_context, "cancel_pending_order")
    retry = gateway.call(retry_context, "cancel_pending_order", cancellation(order))
    assert retry["operation_id"] == operation["operation_id"]
    assert gateway.records[-1]["request_key"] == original_record["request_key"]
    operation["preview"]["status"] = "modified-by-caller"
    assert original_record["result"]["preview"]["status"] == "cancelled"


def test_new_revision_after_rejection_creates_new_request(
    gateway: ToolGateway, pending_account: tuple
) -> None:
    """任务改口先拒绝旧操作，再用新修订号准备新的参数。"""
    user, order = pending_account
    context = verified_context(gateway, user)
    load_for(gateway, context, "cancel_pending_order")
    operation = gateway.call(context, "cancel_pending_order", cancellation(order))
    changed = {**cancellation(order), "reason": "ordered by mistake"}
    with pytest.raises(BusinessError) as error:
        gateway.call(context, "cancel_pending_order", changed)
    assert error.value.code == "request_conflict"
    rejected = gateway.reject(context, operation["operation_id"], "用户修改取消原因")
    assert rejected["status"] == "rejected"
    replacement = gateway.call(replace(context, revision=2), "cancel_pending_order", changed)
    assert replacement["operation_id"] != operation["operation_id"]
    assert replacement["request_key"] != operation["request_key"]


def test_write_preparation_requires_task_id(gateway: ToolGateway, pending_account: tuple) -> None:
    """写请求必须由代码关联到具体任务，才能跨实例去重。"""
    user, order = pending_account
    context = replace(verified_context(gateway, user), task_id=None)
    load_for(gateway, context, "cancel_pending_order")
    with pytest.raises(BusinessError) as error:
        gateway.call(context, "cancel_pending_order", cancellation(order))
    assert error.value.code == "task_required"


@pytest.mark.parametrize(
    ("name", "status", "empty_field"),
    [
        ("return_delivered_order_items", "delivered", "item_ids"),
        ("exchange_delivered_order_items", "delivered", "item_ids"),
        ("exchange_delivered_order_items", "delivered", "new_item_ids"),
        ("modify_pending_order_items", "pending", "item_ids"),
        ("modify_pending_order_items", "pending", "new_item_ids"),
    ],
)
def test_empty_item_lists_leave_order_available_for_valid_request(
    gateway: ToolGateway,
    seed_snapshot: dict,
    name: str,
    status: str,
    empty_field: str,
) -> None:
    """空清单在准备前报参数错误，随后有效请求仍能占用并办理该订单。"""
    order = next(order for order in seed_snapshot["orders"].values() if order["status"] == status)
    context = verified_context(gateway, seed_snapshot["users"][order["user_id"]])
    if status == "delivered":
        context = replace(context, capability="after_sales")
    load_for(gateway, context, name)
    args = {
        "order_id": order["order_id"],
        "item_ids": ["a"],
        "payment_method_id": "card",
    }
    if name != "return_delivered_order_items":
        args["new_item_ids"] = ["b"]
    with pytest.raises(BusinessError) as error:
        gateway.call(context, name, {**args, empty_field: []})
    assert error.value.code == "invalid_arguments"
    assert gateway.call(context, name, args)["status"] == "prepared"


def test_code_commit_keeps_backend_results_and_checks_displayed_arguments(
    gateway: ToolGateway, backend: RetailBackend, pending_account: tuple
) -> None:
    """代码提交使用准备时的参数，真实结果写入调用记录并可重复对账。"""
    user, order = pending_account
    context = verified_context(gateway, user)
    load_for(gateway, context, "cancel_pending_order")
    operation = gateway.call(context, "cancel_pending_order", cancellation(order))
    coordinator = replace(context, capability="coordinator", instance_id="coordinator")
    with pytest.raises(BusinessError) as error:
        gateway.commit(coordinator, operation["operation_id"], {})
    assert error.value.code == "parameter_mismatch"
    committed = gateway.commit(coordinator, operation["operation_id"], cancellation(order))
    assert committed["status"] == "committed"
    assert backend.export_snapshot()["orders"][order["order_id"]]["status"] == "cancelled"
    assert gateway.records[-1]["kind"] == "commit"
    assert gateway.records[-1]["action"] == "cancel_pending_order"
    assert gateway.records[-1]["result"] == committed
    assert gateway.commit(coordinator, operation["operation_id"], cancellation(order)) == committed


def test_code_resolution_requires_identity_and_operation_ownership(
    gateway: ToolGateway, seed_snapshot: dict, pending_account: tuple
) -> None:
    """代码入口同样要求已验证身份，并由后端检查操作归属。"""
    user, order = pending_account
    context = verified_context(gateway, user)
    load_for(gateway, context, "cancel_pending_order")
    operation = gateway.call(context, "cancel_pending_order", cancellation(order))
    guest = CallContext(SessionContext("guest"), "coordinator", "guest")
    with pytest.raises(BusinessError) as error:
        gateway.commit(guest, operation["operation_id"], cancellation(order))
    assert error.value.code == "identity_required"
    other = next(other for other in seed_snapshot["users"].values() if other != user)
    other_context = verified_context(gateway, other)
    with pytest.raises(BusinessError):
        gateway.reject(other_context, operation["operation_id"], "用户拒绝")
    assert gateway.records[-1]["status"] == "failed"


def test_code_operation_refresh_records_current_expiry(
    gateway: ToolGateway,
    backend: RetailBackend,
    pending_account: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """对账会读取后端当前过期状态，保留准备历史并记录实际查询结果。"""
    user, order = pending_account
    context = verified_context(gateway, user)
    load_for(gateway, context, "cancel_pending_order")
    operation = gateway.call(context, "cancel_pending_order", cancellation(order))
    prepared = gateway.records[-1]
    monkeypatch.setattr(backend, "clock", lambda: operation["expires_at"])
    current = gateway.get_operation(context, operation["operation_id"])
    assert current["status"] == "expired"
    assert prepared["status"] == "prepared"
    record = gateway.records[-1]
    assert record["kind"] == "operation"
    assert record["read_write"] == "read"
    assert record["action"] == "cancel_pending_order"
    assert record["status"] == "expired"
    assert record["operation_id"] == operation["operation_id"]
    assert record["request_key"] == operation["request_key"]
    assert record["result"] == current
    assert (
        backend.query(user["user_id"], "get_order_details", {"order_id": order["order_id"]})[
            "status"
        ]
        == "pending"
    )


def test_code_operation_refresh_checks_identity_and_ownership(
    gateway: ToolGateway, pending_account: tuple, seed_snapshot: dict
) -> None:
    """操作对账入口要求身份验证，并拒绝读取其他用户的操作记录。"""
    user, order = pending_account
    context = verified_context(gateway, user)
    load_for(gateway, context, "cancel_pending_order")
    operation = gateway.call(context, "cancel_pending_order", cancellation(order))
    guest = CallContext(SessionContext("guest"), "coordinator", "coordinator")
    with pytest.raises(BusinessError) as error:
        gateway.get_operation(guest, operation["operation_id"])
    assert error.value.code == "identity_required"
    other = next(other for other in seed_snapshot["users"].values() if other != user)
    other_context = verified_context(gateway, other)
    with pytest.raises(BusinessError) as denied:
        gateway.get_operation(other_context, operation["operation_id"])
    assert denied.value.code == "forbidden"
    assert gateway.records[-1]["kind"] == "operation"
    assert gateway.records[-1]["status"] == "failed"
    assert "result" not in gateway.records[-1]

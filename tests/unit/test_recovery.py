"""用真实业务库验证跨会话操作找回、去重、清理与重新核验。"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import func, select, update

from ttwill.backend import RetailBackend, request_key
from ttwill.backend.store import operations, orders
from ttwill.core.coordinator import ConversationState
from ttwill.core.recovery import reconcile_restored_tasks
from ttwill.core.tasks import Task
from ttwill.gateway import BusinessError, CallContext, SessionContext, ToolGateway
from ttwill.gateway.skills import SkillCatalog
from ttwill.tracing import Trace

ACTION = "cancel_pending_order"
ARGS = {"order_id": "#W0000001", "reason": "no longer needed"}


@pytest.fixture
def gateway(backend: RetailBackend, tmp_path: Path, skill_catalog: SkillCatalog) -> ToolGateway:
    """创建恢复测试使用的真实业务网关。

    Args:
        backend: 当前测试独立的 SQLite 后端。
        tmp_path: 保存恢复会话轨迹的临时目录。
        skill_catalog: 覆盖业务写工具的测试 Skill 目录。

    Returns:
        连接真实业务库、尚未绑定身份的网关。
    """
    return ToolGateway(backend, Trace(tmp_path / "run", "new-chat"), skill_catalog)


@pytest.fixture
def state(gateway: ToolGateway) -> ConversationState:
    """构造已重新验证 u1 身份、含一次未完成取消任务的恢复状态。

    Args:
        gateway: 执行真实邮箱身份查询的网关。

    Returns:
        含稳定任务 ID、修订号以及旧聊天展示信息的对话状态。
    """
    result = ConversationState(SessionContext("new-chat"))
    gateway.call(
        CallContext(result.session, "coordinator", "new-instance"),
        "find_user_id_by_email",
        {"email": "u1@example.com"},
    )
    result.tasks.tasks["task-1"] = Task(
        "task-1",
        "order_service",
        ACTION,
        "取消订单 #W0000001",
        order_id="#W0000001",
        status="running",
    )
    result.last_display = [{"session_id": "old-chat", "content": "请确认取消订单"}]
    result.last_reply = "这是旧聊天的取消说明。\n\n请确认取消订单"
    return result


def recover(state: ConversationState, gateway: ToolGateway) -> None:
    """使用新实例上下文执行恢复对账。

    Args:
        state: 含当前会话身份与检查点任务的恢复状态。
        gateway: 连接原业务库的网关。
    """
    reconcile_restored_tasks(state, gateway, CallContext(state.session, "coordinator", "restore"))


def prepare(gateway: ToolGateway, state: ConversationState, **args: object) -> dict:
    """用恢复前的相同任务请求准备业务操作。

    Args:
        gateway: 连接当前测试业务库的网关。
        state: 提供稳定任务 ID 和修订号的检查点状态。
        **args: 覆盖默认取消参数的字段。

    Returns:
        后端真实生成的待提交操作记录。
    """
    task = state.tasks.tasks["task-1"]
    return gateway.backend.prepare(
        "u1", task.action, {**ARGS, **args}, request_key(task.id, task.revision, task.action)
    )


@pytest.mark.parametrize("status", ["pending", "running"])
def test_prepare_without_checkpoint_reference_is_found_and_not_authorized(
    gateway: ToolGateway, state: ConversationState, status: str
) -> None:
    """准备成功但操作引用未保存时，恢复找回操作并要求重新展示。"""
    before = gateway.backend.export_snapshot()
    operation = prepare(gateway, state)
    state.tasks.tasks["task-1"].status = status
    recover(state, gateway)
    task = state.tasks.tasks["task-1"]
    assert task.operation_ids == [operation["operation_id"]]
    assert task.status == "pending_confirm"
    assert task.revision == 1
    assert state.last_display == []
    assert state.last_reply == ""
    assert task.actions[-1]["result"] == operation
    assert gateway.backend.export_snapshot() == before
    assert not any(record["kind"] == "commit" for record in gateway.records)


def test_same_request_across_sessions_and_instances_does_not_duplicate_operation(
    gateway: ToolGateway, state: ConversationState, tmp_path: Path
) -> None:
    """新的会话和实例使用相同请求标识，恢复和重放均只保留原操作。"""
    operation = prepare(gateway, state)
    other_gateway = ToolGateway(gateway.backend, Trace(tmp_path / "other", "later"), gateway.skills)
    state.session = SessionContext("later")
    context = CallContext(state.session, "coordinator", "third-instance", "task-1")
    other_gateway.call(context, "find_user_id_by_email", {"email": "u1@example.com"})
    recover(state, other_gateway)
    actions = deepcopy(state.tasks.tasks["task-1"].actions)
    recover(state, other_gateway)
    assert state.tasks.tasks["task-1"].actions == actions
    assert other_gateway.find_operation(context, ACTION) == operation
    assert prepare(other_gateway, state)["operation_id"] == operation["operation_id"]
    with gateway.backend.engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(operations)).scalar_one() == 1


@pytest.mark.parametrize("with_reference", [False, True])
def test_committed_before_checkpoint_is_verified_without_second_refund(
    gateway: ToolGateway, state: ConversationState, with_reference: bool
) -> None:
    """提交成功后中断时按业务记录核验完成，不再提交或追加退款。"""
    operation = prepare(gateway, state)
    gateway.backend.commit("u1", operation["operation_id"], ARGS)
    task = state.tasks.tasks["task-1"]
    if with_reference:
        task.operation_ids = [operation["operation_id"]]
        task.status = "pending_confirm"
    before = gateway.backend.export_snapshot()
    recover(state, gateway)
    recover(state, gateway)
    assert task.status == "done"
    assert task.actions[-1]["status"] == "committed"
    assert task.facts[-1]["verification"]["verified"] is True
    assert len(task.facts) == 1
    assert gateway.backend.export_snapshot() == before
    history = before["orders"]["#W0000001"]["payment_history"]
    assert [item["transaction_type"] for item in history] == ["payment", "refund"]
    assert not any(record["kind"] == "commit" for record in gateway.records)


def test_committed_record_requires_current_business_verification(
    gateway: ToolGateway, state: ConversationState
) -> None:
    """操作已提交但当前实体不匹配时保留提交事实并失败。"""
    operation = prepare(gateway, state)
    gateway.backend.commit("u1", operation["operation_id"], ARGS)
    with gateway.backend.engine.begin() as connection:
        entity = connection.execute(
            select(orders.c.data).where(orders.c.id == "#W0000001")
        ).scalar_one()
        entity["status"] = "pending"
        connection.execute(update(orders).where(orders.c.id == "#W0000001").values(data=entity))
    recover(state, gateway)
    task = state.tasks.tasks["task-1"]
    assert task.status == "failed"
    assert task.operation_ids == [operation["operation_id"]]
    assert task.facts[-1]["verification"]["verified"] is False
    assert task.reason


@pytest.mark.parametrize("status", ["cancelled", "failed"])
def test_terminal_task_cleans_unreferenced_prepare_and_releases_order(
    gateway: ToolGateway, state: ConversationState, status: str
) -> None:
    """已终止任务仍找回未保存引用的准备项，清理后释放同单占用。"""
    operation = prepare(gateway, state)
    task = state.tasks.tasks["task-1"]
    task.status, task.reason = status, "不再办理"
    recover(state, gateway)
    assert task.status == status
    assert task.reason == "不再办理"
    assert task.actions[-1]["status"] == "rejected"
    assert (
        gateway.backend.get_operation("u1", operation_id=operation["operation_id"])["status"]
        == "rejected"
    )
    replacement = gateway.backend.prepare("u1", ACTION, ARGS, request_key("another", 1, ACTION))
    assert replacement["status"] == "prepared"


@pytest.mark.parametrize("condition", [None, "换货失败时取消"])
def test_invalid_operation_advances_revision_before_new_prepare(
    gateway: ToolGateway, state: ConversationState, condition: str | None
) -> None:
    """失效请求递增修订号，新准备不会取回旧的已拒绝记录。"""
    operation = prepare(gateway, state)
    task = state.tasks.tasks["task-1"]
    task.condition = condition
    gateway.backend.reject("u1", operation["operation_id"], "旧准备不再有效")
    recover(state, gateway)
    assert task.status == ("conditional" if condition else "pending")
    assert task.revision == 2
    assert task.operation_ids == []
    assert task.actions[-1]["status"] == "rejected"
    replacement = prepare(gateway, state)
    assert replacement["operation_id"] != operation["operation_id"]
    assert replacement["request_key"] != operation["request_key"]


def test_expired_operation_is_not_restored_as_confirmable(
    gateway: ToolGateway, state: ConversationState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """后端 TTL 到期后恢复按失效处理，不能沿用旧操作接受确认。"""
    operation = prepare(gateway, state)
    monkeypatch.setattr(gateway.backend, "clock", lambda: operation["expires_at"] + 1)
    recover(state, gateway)
    task = state.tasks.tasks["task-1"]
    assert task.status == "pending"
    assert task.revision == 2
    assert task.actions[-1]["status"] == "expired"
    assert task.operation_ids == []


@pytest.mark.parametrize("action", [ACTION, "get_order_details"])
@pytest.mark.parametrize("condition", [None, "前一任务无法办理"])
def test_running_without_operation_returns_to_dispatchable_state(
    gateway: ToolGateway, state: ConversationState, action: str, condition: str | None
) -> None:
    """运行中断而没有写入记录的任务保留原修订，恢复为可重新分派。"""
    task = state.tasks.tasks["task-1"]
    task.action, task.condition = action, condition
    recover(state, gateway)
    assert task.status == ("conditional" if condition else "pending")
    assert task.revision == 1
    assert task.operation_ids == []


def test_done_task_is_not_repeated_or_reverified(
    gateway: ToolGateway, state: ConversationState
) -> None:
    """已由核验完成的任务不因普通恢复重新执行或解释为新任务。"""
    task = state.tasks.tasks["task-1"]
    task.status = "done"
    before = deepcopy(task.to_dict())
    record_count = len(gateway.records)
    recover(state, gateway)
    assert task.to_dict() == before
    assert len(gateway.records) == record_count
    assert state.last_display == []
    assert state.last_reply == ""


def test_unverified_identity_cannot_recover(gateway: ToolGateway, state: ConversationState) -> None:
    """即使检查点保留任务，未重验身份也不能查找操作。"""
    prepare(gateway, state)
    state.session.user_id = None
    before = state.tasks.snapshot()
    previous_reply = state.last_reply
    with pytest.raises(BusinessError, match="重新验证") as caught:
        recover(state, gateway)
    assert caught.value.code == "identity_required"
    assert state.tasks.snapshot() == before
    assert state.last_reply == previous_reply


def test_wrong_verified_user_cannot_read_or_modify_recovered_operations(
    gateway: ToolGateway, state: ConversationState
) -> None:
    """新会话验证为其他用户时，后端归属检查拒绝读取原任务操作。"""
    operation = prepare(gateway, state)
    state.session = SessionContext("other-user")
    gateway.call(
        CallContext(state.session, "coordinator", "identity"),
        "find_user_id_by_email",
        {"email": "u2@example.com"},
    )
    before = state.tasks.snapshot()
    previous_reply = state.last_reply
    with pytest.raises(BusinessError) as caught:
        recover(state, gateway)
    assert caught.value.code == "forbidden"
    assert state.tasks.snapshot() == before
    assert state.last_reply == previous_reply
    assert (
        gateway.backend.get_operation("u1", operation_id=operation["operation_id"])["status"]
        == "prepared"
    )


def test_context_must_belong_to_current_verified_session(
    gateway: ToolGateway, state: ConversationState
) -> None:
    """不能用旧聊天或其他账户的调用上下文对账当前恢复状态。"""
    context = CallContext(SessionContext("old-chat", "u1"), "coordinator", "restore")
    with pytest.raises(BusinessError) as caught:
        reconcile_restored_tasks(state, gateway, context)
    assert caught.value.code == "recovery_identity_mismatch"


def test_wrong_order_prepare_is_rejected_instead_of_becoming_confirmable(
    gateway: ToolGateway, state: ConversationState
) -> None:
    """即使关联请求正确，准备了错误订单也必须失败并释放占用。"""
    operation = prepare(gateway, state, order_id="#W0000004")
    recover(state, gateway)
    task = state.tasks.tasks["task-1"]
    assert task.status == "failed"
    assert "目标" in task.reason
    assert task.operation_ids == [operation["operation_id"]]
    assert task.actions[-1]["status"] == "rejected"


def test_wrong_action_record_is_rejected_despite_matching_request_key(
    gateway: ToolGateway, state: ConversationState
) -> None:
    """请求关联记录动作不匹配时不能只凭关联标识恢复为可确认。"""
    operation = prepare(gateway, state)
    with gateway.backend.engine.begin() as connection:
        connection.execute(
            update(operations)
            .where(operations.c.operation_id == operation["operation_id"])
            .values(action="modify_pending_order_items")
        )
    recover(state, gateway)
    task = state.tasks.tasks["task-1"]
    assert task.status == "failed"
    assert "动作" in task.reason
    assert task.actions[-1]["status"] == "rejected"


def test_checkpoint_reference_to_other_request_blocks_recovery(
    gateway: ToolGateway, state: ConversationState
) -> None:
    """检查点引用了其他请求时不能将其当作当前任务结果或静默重新准备。"""
    operation = gateway.backend.prepare("u1", ACTION, ARGS, request_key("other-task", 1, ACTION))
    state.tasks.tasks["task-1"].operation_ids = [operation["operation_id"]]
    with pytest.raises(BusinessError) as caught:
        recover(state, gateway)
    assert caught.value.code == "recovery_operation_mismatch"
    assert state.tasks.tasks["task-1"].revision == 1
    assert (
        gateway.backend.get_operation("u1", operation_id=operation["operation_id"])["status"]
        == "prepared"
    )


def test_find_operation_is_code_only_and_enforces_valid_request(
    gateway: ToolGateway, state: ConversationState
) -> None:
    """按请求查找不会开放给模型，尚无请求返回空而无效关联直接拒绝。"""
    context = CallContext(state.session, "coordinator", "restore", "task-1")
    assert gateway.find_operation(context, ACTION) is None
    assert "find_operation" not in {
        schema["function"]["name"] for schema in gateway.schemas(context)
    }
    with pytest.raises(BusinessError) as caught:
        gateway.find_operation(replace(context, task_id=None), ACTION)
    assert caught.value.code == "invalid_request_key"

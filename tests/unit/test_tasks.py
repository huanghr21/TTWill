"""用真实业务库检查任务更新边界、同单聚合与旧操作作废。"""

import json
from copy import deepcopy
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from ttwill.backend import RetailBackend
from ttwill.core.tasks import Task, TaskChange, TaskStore
from ttwill.gateway import BusinessError, CallContext, SessionContext, ToolGateway
from ttwill.gateway.skills import SkillCatalog
from ttwill.tracing import Trace


@pytest.fixture
def runtime(
    backend: RetailBackend, tmp_path: Path, skill_catalog: SkillCatalog
) -> tuple[TaskStore, ToolGateway, CallContext]:
    """为任务测试提供真实网关和已验证身份。

    Args:
        backend: 每项测试独立的 SQLite 业务后端。
        tmp_path: 保存本项测试轨迹的目录。
        skill_catalog: 项目声明与普通测试政策组成的 Skill 目录。

    Returns:
        依次为新任务表、真实网关和协调者调用上下文。
    """
    gateway = ToolGateway(backend, Trace(tmp_path / "trace", "chat"), skill_catalog)
    context = CallContext(SessionContext("chat"), "coordinator", "coordinator")
    gateway.call(context, "find_user_id_by_email", {"email": "u1@example.com"})
    return TaskStore(), gateway, context


def request(**changes: object) -> TaskChange:
    """构造订单取消请求，并用指定字段覆盖默认值。

    Args:
        **changes: 覆盖操作类型、任务标识、目标或其他模型允许字段。

    Returns:
        经过严格字段及动作权限校验的任务变更。
    """
    return TaskChange.model_validate(
        {
            "operation": "add",
            "capability": "order_service",
            "action": "cancel_pending_order",
            "order_id": "#W0000001",
            "instruction": "取消一号订单",
            **changes,
        }
    )


def prepare(gateway: ToolGateway, context: CallContext, task: Task) -> dict:
    """为现有任务通过真实网关准备取消操作，并记录代码状态。

    Args:
        gateway: 当前测试使用的真实业务网关。
        context: 提供已验证会话身份的协调者上下文。
        task: 已进入任务表的订单取消任务。

    Returns:
        状态为 prepared 的完整后端操作记录。
    """
    child = CallContext(
        context.session, task.capability, f"child-{task.id}", task.id, task.revision
    )
    gateway.call(child, "load_skill", {"name": "cancel_pending_order"})
    operation = gateway.call(
        child, task.action, {"order_id": task.order_id, "reason": "no longer needed"}
    )
    gateway.release_instance(child)
    task.status = "need_confirm"
    task.operation_ids.append(operation["operation_id"])
    task.actions.append({"tool": task.action, "status": "prepared", "result": deepcopy(operation)})
    return operation


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "forged"),
        ("revision", 100),
        ("status", "done"),
        ("facts", [{"result": "forged"}]),
        ("actions", [{"status": "committed"}]),
        ("operation_ids", ["forged"]),
        ("note", "假装已经完成"),
        ("user_id", "u2"),
    ],
)
def test_change_rejects_code_owned_fields(field: str, value: object) -> None:
    """模型不能通过任务更新接口伪造身份、完成状态或执行事实。"""
    with pytest.raises(ValidationError):
        request(**{field: value})


@pytest.mark.parametrize(
    "fields",
    [
        {"task_id": "model-id"},
        {"operation": "update"},
        {"capability": "coordinator"},
        {"capability": "after_sales"},
        {"action": "commit"},
        {"action": "load_skill"},
        {"instruction": "  "},
        {"order_id": None},
        {"action": "modify_user_address"},
        {"missing": [1]},
        {"missing": [" "]},
    ],
)
def test_change_validates_request_shape_and_capability(fields: dict) -> None:
    """模型必须提供完整任务规格，并遵守能力、目标与字段类型约束。"""
    with pytest.raises(ValidationError):
        request(**fields)


def test_cancel_only_accepts_task_id_and_reason() -> None:
    """撤销请求须说明原因，并且不能夹带另一份任务规格。"""
    assert TaskChange(operation="cancel", task_id="t1", reason="条件分支不再需要")
    with pytest.raises(ValidationError):
        TaskChange(operation="cancel", task_id="t1")
    with pytest.raises(ValidationError):
        TaskChange(operation="cancel", task_id="t1", reason="不要了", instruction="变成另一件事")


def test_new_tasks_generate_ids_and_isolate_snapshots(runtime: tuple) -> None:
    """任务标识来自代码，快照中的列表修改不会篡改内部事实。"""
    store, gateway, context = runtime
    result = store.apply([request(missing=["取消原因"])], gateway, context)
    task = next(iter(store.tasks.values()))
    assert UUID(task.id).version == 4
    assert task.revision == 1 and task.is_write and task.status == "pending"
    assert result == [{"task_id": task.id, "status": "pending", "reason": None, "changed": True}]
    snapshot = store.snapshot()
    snapshot[0]["missing"].append("伪造字段")
    snapshot[0]["facts"].append({"fake": True})
    assert task.missing == ["取消原因"] and task.facts == []


def test_same_order_action_is_aggregated_once_before_first_dispatch(runtime: tuple) -> None:
    """同一批同单同类商品需求合为一个任务，并保留每项诉求和信息缺口。"""
    store, gateway, context = runtime
    store.apply(
        [
            request(
                action="modify_pending_order_items", instruction="蓝衬衫换红衬衫", missing=["卡号"]
            ),
            request(
                action="modify_pending_order_items",
                instruction="小码鞋换大码鞋",
                missing=["卡号", "颜色"],
            ),
        ],
        gateway,
        context,
    )
    assert len(store.tasks) == 1
    task = next(iter(store.tasks.values()))
    assert task.instruction == "蓝衬衫换红衬衫\n小码鞋换大码鞋"
    assert task.missing == ["卡号", "颜色"]
    assert task.revision == 1


def test_same_order_action_add_revises_existing_request_only_once(runtime: tuple) -> None:
    """向已有请求补充同类诉求时，同一批汇总只增加一次修订号。"""
    store, gateway, context = runtime
    store.apply([request(instruction="现有诉求")], gateway, context)
    task_id = next(iter(store.tasks))
    store.apply([request(instruction="补充一"), request(instruction="补充二")], gateway, context)
    task = store.tasks[task_id]
    assert task.revision == 2
    assert task.instruction == "现有诉求\n补充一\n补充二"


def test_distinct_actions_orders_and_read_tasks_are_separate(runtime: tuple) -> None:
    """不同订单和不同动作独立建任务，重复查询也允许独立执行。"""
    store, gateway, context = runtime
    store.apply(
        [
            request(),
            request(order_id="#W0000004"),
            request(action="modify_pending_order_address", instruction="修改地址"),
            request(action="get_order_details"),
            request(action="get_order_details"),
        ],
        gateway,
        context,
    )
    assert len(store.tasks) == 5
    assert [task.is_write for task in store.tasks.values()] == [True, True, True, False, False]


def test_user_address_has_no_order_and_condition_stays_natural_language(runtime: tuple) -> None:
    """默认地址任务不绑定订单，条件原文保留给协调者判断。"""
    store, gateway, context = runtime
    condition = "如果用户仍打算保留账户，则执行"
    store.apply(
        [request(action="modify_user_address", order_id=None, condition=condition)],
        gateway,
        context,
    )
    task = next(iter(store.tasks.values()))
    assert task.status == "conditional"
    assert task.condition == condition and task.order_id is None


def test_update_invalidates_prepared_operation_and_preserves_history(runtime: tuple) -> None:
    """改口先作废旧操作，增加修订号，保留事实与操作审计并清理旧提示。"""
    store, gateway, context = runtime
    store.apply([request(missing=["旧缺口"])], gateway, context)
    task = next(iter(store.tasks.values()))
    task.facts.append({"tool": "get_order_details", "result": {"status": "pending"}})
    task.note = "旧提示"
    operation = prepare(gateway, context, task)
    store.apply(
        [request(operation="update", task_id=task.id, instruction="换一个取消原因")],
        gateway,
        context,
    )
    updated = store.tasks[task.id]
    assert updated.revision == 2 and updated.status == "pending"
    assert updated.operation_ids == [] and updated.missing == [] and updated.note == ""
    assert updated.facts == task.facts
    assert updated.actions[0] == task.actions[0]
    assert updated.actions[-1]["status"] == "rejected"
    assert gateway.get_operation(context, operation["operation_id"])["status"] == "rejected"
    replacement = prepare(gateway, context, updated)
    assert replacement["request_key"] != operation["request_key"]
    assert replacement["operation_id"] != operation["operation_id"]


def test_cancel_releases_order_and_records_reason(runtime: tuple) -> None:
    """撤销备用或准备中的任务是终止状态，保留原因并释放订单占用。"""
    store, gateway, context = runtime
    store.apply([request()], gateway, context)
    task = next(iter(store.tasks.values()))
    operation = prepare(gateway, context, task)
    task.missing = ["过时的取消原因"]
    task.note = "请继续补充资料"
    task.facts.append({"tool": "get_order_details", "result": {"items": [{"item_id": "a"}]}})
    previous_facts, previous_actions = deepcopy(task.facts), deepcopy(task.actions)
    store.apply(
        [TaskChange(operation="cancel", task_id=task.id, reason="用户改变主意")], gateway, context
    )
    cancelled = store.tasks[task.id]
    assert cancelled.status == "cancelled" and cancelled.reason == "用户改变主意"
    assert cancelled.operation_ids == []
    assert cancelled.missing == [] and cancelled.note == ""
    assert cancelled.facts == previous_facts
    assert cancelled.actions[:-1] == previous_actions
    assert cancelled.actions[-1]["status"] == "rejected"
    assert gateway.get_operation(context, operation["operation_id"])["status"] == "rejected"
    store.apply([request()], gateway, context)
    fresh = next(item for item in store.tasks.values() if item.id != task.id)
    assert prepare(gateway, context, fresh)["status"] == "prepared"


@pytest.mark.parametrize("status", ["done", "cancelled", "failed"])
def test_terminal_field_cleanup_preserves_full_execution_evidence(status):
    """终态清理仅删除信息缺口和备注，事实、操作及其他任务字段保持完整。"""
    task = Task(
        "t1",
        "order_service",
        "cancel_pending_order",
        "取消一号订单",
        status=status,
        missing=["取消原因"],
        note="请再确认",
        facts=[{"result": {"items": [{"item_id": "a", "price": 100}]}}],
        actions=[{"result": {"operation_id": "op1", "status": "committed", "preview": {}}}],
        operation_ids=["op1"],
        reason="核验后的状态说明",
    )
    before = task.to_dict()
    assert before["missing"] == ["取消原因"] and before["note"] == "请再确认"

    task.clear_terminal_fields()
    task.clear_terminal_fields()

    assert task.to_dict() == before | {"missing": [], "note": ""}


@pytest.mark.parametrize(
    "status",
    ["pending", "conditional", "running", "waiting_info", "pending_confirm", "needs_verification"],
)
def test_active_task_cleanup_keeps_missing_information_and_note(status):
    """尚未终止的任务保留信息缺口与备注，不因共用清理入口丢失办理要求。"""
    task = Task(
        "t1",
        "order_service",
        "cancel_pending_order",
        "取消一号订单",
        status=status,
        missing=["取消原因"],
        note="请补充取消原因",
    )
    before = task.to_dict()

    task.clear_terminal_fields()

    assert task.to_dict() == before


@pytest.mark.parametrize("status", ["done", "cancelled"])
@pytest.mark.parametrize("operation", ["update", "cancel"])
def test_terminal_task_history_cannot_be_rewritten(
    runtime: tuple, status: str, operation: str
) -> None:
    """已完成或已撤销任务不能被普通更新接口覆写。"""
    store, gateway, context = runtime
    store.apply([request()], gateway, context)
    task = next(iter(store.tasks.values()))
    task.status = status
    before = store.snapshot()
    change = (
        request(operation="update", task_id=task.id)
        if operation == "update"
        else TaskChange(operation="cancel", task_id=task.id, reason="不需要了")
    )
    with pytest.raises(BusinessError, match="已终止"):
        store.apply([change], gateway, context)
    assert store.snapshot() == before
    store.apply([request()], gateway, context)
    assert len(store.tasks) == 2


@pytest.mark.parametrize("operation", ["update", "cancel", "add"])
def test_committed_but_unverified_task_cannot_be_changed(runtime: tuple, operation: str) -> None:
    """即使内存还停在待确认，后端已提交也阻止改口、撤销或合并。"""
    store, gateway, context = runtime
    store.apply([request()], gateway, context)
    task = next(iter(store.tasks.values()))
    prepared = prepare(gateway, context, task)
    gateway.commit(context, prepared["operation_id"], prepared["args"])
    before = store.snapshot()
    change = (
        TaskChange(operation="cancel", task_id=task.id, reason="反悔")
        if operation == "cancel"
        else request(operation=operation, **({"task_id": task.id} if operation == "update" else {}))
    )
    with pytest.raises(BusinessError) as error:
        store.apply([change], gateway, context)
    assert error.value.code == "task_committed"
    assert store.snapshot() == before
    assert gateway.get_operation(context, prepared["operation_id"])["status"] == "committed"


def test_invalid_later_change_has_no_partial_state_or_rejection(runtime: tuple) -> None:
    """后项引用不存在的任务时，前项修改不发布，也不作废原准备项。"""
    store, gateway, context = runtime
    store.apply([request()], gateway, context)
    task = next(iter(store.tasks.values()))
    prepared = prepare(gateway, context, task)
    before = store.snapshot()
    with pytest.raises(BusinessError) as error:
        store.apply(
            [
                request(operation="update", task_id=task.id, instruction="新诉求"),
                TaskChange(operation="cancel", task_id="missing", reason="不要了"),
            ],
            gateway,
            context,
        )
    assert error.value.code == "task_not_found"
    assert store.snapshot() == before
    assert gateway.get_operation(context, prepared["operation_id"])["status"] == "prepared"


def test_all_operations_are_checked_before_any_rejection(runtime: tuple) -> None:
    """批次后项已经提交时，前项仍待确认的操作保持有效。"""
    store, gateway, context = runtime
    store.apply([request(), request(order_id="#W0000004")], gateway, context)
    first, second = store.tasks.values()
    pending = prepare(gateway, context, first)
    committed = prepare(gateway, context, second)
    gateway.commit(context, committed["operation_id"], committed["args"])
    before = store.snapshot()
    with pytest.raises(BusinessError, match="须先核验"):
        store.apply(
            [
                TaskChange(operation="cancel", task_id=first.id, reason="不要了"),
                TaskChange(operation="cancel", task_id=second.id, reason="不要了"),
            ],
            gateway,
            context,
        )
    assert store.snapshot() == before
    assert gateway.get_operation(context, pending["operation_id"])["status"] == "prepared"


def test_update_cannot_duplicate_another_active_order_action(runtime: tuple) -> None:
    """完整规格更新不能绕过同单同类唯一活跃任务的约束。"""
    store, gateway, context = runtime
    store.apply([request(), request(order_id="#W0000004")], gateway, context)
    _, second = store.tasks.values()
    before = store.snapshot()
    with pytest.raises(BusinessError, match="已有同单同类任务"):
        store.apply([request(operation="update", task_id=second.id)], gateway, context)
    assert store.snapshot() == before


def test_conflicting_conditions_are_not_silently_interpreted(runtime: tuple) -> None:
    """代码不擅自把两个自然语言条件合成新的执行逻辑。"""
    store, gateway, context = runtime
    with pytest.raises(BusinessError, match="条件不同"):
        store.apply([request(condition="失败时"), request(condition="成功时")], gateway, context)
    assert store.snapshot() == []


@pytest.mark.parametrize("first_operation", ["add", "update", "cancel"])
def test_mixed_or_duplicate_changes_to_same_task_are_rejected(
    runtime: tuple, first_operation: str
) -> None:
    """同批同任务的混合或重复改动须先明确最终规格，不能被预聚合改写顺序。"""
    store, gateway, context = runtime
    store.apply([request()], gateway, context)
    task = next(iter(store.tasks.values()))
    prepared = prepare(gateway, context, task)
    before = store.snapshot()
    first = (
        TaskChange(operation="cancel", task_id=task.id, reason="不要了")
        if first_operation == "cancel"
        else request(
            operation=first_operation,
            **({"task_id": task.id} if first_operation == "update" else {}),
        )
    )
    with pytest.raises(BusinessError):
        store.apply(
            [first, request(operation="update", task_id=task.id, instruction="最终诉求")],
            gateway,
            context,
        )
    assert store.snapshot() == before
    assert gateway.get_operation(context, prepared["operation_id"])["status"] == "prepared"


def test_task_updates_and_rejections_have_trace_evidence(runtime: tuple) -> None:
    """成功变更保留前后快照，拒绝更新保留护栏与明确原因。"""
    store, gateway, context = runtime
    store.apply([request()], gateway, context)
    with pytest.raises(BusinessError):
        store.apply(
            [TaskChange(operation="cancel", task_id="missing", reason="撤销")], gateway, context
        )
    events = [json.loads(line) for line in gateway.trace.path.read_text().splitlines()]
    updates = [event for event in events if event["kind"] == "task_update" and event["end"]]
    assert updates[-1]["attrs"]["before"] == []
    assert updates[-1]["attrs"]["after"] == store.snapshot()
    blocked = [
        event for event in events if event["kind"] == "guard" and event["status"] == "blocked"
    ]
    assert blocked[-1]["name"] == "task_not_found"

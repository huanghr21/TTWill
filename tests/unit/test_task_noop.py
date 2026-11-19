"""复现 88 题重复规格更新，验证任务无变化时保留已准备操作。"""

import json
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest

from ttwill.backend import RetailBackend
from ttwill.core.tasks import Task, TaskChange, TaskStore
from ttwill.gateway import BusinessError, CallContext, SessionContext, ToolGateway
from ttwill.gateway.skills import SkillCatalog
from ttwill.tracing import Trace

# 88 题第 7 轮两次 update_tasks 的请求规格一致；只保留业务输入作为离线证据。
REPLAY_CHANGE = {
    "operation": "update",
    "task_id": "373d11e9d8574e8a855493d7f52bd2e7",
    "capability": "order_service",
    "action": "cancel_pending_order",
    "order_id": "#W8835847",
    "condition": "用户已确认取消整个订单 #W8835847",
    "instruction": (
        "取消待发货订单 #W8835847 的全部商品（T恤、游戏鼠标、书架 玻璃/白色/5英尺），"
        "订单金额 ¥4,829.79，取消原因：误下单（ordered by mistake）。"
    ),
    "missing": [],
}


@pytest.fixture
def prepared_runtime(
    tmp_path: Path, seed_snapshot: dict, skill_catalog: SkillCatalog
) -> Iterator[tuple[TaskStore, ToolGateway, CallContext, Task, dict]]:
    """用真实准备操作复现重复更新前的任务状态。

    Args:
        tmp_path: 独立业务库与轨迹目录。
        seed_snapshot: 可修改的微型业务种子，不加载完整开发数据。
        skill_catalog: 为取消工具解锁提供测试政策。

    Yields:
        任务表、网关、协调上下文、现有任务与后端准备记录。
    """
    order = seed_snapshot["orders"].pop("#W0000001")
    order["order_id"] = "#W8835847"
    seed_snapshot["orders"][order["order_id"]] = order
    seed_snapshot["users"]["u1"]["orders"][0] = order["order_id"]
    backend = RetailBackend(tmp_path / "business.sqlite")
    backend.initialize(seed_snapshot)
    try:
        gateway = ToolGateway(backend, Trace(tmp_path / "trace", "chat"), skill_catalog)
        context = CallContext(SessionContext("chat"), "coordinator", "coordinator")
        gateway.call(context, "find_user_id_by_email", {"email": "u1@example.com"})
        task = Task(
            id=REPLAY_CHANGE["task_id"],
            revision=2,
            status="pending_confirm",
            note="已准备取消，请确认。",
            reason="等待用户确认",
            **{
                key: deepcopy(value)
                for key, value in REPLAY_CHANGE.items()
                if key not in {"operation", "task_id"}
            },
        )
        store = TaskStore()
        store.tasks[task.id] = task
        child = CallContext(context.session, task.capability, "child", task.id, task.revision)
        gateway.call(child, "load_skill", {"name": task.action})
        operation = gateway.call(
            child, task.action, {"order_id": task.order_id, "reason": "ordered by mistake"}
        )
        gateway.release_instance(child)
        task.operation_ids = [operation["operation_id"]]
        task.facts = [{"tool": "get_order_details", "result": deepcopy(order)}]
        task.actions = [{"tool": task.action, "result": deepcopy(operation)}]
        yield store, gateway, context, task, operation
    finally:
        backend.close()


def replay(**updates: object) -> TaskChange:
    """构造真实重复规格，并覆盖当前用例需要变化的字段。

    Args:
        **updates: 覆盖真实规格中的任务请求字段。

    Returns:
        经过字段约束校验的任务变更。
    """
    return TaskChange.model_validate({**REPLAY_CHANGE, **updates})


@pytest.mark.parametrize("order_id", ["#W8835847", " w8835847 ", " #w8835847 "])
def test_replayed_identical_update_preserves_prepared_request(prepared_runtime, order_id) -> None:
    """88 的相同规格重发或订单号规范化同值，不作废准备项也不清除代码状态。"""
    store, gateway, context, task, operation = prepared_runtime
    before = store.snapshot()
    records_before = len(gateway.records)
    result = store.apply([replay(order_id=order_id)], gateway, context)
    assert result == [
        {"task_id": task.id, "status": "pending_confirm", "reason": "无变化", "changed": False}
    ]
    assert store.snapshot() == before
    assert gateway.get_operation(context, operation["operation_id"])["status"] == "prepared"
    assert not any(
        record["kind"] in {"prepare", "reject"} for record in gateway.records[records_before:]
    )


def test_identical_add_merge_keeps_existing_prepared_request(prepared_runtime) -> None:
    """相同新增诉求与现有同单同类任务合并后内容不变，准备项继续有效。"""
    store, gateway, context, task, operation = prepared_runtime
    before = store.snapshot()
    result = store.apply([replay(operation="add", task_id=None)], gateway, context)
    assert result[0]["changed"] is False and result[0]["reason"] == "无变化"
    assert result[0]["task_id"] == task.id
    assert store.snapshot() == before
    assert gateway.get_operation(context, operation["operation_id"])["status"] == "prepared"


@pytest.mark.parametrize(
    "updates",
    [
        {"instruction": "请按新要求取消订单"},
        {"condition": None},
        {"missing": ["取消原因"]},
        {"order_id": "#W0000004"},
        {"action": "modify_pending_order_address"},
    ],
)
def test_real_request_changes_still_invalidate_old_prepare(prepared_runtime, updates) -> None:
    """任一实际业务规格变化仍增加修订、作废旧准备并保留完整历史事实。"""
    store, gateway, context, task, operation = prepared_runtime
    result = store.apply([replay(**updates)], gateway, context)
    current = store.tasks[task.id]
    assert result == [
        {"task_id": task.id, "status": current.status, "reason": None, "changed": True}
    ]
    assert current.revision == task.revision + 1 and current.operation_ids == []
    assert current.note == "" and current.facts == task.facts
    assert current.actions[:-1] == task.actions
    assert gateway.get_operation(context, operation["operation_id"])["status"] == "rejected"


def test_capability_change_is_a_real_request_change(prepared_runtime) -> None:
    """可执行同一查询的能力发生变化时，仍视为真实任务修订。"""
    store, gateway, context, _, _ = prepared_runtime
    task = Task("read", "order_service", "get_order_details", "查询订单", order_id="#W0000004")
    store.tasks[task.id] = task
    result = store.apply(
        [
            TaskChange(
                operation="update",
                task_id=task.id,
                capability="after_sales",
                action=task.action,
                instruction=task.instruction,
                order_id=task.order_id,
            )
        ],
        gateway,
        context,
    )
    assert result[0]["changed"] is True
    assert store.tasks[task.id].revision == 2
    assert store.tasks[task.id].capability == "after_sales"


def test_noop_and_changed_tasks_have_only_affected_briefs(prepared_runtime) -> None:
    """混合批次只返回涉及任务的摘要，无变化与新建各自标识且不泄露业务快照。"""
    store, gateway, context, task, operation = prepared_runtime
    store.tasks["untouched"] = Task("untouched", "order_service", "get_user_details", "查询用户")
    result = store.apply(
        [replay(), replay(operation="add", task_id=None, order_id="#W0000004")], gateway, context
    )
    assert len(result) == 2 and [item["changed"] for item in result] == [False, True]
    assert all(set(item) == {"task_id", "status", "reason", "changed"} for item in result)
    assert result[0]["task_id"] == task.id
    assert "untouched" not in {item["task_id"] for item in result}
    result[0]["status"] = "done"
    assert store.tasks[task.id].status == "pending_confirm"
    assert gateway.get_operation(context, operation["operation_id"])["status"] == "prepared"


@pytest.mark.parametrize("operation", ["update", "add"])
def test_noop_cannot_bypass_same_batch_duplicate_guard(prepared_runtime, operation) -> None:
    """无变化更新或新增合并仍占用同批任务标识，后续重复变更不得绕过护栏。"""
    store, gateway, context, _, operation_record = prepared_runtime
    before = store.snapshot()
    first = replay(
        operation=operation, task_id=None if operation == "add" else REPLAY_CHANGE["task_id"]
    )
    with pytest.raises(BusinessError, match="同批次不能重复"):
        store.apply([first, replay(instruction="另一条诉求")], gateway, context)
    assert store.snapshot() == before
    assert gateway.get_operation(context, operation_record["operation_id"])["status"] == "prepared"


def test_noop_mixed_with_invalid_target_keeps_all_state(prepared_runtime) -> None:
    """无变化更新与非法订单混批时整批拒绝，原准备项保持有效。"""
    store, gateway, context, _, operation = prepared_runtime
    before = store.snapshot()
    with pytest.raises(BusinessError, match="order_id 不合法"):
        store.apply(
            [replay(), replay(operation="add", task_id=None, order_id="#W0000003")],
            gateway,
            context,
        )
    assert store.snapshot() == before
    assert gateway.get_operation(context, operation["operation_id"])["status"] == "prepared"


def test_mixed_existing_requests_only_reject_changed_prepare(prepared_runtime) -> None:
    """同批无变化与实质修改的已有任务各自处理，只拒绝实际修改的准备项。"""
    store, gateway, context, first, first_operation = prepared_runtime
    store.apply([replay(operation="add", task_id=None, order_id="#W0000004")], gateway, context)
    second = next(task for task in store.tasks.values() if task.id != first.id)
    child = CallContext(context.session, second.capability, "second", second.id, second.revision)
    gateway.call(child, "load_skill", {"name": second.action})
    second_operation = gateway.call(
        child, second.action, {"order_id": second.order_id, "reason": "ordered by mistake"}
    )
    gateway.release_instance(child)
    second.operation_ids = [second_operation["operation_id"]]
    second.status = "pending_confirm"
    result = store.apply(
        [replay(), replay(task_id=second.id, order_id=second.order_id, instruction="改为不再需要")],
        gateway,
        context,
    )
    assert [item["changed"] for item in result] == [False, True]
    assert store.tasks[first.id].to_dict() == first.to_dict()
    assert store.tasks[second.id].revision == second.revision + 1
    assert gateway.get_operation(context, first_operation["operation_id"])["status"] == "prepared"
    assert gateway.get_operation(context, second_operation["operation_id"])["status"] == "rejected"


def test_noop_keeps_full_trace_but_returns_only_brief(prepared_runtime) -> None:
    """模型只收到无变化摘要，观测轨迹仍保存相同的完整前后快照。"""
    store, gateway, context, _, _ = prepared_runtime
    before = store.snapshot()
    result = store.apply([replay(missing=None)], gateway, context)
    events = [json.loads(line) for line in gateway.trace.path.read_text().splitlines()]
    updates = [
        event for event in events if event["kind"] == "task_update" and event["event"] == "end"
    ]
    assert updates[-1]["attrs"] == {"before": before, "after": before}
    assert result[0]["changed"] is False
    assert not {"facts", "actions", "operation_ids", "instruction"}.intersection(result[0])

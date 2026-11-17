"""在重新验证身份后，用后端操作记录对账恢复的任务检查点。"""

from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

from ttwill.core.tasks import Task
from ttwill.gateway import BusinessError, CallContext, ToolGateway

if TYPE_CHECKING:
    from ttwill.core.coordinator import ConversationState


def reconcile_restored_tasks(
    state: "ConversationState", gateway: ToolGateway, context: CallContext
) -> None:
    """找回准备或提交成功但检查点尚未更新的操作，清除旧展示授权。

    恢复只读取业务结果、核验或清理不用的准备项，不提交任何操作。任务状态
    原地更新，由调用方随正常图节点保存至检查点；本模块不另存任务状态。

    Args:
        state: 从检查点还原并绑定本段聊天已验证身份的对话状态。
        gateway: 连接该任务原业务库的网关。
        context: 本段聊天重新验证身份后的代码上下文，须与 state.session 一致。

    Raises:
        BusinessError: 身份未验证、上下文身份不一致、操作不属于当前用户，或
            检查点已有操作引用无法在当前请求中对账。
        OSError: 业务库或轨迹读写失败；调用方须保留原检查点并在恢复时重试。
    """
    if state.session.user_id is None or context.session.user_id is None:
        raise BusinessError("恢复任务前须重新验证用户身份", code="identity_required")
    if (
        state.session.session_id != context.session.session_id
        or state.session.user_id != context.session.user_id
    ):
        raise BusinessError("恢复上下文与当前会话身份不一致", code="recovery_identity_mismatch")

    before = state.tasks.snapshot()
    with gateway.trace.span("recovery", "reconcile_restored_tasks") as span:
        # 先检查所有操作归属，再修改任务，防止错误账户取得部分恢复状态。
        recovered = []
        for task in state.tasks.tasks.values():
            if task.status == "done" or not task.is_write:
                continue
            task_context = replace(context, task_id=task.id, revision=task.revision)
            operation = gateway.find_operation(task_context, task.action)
            if task.operation_ids and (
                operation is None or set(task.operation_ids) != {operation["operation_id"]}
            ):
                for operation_id in task.operation_ids:
                    gateway.get_operation(task_context, operation_id)
                reason = "检查点操作引用与当前任务请求不一致，不能重新准备或提交"
                _guard(gateway, reason)
                raise BusinessError(reason, code="recovery_operation_mismatch")
            recovered.append((task, task_context, operation))

        state.last_display = []
        for task in state.tasks.tasks.values():
            if not task.is_write and task.status == "running":
                task.status = "conditional" if task.condition else "pending"
        for task, task_context, operation in recovered:
            _restore_task(task, gateway, task_context, operation)
        span.attrs.update(before=before, after=state.tasks.snapshot())


def _restore_task(
    task: Task, gateway: ToolGateway, context: CallContext, operation: dict | None
) -> None:
    """按当前请求的真实后端状态更新单项任务。

    Args:
        task: 要原地更新的任务，当前状态尚未完成。
        gateway: 执行拒绝与只读核验的业务网关。
        context: 已绑定该任务 ID 和修订号的代码上下文。
        operation: 按当前任务请求查到的完整操作；None 表示没有操作记录。
    """
    if operation is None:
        if task.status in {"running", "pending_confirm", "needs_verification"}:
            task.status = "conditional" if task.condition else "pending"
        return

    task.operation_ids = [operation["operation_id"]]
    _record_operation(task, operation)
    target_matches = (
        operation["args"].get("order_id") == task.order_id
        if task.order_id is not None
        else operation["args"].get("user_id") == context.session.user_id
    )
    if operation["action"] != task.action or not target_matches:
        reason = "恢复的操作动作或业务目标与任务不一致"
        _guard(gateway, reason)
        if operation["status"] == "prepared":
            operation = gateway.reject(context, operation["operation_id"], reason)
            _record_operation(task, operation)
        task.status, task.reason = "failed", reason
        task.reason_internal = True
        return

    if operation["status"] == "committed":
        task.status = "needs_verification"
        verification = gateway.verify_operation(context, operation["operation_id"])
        fact = {"verification": verification}
        if fact not in task.facts:
            task.facts.append(deepcopy(fact))
        task.status = "done" if verification["verified"] else "failed"
        task.reason = verification["reason"]
        task.reason_internal = False
        task.missing = []
    elif task.status in {"cancelled", "failed"}:
        if operation["status"] == "prepared":
            operation = gateway.reject(
                context, operation["operation_id"], task.reason or "任务已经终止"
            )
            _record_operation(task, operation)
    elif operation["status"] == "prepared":
        task.status = "pending_confirm"
        task.reason = None
        task.reason_internal = False
        task.missing = []
    else:
        task.operation_ids = []
        task.revision += 1
        task.status = "conditional" if task.condition else "pending"
        task.reason = f"旧操作已失效：{operation['status']}，需重新准备与展示"
        task.reason_internal = False


def _record_operation(task: Task, operation: dict) -> None:
    """将实际操作记录补入任务历史，重复恢复不重复追加同一结果。

    Args:
        task: 保存恢复操作历史的任务。
        operation: 网关从业务库读到或拒绝后返回的完整操作记录。
    """
    entry = {
        "kind": "recovery",
        "tool": operation["action"],
        "args": deepcopy(operation["args"]),
        "status": operation["status"],
        "operation_id": operation["operation_id"],
        "result": deepcopy(operation),
    }
    if entry not in task.actions:
        task.actions.append(entry)


def _guard(gateway: ToolGateway, reason: str) -> None:
    """记录恢复时发现的任务与后端操作不一致。"""
    with gateway.trace.span(
        "guard", "recovery_matches", attrs={"allowed": False, "reason": reason}
    ) as span:
        span.status = "blocked"

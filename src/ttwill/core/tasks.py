"""校验模型提出的任务变更，保留代码拥有的事实与执行状态。"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ttwill.gateway import (
    CAPABILITY_TOOLS,
    READ_TOOLS,
    WRITE_TOOLS,
    BusinessError,
    CallContext,
    ToolGateway,
)


@dataclass
class Task:
    """保存代码分配的任务标识、请求和执行事实。

    Attributes:
        id: 由代码生成的稳定任务标识。
        capability: 执行该请求的业务能力名称。
        action: 本任务请求的实际业务工具名称。
        instruction: 用户诉求与办理要求，同单同类诉求合并保存。
        revision: 请求修订号，初始为 1，修改请求时递增。
        order_id: 订单目标；用户资料任务或未定位订单时为 None。
        condition: 协调者解释的自然语言条件；None 表示无前置条件。
        status: 代码维护的办理状态，写任务完成须经核验代码设置。
        missing: 模型报告的信息缺口。
        facts: 由实际业务查询形成的事实。
        actions: 由网关记录形成的操作历史，包括已作废请求。
        operation_ids: 当前修订关联的后端操作标识。
        note: 子 Agent 提供给协调者的备注，不作为执行证据。
        reason: 撤销或失败原因；正常待办时为 None。
        reason_internal: 原因是否只用于任务状态与轨迹，由代码设置并随检查点保存。
    """

    id: str
    capability: str
    action: str
    instruction: str
    revision: int = 1
    order_id: str | None = None
    condition: str | None = None
    status: str = "pending"
    missing: list[str] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    operation_ids: list[str] = field(default_factory=list)
    note: str = ""
    reason: str | None = None
    reason_internal: bool = False

    @property
    def is_write(self) -> bool:
        """根据固定业务工具集合判断本任务是否请求写操作。"""
        return self.action in WRITE_TOOLS

    def to_dict(self) -> dict[str, Any]:
        """返回与内部可变字段隔离的完整任务快照。"""
        return asdict(self)


class TaskChange(BaseModel):
    """限定模型只能提出请求增改、撤销原因和信息缺口。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    operation: Literal["add", "update", "cancel"]
    task_id: str | None = None
    capability: str | None = None
    action: str | None = None
    order_id: str | None = None
    instruction: str | None = None
    condition: str | None = None
    reason: str | None = None
    missing: list[str] | None = None

    @field_validator(
        "task_id", "capability", "action", "order_id", "instruction", "condition", "reason"
    )
    @classmethod
    def nonempty_text(cls, value: str | None) -> str | None:
        """拒绝仅由空白组成的请求字段，保留有效原文。"""
        if value is not None and not value.strip():
            raise ValueError("任务文字字段须包含非空白字符")
        return value

    @field_validator("missing")
    @classmethod
    def nonempty_missing(cls, value: list[str] | None) -> list[str] | None:
        """拒绝空白信息缺口并保留每项的原始说明。"""
        if value is not None and any(not item.strip() for item in value):
            raise ValueError("信息缺口须包含非空白字符")
        return value

    @model_validator(mode="after")
    def validate_operation(self) -> "TaskChange":
        """核对不同变更类型的必填字段、工具权限与目标约束。

        Returns:
            已满足新增、更新或撤销请求约束的当前对象。

        Raises:
            ValueError: 必填字段缺失、工具越权或目标字段不符合该动作。
        """
        if self.operation == "cancel":
            if self.task_id is None or self.reason is None:
                raise ValueError("撤销任务须提供 task_id 和 reason")
            if any(
                getattr(self, name) is not None
                for name in (
                    "capability",
                    "action",
                    "order_id",
                    "instruction",
                    "condition",
                    "missing",
                )
            ):
                raise ValueError("撤销任务只接受 task_id 和 reason")
            return self
        if self.operation == "add" and self.task_id is not None:
            raise ValueError("新增任务的标识由代码生成")
        if self.operation == "update" and self.task_id is None:
            raise ValueError("更新任务须提供 task_id")
        if self.capability is None or self.action is None or self.instruction is None:
            raise ValueError("新增或更新任务须提供完整 capability、action 和 instruction")
        allowed = CAPABILITY_TOOLS.get(self.capability, frozenset())
        if self.action not in (READ_TOOLS | WRITE_TOOLS) & allowed:
            raise ValueError("任务动作不在该业务能力的工具白名单内")
        if self.action in WRITE_TOOLS and self.action != "modify_user_address":
            if self.order_id is None:
                raise ValueError("订单写任务须提供 order_id")
        if self.action == "modify_user_address" and self.order_id is not None:
            raise ValueError("修改用户地址不接受订单目标")
        return self


class TaskStore:
    """在内存中保存任务，批量校验后应用模型允许提出的变更。"""

    def __init__(self) -> None:
        """初始化空任务表；持久化由后续运行时检查点负责。"""
        self.tasks: dict[str, Task] = {}

    def snapshot(self) -> list[dict[str, Any]]:
        """按创建顺序返回全部任务的独立快照。"""
        return [task.to_dict() for task in self.tasks.values()]

    def apply(
        self, changes: list[TaskChange], gateway: ToolGateway, context: CallContext
    ) -> list[dict[str, Any]]:
        """校验整批请求，再作废旧准备项并发布任务变更。

        更新或撤销前统一对账受影响任务的全部操作。已提交操作必须先核验，
        不能通过模型改口抹去；已完成或已撤销任务保持历史不变。更新保留历史
        事实和动作。请求内容未变化时保留当前修订、状态和准备项；有变化时清除
        旧操作引用，新的准备请求使用递增后的修订号。

        Args:
            changes: 已通过严格字段校验的增、改、撤请求列表。
            gateway: 用于对账、拒绝旧准备项并记录任务变化的业务网关。
            context: 当前协调者的代码调用上下文，旧操作按原任务关联记录。

        Returns:
            按请求顺序返回本批涉及任务的摘要，仅含 task_id、status、reason、
            changed；无变化时 changed 为 False、reason 为“无变化”。

        Raises:
            BusinessError: 任务不存在、已经终止、存在未核验提交，合并条件
                不一致，或操作对账、作废被业务后端拒绝。
            OSError: 轨迹写入失败。
        """
        planned = deepcopy(self.tasks)
        affected: dict[str, str] = {}
        changes_applied: dict[str, bool] = {}
        try:
            normalized = [
                change.model_copy(
                    update={"order_id": gateway.normalize_order_id(context, change.order_id)}
                )
                if change.order_id is not None
                else change
                for change in changes
            ]
            for change in self._coalesce(normalized):
                if change.task_id in changes_applied:
                    raise BusinessError(
                        "同批次不能重复修改或撤销同一任务，请一次给出最终完整规格",
                        code="invalid_task_change",
                    )
                task_id, changed = self._plan(planned, change)
                if task_id in changes_applied:
                    raise BusinessError(
                        "同批次不能重复修改或撤销同一任务，请一次给出最终完整规格",
                        code="invalid_task_change",
                    )
                changes_applied[task_id] = changed
                if task_id in self.tasks:
                    affected[task_id] = change.reason or "任务请求已修改，作废旧准备操作"
            operations = self._reconcile(affected, gateway, context)
        except BusinessError as exc:
            self._blocked(gateway, str(exc), exc.code)
            raise

        for task_id, records in operations.items():
            if not changes_applied[task_id]:
                continue
            task_context = replace(context, task_id=task_id, revision=self.tasks[task_id].revision)
            for operation in records:
                if operation["status"] == "prepared":
                    operation = gateway.reject(
                        task_context, operation["operation_id"], affected[task_id]
                    )
                    planned[task_id].actions.append(
                        {
                            "kind": "reject",
                            "tool": operation["action"],
                            "args": {"reason": affected[task_id]},
                            "status": operation["status"],
                            "operation_id": operation["operation_id"],
                            "result": deepcopy(operation),
                        }
                    )
        before = self.snapshot()
        after = [task.to_dict() for task in planned.values()]
        with gateway.trace.span(
            "task_update", "apply_tasks", attrs={"before": before, "after": after}
        ):
            self.tasks = planned
        return [
            {
                "task_id": task_id,
                "status": planned[task_id].status,
                "reason": planned[task_id].reason if changed else "无变化",
                "changed": changed,
            }
            for task_id, changed in changes_applied.items()
        ]

    @staticmethod
    def _coalesce(changes: list[TaskChange]) -> list[TaskChange]:
        """在应用前汇总同一批次、同单同类的新增写诉求。

        Args:
            changes: 经过字段验证且保持模型提出顺序的变更列表。

        Returns:
            保持首个出现位置的合并列表；查询和用户地址请求保持独立。

        Raises:
            BusinessError: 同单同类的条件不同，无法由代码合并解释。
        """
        result = []
        positions: dict[tuple[str, str], int] = {}
        for change in changes:
            if (
                change.operation != "add"
                or change.action not in WRITE_TOOLS
                or change.order_id is None
            ):
                result.append(change)
                continue
            key = (change.order_id, change.action)
            if key not in positions:
                positions[key] = len(result)
                result.append(change)
                continue
            position = positions[key]
            previous = result[position]
            if previous.condition != change.condition:
                raise BusinessError(
                    "同单同类诉求的条件不同，请明确合并后的条件", code="invalid_task_change"
                )
            result[position] = previous.model_copy(
                update={
                    "instruction": TaskStore._join(previous.instruction, change.instruction),
                    "missing": list(
                        dict.fromkeys((previous.missing or []) + (change.missing or []))
                    ),
                }
            )
        return result

    @staticmethod
    def _join(first: str, second: str) -> str:
        """按出现顺序保留两条独立诉求，完全相同的内容只保留一次。"""
        return first if first == second else f"{first}\n{second}"

    def _plan(self, planned: dict[str, Task], change: TaskChange) -> tuple[str, bool]:
        """在独立任务副本上校验并模拟一项变更。

        Args:
            planned: 本批次尚未发布的任务副本。
            change: 单项增、改、撤请求。

        Returns:
            本项请求涉及的任务标识，以及请求内容是否实际变化。

        Raises:
            BusinessError: 任务不存在、已经终止、更新后与另一活跃任务重复，
                或新增合并时自然语言条件不一致。
        """
        existing = None
        if change.operation == "add" and change.action in WRITE_TOOLS and change.order_id:
            existing = self._same_request(planned, change.order_id, change.action)
        if change.operation == "add" and existing is None:
            task = Task(
                id=uuid4().hex,
                capability=change.capability,
                action=change.action,
                instruction=change.instruction,
                order_id=change.order_id,
                condition=change.condition,
                status="conditional" if change.condition else "pending",
                missing=deepcopy(change.missing or []),
            )
            planned[task.id] = task
            return task.id, True
        task_id = existing.id if existing is not None else change.task_id
        if task_id not in planned:
            raise BusinessError("任务不存在", code="task_not_found")
        task = planned[task_id]
        if task.status in {"done", "cancelled"}:
            raise BusinessError("已终止的任务不能修改；新的诉求须新建任务", code="task_finalized")
        if change.operation == "add" and task.condition != change.condition:
            raise BusinessError(
                "同单同类诉求的条件不同，请明确合并后的条件", code="invalid_task_change"
            )
        if change.operation == "update" and change.action in WRITE_TOOLS and change.order_id:
            duplicate = self._same_request(planned, change.order_id, change.action, task.id)
            if duplicate is not None:
                raise BusinessError(
                    "更新目标已有同单同类任务，请更新现有任务", code="invalid_task_change"
                )
        candidate = {}
        if change.operation != "cancel":
            candidate = {
                "capability": change.capability,
                "action": change.action,
                "order_id": change.order_id,
                "condition": change.condition,
                "instruction": (
                    self._join(task.instruction, change.instruction)
                    if change.operation == "add"
                    else change.instruction
                ),
                "missing": (
                    list(dict.fromkeys(task.missing + (change.missing or [])))
                    if change.operation == "add"
                    else deepcopy(change.missing or [])
                ),
            }
            if all(getattr(task, name) == value for name, value in candidate.items()):
                return task.id, False
        task.revision += 1
        task.operation_ids = []
        task.note = ""
        task.reason_internal = False
        if change.operation == "cancel":
            task.status = "cancelled"
            task.reason = change.reason
            task.missing = []
            return task.id, True
        for name, value in candidate.items():
            setattr(task, name, value)
        task.status = "conditional" if change.condition else "pending"
        task.reason = None
        return task.id, True

    @staticmethod
    def _same_request(
        tasks: dict[str, Task], order_id: str, action: str, exclude: str | None = None
    ) -> Task | None:
        """查找尚未终止的同单同类写任务。

        Args:
            tasks: 待检查的任务表。
            order_id: 目标订单标识。
            action: 实际写工具名称。
            exclude: 要跳过的任务标识；None 表示检查全部任务。

        Returns:
            匹配的活跃任务；不存在时返回 None。
        """
        return next(
            (
                task
                for task in tasks.values()
                if task.id != exclude
                and task.status not in {"done", "cancelled"}
                and task.order_id == order_id
                and task.action == action
            ),
            None,
        )

    def _reconcile(
        self, affected: dict[str, str], gateway: ToolGateway, context: CallContext
    ) -> dict[str, list[dict[str, Any]]]:
        """在拒绝任何准备项前检查所有受影响任务的后端操作。

        Args:
            affected: 受修改影响的已有任务标识与作废原因。
            gateway: 提供权威操作记录的工具网关。
            context: 对账身份来源，任务关联使用各任务的原修订号。

        Returns:
            各受影响任务的当前后端操作记录。

        Raises:
            BusinessError: 任一操作已经提交但尚未核验，或后端对账失败。
        """
        operations = {}
        for task_id in affected:
            task = self.tasks[task_id]
            task_context = replace(context, task_id=task.id, revision=task.revision)
            records = [
                gateway.get_operation(task_context, operation_id)
                for operation_id in task.operation_ids
            ]
            if any(record["status"] == "committed" for record in records):
                raise BusinessError(
                    "任务已有提交结果，须先核验，不能修改或撤销", code="task_committed"
                )
            operations[task_id] = records
        return operations

    @staticmethod
    def _blocked(gateway: ToolGateway, reason: str, code: str) -> None:
        """记录被拒绝的任务变更及原因。

        Args:
            gateway: 提供当前会话轨迹的业务网关。
            reason: 本次变更被拒绝的说明。
            code: 区分拒绝原因的业务错误码。
        """
        with gateway.trace.span("guard", code, attrs={"allowed": False, "reason": reason}) as span:
            span.status = "blocked"

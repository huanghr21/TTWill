"""维护本次运行的任务清单，协调分派、用户确认和执行后核验。"""

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass, field
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ttwill.core.confirmation import ConfirmationJudge
from ttwill.core.subagent import DispatchTask, SubAgentRunner
from ttwill.core.tasks import Task, TaskChange, TaskStore
from ttwill.gateway import BusinessError, CallContext, SessionContext, ToolGateway
from ttwill.gateway.presentation import action_label, operation_content
from ttwill.llm import LLMClient
from ttwill.llm.types import LLMError


class UpdateTasks(BaseModel):
    """只接受任务增、改、撤提议，不接受业务完成或用户授权字段。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    changes: list[TaskChange] = Field(min_length=1)


class CompletePlan(BaseModel):
    """声明本轮任务修改已经提出，由代码开始独立判定用户原话。"""

    model_config = ConfigDict(extra="forbid", strict=True)


class DispatchTasks(BaseModel):
    """选择由代码建立的任务，每批可分派不同订单的多个实例。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    task_ids: list[str] = Field(min_length=1)


class Reply(BaseModel):
    """提供回复要点及要展示的任务，业务状态和操作详情由代码填写。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    message_hint: str
    show_task_ids: list[str] | None = None


@dataclass
class ConversationState:
    """保存一次运行中的对话与可信任务状态，第 5 步再接入检查点。

    Attributes:
        session: 本段聊天的身份，由网关验证。
        tasks: 代码维护的任务集合。
        history: 实际接收和发出的用户、助手消息，仅驻留内存。
        last_display: 最近一次助手回复实际展示的操作及参数快照。
        turn: 已开始处理的用户消息数。
    """

    session: SessionContext
    tasks: TaskStore = field(default_factory=TaskStore)
    history: list[dict] = field(default_factory=list)
    last_display: list[dict] = field(default_factory=list)
    turn: int = 0


class Coordinator:
    """通过模型选择任务与工具，由代码控制授权、提交和完成状态。"""

    def __init__(
        self,
        gateway: ToolGateway,
        runner: SubAgentRunner,
        llm: LLMClient,
        *,
        max_rounds: int,
        model_tier: str = "flash",
        judge: ConfirmationJudge | None = None,
    ) -> None:
        """绑定本次运行的业务网关、子 Agent 和模型。

        Args:
            gateway: 所有业务访问及轨迹使用的工具网关。
            runner: 按能力执行独立子任务的循环。
            llm: 协调决策使用的统一模型调用层。
            max_rounds: 每条用户消息允许的最多协调决策轮数，须为正整数。
            model_tier: 协调者及默认确认判定器使用的模型档位。
            judge: 独立确认判定器；None 时使用协调者的模型调用层构造。

        Raises:
            ValueError: max_rounds 不是正整数。
        """
        if type(max_rounds) is not int or max_rounds < 1:
            raise ValueError("协调决策轮数上限必须为正整数")
        self.gateway, self.runner, self.llm = gateway, runner, llm
        self.max_rounds, self.model_tier = max_rounds, model_tier
        self.judge = judge or ConfirmationJudge(llm, model_tier=model_tier)

    async def handle_turn(self, state: ConversationState, user_message: str) -> dict:
        """处理一条真实用户消息，返回回复与可信任务快照。

        先允许模型提出本轮全部改口或撤销；complete_plan 后才判定上一回复的
        操作。模型不能提供确认结果，不能调用提交。未完成工作会阻止提前回复。

        Args:
            state: 当前聊天的内存状态，原地更新，不写独立任务状态文件。
            user_message: 调用入口收到的真实用户原话，不得替换成模型摘要。

        Returns:
            包含 reply、tasks、displayed、confirmation_results、rounds 的字典；
            displayed 仅列本次实际回复中出现的操作，供下一条消息判定。

        Raises:
            ValueError: 用户消息为空。
            asyncio.CancelledError: 本轮调用被取消。
            OSError: 业务或轨迹读写失败；提交结果未知时保留待对账状态。
        """
        if not user_message.strip():
            raise ValueError("用户消息不能为空")
        state.turn += 1
        previous = deepcopy(state.last_display)
        state.last_display = []
        state.history.append({"role": "user", "content": user_message})
        context = CallContext(state.session, "coordinator", uuid4().hex)
        confirmed = False
        decisions = []
        with self.gateway.trace.span("turn", "处理用户消息", turn=state.turn):
            with self.gateway.trace.span("coordinator", "任务协调"):
                self._reconcile(state, context)
                messages = self._messages(state)
                for round_number in range(1, self.max_rounds + 1):
                    try:
                        response = await self.llm.complete(
                            self.model_tier, messages, tools=self._schemas(context, confirmed)
                        )
                    except (LLMError, TimeoutError) as exc:
                        reason = f"协调模型调用失败：{type(exc).__name__}"
                        self._fail_unfinished(state, context, reason)
                        return self._reply(
                            state,
                            context,
                            Reply(message_hint=""),
                            decisions,
                            round_number,
                            notice=reason,
                        )
                    calls = response.message.get("tool_calls") or []
                    if len(calls) != 1:
                        self._guard(
                            "coordinator_format",
                            "每轮须调用一个协调工具，可在 dispatch_tasks 中批量分派",
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "请只调用一个工具。需要多个任务时在同一 dispatch_tasks 中列出。"
                                ),
                            }
                        )
                        continue
                    messages.append(deepcopy(response.message))
                    call = calls[0]
                    name = call["function"]["name"]
                    try:
                        arguments = json.loads(call["function"]["arguments"])
                        if not isinstance(arguments, dict):
                            raise ValueError("工具参数必须是对象")
                        if name == "update_tasks":
                            self._require_identity(state)
                            updates = UpdateTasks.model_validate(arguments)
                            for change in updates.changes:
                                if change.capability is not None:
                                    self.runner.registry.get(change.capability)
                            outcome = state.tasks.apply(updates.changes, self.gateway, context)
                        elif name == "complete_plan":
                            CompletePlan.model_validate(arguments)
                            if confirmed:
                                raise ValueError("本轮已经判定过用户确认")
                            decisions = await self._confirm(state, context, user_message, previous)
                            confirmed = True
                            outcome = {
                                "confirmation_results": decisions,
                                "tasks": state.tasks.snapshot(),
                            }
                        elif name == "dispatch_tasks":
                            self._require_identity(state)
                            if not confirmed:
                                raise ValueError("先提出所有任务修改并调用 complete_plan，再分派")
                            selection = DispatchTasks.model_validate(arguments)
                            outcome = await self._dispatch(state, context, selection.task_ids)
                        elif name == "reply":
                            proposal = Reply.model_validate(arguments)
                            if not confirmed:
                                raise ValueError("先提出所有任务修改并调用 complete_plan，再回复")
                            self._reconcile(state, context)
                            unfinished = self._actionable(state)
                            if unfinished:
                                raise ValueError(
                                    f"任务仍需推进或判断条件：{unfinished}；先分派、补充缺失信息或撤销不再需要的条件分支"
                                )
                            return self._reply(state, context, proposal, decisions, round_number)
                        elif name in {
                            schema["function"]["name"] for schema in self.gateway.schemas(context)
                        }:
                            outcome = self.gateway.call(context, name, arguments)
                        else:
                            raise ValueError(f"当前阶段没有工具：{name}")
                    except ValueError as exc:
                        self._guard("coordinator_tool", str(exc))
                        outcome = {
                            "error": getattr(exc, "code", "invalid_tool_call"),
                            "reason": str(exc),
                        }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(outcome, ensure_ascii=False),
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "tasks": state.tasks.snapshot(),
                                    "verified": state.session.user_id is not None,
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
                reason = f"协调者达到本轮上限 {self.max_rounds}，未完成本轮处理"
                self._fail_unfinished(state, context, reason)
                return self._reply(
                    state,
                    context,
                    Reply(message_hint=""),
                    decisions,
                    self.max_rounds,
                    notice=reason,
                )

    def _messages(self, state: ConversationState) -> list[dict]:
        """组装固定政策、历史和本轮可信任务快照。

        Args:
            state: 当前会话的真实消息和代码任务状态。

        Returns:
            新的 Chat Completions 消息列表；旧对话仅供理解，不作为当前授权。
        """
        prompt = (
            "你是电商客服协调者，负责理解全部诉求并维护任务。每次调用一个工具。"
            "未验证身份时先查找用户；用户只能服务一个账户。"
            "首先用 update_tasks 提出本轮所有新增、修改、撤销，再用 complete_plan 结束计划。"
            "用户改口时必须在 complete_plan 之前更新旧任务，不能只据‘可以’提交。"
            "此后用 dispatch_tasks 选择任务，或用 update_tasks 撤销已不需要的条件备用任务。"
            "条件保持自然语言，由你判断；条件满足时明确分派该任务，不再需要则撤销并说明原因。"
            "写任务按订单×动作类型聚合完整商品清单；同单不同动作逐个确认，先地址/支付后商品。"
            "每个任务提供完整 instruction、capability、action，订单动作还要 order_id。"
            "capability 从下方能力目录选择，action 使用该能力业务工具的完整名称。"
            "模型只提出任务增改撤、missing与message_hint；任何status、revision或确认字段均由代码维护。"
            "不能声明写任务完成，不能提交操作，用户确认只由独立判定器读取当前用户原话。"
            "reply 只能在所有任务完成、撤销、等待用户或失败后调用。"
            "show_task_ids为要展示待确认详情的任务，省略展示全部，空列表不展示。"
            "任务结果与操作详情由代码生成，message_hint只作回复要点。\n"
            + self.gateway.skills.common_policy
            + "\n可分派能力：\n"
            + json.dumps(self.runner.registry.catalog(), ensure_ascii=False)
        )
        return [
            {"role": "system", "content": prompt},
            *deepcopy(state.history),
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "tasks": state.tasks.snapshot(),
                        "verified": state.session.user_id is not None,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    def _schemas(self, context: CallContext, confirmed: bool) -> list[dict]:
        """按身份和本轮阶段开放协调工具，不开放提交与核验。

        Args:
            context: 当前协调者的调用身份。
            confirmed: 本轮是否已经完成独立确认判定。

        Returns:
            当前可见的函数工具定义。
        """
        models = {"reply": Reply} if confirmed else {"complete_plan": CompletePlan}
        if context.session.user_id is not None:
            models["update_tasks"] = UpdateTasks
            if confirmed:
                models["dispatch_tasks"] = DispatchTasks
        return self.gateway.schemas(context) + [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": model.__doc__,
                    "parameters": model.model_json_schema(),
                },
            }
            for name, model in models.items()
        ]

    @staticmethod
    def _require_identity(state: ConversationState) -> None:
        """在任务更新和分派前要求会话身份已由网关验证。"""
        if state.session.user_id is None:
            raise ValueError("请先验证用户身份")

    def _guard(self, name: str, reason: str) -> None:
        """记录当前协调过程拒绝继续执行的具体原因。"""
        with self.gateway.trace.span(
            "guard", name, attrs={"allowed": False, "reason": reason}
        ) as span:
            span.status = "blocked"

    def _reconcile(self, state: ConversationState, context: CallContext) -> None:
        """刷新旧操作，释放失败任务遗留的准备项，核验已提交结果。

        Args:
            state: 原地更新的任务状态；过期任务递增修订后重新准备。
            context: 读取和处理操作时使用的已验证身份。
        """
        if state.session.user_id is None:
            return
        for task in state.tasks.tasks.values():
            if task.status not in {"pending_confirm", "needs_verification", "failed"}:
                continue
            for operation_id in list(task.operation_ids):
                operation = self.gateway.get_operation(context, operation_id)
                if task.status == "failed" and operation["status"] != "committed":
                    if operation["status"] == "prepared":
                        self.gateway.reject(context, operation_id, task.reason or "任务已失败")
                    continue
                if operation["status"] == "committed":
                    self._verify(task, context, operation)
                elif operation["status"] == "prepared":
                    task.status = "pending_confirm"
                else:
                    task.operation_ids = []
                    task.revision += 1
                    task.status = "conditional" if task.condition else "pending"
                    task.reason = f"旧操作已失效：{operation['status']}，需重新准备与展示"

    async def _confirm(
        self,
        state: ConversationState,
        context: CallContext,
        user_message: str,
        previous: list[dict],
    ) -> list[dict]:
        """只判定前一条实际回复中仍与当前任务修订一致的操作。

        Args:
            state: 已应用本轮任务修改的状态。
            context: 代码提交、拒绝和核验使用的身份。
            user_message: 当前用户原话。
            previous: 上次成功发出的回复的展示快照。

        Returns:
            按操作列出的判定、执行状态及核验结果，部分失败不覆盖其他结果。
        """
        eligible = {}
        for shown in previous:
            task = state.tasks.tasks.get(shown["task_id"])
            if (
                task is None
                or task.revision != shown["revision"]
                or task.status != "pending_confirm"
                or shown["operation_id"] not in task.operation_ids
                or shown["session_id"] != state.session.session_id
                or shown["turn"] != state.turn - 1
            ):
                continue
            operation = self.gateway.get_operation(context, shown["operation_id"])
            if (
                operation["status"] == "prepared"
                and operation["args"] == shown["args"]
                and operation["preview"] == shown["preview"]
            ):
                eligible[operation["operation_id"]] = (task, shown)
        if not eligible:
            return []
        # 保留实际展示范围，避免撤销一项后把多项短回复误判为单项确认。
        judgments = await self.judge.judge(user_message, previous)
        results = []
        for judgment in judgments:
            if judgment["operation_id"] not in eligible:
                continue
            task, shown = eligible[judgment["operation_id"]]
            entry = dict(judgment)
            try:
                if judgment["decision"] == "confirm":
                    task.status = "needs_verification"
                    task.reason = "正在提交并核验；结果未知时先对账"
                    operation = self.gateway.commit(context, shown["operation_id"], shown["args"])
                    task.actions.append({"kind": "commit", "result": deepcopy(operation)})
                    entry["verification"] = self._verify(task, context, operation)
                elif judgment["decision"] in {"reject", "modify"}:
                    operation = self.gateway.reject(
                        context, shown["operation_id"], judgment["reason"]
                    )
                    task.actions.append({"kind": "reject", "result": deepcopy(operation)})
                    task.operation_ids = []
                    task.reason = judgment["reason"]
                    if judgment["decision"] == "modify":
                        task.revision += 1
                        task.status = "waiting_info"
                        task.missing = ["请补充修改后的完整要求"]
                    else:
                        task.status = "cancelled"
                entry["status"] = task.status
            except BusinessError as exc:
                operation = self.gateway.get_operation(context, shown["operation_id"])
                if operation["status"] == "committed":
                    task.status = "needs_verification"
                else:
                    if operation["status"] == "prepared":
                        self.gateway.reject(context, operation["operation_id"], str(exc))
                    task.status = "failed"
                task.reason = str(exc)
                entry.update(status=task.status, error=exc.code, reason=str(exc))
            results.append(entry)
        return results

    def _verify(self, task: Task, context: CallContext, operation: dict) -> dict:
        """重读业务实体核验提交结果，只有此处允许写任务进入 done。

        Args:
            task: 等待核验的写任务，原地记录状态与事实。
            context: 核验使用的已验证身份。
            operation: 后端已提交的操作记录。

        Returns:
            网关生成的完整核验结果；失败保留已提交事实并记录原因。
        """
        verification = self.gateway.verify_operation(context, operation["operation_id"])
        task.facts.append({"verification": deepcopy(verification)})
        task.status = "done" if verification["verified"] else "failed"
        task.reason = None if verification["verified"] else verification["reason"]
        return verification

    async def _dispatch(
        self, state: ConversationState, context: CallContext, task_ids: list[str]
    ) -> dict:
        """选择不冲突的任务并行执行，同一订单只推进一项写操作。

        Args:
            state: 当前任务清单。
            context: 结果异常时作废准备项的代码身份。
            task_ids: 模型明确选择的现有任务 ID，不能重复。

        Returns:
            dispatched 的任务快照和 blocked 的占用原因。

        Raises:
            ValueError: 任务不存在、重复，或任务已结束、已经等待确认。
            asyncio.CancelledError: 调用被取消；等待并行实例退出后向上传播。
            OSError: 业务或轨迹读写失败；先保留操作引用并停止并行实例。
        """
        if len(set(task_ids)) != len(task_ids) or any(
            key not in state.tasks.tasks for key in task_ids
        ):
            raise ValueError("分派任务不存在或重复")
        tasks = [state.tasks.tasks[key] for key in task_ids]
        if any(task.status not in {"pending", "conditional"} for task in tasks):
            raise ValueError("只可分派待执行或条件已满足的任务；缺信息任务先更新完整要求")
        occupied = {
            task.order_id
            for task in state.tasks.tasks.values()
            if task.is_write
            and task.order_id
            and task.status in {"pending_confirm", "needs_verification", "running"}
        }
        selected, blocked = [], []
        for task in tasks:
            if task.is_write and task.order_id in occupied:
                blocked.append({"task_id": task.id, "reason": "同一订单的前项操作尚未提交或作废"})
                continue
            if task.is_write and task.order_id:
                occupied.add(task.order_id)
            task.status = "running"
            selected.append(task)
        if blocked:
            self._guard("order_serialization", json.dumps(blocked, ensure_ascii=False))
        executions = [
            asyncio.create_task(self._dispatch_one(state, context, task)) for task in selected
        ]
        try:
            await asyncio.gather(*executions)
        except (asyncio.CancelledError, OSError) as exc:
            for execution in executions:
                if not execution.done():
                    execution.cancel()
            await asyncio.gather(*executions, return_exceptions=True)
            for task in selected:
                if task.status == "running":
                    task.status, task.reason = "failed", f"分派中断：{type(exc).__name__}"
                    self._interrupted_records(state, task)
            raise
        return {"dispatched": [task.to_dict() for task in selected], "blocked": blocked}

    async def _dispatch_one(
        self, state: ConversationState, context: CallContext, task: Task
    ) -> None:
        """执行一个任务，校验操作目标后将实际返回并入可信状态。

        Args:
            state: 当前会话身份。
            context: 对账和清理操作使用的协调者上下文。
            task: 待执行任务，原地更新状态与实际事实。

        Raises:
            asyncio.CancelledError: 实例被取消，保留事实并释放准备项后向上传播。
            OSError: 执行或结果对账失败；保留操作引用供再次对账。
        """
        try:
            result = await self.runner.run(
                state.session,
                task.capability,
                DispatchTask(
                    task.id,
                    f"{task.instruction}\n本次任务动作：{task.action}；目标订单：{task.order_id}。只准备这一项完整请求。",
                    task.revision,
                    task.is_write,
                    deepcopy(task.facts),
                ),
            )
            self._merge_result(context, task, result)
        except (LLMError, TimeoutError, ValueError) as exc:
            task.status, task.reason = "failed", f"子 Agent 调用失败：{type(exc).__name__}"
            self._interrupted_records(state, task)
            self._discard_prepared(context, task)
            return
        except (asyncio.CancelledError, OSError) as exc:
            task.status, task.reason = "failed", f"子 Agent 中断：{type(exc).__name__}"
            self._interrupted_records(state, task)
            self._discard_prepared(context, task)
            raise

    def _merge_result(self, context: CallContext, task: Task, result: dict) -> None:
        """保存真实执行记录，核对动作与目标后更新任务状态。

        Args:
            context: 读取后端及核验结果使用的代码身份。
            task: 要原地补入操作引用、事实和状态的任务。
            result: 子 Agent 循环由网关记录组装的结构化结果。

        Raises:
            BusinessError: 对账、清理或核验被后端拒绝。
            OSError: 后端或轨迹读写失败。
        """
        task.actions.extend(deepcopy(result["actions"]))
        task.facts.extend(deepcopy(result["facts"]))
        task.missing, task.message_hint = result["missing"], result["message_hint"]
        operation_ids = list(
            dict.fromkeys(
                action["operation_id"] for action in result["actions"] if action.get("operation_id")
            )
        )
        task.operation_ids = operation_ids
        operations = [self.gateway.get_operation(context, key) for key in operation_ids]
        valid = len(operations) <= 1 and all(
            task.is_write
            and op["action"] == task.action
            and op["args"].get("order_id") == task.order_id
            for op in operations
        )
        if not valid:
            for operation in operations:
                if operation["status"] == "prepared":
                    self.gateway.reject(
                        context, operation["operation_id"], "子 Agent 操作与分派任务不一致"
                    )
            task.status, task.reason = "failed", "子 Agent 操作与分派任务不一致"
            self._guard("dispatch_result", task.reason)
            return
        task.operation_ids = operation_ids
        task.reason = result["reason"]
        task.status = {
            "need_confirm": "pending_confirm",
            "need_info": "waiting_info",
            "ready": "done",
        }.get(result["status"], result["status"])
        if task.is_write and task.status == "done":
            task.status, task.reason = "failed", "写任务必须提交后核验"
        if (
            not task.is_write
            and task.status == "done"
            and not any(
                fact["tool"] == task.action
                and (task.order_id is None or fact["args"].get("order_id") == task.order_id)
                for fact in result["facts"]
            )
        ):
            task.status, task.reason = "failed", "未取得当前任务动作与目标对应的查询事实"
            self._guard("dispatch_read_result", task.reason)
        for operation in operations:
            if operation["status"] == "committed":
                self._verify(task, context, operation)

    def _discard_prepared(self, context: CallContext, task: Task) -> None:
        """清理中断任务的准备项，已提交操作保留为待核验。

        Args:
            context: 读取与拒绝操作使用的代码身份。
            task: 已保存操作引用的中断任务；清理读写失败时仍保留引用。

        Raises:
            BusinessError: 操作不存在或身份归属不符。
            OSError: 清理仍无法读写；下一轮按保存的引用继续对账。
        """
        for operation_id in task.operation_ids:
            operation = self.gateway.get_operation(context, operation_id)
            if operation["status"] == "prepared":
                rejected = self.gateway.reject(context, operation_id, task.reason)
                task.actions.append({"kind": "reject", "result": deepcopy(rejected)})
            elif operation["status"] == "committed":
                task.status = "needs_verification"

    def _interrupted_records(self, state: ConversationState, task: Task) -> None:
        """保留中断实例已产生的准备与清理事实，避免丢失操作引用。

        Args:
            state: 用于限定会话身份的当前状态。
            task: 已标为失败的任务，补入本修订的真实网关记录。
        """
        records = [
            record
            for record in self.gateway.records
            if record.get("task_id") == task.id
            and record.get("revision") == task.revision
            and record.get("session_id") == state.session.session_id
            and record.get("user_id") == state.session.user_id
        ]
        task.actions.extend(deepcopy(records))
        task.operation_ids = list(
            dict.fromkeys(
                record["operation_id"] for record in records if record.get("operation_id")
            )
        )

    @staticmethod
    def _actionable(state: ConversationState) -> list[str]:
        """列出仍须推进或判断条件的任务。

        Args:
            state: 当前任务清单，包含正在等待用户的前置事项。

        Returns:
            需要继续处理的任务 ID；等待同单确认的后项可留到下一轮。
        """
        waiting = [
            task
            for task in state.tasks.tasks.values()
            if task.status in {"pending_confirm", "waiting_info", "needs_verification"}
        ]
        occupied = {
            task.order_id
            for task in waiting
            if task.is_write and task.order_id and task.status != "waiting_info"
        }
        return [
            task.id
            for task in state.tasks.tasks.values()
            if task.status == "running"
            or (task.status == "pending" and not (task.is_write and task.order_id in occupied))
            or (task.status == "conditional" and not waiting)
        ]

    def _fail_unfinished(self, state: ConversationState, context: CallContext, reason: str) -> None:
        """终止本轮仍可推进的任务并释放其准备项，保留已经等待用户的结果。

        Args:
            state: 要原地更新的任务状态。
            context: 对账和释放准备项使用的代码身份。
            reason: 本轮超限或协调失败的具体原因，同时记录到 guard。
        """
        self._guard("coordinator_limit", reason)
        for task_id in self._actionable(state):
            task = state.tasks.tasks[task_id]
            for operation_id in task.operation_ids:
                operation = self.gateway.get_operation(context, operation_id)
                if operation["status"] == "prepared":
                    self.gateway.reject(context, operation_id, reason)
                elif operation["status"] == "committed":
                    self._verify(task, context, operation)
            if task.status != "done":
                task.status, task.reason = "failed", reason

    def _reply(
        self,
        state: ConversationState,
        context: CallContext,
        proposal: Reply,
        decisions: list[dict],
        rounds: int,
        *,
        notice: str | None = None,
    ) -> dict:
        """用实际状态与后端详情生成回复，成功记录后才保存可确认的展示快照。

        Args:
            state: 当前任务及消息状态。
            context: 刷新展示操作时使用的身份。
            proposal: 模型要点和希望展示的任务 ID，不含可信业务字段。
            decisions: 本轮逐操作确认与执行结果。
            rounds: 本轮已经使用的协调决策轮数。
            notice: 代码产生的本轮异常说明；None 表示无需附加通知。

        Returns:
            回复、任务快照、实际展示项、确认结果与轮数。

        Raises:
            ValueError: 指定的展示任务不存在或重复。
            OSError: 回复或轨迹无法记录；不保存新的展示授权范围。
        """
        selected = proposal.show_task_ids
        if selected is not None and (
            len(set(selected)) != len(selected)
            or any(key not in state.tasks.tasks for key in selected)
        ):
            raise ValueError("展示任务不存在或重复")
        lines, displayed = ([notice] if notice else []), []
        labels = {
            "pending_confirm": "等待确认",
            "waiting_info": "需要补充信息",
            "done": "已完成",
            "cancelled": "已撤销",
            "failed": "未完成",
            "pending": "等待前项操作",
            "conditional": "等待条件确定",
            "needs_verification": "等待核对操作结果",
        }
        for task in state.tasks.tasks.values():
            label = task.order_id or "账户资料"
            lines.append(
                f"{label} · {action_label(task.action)}：{labels.get(task.status, task.status)}。"
            )
            if task.missing:
                lines.append("需要补充：" + "；".join(task.missing))
            if task.reason:
                lines.append(task.reason)
            if task.status == "done" and not task.is_write:
                lines.extend(
                    json.dumps(fact["result"], ensure_ascii=False)
                    for fact in task.facts
                    if "result" in fact
                )
            if task.status != "pending_confirm" or (
                selected is not None and task.id not in selected
            ):
                continue
            for operation_id in task.operation_ids:
                operation = self.gateway.get_operation(context, operation_id)
                if operation["status"] != "prepared":
                    continue
                content = operation_content(operation)
                lines.append(content)
                displayed.append(
                    {
                        "operation_id": operation_id,
                        "task_id": task.id,
                        "revision": task.revision,
                        "session_id": state.session.session_id,
                        "turn": state.turn,
                        "args": deepcopy(operation["args"]),
                        "preview": deepcopy(operation["preview"]),
                        "content": content,
                    }
                )
        if displayed:
            lines.append("请确认以上操作；多项可指明具体操作，或明确回复“全部确认”。")
        if not lines:
            lines.append(proposal.message_hint or "请提供邮箱，或姓名和邮编，以便验证身份。")
        reply = "\n\n".join(lines)
        with self.gateway.trace.span(
            "reply", "回复用户", attrs={"content": reply, "displayed": displayed}
        ):
            pass
        with self.gateway.trace.span(
            "state_snapshot", "本轮任务状态", attrs={"tasks": state.tasks.snapshot()}
        ):
            pass
        state.last_display = deepcopy(displayed)
        state.history.append({"role": "assistant", "content": reply})
        return {
            "reply": reply,
            "tasks": state.tasks.snapshot(),
            "displayed": deepcopy(displayed),
            "confirmation_results": decisions,
            "rounds": rounds,
        }

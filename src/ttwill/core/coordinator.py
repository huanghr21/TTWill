"""维护本次运行的任务清单，协调分派、用户确认和执行后核验。"""

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass, field, replace
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ttwill.core.business_data import business_updates, limit_business_data
from ttwill.core.confirmation import ConfirmationJudge
from ttwill.core.context import DATA_NOTICE, STATE_NOTICE
from ttwill.core.subagent import DispatchTask, SubAgentRunner
from ttwill.core.task_view import task_brief, task_view
from ttwill.core.tasks import Task, TaskChange, TaskStore
from ttwill.gateway import (
    IDENTITY_TOOLS,
    READ_TOOLS,
    BusinessError,
    CallContext,
    SessionContext,
    ToolGateway,
)
from ttwill.llm import LLMClient
from ttwill.llm.types import LLMError

STOP_NOTICE = "抱歉，这次处理没有完成，请稍后重试或说明是否需要转人工。"
PREPARED_STOP_NOTICE = (
    "以下操作已准备好，请核对后确认；其余事项本轮未能处理完，可稍后继续或转人工。"
)
COMPLETION_CLAIMS = ("已取消", "已退款", "已退货", "已换货", "已修改", "已提交")


class UpdateTasks(BaseModel):
    """只接受任务增、改、撤提议，不接受业务完成或用户授权字段。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    changes: list[TaskChange] = Field(min_length=1)


class CompletePlan(BaseModel):
    """结束本轮计划，由代码独立判定用户原话，再分派 task_ids 指定的已有任务。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    task_ids: list[str] = Field(
        default_factory=list,
        description="要在确认判定后分派的待执行或条件已满足任务；省略或空列表只结束计划",
    )


class DispatchTasks(BaseModel):
    """选择由代码建立的任务，每批可分派不同订单的多个实例。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    task_ids: list[str] = Field(min_length=1)


class Reply(BaseModel):
    """撰写原样发送的客服正文，并选择由代码附加确认卡片的任务。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    message: str = Field(min_length=1, description="原样发送给用户的非空回复正文")
    show_task_ids: list[str] | None = None

    @field_validator("message")
    @classmethod
    def nonempty_message(cls, value: str) -> str:
        """拒绝空白回复，保留有效正文的原始空格与换行。"""
        if not value.strip():
            raise ValueError("回复正文须包含非空白字符")
        return value


@dataclass
class ConversationState:
    """保存协调者使用的对话与可信任务状态，由运行时决定内存或检查点生命周期。

    Attributes:
        session: 本段聊天的身份，由网关验证。
        tasks: 代码维护的任务集合。
        history: 直接调用时的消息历史；持久运行时完整原文另存对话档案。
        last_display: 最近一次助手回复实际展示的操作及参数快照。
        last_reply: 用户实际收到的上一条完整回复，供确认判定消除正文歧义。
        business_data: 按实体去重的业务参考数据，随检查点保存，不作为授权或执行事实。
        turn: 已开始处理的用户消息数。
    """

    session: SessionContext
    tasks: TaskStore = field(default_factory=TaskStore)
    history: list[dict] = field(default_factory=list)
    last_display: list[dict] = field(default_factory=list)
    last_reply: str = ""
    business_data: dict[str, dict] = field(default_factory=dict)
    turn: int = 0


@dataclass
class TurnState:
    """保存一条用户消息内的协调进度，供图节点之间写入检查点。

    Attributes:
        user_message: 本轮真实用户原话，独立确认只读取此字段。
        instance_id: 本轮协调实例标识，不参与业务准备的确定性请求键。
        previous: 本轮开始时保留的上一回复实际展示快照。
        previous_reply: 上一条实际发送的完整回复，保留正文与确认卡片。
        business_snapshot: 轮初冻结的业务数据区，本轮新结果通过工具往返传递。
        messages: 未压缩的本轮模型与工具往返，持久运行时同时归档。
        base_messages: 初始提示与历史占用的消息数，之后的条目属于本轮往返。
        confirmed: 是否已完成本轮独立确认判定。
        decisions: 本轮逐操作确认与执行结果。
        rounds: 已使用的协调决策次数。
        invalid_outputs: 不合格的模型原文，仅供档案观察，不作为可执行工具往返。
        result: 本轮已生成的回复；None 表示还需继续决策。
    """

    user_message: str
    instance_id: str
    previous: list[dict]
    messages: list[dict]
    base_messages: int
    previous_reply: str = ""
    business_snapshot: list[dict] = field(default_factory=list)
    confirmed: bool = False
    decisions: list[dict] = field(default_factory=list)
    rounds: int = 0
    invalid_outputs: list[dict] = field(default_factory=list)
    result: dict | None = None


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
        data_max_chars: int = 16000,
    ) -> None:
        """绑定本次运行的业务网关、子 Agent 和模型。

        Args:
            gateway: 所有业务访问及轨迹使用的工具网关。
            runner: 按能力执行独立子任务的循环。
            llm: 协调决策使用的统一模型调用层。
            max_rounds: 每条用户消息允许的最多协调决策轮数，须为正整数。
            model_tier: 协调者及默认确认判定器使用的模型档位。
            judge: 独立确认判定器；None 时使用协调者的模型调用层构造。
            data_max_chars: 轮初业务数据区的 JSON 字符上限，最少容纳空列表。

        Raises:
            ValueError: max_rounds 不是正整数。
        """
        if type(max_rounds) is not int or max_rounds < 1:
            raise ValueError("协调决策轮数上限必须为正整数")
        if type(data_max_chars) is not int or data_max_chars < 2:
            raise ValueError("业务数据字符上限必须为不小于 2 的整数")
        self.gateway, self.runner, self.llm = gateway, runner, llm
        self.max_rounds, self.model_tier = max_rounds, model_tier
        self.judge = judge or ConfirmationJudge(llm, model_tier=model_tier)
        self.data_max_chars = data_max_chars

    async def handle_turn(self, state: ConversationState, user_message: str) -> dict:
        """在内存中处理一条真实用户消息，复用持久运行时的逐轮执行接口。

        Args:
            state: 当前聊天状态，原地更新；此入口不保存检查点。
            user_message: 调用入口收到的真实用户原话，不得替换成模型摘要。

        Returns:
            含 reply、tasks、displayed、confirmation_results、rounds 的回复字典。

        Raises:
            ValueError: 用户消息为空。
            asyncio.CancelledError: 本轮调用被取消。
            OSError: 业务或轨迹读写失败；保留已知操作引用供对账。
        """
        with self.gateway.trace.span("turn", "处理用户消息", turn=state.turn + 1):
            with self.gateway.trace.span("coordinator", "任务协调"):
                work = self.begin_turn(state, user_message)
                while work.result is None:
                    await self.advance_turn(state, work)
                return work.result

    def begin_turn(self, state: ConversationState, user_message: str) -> TurnState:
        """建立本轮进度并对账已有操作，不分派子 Agent。

        Args:
            state: 本轮会话与任务状态，原地递增轮次并记录用户原话。
            user_message: 当前入口收到的非空用户消息。

        Returns:
            新的协调进度；持久运行时在进入首个决策节点前保存该进度。

        Raises:
            ValueError: 用户消息不是非空字符串。
            BusinessError: 已有操作无法按当前身份对账。
            OSError: 对账或轨迹读写失败。
        """
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("用户消息不能为空")
        state.turn += 1
        previous = deepcopy(state.last_display)
        previous_reply = state.last_reply
        state.last_display = []
        state.last_reply = ""
        state.history.append({"role": "user", "content": user_message})
        instance_id = uuid4().hex
        record_offset = len(self.gateway.records)
        self._reconcile(state, CallContext(state.session, "coordinator", instance_id))
        self._remember_business(state, self.gateway.records[record_offset:])
        messages = self._messages(state)
        return TurnState(
            user_message,
            instance_id,
            previous,
            messages,
            len(messages),
            previous_reply=previous_reply,
            business_snapshot=deepcopy(list(state.business_data.values())),
        )

    async def advance_turn(
        self, state: ConversationState, work: TurnState, *, messages: list[dict] | None = None
    ) -> None:
        """执行一次协调决策；任务更新与后续分派分属不同调用。

        Args:
            state: 当前会话与任务状态，工具执行结果写回此对象。
            work: 原地更新的本轮进度；产生最终回复时写入 result。
            messages: 上下文编译器本次生成的模型输入；None 时给原始往返注入最新状态。
                确认判定始终使用 work.user_message 与 work.previous。

        Raises:
            ValueError: 本轮已结束，不能继续推进。
            asyncio.CancelledError: 调用被取消，实例收尾后向上传播。
            OSError: 业务或轨迹读写失败；图运行时保留上一个成功节点的检查点。
        """
        if work.result is not None:
            raise ValueError("本轮已结束")
        if work.rounds >= self.max_rounds:
            self.stop_turn(state, work, f"协调者达到本轮上限 {self.max_rounds}，未完成本轮处理")
            return
        context = CallContext(state.session, "coordinator", work.instance_id)
        work.rounds += 1
        # 本次响应只能使用发出请求时的权限，身份查询不能给同一批后续调用提权。
        business_tools = {schema["function"]["name"] for schema in self.gateway.schemas(context)}
        try:
            response = await self.llm.complete(
                self.model_tier,
                self._model_messages(state, work) if messages is None else messages,
                tools=self._schemas(state, work),
            )
        except (LLMError, TimeoutError) as exc:
            self.stop_turn(state, work, f"协调模型调用失败：{type(exc).__name__}")
            return
        calls = response.message.get("tool_calls") or []
        text = response.message.get("content") or ""
        if not calls and text.strip() and self._can_reply(state, work):
            # 模型常以纯文本回复用户；仅在 reply 工具同样会被接受时按 reply 处理。
            if state.session.user_id is not None:
                self._reconcile(state, context)
            if state.session.user_id is None or not self._actionable(state):
                work.messages.append(deepcopy(response.message))
                work.result = self._reply(
                    state, context, Reply(message=text), work.decisions, work.rounds
                )
                if response.message["content"] == work.result["reply"]:
                    state.history[-1] = deepcopy(response.message)
                return
        readonly_batch = calls and all(
            call["function"]["name"] in READ_TOOLS | IDENTITY_TOOLS for call in calls
        )
        if not calls or (len(calls) > 1 and not readonly_batch):
            work.invalid_outputs.append(deepcopy(response.message))
            self._guard(
                "coordinator_format",
                "多个调用必须全部是只读业务工具；协调工具每次只能调用一个",
            )
            work.messages.append(
                {
                    "role": "user",
                    "content": "可一次调用多个只读业务工具；协调工具每次只调用一个。"
                    "批量任务更新用 update_tasks.changes，批量分派用 task_ids。",
                }
            )
        else:
            work.messages.append(deepcopy(response.message))
            for call in calls:
                record_offset = len(self.gateway.records)
                try:
                    arguments = json.loads(call["function"]["arguments"])
                    if not isinstance(arguments, dict):
                        raise ValueError("工具参数必须是对象")
                    outcome = await self._execute_call(
                        state, work, context, call["function"]["name"], arguments, business_tools
                    )
                except ValueError as exc:
                    self._guard("coordinator_tool", str(exc))
                    outcome = {
                        "error": getattr(exc, "code", "invalid_tool_call"),
                        "reason": str(exc),
                    }
                records = self.gateway.records[record_offset:]
                self._remember_business(state, records)
                if call["function"]["name"] in {"complete_plan", "dispatch_tasks"}:
                    delta = business_updates(
                        records,
                        user_id=state.session.user_id,
                        session_id=state.session.session_id,
                        turn=state.turn,
                    )
                    if delta:
                        outcome["business_data"] = list(delta.values())
                work.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(outcome, ensure_ascii=False),
                    }
                )
        if work.result is None and work.rounds == self.max_rounds:
            self.stop_turn(state, work, f"协调者达到本轮上限 {self.max_rounds}，未完成本轮处理")

    async def _execute_call(
        self,
        state: ConversationState,
        work: TurnState,
        context: CallContext,
        name: str,
        arguments: dict,
        business_tools: set[str],
    ) -> dict | list[dict] | str:
        """执行一个已通过批次形状检查的调用，任务与确认护栏仍逐项验证。

        Args:
            state: 当前会话与可信任务状态，按调用结果更新。
            work: 本轮进度，保存确认结果及最终回复。
            context: 本轮协调者的代码调用身份。
            name: 模型选出的单个工具名称。
            arguments: 已解析为对象的工具参数。
            business_tools: 本次模型请求发出时可用的业务工具名称。

        Returns:
            要与当前 tool_call_id 配对保存的业务或协调结果。

        Raises:
            ValueError: 参数、阶段、身份或业务护栏拒绝此次调用。
        """
        if name == "update_tasks":
            self._require_identity(state)
            updates = UpdateTasks.model_validate(arguments)
            for change in updates.changes:
                if change.capability is not None:
                    self.runner.registry.get(change.capability)
            return state.tasks.apply(updates.changes, self.gateway, context)
        if name == "complete_plan":
            self._require_identity(state)
            plan = CompletePlan.model_validate(arguments)
            if work.confirmed:
                raise ValueError("本轮已经判定过用户确认")
            work.decisions = await self._confirm(
                state, context, work.user_message, work.previous, work.previous_reply
            )
            work.confirmed = True
            selected, blocked = self._plan_candidates(state, plan.task_ids)
            dispatched = (
                await self._dispatch(state, context, selected)
                if selected
                else {"dispatched": [], "blocked": []}
            )
            return {
                "confirmation_results": self._confirmation_view(work.decisions),
                "dispatched": dispatched["dispatched"],
                "blocked": blocked + dispatched["blocked"],
            }
        if name == "dispatch_tasks":
            self._require_identity(state)
            if self._requires_plan(state, work):
                raise ValueError("先提出所有任务修改并调用 complete_plan，再分派")
            selection = DispatchTasks.model_validate(arguments)
            return await self._dispatch(state, context, selection.task_ids)
        if name == "reply":
            proposal = Reply.model_validate(arguments)
            if not self._can_reply(state, work):
                raise ValueError("先提出所有任务修改并调用 complete_plan，再回复")
            self._reconcile(state, context)
            unfinished = self._actionable(state)
            if unfinished:
                raise ValueError(
                    f"任务仍需推进或判断条件：{unfinished}；先分派、补充缺失信息或撤销不再需要的条件分支"
                )
            work.result = self._reply(state, context, proposal, work.decisions, work.rounds)
            return {"replied": True}
        if name in business_tools:
            return self.gateway.call(context, name, arguments)
        raise ValueError(f"当前阶段没有工具：{name}")

    def stop_turn(self, state: ConversationState, work: TurnState, reason: str) -> None:
        """说明本轮无法继续的原因，终止可推进任务并保留已等待用户的结果。

        Args:
            state: 要原地更新的会话与任务状态。
            work: 本轮进度，最终回复写入 result。
            reason: 代码确定的超限、上下文或模型调用失败原因。
        """
        context = CallContext(state.session, "coordinator", work.instance_id)
        self._fail_unfinished(state, context, reason)
        work.result = self._reply(
            state,
            context,
            Reply(message=STOP_NOTICE),
            work.decisions,
            work.rounds,
            notice=STOP_NOTICE,
        )

    def _messages(self, state: ConversationState) -> list[dict]:
        """组装固定政策和原始历史，权威状态在每次模型请求前单独注入。

        Args:
            state: 当前会话的真实消息和代码任务状态。

        Returns:
            新的 Chat Completions 消息列表；旧对话仅供理解，不作为当前授权。
        """
        prompt = (
            "你是与用户多轮交互的电商客服主 Agent，负责理解全部诉求、查询、解释、追问和维护任务。"
            "可以一次调用多个只读业务工具，代码按顺序执行；含协调工具时只能调用一个，"
            "批量更新或分派使用该工具的列表参数。"
            "未验证身份时先查找用户，缺少邮箱或姓名+邮编时直接用 reply 询问；用户只能服务一个账户。"
            "身份验证后可直接读取用户资料、订单详情和商品规格。订单不明时先自行查询定位；"
            "只有多张订单都符合且确实无法区分时才询问用户，不创建没有 order_id 的订单写任务。"
            "写操作必须由子 Agent 加载 Skill 后准备，协调者只能读取业务数据。"
            "上一回复展示过且仍有效的待确认操作存在时，先用 update_tasks 提出本轮全部新增、"
            "改口和撤销，再用 complete_plan 让代码独立判定用户原话、提交并核验。"
            "没有这样的待确认操作时，可直接 reply 对话或用 dispatch_tasks 分派，"
            "不需要空调 complete_plan；仍有可推进任务时不能提前回复。"
            "complete_plan(task_ids=[...]) 可在一次决策中结束计划并分派已有任务。"
            "其中不存在、重复或状态不符的分派项会单独说明，不影响当前原话的确认判定。"
            "task_ids 不表示用户授权，新准备项须展示后等待确认。"
            "用户改口时必须在 complete_plan 之前更新旧任务，不能只据‘可以’提交。"
            "条件保持自然语言，由你判断；条件满足时明确分派该任务，不再需要则撤销并说明原因。"
            "写任务按订单×动作类型聚合完整商品清单；同单不同动作逐个确认，先地址/支付后商品。"
            "每个任务提供完整 instruction、capability、action，订单动作还要 order_id。"
            "capability 从下方能力目录选择，action 使用该能力业务工具的完整名称。"
            "模型只提出任务增改撤与missing；子Agent的note是给你的备注，不是直接回复用户的正文。"
            "任何status、revision或确认字段均由代码维护。"
            "不能声明写任务完成，不能提交操作，用户确认只由独立判定器读取当前用户原话。"
            "reply 只能在所有任务完成、撤销、等待用户或失败后调用。"
            "show_task_ids为要展示待确认详情的任务，省略展示全部，空列表不展示。"
            "协调工具返回变化摘要；当前完整任务清单只看本次请求附带的权威状态。"
            "任务区不含事实和动作历史；业务明细优先看本轮工具结果，其次看轮初业务参考数据。"
            "本轮子Agent查询和核验结果在协调工具返回的business_data中；它们优先于轮初旧数据。"
            "数据标记已变更或已移出时须重新查询，不能用旧值推断。"
            "reply.message 是原样发送给用户的非空正文；代码只在正文之后附加所选操作的确认卡片。"
            "请用自然语言回答问题、解释、追问和汇总办理结果，不贴JSON、内部标识或任务字段名。"
            "不另造卡片中的办理参数或金额；引用卡片金额解释用户疑问时保持一致。"
            "写任务核验完成之前不能说已取消、已退款、已退货、已换货、已修改或已提交；"
            "done只表示该动作已核验，退货申请完成不表示已收到退货或退款到账。"
            "撤销、失败等变化只在发生的本轮说明；随后用户追问时按实际状态回答。\n"
            + self.gateway.skills.common_policy
            + "\n可分派能力：\n"
            + json.dumps(self.runner.registry.catalog(), ensure_ascii=False)
        )
        return [{"role": "system", "content": prompt}, *deepcopy(state.history)]

    @staticmethod
    def _model_messages(state: ConversationState, work: TurnState) -> list[dict]:
        """给内存入口的原始消息临时注入一份最新状态，不修改归档消息下标。

        Args:
            state: 当前身份与完整可信任务，投影只用于模型输入。
            work: 初始历史与本轮真实往返，base_messages 在整个轮次保持稳定。

        Returns:
            在当前用户原话之前插入一份精简状态的新消息列表。
        """
        snapshot = {
            "role": "system",
            "content": STATE_NOTICE
            + json.dumps(
                {
                    "tasks": [task_view(task) for task in state.tasks.snapshot()],
                    "verified": state.session.user_id is not None,
                },
                ensure_ascii=False,
            ),
        }
        boundary = work.base_messages - 1
        data = {
            "role": "system",
            "content": DATA_NOTICE
            + json.dumps(work.business_snapshot, ensure_ascii=False, separators=(",", ":")),
        }
        return [*work.messages[:boundary], data, snapshot, *work.messages[boundary:]]

    def _remember_business(self, state: ConversationState, records: list[dict]) -> None:
        """将本次真实查询并入持久参考缓存，轮初模型视图在本轮保持不变。

        Args:
            state: 当前会话与参考缓存，身份未验证时不写入业务数据。
            records: 本次网关新增记录，包含子 Agent、协调者及代码核验调用。
        """
        if state.session.user_id is None:
            return
        updates = business_updates(
            records,
            user_id=state.session.user_id,
            session_id=state.session.session_id,
            turn=state.turn,
        )
        for key, entry in updates.items():
            state.business_data.pop(key, None)
            state.business_data[key] = entry
        limit_business_data(state.business_data, self.data_max_chars)

    @staticmethod
    def _confirmation_view(decisions: list[dict]) -> list[dict]:
        """精简模型可见的逐操作判定，完整核验证据仍保存在本轮状态中。

        Args:
            decisions: 含完整核验 expected、actual 和 operation 的内部确认结果。

        Returns:
            操作标识、判断、执行状态、原因与核验结论，不包含完整业务对象。
        """
        result = []
        for decision in decisions:
            item = {
                key: deepcopy(decision[key])
                for key in ("operation_id", "decision", "status", "reason", "error")
                if key in decision
            }
            if "verification" in decision:
                item["verification"] = {
                    key: deepcopy(decision["verification"][key])
                    for key in ("verified", "reason", "checked_fields")
                    if key in decision["verification"]
                }
            result.append(item)
        return result

    def _schemas(self, state: ConversationState, work: TurnState) -> list[dict]:
        """按身份和有效待确认展示开放工具，不开放提交与核验。

        Args:
            state: 当前会话与可信任务状态。
            work: 本轮独立确认进度和上一回复的实际展示快照。

        Returns:
            当前可见的业务与协调函数工具定义。
        """
        context = CallContext(state.session, "coordinator", work.instance_id)
        models = {"reply": Reply}
        if state.session.user_id is not None:
            models = {"update_tasks": UpdateTasks}
            if not work.confirmed:
                models["complete_plan"] = CompletePlan
            if not self._requires_plan(state, work):
                models.update(reply=Reply, dispatch_tasks=DispatchTasks)
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

    def _requires_plan(self, state: ConversationState, work: TurnState) -> bool:
        """仅对上一回复展示过且仍有效的待确认操作要求先处理本轮计划。"""
        if work.confirmed or state.session.user_id is None:
            return False
        context = CallContext(state.session, "coordinator", work.instance_id)
        return bool(self._eligible_confirmations(state, context, work.previous))

    def _can_reply(self, state: ConversationState, work: TurnState) -> bool:
        """允许身份追问或无有效旧授权时对话，任务可推进性由回复入口另行检查。"""
        return not self._requires_plan(state, work)

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
            task.clear_terminal_fields()
            if task.status not in {"pending_confirm", "needs_verification", "failed"}:
                continue
            task_context = replace(context, task_id=task.id, revision=task.revision)
            for operation_id in list(task.operation_ids):
                operation = self.gateway.get_operation(task_context, operation_id)
                if task.status == "failed" and operation["status"] != "committed":
                    if operation["status"] == "prepared":
                        self.gateway.reject(task_context, operation_id, task.reason or "任务已失败")
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
                    task.reason_internal = False

    def _eligible_confirmations(
        self, state: ConversationState, context: CallContext, previous: list[dict]
    ) -> dict[str, tuple[Task, dict]]:
        """选出上一回复中仍可由当前用户原话确认的操作。

        Args:
            state: 当前任务修订、状态和会话轮次。
            context: 查询后端操作时使用的已验证身份。
            previous: 上一回复实际展示的完整操作及参数快照。

        Returns:
            操作 ID 到任务及展示快照的映射，过期、改口及未展示项不在其中。
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
            task_context = replace(context, task_id=task.id, revision=task.revision)
            operation = self.gateway.get_operation(task_context, shown["operation_id"])
            if (
                operation["status"] == "prepared"
                and operation["args"] == shown["args"]
                and operation["preview"] == shown["preview"]
            ):
                eligible[operation["operation_id"]] = (task, shown)
        return eligible

    async def _confirm(
        self,
        state: ConversationState,
        context: CallContext,
        user_message: str,
        previous: list[dict],
        previous_reply: str | None = None,
    ) -> list[dict]:
        """只判定前一条实际回复中仍与当前任务修订一致的操作。

        Args:
            state: 已应用本轮任务修改的状态。
            context: 代码提交、拒绝和核验使用的身份。
            user_message: 当前用户原话。
            previous: 上次成功发出的回复的展示快照。
            previous_reply: 上次实际回复正文与卡片；None 用于独立判定测试入口。

        Returns:
            按操作列出的判定、执行状态及核验结果，部分失败不覆盖其他结果。
        """
        eligible = self._eligible_confirmations(state, context, previous)
        if not eligible:
            return []
        # 保留实际展示范围，避免撤销一项后把多项短回复误判为单项确认。
        judgments = await self.judge.judge(user_message, previous, assistant_reply=previous_reply)
        results = []
        for judgment in judgments:
            if judgment["operation_id"] not in eligible:
                continue
            task, shown = eligible[judgment["operation_id"]]
            task_context = replace(context, task_id=task.id, revision=task.revision)
            entry = dict(judgment)
            try:
                if judgment["decision"] == "confirm":
                    task.status = "needs_verification"
                    task.reason = "正在提交并核验；结果未知时先对账"
                    task.reason_internal = False
                    operation = self.gateway.commit(
                        task_context, shown["operation_id"], shown["args"]
                    )
                    task.actions.append({"kind": "commit", "result": deepcopy(operation)})
                    entry["verification"] = self._verify(task, context, operation)
                elif judgment["decision"] in {"reject", "modify"}:
                    operation = self.gateway.reject(
                        task_context, shown["operation_id"], judgment["reason"]
                    )
                    task.actions.append({"kind": "reject", "result": deepcopy(operation)})
                    task.operation_ids = []
                    task.reason = judgment["reason"]
                    task.reason_internal = False
                    if judgment["decision"] == "modify":
                        task.revision += 1
                        task.status = "waiting_info"
                        task.missing = ["请补充修改后的完整要求"]
                    else:
                        task.status = "cancelled"
                entry["status"] = task.status
            except BusinessError as exc:
                operation = self.gateway.get_operation(task_context, shown["operation_id"])
                if operation["status"] == "committed":
                    task.status = "needs_verification"
                else:
                    if operation["status"] == "prepared":
                        self.gateway.reject(task_context, operation["operation_id"], str(exc))
                    task.status = "failed"
                task.reason = str(exc)
                task.reason_internal = False
                entry.update(status=task.status, error=exc.code, reason=str(exc))
            task.clear_terminal_fields()
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
        task_context = replace(context, task_id=task.id, revision=task.revision)
        verification = self.gateway.verify_operation(task_context, operation["operation_id"])
        task.facts.append({"verification": deepcopy(verification)})
        task.status = "done" if verification["verified"] else "failed"
        task.reason = None if verification["verified"] else verification["reason"]
        task.reason_internal = False
        task.clear_terminal_fields()
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
            dispatched 的任务状态摘要和 blocked 的占用原因。

        Raises:
            ValueError: 任务不存在、重复，或任务已结束、已经等待确认。
            asyncio.CancelledError: 调用被取消；等待并行实例退出后向上传播。
            OSError: 业务或轨迹读写失败；先保留操作引用并停止并行实例。
        """
        tasks = self._dispatch_candidates(state, task_ids)
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
                    task.reason_internal = True
                    task.clear_terminal_fields()
                    self._interrupted_records(state, task)
            raise
        return {"dispatched": [task_brief(task.to_dict()) for task in selected], "blocked": blocked}

    @staticmethod
    def _plan_candidates(
        state: ConversationState, task_ids: list[str]
    ) -> tuple[list[str], list[dict]]:
        """在确认已处理后逐项筛选合并分派选择，非法项不影响其他任务。

        Args:
            state: 确认、提交和核验后的可信任务状态。
            task_ids: 结构已校验的模型分派列表；首次合法出现最多执行一次。

        Returns:
            可分派任务 ID 列表和每项被跳过的状态、原因；同单锁另由分派入口检查。
        """
        selected, blocked, seen = [], [], set()
        for task_id in task_ids:
            task = state.tasks.tasks.get(task_id)
            reason = None
            if task_id in seen:
                reason = "分派任务 ID 重复，本次不重复执行"
            elif task is None:
                reason = "分派任务不存在"
            elif task.status not in {"pending", "conditional"}:
                reason = "任务状态不允许分派；只可分派待执行或条件已满足的任务"
            seen.add(task_id)
            if reason is not None:
                blocked.append(
                    {
                        "task_id": task_id,
                        "status": None if task is None else task.status,
                        "reason": reason,
                    }
                )
            else:
                selected.append(task_id)
        return selected, blocked

    @staticmethod
    def _dispatch_candidates(state: ConversationState, task_ids: list[str]) -> list[Task]:
        """校验分派选择，不改变任务状态或触发业务操作。

        Args:
            state: 提供任务身份和当前状态的可信清单。
            task_ids: 由工具参数校验后的任务 ID；空列表表示不分派。

        Returns:
            按选择顺序排列的任务；订单占用在实际分派时检查。

        Raises:
            ValueError: ID 重复、不存在，或任务不处于待执行 / 条件状态。
        """
        if len(set(task_ids)) != len(task_ids) or any(
            key not in state.tasks.tasks for key in task_ids
        ):
            raise ValueError("分派任务不存在或重复")
        tasks = [state.tasks.tasks[key] for key in task_ids]
        if any(task.status not in {"pending", "conditional"} for task in tasks):
            raise ValueError("只可分派待执行或条件已满足的任务；缺信息任务先更新完整要求")
        return tasks

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
                    f"{task.instruction}\n本次任务动作：{task.action}；目标订单：{task.order_id}。只准备这一项完整请求。"
                    f"\n用户身份已由协调者验证为 {state.session.user_id}，不要再要求用户验证身份。",
                    task.revision,
                    task.is_write,
                    deepcopy(task.facts),
                ),
            )
            self._merge_result(context, task, result)
        except (LLMError, TimeoutError, ValueError) as exc:
            task.status, task.reason = "failed", f"子 Agent 调用失败：{type(exc).__name__}"
            task.reason_internal = True
            task.clear_terminal_fields()
            self._interrupted_records(state, task)
            self._discard_prepared(context, task)
            return
        except (asyncio.CancelledError, OSError) as exc:
            task.status, task.reason = "failed", f"子 Agent 中断：{type(exc).__name__}"
            task.reason_internal = True
            task.clear_terminal_fields()
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
        context = replace(context, task_id=task.id, revision=task.revision)
        task.actions.extend(deepcopy(result["actions"]))
        task.facts.extend(deepcopy(result["facts"]))
        task.missing, task.note = result["missing"], result["note"]
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
            task.reason_internal = True
            task.clear_terminal_fields()
            self._guard("dispatch_result", task.reason)
            return
        task.operation_ids = operation_ids
        task.reason = result["reason"]
        task.reason_internal = task.reason is not None
        task.status = {
            "need_confirm": "pending_confirm",
            "need_info": "waiting_info",
            "ready": "done",
        }.get(result["status"], result["status"])
        if task.is_write and task.status == "done":
            task.status, task.reason = "failed", "写任务必须提交后核验"
            task.reason_internal = True
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
            task.reason_internal = True
            self._guard("dispatch_read_result", task.reason)
        for operation in operations:
            if operation["status"] == "committed":
                self._verify(task, context, operation)

        task.clear_terminal_fields()

    def _discard_prepared(self, context: CallContext, task: Task) -> None:
        """清理中断任务的准备项，已提交操作保留为待核验。

        Args:
            context: 读取与拒绝操作使用的代码身份。
            task: 已保存操作引用的中断任务；清理读写失败时仍保留引用。

        Raises:
            BusinessError: 操作不存在或身份归属不符。
            OSError: 清理仍无法读写；下一轮按保存的引用继续对账。
        """
        context = replace(context, task_id=task.id, revision=task.revision)
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
            task_context = replace(context, task_id=task.id, revision=task.revision)
            for operation_id in task.operation_ids:
                operation = self.gateway.get_operation(task_context, operation_id)
                if operation["status"] == "prepared":
                    self.gateway.reject(task_context, operation_id, reason)
                elif operation["status"] == "committed":
                    self._verify(task, context, operation)
            if task.status != "done":
                task.status, task.reason = "failed", reason
                task.reason_internal = True
            task.clear_terminal_fields()

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
        """原样发送模型正文并附代码生成的确认卡片，记录成功后保存实际回复。

        Args:
            state: 当前任务及消息状态。
            context: 刷新展示操作时使用的身份。
            proposal: 模型正文和希望展示的任务 ID，正文不能替代卡片或执行事实。
            decisions: 本轮逐操作确认与执行结果。
            rounds: 本轮已经使用的协调决策轮数。
            notice: 停止本轮时使用的固定说明；None 表示发送模型正文。

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
        cards, displayed = [], []
        for task in state.tasks.tasks.values():
            if task.status != "pending_confirm" or (
                selected is not None and task.id not in selected
            ):
                continue
            for operation_id in task.operation_ids:
                task_context = replace(context, task_id=task.id, revision=task.revision)
                operation = self.gateway.get_operation(task_context, operation_id)
                if operation["status"] != "prepared":
                    continue
                content = self.gateway.operation_content(task_context, operation)
                cards.append(content)
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
        message = proposal.message
        if notice is not None:
            message = PREPARED_STOP_NOTICE if displayed else notice
        else:
            self._record_reply_claims(state, message)
        parts = [message, *cards]
        if displayed:
            parts.append("请确认以上操作；多项可指明具体操作，或明确回复“全部确认”。")
        reply = "\n\n".join(parts)
        with self.gateway.trace.span(
            "reply", "回复用户", attrs={"content": reply, "displayed": displayed}
        ):
            pass
        with self.gateway.trace.span(
            "state_snapshot", "本轮任务状态", attrs={"tasks": state.tasks.snapshot()}
        ):
            pass
        state.last_display = deepcopy(displayed)
        state.last_reply = reply
        state.history.append({"role": "assistant", "content": reply})
        return {
            "reply": reply,
            "tasks": state.tasks.snapshot(),
            "displayed": deepcopy(displayed),
            "confirmation_results": decisions,
            "rounds": rounds,
        }

    def _record_reply_claims(self, state: ConversationState, message: str) -> None:
        """记录正文中的完成措辞及核验状态，供人工分析，不改写或阻断回复。

        Args:
            state: 本轮可信任务状态，保留动作、目标与最近核验结果作为观察依据。
            message: 模型原始回复正文；不扫描代码生成的确认卡片。
        """
        matched = [word for word in COMPLETION_CLAIMS if word in message]
        if not matched:
            return
        tasks = []
        for task in state.tasks.tasks.values():
            if not task.is_write:
                continue
            verification = next(
                (fact["verification"] for fact in reversed(task.facts) if "verification" in fact),
                {},
            )
            tasks.append(
                {
                    "task_id": task.id,
                    "action": task.action,
                    "order_id": task.order_id,
                    "status": task.status,
                    "verification": {
                        key: deepcopy(verification[key])
                        for key in ("operation_id", "verified", "reason", "checked_fields")
                        if key in verification
                    },
                }
            )
        with self.gateway.trace.span(
            "reply_claim",
            "回复完成表述观察",
            attrs={"message": message, "matched": matched, "tasks": tasks},
        ):
            pass

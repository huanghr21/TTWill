"""以 LangGraph 检查点保存协调进度，重新验证身份后恢复同一业务任务。"""

import asyncio
from copy import deepcopy
from dataclasses import asdict
from typing import TypedDict
from uuid import uuid4
from weakref import WeakValueDictionary

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt

from ttwill.core.context import ContextBudgetExceeded, ContextCompiler, context_chars
from ttwill.core.coordinator import ConversationState, Coordinator, TurnState
from ttwill.core.recovery import reconcile_restored_tasks
from ttwill.core.tasks import Task
from ttwill.gateway import IDENTITY_TOOLS, BusinessError, CallContext, SessionContext
from ttwill.llm import LLMError
from ttwill.persistence import Archive

_thread_locks: WeakValueDictionary[tuple[str, str], asyncio.Lock] = WeakValueDictionary()


class RuntimeState(TypedDict):
    """描述检查点中的唯一任务状态及当前节点进度，全部字段使用 JSON 数据。"""

    owner_id: str
    database_path: str
    opening_id: str
    conversation: dict
    work: dict | None
    request: str | None
    result: dict | None


class PersistentRuntime:
    """将任务更新与后续分派分隔为图节点，在等待用户时暂停。"""

    def __init__(
        self,
        coordinator: Coordinator,
        checkpointer: AsyncSqliteSaver,
        archive: Archive,
        *,
        thread_id: str,
        compiler: ContextCompiler,
    ) -> None:
        """绑定图、检查点和原始对话档案，不在验证身份前读取历史数据。

        Args:
            coordinator: 复用任务管理、确认、分派和核验的协调者。
            checkpointer: 调用方管理生命周期的原生异步 SQLite checkpointer。
            archive: 完整消息与摘要档案，不能用于还原任务状态。
            thread_id: 本次业务任务的稳定标识；不同评测任务或重复运行使用不同范围。
            compiler: 每次协调决策使用的上下文预算与摘要编译器。

        Raises:
            ValueError: thread_id 为空，或编译器使用了另一份档案。
        """
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("任务线程标识不能为空")
        if compiler.archive is not archive:
            raise ValueError("运行时与上下文编译器必须使用同一档案")
        self.coordinator, self.archive, self.compiler = coordinator, archive, compiler
        self.thread_id = thread_id
        self._session: SessionContext | None = None
        self._state: ConversationState | None = None
        self._intake = ConversationState(SessionContext(coordinator.gateway.trace.session_id))
        self._intake_turns: list[TurnState] = []
        lock_key = (str(coordinator.gateway.database_path), thread_id)
        self._lock = _thread_locks.setdefault(lock_key, asyncio.Lock())
        self._config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": coordinator.max_rounds + 8,
        }
        builder = StateGraph(RuntimeState)
        builder.add_node("restore", self._restore)
        builder.add_node("wait", self._wait)
        builder.add_node("begin", self._begin)
        builder.add_node("advance", self._advance)
        builder.add_conditional_edges(START, self._start_node)
        builder.add_edge("restore", "wait")
        builder.add_edge("wait", "begin")
        builder.add_edge("begin", "advance")
        builder.add_conditional_edges("advance", self._next_node)
        self.graph = builder.compile(checkpointer=checkpointer)

    @property
    def state(self) -> ConversationState:
        """返回本实例最近载入的状态；只有成功验证并打开会话后可读取。

        Returns:
            当前节点使用的内存对象；持久状态仍以图检查点为准。

        Raises:
            BusinessError: 尚未打开已验证会话。
        """
        self._require_session()
        if self._state is None:
            raise BusinessError("会话尚未完成恢复", code="session_not_open")
        return self._state

    async def open_session(self, identity_tool: str, identity_args: dict) -> dict:
        """重新验证身份，再打开原业务库中的进度并清除旧聊天授权。

        Args:
            identity_tool: 网关提供的邮箱或姓名与邮编身份查找工具。
            identity_args: 本次用户重新提供的身份验证参数，不接受 verified 或 user_id。

        Returns:
            含 thread_id、session_id、tasks 和 resumed 的状态概览。

        Raises:
            BusinessError: 身份失败、用户不匹配、业务库不匹配或会话已打开。
            OSError: 业务或轨迹读写失败，原检查点保留供后续恢复。
        """
        async with self._lock:
            if self._session is not None:
                raise BusinessError("此实例已打开会话", code="session_already_open")
            if identity_tool not in IDENTITY_TOOLS:
                raise BusinessError("恢复前须使用身份查找工具", code="identity_required")
            gateway = self.coordinator.gateway
            session = SessionContext(gateway.trace.session_id)
            context = CallContext(session, "coordinator", uuid4().hex)
            gateway.call(context, identity_tool, identity_args)
            if session.user_id is None:
                raise BusinessError("未查到唯一用户，不能打开任务", code="identity_required")
            # 身份验证之前不得读取检查点、摘要或完整对话。
            saved = await self.graph.aget_state(self._config)
            values = saved.values
            if values and values["owner_id"] != session.user_id:
                raise BusinessError("任务检查点不属于当前用户", code="checkpoint_owner_mismatch")
            database_path = str(gateway.database_path)
            if values and values["database_path"] != database_path:
                raise BusinessError("续办须使用任务原业务库", code="checkpoint_database_mismatch")
            state = (
                _conversation(values["conversation"], session)
                if values
                else ConversationState(session)
            )
            self._session = session
            self._state = state
            initial: RuntimeState = {
                "owner_id": session.user_id,
                "database_path": database_path,
                "opening_id": uuid4().hex,
                "conversation": _snapshot(state),
                "work": None,
                "request": None,
                "result": None,
            }
            opened = False
            try:
                await self.graph.ainvoke(initial, self._config, durability="sync")
                opened = True
            finally:
                if not opened:
                    self._session = None
                    self._state = None
            return {
                "thread_id": self.thread_id,
                "session_id": session.session_id,
                "tasks": self.state.tasks.snapshot(),
                "resumed": bool(values),
            }

    async def receive_message(self, user_message: str) -> dict:
        """从自然对话验证身份，再在同一轮接入持久化任务处理。

        Args:
            user_message: 当前用户原话；邮箱或姓名与邮编由协调模型从对话提取。

        Returns:
            含 reply、tasks、displayed、confirmation_results 和 rounds 的回复。
            身份确认前只能查找用户或回复，不能创建任务，也不读取历史检查点和档案。

        Raises:
            ValueError: 用户消息为空。
            BusinessError: 验证后的用户或业务库与原检查点不符，或会话已被接续。
            asyncio.CancelledError: 调用被取消；认证接管失败后须重新验证身份。
            OSError: 轨迹、档案或检查点读写失败。
        """
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("用户消息不能为空")
        async with self._lock:
            if self._session is not None:
                return await self._send_message(user_message)
            state = self._intake
            trace = self.coordinator.gateway.trace
            with trace.span("turn", "自然对话身份接入", turn=state.turn + 1):
                with trace.span("coordinator", "身份接入协调", attrs={"thread_id": self.thread_id}):
                    work = self.coordinator.begin_turn(state, user_message)
                    self._intake_turns.append(work)
                    while work.result is None:
                        context = CallContext(state.session, "coordinator", work.instance_id)
                        tools = self.coordinator._schemas(context, work.confirmed)
                        if context_chars(work.messages, tools) > self.compiler.max_chars:
                            self.coordinator.stop_turn(
                                state, work, "身份验证前的对话超过上下文字符预算，请重新开始会话"
                            )
                        else:
                            try:
                                await self.coordinator.advance_turn(
                                    state, work, messages=work.messages
                                )
                            except (asyncio.CancelledError, OSError):
                                self._intake = ConversationState(SessionContext(trace.session_id))
                                self._intake_turns = []
                                raise
                        if state.session.user_id is not None:
                            return await self._activate_intake(work)
                    return deepcopy(work.result)

    async def _activate_intake(self, work: TurnState) -> dict:
        """在身份查询成功后接管当前轮次，任务创建前先保存图检查点。

        Args:
            work: 尚未分派业务任务的本轮进度，保留原话、工具往返和已用轮数。

        Returns:
            接管后推进到等待节点产生的最终回复。

        Raises:
            BusinessError: 当前身份或业务库与旧检查点不符。
            OSError: 恢复、归档或图推进失败；再次接入须重新提供身份。
        """
        session = self._intake.session
        activated = False
        try:
            saved = await self.graph.aget_state(self._config)
            values = saved.values
            if values and values["owner_id"] != session.user_id:
                raise BusinessError("任务检查点不属于当前用户", code="checkpoint_owner_mismatch")
            database_path = str(self.coordinator.gateway.database_path)
            if values and values["database_path"] != database_path:
                raise BusinessError("续办须使用任务原业务库", code="checkpoint_database_mismatch")
            state = (
                _conversation(values["conversation"], session)
                if values
                else ConversationState(session)
            )
            self._session = session
            initial: RuntimeState = {
                "owner_id": session.user_id,
                "database_path": database_path,
                "opening_id": uuid4().hex,
                "conversation": _snapshot(state),
                "work": None,
                "request": None,
                "result": None,
            }
            restored = self._restore(initial)
            state = _conversation(restored["conversation"], session)
            turn_offset = state.turn
            state.turn += len(self._intake_turns)
            self._state = state
            with self.coordinator.gateway.trace.span(
                "recovery",
                "认证后绑定对话轮次",
                turn=state.turn,
                attrs={
                    "opening_id": initial["opening_id"],
                    "turn_offset": turn_offset,
                    "turn_mapping": [
                        {
                            "instance_id": intake_work.instance_id,
                            "local_turn": index,
                            "persistent_turn": turn_offset + index,
                        }
                        for index, intake_work in enumerate(self._intake_turns, start=1)
                    ],
                },
            ):
                pass
            # 新聊天没有旧授权；本轮原话只可确认本聊天上一条实际展示。
            work.previous = []
            work.decisions = []
            if work.result is not None:
                self.coordinator.stop_turn(
                    state, work, f"协调者达到本轮上限 {self.coordinator.max_rounds}，未完成本轮处理"
                )
            for index, intake_work in enumerate(self._intake_turns, start=1):
                self._archive_intake(intake_work, turn_offset + index)
            current_messages = work.messages[work.base_messages :]
            work.messages = [
                work.messages[0],
                {"role": "user", "content": work.user_message},
                *current_messages,
            ]
            work.base_messages = 2
            initial.update(conversation=_snapshot(state), work=asdict(work), result=work.result)
            result = await self.graph.ainvoke(initial, self._config, durability="sync")
            self._state = _conversation(result["conversation"], session)
            activated = True
            return deepcopy(result["result"])
        finally:
            # 已验证但未接管成功的临时对象不能在下一轮绕过身份入口。
            self._intake = ConversationState(
                SessionContext(self.coordinator.gateway.trace.session_id)
            )
            self._intake_turns = []
            if not activated:
                self._session = None
                self._state = None

    def _archive_intake(self, work: TurnState, turn: int) -> None:
        """在身份归属确定后补存认证对话，不把临时任务快照写入历史。

        Args:
            work: 认证阶段某一轮的原话、真实工具往返和可选最终回复。
            turn: 以原检查点和档案最大轮次为偏移的全线程轮次。
        """
        session = self._require_session()
        messages = [
            (f"{work.instance_id}:user", {"role": "user", "content": work.user_message}),
            *[
                (f"{work.instance_id}:intake:{index}", message)
                for index, message in enumerate(work.messages[work.base_messages :])
            ],
        ]
        if work.result is not None:
            messages.append(
                (
                    f"{work.instance_id}:reply",
                    {"role": "assistant", "content": work.result["reply"]},
                )
            )
        self.archive.append_messages(self.thread_id, session.user_id, messages, turn=turn)
        for index, message in enumerate(work.invalid_outputs):
            self.archive.append(
                self.thread_id,
                session.user_id,
                f"{work.instance_id}:invalid:{index}",
                "event",
                {"event": "invalid_model_output", "message": message},
                turn=turn,
            )

    async def send_message(self, user_message: str) -> dict:
        """从等待节点接收真实用户消息，执行到下一次等待并返回已保存的回复。

        Args:
            user_message: 用户当前消息原文，不接受模型生成的确认标志。

        Returns:
            协调者生成的 reply、tasks、displayed、confirmation_results 和 rounds。

        Raises:
            ValueError: 用户消息为空。
            BusinessError: 尚未验证、已被另一会话接续，或上次执行中断须重新打开。
            asyncio.CancelledError: 调用被取消；再次运行须新建会话并重新验证。
            OSError: 持久化或业务读写失败。
        """
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("用户消息不能为空")
        async with self._lock:
            return await self._send_message(user_message)

    async def _send_message(self, user_message: str) -> dict:
        """在调用方已持有线程锁时推进已验证会话的一轮消息。

        Args:
            user_message: 已由公开入口校验的非空用户原话。

        Returns:
            下一等待节点保存的协调结果。
        """
        session = self._require_session()
        saved = await self.graph.aget_state(self._config)
        self._check_saved_session(saved.values)
        if saved.next != ("wait",) or not any(task.interrupts for task in saved.tasks):
            raise BusinessError("上次执行尚未完成，请重新验证身份后恢复", code="reopen_required")
        trace = self.coordinator.gateway.trace
        with trace.span(
            "turn", "持久化处理用户消息", turn=saved.values["conversation"]["turn"] + 1
        ):
            with trace.span("coordinator", "检查点协调", attrs={"thread_id": self.thread_id}):
                result = await self.graph.ainvoke(
                    Command(resume=user_message), self._config, durability="sync"
                )
        self._state = _conversation(result["conversation"], session)
        return deepcopy(result["result"])

    @staticmethod
    def _start_node(values: RuntimeState) -> str:
        """身份接管保留当前决策进度，显式打开会话则先执行恢复节点。"""
        if values.get("work") is None:
            return "restore"
        return "wait" if values.get("result") is not None else "advance"

    async def snapshot(self) -> ConversationState:
        """在当前会话仍有效时读取唯一的图检查点状态。

        Returns:
            与内部对象隔离的会话及任务状态，不从档案或轨迹重建。

        Raises:
            BusinessError: 尚未验证或任务已被另一会话接续。
        """
        async with self._lock:
            session = self._require_session()
            saved = await self.graph.aget_state(self._config)
            self._check_saved_session(saved.values)
            return _conversation(saved.values["conversation"], session)

    def _require_session(self) -> SessionContext:
        """取得已由本实例网关验证的会话，拒绝未验证访问。"""
        if self._session is None or self._session.user_id is None:
            raise BusinessError("请先验证身份并打开会话", code="identity_required")
        return self._session

    def _check_saved_session(self, values: dict) -> None:
        """检查当前会话仍拥有该任务的推进位置。

        Args:
            values: 从图检查点读取的当前状态。

        Raises:
            BusinessError: 已由另一会话接续，或身份不匹配。
        """
        session = self._require_session()
        if (
            not values
            or values["owner_id"] != session.user_id
            or values["conversation"]["session_id"] != session.session_id
        ):
            raise BusinessError("任务已由另一会话接续，请重新验证后打开", code="session_replaced")

    def _restore(self, values: RuntimeState) -> dict:
        """在独立节点对账已验证任务，恢复后始终先等待新的用户消息。

        Args:
            values: 已核对身份与业务库的检查点输入；opening_id 在本次打开尝试内
                固定，使节点重放复用同一档案事件，重新验证后的新尝试使用新标识。

        Returns:
            清除旧授权并核对后端结果的任务状态与空的本轮进度。
        """
        session = self._require_session()
        state = _conversation(values["conversation"], session)
        reconcile_restored_tasks(
            state, self.coordinator.gateway, CallContext(session, "coordinator", uuid4().hex)
        )
        checkpoint_turn = state.turn
        # 原话已归档而 begin 检查点未写时，保留消息编号单调，不重放该输入。
        state.turn = max(
            [state.turn]
            + [entry["turn"] for entry in self.archive.messages(self.thread_id, session.user_id)]
        )
        self._state = state
        self.archive.append(
            self.thread_id,
            session.user_id,
            f"{session.session_id}:opened:{values['opening_id']}",
            "event",
            {
                "event": "session_opened",
                "session_id": session.session_id,
                "opening_id": values["opening_id"],
                "checkpoint_turn": checkpoint_turn,
                "message_turn": state.turn,
            },
            turn=state.turn,
        )
        return {"conversation": _snapshot(state), "work": None, "request": None, "result": None}

    @staticmethod
    def _wait(values: RuntimeState) -> dict:
        """仅暂停并接收新消息，节点重放时不重做任何业务副作用。

        Args:
            values: 最近一个已持久化节点的状态。

        Returns:
            用户通过 Command(resume=...) 提交的真实消息。
        """
        message = interrupt({"status": "waiting_for_user", "result": values.get("result")})
        return {"request": message, "result": None, "work": None}

    def _begin(self, values: RuntimeState) -> dict:
        """建立本轮协调状态并完整归档真实用户消息。

        Args:
            values: 等待节点接收消息后的图状态。

        Returns:
            新轮次的会话快照与协调进度，后续决策在其检查点之后执行。
        """
        state = _conversation(values["conversation"], self._require_session())
        self._state = state
        work = self.coordinator.begin_turn(state, values["request"])
        self.archive.append_message(
            self.thread_id,
            state.session.user_id,
            f"{work.instance_id}:user",
            {"role": "user", "content": work.user_message},
            turn=state.turn,
        )
        return {"conversation": _snapshot(state), "work": asdict(work)}

    async def _advance(self, values: RuntimeState) -> dict:
        """编译上下文、执行一次协调决策并归档完整往返，返回新的检查点状态。

        Args:
            values: 上一个决策节点已保存的任务及本轮进度。

        Returns:
            更新后的任务、协调进度及可选最终回复。

        Raises:
            asyncio.CancelledError: 当前执行被取消。
            OSError: 业务或存储失败，任务恢复仍以上一检查点为准。
        """
        state = _conversation(values["conversation"], self._require_session())
        self._state = state
        work = TurnState(**deepcopy(values["work"]))
        context = CallContext(state.session, "coordinator", work.instance_id)
        offset = len(work.messages)
        invalid_offset = len(work.invalid_outputs)
        try:
            try:
                messages = await self.compiler.compile(
                    thread_id=self.thread_id,
                    owner_id=state.session.user_id,
                    system_messages=work.messages[:1],
                    task_snapshot={"tasks": state.tasks.snapshot(), "verified": True},
                    current_message=work.user_message,
                    current_turn=state.turn,
                    tools=self.coordinator._schemas(context, work.confirmed),
                    current_messages=work.messages[work.base_messages :],
                )
            except (ContextBudgetExceeded, LLMError, TimeoutError) as exc:
                reason = str(exc) if isinstance(exc, ContextBudgetExceeded) else type(exc).__name__
                self.coordinator.stop_turn(state, work, f"上下文编译无法继续：{reason}")
            else:
                await self.coordinator.advance_turn(state, work, messages=messages)
        except (asyncio.CancelledError, OSError):
            self.archive.append(
                self.thread_id,
                state.session.user_id,
                f"{work.instance_id}:interrupted:{work.rounds}",
                "event",
                {"event": "decision_interrupted", "messages": work.messages[offset:]},
                turn=state.turn,
            )
            raise
        messages_to_archive = [
            (f"{work.instance_id}:message:{index}", message)
            for index, message in enumerate(work.messages[offset:], start=offset)
        ]
        if work.result is not None:
            messages_to_archive.append(
                (
                    f"{work.instance_id}:reply",
                    {"role": "assistant", "content": work.result["reply"]},
                )
            )
        self.archive.append_messages(
            self.thread_id, state.session.user_id, messages_to_archive, turn=state.turn
        )
        for index, message in enumerate(
            work.invalid_outputs[invalid_offset:], start=invalid_offset
        ):
            self.archive.append(
                self.thread_id,
                state.session.user_id,
                f"{work.instance_id}:invalid:{index}",
                "event",
                {"event": "invalid_model_output", "message": message},
                turn=state.turn,
            )
        return {"conversation": _snapshot(state), "work": asdict(work), "result": work.result}

    @staticmethod
    def _next_node(values: RuntimeState) -> str:
        """有最终回复时等待用户，否则进入下一个协调决策节点。"""
        return "wait" if values.get("result") is not None else "advance"


def _snapshot(state: ConversationState) -> dict:
    """将任务状态转换为 JSON 检查点数据，完整历史另由档案保存。

    Args:
        state: 当前已验证会话的可信任务与展示状态。

    Returns:
        会话标识、任务、展示范围和轮次；不包含可从档案取得的聊天历史。
    """
    return {
        "session_id": state.session.session_id,
        "tasks": state.tasks.snapshot(),
        "last_display": deepcopy(state.last_display),
        "turn": state.turn,
    }


def _conversation(snapshot: dict, session: SessionContext) -> ConversationState:
    """用检查点重建任务并绑定本次已验证身份，不沿用旧会话验证标记。

    Args:
        snapshot: 图检查点中的 JSON 会话状态。
        session: 当前网关重新验证后的会话身份。

    Returns:
        与输入隔离的内存状态；历史消息由上下文编译器从档案读取。
    """
    state = ConversationState(session)
    tasks = [Task(**deepcopy(item)) for item in snapshot["tasks"]]
    state.tasks.tasks = {task.id: task for task in tasks}
    state.last_display = deepcopy(snapshot["last_display"])
    state.turn = snapshot["turn"]
    return state

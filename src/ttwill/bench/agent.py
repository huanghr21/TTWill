"""把持久化本体包装为只交换用户可见文本的 τ² 半双工 Agent。"""

import asyncio
from contextvars import copy_context
from dataclasses import dataclass

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import AssistantMessage, Message, UserMessage

from ttwill.core.runtime import PersistentRuntime


@dataclass(frozen=True)
class AgentState:
    """记录协议层的运行标识与消息数，不保存任何业务任务状态。

    Attributes:
        thread_id: 持久化运行时的业务任务线程标识。
        turns: 本包装器已处理的用户消息数。
    """

    thread_id: str
    turns: int = 0


class TTWillAgent(HalfDuplexAgent[AgentState]):
    """将官方同步消息接口接到本体异步入口，不接收任务答案或用户场景。"""

    def __init__(self, runtime: PersistentRuntime, runner: asyncio.Runner) -> None:
        """绑定本体及整个评测生命周期共用的异步执行器。

        Args:
            runtime: 仅处理真实对话的本体运行时，身份由协调模型和网关验证。
            runner: 调用方创建并关闭的执行器；检查点与模型资源须在同一循环中打开。
        """
        super().__init__(tools=[], domain_policy="")
        self.runtime, self.runner = runtime, runner

    def get_init_state(self, message_history: list[Message] | None = None) -> AgentState:
        """接受官方固定开场白，建立不含任务数据的协议状态。

        Args:
            message_history: 空历史，或官方 Orchestrator 提供的一条纯文本助手开场白。
                当前适配不接收预置用户对话或工具历史。

        Returns:
            当前业务线程的零轮次协议状态；任务仍由运行时检查点维护。

        Raises:
            ValueError: 历史包含用户、工具、多个消息或非文本助手开场白。
        """
        history = message_history or []
        if history and (
            len(history) != 1
            or not isinstance(history[0], AssistantMessage)
            or not history[0].has_text_content()
            or history[0].is_tool_call()
            or history[0].is_audio
        ):
            raise ValueError("当前评测仅接受空历史或一条纯文本助手开场白")
        return AgentState(self.runtime.thread_id)

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: AgentState
    ) -> tuple[AssistantMessage, AgentState]:
        """处理一条模拟用户原话，向官方轨迹只返回实际客服回复。

        Args:
            message: 官方模拟器产生的非空纯文本 UserMessage。
            state: get_init_state 或上次调用返回的协议状态。

        Returns:
            无工具调用的助手文本消息，以及递增轮次后的协议状态。

        Raises:
            ValueError: 输入不是纯文本用户消息，或协议状态属于另一业务线程。
            RuntimeError: 调用线程已经运行其他事件循环，无法进入同步执行器。
        """
        if (
            not isinstance(message, UserMessage)
            or not isinstance(message.content, str)
            or not message.content.strip()
            or message.is_tool_call()
            or message.is_audio
        ):
            raise ValueError("TTWill 评测入口只接受非空纯文本用户消息")
        if state.thread_id != self.runtime.thread_id:
            raise ValueError("协议状态不属于当前业务线程")
        result = self.runner.run(
            self.runtime.receive_message(message.content), context=copy_context()
        )
        reply = AssistantMessage(role="assistant", content=result["reply"])
        return reply, AgentState(state.thread_id, state.turns + 1)

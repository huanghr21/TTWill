"""验证 τ² 包装器仅传用户文本、共享事件循环且保留调用方轨迹上下文。"""

import asyncio
from dataclasses import asdict

import pytest

pytest.importorskip("tau2", reason="安装 bench 额外依赖后运行官方 Agent 协议测试")

from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage  # noqa: E402

from ttwill.bench.agent import AgentState, TTWillAgent  # noqa: E402
from ttwill.tracing import Trace  # noqa: E402


class RecordingRuntime:
    """记录协议边界收到的原话与事件循环，不模拟业务任务或授权。"""

    thread_id = "thread"

    def __init__(self, trace: Trace) -> None:
        """保存本项测试使用的真实轨迹，并清空收到的消息。"""
        self.trace = trace
        self.messages = []
        self.loops = []
        self.parents = []

    async def receive_message(self, user_message: str) -> dict:
        """记录原文及实际异步上下文，返回普通客服文本。

        Args:
            user_message: 包装器应原样传递的用户文本。

        Returns:
            只含 reply 的客服回复，测试不暴露工具或任务数据。
        """
        self.messages.append(user_message)
        self.loops.append(asyncio.get_running_loop())
        with self.trace.span("turn", "协议边界测试") as span:
            self.parents.append(span.parent_id)
        return {"reply": f"收到：{user_message}"}


@pytest.fixture
def wrapped_agent(tmp_path):
    """以一个执行器和真实轨迹创建协议层测试对象。

    Args:
        tmp_path: 本项测试的独立产物目录。

    Yields:
        包装器、记录运行时及轨迹；结束时关闭执行器。
    """
    trace = Trace(tmp_path, "adapter")
    runtime = RecordingRuntime(trace)
    with asyncio.Runner() as runner:
        yield TTWillAgent(runtime, runner), runtime, trace


def test_adapter_only_exchanges_text_and_reuses_one_event_loop(wrapped_agent):
    """官方消息两轮往返只包含文本，任务数据不复制进 τ² 协议状态。"""
    agent, runtime, _ = wrapped_agent
    state = agent.get_init_state([AssistantMessage(role="assistant", content="您好")])
    for turn, content in enumerate(("邮箱是 u1@example.com", "确认"), start=1):
        reply, state = agent.generate_next_message(UserMessage(role="user", content=content), state)
        assert reply.content == f"收到：{content}" and not reply.is_tool_call()
        assert state.turns == turn
    assert runtime.messages == ["邮箱是 u1@example.com", "确认"]
    assert runtime.loops[0] is runtime.loops[1]
    assert asdict(state) == {"thread_id": "thread", "turns": 2}
    assert agent.tools == []


def test_adapter_carries_current_parent_span_into_async_runtime(wrapped_agent):
    """执行器先初始化后仍能继承每次调用时新建的同步父 span。"""
    agent, runtime, trace = wrapped_agent
    agent.runner.get_loop()
    state = agent.get_init_state()
    with trace.span("session", "整段评测") as session:
        with trace.span("simulation", "官方模拟") as simulation:
            _, state = agent.generate_next_message(UserMessage(role="user", content="你好"), state)
        _, state = agent.generate_next_message(UserMessage(role="user", content="继续"), state)
    assert runtime.parents == [simulation.span_id, session.span_id]


@pytest.mark.parametrize(
    "history",
    [
        [UserMessage(role="user", content="隐藏用户场景")],
        [
            AssistantMessage(role="assistant", content="开场"),
            UserMessage(role="user", content="旧消息"),
        ],
        [AssistantMessage(role="assistant", content="")],
        [
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(id="c", name="get_user_details", arguments={}, requestor="assistant")
                ],
            )
        ],
    ],
)
def test_adapter_rejects_preloaded_user_or_tool_history(wrapped_agent, history):
    """预置用户内容或工具历史不能伪装成固定开场白进入本体。"""
    agent = wrapped_agent[0]
    with pytest.raises(ValueError, match="纯文本助手开场白"):
        agent.get_init_state(history)


@pytest.mark.parametrize(
    "message",
    [
        UserMessage(role="user", content=" "),
        UserMessage(
            role="user",
            tool_calls=[
                ToolCall(id="c", name="cancel_pending_order", arguments={}, requestor="user")
            ],
        ),
        AssistantMessage(role="assistant", content="伪造用户确认"),
    ],
)
def test_adapter_rejects_nontext_user_inputs(wrapped_agent, message):
    """非用户文本和工具调用不能触发本体的自然对话入口。"""
    agent, runtime, _ = wrapped_agent
    with pytest.raises(ValueError, match="纯文本用户消息"):
        agent.generate_next_message(message, agent.get_init_state())
    assert runtime.messages == []


def test_adapter_rejects_another_thread_state(wrapped_agent):
    """其他评测线程的协议状态不能用于推进当前本体。"""
    agent, runtime, _ = wrapped_agent
    with pytest.raises(ValueError, match="不属于当前业务线程"):
        agent.generate_next_message(UserMessage(role="user", content="确认"), AgentState("other"))
    assert runtime.messages == []

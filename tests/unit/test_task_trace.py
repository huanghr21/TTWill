"""验证任务改修订后，业务写入与子 Agent 派工仍能从轨迹准确归属。"""

import asyncio
import json
from pathlib import Path

import pytest

from ttwill.agents import build_registry
from ttwill.backend import RetailBackend, request_key
from ttwill.config import AppConfig, LLMConfig, ModelConfig
from ttwill.core.coordinator import ConversationState, Coordinator, Reply
from ttwill.core.subagent import DispatchTask, SubAgentRunner
from ttwill.core.tasks import Task
from ttwill.gateway import CallContext, SessionContext, ToolGateway
from ttwill.gateway.skills import SkillCatalog
from ttwill.llm import FakeModel, LLMClient
from ttwill.tracing import Trace


@pytest.fixture
def trace_runtime(
    tmp_path: Path, backend: RetailBackend, skill_catalog: SkillCatalog
) -> tuple[Coordinator, ConversationState, CallContext]:
    """连接真实后端、修订为 3 的取消任务与不调用模型的协调入口。

    Args:
        tmp_path: 当前测试的独立轨迹目录。
        backend: 初始化测试种子的 SQLite 后端。
        skill_catalog: 与业务动作匹配的 Skill 目录。

    Returns:
        协调者、已验证的会话状态、尚未绑定具体任务的协调上下文。
    """
    trace = Trace(tmp_path / "run", "chat")
    gateway = ToolGateway(backend, trace, skill_catalog)
    config = AppConfig(
        tmp_path, tmp_path / "run", LLMConfig(1, 0, 0), {"flash": ModelConfig("fake")}
    )
    llm = LLMClient(config, trace, FakeModel([]))
    runner = SubAgentRunner(build_registry(skill_catalog), gateway, llm, max_steps=4)
    coordinator = Coordinator(gateway, runner, llm, max_rounds=4)
    state = ConversationState(SessionContext("chat"), turn=1)
    context = CallContext(state.session, "coordinator", "coordinator-instance")
    gateway.call(context, "find_user_id_by_email", {"email": "u1@example.com"})
    task = Task(
        "revised-task",
        "order_service",
        "cancel_pending_order",
        "取消订单 #W0000001",
        revision=3,
        order_id="#W0000001",
        status="pending_confirm",
    )
    operation = backend.prepare(
        "u1",
        task.action,
        {"order_id": "#W0000001", "reason": "no longer needed"},
        request_key(task.id, task.revision, task.action),
    )
    task.operation_ids = [operation["operation_id"]]
    state.tasks.tasks[task.id] = task
    return coordinator, state, context


@pytest.mark.parametrize(
    "message,kind,status", [("确认", "commit", "done"), ("不同意", "reject", "cancelled")]
)
def test_confirmation_records_real_task_revision(trace_runtime, message, kind, status) -> None:
    """确认提交或拒绝改修订后的任务，轨迹保留真实任务 ID 和修订号。"""
    coordinator, state, context = trace_runtime
    previous = coordinator._reply(state, context, Reply(message_hint=""), [], 1)["displayed"]
    state.turn = 2
    result = asyncio.run(coordinator._confirm(state, context, message, previous))
    assert result[0]["status"] == status
    records = coordinator.gateway.records
    writes = [record for record in records if record["kind"] == kind]
    assert len(writes) == 1
    assert writes[0]["task_id"] == "revised-task" and writes[0]["revision"] == 3
    if kind == "commit":
        checks = [
            record for record in records if record["kind"] in {"verification", "verification_read"}
        ]
        assert {record["kind"] for record in checks} == {"verification", "verification_read"}
        assert all(
            record["task_id"] == "revised-task" and record["revision"] == 3 for record in checks
        )
    events = [json.loads(line) for line in coordinator.gateway.trace.path.read_text().splitlines()]
    persisted = [
        event for event in events if event["event"] == "end" and event["attrs"].get("kind") == kind
    ]
    assert persisted[0]["attrs"] == writes[0]


def test_cleanup_records_revised_task(trace_runtime) -> None:
    """任务失败清理准备项时，拒绝轨迹仍指向当前任务修订。"""
    coordinator, state, context = trace_runtime
    task = state.tasks.tasks["revised-task"]
    task.status, task.reason = "failed", "本轮任务失败"
    coordinator._discard_prepared(context, task)
    rejected = [record for record in coordinator.gateway.records if record["kind"] == "reject"]
    assert len(rejected) == 1
    assert rejected[0]["task_id"] == task.id and rejected[0]["revision"] == task.revision


def test_dispatch_records_exact_instruction(trace_runtime) -> None:
    """派工节点保存实际交给子 Agent 的完整任务说明及修订号。"""
    coordinator, state, _ = trace_runtime
    message = {
        "role": "assistant",
        "content": json.dumps(
            {"missing": ["订单号"], "message_hint": "请补充订单号"}, ensure_ascii=False
        ),
    }
    coordinator.runner.llm.backend = FakeModel(
        [{"model": "fake", "choices": [{"finish_reason": "stop", "message": message}]}]
    )
    instruction = "请查询待补充的订单号，保留用户原话中的全部要求。"
    asyncio.run(
        coordinator.runner.run(
            state.session, "order_service", DispatchTask("read-task", instruction, revision=4)
        )
    )
    events = [json.loads(line) for line in coordinator.gateway.trace.path.read_text().splitlines()]
    dispatch = [
        event for event in events if event["kind"] == "dispatch" and event["event"] == "end"
    ]
    assert len(dispatch) == 1
    assert dispatch[0]["attrs"]["instruction"] == instruction
    assert dispatch[0]["attrs"]["task_id"] == "read-task"
    assert dispatch[0]["attrs"]["revision"] == 4

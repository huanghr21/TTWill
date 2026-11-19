"""验证五区上下文、纯对话历史、一次摘要与本轮输入的完整保留。"""

import asyncio
import json

import pytest

from ttwill.config import AppConfig, LLMConfig, ModelConfig
from ttwill.core.context import (
    DATA_NOTICE,
    STATE_NOTICE,
    SUMMARY_NOTICE,
    ContextBudgetExceeded,
    ContextCompiler,
    context_chars,
)
from ttwill.llm import FakeModel, LLMClient, LLMError
from ttwill.persistence import Archive
from ttwill.tracing import Trace


def response(text: str) -> dict:
    """将摘要正文包装成统一调用层使用的假响应。"""
    return {
        "model": "fake",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }


@pytest.fixture
def runtime(tmp_path):
    """提供独立档案与假模型的上下文编译器工厂，测试结束时关闭所有连接。"""
    archives = []

    def make(script=(), *, budget=3000, recent=1):
        """按指定预算与脚本构造编译器。

        Args:
            script: FakeModel 按序消费的摘要响应或错误。
            budget: 协调输入的最大字符数。
            recent: 完整保留的历史用户轮次数。

        Returns:
            依次返回编译器、档案和假模型。
        """
        index = len(archives)
        archive = Archive(tmp_path / f"archive-{index}.sqlite")
        archives.append(archive)
        config = AppConfig(
            tmp_path,
            tmp_path / "runs",
            LLMConfig(1, 0, 0),
            {"flash": ModelConfig("fake"), "cheap": ModelConfig("fake-cheap")},
        )
        fake = FakeModel(script)
        llm = LLMClient(config, Trace(config.runs_dir, f"context-{index}"), fake)
        return ContextCompiler(archive, llm, max_chars=budget, recent_turns=recent), archive, fake

    yield make
    for archive in archives:
        archive.close()


def compile_context(compiler: ContextCompiler, **overrides) -> list[dict]:
    """用固定线程与权威输入执行一次编译，允许测试覆盖调用参数。

    Args:
        compiler: 当前测试的上下文编译器。
        overrides: 覆盖当前轮次、任务快照等 compile 参数。

    Returns:
        编译后的消息列表。
    """
    arguments = {
        "thread_id": "thread",
        "owner_id": "u1",
        "system_messages": [{"role": "system", "content": "稳定系统提示"}],
        "task_snapshot": [{"id": "task", "status": "pending_confirm", "revision": 2}],
        "current_message": "当前只改地址，不要取消订单",
        "current_turn": 3,
    }
    arguments.update(overrides)
    return asyncio.run(compiler.compile(**arguments))


def add_message(archive: Archive, event_id: str, text: str, turn: int, role: str = "user") -> None:
    """在固定用户与线程中保存并标记指定轮次的实际对话。"""
    archive.append_message(
        "thread", "u1", event_id, {"role": role, "content": text}, turn=turn, dialogue=True
    )


def test_small_context_keeps_exact_inputs_without_model_call(runtime):
    """正常上下文不调用摘要模型，当前用户消息不会因已归档而重复出现。"""
    compiler, archive, fake = runtime()
    add_message(archive, "old", "用户最初的话", 1)
    add_message(archive, "recent", "上一轮回复", 2, "assistant")
    add_message(archive, "current", "当前只改地址，不要取消订单", 3)
    snapshot = [{"id": "t", "status": "pending_confirm"}]
    result = compile_context(compiler, task_snapshot=snapshot)
    assert [item["content"] for item in result[:3]] == [
        "稳定系统提示",
        "用户最初的话",
        "上一轮回复",
    ]
    assert result[-2]["content"] == STATE_NOTICE + '[{"id":"t","status":"pending_confirm"}]'
    assert result[-3] == {"role": "system", "content": DATA_NOTICE + "[]"}
    assert result[-1] == {"role": "user", "content": "当前只改地址，不要取消订单"}
    result[-2]["content"] = "外部改动"
    assert snapshot == [{"id": "t", "status": "pending_confirm"}]
    assert not fake.requests


def test_internal_history_stays_archived_without_entering_dialogue(runtime):
    """内部调用、错误和同文未标记消息不进入跨轮上下文，完整档案仍可回查。"""
    compiler, archive, fake = runtime(budget=1100)
    call = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "测试旧轮思考标记",
        "tool_calls": [
            {"id": "call1", "type": "function", "function": {"name": "query", "arguments": "{}"}}
        ],
    }
    archive.append_message("thread", "u1", "call", call, turn=1)
    original = {"role": "tool", "tool_call_id": "call1", "content": "商品结果" * 3000}
    archive.append_message("thread", "u1", "result", original, turn=1)
    recent = {
        "role": "assistant",
        "content": "订单已查询。",
        "reasoning_content": "测试近期无工具回复的思考标记",
    }
    archive.append_message("thread", "u1", "recent", recent, turn=2, dialogue=True)
    archive.append_message(
        "thread", "u1", "internal-recent", {"role": "assistant", "content": "内部拒绝"}, turn=2
    )
    archive.append_message("thread", "u1", "unmarked-copy", recent, turn=2)
    result = compile_context(compiler)
    assert all(message["role"] != "tool" for message in result)
    assert call not in result
    assert result.count(recent) == 1
    assert "内部拒绝" not in str(result)
    assert "_ttwill_dialogue" not in str(result)
    assert archive.get_event("thread", "u1", "call")["payload"] == call
    assert archive.get_event("thread", "u1", "result")["payload"] == original
    assert context_chars(result) <= 1100
    assert not fake.requests


def test_summary_reads_only_dialogue_and_preserves_current_tool_exchange(runtime):
    """历史只摘要真实对话，本轮调用的思考、工具结果和错误则原样保留。"""
    compiler, archive, fake = runtime([response("此前已查询第一笔订单。")], budget=1500)
    add_message(archive, "old-user", "真实用户旧原话" * 500, 1)
    add_message(archive, "old-reply", "真实助手旧回复", 1, "assistant")
    add_message(archive, "recent-user", "真实用户近期原话", 2)
    old_call = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "旧轮思考标记" * 500,
        "tool_calls": [
            {"id": "old-call", "type": "function", "function": {"name": "query", "arguments": "{}"}}
        ],
    }
    old_result = {"role": "tool", "tool_call_id": "old-call", "content": "旧查询结果" * 500}
    current_call = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "本轮思考标记",
        "tool_calls": [
            {
                "id": "current-call",
                "type": "function",
                "function": {"name": "query", "arguments": "{}"},
            }
        ],
    }
    current_result = {"role": "tool", "tool_call_id": "current-call", "content": "本轮查询错误"}
    for event_id, message, turn in (
        ("old-call", old_call, 1),
        ("old-result", old_result, 1),
        ("recent-error", {"role": "assistant", "content": "近期内部错误噪声"}, 2),
    ):
        archive.append_message("thread", "u1", event_id, message, turn=turn)

    business = [{"kind": "order", "id": "#1", "status": "历史业务状态"}]
    messages = compile_context(
        compiler, current_messages=[current_call, current_result], business_data=business
    )

    assert len(fake.requests) == 1
    assert messages[1]["content"] == SUMMARY_NOTICE + "\n此前已查询第一笔订单。"
    assert messages[-2:] == [current_call, current_result]
    assert "真实用户近期原话" in str(messages)
    assert "近期内部错误噪声" not in str(messages)
    assert "old-call" not in json.dumps(messages)
    summary_request = json.loads(fake.requests[0]["messages"][-1]["content"])
    assert summary_request["history"] == [
        {"role": "user", "content": "真实用户旧原话" * 500},
        {"role": "assistant", "content": "真实助手旧回复"},
    ]
    assert "历史业务状态" not in str(summary_request)
    assert "本轮查询错误" not in str(summary_request)
    assert context_chars(messages) <= 1500
    assert archive.get_event("thread", "u1", "old-call")["payload"] == old_call
    assert archive.get_event("thread", "u1", "old-result")["payload"] == old_result


def test_business_data_has_fixed_position_and_cannot_change_input(runtime):
    """业务数据置于实际对话之后和权威快照之前，明确本轮新结果优先。"""
    compiler, archive, fake = runtime()
    add_message(archive, "old", "用户历史原话", 1)
    add_message(archive, "recent", "已展示历史卡片", 2, "assistant")
    data = [{"kind": "order", "id": "#1", "data": {"status": "pending"}}]
    current = [{"role": "tool", "tool_call_id": "c", "content": "新状态：delivered"}]
    result = compile_context(compiler, business_data=data, current_messages=current)
    assert [message["role"] for message in result] == [
        "system",
        "user",
        "assistant",
        "system",
        "system",
        "user",
        "tool",
    ]
    assert result[3]["content"].startswith(DATA_NOTICE)
    assert json.loads(result[3]["content"][len(DATA_NOTICE) :]) == data
    assert "本轮工具返回的新结果优先" in result[3]["content"]
    assert result[4]["content"].startswith(STATE_NOTICE)
    assert result[-1] == current[0]
    result[3]["content"] = "外部改动"
    result[-1]["content"] = "外部改动"
    assert data[0]["data"]["status"] == "pending"
    assert current[0]["content"] == "新状态：delivered"
    assert not fake.requests


def test_summary_is_single_call_marked_reference_and_reused(runtime):
    """超限旧轮次只摘要一次，之后新增近期轮次仍能复用原前缀摘要。"""
    compiler, archive, fake = runtime(
        [response("用户此前咨询取消订单，尚需以当前要求为准。")], budget=800
    )
    add_message(archive, "old1", "历史需求" * 300, 1)
    add_message(archive, "recent2", "近期话", 2)
    first = compile_context(compiler)
    assert first[1]["content"].startswith(SUMMARY_NOTICE)
    assert len(fake.requests) == 1
    assert "pending_confirm" not in fake.requests[0]["messages"][-1]["content"]
    assert "当前只改地址" not in fake.requests[0]["messages"][-1]["content"]
    saved = archive.entries("thread", "u1", kind="summary")
    assert len(saved) == 1
    assert "task_snapshot" not in saved[0]["payload"]
    add_message(archive, "recent3", "最新一轮话", 3)
    archive.append_message(
        "thread", "u1", "internal-late", {"role": "assistant", "content": "内部错误" * 500}, turn=1
    )
    second = compile_context(compiler, current_turn=4)
    assert second[1] == first[1]
    assert "近期话" in str(second)
    assert "最新一轮话" in str(second)
    assert len(fake.requests) == 1
    assert len(archive.messages("thread", "u1")) == 4
    assert len(archive.dialogue_messages("thread", "u1")) == 3


def test_summary_is_reusable_after_reopening_archive(runtime):
    """摘要保存在 SQLite 中，重新打开档案后不必再次调用模型。"""
    compiler, archive, fake = runtime([response("已压缩旧历史。")], budget=700)
    add_message(archive, "old", "很长的历史" * 200, 1)
    expected = compile_context(compiler)
    with Archive(archive.path) as reopened:
        restored = ContextCompiler(reopened, compiler.llm, max_chars=700, recent_turns=1)
        assert compile_context(restored) == expected
    assert len(fake.requests) == 1


def test_additional_old_history_triggers_one_new_summary_when_needed(runtime):
    """再次超限时只摘要已有摘要和新增旧轮次，不重复发送已压缩原文。"""
    compiler, archive, fake = runtime(
        [response("第一批历史摘要"), response("两批历史摘要")], budget=750
    )
    add_message(archive, "old1", "最早原文" * 300, 1)
    compile_context(compiler)
    add_message(archive, "old2", "第二批历史" * 300, 2)
    second = compile_context(compiler, current_turn=4)
    assert "两批历史摘要" in str(second)
    assert len(fake.requests) == 2
    request = fake.requests[1]["messages"][-1]["content"]
    assert "第一批历史摘要" in request
    assert "最早原文" not in request
    assert len(archive.entries("thread", "u1", kind="summary")) == 2


@pytest.mark.parametrize(
    "field", ["task_snapshot", "current_message", "current_messages", "business_data", "tools"]
)
def test_authoritative_and_current_content_is_never_truncated(runtime, field):
    """当前任务、原话、本轮工具往返和工具定义超限时明确失败，不截断或调用摘要。"""
    compiler, archive, fake = runtime(budget=500)
    add_message(archive, "old", "旧历史" * 300, 1)
    values = {
        "task_snapshot": [{"instruction": "任务原文" * 500}],
        "current_message": "本轮原话" * 500,
        "current_messages": [{"role": "tool", "content": "本轮结果" * 500, "tool_call_id": "c"}],
        "business_data": [{"data": "历史业务数据" * 500}],
        "tools": [{"description": "工具定义" * 500}],
    }
    with pytest.raises(ContextBudgetExceeded):
        compile_context(compiler, **{field: values[field]})
    assert not fake.requests
    assert archive.entries("thread", "u1", kind="summary") == []


def test_recent_rounds_are_not_summarized(runtime):
    """近期完整轮次占满预算时保留原文并报错，不擅自缩短近期窗口。"""
    compiler, archive, fake = runtime(budget=700)
    add_message(archive, "recent", "近期重要约束" * 500, 2)
    with pytest.raises(ContextBudgetExceeded):
        compile_context(compiler)
    assert not fake.requests


def test_oversized_summary_is_not_silently_truncated_or_retried(runtime):
    """一次摘要仍超限则明确失败，不能循环摘要或截断摘要与任务。"""
    compiler, archive, fake = runtime([response("摘要仍然太长" * 500)], budget=700)
    add_message(archive, "old", "旧历史" * 500, 1)
    with pytest.raises(ContextBudgetExceeded, match="一次历史摘要"):
        compile_context(compiler)
    assert len(fake.requests) == 1
    assert archive.entries("thread", "u1", kind="summary") == []


def test_summary_failure_propagates_without_losing_history(runtime):
    """摘要服务失败原样传给调用方，完整消息保持在档案中。"""
    compiler, archive, fake = runtime([LLMError("摘要失败")], budget=700)
    add_message(archive, "old", "完整旧历史" * 500, 1)
    with pytest.raises(LLMError, match="摘要失败"):
        compile_context(compiler)
    assert len(fake.requests) == 1
    assert archive.get_event("thread", "u1", "old")["payload"]["content"] == "完整旧历史" * 500


def test_other_owner_history_and_summary_are_never_loaded(runtime):
    """编译上下文不能加载其他用户的原文或摘要。"""
    compiler, archive, fake = runtime()
    archive.append_message(
        "thread",
        "u2",
        "private",
        {"role": "user", "content": "另一用户秘密"},
        turn=1,
        dialogue=True,
    )
    archive.append("thread", "u2", "summary", "summary", {"text": "另一用户秘密摘要"})
    result = compile_context(compiler)
    assert "秘密" not in str(result)
    assert not fake.requests


def test_summary_cache_rejects_unmatched_source_digest(runtime):
    """伪造或不匹配旧原文的摘要来源摘要不会替换真实消息。"""
    compiler, archive, fake = runtime()
    add_message(archive, "old", "必须保留的真实原话", 1)
    archive.append(
        "thread",
        "u1",
        "bad-summary",
        "summary",
        {
            "through_sequence": 1,
            "source_digest": "错误摘要值",
            "text": "错误历史",
        },
    )
    result = compile_context(compiler)
    assert "必须保留的真实原话" in str(result)
    assert "错误历史" not in str(result)
    assert not fake.requests


@pytest.mark.parametrize(("budget", "recent"), [(0, 1), (True, 1), (100, -1), (100, True)])
def test_invalid_context_limits_are_rejected(runtime, budget, recent):
    """上下文预算必须为正整数，近期轮数必须为非负整数。"""
    with pytest.raises(ValueError):
        runtime(budget=budget, recent=recent)

"""验证上下文预算、历史引用、一次摘要与权威输入的完整保留。"""

import asyncio
import json

import pytest

from ttwill.config import AppConfig, LLMConfig, ModelConfig
from ttwill.core.context import (
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
    """在固定用户与线程中保存指定轮次的完整消息。"""
    archive.append_message("thread", "u1", event_id, {"role": role, "content": text}, turn=turn)


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
    assert result[-1] == {"role": "user", "content": "当前只改地址，不要取消订单"}
    result[-2]["content"] = "外部改动"
    assert snapshot == [{"id": "t", "status": "pending_confirm"}]
    assert not fake.requests


def test_old_tool_result_is_referenced_before_any_summary(runtime):
    """旧工具正文引用化后满足预算时，不增加模型调用且仍能回查完整结果。"""
    compiler, archive, fake = runtime(budget=1100)
    call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call1", "type": "function", "function": {"name": "query", "arguments": "{}"}}
        ],
    }
    archive.append_message("thread", "u1", "call", call, turn=1)
    original = {"role": "tool", "tool_call_id": "call1", "content": "商品结果" * 3000}
    archive.append_message("thread", "u1", "result", original, turn=1)
    result = compile_context(compiler)
    tool = next(message for message in result if message["role"] == "tool")
    assert json.loads(tool["content"])["archive_reference"] == "result"
    assert tool["tool_call_id"] == "call1"
    assert call in result
    assert archive.get_event("thread", "u1", "result")["payload"] == original
    assert context_chars(result) <= 1100
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
    second = compile_context(compiler, current_turn=4)
    assert second[1] == first[1]
    assert "近期话" in str(second)
    assert "最新一轮话" in str(second)
    assert len(fake.requests) == 1
    assert len(archive.messages("thread", "u1")) == 3


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


@pytest.mark.parametrize("field", ["task_snapshot", "current_message", "current_messages", "tools"])
def test_authoritative_and_current_content_is_never_truncated(runtime, field):
    """当前任务、原话、本轮工具往返和工具定义超限时明确失败，不截断或调用摘要。"""
    compiler, archive, fake = runtime(budget=500)
    add_message(archive, "old", "旧历史" * 300, 1)
    values = {
        "task_snapshot": [{"instruction": "任务原文" * 500}],
        "current_message": "本轮原话" * 500,
        "current_messages": [{"role": "tool", "content": "本轮结果" * 500, "tool_call_id": "c"}],
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
        "thread", "u2", "private", {"role": "user", "content": "另一用户秘密"}, turn=1
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

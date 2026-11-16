"""验证确认规则、独立模型上下文及严格的逐操作判断范围。"""

import asyncio
import json
from pathlib import Path

import pytest

from ttwill.config import AppConfig, LLMConfig, ModelConfig
from ttwill.core.confirmation import ConfirmationJudge
from ttwill.llm import FakeModel, LLMClient, LLMError
from ttwill.tracing import Trace


def response(content) -> dict:
    """将判断列表或指定文本包装成统一调用层的假模型响应。"""
    return {
        "model": "fake",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content if isinstance(content, str) else json.dumps(content),
                },
                "finish_reason": "stop",
            }
        ],
    }


def decision(operation_id: str, value: str, reason: str = "用户明确指明该操作") -> dict:
    """构造用于模型响应的单项判断。"""
    return {"operation_id": operation_id, "decision": value, "reason": reason}


def displayed(*operation_ids: str) -> list[dict]:
    """为指定标识构造互不相同的实际展示文本。"""
    return [
        {"operation_id": operation_id, "content": f"请确认操作 {operation_id} 的完整详情。"}
        for operation_id in operation_ids
    ]


@pytest.fixture
def runtime(tmp_path):
    """提供绑定真实调用层和临时轨迹的假模型判定器工厂。

    Args:
        tmp_path: 当前测试独占的文件目录。

    Returns:
        make：接收响应脚本与可选超时秒数，依次返回判定器、假模型与轨迹。
    """

    def make(script, *, timeout_seconds=1):
        """创建单个判定器及其完整调用记录。

        Args:
            script: 按调用顺序消费的 FakeModel 响应或 LLMError。
            timeout_seconds: 模型单次请求的总超时秒数。

        Returns:
            依次返回 ConfirmationJudge、FakeModel 与 Trace。
        """
        config = AppConfig(
            tmp_path,
            tmp_path / "runs",
            LLMConfig(timeout_seconds, 0, 0),
            {"flash": ModelConfig("fake"), "pro": ModelConfig("fake-pro")},
        )
        trace = Trace(config.runs_dir, "confirmation")
        fake = FakeModel(script)
        return ConfirmationJudge(LLMClient(config, trace, fake)), fake, trace

    return make


def confirmation_event(path: Path) -> dict:
    """读取最近一次确认判定结束事件。"""
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [
        event for event in events if event["kind"] == "confirmation" and event["event"] == "end"
    ][-1]


@pytest.mark.parametrize("message", ["好的", "确认", "可以", "  同意！  ", "确认提交。"])
def test_single_explicit_short_confirmation_uses_rule(runtime, message):
    """单项完整短同意通过规则确认，轨迹保留真实原话和展示文本。"""
    judge, fake, trace = runtime([])
    shown = displayed("op-1")
    result = asyncio.run(judge.judge(message, shown))
    assert result == [decision("op-1", "confirm", "只展示一个操作，用户明确同意且未提出修改")]
    assert not fake.requests
    event = confirmation_event(trace.path)
    assert event["attrs"]["user_message"] == message
    assert event["attrs"]["displayed"] == shown
    assert event["attrs"]["results"] == result
    assert event["attrs"]["source"] == "rule"


@pytest.mark.parametrize("message", ["不同意", "拒绝", "不确认", "不要执行！", "不用了"])
def test_single_explicit_rejection_uses_rule(runtime, message):
    """单项拒绝直接返回 reject，不会调用模型或执行写入。"""
    judge, fake, _ = runtime([])
    result = asyncio.run(judge.judge(message, displayed("op-1")))
    assert result[0]["decision"] == "reject"
    assert result[0]["reason"]
    assert not fake.requests


@pytest.mark.parametrize("message", ["全部确认", "都可以", "全部同意！"])
def test_explicit_all_confirmation_covers_only_displayed_operations(runtime, message):
    """明确的全部同意只覆盖传入的当前展示项。"""
    judge, fake, _ = runtime([])
    result = asyncio.run(judge.judge(message, displayed("op-2", "op-1")))
    assert [item["operation_id"] for item in result] == ["op-2", "op-1"]
    assert [item["decision"] for item in result] == ["confirm", "confirm"]
    assert not fake.requests


@pytest.mark.parametrize("message", ["好的", "确认", "可以", "同意", "不同意", "拒绝"])
def test_ambiguous_multi_operation_short_reply_never_reaches_model(runtime, message):
    """多项范围不明短回复由规则拦住，即使假模型计划全部确认也不能扩大授权。"""
    judge, fake, trace = runtime(
        [response([decision("op-1", "confirm"), decision("op-2", "confirm")])]
    )
    result = asyncio.run(judge.judge(message, displayed("op-1", "op-2")))
    assert [item["decision"] for item in result] == ["unrelated", "unrelated"]
    assert not fake.requests
    assert confirmation_event(trace.path)["attrs"]["source"] == "rule"


def test_partial_confirmation_is_judged_per_id_and_restored_to_display_order(runtime):
    """用户指名确认和拒绝分别处理，模型换序不能改变输入与结果的对应关系。"""
    judge, fake, trace = runtime(
        [
            response(
                [
                    decision("op-2", "reject", "用户明确不要办理第二项"),
                    decision("op-1", "confirm", "用户明确同意第一项"),
                ]
            )
        ]
    )
    result = asyncio.run(judge.judge("第一项确认，第二项不要办理", displayed("op-1", "op-2")))
    assert result == [
        decision("op-1", "confirm", "用户明确同意第一项"),
        decision("op-2", "reject", "用户明确不要办理第二项"),
    ]
    assert len(fake.requests) == 1
    assert "tools" not in fake.requests[0]
    event = confirmation_event(trace.path)
    assert event["attrs"]["source"] == "model"
    assert event["attrs"]["results"] == result


@pytest.mark.parametrize(
    "message",
    ["可以，不过换成白色", "全部确认，不过第一项换成白色", "好的，请将地址换成新地址"],
)
def test_confirmation_with_modification_bypasses_short_rule(runtime, message):
    """完整回复含修改内容时必须独立判定，不能从前缀同意短语直接确认。"""
    judge, fake, _ = runtime([response([decision("op-1", "modify", "用户提出了修改要求")])])
    result = asyncio.run(judge.judge(message, displayed("op-1")))
    assert result[0]["decision"] == "modify"
    assert len(fake.requests) == 1


def test_question_punctuation_does_not_become_explicit_confirmation(runtime):
    """询问语气的“可以？”不按明确同意规则处理。"""
    judge, fake, _ = runtime([response([decision("op-1", "unrelated", "用户仍在询问")])])
    result = asyncio.run(judge.judge("可以？", displayed("op-1")))
    assert result[0]["decision"] == "unrelated"
    assert len(fake.requests) == 1


@pytest.mark.parametrize(
    "payload",
    [
        [decision("unshown", "confirm")],
        [decision("op-1", "confirm"), decision("op-1", "confirm")],
        [],
        [decision("op-1", "confirm"), decision("unshown", "confirm")],
        [{**decision("op-1", "confirm"), "authorized": True}],
        [{**decision("op-1", "confirm"), "reason": " \n "}],
        [{**decision("op-1", "confirm"), "reason": 1}],
        [{"operation_id": "op-1", "decision": "confirm"}],
        [decision("op-1", "done")],
        {"decisions": [decision("op-1", "confirm")]},
        "```json\n[]\n```",
        "不是 JSON",
    ],
)
def test_invalid_model_output_never_authorizes_any_operation(runtime, payload):
    """伪造、重复、缺项、额外字段和非严格输出全部按未确认处理且记录失败。"""
    judge, fake, trace = runtime([response(payload)])
    result = asyncio.run(judge.judge("请办理我指的这一项", displayed("op-1")))
    assert result == [decision("op-1", "unrelated", "确认判定输出不合格，本轮未确认")]
    assert len(fake.requests) == 1
    event = confirmation_event(trace.path)
    assert event["status"] == "blocked"
    assert event["attrs"]["failure_reason"]
    assert event["attrs"]["results"] == result


def test_missing_one_of_several_ids_invalidates_entire_model_judgment(runtime):
    """模型遗漏一个展示项时不能部分采纳其余确认。"""
    judge, _, _ = runtime([response([decision("op-1", "confirm")])])
    result = asyncio.run(judge.judge("请帮我办理相关项目", displayed("op-1", "op-2")))
    assert [item["decision"] for item in result] == ["unrelated", "unrelated"]


@pytest.mark.parametrize("kind", ["tool", "refusal"])
def test_tool_call_or_refusal_cannot_be_used_as_confirmation(runtime, kind):
    """即使伴有合法判断文本，工具调用和拒答响应也不构成确认。"""
    raw = response([decision("op-1", "confirm")])
    message = raw["choices"][0]["message"]
    if kind == "tool":
        message["tool_calls"] = [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "commit", "arguments": "{}"},
            }
        ]
    else:
        message["refusal"] = "无法判断。"
    judge, _, _ = runtime([raw])
    result = asyncio.run(judge.judge("请继续办理这一项", displayed("op-1")))
    assert result[0]["decision"] == "unrelated"


def test_model_failure_returns_unrelated_and_records_failure_type(runtime):
    """模型调用失败按未确认处理，失败原因不丢失。"""
    judge, fake, trace = runtime([LLMError("服务不可用")])
    result = asyncio.run(judge.judge("第一项确认", displayed("op-1", "op-2")))
    assert [item["decision"] for item in result] == ["unrelated", "unrelated"]
    assert len(fake.requests) == 1
    event = confirmation_event(trace.path)
    assert event["status"] == "blocked"
    assert event["attrs"]["error_type"] == "LLMError"
    assert "调用失败" in event["attrs"]["failure_reason"]


def test_timeout_returns_unrelated_without_second_judgment(runtime, monkeypatch):
    """统一调用层超时后不另行发起确认判定。"""
    judge, fake, trace = runtime([], timeout_seconds=0.001)

    async def delayed_send(model, request, timeout_seconds):
        """等待超过调用层时限，以触发真实总超时路径。"""
        await asyncio.sleep(1)
        raise AssertionError("超时后不应继续执行假模型")

    monkeypatch.setattr(fake, "send", delayed_send)
    result = asyncio.run(judge.judge("请继续办理这一项", displayed("op-1")))
    assert result[0]["decision"] == "unrelated"
    assert confirmation_event(trace.path)["attrs"]["error_type"] == "TimeoutError"


@pytest.mark.parametrize("error", [asyncio.CancelledError(), OSError("记录失败")])
def test_cancellation_and_oserror_propagate(runtime, monkeypatch, error):
    """取消与文件系统异常原样向上传播，不伪装成正常未确认结果。"""
    judge, fake, trace = runtime([])

    async def interrupted_send(model, request, timeout_seconds):
        """触发指定中断异常，验证判定器的异常边界。"""
        raise error

    monkeypatch.setattr(fake, "send", interrupted_send)
    with pytest.raises(type(error)):
        asyncio.run(judge.judge("请继续办理这一项", displayed("op-1")))
    event = confirmation_event(trace.path)
    assert event["status"] == "error"
    assert event["attrs"]["error_type"] == type(error).__name__


def test_model_input_contains_only_current_words_and_displayed_fields(runtime):
    """模型只看到当前原话和展示标识、正文，不获得历史、任务状态或隐藏操作。"""
    judge, fake, _ = runtime(
        [
            response([decision("op-1", "confirm")]),
            response([decision("op-2", "unrelated")]),
        ]
    )
    first = [
        {
            **displayed("op-1")[0],
            "task_id": "hidden-task",
            "history": "曾经同意隐藏操作",
            "expected_args": {"private": "internal"},
        }
    ]
    asyncio.run(judge.judge("仅确认第一项", first))
    asyncio.run(judge.judge("现在只想查询物流", displayed("op-2")))
    assert len(fake.requests) == 2
    for request in fake.requests:
        assert len(request["messages"]) == 2
        assert "tools" not in request
        data = json.loads(request["messages"][1]["content"])
        assert set(data) == {"user_message", "displayed"}
        assert all(set(item) == {"operation_id", "content"} for item in data["displayed"])
        assert "hidden-task" not in json.dumps(request)
        assert "internal" not in json.dumps(request)
    second = json.loads(fake.requests[1]["messages"][1]["content"])
    assert second == {"user_message": "现在只想查询物流", "displayed": displayed("op-2")}


def test_empty_displayed_or_user_message_needs_no_model(runtime):
    """无展示项或空用户输入没有可确认的内容。"""
    judge, fake, _ = runtime([])
    assert asyncio.run(judge.judge("全部确认", [])) == []
    result = asyncio.run(judge.judge("   ", displayed("op-1")))
    assert result[0]["decision"] == "unrelated"
    assert not fake.requests


@pytest.mark.parametrize(
    "shown",
    [
        displayed("op-1", "op-1"),
        [{"operation_id": "", "content": "详情"}],
        [{"operation_id": "op-1", "content": " "}],
        [{"operation_id": 1, "content": "详情"}],
        [{"operation_id": "op-1", "content": {"text": "详情"}}],
        [{"operation_id": "op-1"}],
    ],
)
def test_invalid_displayed_input_is_rejected_at_boundary(runtime, shown):
    """代码传入的展示标识与实际文本必须合法且唯一。"""
    judge, fake, _ = runtime([])
    with pytest.raises(ValueError):
        asyncio.run(judge.judge("确认", shown))
    assert not fake.requests


def test_invalid_user_message_and_unknown_tier_raise_valueerror(runtime):
    """用户原话必须是字符串，判定档位必须已经配置。"""
    judge, _, _ = runtime([])
    with pytest.raises(ValueError, match="用户原话"):
        asyncio.run(judge.judge(None, displayed("op-1")))
    with pytest.raises(ValueError, match="档位"):
        ConfirmationJudge(judge.llm, model_tier="unknown")


def test_configured_model_tier_is_used(runtime):
    """确认判定可选择已配置档位，并继续使用统一调用层。"""
    original, fake, _ = runtime([response([decision("op-1", "confirm")])])
    judge = ConfirmationJudge(original.llm, model_tier="pro")
    asyncio.run(judge.judge("请办理第一项", displayed("op-1")))
    assert fake.requests[0]["model"] == "fake-pro"

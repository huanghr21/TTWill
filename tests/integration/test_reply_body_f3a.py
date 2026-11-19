"""验证 F3a 回复正文、确认卡片与观察轨迹各自保留准确内容。"""

import asyncio
import json

import pytest
from pydantic import ValidationError
from test_coordinator import cancel_steps, cancel_task, task_ids, text_response, tool, turn
from test_coordinator import runtime as runtime

from ttwill.core.coordinator import PREPARED_STOP_NOTICE, STOP_NOTICE, Reply
from ttwill.core.tasks import Task
from ttwill.gateway import CallContext


@pytest.mark.parametrize("arguments", [{}, {"message": ""}, {"message": " \n "}, {"note": "回答"}])
def test_reply_requires_nonempty_message(arguments):
    """回复工具只接受非空正文，不把子 Agent 备注或空文本当作回复。"""
    with pytest.raises(ValidationError):
        Reply.model_validate(arguments)


@pytest.mark.parametrize("plain_text", [False, True])
@pytest.mark.parametrize("status", ["cancelled", "done"])
def test_existing_task_does_not_replace_reply_body(runtime, plain_text, status):
    """重放 24 的材质回答，终态任务及只读事实不再覆盖正文或输出 JSON。"""
    message = "  两件 T 恤的材质分别为棉、聚酯纤维。\n请按穿着需求选择。  "

    def script(state):
        """验证身份后直接发送带空格换行的回答，验证文本不被改写。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield text_response(message) if plain_text else tool("reply", {"message": message})

    coordinator, state, _, _, _ = runtime(script, [])
    state.tasks.tasks["old-task"] = Task(
        "old-task",
        "order_service",
        "get_order_details",
        "查询商品材质",
        order_id="#W0000001",
        status=status,
        missing=["过时信息缺口"],
        reason="过时撤销说明",
        facts=[{"result": {"items": [{"material": "棉"}, {"material": "聚酯纤维"}]}}],
    )
    result = turn(coordinator, state, "T 恤是什么材质？")
    assert result["reply"] == message
    assert result["displayed"] == []
    assert state.last_reply == message
    assert state.history[-1]["content"] == message
    assert "过时" not in result["reply"]


def test_cards_remain_separate_and_next_turn_preserves_actual_reply(runtime):
    """确认判定取得完整实际回复，卡片记录仍只保留操作内容。"""
    message = "请核对下方操作；运费问题还需要客服专员说明。"

    def script(state):
        """准备一项取消并发出正文和卡片。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {"task_ids": task_ids(state)})
        yield tool("reply", {"message": message})

    coordinator, state, gateway, _, _ = runtime(script, cancel_steps("#W0000001"))
    result = turn(coordinator, state, "取消订单")
    shown = result["displayed"][0]
    assert result["reply"].startswith(message + "\n\n" + shown["content"])
    assert message not in shown["content"]
    assert state.last_reply == result["reply"]
    work = coordinator.begin_turn(state, "好的")
    assert work.previous_reply == result["reply"]
    assert work.previous == result["displayed"]
    assert state.last_reply == "" and state.last_display == []

    class RecordingJudge:
        """记录完整回复的传递，不执行语义推断或授权业务操作。"""

        async def judge(self, user_message, displayed, *, assistant_reply=None):
            """核对真实原话和回复，再返回确定的未确认测试结果。"""
            assert user_message == "好的"
            assert assistant_reply == result["reply"]
            assert displayed == result["displayed"]
            return [
                {
                    "operation_id": shown["operation_id"],
                    "decision": "unrelated",
                    "reason": "本测试只验证上下文传递",
                }
            ]

    coordinator.judge = RecordingJudge()
    judgments = asyncio.run(
        coordinator._confirm(
            state,
            CallContext(state.session, "coordinator", work.instance_id),
            work.user_message,
            work.previous,
            work.previous_reply,
        )
    )
    assert judgments[0]["decision"] == "unrelated"
    assert not any(record["kind"] == "commit" for record in gateway.records)


@pytest.mark.parametrize("status", ["pending_confirm", "done"])
def test_completion_claims_are_logged_without_rewriting_or_done_exemption(runtime, status):
    """完成词只作观察，任务 done 也不能证明正文退款到账的表述正确。"""
    message = "旧订单已取消；这次已退款。"
    coordinator, state, gateway, _, _ = runtime(lambda state: iter(()), [])
    state.tasks.tasks["write-task"] = Task(
        "write-task",
        "after_sales",
        "return_delivered_order_items",
        "申请退货",
        order_id="#W0000003",
        status=status,
        facts=[{"verification": {"verified": True, "checked_fields": ["status"]}}],
    )
    result = coordinator._reply(
        state,
        CallContext(state.session, "coordinator", "test"),
        Reply(message=message, show_task_ids=[]),
        [],
        1,
    )
    assert result["reply"] == message
    events = [json.loads(line) for line in gateway.trace.path.read_text().splitlines()]
    claims = [
        event for event in events if event["kind"] == "reply_claim" and event["event"] == "end"
    ]
    assert len(claims) == 1
    assert claims[0]["attrs"]["message"] == message
    assert claims[0]["attrs"]["matched"] == ["已取消", "已退款"]
    assert claims[0]["attrs"]["tasks"][0]["status"] == status
    assert claims[0]["attrs"]["tasks"][0]["verification"]["verified"] is True


def test_stop_preserves_prepared_cards_without_claiming_the_whole_request_failed(runtime, backend):
    """分派用尽本轮预算时，仍展示有效准备项并准确说明哪些内容等待确认。"""

    def script(state):
        """第三次决策准备取消，随后由代码预算结束本轮。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {"task_ids": task_ids(state)})

    coordinator, state, _, _, _ = runtime(script, cancel_steps("#W0000001"), max_rounds=3)
    result = turn(coordinator, state, "取消订单")
    assert result["reply"].startswith(PREPARED_STOP_NOTICE + "\n\n")
    assert STOP_NOTICE not in result["reply"]
    assert len(result["displayed"]) == 1
    assert state.last_reply == result["reply"]
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "pending"

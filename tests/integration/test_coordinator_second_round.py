"""复现第二轮冒烟中的确认分派耦合与重复状态载荷，保留真实后端授权边界。"""

import asyncio
import json

import pytest
from test_coordinator import (
    ADDRESS,
    add_task,
    cancel_steps,
    cancel_task,
    reply,
    task_ids,
    text_response,
    tool,
    turn,
)
from test_coordinator import runtime as runtime

from ttwill.core.context import STATE_NOTICE
from ttwill.llm import FakeModel


def tool_results(fake, name: str) -> list[dict | list]:
    """从最后一次模型请求提取指定协调工具的配对返回值。

    Args:
        fake: 保存完整模型请求的假模型。
        name: 要读取结果的协调工具名。

    Returns:
        按执行顺序排列的已解析工具结果，不包含合成状态消息。
    """
    found = []
    selected = False
    for message in fake.requests[-1]["messages"]:
        if message["role"] == "assistant":
            calls = message.get("tool_calls", [])
            selected = len(calls) == 1 and calls[0]["function"]["name"] == name
        elif message["role"] == "tool" and selected:
            found.append(json.loads(message["content"]))
            selected = False
    return found


def task_snapshots(messages: list[dict]) -> list[dict]:
    """提取内存入口生成的权威任务消息，排除真实用户原话与工具往返。

    Args:
        messages: 一次模型请求的完整消息列表。

    Returns:
        含 tasks 与 verified 的代码状态列表。
    """
    snapshots = []
    for message in messages:
        content = message.get("content") or ""
        if message["role"] != "system" or not content.startswith(STATE_NOTICE):
            continue
        value = json.loads(content.removeprefix(STATE_NOTICE))
        if isinstance(value, dict) and "tasks" in value and "verified" in value:
            snapshots.append(value)
    return snapshots


def test_88_confirm_with_pending_task_selection_commits_and_verifies(runtime, backend):
    """重放 88 的“我全部确认”与待确认 task_ids，分派状态错误不能阻止独立确认。"""

    def script(state):
        """先准备并展示取消，再按真实失败参数处理用户的明确确认。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {"task_ids": task_ids(state)})
        yield reply()
        yield tool("complete_plan", {"task_ids": task_ids(state)})
        yield reply()

    def judgments():
        """独立判定器使用真实 88 轨迹中的明确授权测试响应。"""
        yield text_response(
            json.dumps(
                [
                    {
                        "operation_id": task.operation_ids[0],
                        "decision": "confirm",
                        "reason": "用户说我全部确认，明确授权最近展示的取消操作",
                    }
                    for task in state.tasks.tasks.values()
                ],
                ensure_ascii=False,
            )
        )

    coordinator, state, gateway, fake, child = runtime(script, cancel_steps("#W0000001"))
    coordinator.judge.llm.backend = FakeModel(judgments())
    first = turn(coordinator, state, "取消 #W0000001")
    second = turn(coordinator, state, "我全部确认")
    assert second["tasks"][0]["status"] == "done"
    assert second["confirmation_results"][0]["verification"]["verified"]
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "cancelled"
    assert len(child.requests) == 3
    assert sum(record["kind"] == "commit" for record in gateway.records) == 1
    outcome = tool_results(fake, "complete_plan")[-1]
    assert "tasks" not in outcome and outcome["dispatched"] == []
    assert outcome["blocked"][0]["task_id"] == first["tasks"][0]["id"]
    assert outcome["blocked"][0]["status"] == "done"
    assert outcome["confirmation_results"][0]["verification"] == {
        key: second["confirmation_results"][0]["verification"][key]
        for key in ("verified", "reason", "checked_fields")
    }
    assert "actual" in second["confirmation_results"][0]["verification"]
    assert len(task_snapshots(fake.requests[-1]["messages"])) == 1


def test_mixed_plan_ids_confirm_independently_and_dispatch_valid_task_once(runtime, backend):
    """混合未知、重复、已完成与合法 ID 时逐项说明，合法新任务只准备一次且不被授权。"""

    def script(state):
        """确认首单时混选第二单、未知 ID 和旧项，第二单仍须下一轮确认。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {"task_ids": task_ids(state)})
        yield reply()
        yield tool("update_tasks", {"changes": [cancel_task("#W0000004")]})
        old_id, new_id = task_ids(state)
        yield tool("complete_plan", {"task_ids": ["unknown", new_id, old_id, new_id]})
        yield reply()

    coordinator, state, gateway, fake, child = runtime(
        script, cancel_steps("#W0000001") + cancel_steps("#W0000004")
    )
    turn(coordinator, state, "取消 #W0000001")
    second = turn(coordinator, state, "全部确认")
    assert [task["status"] for task in second["tasks"]] == ["done", "pending_confirm"]
    assert len(child.requests) == 6
    assert sum(record["kind"] == "commit" for record in gateway.records) == 1
    assert backend.export_snapshot()["orders"]["#W0000004"]["status"] == "pending"
    outcome = tool_results(fake, "complete_plan")[-1]
    old_id, new_id = task_ids(state)
    assert outcome["dispatched"] == [
        {"task_id": new_id, "status": "pending_confirm", "reason": None}
    ]
    assert [item["task_id"] for item in outcome["blocked"]] == ["unknown", old_id, new_id]
    assert "不存在" in outcome["blocked"][0]["reason"]
    assert "状态" in outcome["blocked"][1]["reason"]
    assert "重复" in outcome["blocked"][2]["reason"]


@pytest.mark.parametrize("user_message", ["全部确认", "不同意"])
def test_unknown_selection_does_not_block_confirm_or_reject(runtime, backend, user_message):
    """未知分派 ID 不影响真正用户原话对已展示操作的确认或拒绝。"""

    def script(state):
        """把未知分派选择与真实取消操作的用户决策放在同一次工具调用中。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {"task_ids": task_ids(state)})
        yield reply()
        yield tool("complete_plan", {"task_ids": ["unknown"]})
        yield reply()

    coordinator, state, gateway, fake, child = runtime(script, cancel_steps("#W0000001"))
    turn(coordinator, state, "取消 #W0000001")
    second = turn(coordinator, state, user_message)
    confirmed = user_message == "全部确认"
    assert second["tasks"][0]["status"] == ("done" if confirmed else "cancelled")
    assert second["confirmation_results"][0]["decision"] == ("confirm" if confirmed else "reject")
    assert sum(record["kind"] == "commit" for record in gateway.records) == int(confirmed)
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == (
        "cancelled" if confirmed else "pending"
    )
    assert len(child.requests) == 3
    assert tool_results(fake, "complete_plan")[-1]["blocked"][0]["task_id"] == "unknown"


def test_selected_but_unshown_prepared_operation_is_not_authorized(runtime, backend):
    """模型选择未展示的准备项不能创造用户授权，即使用户输入全部确认。"""

    def script(state):
        """首轮故意不展示准备项，次轮仍将同一任务作为分派参数。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {"task_ids": task_ids(state)})
        yield reply([])
        yield tool("complete_plan", {"task_ids": task_ids(state)})
        yield reply()

    coordinator, state, gateway, fake, child = runtime(script, cancel_steps("#W0000001"))
    assert turn(coordinator, state, "取消 #W0000001")["displayed"] == []
    second = turn(coordinator, state, "全部确认")
    assert second["confirmation_results"] == []
    assert second["tasks"][0]["status"] == "pending_confirm"
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "pending"
    assert len(child.requests) == 3
    assert not any(record["kind"] == "commit" for record in gateway.records)
    assert tool_results(fake, "complete_plan")[-1]["blocked"][0]["status"] == "pending_confirm"


def test_unrelated_confirmation_keeps_same_order_lock(runtime, backend):
    """独立判定未授权时，即使混选合法后续任务，也不能绕过同单准备锁。"""

    def script(state):
        """用待确认取消占住订单，次轮新增同单地址操作并明确不授权旧项。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("complete_plan", {"task_ids": task_ids(state)})
        yield reply()
        yield tool(
            "update_tasks",
            {"changes": [add_task("modify_pending_order_address", "#W0000001", str(ADDRESS))]},
        )
        yield tool("complete_plan", {"task_ids": task_ids(state)})
        yield reply()

    def judgments():
        """独立判断本轮只问状态，因此保留尚未确认操作及同单占用。"""
        yield text_response(
            json.dumps(
                [
                    {
                        "operation_id": next(iter(state.tasks.tasks.values())).operation_ids[0],
                        "decision": "unrelated",
                        "reason": "用户只询问订单，未同意取消",
                    }
                ]
            )
        )

    coordinator, state, gateway, fake, child = runtime(script, cancel_steps("#W0000001"))
    coordinator.judge.llm.backend = FakeModel(judgments())
    turn(coordinator, state, "取消 #W0000001")
    second = turn(coordinator, state, "先告诉我订单状态")
    assert [task["status"] for task in second["tasks"]] == ["pending_confirm", "pending"]
    assert len(child.requests) == 3
    assert not any(record["kind"] == "commit" for record in gateway.records)
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "pending"
    outcome = tool_results(fake, "complete_plan")[-1]
    assert outcome["dispatched"] == [] and len(outcome["blocked"]) == 2
    assert "同一订单" in outcome["blocked"][1]["reason"]


def test_memory_requests_refresh_one_compact_snapshot_without_changing_message_offsets(runtime):
    """内存入口逐决策只注入一份最新精简状态，原始往返和稳定归档下标保持完整。"""

    def script(state):
        """身份、更新与分派后每次检查代码状态，最终展示准备操作。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()

    coordinator, state, _, fake, _ = runtime(script, cancel_steps("#W0000001"))

    async def run():
        """逐步推进并核对本轮记录只追加真实模型与配对工具消息。"""
        work = coordinator.begin_turn(state, "取消 #W0000001")
        base = work.base_messages
        while work.result is None:
            offset = len(work.messages)
            await coordinator.advance_turn(state, work)
            assert work.base_messages == base
            assert [message["role"] for message in work.messages[offset:]] == [
                "assistant",
                "tool",
            ]
            assert task_snapshots(work.messages) == []
        return work.result

    result = asyncio.run(run())
    snapshots = [task_snapshots(request["messages"]) for request in fake.requests]
    assert all(len(items) == 1 for items in snapshots)
    assert [items[0]["verified"] for items in snapshots] == [False, True, True, True]
    assert snapshots[0][0]["tasks"] == snapshots[1][0]["tasks"] == []
    assert snapshots[2][0]["tasks"][0]["status"] == "pending"
    view = snapshots[3][0]["tasks"][0]
    assert view["status"] == "pending_confirm"
    assert "payment_history" not in json.dumps(view)
    assert "preview" not in json.dumps(view)
    assert "preview" in json.dumps(result["tasks"][0]["actions"])
    assert tool_results(fake, "dispatch_tasks")[0]["dispatched"] == [
        {"task_id": task_ids(state)[0], "status": "pending_confirm", "reason": None}
    ]
    assert set(tool_results(fake, "update_tasks")[0][0]) == {
        "task_id",
        "status",
        "reason",
        "changed",
    }

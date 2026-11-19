"""用真实图检查点核对五区上下文、业务数据增量与终态任务清理。"""

import asyncio
import json

from test_runtime import child_script, followup_script, open_u1, prepare_script, tool
from test_runtime import runtime_factory as runtime_factory

from ttwill.core.context import DATA_NOTICE, STATE_NOTICE
from ttwill.gateway import CallContext


def zone(request: dict, notice: str) -> list | dict:
    """读取一个模型请求中唯一的合成区域。

    Args:
        request: 协调假模型记录的实际请求。
        notice: 区域消息的固定前缀。

    Returns:
        前缀之后的 JSON 业务数据或权威任务快照。
    """
    messages = [
        message["content"]
        for message in request["messages"]
        if isinstance(message.get("content"), str) and message["content"].startswith(notice)
    ]
    assert len(messages) == 1
    return json.loads(messages[0].removeprefix(notice))


def tool_results(request: dict, name: str) -> list[dict]:
    """按配对顺序读取指定工具在当前请求中的返回正文。

    Args:
        request: 含本轮完整工具调用和结果的模型请求。
        name: 需要核对的协调工具名称。

    Returns:
        保持出现顺序的 JSON 工具结果列表。
    """
    selected = set()
    results = []
    for message in request["messages"]:
        for call in message.get("tool_calls", []):
            if call["function"]["name"] == name:
                selected.add(call["id"])
            else:
                selected.discard(call["id"])
        if message["role"] == "tool" and message["tool_call_id"] in selected:
            results.append(json.loads(message["content"]))
            selected.discard(message["tool_call_id"])
    return results


def entities(data: list[dict]) -> dict[tuple[str, str], dict]:
    """把无重复实体的数据区转为便于核对的映射。

    Args:
        data: 数据区或协调工具返回的业务实体列表。

    Returns:
        以类型与标识二元组为键的条目映射；存在重复时测试失败。
    """
    mapped = {(entry["type"], entry["id"]): entry for entry in data}
    assert len(mapped) == len(data)
    return mapped


def test_child_reads_are_visible_in_same_turn_and_unique_next_turn(runtime_factory):
    """子 Agent 的订单与商品具体值同轮可见，下一轮仅由数据区提供最新实体。"""

    def script(ids):
        """创建读取任务并在下一轮继续对话，不重新查询已有实体。"""
        yield tool(
            "update_tasks",
            {
                "changes": [
                    {
                        "operation": "add",
                        "capability": "order_service",
                        "action": "get_order_details",
                        "order_id": "#W0000001",
                        "instruction": "查询订单地址与衬衫规格价格",
                    }
                ]
            },
        )
        yield tool("complete_plan", {})
        yield tool("dispatch_tasks", {"task_ids": ids()})
        yield tool("reply", {"message": "订单待发货，蓝色衬衫价格为20元。"})
        yield from followup_script(ids)

    children = [
        tool("get_order_details", {"order_id": "#W0000001"}),
        tool("get_product_details", {"product_id": "p1"}),
        tool("get_order_details", {"order_id": "#W0000001"}),
        tool("finish", {"missing": [], "note": "订单与商品已查询"}),
    ]

    async def scenario():
        """检查实际模型请求、完整检查点与跨轮数据区的不同职责。"""
        async with runtime_factory("reads", script, children) as bundle:
            runtime, _, gateway, fake, _, _ = bundle
            await open_u1(runtime)
            result = await runtime.send_message("查订单和衬衫")
            assert result["tasks"][0]["status"] == "done"
            assert len(result["tasks"][0]["facts"]) == 3
            request = fake.requests[-1]
            assert zone(request, DATA_NOTICE) == []
            updates = entities(tool_results(request, "dispatch_tasks")[-1]["business_data"])
            order = updates[("order", "#W0000001")]["result"]
            product = updates[("product", "p1")]["result"]
            assert order["address"]["address1"] == "原地址"
            assert order["items"][0]["options"] == {"颜色": "蓝"}
            assert product["variants"]["b"]["price"] == 30.0
            assert product["variants"]["b"]["options"] == {"颜色": "红"}
            assert all(
                "facts" not in task and "actions" not in task
                for task in zone(request, STATE_NOTICE)["tasks"]
            )
            requests_before = len(fake.requests)
            reads_before = sum(record.get("read_write") == "read" for record in gateway.records)
            await runtime.send_message("记住刚才的查询即可")
            for request in fake.requests[requests_before:]:
                assert entities(zone(request, DATA_NOTICE)) == updates
                assert tool_results(request, "dispatch_tasks") == []
            assert (
                sum(record.get("read_write") == "read" for record in gateway.records)
                == reads_before
            )
            saved = await runtime.snapshot()
            assert saved.tasks.snapshot() == result["tasks"]
            assert entities(list(saved.business_data.values())) == updates

    asyncio.run(scenario())


def test_confirmation_verification_publishes_new_order_with_frozen_turn_snapshot(runtime_factory):
    """提交核验的新订单经本轮工具返回可见，轮初数据区保持原值直到下一轮。"""

    def script(ids):
        """先准备取消，下一轮确认核验，再开一轮读取最新缓存。"""
        yield from prepare_script(ids, combined=True)
        yield tool("complete_plan", {"task_ids": ids()})
        yield tool("reply", {"message": "取消已核验。"})
        yield from followup_script(ids)

    children = [
        tool("get_order_details", {"order_id": "#W0000001"}),
        *child_script()[:-1],
        tool("finish", {"missing": ["旧追问"], "note": "旧待确认备注"}),
    ]

    async def scenario():
        """比较核验前后真实模型输入及可信缓存，不依赖模型声明完成。"""
        async with runtime_factory("commit", script, children) as bundle:
            runtime, _, gateway, fake, _, _ = bundle
            await open_u1(runtime)
            first = await runtime.send_message("请取消订单")
            assert first["tasks"][0]["status"] == "pending_confirm"
            assert first["tasks"][0]["missing"] == ["旧追问"]
            before = len(fake.requests)
            result = await runtime.send_message("确认")
            assert result["tasks"][0]["status"] == "done"
            assert result["tasks"][0]["missing"] == []
            assert result["tasks"][0]["note"] == ""
            for request in fake.requests[before:]:
                assert (
                    entities(zone(request, DATA_NOTICE))[("order", "#W0000001")]["result"]["status"]
                    == "pending"
                )
            updates = entities(
                tool_results(fake.requests[-1], "complete_plan")[-1]["business_data"]
            )
            assert updates[("order", "#W0000001")]["result"]["status"] == "cancelled"
            assert updates[("user", "u1")]["needs_refresh"] is True
            assert any(record["kind"] == "verification_read" for record in gateway.records)
            saved = await runtime.snapshot()
            assert saved.business_data["order:#W0000001"]["result"]["status"] == "cancelled"
            assert len(saved.tasks.snapshot()[0]["actions"]) == 2
            before = len(fake.requests)
            await runtime.send_message("好的")
            for request in fake.requests[before:]:
                assert entities(zone(request, DATA_NOTICE)) == updates
                assert tool_results(request, "complete_plan") in (
                    [],
                    [{"confirmation_results": [], "dispatched": [], "blocked": []}],
                )

    asyncio.run(scenario())


def test_cross_turn_history_contains_only_actual_dialogue_and_archive_keeps_failures(
    runtime_factory,
):
    """reply 参数、工具错误和调用仅留完整档案，跨轮上下文只带实际原话与回复。"""

    def script(ids):
        """先产生可恢复的参数错误，再正常回复并继续下一轮。"""
        yield tool("reply", {})
        yield tool("complete_plan", {})
        yield tool("reply", {"message": "请告诉我需要哪项帮助。"})
        yield from followup_script(ids)

    async def scenario():
        """用归档与下一轮实际模型输入对照验证历史过滤。"""
        async with runtime_factory("history", script) as bundle:
            runtime, _, _, fake, _, archive = bundle
            await open_u1(runtime)
            first = await runtime.send_message("我想咨询")
            entries = archive.messages("thread", "u1")
            assert any(entry["payload"].get("tool_calls") for entry in entries)
            assert any(
                entry["payload"]["role"] == "tool" and '"error"' in entry["payload"]["content"]
                for entry in entries
            )
            assert [entry["payload"] for entry in archive.dialogue_messages("thread", "u1")] == [
                {"role": "user", "content": "我想咨询"},
                {"role": "assistant", "content": first["reply"]},
            ]
            before = len(fake.requests)
            await runtime.send_message("我再想想")
            first_request = fake.requests[before]
            history = first_request["messages"][1:-3]
            assert history == [
                {"role": "user", "content": "我想咨询"},
                {"role": "assistant", "content": first["reply"]},
            ]
            assert "_ttwill_dialogue" not in str(first_request)
            assert all(message["role"] != "tool" for message in first_request["messages"])
            assert not any(message.get("tool_calls") for message in first_request["messages"])
            assert archive.messages("thread", "u1")[: len(entries)] == entries

    asyncio.run(scenario())


def test_new_session_clears_business_data_but_keeps_full_task_state(runtime_factory):
    """重验身份恢复任务时不带旧会话缓存，完整事实、操作引用与用户诉求仍可恢复。"""

    async def scenario():
        """关闭并重开真实检查点，检查恢复后首次请求的数据边界。"""
        children = [tool("get_order_details", {"order_id": "#W0000001"}), *child_script()]
        async with runtime_factory("old", prepare_script, children) as bundle:
            runtime = bundle[0]
            await open_u1(runtime)
            result = await runtime.send_message("请取消订单")
            saved = await runtime.snapshot()
            assert saved.business_data and result["tasks"][0]["facts"]
        async with runtime_factory("new", followup_script) as bundle:
            runtime, _, gateway, fake, _, _ = bundle
            opened = await open_u1(runtime)
            restored = await runtime.snapshot()
            assert opened["resumed"] is True
            assert restored.business_data == {}
            restored_task = restored.tasks.snapshot()[0]
            saved_task = saved.tasks.snapshot()[0]
            assert {key: value for key, value in restored_task.items() if key != "actions"} == {
                key: value for key, value in saved_task.items() if key != "actions"
            }
            assert restored_task["actions"][:-1] == saved_task["actions"]
            recovered = restored_task["actions"][-1]
            assert recovered["kind"] == "recovery"
            assert recovered["result"] == saved_task["actions"][0]["result"]
            assert restored.last_display == [] and restored.last_reply == ""
            await runtime.send_message("继续办理")
            assert all(zone(request, DATA_NOTICE) == [] for request in fake.requests)
            assert not any(record["kind"] == "commit" for record in gateway.records)

    asyncio.run(scenario())


def test_cancelled_and_failed_tasks_clear_stale_missing_and_note(runtime_factory):
    """终态清除旧追问与备注，检查点仍保留任务主体和失败原因。"""

    def script(ids):
        """先令子任务缺信息，再撤销；第二任务无实际操作便结束，须进入失败。"""
        yield from prepare_script(ids, combined=True)
        yield tool(
            "update_tasks",
            {"changes": [{"operation": "cancel", "task_id": ids()[0], "reason": "用户撤销"}]},
        )
        yield tool("complete_plan", {})
        yield tool("reply", {"message": "已停止办理。"})
        yield next(prepare_script(ids))
        yield tool("complete_plan", {"task_ids": [ids()[-1]]})
        yield tool("reply", {"message": "目前无法完成，请稍后重试。"})

    children = [
        tool("finish", {"missing": ["取消原因"], "note": "请补充旧原因"}),
        tool("finish", {"missing": [], "note": "失败时的旧备注"}),
    ]

    async def scenario():
        """真实分派与检查点分别覆盖取消和失败终态。"""
        async with runtime_factory("terminal", script, children) as bundle:
            runtime, _, _, _, _, _ = bundle
            await open_u1(runtime)
            waiting = await runtime.send_message("请帮我取消")
            assert waiting["tasks"][0]["status"] == "waiting_info"
            assert waiting["tasks"][0]["note"] == "请补充旧原因"
            cancelled = await runtime.send_message("不办了")
            assert cancelled["tasks"][0]["status"] == "cancelled"
            assert cancelled["tasks"][0]["missing"] == []
            assert cancelled["tasks"][0]["note"] == ""
            failed = await runtime.send_message("还是取消订单")
            assert failed["tasks"][-1]["status"] == "failed"
            assert failed["tasks"][-1]["missing"] == []
            assert failed["tasks"][-1]["note"] == ""
            assert failed["tasks"][-1]["reason"]
            assert (await runtime.snapshot()).tasks.snapshot() == failed["tasks"]

    asyncio.run(scenario())


def test_same_turn_reread_keeps_newest_entity_when_data_budget_shrinks(runtime_factory):
    """同轮真实读取 A、B、A 后，容量不足时保留最近重读的 A 并移出 B 正文。"""

    async def scenario():
        """用相近大小的真实订单结果验证重读时间，不依赖缓存映射的内部排序断言。"""
        async with runtime_factory("same-turn-lru", followup_script) as bundle:
            runtime, coordinator, gateway, _, _, _ = bundle
            await open_u1(runtime)
            state = runtime.state
            coordinator.begin_turn(state, "比较两张待发货订单，再看第一张")
            context = CallContext(state.session, "coordinator", "same-turn-reader")
            for order_id in ("#W0000001", "#W0000004"):
                offset = len(gateway.records)
                gateway.call(context, "get_order_details", {"order_id": order_id})
                coordinator._remember_business(state, gateway.records[offset:])
            assert len(state.business_data) == 2
            assert {entry["read_turn"] for entry in state.business_data.values()} == {state.turn}
            first = state.business_data["order:#W0000001"]
            single_chars = len(json.dumps([first], ensure_ascii=False, separators=(",", ":")))
            coordinator.data_max_chars = single_chars + 200
            assert coordinator.data_max_chars < len(
                json.dumps(
                    list(state.business_data.values()), ensure_ascii=False, separators=(",", ":")
                )
            )

            offset = len(gateway.records)
            latest = gateway.call(context, "get_order_details", {"order_id": "#W0000001"})
            coordinator._remember_business(state, gateway.records[offset:])

            assert state.business_data["order:#W0000001"]["result"] == latest
            removed = state.business_data.get("order:#W0000004")
            assert removed is None or (removed["needs_refresh"] is True and "result" not in removed)
            assert [
                record["args"]["order_id"]
                for record in gateway.records
                if record["action"] == "get_order_details"
            ] == ["#W0000001", "#W0000004", "#W0000001"]

    asyncio.run(scenario())

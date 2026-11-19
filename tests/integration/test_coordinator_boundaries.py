"""复现 60 的订单表示差异和 88 的对话、只读批量调用，验证确认与检查点边界。"""

import asyncio
import json
from copy import deepcopy

import pytest
from test_coordinator import (
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
from test_runtime import runtime_factory as runtime_factory


def batch(*responses: dict) -> dict:
    """把多个业务调用合成一次模型决策，并分配互不重复的调用标识。

    Args:
        *responses: 单工具假模型响应，按所需执行顺序传入。

    Returns:
        含完整合成思考字段和全部工具调用的假响应，供消息配对回归使用。
    """
    result = deepcopy(responses[0])
    message = result["choices"][0]["message"]
    message["reasoning_content"] = "这是离线测试构造的思考字段，用于验证完整保留。"
    message["tool_calls"] = []
    for index, response in enumerate(responses):
        call = deepcopy(response["choices"][0]["message"]["tool_calls"][0])
        call["id"] = f"batch_{index}"
        message["tool_calls"].append(call)
    return result


def guards(gateway) -> list[dict]:
    """读取协调者实际结束的拒绝事件，排除 span 开始和网关内部事件。"""
    return [
        event
        for event in map(json.loads, gateway.trace.path.read_text().splitlines())
        if event["event"] == "end"
        and event["kind"] == "guard"
        and event["name"].startswith("coordinator_")
    ]


def test_88_identity_then_question_does_not_require_empty_plan(runtime):
    """重放 88 的身份后追问结构，无展示待确认项时 reply 无须先空调计划。"""
    question = "您账户下有两个订单，请告知哪一个是书架订单，我再核对规格。"

    def script(state):
        """使用真实失败输出的 reply 结构，仅将账户改为独立测试数据。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("reply", {"message": question})

    coordinator, state, gateway, fake, child = runtime(script, [])
    result = turn(coordinator, state, "我要把书架改成 4 英尺，邮箱 u1@example.com")
    assert result["reply"] == question
    assert result["rounds"] == 2 and result["tasks"] == []
    assert child.requests == [] and guards(gateway) == []
    names = {schema["function"]["name"] for schema in fake.requests[-1]["tools"]}
    assert {"reply", "dispatch_tasks", "get_order_details", "get_product_details"} <= names


def test_88_two_order_reads_share_one_decision_and_keep_message_pairs(runtime, backend):
    """重放 88 的两个订单查询，顺序执行且只计一次决策，完整保留原消息及思考。"""
    # 88/6eeacb2d116f43a39a1e9b14aad686f5-1.json 的两个订单映射到小库中的前两单。
    reads = batch(
        tool("get_order_details", {"order_id": "#W0000001"}),
        tool("get_order_details", {"order_id": "#W0000002"}),
    )
    before = backend.export_snapshot()

    def script(state):
        """直接查询两单并解释，不创建只读派工任务。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield reads
        yield tool("reply", {"message": "已查到两单，第一单待发货，第二单已送达。"})

    coordinator, state, gateway, fake, child = runtime(script, [], max_rounds=3)
    result = turn(coordinator, state, "查一下我的两张订单")
    assert result["rounds"] == 3 and result["tasks"] == []
    assert child.requests == [] and guards(gateway) == []
    assert backend.export_snapshot() == before
    records = [record for record in gateway.records if record["action"] == "get_order_details"]
    assert [record["args"]["order_id"] for record in records] == ["#W0000001", "#W0000002"]
    messages = fake.requests[-1]["messages"]
    original = reads["choices"][0]["message"]
    index = messages.index(original)
    assert [message["tool_call_id"] for message in messages[index + 1 : index + 3]] == [
        "batch_0",
        "batch_1",
    ]
    assert [
        json.loads(message["content"])["order_id"] for message in messages[index + 1 : index + 3]
    ] == ["#W0000001", "#W0000002"]


@pytest.mark.parametrize(
    "coordination", ["reply", "update_tasks", "complete_plan", "dispatch_tasks"]
)
def test_read_mixed_with_coordination_rejects_entire_response(runtime, coordination):
    """只读和协调工具混在一次响应时整批拒绝，首个只读调用也不执行。"""
    arguments = {
        "reply": {"message": "请确认"},
        "update_tasks": {"changes": [cancel_task("#W0000001")]},
        "complete_plan": {},
        "dispatch_tasks": {"task_ids": ["guessed"]},
    }
    mixed = batch(
        tool("get_order_details", {"order_id": "#W0000001"}),
        tool(coordination, arguments[coordination]),
    )

    def script(state):
        """发出不合法的混合调用，再用合法回复结束。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield mixed
        yield tool("reply", {"message": "请说明具体需求。"})

    coordinator, state, gateway, fake, child = runtime(script, [])
    result = turn(coordinator, state, "你好")
    assert result["rounds"] == 3 and result["tasks"] == [] and child.requests == []
    assert not any(record["action"] == "get_order_details" for record in gateway.records)
    assert [event["name"] for event in guards(gateway)] == ["coordinator_format"]
    assert mixed["choices"][0]["message"] not in fake.requests[-1]["messages"]


def test_identity_batch_cannot_unlock_business_reads_inside_same_response(runtime):
    """认证前响应的权限固定，前一个查找成功不能让同批次后一个业务查询提权。"""
    reads = batch(
        tool("find_user_id_by_email", {"email": "u1@example.com"}),
        tool("get_order_details", {"order_id": "#W0000001"}),
    )

    def script(state):
        """混合身份查找与业务查询，之后正常结束对话。"""
        yield reads
        yield tool("reply", {"message": "已验证身份。"})

    coordinator, state, gateway, fake, _ = runtime(script, [])
    result = turn(coordinator, state, "u1@example.com")
    assert state.session.user_id == "u1" and result["rounds"] == 2
    assert not any(record["action"] == "get_order_details" for record in gateway.records)
    results = [
        json.loads(message["content"])
        for message in fake.requests[-1]["messages"]
        if message["role"] == "tool"
    ]
    assert results[0] == "u1"
    assert results[1]["reason"] == "当前阶段没有工具：get_order_details"


def test_multiple_identity_lookups_cannot_switch_user(runtime):
    """同一批只读身份查找仍受单用户锁约束，后一个查询不能切换会话归属。"""

    def script(state):
        """一次响应查找两个账户，然后读取第一位用户的资料。"""
        yield batch(
            tool("find_user_id_by_email", {"email": "u1@example.com"}),
            tool("find_user_id_by_email", {"email": "u2@example.com"}),
        )
        yield tool("get_user_details", {"user_id": "u1"})
        yield tool("reply", {"message": "已读取当前账户。"})

    coordinator, state, gateway, fake, _ = runtime(script, [])
    result = turn(coordinator, state, "查询账户")
    assert state.session.user_id == "u1" and result["rounds"] == 3
    messages = fake.requests[1]["messages"]
    outcomes = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
    assert outcomes[0] == "u1" and outcomes[1]["error"] == "identity_locked"
    assert len(guards(gateway)) == 1


def test_read_batch_error_keeps_remaining_results_paired(runtime):
    """只读批次中单项业务拒绝仍逐项返回错误和成功结果，不破坏消息配对。"""

    def script(state):
        """先读不属于本用户的订单，再读合法订单。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield batch(
            tool("get_order_details", {"order_id": "#W0000003"}),
            tool("get_order_details", {"order_id": "#W0000001"}),
        )
        yield tool("reply", {"message": "已找到当前用户的订单。"})

    coordinator, state, gateway, fake, _ = runtime(script, [])
    result = turn(coordinator, state, "查询订单")
    assert result["rounds"] == 3 and len(guards(gateway)) == 1
    outcomes = {
        message["tool_call_id"]: json.loads(message["content"])
        for message in fake.requests[-1]["messages"]
        if message["role"] == "tool"
    }
    assert "error" in outcomes["batch_0"]
    assert outcomes["batch_1"]["order_id"] == "#W0000001"


@pytest.mark.parametrize("plain_text", [True, False])
def test_previous_display_requires_plan_before_dispatch_or_reply(runtime, backend, plain_text):
    """有效上一展示仍先处理改口与确认，直接分派和两种回复都不能绕过授权。"""

    def script(state):
        """首轮直接分派；次轮先尝试绕过旧确认，最后合法提交旧项并准备新项。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()
        yield tool("update_tasks", {"changes": [cancel_task("#W0000004")]})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state, status="pending")})
        yield text_response("已处理") if plain_text else reply()
        assert len(child.requests) == 3
        assert not any(record["kind"] == "commit" for record in gateway.records)
        yield tool("complete_plan", {"task_ids": task_ids(state, status="pending")})
        yield reply()

    coordinator, state, gateway, fake, child = runtime(
        script, cancel_steps("#W0000001") + cancel_steps("#W0000004")
    )
    first = turn(coordinator, state, "取消第一单")
    assert first["rounds"] == 4 and len(first["displayed"]) == 1
    second = turn(coordinator, state, "确认")
    assert second["rounds"] == 5
    assert [task["status"] for task in second["tasks"]] == ["done", "pending_confirm"]
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "cancelled"
    assert backend.export_snapshot()["orders"]["#W0000004"]["status"] == "pending"
    assert len(second["confirmation_results"]) == 1
    assert len(guards(gateway)) == 2
    names = {schema["function"]["name"] for schema in fake.requests[4]["tools"]}
    assert "complete_plan" in names and not {"reply", "dispatch_tasks"} & names


def test_cancelled_previous_operation_no_longer_requires_plan(runtime, backend):
    """先撤销旧任务后可直接回复，已失效展示不再强制空计划，也绝不提交旧操作。"""

    def script(state):
        """先准备并展示取消订单，用户改口后先撤销任务再直接回复。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply()
        yield tool(
            "update_tasks",
            {
                "changes": [
                    {"operation": "cancel", "task_id": task_ids(state)[0], "reason": "用户不再取消"}
                ]
            },
        )
        yield reply()

    coordinator, state, gateway, _, _ = runtime(script, cancel_steps("#W0000001"))
    first = turn(coordinator, state, "取消这单")
    second = turn(coordinator, state, "可以，等等，不要取消了")
    assert second["rounds"] == 2 and second["tasks"][0]["status"] == "cancelled"
    assert second["confirmation_results"] == [] and guards(gateway) == []
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "pending"
    assert not any(record["kind"] == "commit" for record in gateway.records)
    assert first["displayed"]


def test_unshown_prepared_operation_can_be_redisplayed_without_plan(runtime, backend):
    """上一回复未展示的准备项不能被确认，下一轮可直接重新展示而无须空计划。"""

    def script(state):
        """先隐藏准备项，再直接回复展示，最后才以真实后续确认提交。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool("update_tasks", {"changes": [cancel_task("#W0000001")]})
        yield tool("dispatch_tasks", {"task_ids": task_ids(state)})
        yield reply([])
        yield reply()
        yield tool("complete_plan", {})
        yield reply()

    coordinator, state, gateway, _, _ = runtime(script, cancel_steps("#W0000001"))
    first = turn(coordinator, state, "取消这单")
    second = turn(coordinator, state, "确认")
    assert first["displayed"] == []
    assert second["rounds"] == 1 and second["tasks"][0]["status"] == "pending_confirm"
    assert second["confirmation_results"] == [] and len(second["displayed"]) == 1
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "pending"
    third = turn(coordinator, state, "确认")
    assert third["tasks"][0]["status"] == "done" and guards(gateway) == []


def test_60_order_format_matches_read_facts_through_checkpoints(runtime_factory):
    """重放 60 的无井号订单任务与规范查询事实，经检查点保存后仍正确判读任务完成。"""

    def script(ids):
        """沿用 60 的读任务字段结构，将业务标识映射到本地小库。"""
        yield tool("find_user_id_by_email", {"email": "u1@example.com"})
        yield tool(
            "update_tasks",
            {"changes": [add_task("get_order_details", " w0000001 ", "查询订单明细")]},
        )
        yield tool("dispatch_tasks", {"task_ids": ids()})
        yield tool("reply", {"message": "已查到订单。"})

    async def scenario():
        """使用真实 SQLite 检查点验证标准化目标、事实和恢复状态一致。"""
        children = [
            tool("get_order_details", {"order_id": "#W0000001"}),
            tool("finish", {"missing": [], "note": "已查询"}),
        ]
        async with runtime_factory("format", script, children) as bundle:
            runtime, _, gateway, _, _, _ = bundle
            result = await runtime.receive_message("我的邮箱是 u1@example.com，查询 w0000001")
            task = result["tasks"][0]
            assert task["status"] == "done" and task["order_id"] == "#W0000001"
            assert task["facts"][0]["args"]["order_id"] == "#W0000001"
            state = await runtime.snapshot()
            assert state.tasks.tasks[task["id"]].status == "done"
            assert not any(record["name"] == "dispatch_read_result" for record in guards(gateway))

    asyncio.run(scenario())


def test_persistent_read_batch_keeps_reasoning_and_pairs_after_identity_activation(runtime_factory):
    """认证接管及后续上下文编译保留完整批量消息，每个工具结果只归档一次。"""
    lookups = batch(
        tool("find_user_id_by_email", {"email": "u1@example.com"}),
        tool("find_user_id_by_email", {"email": "u1@example.com"}),
    )
    reads = batch(
        tool("get_order_details", {"order_id": "#W0000001"}),
        tool("get_order_details", {"order_id": "#W0000002"}),
    )

    def script(ids):
        """认证批次后直接查询两单，再回复，下一轮检查历史编译。"""
        yield lookups
        yield reads
        yield tool("reply", {"message": "已查询两单。"})
        yield tool("reply", {"message": "还需要什么帮助？"})

    async def scenario():
        """逐项检查消息配对、档案唯一性和编译后的完整思考字段。"""
        async with runtime_factory("batch", script) as bundle:
            runtime, _, gateway, fake, child, archive = bundle
            result = await runtime.receive_message("u1@example.com，查两单")
            assert result["rounds"] == 3 and child.requests == []
            first_turn_requests = list(fake.requests)
            await runtime.receive_message("谢谢")
            archived = [entry["payload"] for entry in archive.messages("thread", "u1")]
            for response in (lookups, reads):
                original = response["choices"][0]["message"]
                assert archived.count(original) == 1
                assert any(
                    request["messages"].count(original) == 1 for request in first_turn_requests
                )
                assert original not in fake.requests[-1]["messages"]
                index = archived.index(original)
                following = archived[index + 1 : index + 3]
                assert [message["tool_call_id"] for message in following] == ["batch_0", "batch_1"]
            assert guards(gateway) == []

    asyncio.run(scenario())

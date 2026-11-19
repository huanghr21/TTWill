"""验证持久化上下文精简与完整检查点、业务核验可以同时成立。"""

import asyncio
import json

from test_runtime import child_script, open_u1, prepare_script, tool
from test_runtime import runtime_factory as runtime_factory

from ttwill.core.context import STATE_NOTICE


def test_projected_requests_keep_full_checkpoint_and_confirmable_operation(runtime_factory):
    """准备和提交的模型输入均不含完整操作对象，检查点与核验事实仍保持完整。"""

    def script(ids):
        """模拟 88 将待确认任务列入 complete_plan，再按真实授权提交。"""
        yield from prepare_script(ids, combined=True)
        yield tool("complete_plan", {"task_ids": ids()})
        yield tool("reply", {"message": "办理完成"})

    async def scenario():
        """运行真实图检查点两轮，逐请求检查唯一状态和归档工具摘要。"""
        async with runtime_factory(
            "projection", script, child_script(), confirmation_decisions={"全部确认": "confirm"}
        ) as bundle:
            runtime, _, gateway, fake, _, archive = bundle
            await open_u1(runtime)
            first = await runtime.send_message("取消订单")
            saved = await runtime.snapshot()
            assert saved.tasks.snapshot() == first["tasks"]
            action = first["tasks"][0]["actions"][0]
            assert "items" in action["result"]["preview"]
            result = await runtime.send_message("全部确认")
            assert result["tasks"][0]["status"] == "done"
            assert sum(record["kind"] == "commit" for record in gateway.records) == 1
            assert "expected" in result["tasks"][0]["facts"][-1]["verification"]
            for request in fake.requests:
                states = [
                    json.loads(message["content"][len(STATE_NOTICE) :])
                    for message in request["messages"]
                    if isinstance(message.get("content"), str)
                    and message["content"].startswith(STATE_NOTICE)
                ]
                assert len(states) == 1
                for task in states[0]["tasks"]:
                    encoded = json.dumps(task)
                    assert '"preview"' not in encoded and '"expected"' not in encoded
                    assert "facts" not in task and "actions" not in task
                    assert all(
                        set(operation) == {"operation_id", "status"}
                        for operation in task.get("operations", [])
                    )
            entries = archive.messages("thread", "u1")
            assert not any(
                str(entry["payload"].get("content", "")).startswith(STATE_NOTICE)
                for entry in entries
            )
            for entry in entries:
                message = entry["payload"]
                if message["role"] == "tool":
                    content = message["content"]
                    assert '"preview"' not in content and '"facts"' not in content

    asyncio.run(scenario())

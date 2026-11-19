"""验证模型输入的任务投影不携带完整业务对象，也不改变可信状态。"""

import json
from copy import deepcopy

from ttwill.core.task_view import facts_view, task_brief, task_view


def detailed_task() -> dict:
    """构造含查询事实、准备记录与完整核验对象的任务快照。"""
    order = {
        "order_id": "#W5061109",
        "status": "pending",
        "items": [{"item_id": "old", "private_detail": "完整业务对象" * 1000}],
        "payment_history": [{"amount": 100}],
    }
    operation = {
        "operation_id": "op1",
        "action": "modify_pending_order_items",
        "status": "prepared",
        "args": {"order_id": "#W5061109", "item_ids": ["old"], "new_item_ids": ["new"]},
        "preview": order,
        "result": order,
    }
    return {
        "id": "t1",
        "status": "pending_confirm",
        "revision": 2,
        "instruction": "仅修改耳机规格",
        "reason": None,
        "missing": [],
        "operation_ids": ["op1"],
        "facts": [
            {"tool": "get_order_details", "args": {"order_id": "#W5061109"}, "result": order},
            {
                "verification": {
                    "verified": True,
                    "reason": None,
                    "expected": order,
                    "actual": order,
                }
            },
        ],
        "actions": [{"kind": "call", "tool": operation["action"], "result": operation}],
    }


def test_task_view_removes_nested_business_objects_and_preserves_identity():
    """事实及操作只投影关键标识和结果摘要，完整原状态保持独立且可核验。"""
    task = detailed_task()
    before = deepcopy(task)
    result = task_view(task)
    assert result["id"] == "t1" and result["revision"] == 2
    assert result["instruction"] == task["instruction"]
    assert result["operation_ids"] == ["op1"]
    assert result["facts"][0]["result_summary"]["order_id"] == "#W5061109"
    assert result["facts"][1]["verification"]["verified"] is True
    assert result["actions"][0]["operation_id"] == "op1"
    encoded = json.dumps(result, ensure_ascii=False)
    assert "完整业务对象" not in encoded
    assert '"preview"' not in encoded and '"expected"' not in encoded and '"actual"' not in encoded
    assert len(encoded) < len(json.dumps(task, ensure_ascii=False)) / 10
    result["operation_ids"].clear()
    result["actions"][0]["args"]["item_ids"].clear()
    assert task == before


def test_facts_view_keeps_query_identifiers_without_full_results():
    """派工事实保留查询方法、目标和可复查来源，不传完整订单。"""
    facts = detailed_task()["facts"]
    facts[0]["source"] = {"session_id": "s", "instance_id": "child", "record_index": 0}
    before = deepcopy(facts)
    result = facts_view(facts)
    assert result[0]["tool"] == "get_order_details"
    assert result[0]["args"] == {"order_id": "#W5061109"}
    assert result[0]["source"] == before[0]["source"]
    assert "result" not in result[0]
    assert facts == before


def test_task_brief_contains_only_execution_summary():
    """协调工具的任务摘要只给标识、状态和原因，不夹带任务内容。"""
    assert task_brief(detailed_task()) == {
        "task_id": "t1",
        "status": "pending_confirm",
        "reason": None,
    }

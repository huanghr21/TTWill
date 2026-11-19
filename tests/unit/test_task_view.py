"""验证模型输入的任务投影不携带完整业务对象，也不改变可信状态。"""

import json
from copy import deepcopy

import pytest

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
        "capability": "order_service",
        "action": "modify_pending_order_items",
        "order_id": "#W5061109",
        "status": "pending_confirm",
        "revision": 2,
        "instruction": "仅修改耳机规格",
        "condition": "价差不超过 100 元",
        "reason": None,
        "reason_internal": False,
        "missing": ["是否采用新规格"],
        "note": "请先解释价差",
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


@pytest.mark.parametrize(
    "status",
    ["pending", "conditional", "running", "waiting_info", "pending_confirm", "needs_verification"],
)
def test_task_view_keeps_active_request_without_business_results(status):
    """活跃任务保留请求和当前操作状态，业务事实与操作历史不进入模型视图。"""
    task = detailed_task()
    task["status"] = status
    before = deepcopy(task)
    result = task_view(task)
    assert result == {
        key: value
        for key, value in task.items()
        if key not in {"facts", "actions", "operation_ids"}
    } | {"operations": [{"operation_id": "op1", "status": "prepared"}]}
    encoded = json.dumps(result, ensure_ascii=False)
    assert "完整业务对象" not in encoded
    assert '"preview"' not in encoded and '"expected"' not in encoded and '"actual"' not in encoded
    assert len(encoded) < len(json.dumps(task, ensure_ascii=False)) / 10
    result["missing"].clear()
    result["operations"][0]["status"] = "伪造状态"
    assert task == before


@pytest.mark.parametrize("status", ["done", "cancelled", "failed"])
def test_terminal_task_view_is_only_one_status_line_without_mutating_task(status):
    """终态只展示业务目标与状态原因，完整任务中的备注、事实和操作仍保留。"""
    task = detailed_task()
    task["status"] = status
    task["reason"] = "本次办理已结束"
    before = deepcopy(task)

    result = task_view(task)

    assert result == {
        "id": "t1",
        "action": "modify_pending_order_items",
        "order_id": "#W5061109",
        "status": status,
        "reason": "本次办理已结束",
        "reason_internal": False,
    }
    assert task == before


def test_current_operations_use_latest_actual_record_and_exclude_old_revision():
    """当前修订只按有效操作引用取最新真实状态，旧修订和完整参数不进入投影。"""
    task = detailed_task()
    task["status"] = "needs_verification"
    task["actions"] = [
        {"operation_id": "old-op", "status": "prepared"},
        {"operation_id": "old-op", "status": "rejected"},
        *task["actions"],
        {"kind": "commit", "result": {"operation_id": "op1", "status": "committed"}},
    ]
    before = deepcopy(task)

    result = task_view(task)

    assert result["operations"] == [{"operation_id": "op1", "status": "committed"}]
    assert "old-op" not in json.dumps(result)
    assert "actions" not in result and "facts" not in result
    assert task == before


def test_task_status_does_not_fabricate_missing_operation_status():
    """没有实际操作记录时保留引用并明确状态未知，不根据任务状态猜测后端状态。"""
    task = detailed_task()
    task["actions"] = []

    assert task_view(task)["operations"] == [{"operation_id": "op1", "status": None}]


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

"""为模型生成任务、事实和动作的精简视图，完整执行数据留在可信状态中。"""

from copy import deepcopy
from typing import Any


def _result_summary(result: Any) -> Any:
    """保留结果的标识和简单状态，嵌套业务集合只报告条目数量。"""
    if isinstance(result, list):
        return {"count": len(result)}
    if not isinstance(result, dict):
        return result
    summary = {}
    for key, value in result.items():
        if key in {"expected", "actual", "preview", "result"}:
            continue
        if isinstance(value, (list, dict)):
            summary[key] = {"count": len(value)}
        elif key.endswith("_id") or key in {
            "id",
            "action",
            "status",
            "name",
            "price",
            "available",
            "verified",
            "reason",
            "error",
        }:
            summary[key] = value
    return summary


def facts_view(facts: list[dict]) -> list[dict]:
    """将可信事实投影成查询方法、目标和结果摘要。

    Args:
        facts: 代码保存的查询或核验事实，完整结果不修改。

    Returns:
        与原对象隔离的精简列表；查询细节须按保留的动作与参数重新读取。
    """
    result = []
    for fact in facts:
        if "verification" in fact:
            result.append({"verification": _result_summary(fact["verification"])})
            continue
        entry = {key: deepcopy(fact[key]) for key in ("tool", "args", "source") if key in fact}
        if "result" in fact:
            entry["result_summary"] = _result_summary(fact["result"])
        result.append(entry)
    return result


def task_view(task: dict) -> dict:
    """生成一次模型决策所需的任务状态，不携带完整业务对象与操作预览。

    Args:
        task: 来自当前可信任务状态的完整快照。

    Returns:
        保留请求、修订、状态、操作引用及精简事实和动作的独立字典；不能用于恢复。
    """
    view = {key: deepcopy(value) for key, value in task.items() if key not in {"facts", "actions"}}
    view["facts"] = facts_view(task.get("facts", []))
    view["actions"] = []
    for action in task.get("actions", []):
        operation = action.get("result")
        operation = operation if isinstance(operation, dict) else {}
        brief = {key: deepcopy(action[key]) for key in ("kind", "tool") if key in action}
        for key in ("operation_id", "status", "args", "reason"):
            if key in action or key in operation:
                brief[key] = deepcopy(action[key] if key in action else operation[key])
        if "tool" not in brief and "action" in operation:
            brief["tool"] = operation["action"]
        view["actions"].append(brief)
    return view


def task_brief(task: dict) -> dict:
    """给协调工具返回任务执行摘要，不附带整份任务状态。

    Args:
        task: 当前任务的完整或模型可见快照。

    Returns:
        仅包含 task_id、status、reason 的字典。
    """
    return {"task_id": task["id"], "status": task["status"], "reason": task.get("reason")}

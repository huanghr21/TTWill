"""用官方零售工具与独立业务库重放参考动作，完整记录终态差异。"""

import json
import os
from collections import Counter
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

from ttwill.backend import RetailBackend, request_key
from ttwill.bench.retail import RetailBundle

READ_ACTIONS = {
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_user_details",
    "get_order_details",
    "get_product_details",
    "get_item_details",
    "list_all_product_types",
}
WRITE_ACTIONS = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
}
GENERIC_ACTIONS = {"calculate", "transfer_to_human_agents"}


def _reference_types() -> tuple[type, type]:
    """导入官方零售数据与工具类，并设置 LiteLLM 使用本地价目表。

    Returns:
        按顺序包含以下对象的二元组：
            RetailDB: 官方零售数据库模型类。
            RetailTools: 绑定数据库后可执行参考动作的官方工具类。

    Raises:
        ModuleNotFoundError: 尚未安装包含 τ² 的 bench 可选依赖。
    """
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.retail.tools import RetailTools

    return RetailDB, RetailTools


def value_differences(expected: Any, actual: Any, path: str = "$") -> list[dict[str, Any]]:
    """递归比较两个值并记录差异所在字段。

    Args:
        expected: 官方结果或数据库终态中的参考值。
        actual: 本地后端对应的结果或终态值。
        path: 当前值的字段路径，根路径默认为 $。

    Returns:
        差异列表，每项包含 path、reference 和 backend；相等时为空列表。
        一侧缺少字典键或列表元素时，该侧记录为 None。
    """
    differences = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            item_path = f"{path}.{key}"
            if key not in expected or key not in actual:
                differences.append(
                    {"path": item_path, "reference": expected.get(key), "backend": actual.get(key)}
                )
            else:
                differences.extend(value_differences(expected[key], actual[key], item_path))
    elif isinstance(expected, list) and isinstance(actual, list):
        for index in range(max(len(expected), len(actual))):
            item_path = f"{path}[{index}]"
            if index >= len(expected) or index >= len(actual):
                differences.append(
                    {
                        "path": item_path,
                        "reference": expected[index] if index < len(expected) else None,
                        "backend": actual[index] if index < len(actual) else None,
                    }
                )
            else:
                differences.extend(value_differences(expected[index], actual[index], item_path))
    elif expected != actual:
        differences.append({"path": path, "reference": expected, "backend": actual})
    return differences


def _normalize_result(action: str, result: Any) -> Any:
    """通过官方数据模型补齐结果字段，并统一商品目录的 JSON 表示。

    Args:
        action: 产生结果的工具名称，用于选择对应的数据模型。
        result: 官方工具或本地后端返回的原始结果。

    Returns:
        可直接比较的结果；实体转换为普通字典，商品目录字符串解析为 JSON 值。

    Raises:
        ValueError: 结果无法解析为 JSON，或不符合对应的官方数据模型。
    """
    from tau2.domains.retail.data_model import Order, Product, User, Variant

    if hasattr(result, "model_dump"):
        return result.model_dump()
    if action == "list_all_product_types":
        return json.loads(result) if isinstance(result, str) else result
    if action in {"get_user_details", "modify_user_address"}:
        return User.model_validate(result).model_dump()
    if action == "get_product_details":
        return Product.model_validate(result).model_dump()
    if action == "get_item_details":
        return Variant.model_validate(result).model_dump()
    if action == "get_order_details" or action in WRITE_ACTIONS:
        return Order.model_validate(result).model_dump()
    return result


def _target_user(task: dict[str, Any], database: dict[str, Any]) -> str | None:
    """从参考动作的业务目标确定本题使用的测试身份。

    Args:
        task: 包含 evaluation_criteria.actions 的官方任务。
        database: 同源业务快照，用于查找目标订单的所属用户。

    Returns:
        动作中唯一的目标用户 ID；找不到用户目标时为 None。该身份只用于重放测试。

    Raises:
        ValueError: 同一任务的业务目标涉及多个用户。
    """
    users = set()
    for action in task["evaluation_criteria"]["actions"]:
        args = action["arguments"]
        if "user_id" in args:
            users.add(args["user_id"])
        if args.get("order_id") in database["orders"]:
            users.add(database["orders"][args["order_id"]]["user_id"])
    if len(users) > 1:
        raise ValueError(f"参考任务 {task['id']} 包含不同用户的业务目标")
    return next(iter(users), None)


def _backend_action(
    backend: RetailBackend, user_id: str | None, action: dict[str, Any], task_id: str
) -> Any:
    """将官方业务动作映射为本地查询或准备后提交。

    Args:
        backend: 当前任务专用的本地业务后端。
        user_id: 测试身份；身份查找动作可以传 None，其他动作需要明确的用户 ID。
        action: 包含 name、arguments 和 action_id 的参考动作。
        task_id: 参考任务 ID，与动作 ID 共同生成幂等请求标识。

    Returns:
        查询结果，或写操作提交记录中的业务结果。

    Raises:
        ValueError: 动作不受支持、缺少测试身份，或后端拒绝参数和业务操作。
    """
    name, args = action["name"], action["arguments"]
    if name in {"find_user_id_by_email", "find_user_id_by_name_zip"}:
        return getattr(backend, name)(**args)
    if user_id is None:
        raise ValueError("参考业务动作没有可确定的用户身份")
    if name in WRITE_ACTIONS:
        key = request_key(f"reference:{task_id}:{action['action_id']}", 1, name)
        prepared = backend.prepare(user_id, name, args, key)
        committed = backend.commit(user_id, prepared["operation_id"], args)
        return committed["result"]
    if name in READ_ACTIONS:
        return backend.query(user_id, name, args)
    raise ValueError(f"不支持的参考动作：{name}")


def _compare_action(
    reference: Any,
    backend: RetailBackend,
    user_id: str | None,
    action: dict[str, Any],
    task_id: str,
) -> dict[str, Any]:
    """执行两侧动作，比较结果并保存实际错误。

    Args:
        reference: 绑定本题参考数据库的官方工具实例。
        backend: 本题专用的本地业务后端。
        user_id: 本地后端使用的测试身份；身份查找和通用工具可为 None。
        action: 包含 name、arguments 和 action_id 的参考动作。
        task_id: 用于生成本地幂等请求标识的参考任务 ID。

    Returns:
        动作比较记录，包含名称、错误和字段差异。业务动作通过 matched 表示结果一致
        或两侧均拒绝；通用工具只记录官方结果，并将 compared 设为 False。
    """
    name, args = action["name"], action["arguments"]
    expected = actual = None
    reference_error = backend_error = None
    try:
        expected = _normalize_result(name, getattr(reference, name)(**args))
    except ValueError as exc:
        reference_error = str(exc)
    if name in GENERIC_ACTIONS:
        return {
            "action_id": action["action_id"],
            "name": name,
            "compared": False,
            "reason": "通用工具没有业务库副作用，仅记录官方执行结果",
            "reference_result": expected,
            "reference_error": reference_error,
        }
    try:
        actual = _normalize_result(name, _backend_action(backend, user_id, action, task_id))
    except ValueError as exc:
        backend_error = str(exc)
    both_rejected = reference_error is not None and backend_error is not None
    differences = value_differences(expected, actual)
    return {
        "action_id": action["action_id"],
        "name": name,
        "compared": True,
        "matched": both_rejected
        or (reference_error is None and backend_error is None and not differences),
        "reference_error": reference_error,
        "backend_error": backend_error,
        "differences": differences,
    }


def replay_task(bundle: RetailBundle, task: dict[str, Any], database_path: Path) -> dict[str, Any]:
    """在独立数据库中重放一题并比较全部动作与终态。

    调用会创建并初始化本题数据库、执行参考动作，结束时关闭本地后端连接池。

    Args:
        bundle: 包含初始业务数据及任务元信息的官方中文数据包。
        task: 要重放的参考任务，initial_state 需为 None。
        database_path: 本题使用的 SQLite 文件路径，业务库应尚未初始化。

    Returns:
        本题报告，包含任务标识、测试身份及以下比较结果：
            reference_hash: 官方工具执行后的数据库哈希。
            backend_hash: 本地后端执行后的数据库哈希。
            hash_equal: 两侧终态哈希是否一致。
            actions_equal: 所有参与比较的动作是否匹配。
            actions: 逐项动作比较记录。
            differences: 数据库终态的逐字段差异。

    Raises:
        ValueError: initial_state 非空、目标身份不唯一，或数据库已经初始化。
        ModuleNotFoundError: 尚未安装包含 τ² 的 bench 可选依赖。
    """
    RetailDB, RetailTools = _reference_types()

    if task["initial_state"] is not None:
        raise ValueError("当前重放器尚未接入 initial_state 初始化动作")
    reference_db = RetailDB.model_validate(deepcopy(bundle.database))
    reference = RetailTools(reference_db)
    backend = RetailBackend(database_path)
    try:
        backend.initialize(bundle.database)
        user_id = _target_user(task, bundle.database)
        actions = []
        for action in task["evaluation_criteria"]["actions"]:
            comparison = _compare_action(reference, backend, user_id, action, task["id"])
            actions.append(comparison)
        actual_db = RetailDB.model_validate(backend.export_snapshot())
        reference_hash, backend_hash = reference_db.get_hash(), actual_db.get_hash()
        return {
            "task_id": task["id"],
            "canonical_task_id": bundle.metadata[task["id"]]["canonical_task_id"],
            "fixture_user_id": user_id,
            "reference_hash": reference_hash,
            "backend_hash": backend_hash,
            "hash_equal": reference_hash == backend_hash,
            "actions_equal": all(action["matched"] for action in actions if action["compared"]),
            "actions": actions,
            "differences": value_differences(reference_db.model_dump(), actual_db.model_dump()),
        }
    finally:
        backend.close()


def compare_read_tools(bundle: RetailBundle, database_path: Path) -> list[dict[str, Any]]:
    """对比全部七个业务读工具，包括参考轨迹未调用的商品目录。

    Args:
        bundle: 提供用户、订单、商品及规格样本的官方中文数据包。
        database_path: 读工具校验专用的 SQLite 路径，业务库应尚未初始化。

    Returns:
        七项读工具比较记录，每项包含匹配结果、两侧错误及字段差异。

    Raises:
        ValueError: 数据库已经初始化，或初始数据无法导入。
        ModuleNotFoundError: 尚未安装包含 τ² 的 bench 可选依赖。
    """
    RetailDB, RetailTools = _reference_types()

    user = next(iter(bundle.database["users"].values()))
    order = next(
        order for order in bundle.database["orders"].values() if order["user_id"] == user["user_id"]
    )
    product = next(iter(bundle.database["products"].values()))
    variant = next(iter(product["variants"].values()))
    queries = [
        ("find_user_id_by_email", {"email": user["email"]}),
        ("find_user_id_by_name_zip", {**user["name"], "zip": user["address"]["zip"]}),
        ("get_user_details", {"user_id": user["user_id"]}),
        ("get_order_details", {"order_id": order["order_id"]}),
        ("get_product_details", {"product_id": product["product_id"]}),
        ("get_item_details", {"item_id": variant["item_id"]}),
        ("list_all_product_types", {}),
    ]
    reference = RetailTools(RetailDB.model_validate(deepcopy(bundle.database)))
    backend = RetailBackend(database_path)
    try:
        backend.initialize(bundle.database)
        return [
            _compare_action(
                reference,
                backend,
                user["user_id"],
                {"action_id": name, "name": name, "arguments": args},
                "read_tools",
            )
            for name, args in queries
        ]
    finally:
        backend.close()


def replay_tasks(bundle: RetailBundle, output_dir: Path) -> dict[str, Any]:
    """执行无模型一致性校验，并保存各题数据库与原始报告。

    Args:
        bundle: 来自固定提交的中文数据包，任务可为官方 50 题或已校验的开发题。
        output_dir: 本次运行的新目录，保存各题数据库、读工具校验库及 report.json。

    Returns:
        汇总报告，包含来源信息、工具覆盖、各题哈希与动作比较，以及七项读工具校验。
        所有哈希差异都计入不一致数量。

    Raises:
        ValueError: 未安装 τ²、其提交与数据来源不符，或任务数据不满足重放要求。
        FileExistsError: 输出目录已存在。
    """
    try:
        installed = distribution("tau2")
    except PackageNotFoundError as exc:
        raise ValueError("参考重放需要可选依赖，请运行 uv sync --locked --extra bench") from exc
    direct_url = installed.read_text("direct_url.json")
    installation = json.loads(direct_url) if direct_url is not None else {}
    if installation.get("vcs_info", {}).get("commit_id") != bundle.manifest["commit"]:
        raise ValueError("已安装 τ² 的提交与数据来源不一致，请按锁文件安装 bench 依赖")
    output_dir.mkdir(parents=True, exist_ok=False)
    coverage = Counter(
        action["name"] for task in bundle.tasks for action in task["evaluation_criteria"]["actions"]
    )
    tasks = [
        replay_task(bundle, task, output_dir / f"{task['id']}.sqlite") for task in bundle.tasks
    ]
    read_checks = compare_read_tools(bundle, output_dir / "read_tools.sqlite")
    report = {
        "source": bundle.manifest,
        "tau2_version": installed.version,
        "tau2_installation": installation,
        "model_calls": 0,
        "task_count": len(tasks),
        "hash_equal_count": sum(task["hash_equal"] for task in tasks),
        "actions_equal_count": sum(task["actions_equal"] for task in tasks),
        "write_tool_coverage": {name: coverage[name] for name in sorted(WRITE_ACTIONS)},
        "reference_action_coverage": dict(sorted(coverage.items())),
        "read_tool_checks": read_checks,
        "tasks": tasks,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report

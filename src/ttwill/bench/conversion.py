"""离线转换固定版本的开发题，并用官方五十题逐字段核对转换来源。"""

import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from copy import deepcopy
from decimal import Decimal
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ttwill.bench.retail import load_retail_bundle

CONVERSION_VERSION = "ttwill-retail-dev-v1"
REVIEW_VERSION = "ttwill-retail-localization-review-v1"
CONVERSION_FILES = {
    "tau2_retail_tasks.json": "tau2/domains/retail/tasks.json",
    "tau2_retail_database.json": "tau2/domains/retail/db.json",
    "tau2_retail_policy.md": "tau2/domains/retail/policy.md",
    "tau2_zh_retail_localization.json": "tau2/multilingual/zh/localization/retail.json",
}
ADDRESS_FIELDS = ("address1", "address2", "city", "state", "country", "zip")
MONEY_PATTERN = r"\$(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"


def _leaves(value: Any, path: tuple = ()):
    """按稳定字段路径遍历 JSON 叶子，用于逐字段检查与审阅清单。

    Args:
        value: 待遍历的 JSON 值。
        path: 当前字段路径，供递归调用延伸。

    Yields:
        叶子的字段路径和原始标量值。
    """
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaves(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _leaves(child, (*path, index))
    else:
        yield path, value


def _set(value: Any, path: tuple | list, replacement: Any) -> None:
    """按已有路径替换叶子，不创建任务中不存在的字段。"""
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


def field_differences(actual: Any, expected: Any, path: tuple = ()) -> list[dict]:
    """比较 JSON 的字段、类型与值，不忽略缺失字段或列表顺序。

    Args:
        actual: 转换生成的 JSON 值。
        expected: 官方发布的参考 JSON 值。
        path: 当前字段路径，供递归调用使用。

    Returns:
        含字段路径、实际值、期望值及差异类型的记录；完全相同时为空列表。
    """
    if type(actual) is not type(expected):
        return [{"path": list(path), "actual": actual, "expected": expected, "kind": "type"}]
    if isinstance(actual, dict):
        rows = []
        for key in sorted(actual.keys() | expected.keys()):
            if key not in actual or key not in expected:
                rows.append(
                    {
                        "path": [*path, key],
                        "actual": actual.get(key),
                        "expected": expected.get(key),
                        "kind": "missing_field",
                    }
                )
            else:
                rows.extend(field_differences(actual[key], expected[key], (*path, key)))
        return rows
    if isinstance(actual, list):
        rows = []
        if len(actual) != len(expected):
            rows.append(
                {
                    "path": list(path),
                    "actual": len(actual),
                    "expected": len(expected),
                    "kind": "length",
                }
            )
        for index, (left, right) in enumerate(zip(actual, expected)):
            rows.extend(field_differences(left, right, (*path, index)))
        return rows
    return (
        []
        if actual == expected
        else [{"path": list(path), "actual": actual, "expected": expected, "kind": "value"}]
    )


def _official_compile(source_dir: Path) -> tuple[dict, dict, dict]:
    """校验原样源文件，复用固定上游编译器还原数据库、政策与五十题。

    Args:
        source_dir: 固定来源清单与官方原样文件的目录。

    Returns:
        依次返回逐字段校验报告、来源清单和按文档 ID 索引的编译结果。

    Raises:
        ValueError: 来源哈希或已安装官方依赖提交不符，或编译校验失败。
    """
    bundle = load_retail_bundle(source_dir)
    files = bundle.manifest["files"]
    for name in CONVERSION_FILES:
        raw = (source_dir / name).read_bytes()
        expected = files[name]
        if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError(f"来源文件校验失败：{name}")
    try:
        installed = distribution("tau2")
    except PackageNotFoundError as exc:
        raise ValueError("开发题转换需要可选依赖，请运行 uv sync --locked --extra bench") from exc
    installation = json.loads(installed.read_text("direct_url.json") or "{}")
    if installation.get("vcs_info", {}).get("commit_id") != bundle.manifest["commit"]:
        raise ValueError("已安装 τ² 的提交与数据来源不一致，请按锁文件安装 bench 依赖")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    from tau2.data_model.tasks import Task
    from tau2.multilingual.factory.localization_artifact import read_localization_map
    from tau2.multilingual.factory.localization_compile import compile_localization

    with TemporaryDirectory(prefix="ttwill-localization-") as temporary:
        data_root = Path(temporary)
        for name, relative in CONVERSION_FILES.items():
            destination = data_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_dir / name, destination)
        subset = data_root / "tau2/task_subsets/retail_50.json"
        subset.parent.mkdir(parents=True)
        shutil.copyfile(source_dir / "retail_50.json", subset)
        proposal = read_localization_map(
            data_root / CONVERSION_FILES["tau2_zh_retail_localization.json"], data_root
        )
        documents = {document.id: document.content for document in compile_localization(proposal)}
    official = {task["id"]: task for task in bundle.tasks}
    metadata = {row["canonical_task_id"]: task_id for task_id, row in bundle.metadata.items()}
    checks = []
    for raw in documents["tasks"]:
        task = Task.model_validate(raw).model_copy(update={"id": metadata[raw["id"]]})
        converted = task.model_dump(mode="json")
        differences = field_differences(converted, official[task.id])
        checks.append({"task_id": task.id, "matched": not differences, "differences": differences})
    report = {
        "official_50": {
            "total": len(checks),
            "matched": sum(row["matched"] for row in checks),
            "tasks": checks,
        },
        "database_differences": field_differences(documents["database"], bundle.database),
        "policy_differences": field_differences(documents["policy:policy"], bundle.policy),
    }
    return report, bundle.manifest, documents


def _review_entries(path: Path) -> tuple[dict, str | None]:
    """读取绑定原文字段的本地审阅值，拒绝重复或缺少说明的条目。

    Args:
        path: 项目自行维护的本地化审阅 JSON 文件。

    Returns:
        按任务 ID 与字段路径索引的条目及文件字节哈希；文件不存在时返回空索引和 None。

    Raises:
        ValueError: 版本不支持、条目重复或缺少审阅说明。
    """
    if not path.exists():
        return {}, None
    raw = path.read_bytes()
    document = json.loads(raw)
    if document["version"] != REVIEW_VERSION:
        raise ValueError("开发题本地化审阅版本不受支持")
    entries = {}
    for row in document["entries"]:
        key = (row["canonical_task_id"], tuple(row["path"]))
        field_path = key[1]
        editable = _is_prose(field_path) or (
            len(field_path) == 5
            and field_path[:2] == ("evaluation_criteria", "actions")
            and field_path[3] == "arguments"
            and field_path[4] in {*ADDRESS_FIELDS, "first_name", "last_name", "email", "expression"}
        )
        if not editable:
            raise ValueError("本地化审阅不能修改机器标识、动作名称或奖励协议")
        if key in entries or not row["reason"] or not isinstance(row["native"], str):
            raise ValueError("开发题本地化审阅条目重复或缺少说明")
        entries[key] = row
    return entries, hashlib.sha256(raw).hexdigest()


def _scaled(value: str, scale: Decimal) -> str:
    """按基准固定倍数转换金额，保留整数和小数金额的区别。"""
    result = Decimal(value.replace(",", "")) * scale
    return str(result) if "." in value else str(int(result))


def _replace_text(text: str, pairs: dict[str, str]) -> str:
    """一次性替换完整实体，保护未知邮箱并按大小写区分州码。

    Args:
        text: 尚未本地化的任务或判分文本。
        pairs: 当前用户和商品上下文中已确定的原值与本地化值。

    Returns:
        仅替换完整实体后的文本；已有替换结果不会再次参与匹配。
    """
    if not pairs:
        return text
    folded = {key.casefold(): value for key, value in pairs.items() if key}
    entities = []
    for key in sorted(pairs, key=lambda key: (-len(key), key)):
        escaped = re.escape(key)
        entities.append(f"(?-i:{escaped})" if len(key) == 2 and key.isupper() else escaped)
    pattern = r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<![\w@])(?:" + "|".join(entities) + r")(?![\w@])"
    return re.sub(
        pattern, lambda match: folded.get(match.group().casefold(), match.group()), text, flags=re.I
    )


def _addresses(
    database: dict, localized: dict, user_id: str, alias: str
) -> list[tuple[dict, dict]]:
    """对齐同一用户的默认地址与订单地址，避免跨用户共用城市映射。

    Args:
        database: 官方原始数据库。
        localized: 同源的本地化数据库。
        user_id: 原始用户 ID。
        alias: 同一用户的本地化 ID。

    Returns:
        默认地址及各订单地址的原值、本地化值二元组列表。
    """
    rows = [(database["users"][user_id]["address"], localized["users"][alias]["address"])]
    for order_id in database["users"][user_id]["orders"]:
        rows.append(
            (database["orders"][order_id]["address"], localized["orders"][order_id]["address"])
        )
    return rows


def _canonical_caller(raw: dict, database: dict) -> str:
    """从参考身份、订单归属和已知姓名邮编确定唯一调用者。

    Args:
        raw: 尚未本地化的官方任务。
        database: 官方原始数据库。

    Returns:
        具有明确身份或业务归属证据的唯一用户 ID。

    Raises:
        ValueError: 任务证据无法确定唯一用户。
    """
    candidates = set()
    for action in raw["evaluation_criteria"]["actions"]:
        arguments = action["arguments"]
        if arguments.get("user_id") in database["users"]:
            candidates.add(arguments["user_id"])
        if arguments.get("order_id") in database["orders"]:
            candidates.add(database["orders"][arguments["order_id"]]["user_id"])
        for user_id, user in database["users"].items():
            if action["name"] == "find_user_id_by_email" and arguments["email"] == user["email"]:
                candidates.add(user_id)
            if (
                action["name"] == "find_user_id_by_name_zip"
                and all(
                    arguments[name] == user["name"][name] for name in ("first_name", "last_name")
                )
                and arguments["zip"] == user["address"]["zip"]
            ):
                candidates.add(user_id)
    if not candidates:
        known = raw["user_scenario"]["instructions"]["known_info"] or ""
        names = set()
        for user_id, user in database["users"].items():
            full_name = f"{user['name']['first_name']} {user['name']['last_name']}"
            if user_id in known or user["email"] in known:
                candidates.add(user_id)
            if full_name in known:
                names.add(user_id)
        if not candidates:
            candidates = names
            if len(names) > 1:
                candidates = {
                    user_id
                    for user_id in names
                    if re.search(
                        r"(?<!\d)"
                        + re.escape(database["users"][user_id]["address"]["zip"])
                        + r"(?!\d)",
                        known,
                    )
                }
    if len(candidates) != 1:
        raise ValueError(f"开发题 {raw['id']} 无法确定唯一的结构化调用者")
    return next(iter(candidates))


def _is_prose(path: tuple) -> bool:
    """判断字段是否属于用户场景正文或判分文本。"""
    return bool(path) and (
        path[0] == "user_scenario"
        and path[-1] != "domain"
        or any(part in {"nl_assertions", "communicate_info"} for part in path)
    )


def compile_development_tasks(source_dir: Path, overrides_path: Path | None = None) -> dict:
    """生成开发题草稿、官方逐字段校验和需审阅的原文对照，不写产物。

    Args:
        source_dir: 已固定来源清单及八份官方原样文件的目录。
        overrides_path: 绑定原文的本地审阅补充；默认读取 data/tasks 下的同名文件。

    Returns:
        dataset 为完整开发题数据，report 为校验与未解决项，prose_review 为文本前后对照。

    Raises:
        ValueError: 来源校验、官方编译、任务身份或审阅原文检查失败。
    """
    report, manifest, documents = _official_compile(source_dir)
    from tau2.data_model.tasks import Task
    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.retail.tools import RetailTools
    from tau2.multilingual.factory.localization_inventory import _Inventory
    from tau2.multilingual.names import clarify_retail_name

    canonical = json.loads((source_dir / "tau2_retail_database.json").read_text())
    source_tasks = json.loads((source_dir / "tau2_retail_tasks.json").read_text())
    mapping = json.loads((source_dir / "tau2_zh_retail_localization.json").read_text())
    frozen = set(json.loads((source_dir / "retail_50.json").read_text())["task_ids"])
    aliases = {row["canonical"]: row["localized"] for row in mapping["user_id_revision"]["aliases"]}
    entries, review_hash = _review_entries(
        overrides_path or source_dir.parent / "tasks/localization_overrides.json"
    )
    scale = Decimal(mapping["settings"]["currency_scale"])
    authored = {entry["id"]: entry for entry in mapping["entries"]}
    inventory = _Inventory("retail", canonical)
    inventory.walk("tasks", source_tasks, [])
    mapped = {}
    for entry in inventory.entries.values():
        if entry.id not in authored:
            continue
        for occurrence in entry.occurrences:
            index, *path = occurrence.path
            mapped[(source_tasks[index]["id"], tuple(path))] = authored[entry.id]["forms"]["native"]
    localized = documents["database"]
    tools = RetailTools(RetailDB.model_validate(canonical))
    localized_tools = RetailTools(RetailDB.model_validate(localized))
    exact_prose = {}
    candidates = defaultdict(set)
    contacts = {}
    for entry in mapping["entries"]:
        source, native = entry["source"], entry["forms"]["native"]
        if not isinstance(source, str) or not isinstance(native, str):
            continue
        if entry["category"] == "prose":
            exact_prose[source] = native
        if entry["category"] == "product_name":
            candidates[source.casefold()].add(native)
        if entry["category"] == "contact":
            contacts[source] = native
    common = {key: next(iter(values)) for key, values in candidates.items() if len(values) == 1}
    common.update(contacts)
    common.update(aliases)
    published = json.loads((source_dir / "tau2_zh_retail.json").read_text())
    by_id = {task["id"]: task for task in source_tasks}
    gender_by_user = {}
    for row in published["metadata"].values():
        user_id = _canonical_caller(by_id[row["canonical_task_id"]], canonical)
        if user_id in gender_by_user and gender_by_user[user_id] != row["caller_gender"]:
            raise ValueError("官方任务对同一用户给出了冲突的性别标签")
        gender_by_user[user_id] = row["caller_gender"]
    caller_by_task = {}
    results, reviews, unresolved, used = [], [], [], set()
    for raw in source_tasks:
        caller = tools.db.users[_canonical_caller(raw, canonical)]
        caller_by_task[raw["id"]] = caller.user_id
        user_id, alias = caller.user_id, aliases[caller.user_id]
        target = localized["users"][alias]
        addresses = _addresses(canonical, localized, user_id, alias)
        pairs = dict(common)
        pairs[f"{caller.name.first_name} {caller.name.last_name}"] = (
            target["name"]["last_name"] + target["name"]["first_name"]
        )
        pairs[caller.name.first_name] = target["name"]["first_name"]
        pairs[caller.name.last_name] = target["name"]["last_name"]
        address_values = defaultdict(set)
        for before, after in addresses:
            for field in ADDRESS_FIELDS:
                if before[field]:
                    address_values[before[field]].add(after[field])
        pairs.update(
            {key: next(iter(values)) for key, values in address_values.items() if len(values) == 1}
        )
        converted = deepcopy(raw)
        money_values = {
            Decimal(match.replace(",", ""))
            for assertion in raw["evaluation_criteria"].get("nl_assertions") or []
            for match in re.findall(MONEY_PATTERN, assertion)
        }
        for path, value in _leaves(raw):
            if not isinstance(value, str):
                continue
            key = (raw["id"], path)
            prose = _is_prose(path)
            if key in entries:
                row = entries[key]
                if row["source"] != value:
                    raise ValueError(f"本地化审阅原文已变化：{key}")
                if (
                    path[-1] == "expression"
                    and raw["evaluation_criteria"]["actions"][path[2]]["name"] != "calculate"
                ):
                    raise ValueError("表达式本地化审阅仅允许修改 calculate 工具的 expression 参数")
                replacement = row["native"]
                used.add(key)
            elif key in mapped:
                replacement = mapped[key]
            elif prose:
                if value in exact_prose:
                    replacement = exact_prose[value]
                else:
                    replacement = _replace_text(value, pairs)
                    replacement = re.sub(
                        MONEY_PATTERN,
                        lambda match: "CNY " + _scaled(match[1], scale),
                        replacement,
                    )
                    if "communicate_info" in path and re.fullmatch(r"\d+(?:\.\d+)?", value):
                        if Decimal(value) in money_values:
                            replacement = _scaled(value, scale)
            elif "arguments" in path:
                field = path[-1]
                if field == "user_id":
                    replacement = aliases[value]
                elif field in {"first_name", "last_name"}:
                    replacement = (
                        target["name"][field] if value == getattr(caller.name, field) else value
                    )
                elif field == "email":
                    replacement = contacts.get(value, value)
                    if value not in contacts:
                        unresolved.append(
                            {
                                "canonical_task_id": raw["id"],
                                "path": list(path),
                                "source": value,
                                "reason": "未映射的身份邮箱需审阅错误值语义",
                            }
                        )
                elif field in ADDRESS_FIELDS:
                    arguments = raw["evaluation_criteria"]["actions"][path[2]]["arguments"]
                    matches = [
                        after
                        for before, after in addresses
                        if all(
                            arguments.get(name) == before[name]
                            for name in ADDRESS_FIELDS
                            if name in arguments
                        )
                    ]
                    values = {address[field] for address in matches}
                    if len(values) == 1:
                        replacement = next(iter(values))
                    else:
                        replacement = value
                        unresolved.append(
                            {
                                "canonical_task_id": raw["id"],
                                "path": list(path),
                                "source": value,
                                "reason": "地址或错误邮编需明确审阅",
                            }
                        )
                else:
                    replacement = value
            else:
                replacement = value
            _set(converted, path, replacement)
            if prose and raw["id"] not in frozen:
                reviews.append(
                    {
                        "canonical_task_id": raw["id"],
                        "path": list(path),
                        "source": value,
                        "native": replacement,
                        "method": "review"
                        if key in entries
                        else "official"
                        if key in mapped or value in exact_prose
                        else "entities",
                    }
                )
        converted["id"] = raw["id"] + "_zh_identity"
        task = clarify_retail_name(
            Task.model_validate(converted),
            caller_id=alias,
            caller=localized_tools.db.users[alias],
            name_order=mapping["settings"]["name_order"],
        )
        results.append(task.model_dump(mode="json"))
    if used != entries.keys():
        raise ValueError(f"本地化审阅包含无效字段：{sorted(entries.keys() - used)}")
    official = {task["id"]: task for task in published["tasks"]}
    checks = []
    for task in results:
        if task["id"] not in official:
            continue
        differences = field_differences(task, official[task["id"]])
        checks.append(
            {"task_id": task["id"], "matched": not differences, "differences": differences}
        )
    report["source_compilation_50_matches"] = report["official_50"]["matched"]
    report["official_50"] = {
        "total": len(checks),
        "matched": sum(row["matched"] for row in checks),
        "tasks": checks,
    }
    results = [task for task in results if task["id"] not in official]
    report["unresolved"] = unresolved
    report["development_tasks"] = len(results)
    report["unknown_caller_gender"] = [
        task["id"]
        for task in results
        if gender_by_user.get(caller_by_task[task["id"].removesuffix("_zh_identity")]) is None
    ]
    dataset = {
        "conversion_version": CONVERSION_VERSION,
        "source": {
            "commit": manifest["commit"],
            "files": {
                name: {"sha256": row["sha256"], "bytes": row["bytes"]}
                for name, row in manifest["files"].items()
            },
            "localization_overrides_sha256": review_hash,
        },
        "tasks": results,
        "metadata": {
            task["id"]: {
                "canonical_task_id": task["id"].removesuffix("_zh_identity"),
                "caller_gender": gender_by_user.get(
                    caller_by_task[task["id"].removesuffix("_zh_identity")]
                ),
                "persona_id": None,
            }
            for task in results
        },
    }
    return {"dataset": dataset, "report": report, "prose_review": reviews}


def build_development_tasks(source_dir: Path, output_dir: Path) -> dict:
    """生成开发集及转换报告；存在官方字段差异或未解决语义时只写报告。

    Args:
        source_dir: 官方原样文件与来源校验清单目录。
        output_dir: 接收 dev_64.json 和 conversion_report.json 的目录。

    Returns:
        官方五十题逐字段比对、数据库与政策比对及开发题转换检查结果。

    Raises:
        ValueError: 官方校验未通过或仍存在未审阅的身份、地址转换。
        FileExistsError: 输出目录已有开发题或转换报告，需另选新目录保留已有产物。
    """
    for name in ("dev_64.json", "conversion_report.json"):
        if (output_dir / name).exists():
            raise FileExistsError(f"转换产物已存在：{output_dir / name}")
    result = compile_development_tasks(source_dir, output_dir / "localization_overrides.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = result["report"]
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if (
        report["official_50"]["matched"] != 50
        or report["database_differences"]
        or report["policy_differences"]
        or report["unresolved"]
    ):
        raise ValueError("开发题转换仍有差异或未审阅字段，详见 conversion_report.json")
    (output_dir / "dev_64.json").write_text(
        json.dumps(result["dataset"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report

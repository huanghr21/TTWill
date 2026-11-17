"""读取开发任务并按固定规则生成可复现的开发快速集。"""

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from ttwill.bench.conversion import CONVERSION_VERSION
from ttwill.bench.replay import WRITE_ACTIONS
from ttwill.bench.retail import RetailBundle, load_retail_bundle

QUICK_ALGORITHM = "write-type-count-sha256-v1"


def _digest(raw: bytes) -> dict[str, Any]:
    """记录文件字节数与 SHA256，供来源校验和运行报告复用。"""
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _development(raw: bytes) -> dict[str, Any]:
    """读取转换产物并核对转换版本、64 题和元信息的一一对应关系。

    Args:
        raw: dev_64.json 的完整文件字节。

    Returns:
        已解析且通过版本、任务 ID 和原题 ID 唯一性检查的开发集对象。

    Raises:
        ValueError: 转换版本不符、题数错误，或任务与元信息不能唯一对应。
    """
    development = json.loads(raw)
    if development["conversion_version"] != CONVERSION_VERSION:
        raise ValueError("开发集转换版本与当前加载器不一致")
    ids = [task["id"] for task in development["tasks"]]
    if len(ids) != 64 or len(set(ids)) != 64:
        raise ValueError("开发集必须包含 64 道 ID 不重复的任务")
    if set(development["metadata"]) != set(ids):
        raise ValueError("开发任务与元信息必须一一对应")
    canonical_ids = [development["metadata"][task_id]["canonical_task_id"] for task_id in ids]
    if len(set(canonical_ids)) != 64:
        raise ValueError("开发任务不能重复引用同一道原题")
    return development


def _load_development(tasks_dir: Path) -> tuple[bytes, dict[str, Any]]:
    """读取开发集并核对参与转换的审阅补充规则文件。

    Args:
        tasks_dir: 开发集及可选 localization_overrides.json 所在目录。

    Returns:
        依次返回 dev_64.json 原始字节与解析后的开发集。

    Raises:
        ValueError: 审阅补充规则的存在状态或 SHA256 与转换记录不符。
    """
    raw = (tasks_dir / "dev_64.json").read_bytes()
    development = _development(raw)
    overrides = tasks_dir / "localization_overrides.json"
    actual = hashlib.sha256(overrides.read_bytes()).hexdigest() if overrides.exists() else None
    if actual != development["source"]["localization_overrides_sha256"]:
        raise ValueError("开发集审阅补充规则的来源摘要不一致")
    return raw, development


def load_task_bundle(source_dir: Path, tasks_dir: Path, dataset: str) -> RetailBundle:
    """按任务集加载同源的中文业务数据，并验证开发题与快速清单的来源。

    Args:
        source_dir: 含固定上游数据及来源清单的目录。
        tasks_dir: 含 dev_64.json 和冻结 dev_quick.json 的目录。
        dataset: retail_50、dev_64 或 dev_quick。

    Returns:
        对应任务及元信息的中文数据包；manifest 记录实际任务文件摘要与任务 ID。

    Raises:
        FileNotFoundError: 所需来源文件或任务清单缺失。
        ValueError: 任务集名称、来源摘要、任务隔离、抽样算法或冻结清单不符合约定。
    """
    if dataset not in {"retail_50", "dev_64", "dev_quick"}:
        raise ValueError(f"未知任务集：{dataset}")
    bundle = load_retail_bundle(source_dir)
    if dataset == "retail_50":
        return bundle
    raw, development = _load_development(tasks_dir)
    source = development["source"]
    if source["commit"] != bundle.manifest["commit"]:
        raise ValueError("开发集来源提交与中文数据包不一致")
    if set(source["files"]) != set(bundle.manifest["files"]):
        raise ValueError("开发集来源文件与固定清单不一致")
    for name, recorded in source["files"].items():
        expected = bundle.manifest["files"][name]
        if recorded != {key: expected[key] for key in ("sha256", "bytes")}:
            raise ValueError(f"开发集来源摘要与固定清单不一致：{name}")
        if _digest((source_dir / name).read_bytes()) != recorded:
            raise ValueError(f"开发集来源文件校验失败：{name}")
    upstream = json.loads((source_dir / "tau2_retail_tasks.json").read_bytes())
    evaluation_ids = {item["canonical_task_id"] for item in bundle.metadata.values()}
    canonical_ids = {item["canonical_task_id"] for item in development["metadata"].values()}
    if canonical_ids != {task["id"] for task in upstream} - evaluation_ids:
        raise ValueError("开发任务必须是官方任务全集与 retail_50 的差集")

    tasks = development["tasks"]
    metadata = development["metadata"]
    task_files = {"dev_64.json": _digest(raw)}
    if dataset == "dev_quick":
        quick_raw = (tasks_dir / "dev_quick.json").read_bytes()
        quick = json.loads(quick_raw)
        if quick["source_sha256"] != task_files["dev_64.json"]["sha256"]:
            raise ValueError("快速集来源摘要与当前 dev_64.json 不一致")
        expected_quick = _quick_manifest(
            development, quick["source_sha256"], seed=quick["seed"], size=quick["size"]
        )
        if quick != expected_quick:
            raise ValueError("快速集清单与固定抽样算法不一致")
        by_id = {task["id"]: task for task in tasks}
        tasks = [by_id[task_id] for task_id in quick["task_ids"]]
        metadata = {task["id"]: metadata[task["id"]] for task in tasks}
        task_files["dev_quick.json"] = _digest(quick_raw)
    manifest = {
        **bundle.manifest,
        "subset": dataset,
        "conversion_version": development["conversion_version"],
        "localization_overrides_sha256": source["localization_overrides_sha256"],
        "task_files": task_files,
        "task_ids": [task["id"] for task in tasks],
    }
    return replace(bundle, tasks=tasks, metadata=metadata, manifest=manifest)


def _write_names(task: dict[str, Any]) -> list[str]:
    """提取参考动作中的写工具名称，保留重复调用以统计次数。"""
    return [
        action["name"]
        for action in task["evaluation_criteria"]["actions"]
        if action["name"] in WRITE_ACTIONS
    ]


def _stratum(task: dict[str, Any]) -> str:
    """按写类型与次数分层，将多类型和三次以上的稀疏组合归组。"""
    names = _write_names(task)
    kind = "read_only" if not names else names[0] if len(set(names)) == 1 else "mixed"
    count = "3+" if len(names) >= 3 else str(len(names))
    return f"{kind}:{count}"


def _rank(seed: int, value: str) -> str:
    """生成与输入列表顺序和 Python 随机库版本无关的固定排序键。"""
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _coverage(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """统计实际覆盖的写工具、只读题以及多写题数量。

    Args:
        tasks: 含参考动作的任务列表。

    Returns:
        按工具名排列的覆盖与缺失列表、只读题数、多写题数和写次数分布。
    """
    names = [_write_names(task) for task in tasks]
    available = set().union(*map(set, names))
    counts = Counter(len(item) for item in names)
    return {
        "write_tools": sorted(available),
        "missing_write_tools": sorted(WRITE_ACTIONS - available),
        "read_only_tasks": counts[0],
        "multi_write_tasks": sum(count for size, count in counts.items() if size > 1),
        "write_count_distribution": {str(size): counts[size] for size in sorted(counts)},
    }


def _quick_manifest(
    development: dict[str, Any], source_sha256: str, *, seed: int, size: int
) -> dict[str, Any]:
    """按每层保底加剩余容量比例抽样，构造可重算的冻结清单。

    Args:
        development: 包含 tasks 的开发集文件内容。
        source_sha256: 开发集文件原始字节的 SHA256。
        seed: 用于层内选择和配额平局处理的固定整数种子。
        size: 快速集题数，取值为 15～20。

    Returns:
        包含算法、来源摘要、种子、各层配额和任务 ID 的完整清单。

    Raises:
        ValueError: 参数非法、任务 ID 重复、题数不足、层数超过预算或覆盖不全。
    """
    if type(seed) is not int or type(size) is not int or not 15 <= size <= 20:
        raise ValueError("快速集要求整数种子且题数在 15～20 之间")
    tasks = development["tasks"]
    ids = [task["id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("开发任务 ID 不得重复")
    if len(tasks) < size:
        raise ValueError("开发任务数量不足以生成指定规模的快速集")
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        strata[_stratum(task)].append(task)
    if len(strata) > size:
        raise ValueError("快速集题数不足以为每个分层保留一道题")
    for key, members in strata.items():
        members.sort(key=lambda task: (_rank(seed, f"{key}:{task['id']}"), task["id"]))

    capacities = {key: len(members) - 1 for key, members in strata.items()}
    remaining = size - len(strata)
    total_capacity = sum(capacities.values())
    quotas = {key: 1 for key in strata}
    if remaining:
        for key, capacity in capacities.items():
            quotas[key] += remaining * capacity // total_capacity
        remainder_order = sorted(
            strata,
            key=lambda key: (
                -(remaining * capacities[key] % total_capacity),
                _rank(seed, key),
                key,
            ),
        )
        for key in remainder_order[: size - sum(quotas.values())]:
            quotas[key] += 1
    selected = [task for key in sorted(strata) for task in strata[key][: quotas[key]]]
    population_coverage, selected_coverage = _coverage(tasks), _coverage(selected)
    if (
        population_coverage["write_tools"] != selected_coverage["write_tools"]
        or bool(population_coverage["read_only_tasks"])
        != bool(selected_coverage["read_only_tasks"])
        or bool(population_coverage["multi_write_tasks"])
        != bool(selected_coverage["multi_write_tasks"])
    ):
        raise ValueError("快速集未覆盖开发集中已有的写工具、只读或多写任务")
    return {
        "name": "dev_quick",
        "source_dataset": "dev_64",
        "source_sha256": source_sha256,
        "algorithm": QUICK_ALGORITHM,
        "seed": seed,
        "size": size,
        "stratifier": {
            "write_type": "单一写工具名；多种写工具为 mixed；无写工具为 read_only",
            "write_count": ["0", "1", "2", "3+"],
            "allocation": "每层先取 1 题；余量按各层剩余容量比例，以最大余数法分配",
            "selection": "按 SHA256(seed:层名:任务ID) 排序；配额平局按 SHA256(seed:层名)",
        },
        "population_coverage": population_coverage,
        "selected_coverage": selected_coverage,
        "strata": [
            {
                "key": key,
                "population": len(strata[key]),
                "selected": quotas[key],
                "task_ids": [task["id"] for task in strata[key][: quotas[key]]],
            }
            for key in sorted(strata)
        ],
        "task_ids": [task["id"] for task in selected],
    }


def generate_quick_set(tasks_dir: Path, *, seed: int = 0, size: int = 20) -> dict[str, Any]:
    """从已转换的开发集生成固定快速清单，使用独占写入避免覆盖旧清单。

    Args:
        tasks_dir: 包含 dev_64.json 的任务目录，也是 dev_quick.json 的保存目录。
        seed: 固定抽样种子，默认 0。
        size: 选取题数，允许 15～20，默认 20。

    Returns:
        已写入 dev_quick.json 的清单，包含来源摘要、分层配额和任务 ID。

    Raises:
        FileNotFoundError: 开发集文件不存在。
        FileExistsError: 已有快速集清单，禁止通过重跑抽样覆盖。
        ValueError: 开发集、抽样参数或覆盖结果不符合要求。
    """
    raw, development = _load_development(tasks_dir)
    manifest = _quick_manifest(development, hashlib.sha256(raw).hexdigest(), seed=seed, size=size)
    with (tasks_dir / "dev_quick.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest

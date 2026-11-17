"""读取已固定版本的中文零售数据，不加载 τ² 运行依赖。"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TAU2_COMMIT = "c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4"
SOURCE_FILES = ("tau2_zh_retail.json", "retail_50.json", "LICENSE.tau2", "tau2_zh_pack.yaml")


@dataclass(frozen=True)
class RetailBundle:
    """提供同源数据库、参考任务、政策及可写入报告的来源信息。

    Attributes:
        database: 包含 users、products、orders 的初始业务快照。
        tasks: 中文参考任务列表，包含动作与评测条件。
        policy: 中文零售业务政策正文。
        metadata: 按中文任务 ID 索引的元信息，包含对应的原始任务 ID。
        settings: 数据包的语言等编译设置。
        manifest: 固定提交、来源文件及其校验值的清单。
        language_pack: 校验过的官方中文语言包 YAML 原文，仅由评测适配层解析。
    """

    database: dict[str, Any]
    tasks: list[dict[str, Any]]
    policy: str
    metadata: dict[str, Any]
    settings: dict[str, Any]
    manifest: dict[str, Any]
    language_pack: str


def load_retail_bundle(source_dir: Path) -> RetailBundle:
    """校验来源清单并读取官方中文包。

    Args:
        source_dir: 包含 manifest.json、中文数据包、任务清单和许可证的目录。

    Returns:
        同源的数据库、参考任务、政策、设置及来源信息；业务数据使用普通字典表示。

    Raises:
        FileNotFoundError: 来源清单或其记录的文件缺失。
        ValueError: JSON 无法解析，或版本、文件校验值、语言、任务集合不符合固定来源。
    """
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["commit"] != TAU2_COMMIT:
        raise ValueError("中文零售数据的上游提交与当前转换器不一致")
    contents = {}
    for name in SOURCE_FILES:
        raw = (source_dir / name).read_bytes()
        expected = manifest["files"][name]
        if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError(f"来源文件校验失败：{name}")
        contents[name] = raw

    bundle = json.loads(contents["tau2_zh_retail.json"])
    subset = json.loads(contents["retail_50.json"])
    if bundle["domain"] != "retail" or bundle["settings"]["language"] != "zh":
        raise ValueError("数据包必须是中文 retail 数据")
    task_ids = [bundle["metadata"][task["id"]]["canonical_task_id"] for task in bundle["tasks"]]
    if len(task_ids) != len(set(task_ids)) or set(task_ids) != set(subset["task_ids"]):
        raise ValueError("中文参考任务与 retail_50 清单不一致")
    return RetailBundle(
        database=bundle["database"],
        tasks=bundle["tasks"],
        policy=bundle["policies"]["policy"],
        metadata=bundle["metadata"],
        settings=bundle["settings"],
        manifest=manifest,
        language_pack=contents["tau2_zh_pack.yaml"].decode("utf-8"),
    )

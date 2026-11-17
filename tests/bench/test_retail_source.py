"""校验固定中文来源与转换入口，所有检查均不调用模型。"""

import shutil
from pathlib import Path

import pytest

from ttwill.bench.retail import TAU2_COMMIT, load_retail_bundle

SOURCE_DIR = Path(__file__).resolve().parents[2] / "data" / "source"


def test_official_chinese_bundle_contains_the_frozen_subset() -> None:
    """确认完整中文业务数据与官方 50 题来自同一固定提交。"""
    bundle = load_retail_bundle(SOURCE_DIR)
    assert bundle.manifest["commit"] == TAU2_COMMIT
    assert len(bundle.database["users"]) == 500
    assert len(bundle.database["orders"]) == 1000
    assert len(bundle.database["products"]) == 50
    assert sum(len(product["variants"]) for product in bundle.database["products"].values()) == 591
    assert len(bundle.tasks) == 50
    assert all(task["initial_state"] is None for task in bundle.tasks)
    assert bundle.settings["currency_code"] == "CNY"
    assert "explicit user confirmation" in bundle.policy
    assert "agent_greeting: 您好！请问有什么可以帮您？" in bundle.language_pack


def test_source_hash_rejects_modified_data(tmp_path: Path) -> None:
    """来源数据即使仍是可解析 JSON，也不能绕过字节校验。"""
    target = tmp_path / "source"
    shutil.copytree(SOURCE_DIR, target)
    with (target / "tau2_zh_retail.json").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="来源文件校验失败"):
        load_retail_bundle(target)


def test_source_hash_rejects_modified_language_pack(tmp_path: Path) -> None:
    """中文人设原文与业务数据库使用同样的固定来源字节校验。"""
    target = tmp_path / "source"
    shutil.copytree(SOURCE_DIR, target)
    with (target / "tau2_zh_pack.yaml").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="来源文件校验失败：tau2_zh_pack.yaml"):
        load_retail_bundle(target)

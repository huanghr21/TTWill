"""直接运行已安装的官方工具，验证导入、导出和参考动作重放。"""

from pathlib import Path

import pytest

pytest.importorskip("tau2", reason="安装 bench 额外依赖后运行官方工具一致性测试")

from tau2.domains.retail.data_model import RetailDB  # noqa: E402

from ttwill.backend import RetailBackend  # noqa: E402
from ttwill.bench.replay import compare_read_tools, replay_task  # noqa: E402
from ttwill.bench.retail import load_retail_bundle  # noqa: E402

SOURCE_DIR = Path(__file__).resolve().parents[2] / "data" / "source"


def test_seed_roundtrip_preserves_the_official_database_hash(tmp_path: Path) -> None:
    """完整中文数据经过 SQLite 保存和导出后仍保留官方哈希。"""
    bundle = load_retail_bundle(SOURCE_DIR)
    backend = RetailBackend(tmp_path / "roundtrip.sqlite")
    try:
        backend.initialize(bundle.database)
        actual = RetailDB.model_validate(backend.export_snapshot())
    finally:
        backend.close()
    expected = RetailDB.model_validate(bundle.database)
    assert actual.get_hash() == expected.get_hash()


def test_all_business_read_tools_match_the_official_results(tmp_path: Path) -> None:
    """比较身份、用户、订单、商品、规格和目录七种读取行为。"""
    bundle = load_retail_bundle(SOURCE_DIR)
    results = compare_read_tools(bundle, tmp_path / "queries.sqlite")
    assert len(results) == 7
    assert all(result["matched"] for result in results), results


def test_official_exchange_reference_replays_to_the_same_state(tmp_path: Path) -> None:
    """用官方中文首题的完整参考动作核对两阶段换货与官方终态。"""
    bundle = load_retail_bundle(SOURCE_DIR)
    task = next(task for task in bundle.tasks if task["id"] == "6_zh_identity")
    result = replay_task(bundle, task, tmp_path / "exchange.sqlite")
    assert result["actions_equal"], result["actions"]
    assert result["hash_equal"], result["differences"]


def test_invalid_reference_queries_are_rejected_on_both_sides(tmp_path: Path) -> None:
    """官方第 47 题两次无效订单查询应保留错误而不阻断后续退货。"""
    bundle = load_retail_bundle(SOURCE_DIR)
    task = next(task for task in bundle.tasks if task["id"] == "47_zh_identity")
    result = replay_task(bundle, task, tmp_path / "invalid_queries.sqlite")
    rejected = [action for action in result["actions"] if action.get("reference_error")]
    assert len(rejected) == 2
    assert all(action["backend_error"] and action["matched"] for action in rejected)
    assert result["hash_equal"], result["differences"]

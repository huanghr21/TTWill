"""为本体场景创建固定中文开发种子与真实持久化运行环境。"""

from copy import deepcopy
from pathlib import Path

import pytest

from ttwill.bench.retail import load_retail_bundle
from ttwill.bootstrap import load_skills
from ttwill.config import load_config
from ttwill.llm import CompatibleModel

from .support import USER_ID, AcceptanceCase

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def acceptance_seed() -> dict:
    """抽取一位合成用户及其订单、商品，不使用官方题目或参考动作。

    Returns:
        来源校验通过的用户、四张订单和所需商品，金额、规格与订单字段原样保留。
    """
    database = load_retail_bundle(ROOT / "data" / "source").database
    user = database["users"][USER_ID]
    orders = {key: database["orders"][key] for key in user["orders"]}
    products = {
        item["product_id"]: database["products"][item["product_id"]]
        for order in orders.values()
        for item in order["items"]
    }
    return deepcopy({"users": {USER_ID: user}, "orders": orders, "products": products})


@pytest.fixture
def acceptance_case(tmp_path, request, monkeypatch, acceptance_seed):
    """为一项完整场景建立独立库，并禁止构造真实模型客户端。

    Args:
        tmp_path: pytest 分配的独立运行目录，可用 --basetemp 保留在 runs 下。
        request: 当前用例名称，用于轨迹、检查点和验收报告关联。
        monkeypatch: 测试结束后自动还原模型构造器。
        acceptance_seed: 已校验的只读种子子集，每个用例独立导入。

    Yields:
        能关闭重开原业务库的场景助手；成功路径结束后核对完整轨迹。
    """

    def reject_real_model(*args, **kwargs):
        """防止离线验收意外构造真实服务客户端。"""
        raise AssertionError("本体验收只能使用脚本化假模型")

    monkeypatch.setattr(CompatibleModel, "__init__", reject_real_model)
    case = AcceptanceCase(
        tmp_path,
        request.node.name,
        acceptance_seed,
        load_config(ROOT / "config.toml"),
        load_skills(ROOT),
    )
    yield case
    from .trace_checks import assert_trace_integrity

    assert_trace_integrity(case.run_dir, case.turns, case.name)

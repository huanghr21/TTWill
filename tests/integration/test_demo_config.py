"""验证真实模型配置不会污染四个假模型演示的模型身份与费用账本。"""

import asyncio
import json
from pathlib import Path

import pytest

from ttwill.bootstrap import (
    load_demo_config,
    run_agent_demo,
    run_coordinator_demo,
    run_runtime_demo,
)
from ttwill.cli import run_demo
from ttwill.config import load_config
from ttwill.llm import CompatibleModel

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def paid_config(tmp_path: Path) -> Path:
    """提供带非零价格的隔离配置和只读业务资源，不创建任何密钥文件。

    Args:
        tmp_path: 当前测试的独立配置和运行目录。

    Returns:
        含真实服务配置形态、两个模型档位及自定义运行预算的 TOML 路径。
    """
    path = tmp_path / "config.toml"
    path.write_text(
        """runs_dir = "demo-artifacts"
[llm]
timeout_seconds = 5
max_retries = 0
retry_delay_seconds = 0
[models.flash]
model = "paid-model"
base_url = "https://example.invalid/v1"
api_key_env = "TEST_DEMO_KEY"
input_cost_per_million = 50
output_cost_per_million = 100
cached_input_cost_per_million = 10
[models.strong]
model = "other-paid-model"
base_url = "https://example.invalid/v1"
api_key_env = "TEST_DEMO_KEY"
input_cost_per_million = 80
output_cost_per_million = 150
cached_input_cost_per_million = 20
[backend]
operation_ttl_seconds = 600
[agent]
max_steps = 10
max_rounds = 10
[context]
max_chars = 100000
recent_turns = 2
summary_tier = "strong"
""",
        encoding="utf-8",
    )
    for name in ("skills", "data"):
        (tmp_path / name).symlink_to(ROOT / name, target_is_directory=True)
    return path


def test_demo_config_preserves_runtime_settings_and_original_file(paid_config: Path) -> None:
    """演示副本保留目录与全部循环预算，并覆盖所有档位而不修改真实配置。"""
    before = paid_config.read_bytes()
    original = load_config(paid_config)
    demo = load_demo_config(paid_config)
    assert paid_config.read_bytes() == before
    assert load_config(paid_config) == original
    assert (demo.root, demo.runs_dir, demo.llm, demo.backend, demo.agent, demo.context) == (
        original.root,
        original.runs_dir,
        original.llm,
        original.backend,
        original.agent,
        original.context,
    )
    assert demo.models.keys() == original.models.keys()
    for model in demo.models.values():
        assert model.model == "fake-demo"
        assert model.base_url is None and model.api_key_env is None
        assert model.input_cost_per_million == 0
        assert model.output_cost_per_million == 0
        assert model.cached_input_cost_per_million == 0


@pytest.mark.parametrize("kind", ["demo", "agent-demo", "coordinator-demo", "runtime-demo"])
def test_every_demo_records_fake_model_and_zero_cost(paid_config: Path, kind: str, monkeypatch):
    """四个演示均用 fake-demo 发请求并记录零费用，配置中的非零价格不会传入账本。"""

    def reject_real_client(*args, **kwargs):
        """演示创建真实客户端时立即失败，避免读取密钥或建立网络连接。"""
        raise AssertionError("假模型演示不能创建真实服务客户端")

    monkeypatch.setattr(CompatibleModel, "__init__", reject_real_client)
    if kind == "demo":
        trace = asyncio.run(run_demo(paid_config))
    elif kind == "agent-demo":
        trace = Path(asyncio.run(run_agent_demo(paid_config, "order_service"))["trace"])
    elif kind == "coordinator-demo":
        trace = Path(asyncio.run(run_coordinator_demo(paid_config))["trace"])
    else:
        trace = Path(asyncio.run(run_runtime_demo(paid_config))["traces"][0])
    run_dir = trace.parents[1]
    assert run_dir.is_relative_to(paid_config.parent / "demo-artifacts")
    ledger = [json.loads(line) for line in (run_dir / "usage.jsonl").read_text().splitlines()]
    assert ledger and all(entry["cost_usd"] == 0 for entry in ledger)
    records = [json.loads(path.read_text()) for path in (run_dir / "models").glob("*.json")]
    assert records and len(records) == len(ledger)
    assert all(record["request"]["model"] == "fake-demo" for record in records)
    assert all(record["response"]["model"] == "fake-demo" for record in records)

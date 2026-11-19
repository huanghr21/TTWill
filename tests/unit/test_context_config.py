"""验证上下文字符预算、近期历史和摘要模型配置的边界。"""

from pathlib import Path

import pytest

from ttwill.config import ConfigError, ContextConfig, load_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_chars", "0"),
        ("max_chars", "-1"),
        ("max_chars", "true"),
        ("max_chars", "1.5"),
        ("data_max_chars", "0"),
        ("data_max_chars", "1"),
        ("data_max_chars", "-1"),
        ("data_max_chars", "true"),
        ("data_max_chars", "1.5"),
        ("data_max_chars", '"16000"'),
        ("recent_turns", "-1"),
        ("recent_turns", "true"),
        ("recent_turns", "1.5"),
        ("summary_tier", '"missing"'),
        ("summary_tier", '""'),
    ],
)
def test_context_rejects_invalid_budget_and_model(tmp_path: Path, name: str, value: str) -> None:
    """无效预算或不存在的摘要档位在加载配置时即被拒绝。"""
    path = tmp_path / "config.toml"
    original = {
        "max_chars": "64000",
        "data_max_chars": "16000",
        "recent_turns": "4",
        "summary_tier": '"flash"',
    }
    path.write_text(
        (ROOT / "config.toml")
        .read_text()
        .replace(f"{name} = {original[name]}", f"{name} = {value}")
    )
    with pytest.raises(ConfigError, match=f"context.{name}"):
        load_config(path)


def test_context_can_summarize_all_previous_turns(tmp_path: Path) -> None:
    """近期保留轮数可设为零，但当前用户消息始终由编译器完整保留。"""
    path = tmp_path / "config.toml"
    path.write_text(
        (ROOT / "config.toml").read_text().replace("recent_turns = 4", "recent_turns = 0")
    )
    config = load_config(path)
    assert config.context.max_chars == 64000
    assert config.context.recent_turns == 0
    assert config.context.summary_tier == "flash"
    assert config.context.data_max_chars == 16000


def test_business_data_budget_defaults_without_configuration(tmp_path: Path) -> None:
    """未写业务数据预算时采用 16000，已有上下文参数保持原值。"""
    path = tmp_path / "config.toml"
    path.write_text((ROOT / "config.toml").read_text().replace("data_max_chars = 16000\n", ""))
    assert load_config(path).context == ContextConfig()
    assert ContextConfig().data_max_chars == 16000


@pytest.mark.parametrize("budget", [2, 24000])
def test_business_data_budget_accepts_minimum_and_custom_values(
    tmp_path: Path, budget: int
) -> None:
    """独立业务预算可容纳空列表或采用自定义上限，不联动总预算和思考配置。"""
    original = load_config(ROOT / "config.toml")
    path = tmp_path / "config.toml"
    path.write_text(
        (ROOT / "config.toml")
        .read_text()
        .replace("data_max_chars = 16000", f"data_max_chars = {budget}")
    )
    updated = load_config(path)
    assert updated.context.data_max_chars == budget
    assert updated.context.max_chars == original.context.max_chars == 64000
    assert updated.context.recent_turns == original.context.recent_turns == 4
    assert updated.context.summary_tier == original.context.summary_tier == "flash"
    assert updated.agent == original.agent
    assert updated.models == original.models

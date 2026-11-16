"""验证上下文字符预算、近期历史和摘要模型配置的边界。"""

from pathlib import Path

import pytest

from ttwill.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_chars", "0"),
        ("max_chars", "-1"),
        ("max_chars", "true"),
        ("max_chars", "1.5"),
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
    original = {"max_chars": "64000", "recent_turns": "4", "summary_tier": '"flash"'}
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

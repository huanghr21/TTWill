"""验证子 Agent 步数预算的配置边界。"""

from pathlib import Path

import pytest

from ttwill.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "true", '"12"'])
def test_invalid_agent_steps(tmp_path: Path, value: str) -> None:
    """步数只接受正整数，不把布尔值或小数视为有效上限。"""
    path = tmp_path / "config.toml"
    path.write_text(
        (ROOT / "config.toml").read_text().replace("max_steps = 12", f"max_steps = {value}")
    )
    with pytest.raises(ConfigError, match="agent.max_steps"):
        load_config(path)


def test_custom_agent_steps(tmp_path: Path) -> None:
    """派工上限与模型传输重试次数分开配置。"""
    path = tmp_path / "config.toml"
    path.write_text((ROOT / "config.toml").read_text().replace("max_steps = 12", "max_steps = 3"))
    assert load_config(path).agent.max_steps == 3


def test_unknown_agent_option(tmp_path: Path) -> None:
    """未知循环选项不被静默忽略。"""
    path = tmp_path / "config.toml"
    path.write_text(
        (ROOT / "config.toml").read_text().replace("max_steps = 12", "max_steps = 12\nextra = 1")
    )
    with pytest.raises(ConfigError, match="agent.*未支持"):
        load_config(path)


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "true", '"8"'])
def test_invalid_coordinator_rounds(tmp_path: Path, value: str) -> None:
    """协调决策轮数必须是正整数，独立于子 Agent 步数。"""
    path = tmp_path / "config.toml"
    path.write_text(
        (ROOT / "config.toml").read_text().replace("max_rounds = 8", f"max_rounds = {value}")
    )
    with pytest.raises(ConfigError, match="agent.max_rounds"):
        load_config(path)


def test_custom_coordinator_rounds(tmp_path: Path) -> None:
    """协调循环预算可调整，不改变子 Agent 的工具步数。"""
    path = tmp_path / "config.toml"
    path.write_text((ROOT / "config.toml").read_text().replace("max_rounds = 8", "max_rounds = 4"))
    config = load_config(path)
    assert config.agent.max_rounds == 4
    assert config.agent.max_steps == 12

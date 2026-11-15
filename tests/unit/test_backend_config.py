"""验证业务操作有效期的配置边界。"""

from pathlib import Path

import pytest

from ttwill.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "true", '"900"'])
def test_invalid_operation_ttl(tmp_path: Path, value: str) -> None:
    """有效期必须是有限正数，错误类型不能转换为可用期限。"""
    content = (ROOT / "config.toml").read_text(encoding="utf-8")
    path = tmp_path / "config.toml"
    path.write_text(
        content.replace("operation_ttl_seconds = 900", f"operation_ttl_seconds = {value}"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="backend.operation_ttl_seconds"):
        load_config(path)


def test_custom_operation_ttl(tmp_path: Path) -> None:
    """业务有效期可独立于模型超时配置。"""
    content = (ROOT / "config.toml").read_text(encoding="utf-8")
    path = tmp_path / "config.toml"
    path.write_text(
        content.replace("operation_ttl_seconds = 900", "operation_ttl_seconds = 60"),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.backend.operation_ttl_seconds == 60
    assert config.llm.timeout_seconds == 30

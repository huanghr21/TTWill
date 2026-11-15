"""验证配置解析、路径定位、参数约束和密钥隔离。"""

from pathlib import Path

import pytest

from ttwill.config import ConfigError, load_config

BASE_CONFIG = """\
runs_dir = "runs"

[llm]
timeout_seconds = 30.0
max_retries = 2
retry_delay_seconds = 0.5

[models.flash]
model = "fake-demo"
"""


def write_config(tmp_path: Path, content: str = BASE_CONFIG) -> Path:
    """将测试配置写入临时目录。

    Args:
        tmp_path: 接收 config.toml 的目录。
        content: 要写入的 TOML 配置内容，默认使用最小测试配置。

    Returns:
        已写入的配置文件路径。
    """
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_paths_are_relative_to_config_not_working_directory(tmp_path, monkeypatch):
    """相对路径以配置所在目录为基准，加载配置不创建运行目录。"""
    config_dir = tmp_path / "project"
    config_dir.mkdir()
    path = write_config(config_dir, BASE_CONFIG.replace('"runs"', '"../output"'))
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    config = load_config(path)

    assert config.root == config_dir
    assert config.runs_dir == tmp_path / "output"
    assert not config.runs_dir.exists()


def test_optional_model_fields_and_parameters(tmp_path):
    """可选模型字段和调用参数按配置值加载。"""
    path = write_config(
        tmp_path,
        BASE_CONFIG
        + """\
base_url = "https://example.invalid/v1"
api_key_env = "TTWILL_API_KEY"
input_cost_per_million = 0
output_cost_per_million = 2.5
cached_input_cost_per_million = 0.1
[models.flash.parameters]
temperature = 0.2
max_tokens = 100
n = 1
""",
    )

    config = load_config(path)

    assert config.llm.timeout_seconds == 30.0
    assert config.llm.max_retries == 2
    assert config.llm.retry_delay_seconds == 0.5
    model = config.models["flash"]
    assert model.model == "fake-demo"
    assert model.base_url == "https://example.invalid/v1"
    assert model.api_key_env == "TTWILL_API_KEY"
    assert model.input_cost_per_million == 0.0
    assert model.output_cost_per_million == 2.5
    assert model.cached_input_cost_per_million == 0.1
    assert model.parameters == {"temperature": 0.2, "max_tokens": 100, "n": 1}


def test_missing_prices_remain_unknown(tmp_path):
    """缺省价格保持未知，其他可选字段使用约定的默认值。"""
    config = load_config(write_config(tmp_path))

    model = config.models["flash"]
    assert model.base_url is None
    assert model.api_key_env is None
    assert model.parameters == {}
    assert model.input_cost_per_million is None
    assert model.output_cost_per_million is None
    assert model.cached_input_cost_per_million is None


def test_config_does_not_load_env_secrets(tmp_path, monkeypatch):
    """配置对象只保留密钥变量名，不读取文件或进程中的密钥。"""
    (tmp_path / ".env").write_text("TTWILL_API_KEY=file-secret\n", encoding="utf-8")
    monkeypatch.setenv("TTWILL_API_KEY", "environment-secret")
    path = write_config(tmp_path, BASE_CONFIG + 'api_key_env = "TTWILL_API_KEY"\n')

    config = load_config(path)

    assert config.models["flash"].api_key_env == "TTWILL_API_KEY"
    assert "file-secret" not in repr(config)
    assert "environment-secret" not in repr(config)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("timeout_seconds = 30.0", "timeout_seconds = 0"),
        ("timeout_seconds = 30.0", "timeout_seconds = -1"),
        ("timeout_seconds = 30.0", "timeout_seconds = nan"),
        ("timeout_seconds = 30.0", "timeout_seconds = inf"),
        ("timeout_seconds = 30.0", "timeout_seconds = true"),
        ("max_retries = 2", "max_retries = -1"),
        ("max_retries = 2", "max_retries = 1.5"),
        ("max_retries = 2", "max_retries = false"),
        ("retry_delay_seconds = 0.5", "retry_delay_seconds = -0.1"),
        ("retry_delay_seconds = 0.5", 'retry_delay_seconds = "1"'),
        ('model = "fake-demo"', 'model = " "'),
        ('model = "fake-demo"', "model = 3"),
        ('runs_dir = "runs"', "runs_dir = []"),
    ],
)
def test_invalid_field_types_and_ranges(tmp_path, old, new):
    """配置字段的错误类型和越界数值均被拒绝。"""
    path = write_config(tmp_path, BASE_CONFIG.replace(old, new))

    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.parametrize("price", ["-1", "nan", "inf", '"0.5"', "true"])
def test_invalid_prices(tmp_path, price):
    """模型价格必须是有限的非负数值。"""
    path = write_config(tmp_path, BASE_CONFIG + f"input_cost_per_million = {price}\n")

    with pytest.raises(ConfigError, match="input_cost_per_million"):
        load_config(path)


@pytest.mark.parametrize("name", ["model", "messages", "tools", "stream", "timeout"])
def test_parameters_cannot_override_call_fields(tmp_path, name):
    """模型扩展参数不能覆盖调用层保留字段。"""
    path = write_config(tmp_path, BASE_CONFIG + f'[models.flash.parameters]\n{name} = "x"\n')

    with pytest.raises(ConfigError, match="保留"):
        load_config(path)


@pytest.mark.parametrize("value", ["0", "2", "1.0", "true"])
def test_single_response_only(tmp_path, value):
    """单次调用的响应数量只接受整数 1。"""
    path = write_config(tmp_path, BASE_CONFIG + f"[models.flash.parameters]\nn = {value}\n")

    with pytest.raises(ConfigError, match="只支持 1"):
        load_config(path)


@pytest.mark.parametrize(
    "parameter",
    [
        'api_key = "test-secret"',
        'authorization = "test-secret"',
        'headers = { Authorization = "test-secret" }',
        'extra = { api_key = "test-secret" }',
        'extra = [{ password = "test-secret" }]',
    ],
)
def test_credentials_are_rejected_without_echoing_values(tmp_path, parameter):
    """嵌套参数中的认证信息被拒绝，错误消息不回显密钥值。"""
    path = write_config(tmp_path, BASE_CONFIG + f"[models.flash.parameters]\n{parameter}\n")

    with pytest.raises(ConfigError, match="认证信息") as error:
        load_config(path)

    assert "test-secret" not in str(error.value)


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/model",
        "https://user:test-secret@example.invalid/v1",
        "https://example.invalid/v1?api_key=test-secret",
        "https://example.invalid/v1#test-secret",
    ],
)
def test_urls_cannot_carry_credentials(tmp_path, url):
    """模型地址使用 HTTP 协议且不能包含认证信息、查询或片段。"""
    path = write_config(tmp_path, BASE_CONFIG + f'base_url = "{url}"\n')

    with pytest.raises(ConfigError, match="base_url") as error:
        load_config(path)

    assert "test-secret" not in str(error.value)


@pytest.mark.parametrize(
    "content",
    [
        "typo = 1\n" + BASE_CONFIG,
        BASE_CONFIG.replace("[llm]", "[llm]\ntimeout = 30"),
        BASE_CONFIG + "unknown = 1\n",
        BASE_CONFIG + 'api_key = "test-secret"\n',
        BASE_CONFIG.replace("max_retries = 2\n", ""),
        BASE_CONFIG.replace('[models.flash]\nmodel = "fake-demo"', "[models]"),
    ],
)
def test_missing_unknown_and_sensitive_fields_fail_explicitly(tmp_path, content):
    """缺失字段、未知字段和敏感字段产生明确且不含密钥值的错误。"""
    path = write_config(tmp_path, content)

    with pytest.raises(ConfigError) as error:
        load_config(path)

    assert "test-secret" not in str(error.value)


@pytest.mark.parametrize("value", ["nan", "1979-05-27T07:32:00Z"])
def test_parameters_must_serialize_as_json(tmp_path, value):
    """扩展参数必须能序列化为标准 JSON。"""
    path = write_config(tmp_path, BASE_CONFIG + f"[models.flash.parameters]\nvalue = {value}\n")

    with pytest.raises(ConfigError):
        load_config(path)

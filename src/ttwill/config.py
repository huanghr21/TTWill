"""读取非敏感运行配置；密钥由真实模型后端单独从 .env 读取。"""

import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class ConfigError(ValueError):
    """配置字段缺失或不符合约定。"""


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """调用层的超时与重试设置。

    Attributes:
        timeout_seconds: 每次后端传输的总超时及 HTTP 各阶段超时，单位为秒。
        max_retries: 首次调用失败后的最多重试次数，不含首次调用。
        retry_delay_seconds: 指数退避的初始等待秒数。
    """

    timeout_seconds: float
    max_retries: int
    retry_delay_seconds: float


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """一个模型档位对应的实际接口、计价和请求参数。

    Attributes:
        model: 发送给服务的模型名称。
        base_url: 兼容 Chat Completions 的服务基础地址；假模型可设为 None。
        api_key_env: 本地 .env 中的密钥变量名；假模型可设为 None。
        input_cost_per_million: 非缓存输入的美元单价，按百万 token 计；None 表示未知。
        output_cost_per_million: 输出的美元单价，按百万 token 计；None 表示未知。
        cached_input_cost_per_million: 缓存输入的美元单价，按百万 token 计；None 表示未知。
        parameters: 传给模型服务的非敏感生成参数。
    """

    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    cached_input_cost_per_million: float | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """业务操作的准备有效期设置。

    Attributes:
        operation_ttl_seconds: 操作准备完成后允许提交的时长，单位为秒。
    """

    operation_ttl_seconds: float = 900


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """子 Agent 单次派工的工具循环限制。

    Attributes:
        max_steps: 单次派工的最多步数；每个工具调用计一步，无工具回复也计一步。
        max_rounds: 每条用户消息最多进行的协调决策轮数，每轮至多分派一批任务。
    """

    max_steps: int = 12
    max_rounds: int = 8


@dataclass(frozen=True, slots=True)
class AppConfig:
    """解析后的运行配置，不包含密钥值。

    Attributes:
        root: 配置文件所在目录的绝对路径。
        runs_dir: 运行产物目录的绝对路径。
        llm: 模型调用的超时与重试设置。
        models: 按模型档位名称索引的模型配置。
        backend: 业务操作的准备有效期设置。
        agent: 子 Agent 单次派工的步数上限。
    """

    root: Path
    runs_dir: Path
    llm: LLMConfig
    models: dict[str, ModelConfig]
    backend: BackendConfig = field(default_factory=BackendConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


_RESERVED_PARAMETERS = {"model", "messages", "tools", "stream", "timeout"}
_SECRET_FIELDS = {
    "api_key",
    "authorization",
    "password",
    "secret",
    "access_token",
    "token",
    "client_secret",
    "bearer_token",
    "headers",
    "extra_headers",
    "credentials",
    "auth",
}


def _table(value: Any, name: str) -> dict[str, Any]:
    """要求字段为 TOML 表，错误信息用 name 标明配置位置。"""
    if not isinstance(value, dict):
        raise ConfigError(f"{name} 必须是配置表")
    return value


def _check_keys(
    value: dict[str, Any], name: str, required: set[str], optional: set[str] | None = None
) -> None:
    """检查配置表的必填键和允许使用的键。

    Args:
        value: 待检查的配置表。
        name: 错误信息中使用的配置位置。
        required: 必须存在的键。
        optional: 额外允许的键；None 表示没有可选键。

    Raises:
        ConfigError: 缺失必填键或包含未知键。
    """
    missing = required - value.keys()
    unknown = value.keys() - required - (optional or set())
    if missing:
        raise ConfigError(f"{name} 缺少字段：{', '.join(sorted(missing))}")
    if unknown:
        raise ConfigError(f"{name} 包含未支持的字段：{', '.join(sorted(unknown))}")


def _text(value: Any, name: str) -> str:
    """校验非空文本并保留原值，不自动裁剪调用方的内容。"""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} 必须是非空字符串")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    """校验配置数值并转换为浮点数。

    Args:
        value: 待校验的数值，支持整数和浮点数。
        name: 错误信息中使用的配置位置。
        positive: 是否要求大于零；False 时允许零。

    Returns:
        满足取值范围的有限浮点数。

    Raises:
        ConfigError: 输入是布尔值、非数值、超出浮点数范围或不满足取值范围。
    """
    if type(value) not in (int, float):
        raise ConfigError(f"{name} 必须是数值")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ConfigError(f"{name} 超出数值范围") from exc
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        qualifier = "正数" if positive else "非负数"
        raise ConfigError(f"{name} 必须是有限的{qualifier}")
    return result


def _parameter_value(value: Any, name: str) -> None:
    """递归检查模型参数的取值与认证信息。

    Args:
        value: 待检查的参数值，配置表和列表会逐项递归检查。
        name: 错误信息中使用的配置位置。

    Raises:
        ConfigError: 包含认证字段、非有限数值或不支持的参数类型。
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower().replace("-", "_") in _SECRET_FIELDS:
                raise ConfigError(f"{name} 不能包含认证信息；密钥只放在 .env")
            _parameter_value(item, f"{name}.{key}")
    elif isinstance(value, list):
        for item in value:
            _parameter_value(item, name)
    elif type(value) not in (str, bool, int, float):
        raise ConfigError(f"{name} 只能包含 JSON 兼容的参数值")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ConfigError(f"{name} 不能包含非有限数值")


def _model_config(value: Any, name: str) -> ModelConfig:
    """校验单个模型档位的接口、计价与生成参数。

    Args:
        value: 从 TOML 读取的模型档位配置表。
        name: 错误信息中使用的档位配置位置。

    Returns:
        解析后的模型配置；未配置的接口、密钥名和单价保留为 None。

    Raises:
        ConfigError: 字段缺失、未知或取值不符合约定，包括认证信息和保留参数。
    """
    data = _table(value, name)
    price_fields = {
        "input_cost_per_million",
        "output_cost_per_million",
        "cached_input_cost_per_million",
    }
    _check_keys(data, name, {"model"}, {"base_url", "api_key_env", "parameters"} | price_fields)
    model = _text(data["model"], f"{name}.model")
    base_url = None
    if "base_url" in data:
        base_url = _text(data["base_url"], f"{name}.base_url")
        try:
            parsed = urlsplit(base_url)
        except ValueError as exc:
            raise ConfigError(f"{name}.base_url 不是有效 URL") from exc
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigError(f"{name}.base_url 必须是 HTTP 或 HTTPS 地址")
        if (
            parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ConfigError(f"{name}.base_url 不能包含认证信息、查询参数或片段")
    api_key_env = None
    if "api_key_env" in data:
        api_key_env = _text(data["api_key_env"], f"{name}.api_key_env")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
            raise ConfigError(f"{name}.api_key_env 必须是 .env 变量名")
    parameters = _table(data.get("parameters", {}), f"{name}.parameters")
    if _RESERVED_PARAMETERS & parameters.keys():
        raise ConfigError(f"{name}.parameters 不能覆盖调用层保留的参数")
    if "n" in parameters and (type(parameters["n"]) is not int or parameters["n"] != 1):
        raise ConfigError(f"{name}.parameters.n 只支持 1 个响应")
    _parameter_value(parameters, f"{name}.parameters")
    prices = {
        key: _number(data[key], f"{name}.{key}") if key in data else None for key in price_fields
    }
    return ModelConfig(
        model=model, base_url=base_url, api_key_env=api_key_env, parameters=parameters, **prices
    )


def load_config(path: Path) -> AppConfig:
    """读取并校验 TOML 配置，不读取 .env 或修改环境变量。

    Args:
        path: 配置文件路径；其中 runs_dir 的相对路径以该文件所在目录为准。

    Returns:
        含绝对目录路径、调用限制和模型档位映射的运行配置。

    Raises:
        ConfigError: 字段缺失、未知或取值不符合约定。
        tomllib.TOMLDecodeError: 文件不是合法 TOML。
        OSError: 配置文件无法读取。
    """
    path = path.resolve()
    with path.open("rb") as file:
        data = tomllib.load(file)
    _check_keys(data, "配置", {"runs_dir", "llm", "models"}, {"backend", "agent"})
    root = path.parent
    runs_dir = (root / _text(data["runs_dir"], "runs_dir")).resolve()
    llm = _table(data["llm"], "llm")
    _check_keys(llm, "llm", {"timeout_seconds", "max_retries", "retry_delay_seconds"})
    max_retries = llm["max_retries"]
    if type(max_retries) is not int or max_retries < 0:
        raise ConfigError("llm.max_retries 必须是非负整数")
    llm_config = LLMConfig(
        timeout_seconds=_number(llm["timeout_seconds"], "llm.timeout_seconds", positive=True),
        max_retries=max_retries,
        retry_delay_seconds=_number(llm["retry_delay_seconds"], "llm.retry_delay_seconds"),
    )
    models = _table(data["models"], "models")
    if not models:
        raise ConfigError("models 必须至少配置一个模型档位")
    parsed_models = {
        _text(tier, "模型档位"): _model_config(value, f"models.{tier}")
        for tier, value in models.items()
    }
    backend = _table(data.get("backend", {}), "backend")
    _check_keys(backend, "backend", set(), {"operation_ttl_seconds"})
    backend_config = BackendConfig(
        operation_ttl_seconds=_number(
            backend.get("operation_ttl_seconds", 900),
            "backend.operation_ttl_seconds",
            positive=True,
        )
    )
    agent = _table(data.get("agent", {}), "agent")
    _check_keys(agent, "agent", set(), {"max_steps", "max_rounds"})
    max_steps = agent.get("max_steps", 12)
    if type(max_steps) is not int or max_steps < 1:
        raise ConfigError("agent.max_steps 必须是正整数")
    max_rounds = agent.get("max_rounds", 8)
    if type(max_rounds) is not int or max_rounds < 1:
        raise ConfigError("agent.max_rounds 必须是正整数")
    return AppConfig(
        root=root,
        runs_dir=runs_dir,
        llm=llm_config,
        models=parsed_models,
        backend=backend_config,
        agent=AgentConfig(max_steps=max_steps, max_rounds=max_rounds),
    )

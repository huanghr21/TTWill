"""模型边界上的结果与错误。"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from ttwill.config import ModelConfig


class LLMError(RuntimeError):
    """向调用者传递模型错误、重试标记及可选 HTTP 状态码。"""

    def __init__(
        self, message: str, *, retryable: bool = False, status_code: int | None = None
    ) -> None:
        """保存模型错误及重试信息。

        Args:
            message: 调用方已去除敏感信息的错误消息，按原文保存。
            retryable: 是否允许调用层重试本次错误。
            status_code: 相关 HTTP 状态码；None 表示没有状态码。
        """
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class Usage:
    """一次模型响应的 token 用量。

    Attributes:
        input_tokens: 输入 token 总数，包含缓存命中的部分。
        output_tokens: 输出 token 总数。
        cached_input_tokens: 输入中命中缓存的 token 数。
    """

    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0

    def cost_usd(self, model: ModelConfig) -> float | None:
        """按配置中每百万 token 的美元单价计算本次费用。

        输入按缓存命中与未命中分别计价，输出单独计价。

        Args:
            model: 含输入、输出及缓存输入单价的模型配置。

        Returns:
            本次响应的美元费用；缺少所需单价时返回 None。只有实际命中
            缓存时才要求配置缓存单价。
        """
        rates = [model.input_cost_per_million, model.output_cost_per_million]
        if self.cached_input_tokens:
            rates.append(model.cached_input_cost_per_million)
        if any(rate is None for rate in rates):
            return None
        cost = Decimal(self.input_tokens - self.cached_input_tokens) * Decimal(
            str(rates[0])
        ) + Decimal(self.output_tokens) * Decimal(str(rates[1]))
        if self.cached_input_tokens:
            cost += Decimal(self.cached_input_tokens) * Decimal(str(rates[2]))
        return float(cost / 1_000_000)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """已解析的单个模型响应，保留完整消息和原始 JSON 对象。

    Attributes:
        model: 服务返回的实际模型标识。
        message: assistant 消息的原对象引用，包括工具调用、推理内容等扩展字段。
        finish_reason: 服务返回的结束原因；未提供时为 None。
        usage: 已解析的 token 用量；未提供时为 None。
        raw: 完整响应的原对象引用。
    """

    model: str
    message: dict[str, Any]
    finish_reason: str | None
    usage: Usage | None
    raw: dict[str, Any]


def parse_usage(raw: dict[str, Any]) -> Usage | None:
    """从原始响应解析 token 用量。

    缓存数优先取 prompt_cache_hit_tokens，其次取详情中的 cached_tokens，
    两者均缺失时按零处理。

    Args:
        raw: 模型服务返回的完整响应对象。

    Returns:
        输入、输出及缓存输入的用量；usage 缺失或为 null 时返回 None。

    Raises:
        LLMError: 用量字段结构错误、计数不是非负整数，或缓存数超过输入总数。
    """
    data = raw.get("usage")
    if data is None:
        return None
    if not isinstance(data, dict):
        raise LLMError("模型响应的 usage 必须是对象")
    details = data.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        raise LLMError("模型响应的 prompt_tokens_details 必须是对象")
    counts = (
        data.get("prompt_tokens"),
        data.get("completion_tokens"),
        data.get("prompt_cache_hit_tokens", details.get("cached_tokens", 0)),
    )
    if any(type(value) is not int or value < 0 for value in counts):
        raise LLMError("模型响应的 token 用量必须是非负整数")
    usage = Usage(*counts)
    if usage.cached_input_tokens > usage.input_tokens:
        raise LLMError("缓存 token 不能超过输入 token")
    return usage


def parse_response(raw: dict[str, Any], usage: Usage | None) -> ModelResponse:
    """校验单个 assistant 响应，附上已解析用量并保留完整原始内容。

    响应须包含实际模型标识，以及文本、工具调用或拒答内容。工具调用
    只检查结构，arguments 仍保留为字符串，不在此处解析或执行业务。

    Args:
        raw: 模型服务返回的完整响应对象，要求仅包含一个 choice。
        usage: 已解析的用量；None 表示服务未提供用量。

    Returns:
        包含实际模型标识、消息、结束原因和用量的响应，保留消息与原文的对象引用。

    Raises:
        LLMError: 必需字段缺失、已检查字段的结构不符，或消息没有有效内容。
    """
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise LLMError("模型必须返回一个 choice")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise LLMError("模型未返回有效 message")
    message = choice["message"]
    if message.get("role") != "assistant":
        raise LLMError("模型响应的角色必须是 assistant")
    if not isinstance(raw.get("model"), str) or not raw["model"]:
        raise LLMError("模型响应缺少实际 model 标识")
    if message.get("content") is not None and not isinstance(message["content"], str):
        raise LLMError("模型文本响应必须是字符串或 null")
    tool_calls = message.get("tool_calls")
    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            raise LLMError("模型工具调用必须是列表")
        for call in tool_calls:
            if (
                not isinstance(call, dict)
                or not isinstance(call.get("id"), str)
                or call.get("type") != "function"
                or not isinstance(call.get("function"), dict)
                or not isinstance(call["function"].get("name"), str)
                or not isinstance(call["function"].get("arguments"), str)
            ):
                raise LLMError("模型工具调用结构不完整")
    refusal = message.get("refusal")
    if refusal is not None and not isinstance(refusal, str):
        raise LLMError("模型拒答内容必须是字符串或 null")
    if message.get("content") is None and not tool_calls and not refusal:
        raise LLMError("模型响应既没有文本、工具调用，也没有拒答内容")
    return ModelResponse(raw["model"], message, choice.get("finish_reason"), usage, raw)

"""模型边界上的结果与错误。"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from ttwill.config import ModelConfig


class LLMError(RuntimeError):
    """可公开的错误信息，不包含服务返回的原始错误正文或鉴权头。"""

    def __init__(
        self, message: str, *, retryable: bool = False, status_code: int | None = None
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0

    def cost_usd(self, model: ModelConfig) -> float | None:
        """缺少计价所需信息时保留未知，不能把未知费用当作零。"""
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
    model: str
    message: dict[str, Any]
    finish_reason: str | None
    usage: Usage | None
    raw: dict[str, Any]


def parse_usage(raw: dict[str, Any]) -> Usage | None:
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

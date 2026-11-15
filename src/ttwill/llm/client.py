"""按档位调用模型；每次尝试单独留存轨迹、原始结果与费用。"""

import asyncio
import sys
from copy import deepcopy
from dataclasses import asdict
from time import perf_counter
from typing import Any
from uuid import uuid4

from ttwill.config import AppConfig
from ttwill.llm.models import CompatibleModel, FakeModel
from ttwill.llm.types import LLMError, ModelResponse, parse_response, parse_usage
from ttwill.tracing import Trace


class LLMClient:
    """统一模型调用、响应校验与逐次尝试记录，由调用者管理后端连接的生命周期。"""

    def __init__(
        self, config: AppConfig, trace: Trace, backend: CompatibleModel | FakeModel
    ) -> None:
        """绑定运行配置、轨迹和模型后端。

        Args:
            config: 模型档位、超时和重试设置。
            trace: 接收模型调用轨迹、原文和用量记录的会话轨迹。
            backend: 执行模型请求的真实后端或脚本化假模型，生命周期由调用者管理。
        """
        self.config = config
        self.trace = trace
        self.backend = backend

    async def complete(
        self,
        tier: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        """按配置档位请求一个完整响应，并为每次尝试保存原文、轨迹和费用。

        仅对可重试的 LLMError 和传输总超时进行指数退避，最多尝试
        ``max_retries + 1`` 次。配置中的秒数限制每次后端传输的总用时，
        不包含重试等待、响应解析和记录写入；取消会向上传播，不触发重试。
        用量或计价信息缺失时，相应账本字段保留为 None。

        Args:
            tier: 配置中已定义的模型档位名称。
            messages: Chat Completions 消息历史，调用时复制，不修改传入列表。
            tools: 本次可用的工具定义；None 表示请求不携带 tools 字段。

        Returns:
            校验后的响应，保留服务返回的实际模型标识、完整消息和原始对象。

        Raises:
            ValueError: 档位未配置或消息列表为空。
            LLMError: 请求或响应无效，或可重试的模型错误已耗尽重试次数。
            TimeoutError: 最后一次传输尝试超过配置的总时限。
            asyncio.CancelledError: 调用被取消，包括等待重试期间的取消。
            OSError: 轨迹、模型原文或费用账本无法写入。
        """
        if tier not in self.config.models:
            raise ValueError(f"未配置模型档位：{tier}")
        if not messages:
            raise ValueError("模型调用至少需要一条消息")
        model = self.config.models[tier]
        request = deepcopy(model.parameters)
        request.update(model=model.model, messages=deepcopy(messages), stream=False)
        if tools is not None:
            request["tools"] = deepcopy(tools)
        call_id = uuid4().hex
        limits = self.config.llm
        for attempt in range(1, limits.max_retries + 2):
            record_id = f"{call_id}-{attempt}"
            try:
                with self.trace.span(
                    "llm", tier, attrs={"call_id": call_id, "attempt": attempt, "tier": tier}
                ) as span:
                    started = perf_counter()
                    raw = None
                    usage = None
                    completed = False
                    try:
                        # HTTPX 的读超时是分块空闲超时；另限制每次尝试的总用时。
                        async with asyncio.timeout(limits.timeout_seconds):
                            raw = await self.backend.send(model, request, limits.timeout_seconds)
                        usage = parse_usage(raw)
                        response = parse_response(raw, usage)
                        completed = True
                        return response
                    finally:
                        error = sys.exception() if not completed else None
                        latency_ms = (perf_counter() - started) * 1000
                        cost = usage.cost_usd(model) if usage else None
                        error_info = None
                        if error is not None:
                            error_info = {"type": type(error).__name__}
                            if isinstance(error, LLMError) and error.status_code is not None:
                                error_info["status_code"] = error.status_code
                        record_ref = self.trace.write_model_record(
                            record_id,
                            {
                                "call_id": call_id,
                                "attempt": attempt,
                                "tier": tier,
                                "request": request,
                                "response": raw,
                                "error": error_info,
                            },
                        )
                        metrics = {
                            "call_id": call_id,
                            "attempt": attempt,
                            "span_id": span.span_id,
                            "tier": tier,
                            "requested_model": model.model,
                            "model": raw.get("model", model.model) if raw else model.model,
                            "input_tokens": usage.input_tokens if usage else None,
                            "output_tokens": usage.output_tokens if usage else None,
                            "cached_input_tokens": usage.cached_input_tokens if usage else None,
                            "cost_usd": cost,
                            "latency_ms": latency_ms,
                            "record_ref": record_ref,
                            "error": error_info,
                        }
                        span.attrs.update(metrics)
                        self.trace.record_usage(
                            {
                                **metrics,
                                "status": "error" if error else "ok",
                                "usage": asdict(usage) if usage else None,
                            }
                        )
            except (LLMError, TimeoutError) as exc:
                retryable = isinstance(exc, TimeoutError) or exc.retryable
                if not retryable or attempt > limits.max_retries:
                    raise
                await asyncio.sleep(limits.retry_delay_seconds * 2 ** (attempt - 1))
        raise AssertionError("模型重试循环不应无结果退出")

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
    def __init__(
        self, config: AppConfig, trace: Trace, backend: CompatibleModel | FakeModel
    ) -> None:
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

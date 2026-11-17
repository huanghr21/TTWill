"""把 τ² 内部的 LiteLLM 调用接入本体轨迹和费用账本。"""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import Context, ContextVar, copy_context
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from threading import Condition
from typing import Any, Literal
from uuid import uuid4

import litellm
from litellm.integrations.custom_logger import CustomLogger

from ttwill.config import ModelConfig
from ttwill.llm.types import LLMError, parse_usage
from ttwill.tracing import Span, Trace


@dataclass
class _Attempt:
    """保存一次请求及其专属轨迹上下文，供后台完成回调闭合。"""

    number: int
    request: dict[str, Any]
    context: Context
    manager: AbstractContextManager[Span]
    span: Span


@dataclass
class _Capture:
    """汇总一次官方组件操作中发生的请求和落盘错误。"""

    call_id: str = field(default_factory=lambda: uuid4().hex)
    attempts: int = 0
    pending: dict[str, _Attempt] = field(default_factory=dict)
    errors: list[Exception] = field(default_factory=list)


class TauUsageCallback(CustomLogger):
    """记录官方模拟用户或判分调用，等待后台记录完成后再让驱动器继续。"""

    def __init__(
        self, trace: Trace, *, role: Literal["user_simulator", "evaluator"], model: ModelConfig
    ) -> None:
        """绑定会话、评测角色及明确配置的模型价格。

        Args:
            trace: 已携带运行、任务、重复和尝试标识的会话轨迹。
            role: 官方调用的角色，模拟用户或判分器。
            model: 实际模型配置；缺失价格时费用保持未知。

        Raises:
            ValueError: role 不是受支持的评测角色。
        """
        super().__init__()
        if role not in {"user_simulator", "evaluator"}:
            raise ValueError("τ² 模型角色必须为 user_simulator 或 evaluator")
        self.trace = trace
        self.role = role
        self.model = model
        self._id = uuid4().hex
        self._capture: ContextVar[_Capture | None] = ContextVar(self._id, default=None)
        self._condition = Condition()
        self._active = False

    def request_kwargs(self) -> dict[str, Any]:
        """生成传给官方组件 llm_args 的回调参数，关闭不可逐次观测的内部重试。

        Returns:
            每次返回独立字典及列表；含请求级成功、失败回调，关联元数据和
            num_retries / max_retries，并关闭响应缓存。驱动器负责显式重试
            并保留每次尝试。
        """
        return {
            "success_callback": [self],
            "failure_callback": [self],
            "metadata": {"ttwill_callback_id": self._id},
            "num_retries": 0,
            "max_retries": 0,
            "caching": False,
        }

    @contextmanager
    def capture(self) -> Iterator[None]:
        """包围一次同步官方操作，离开时等待所有请求的后台回调完成。

        该操作及其 LiteLLM 调用须在同一线程执行；可把整个上下文放入
        asyncio.to_thread。输入回调按本实例元数据筛选，不收集其他请求。
        同一实例不能重入；不同实例可并行使用。

        Yields:
            无返回对象；内部调用须传入 request_kwargs 产生的参数。

        Raises:
            RuntimeError: 同一实例重入，或完成回调在 30 秒内未交付。
            OSError: 模型原文、轨迹或账本写入失败。
            ValueError: 回调记录的数据无法序列化。
            LLMError: 模型用量字段不合法。
        """
        with self._condition:
            if self._active:
                raise RuntimeError("同一 τ² 用量回调不能同时捕获多个操作")
            self._active = True
        captured = _Capture()
        token = self._capture.set(captured)
        litellm.input_callback.append(self)
        try:
            yield
        finally:
            try:
                with self._condition:
                    completed = self._condition.wait_for(lambda: not captured.pending, timeout=30)
                if not completed:
                    raise RuntimeError("τ² 模型调用的完成回调未在 30 秒内交付")
                if captured.errors:
                    raise captured.errors[0]
            finally:
                litellm.input_callback.remove(self)
                self._capture.reset(token)
                with self._condition:
                    self._active = False

    def log_pre_api_call(self, model: str, messages: list[dict], kwargs: dict) -> None:
        """为带本实例关联元数据的请求记录真实开始时间与完整模型输入。

        Args:
            model: LiteLLM 实际使用的模型名称。
            messages: 发送给模型的完整消息。
            kwargs: LiteLLM 的调用详情，包含关联标识及非敏感生成参数。
        """
        metadata = kwargs.get("litellm_params", {}).get("metadata") or {}
        if metadata.get("ttwill_callback_id") != self._id:
            return
        captured = self._capture.get()
        if captured is None:
            raise RuntimeError("τ² 模型调用必须位于 capture 上下文中")
        self._start(captured, kwargs, model, messages)

    def _start(self, captured: _Capture, kwargs: dict, model: str, messages: list) -> _Attempt:
        """在独立上下文中打开 span，以便不同线程的完成回调安全闭合。

        Args:
            captured: 本次官方操作的请求集合。
            kwargs: 含 LiteLLM 请求 ID 和生成参数的调用详情。
            model: 实际请求模型名称。
            messages: 完整模型输入。

        Returns:
            保存请求、序号及专属轨迹上下文的尝试记录。
        """
        key = kwargs["litellm_call_id"]
        with self._condition:
            if key in captured.pending:
                raise RuntimeError("同一 LiteLLM 请求重复开始；须关闭内部重试")
            captured.attempts += 1
            request = _request_parameters(kwargs.get("optional_params") or {})
            request.update(model=model, messages=deepcopy(messages))
            context = copy_context()
            manager = self.trace.span(
                "llm",
                self.role,
                attrs={
                    "role": self.role,
                    "call_id": captured.call_id,
                    "attempt": captured.attempts,
                },
            )
            try:
                span = context.run(manager.__enter__)
            except (OSError, ValueError, TypeError) as exc:
                captured.errors.append(exc)
                raise
            attempt = _Attempt(captured.attempts, request, context, manager, span)
            captured.pending[key] = attempt
            return attempt

    def log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: datetime, end_time: datetime
    ) -> None:
        """记录成功响应并闭合对应模型 span。

        Args:
            kwargs: LiteLLM 调用详情，包含请求关联 ID。
            response_obj: SDK 返回的完整模型响应对象。
            start_time: 请求开始时间。
            end_time: 请求结束时间。
        """
        self._finish(kwargs, response_obj, start_time, end_time, failed=False)

    def log_failure_event(
        self, kwargs: dict, response_obj: Any, start_time: datetime, end_time: datetime
    ) -> None:
        """记录失败响应，仅保存异常类型及状态码以避免凭据泄露。

        Args:
            kwargs: LiteLLM 调用详情，包含请求关联 ID 和异常对象。
            response_obj: LiteLLM 提供的失败结果，不写入可能带凭据的错误文本。
            start_time: 请求开始时间。
            end_time: 请求结束时间。
        """
        self._finish(kwargs, response_obj, start_time, end_time, failed=True)

    def _finish(
        self, kwargs: dict, response: Any, start: datetime, end: datetime, *, failed: bool
    ) -> None:
        """完成一次请求并唤醒驱动器，回传被 LiteLLM 捕获的记录异常。

        Args:
            kwargs: 含请求 ID 的 LiteLLM 调用详情。
            response: SDK 完整模型响应；请求失败时忽略其错误文本。
            start: 请求开始时间。
            end: 请求结束时间。
            failed: 是否由失败回调调用。
        """
        captured = self._capture.get()
        if captured is None:
            raise RuntimeError("τ² 完成回调缺少 capture 上下文")
        key = kwargs["litellm_call_id"]
        with self._condition:
            attempt = captured.pending.get(key)
        if attempt is None:
            # 参数校验等前置错误可能在输入回调之前发生，也必须进入账本。
            attempt = self._start(
                captured, kwargs, kwargs.get("model", self.model.model), kwargs.get("messages", [])
            )
        failure = None
        try:
            attempt.context.run(
                self._record, captured, attempt, kwargs, response, start, end, failed
            )
        except (OSError, ValueError, TypeError, LLMError) as exc:
            failure = exc
            with self._condition:
                captured.errors.append(exc)
            raise
        finally:
            try:
                attempt.context.run(
                    attempt.manager.__exit__, type(failure) if failure else None, failure, None
                )
            except (OSError, ValueError, TypeError) as exc:
                with self._condition:
                    captured.errors.append(exc)
                raise
            finally:
                with self._condition:
                    captured.pending.pop(key, None)
                    self._condition.notify_all()

    def _record(
        self,
        captured: _Capture,
        attempt: _Attempt,
        kwargs: dict,
        response: Any,
        start: datetime,
        end: datetime,
        failed: bool,
    ) -> None:
        """保存完整响应与允许的请求字段，并按明确价格计算费用。

        Args:
            captured: 当前官方操作的调用集合。
            attempt: 当前请求的原文及轨迹。
            kwargs: LiteLLM 调用详情，失败时包含异常对象。
            response: 成功时的完整模型响应对象。
            start: 请求开始时间。
            end: 请求结束时间。
            failed: 是否发生模型传输失败。
        """
        raw = None if failed else response.model_dump(mode="json")
        error = None
        if failed:
            exception = kwargs.get("exception")
            error = {"type": type(exception).__name__ if exception else "LiteLLMError"}
            status = getattr(exception, "status_code", None)
            if type(status) is int:
                error["status_code"] = status
        usage = None
        usage_error = None
        try:
            usage = parse_usage(raw) if raw else None
        except LLMError as exc:
            # 无效 usage 仍保留原响应；不能丢掉已经实际发生的请求。
            usage_error = exc
            error = {"type": type(exc).__name__}
        record_id = f"{captured.call_id}-{attempt.number}"
        reference = self.trace.write_model_record(
            record_id,
            {
                "call_id": captured.call_id,
                "attempt": attempt.number,
                "role": self.role,
                "request": attempt.request,
                "response": raw,
                "error": error,
            },
        )
        metrics = {
            "call_id": captured.call_id,
            "attempt": attempt.number,
            "span_id": attempt.span.span_id,
            "role": self.role,
            "tier": None,
            "requested_model": self.model.model,
            "model": raw.get("model", self.model.model) if raw else self.model.model,
            "input_tokens": usage.input_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None,
            "cached_input_tokens": usage.cached_input_tokens if usage else None,
            "cost_usd": usage.cost_usd(self.model) if usage else None,
            "latency_ms": (end - start).total_seconds() * 1000,
            "record_ref": reference,
            "error": error,
        }
        attempt.span.attrs.update(metrics)
        attempt.span.status = "error" if error else "ok"
        self.trace.record_usage(
            {**metrics, "status": attempt.span.status, "usage": asdict(usage) if usage else None}
        )
        if usage_error is not None:
            raise usage_error


def _request_parameters(value: dict[str, Any]) -> dict[str, Any]:
    """复制模型生成参数，排除连接凭据、HTTP 头和回调元数据。"""
    excluded = {"api_key", "api_base", "base_url", "authorization", "headers", "extra_headers"}
    result = {}
    for key, item in value.items():
        if key.lower() in excluded or key in {"metadata", "litellm_metadata", "client"}:
            continue
        result[key] = _request_parameters(item) if isinstance(item, dict) else deepcopy(item)
    return result

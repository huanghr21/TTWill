"""兼容 Chat Completions 的传输与可脚本化的假模型。"""

from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values

from ttwill.config import ModelConfig
from ttwill.llm.types import LLMError


class CompatibleModel:
    """复用 HTTP 连接调用兼容服务；应通过 async with 使用并在退出时关闭连接。"""

    def __init__(
        self, env_path: Path, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        """读取指定 .env 的密钥快照并建立客户端，不发出请求。

        不插值变量，也不读取或修改进程环境中的同名密钥。密钥是否存在在
        send 时检查。

        Args:
            env_path: 保存模型密钥的本地 .env 文件路径。
            transport: HTTP 传输对象，可用于测试；None 表示使用 HTTPX 默认传输。
        """
        # 不把密钥载入全局环境，多个运行的配置不会互相污染。
        self._keys = dotenv_values(env_path, interpolate=False)
        self._client = httpx.AsyncClient(transport=transport)

    async def __aenter__(self) -> "CompatibleModel":
        """返回当前客户端，供上下文内的多次请求复用连接。"""
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """关闭 HTTP 客户端，不抑制上下文内的异常。"""
        await self._client.aclose()

    async def send(
        self, model: ModelConfig, request: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        """向配置的 Chat Completions 地址发送一次请求，不在此层重试。

        Args:
            model: 服务地址及 .env 密钥名；模型名和生成参数已在 request 中。
            request: 要发送的完整 JSON 请求体，不包含鉴权头。
            timeout_seconds: HTTPX 各网络阶段的超时秒数；总时限由 LLMClient 控制。

        Returns:
            服务返回的完整 JSON 对象；消息和用量字段由调用层继续校验。

        Raises:
            LLMError: 配置或密钥缺失、传输失败、HTTP 非成功状态，或响应不是
                JSON 对象。传输错误、408、429 和 5xx 标记为可重试；错误消息
                不包含服务错误正文或鉴权头。
            asyncio.CancelledError: 外部取消请求，原样向上传播。
        """
        if not model.base_url or not model.api_key_env:
            raise LLMError("真实模型需要配置 base_url 与 api_key_env")
        key = self._keys.get(model.api_key_env)
        if not key or not key.strip():
            raise LLMError("本地 .env 缺少所需模型密钥")
        try:
            response = await self._client.post(
                model.base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=request,
                timeout=timeout_seconds,
            )
        except httpx.TransportError as exc:
            raise LLMError(type(exc).__name__, retryable=True) from None
        if not response.is_success:
            raise LLMError(
                f"模型服务返回 HTTP {response.status_code}",
                retryable=response.status_code in {408, 429} or response.status_code >= 500,
                status_code=response.status_code,
            )
        try:
            data = response.json()
        except ValueError:
            raise LLMError("模型服务未返回合法 JSON") from None
        if not isinstance(data, dict):
            raise LLMError("模型响应必须是 JSON 对象")
        return data


class FakeModel:
    """按顺序返回响应或抛出错误；脚本耗尽明确失败，不访问网络。"""

    def __init__(self, responses: Iterable[dict[str, Any] | LLMError]) -> None:
        """保存响应脚本并初始化请求历史。

        Args:
            responses: 按调用顺序消费的响应对象或 LLMError 可迭代对象；每次调用
                消费一项，支持按当前运行状态延迟生成响应。
        """
        self._responses = iter(responses)
        self.requests: list[dict[str, Any]] = []

    async def send(
        self, model: ModelConfig, request: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        """记录请求副本并消费下一项响应脚本。

        Args:
            model: 与真实后端保持一致的模型配置参数，不参与响应选择。
            request: 本次请求体；副本会追加到 requests 历史中。
            timeout_seconds: 与真实后端保持一致的超时参数，假模型不执行等待。

        Returns:
            当前脚本项的响应副本。

        Raises:
            LLMError: 当前脚本项是错误，或响应脚本已耗尽。
        """
        self.requests.append(deepcopy(request))
        response = next(self._responses, None)
        if response is None:
            raise LLMError("假模型响应脚本已耗尽")
        if isinstance(response, LLMError):
            raise response
        return deepcopy(response)

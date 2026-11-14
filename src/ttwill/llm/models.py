"""兼容 Chat Completions 的传输与可脚本化的假模型。"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values

from ttwill.config import ModelConfig
from ttwill.llm.types import LLMError


class CompatibleModel:
    def __init__(
        self, env_path: Path, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        # 不把密钥载入全局环境，多个运行的配置不会互相污染。
        self._keys = dotenv_values(env_path, interpolate=False)
        self._client = httpx.AsyncClient(transport=transport)

    async def __aenter__(self) -> "CompatibleModel":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self._client.aclose()

    async def send(
        self, model: ModelConfig, request: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
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

    def __init__(self, responses: list[dict[str, Any] | LLMError]) -> None:
        self._responses = iter(responses)
        self.requests: list[dict[str, Any]] = []

    async def send(
        self, model: ModelConfig, request: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        self.requests.append(deepcopy(request))
        response = next(self._responses, None)
        if response is None:
            raise LLMError("假模型响应脚本已耗尽")
        if isinstance(response, LLMError):
            raise response
        return deepcopy(response)

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional

from aios_core.exceptions import ProviderError, RateLimitError, TimeoutError
from aios_core.models import ChatMessage, ChatResponse, EngineConfig, StreamingChunk, UsageStats
from aios_core.tokenizer import TokenCounter

logger = logging.getLogger("aios.core.provider")


class LLMProvider(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        config: EngineConfig,
        **kwargs: Any,
    ) -> ChatResponse:
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        config: EngineConfig,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingChunk]:
        ...

    @abstractmethod
    async def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        ...


class OpenAIProvider(LLMProvider):
    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._client: Any = None
        self._async_client: Any = None
        self._tokenizer = TokenCounter(config.model)
        self._logger = logging.getLogger("aios.core.provider.openai")
        self._init_clients()

    def _init_clients(self) -> None:
        try:
            import openai

            kwargs: dict[str, Any] = {}
            if self._config.api_key:
                kwargs["api_key"] = self._config.api_key
            if self._config.api_base:
                kwargs["base_url"] = self._config.api_base

            self._client = openai.OpenAI(**kwargs)
            self._async_client = openai.AsyncOpenAI(**kwargs)
            self._logger.info("OpenAI client initialized (model=%s)", self._config.model)
        except ImportError:
            self._logger.warning("openai package not installed; OpenAI provider unavailable")

    async def chat(
        self,
        messages: list[dict[str, Any]],
        config: EngineConfig,
        **kwargs: Any,
    ) -> ChatResponse:
        self._ensure_client()
        request_kwargs = self._build_request_kwargs(config, messages, stream=False, **kwargs)

        try:
            response = await self._async_client.chat.completions.create(**request_kwargs)
            choice = response.choices[0]
            msg = ChatMessage(
                role=choice.message.role or "assistant",
                content=choice.message.content or "",
                tool_calls=[self._format_tool_call(tc) for tc in choice.message.tool_calls] if choice.message.tool_calls else [],
            )
            usage = UsageStats(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )
            return ChatResponse(
                message=msg,
                usage=usage,
                finish_reason=choice.finish_reason or "stop",
            )
        except Exception as exc:
            raise self._map_error(exc)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        config: EngineConfig,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingChunk]:
        self._ensure_client()
        request_kwargs = self._build_request_kwargs(config, messages, stream=True, **kwargs)

        try:
            stream = await self._async_client.chat.completions.create(**request_kwargs)
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue
                finish = chunk.choices[0].finish_reason if chunk.choices else None
                yield StreamingChunk(
                    content=delta.content or "",
                    finish_reason=finish,
                )
        except Exception as exc:
            raise self._map_error(exc)

    async def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        return self._tokenizer.count_messages(messages)

    def _ensure_client(self) -> None:
        if self._async_client is None:
            raise ProviderError("OpenAI client not initialized. Install openai package.")

    def _build_request_kwargs(
        self,
        config: EngineConfig,
        messages: list[dict[str, Any]],
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", config.temperature),
            "max_tokens": kwargs.get("max_tokens", config.max_tokens),
            "top_p": kwargs.get("top_p", config.top_p),
            "stream": stream,
            "timeout": config.timeout_seconds,
        }
        if kwargs.get("tools"):
            request["tools"] = kwargs["tools"]
        if kwargs.get("tool_choice"):
            request["tool_choice"] = kwargs["tool_choice"]
        return request

    @staticmethod
    def _format_tool_call(tc: Any) -> dict[str, Any]:
        return {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        }

    @staticmethod
    def _map_error(exc: Exception) -> ProviderError:
        exc_str = str(exc).lower()
        if "rate limit" in exc_str or "429" in exc_str:
            return RateLimitError(str(exc))
        if "timeout" in exc_str or "timed out" in exc_str:
            return TimeoutError(str(exc))
        if "maximum context length" in exc_str or "context_length_exceeded" in exc_str:
            from aios_core.exceptions import ContextLengthExceededError

            return ContextLengthExceededError(str(exc))
        return ProviderError(str(exc))


class MockProvider(LLMProvider):
    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._logger = logging.getLogger("aios.core.provider.mock")

    async def chat(
        self,
        messages: list[dict[str, Any]],
        config: EngineConfig,
        **kwargs: Any,
    ) -> ChatResponse:
        last = messages[-1]["content"] if messages else ""
        response_text = f"Mock response to: {last[:50]}..."
        await asyncio.sleep(0.05)
        return ChatResponse(
            message=ChatMessage(role="assistant", content=response_text),
            usage=UsageStats(prompt_tokens=len(str(messages)) // 4, completion_tokens=len(response_text) // 4),
            finish_reason="stop",
        )

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        config: EngineConfig,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingChunk]:
        last = messages[-1]["content"] if messages else ""
        words = f"Mock response to: {last[:50]}...".split()
        for word in words:
            await asyncio.sleep(0.01)
            yield StreamingChunk(content=word + " ")
        yield StreamingChunk(content="", finish_reason="stop")

    async def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        return sum(len(m.get("content", "")) for m in messages) // 2

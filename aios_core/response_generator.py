from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional

from aios_core.exceptions import (
    AIOSEngineError,
    ContextLengthExceededError,
    ProviderError,
    RateLimitError,
    TimeoutError,
)
from aios_core.models import ChatMessage, ChatResponse, EngineConfig, StreamingChunk, UsageStats
from aios_core.provider import LLMProvider
from aios_core.token_manager import TokenManager, calculate_cost

logger = logging.getLogger("aios.core.response_generator")


class SafetyAction(str, Enum):
    RAISE = "raise"
    WARN = "warn"
    MASK = "mask"


@dataclass
class SafetyResult:
    passed: bool
    action: SafetyAction
    reason: str = ""
    masked_text: str = ""


@dataclass
class GenConfig:
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 4096
    stop_sequences: list[str] = field(default_factory=list)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    timeout_seconds: int = 60

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }
        if self.top_k > 0:
            d["top_k"] = self.top_k
        if self.stop_sequences:
            d["stop"] = self.stop_sequences
        if self.seed is not None:
            d["seed"] = self.seed
        return d


@dataclass
class StructuredOutputConfig:
    enabled: bool = False
    schema: dict[str, Any] | None = None
    schema_name: str = "response"
    mode: str = "json"
    fallback_on_error: bool = True
    max_retries_on_error: int = 2
    system_prompt_template: str = (
        "You are a helpful assistant that always responds with valid JSON. "
        "Do NOT include any text outside the JSON object. "
        "Respond with a JSON object conforming to this schema: {schema_description}"
    )

    def get_system_prompt(self) -> str:
        schema_desc = json.dumps(self.schema, indent=2) if self.schema else "any valid JSON object"
        return self.system_prompt_template.format(schema_description=schema_desc)


@dataclass
class ResponseMetrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    finish_reason: str = "stop"
    filtered: bool = False
    retry_count: int = 0
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "cost": round(self.cost, 8),
            "finish_reason": self.finish_reason,
            "filtered": self.filtered,
            "retry_count": self.retry_count,
            "cached": self.cached,
        }


@dataclass
class GenerationResult:
    text: str
    finish_reason: str = "stop"
    metrics: ResponseMetrics = field(default_factory=ResponseMetrics)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_chat_response(
        cls,
        response: ChatResponse,
        retry_count: int = 0,
        filtered: bool = False,
    ) -> GenerationResult:
        return cls(
            text=response.message.content,
            finish_reason=response.finish_reason,
            metrics=ResponseMetrics(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                latency_ms=response.latency_ms,
                cost=calculate_cost(
                    response.metadata.get("model", ""),
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                ),
                finish_reason=response.finish_reason,
                filtered=filtered,
                retry_count=retry_count,
                cached=response.cached,
            ),
        )


class SafetyFilterError(AIOSEngineError):
    pass


class SafetyFilter:
    def __init__(
        self,
        blocked_patterns: list[str] | None = None,
        blocked_words: list[str] | None = None,
        action: SafetyAction = SafetyAction.RAISE,
        enabled: bool = True,
    ) -> None:
        self._action = action
        self._enabled = enabled
        self._logger = logging.getLogger("aios.core.safety_filter")

        self._patterns: list[re.Pattern] = []
        if blocked_patterns:
            self._patterns = [re.compile(p, re.IGNORECASE) for p in blocked_patterns]

        self._blocked_words: list[str] = blocked_words or []

    def check_input(self, messages: list[dict[str, Any]]) -> SafetyResult:
        if not self._enabled:
            return SafetyResult(passed=True, action=SafetyAction.RAISE)

        for msg in messages:
            content = msg.get("content", "")
            result = self._check_content(content)
            if not result.passed:
                self._logger.warning(
                    "Input safety check failed: %s (action=%s)", result.reason, result.action.value
                )
                return result
        return SafetyResult(passed=True, action=self._action)

    def check_output(self, text: str) -> SafetyResult:
        if not self._enabled:
            return SafetyResult(passed=True, action=SafetyAction.RAISE)
        return self._check_content(text)

    def _check_content(self, text: str) -> SafetyResult:
        for word in self._blocked_words:
            if word.lower() in text.lower():
                if self._action == SafetyAction.MASK:
                    masked = re.sub(re.escape(word), "[FILTERED]", text, flags=re.IGNORECASE)
                    return SafetyResult(
                        passed=False,
                        action=self._action,
                        reason=f"Blocked word: {word}",
                        masked_text=masked,
                    )
                return SafetyResult(
                    passed=False,
                    action=self._action,
                    reason=f"Blocked word: {word}",
                )

        for pattern in self._patterns:
            match = pattern.search(text)
            if match:
                if self._action == SafetyAction.MASK:
                    masked = pattern.sub("[FILTERED]", text)
                    return SafetyResult(
                        passed=False,
                        action=self._action,
                        reason=f"Blocked pattern: {pattern.pattern}",
                        masked_text=masked,
                    )
                return SafetyResult(
                    passed=False,
                    action=self._action,
                    reason=f"Blocked pattern: {pattern.pattern}",
                )

        return SafetyResult(passed=True, action=self._action)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled


class JSONExtractionError(AIOSEngineError):
    pass


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise JSONExtractionError("No JSON object found in response")

    json_str = text[start : end + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise JSONExtractionError(f"Invalid JSON: {exc}") from exc


def validate_json_schema(data: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def validate_type(value: Any, expected: str, path: str) -> None:
        if expected == "string" and not isinstance(value, str):
            errors.append(f"{path}: expected string, got {type(value).__name__}")
        elif expected == "integer" and not isinstance(value, int):
            errors.append(f"{path}: expected integer, got {type(value).__name__}")
        elif expected == "number" and not isinstance(value, (int, float)):
            errors.append(f"{path}: expected number, got {type(value).__name__}")
        elif expected == "boolean" and not isinstance(value, bool):
            errors.append(f"{path}: expected boolean, got {type(value).__name__}")
        elif expected == "array" and not isinstance(value, list):
            errors.append(f"{path}: expected array, got {type(value).__name__}")
        elif expected == "object" and not isinstance(value, dict):
            errors.append(f"{path}: expected object, got {type(value).__name__}")

    def validate(data: Any, schema: dict[str, Any], path: str = "$") -> None:
        schema_type = schema.get("type", "object")

        if schema_type == "object":
            if not isinstance(data, dict):
                validate_type(data, "object", path)
                return
            required = schema.get("required", [])
            for req_key in required:
                if req_key not in data:
                    errors.append(f"{path}.{req_key}: missing required field")
            properties = schema.get("properties", {})
            for key, prop_schema in properties.items():
                if key in data:
                    validate(data[key], prop_schema, f"{path}.{key}")

        elif schema_type == "array":
            if not isinstance(data, list):
                validate_type(data, "array", path)
                return
            items_schema = schema.get("items", {})
            for i, item in enumerate(data):
                validate(item, items_schema, f"{path}[{i}]")

        else:
            validate_type(data, schema_type, path)

    validate(data, schema)
    return errors


class RetryHandler:
    def __init__(
        self,
        max_retries: int = 3,
        min_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        jitter: float = 0.1,
        retry_on_timeout: bool = True,
        retry_on_rate_limit: bool = True,
        retry_on_provider_error: bool = False,
    ) -> None:
        self.max_retries = max_retries
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retry_on_timeout = retry_on_timeout
        self.retry_on_rate_limit = retry_on_rate_limit
        self.retry_on_provider_error = retry_on_provider_error
        self._logger = logging.getLogger("aios.core.retry_handler")

    async def execute(
        self,
        coro_factory: Any,
        context_truncator: Any = None,
    ) -> Any:
        last_error: Exception | None = None
        delay = self.min_delay
        retry_count = 0

        for attempt in range(self.max_retries + 1):
            try:
                return await coro_factory()
            except RateLimitError as exc:
                last_error = exc
                if attempt < self.max_retries and self.retry_on_rate_limit:
                    jitter_amount = random.uniform(0, self.jitter * delay)
                    wait = delay + jitter_amount
                    self._logger.warning(
                        "Rate limited (attempt %d/%d). Retrying in %.2fs",
                        attempt + 1, self.max_retries + 1, wait,
                    )
                    await asyncio.sleep(wait)
                    delay = min(delay * self.backoff_factor, self.max_delay)
                    retry_count += 1
                else:
                    raise
            except TimeoutError as exc:
                last_error = exc
                if attempt < self.max_retries and self.retry_on_timeout:
                    self._logger.warning(
                        "Timeout (attempt %d/%d). Retrying in %.2fs",
                        attempt + 1, self.max_retries + 1, delay,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * self.backoff_factor, self.max_delay)
                    retry_count += 1
                else:
                    raise
            except ContextLengthExceededError as exc:
                if context_truncator and attempt < self.max_retries:
                    self._logger.warning(
                        "Context length exceeded. Truncating and retrying."
                    )
                    context_truncator()
                    retry_count += 1
                else:
                    raise
            except ProviderError as exc:
                last_error = exc
                if attempt < self.max_retries and self.retry_on_provider_error:
                    self._logger.warning(
                        "Provider error (attempt %d/%d): %s. Retrying in %.2fs",
                        attempt + 1, self.max_retries + 1, exc, delay,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * self.backoff_factor, self.max_delay)
                    retry_count += 1
                else:
                    raise

        raise ProviderError(
            f"Request failed after {self.max_retries} retries"
        ) from last_error

    @property
    def retry_count(self) -> int:
        return 0


class ResponseGenerator:
    def __init__(
        self,
        provider: LLMProvider,
        model: str = "gpt-4o",
        config: GenConfig | None = None,
        structured_output: StructuredOutputConfig | None = None,
        safety_filter: SafetyFilter | None = None,
        retry_handler: RetryHandler | None = None,
        token_manager: TokenManager | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._config = config or GenConfig()
        self._structured_output = structured_output or StructuredOutputConfig()
        self._safety = safety_filter or SafetyFilter()
        self._retry = retry_handler or RetryHandler()
        self._token_manager = token_manager or TokenManager(model=model)
        self._logger = logging.getLogger("aios.core.response_generator")

    @property
    def provider(self) -> LLMProvider:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def safety_filter(self) -> SafetyFilter:
        return self._safety

    @property
    def generation_config(self) -> GenConfig:
        return self._config

    def set_generation_config(self, config: GenConfig) -> None:
        self._config = config
        self._logger.info("Generation config updated")

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[dict[str, Any]],
        config: GenConfig | None = None,
        structured: StructuredOutputConfig | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        gen_config = config or self._config
        struct_config = structured or self._structured_output
        model_name = model or self._model

        safety_result = self._safety.check_input(messages)
        if not safety_result.passed:
            if safety_result.action == SafetyAction.RAISE:
                raise SafetyFilterError(
                    f"Input blocked by safety filter: {safety_result.reason}"
                )

        request_kwargs: dict[str, Any] = gen_config.to_dict()
        request_kwargs.update(kwargs)

        if struct_config.enabled:
            messages = self._inject_structured_system_prompt(
                messages, struct_config
            )
            if struct_config.mode == "json_schema" and struct_config.schema:
                request_kwargs["response_format"] = {
                    "type": "json_object",
                    "schema": struct_config.schema,
                }
            else:
                request_kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        retry_count = 0

        async def execute() -> ChatResponse:
            nonlocal retry_count
            try:
                return await asyncio.wait_for(
                    self._provider.chat(messages, EngineConfig(model=model_name), **request_kwargs),
                    timeout=gen_config.timeout_seconds,
                )
            except Exception as exc:
                raise

        def truncate_context() -> None:
            nonlocal messages
            if len(messages) > 2:
                system = [m for m in messages if m.get("role") == "system"]
                others = [m for m in messages if m.get("role") != "system"]
                messages = system + others[-(len(others) // 2):]

        response = await self._retry.execute(execute, truncate_context)
        latency = (time.perf_counter() - start) * 1000

        text = response.message.content

        # Structured output retry loop for JSON recovery
        if struct_config.enabled and struct_config.fallback_on_error:
            text = await self._ensure_structured_output(
                text, messages, struct_config, gen_config, model_name, request_kwargs,
            )

        # Safety filter on output
        filtered = False
        safety_result = self._safety.check_output(text)
        if not safety_result.passed:
            if safety_result.action == SafetyAction.RAISE:
                raise SafetyFilterError(
                    f"Output blocked by safety filter: {safety_result.reason}"
                )
            elif safety_result.action == SafetyAction.MASK and safety_result.masked_text:
                text = safety_result.masked_text
                filtered = True
            elif safety_result.action == SafetyAction.WARN:
                self._logger.warning(
                    "Output triggered safety filter: %s", safety_result.reason
                )

        result = GenerationResult(
            text=text,
            finish_reason=response.finish_reason,
            metrics=ResponseMetrics(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                latency_ms=latency,
                cost=self._token_manager.calculate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    model_name,
                ),
                finish_reason=response.finish_reason,
                retry_count=retry_count,
                cached=response.cached,
                filtered=filtered,
            ),
            metadata=response.metadata,
        )
        self._token_manager.track_request(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            latency,
        )
        return result

    # ------------------------------------------------------------------
    # Generate JSON / Structured
    # ------------------------------------------------------------------

    async def generate_json(
        self,
        messages: list[dict[str, Any]],
        config: GenConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        struct = StructuredOutputConfig(enabled=True, mode="json")
        result = await self.generate(messages, config, struct, **kwargs)
        return extract_json(result.text)

    async def generate_structured(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        config: GenConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        struct = StructuredOutputConfig(
            enabled=True,
            mode="json_schema",
            schema=schema,
        )
        result = await self.generate(messages, config, struct, **kwargs)

        data = extract_json(result.text)
        validation_errors = validate_json_schema(data, schema)
        if validation_errors:
            raise JSONExtractionError(
                f"JSON response does not match schema: {'; '.join(validation_errors)}"
            )
        return data

    # ------------------------------------------------------------------
    # Stream
    # ------------------------------------------------------------------

    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        config: GenConfig | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingChunk]:
        gen_config = config or self._config
        model_name = model or self._model

        safety_result = self._safety.check_input(messages)
        if not safety_result.passed:
            if safety_result.action == SafetyAction.RAISE:
                raise SafetyFilterError(
                    f"Input blocked by safety filter: {safety_result.reason}"
                )

        request_kwargs = gen_config.to_dict()
        request_kwargs.update(kwargs)

        start = time.perf_counter()
        full_content: list[str] = []
        finish_reason: str | None = None
        usage: UsageStats | None = None

        try:
            async for chunk in self._provider.chat_stream(
                messages, EngineConfig(model=model_name), **request_kwargs
            ):
                full_content.append(chunk.content)
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                if chunk.usage:
                    usage = chunk.usage
                yield chunk
        except Exception as exc:
            self._logger.error("Stream error: %s", exc)
            raise

        latency = (time.perf_counter() - start) * 1000
        text = "".join(full_content)

        filtered = False
        safety_result = self._safety.check_output(text)
        if not safety_result.passed:
            if safety_result.action == SafetyAction.RAISE:
                raise SafetyFilterError(
                    f"Stream output blocked by safety filter: {safety_result.reason}"
                )

        self._token_manager.track_request(
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            latency,
        )

    # ------------------------------------------------------------------
    # Prompt injection
    # ------------------------------------------------------------------

    def build_chat_messages(
        self,
        system_prompt: str = "",
        user_input: str = "",
        history: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        if user_input:
            messages.append({"role": "user", "content": user_input})
        return messages

    def _inject_structured_system_prompt(
        self,
        messages: list[dict[str, Any]],
        struct_config: StructuredOutputConfig,
    ) -> list[dict[str, Any]]:
        result = list(messages)
        sys_prompt = struct_config.get_system_prompt()
        has_system = any(m.get("role") == "system" for m in result)
        if has_system:
            for m in result:
                if m.get("role") == "system":
                    m["content"] = sys_prompt + "\n\n" + m["content"]
                    break
        else:
            result.insert(0, {"role": "system", "content": sys_prompt})
        return result

    async def _ensure_structured_output(
        self,
        text: str,
        messages: list[dict[str, Any]],
        struct_config: StructuredOutputConfig,
        gen_config: GenConfig,
        model_name: str,
        request_kwargs: dict[str, Any],
    ) -> str:
        attempt = 0
        current_text = text
        while attempt < struct_config.max_retries_on_error:
            try:
                extract_json(current_text)
                return current_text
            except JSONExtractionError:
                attempt += 1
                self._logger.warning(
                    "Structured output parse failed (attempt %d/%d). Retrying...",
                    attempt, struct_config.max_retries_on_error,
                )
                correction_msg = (
                    "Your previous response was not valid JSON. "
                    "Respond with ONLY a valid JSON object. No markdown, no explanation."
                )
                retry_messages = list(messages) + [
                    {"role": "assistant", "content": current_text},
                    {"role": "user", "content": correction_msg},
                ]
                try:
                    resp = await self._provider.chat(
                        retry_messages,
                        EngineConfig(model=model_name),
                        **request_kwargs,
                    )
                    current_text = resp.message.content
                except Exception as exc:
                    self._logger.error(
                        "Structured output retry failed: %s", exc
                    )
                    break
        return current_text

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        return self._token_manager.count_messages(messages)

    def estimate_cost(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> dict[str, float]:
        return self._token_manager.estimate_cost(messages, max_tokens)

    def estimate_request(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> dict[str, int | float]:
        token_est = self._token_manager.estimate_request_tokens(messages, max_tokens)
        cost_est = self._token_manager.estimate_cost(messages, max_tokens)
        return {**token_est, **cost_est}

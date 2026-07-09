from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Optional

from aios_core.exceptions import (
    AIOSEngineError,
    ConfigurationError,
    ContextLengthExceededError,
    ProviderError,
    RateLimitError,
    TimeoutError,
)
from aios_core.metrics import EngineMetrics, MetricsCollector
from aios_core.models import (
    ChatMessage,
    ChatResponse,
    EngineConfig,
    MessageRole,
    SessionState,
    StreamingChunk,
    UsageStats,
)
from aios_core.provider import LLMProvider, MockProvider, OpenAIProvider
from aios_core.token_manager import TokenManager

logger = logging.getLogger("aios.core.engine")


class AIOSEngine:
    """Production-grade LLM engine with memory, RAG, tools, and agent support.

    The ``AIOSEngine`` is the central orchestrator for all AI operations.
    It integrates:

    * **LLM providers** — OpenAI, Anthropic, or custom backends.
    * **Memory** — conversation memory + long-term memory via ``memory/``.
    * **RAG** — retrieval-augmented generation via ``rag/``.
    * **Tools** — dynamic tool calling via ``tools/``.
    * **Agents** — task routing via ``agents/``.

    Usage::

        from aios_core import AIOSEngine, EngineConfig

        engine = AIOSEngine(EngineConfig(model="gpt-4o", api_key="..."))

        # Basic chat
        response = await engine.achat([{"role": "user", "content": "Hello!"}])
        print(response.message.content)

        # Stream
        async for chunk in engine.stream([{"role": "user", "content": "Tell me a story"}]):
            print(chunk.content, end="")

        # With memory
        engine.configure_memory(memory_manager)
        sid = engine.create_session()
        response = await engine.achat([{"role": "user", "content": "Remember my name is Alice"}], session_id=sid)

        # With RAG
        engine.configure_rag(rag_pipeline)
        response = await engine.achat_with_rag("What does the documentation say about X?")

        # With tools
        engine.configure_tools(tool_manager)
        response = await engine.achat_with_tools([{"role": "user", "content": "Calculate 2+2"}])

        # Route to agent
        result = await engine.route_to_agent("Write a Python function")

        # Metrics
        print(engine.get_metrics())
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self._logger = logging.getLogger("aios.core.engine")
        self._metrics = MetricsCollector()
        self._provider: LLMProvider = self._init_provider()
        self._token_manager = TokenManager(model=self.config.model)
        self._sessions: dict[str, SessionState] = {}
        self._cache: dict[str, ChatResponse] = {}

        # External module integrations (set via configure_* methods)
        self._memory: Any = None
        self._rag: Any = None
        self._tools: Any = None
        self._agents: Any = None

        self._logger.info(
            "AIOSEngine initialized (provider=%s, model=%s, max_retries=%s)",
            config.provider if config else "default",
            config.model if config else "gpt-4o",
            config.max_retries if config else 3,
        )

    # ------------------------------------------------------------------
    # Provider
    # ------------------------------------------------------------------

    def _init_provider(self) -> LLMProvider:
        provider_map: dict[str, type[LLMProvider]] = {
            "openai": OpenAIProvider,
            "mock": MockProvider,
        }
        provider_cls = provider_map.get(self.config.provider)
        if provider_cls is None:
            self._logger.warning("Unknown provider '%s'; falling back to mock", self.config.provider)
            provider_cls = MockProvider
        return provider_cls(self.config)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_memory(self, memory_manager: Any) -> None:
        self._memory = memory_manager
        self._logger.info("Memory module attached")

    def configure_rag(self, rag_pipeline: Any) -> None:
        self._rag = rag_pipeline
        self._logger.info("RAG module attached")

    def configure_tools(self, tool_manager: Any) -> None:
        self._tools = tool_manager
        self._logger.info("Tool module attached")

    def configure_agents(self, agent_manager: Any) -> None:
        self._agents = agent_manager
        self._logger.info("Agent module attached")

    def set_provider(self, provider: LLMProvider) -> None:
        self._provider = provider
        self._logger.info("Provider set to %s", type(provider).__name__)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def create_session(
        self,
        session_id: str | None = None,
        system_prompt: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        sid = session_id or uuid.uuid4().hex[:16]
        self._sessions[sid] = SessionState(
            session_id=sid,
            system_prompt=system_prompt,
            metadata=metadata or {},
        )
        if system_prompt:
            self._sessions[sid].add_message(
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
            )
        self._logger.debug("Session created: %s", sid)
        return sid

    def get_session(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._logger.debug("Session deleted: %s", session_id)
            return True
        return False

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def clear_sessions(self) -> None:
        self._sessions.clear()
        self._logger.debug("All sessions cleared")

    def set_system_prompt(self, session_id: str, prompt: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            raise AIOSEngineError(f"Session not found: {session_id}")
        session.system_prompt = prompt
        if session.messages and session.messages[0].role == MessageRole.SYSTEM:
            session.messages[0].content = prompt
        else:
            session.messages.insert(0, ChatMessage(role=MessageRole.SYSTEM, content=prompt))

    # ------------------------------------------------------------------
    # Chat — sync
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]] | list[ChatMessage],
        session_id: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        return asyncio.run(self.achat(messages, session_id, **kwargs))

    # ------------------------------------------------------------------
    # Chat — async
    # ------------------------------------------------------------------

    async def achat(
        self,
        messages: list[dict[str, Any]] | list[ChatMessage],
        session_id: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        start = time.perf_counter()
        formatted = self._format_messages(messages)

        if session_id:
            self._update_session(session_id, formatted)

        context = await self._build_context(session_id)
        if context:
            system_msg = {"role": "system", "content": context}
            formatted = [system_msg] + [m for m in formatted if m.get("role") != "system"]

        self._check_context_length(formatted)

        cache_key = self._cache_key(formatted, kwargs)
        cached = self._get_cached(cache_key)
        if cached is not None:
            self._metrics.record_cache_hit()
            return cached

        response = await self._execute_with_retry(formatted, kwargs)
        latency = (time.perf_counter() - start) * 1000
        response.latency_ms = latency

        if session_id and response.message.role == MessageRole.ASSISTANT:
            session = self._sessions.get(session_id)
            if session:
                session.add_message(response.message)

        self._cache_response(cache_key, response)
        self._metrics.record_request(latency, response.usage.prompt_tokens, response.usage.completion_tokens)
        self._token_manager.track_request(response.usage.prompt_tokens, response.usage.completion_tokens, latency)

        await self._store_in_memory(session_id, formatted, response)

        return response

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: list[dict[str, Any]] | list[ChatMessage],
        session_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingChunk]:
        start = time.perf_counter()
        formatted = self._format_messages(messages)

        if session_id:
            self._update_session(session_id, formatted)

        context = await self._build_context(session_id)
        if context:
            system_msg = {"role": "system", "content": context}
            formatted = [system_msg] + [m for m in formatted if m.get("role") != "system"]

        self._check_context_length(formatted)

        full_content: list[str] = []
        finish_reason: str | None = None
        usage: UsageStats | None = None

        try:
            async for chunk in self._provider.chat_stream(formatted, self.config, **kwargs):
                full_content.append(chunk.content)
                yield chunk
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                if chunk.usage:
                    usage = chunk.usage
        except Exception as exc:
            self._logger.error("Stream error: %s", exc)
            raise

        latency = (time.perf_counter() - start) * 1000

        if session_id:
            content = "".join(full_content)
            response_msg = ChatMessage(role=MessageRole.ASSISTANT, content=content)
            session = self._sessions.get(session_id)
            if session:
                session.add_message(response_msg)

        self._metrics.record_request(
            latency,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )
        self._token_manager.track_request(
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            latency,
        )

    # ------------------------------------------------------------------
    # RAG
    # ------------------------------------------------------------------

    async def achat_with_rag(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> ChatResponse:
        if self._rag is None:
            raise ConfigurationError("RAG pipeline not configured. Call configure_rag() first.")

        self._metrics.record_rag_query()

        rag_results = await self._rag.query(query, top_k=top_k)
        context_parts = []
        for r in rag_results:
            text = r.get("text", "")
            score = r.get("score", 0)
            if score > 0.3:
                context_parts.append(text)

        rag_context = "\n\n".join(context_parts[:5]) if context_parts else ""
        system_prompt = (
            "You are a helpful assistant. Answer the user's question based on the "
            "provided context. If the context does not contain the answer, say so.\n\n"
            f"## Context\n{rag_context}" if rag_context else ""
        )

        messages = [{"role": "system", "content": system_prompt}]
        if session_id:
            session = self._sessions.get(session_id)
            if session:
                for msg in session.messages[-10:]:
                    messages.append(msg.to_dict())

        messages.append({"role": "user", "content": query})
        return await self.achat(messages, session_id=None, **kwargs)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    async def achat_with_tools(
        self,
        messages: list[dict[str, Any]] | list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
        max_tool_rounds: int = 5,
        **kwargs: Any,
    ) -> ChatResponse:
        if self._tools is None:
            raise ConfigurationError("Tool manager not configured. Call configure_tools() first.")

        formatted = self._format_messages(messages)
        tool_defs = tools or self._tools.list_tools()
        if not tool_defs:
            return await self.achat(messages, session_id, **kwargs)

        round_num = 0
        while round_num < max_tool_rounds:
            response = await self._execute_with_retry(formatted, {**kwargs, "tools": tool_defs})
            self._metrics.record_tool_call()

            if not response.message.tool_calls:
                return response

            for tc in response.message.tool_calls:
                func_name = tc.get("function", {}).get("name", "")
                func_args_str = tc.get("function", {}).get("arguments", "{}")
                try:
                    func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
                except json.JSONDecodeError:
                    func_args = {}

                tool_result = await self._tools.execute(func_name, func_args, permission_override=True)

                formatted.append({"role": "assistant", "content": "", "tool_calls": [tc]})
                formatted.append({
                    "role": "tool",
                    "content": json.dumps(tool_result.data) if tool_result.success else tool_result.error,
                    "tool_call_id": tc.get("id", ""),
                })

            round_num += 1

        return await self._execute_with_retry(formatted, kwargs)

    # ------------------------------------------------------------------
    # Agent routing
    # ------------------------------------------------------------------

    async def route_to_agent(
        self,
        task: str,
        agent_name: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Any:
        if self._agents is None:
            raise ConfigurationError("Agent manager not configured. Call configure_agents() first.")

        self._metrics.record_agent_route()
        return await self._agents.execute(task, agent_name, context)

    async def route_to_agent_plan(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> list[Any]:
        if self._agents is None:
            raise ConfigurationError("Agent manager not configured. Call configure_agents() first.")

        return await self._agents.execute_plan(task, context)

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        return         self._token_manager.count(text)

    def count_message_tokens(self, messages: list[dict[str, Any]] | list[ChatMessage]) -> int:
        formatted = self._format_messages(messages)
        return         self._token_manager.count_messages(formatted)

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    async def _build_context(self, session_id: str | None) -> str:
        parts: list[str] = []

        if session_id and self._memory:
            search_results = self._memory.search("", session_id=session_id, top_k=5)
            if search_results:
                memory_items = "\n".join(f"- {r.content[:200]}" for r in search_results[:3])
                parts.append(f"## Relevant Memories\n{memory_items}")

        if session_id:
            session = self._sessions.get(session_id)
            if session and session.system_prompt:
                parts.insert(0, session.system_prompt)

        if self._rag and session_id:
            session = self._sessions.get(session_id)
            if session and session.messages:
                last_query = ""
                for msg in reversed(session.messages):
                    if msg.role == MessageRole.USER:
                        last_query = msg.content
                        break
                if last_query:
                    rag_results = await self._rag.query(last_query, top_k=3)
                    rag_context = "\n".join(
                        r.get("text", "") for r in rag_results if r.get("score", 0) > 0.3
                    )
                    if rag_context:
                        parts.append(f"## Retrieved Context\n{rag_context}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    async def _execute_with_retry(
        self,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> ChatResponse:
        last_error: Exception | None = None
        delay = self.config.retry_min_delay

        for attempt in range(self.config.max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self._provider.chat(messages, self.config, **kwargs),
                    timeout=self.config.timeout_seconds,
                )
            except RateLimitError as exc:
                self._metrics.record_rate_limit()
                last_error = exc
                if attempt < self.config.max_retries:
                    jitter = random.uniform(0, 0.1 * delay)
                    await asyncio.sleep(delay + jitter)
                    delay = min(delay * 2, self.config.retry_max_delay)
            except TimeoutError as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self.config.retry_max_delay)
            except ContextLengthExceededError:
                messages = self._truncate_messages(messages)
                last_error = None
            except ProviderError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self.config.retry_max_delay)

        raise ProviderError(f"Request failed after {self.config.max_retries} retries") from last_error

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_messages(
        self,
        messages: list[dict[str, Any]] | list[ChatMessage],
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        if isinstance(messages[0], ChatMessage):
            return [m.to_dict() for m in messages]  # type: ignore[union-attr]
        return messages  # type: ignore[return-value]

    def _update_session(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        session = self._sessions[session_id]
        for msg_dict in messages:
            msg = ChatMessage.from_dict(msg_dict)
            session.add_message(msg)

    def _check_context_length(self, messages: list[dict[str, Any]]) -> None:
        total =         self._token_manager.count_messages(messages)
        if total > self.config.max_context_length:
            raise ContextLengthExceededError(
                f"Context length {total} exceeds maximum {self.config.max_context_length}. "
                "Consider truncating messages or increasing max_context_length."
            )

    def _truncate_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(messages) <= 2:
            return messages
        system_messages = [m for m in messages if m.get("role") == "system"]
        other = [m for m in messages if m.get("role") != "system"]
        truncated = other[-(len(other) // 2):]
        return system_messages + truncated

    def _cache_key(self, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> str:
        content = json.dumps(messages, sort_keys=True) + json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached(self, key: str) -> ChatResponse | None:
        if not self.config.enable_caching:
            return None
        return self._cache.get(key)

    def _cache_response(self, key: str, response: ChatResponse) -> None:
        if not self.config.enable_caching:
            return
        if len(self._cache) > 1000:
            self._cache.pop(next(iter(self._cache)), None)
        self._cache[key] = response

    async def _store_in_memory(
        self,
        session_id: str | None,
        messages: list[dict[str, Any]],
        response: ChatResponse,
    ) -> None:
        if self._memory is None or session_id is None:
            return
        try:
            for msg in messages:
                self._memory.add_message(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    session_id=session_id,
                )
            self._memory.add_message(
                role="assistant",
                content=response.message.content,
                session_id=session_id,
            )
        except Exception as exc:
            self._logger.warning("Failed to store message in memory: %s", exc)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> EngineMetrics:
        return self._metrics.get_metrics()

    def get_metrics_dict(self) -> dict[str, Any]:
        return self._metrics.get_metrics().to_dict()

    def reset_metrics(self) -> None:
        self._metrics.reset()
        self._logger.debug("Metrics reset")

    # ------------------------------------------------------------------
    # Prompt management
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        template: str,
        **variables: Any,
    ) -> str:
        return template.format(**variables)

    def build_chat_prompt(
        self,
        system_prompt: str = "",
        conversation: list[ChatMessage] | None = None,
        user_input: str = "",
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if conversation:
            for msg in conversation:
                messages.append(msg.to_dict())
        if user_input:
            messages.append({"role": "user", "content": user_input})
        return messages

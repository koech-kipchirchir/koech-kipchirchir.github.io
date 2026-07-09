from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

logger = logging.getLogger("aios.core.context")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ContextSource(str, Enum):
    SYSTEM_PROMPT = "system_prompt"
    CONVERSATION = "conversation"
    MEMORY = "memory"
    DOCUMENTS = "documents"
    TOOL_OUTPUTS = "tool_outputs"
    AGENT_RESPONSES = "agent_responses"
    CUSTOM = "custom"


class Priority(int, Enum):
    CRITICAL = 100
    HIGH = 75
    MEDIUM = 50
    LOW = 25
    BACKGROUND = 10


class TruncationStrategy(str, Enum):
    DROP_LOWEST_PRIORITY = "drop_lowest_priority"
    SLIDING_WINDOW = "sliding_window"
    TRUNCATE_EACH = "truncate_each"
    SUMMARIZE = "summarize"
    FAIL = "fail"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ContextBudget:
    """Token budget allocation for each context source type (percentage)."""

    system_prompt: float = 0.10
    conversation: float = 0.40
    memory: float = 0.10
    documents: float = 0.20
    tool_outputs: float = 0.10
    agent_responses: float = 0.10

    def __post_init__(self) -> None:
        total = sum([
            self.system_prompt,
            self.conversation,
            self.memory,
            self.documents,
            self.tool_outputs,
            self.agent_responses,
        ])
        if abs(total - 1.0) > 0.01:
            scale = 1.0 / total
            self.system_prompt *= scale
            self.conversation *= scale
            self.memory *= scale
            self.documents *= scale
            self.tool_outputs *= scale
            self.agent_responses *= scale

    def token_limit_for(self, source: ContextSource, total_budget: int) -> int:
        ratio = self._ratio(source)
        return max(64, int(total_budget * ratio))

    def _ratio(self, source: ContextSource) -> float:
        mapping: dict[ContextSource, float] = {
            ContextSource.SYSTEM_PROMPT: self.system_prompt,
            ContextSource.CONVERSATION: self.conversation,
            ContextSource.MEMORY: self.memory,
            ContextSource.DOCUMENTS: self.documents,
            ContextSource.TOOL_OUTPUTS: self.tool_outputs,
            ContextSource.AGENT_RESPONSES: self.agent_responses,
        }
        return mapping.get(source, 0.05)


@dataclass
class ContextSection:
    """A single section of context with metadata."""

    source: ContextSource
    content: str
    priority: Priority = Priority.MEDIUM
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False


@dataclass
class ContextConfig:
    """Configuration for the context builder."""

    max_context_tokens: int = 8192
    max_sequence_length: int = 128000
    reserve_tokens: int = 1024
    truncation_strategy: TruncationStrategy = TruncationStrategy.DROP_LOWEST_PRIORITY
    budget: ContextBudget = field(default_factory=ContextBudget)
    sliding_window_size: int = 20
    summarize_long_memory: bool = True
    max_document_chars: int = 5000

    @property
    def available_tokens(self) -> int:
        return max(1, min(self.max_context_tokens, self.max_sequence_length) - self.reserve_tokens)


# ---------------------------------------------------------------------------
# Token Counter
# ---------------------------------------------------------------------------

class TokenCounter:
    """Lightweight token estimator with optional tiktoken integration."""

    def __init__(self) -> None:
        self._encoding: Any = None
        self._init_encoding()

    def _init_encoding(self) -> None:
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            pass

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(text, disallowed_special=()))
        return max(1, len(text) // 2)

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            total += self.count(msg.get("content", ""))
            total += self.count(msg.get("role", ""))
        total += len(messages) * 4 + 3
        return total


# ---------------------------------------------------------------------------
# Context Builder
# ---------------------------------------------------------------------------

class ContextBuilder:
    """Build and manage LLM context from multiple sources with token budgeting.

    Merges conversation history, memory, retrieved documents, tool outputs,
    agent responses, and system prompts into a single context string that
    respects configurable token limits.

    Usage::

        from aios_core.context_builder import ContextBuilder, ContextConfig

        cb = ContextBuilder(ContextConfig(max_context_tokens=32000))

        cb.add_system_prompt("You are a helpful AI.")
        cb.add_conversation([{"role": "user", "content": "Hello"}])
        cb.add_documents(["Relevant doc content..."])
        cb.add_tool_output("Calculation result: 42")

        built = cb.build()
        print(built.full_text)
        print(built.token_count, built.truncated_sections)
    """

    def __init__(self, config: ContextConfig | None = None) -> None:
        self.config = config or ContextConfig()
        self._sections: list[ContextSection] = []
        self._lock = threading.Lock()
        self._tokenizer = TokenCounter()
        self._logger = logging.getLogger("aios.core.context.builder")

    # -- Section Adders ----------------------------------------------------

    def add_system_prompt(self, content: str, priority: Priority = Priority.CRITICAL) -> ContextSection:
        return self._add(ContextSource.SYSTEM_PROMPT, content, priority)

    def add_conversation(
        self,
        messages: list[dict[str, Any]],
        priority: Priority = Priority.HIGH,
    ) -> ContextSection:
        text = self._format_messages(messages)
        return self._add(ContextSource.CONVERSATION, text, priority, metadata={"message_count": len(messages)})

    def add_memory(
        self,
        memories: list[str],
        priority: Priority = Priority.LOW,
    ) -> ContextSection:
        if self.config.summarize_long_memory and len(memories) > 5:
            text = self._summarize_memories(memories)
        else:
            text = "\n".join(f"- {m}" for m in memories)
        return self._add(ContextSource.MEMORY, text, priority, metadata={"count": len(memories)})

    def add_documents(
        self,
        documents: list[str],
        priority: Priority = Priority.MEDIUM,
    ) -> ContextSection:
        truncated: list[str] = []
        for doc in documents:
            if len(doc) > self.config.max_document_chars:
                doc = doc[:self.config.max_document_chars] + "\n...[truncated]..."
            truncated.append(doc)
        text = "\n\n".join(f"--- Document {i+1} ---\n{d}" for i, d in enumerate(truncated))
        return self._add(ContextSource.DOCUMENTS, text, priority, metadata={"count": len(documents)})

    def add_tool_output(
        self,
        output: str,
        tool_name: str = "",
        priority: Priority = Priority.HIGH,
    ) -> ContextSection:
        prefix = f"[Tool: {tool_name}]\n" if tool_name else ""
        return self._add(ContextSource.TOOL_OUTPUTS, prefix + output, priority, metadata={"tool": tool_name})

    def add_agent_response(
        self,
        response: str,
        agent_name: str = "",
        priority: Priority = Priority.HIGH,
    ) -> ContextSection:
        prefix = f"[Agent: {agent_name}]\n" if agent_name else ""
        return self._add(ContextSource.AGENT_RESPONSES, prefix + response, priority, metadata={"agent": agent_name})

    def add_custom(
        self,
        content: str,
        source_name: str = "custom",
        priority: Priority = Priority.MEDIUM,
    ) -> ContextSection:
        return self._add(ContextSource.CUSTOM, content, priority, metadata={"source_name": source_name})

    # -- Build -------------------------------------------------------------

    def build(self) -> BuiltContext:
        """Build the final context, applying truncation if over budget."""
        sections = list(self._sections)
        sections.sort(key=lambda s: s.priority.value, reverse=True)

        available = self.config.available_tokens
        sections = self._apply_token_budget(sections, available)
        sections = self._truncate(sections, available)

        full_text_parts: list[str] = []
        total_tokens = 0
        truncated_names: list[str] = []

        for section in sections:
            if not section.content:
                continue
            full_text_parts.append(section.content)
            total_tokens += section.token_count
            if section.truncated:
                truncated_names.append(section.source.value)

        full_text = "\n\n".join(full_text_parts)

        return BuiltContext(
            full_text=full_text,
            token_count=total_tokens,
            sections=sections,
            truncated_sections=truncated_names,
            remaining_budget=available - total_tokens,
            config=self.config,
        )

    def build_chat_messages(
        self,
        system_prompt: str = "",
        user_input: str = "",
    ) -> list[dict[str, Any]]:
        """Build context and return as chat messages for an LLM API."""
        context = self.build()
        messages: list[dict[str, Any]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if context.full_text:
            messages.append({"role": "system", "content": context.full_text})

        if user_input:
            messages.append({"role": "user", "content": user_input})

        return messages

    # -- Reset & Info ------------------------------------------------------

    def reset(self) -> None:
        with self._lock:
            self._sections.clear()

    def current_sections(self) -> list[ContextSection]:
        return list(self._sections)

    def current_token_count(self) -> int:
        return sum(s.token_count for s in self._sections)

    @property
    def section_count(self) -> int:
        return len(self._sections)

    # -- Internal ----------------------------------------------------------

    def _add(
        self,
        source: ContextSource,
        content: str,
        priority: Priority,
        metadata: dict[str, Any] | None = None,
    ) -> ContextSection:
        section = ContextSection(
            source=source,
            content=content.strip(),
            priority=priority,
            token_count=self._tokenizer.count(content),
            metadata=metadata or {},
        )
        with self._lock:
            self._sections.append(section)
        return section

    def _apply_token_budget(
        self,
        sections: list[ContextSection],
        available: int,
    ) -> list[ContextSection]:
        budgeted: list[ContextSection] = []
        for section in sections:
            limit = self.config.budget.token_limit_for(section.source, available)
            if section.token_count > limit:
                ratio = limit / max(section.token_count, 1)
                mid = int(len(section.content) * ratio)
                section.content = section.content[:mid] + "\n...[truncated by budget]..."
                section.token_count = self._tokenizer.count(section.content)
                section.truncated = True
            budgeted.append(section)
        return budgeted

    def _truncate(
        self,
        sections: list[ContextSection],
        available: int,
    ) -> list[ContextSection]:
        total = sum(s.token_count for s in sections)
        if total <= available:
            return sections

        strategy = self.config.truncation_strategy

        if strategy == TruncationStrategy.FAIL:
            raise ContextOverflowError(
                f"Context exceeds limit: {total} > {available} tokens"
            )

        if strategy == TruncationStrategy.DROP_LOWEST_PRIORITY:
            return self._drop_lowest_priority(sections, available)

        if strategy == TruncationStrategy.SLIDING_WINDOW:
            return self._sliding_window(sections, available)

        if strategy == TruncationStrategy.TRUNCATE_EACH:
            return self._truncate_each(sections, available)

        if strategy == TruncationStrategy.SUMMARIZE:
            return self._summarize_overflow(sections, available)

        return sections

    def _drop_lowest_priority(
        self,
        sections: list[ContextSection],
        available: int,
    ) -> list[ContextSection]:
        keep: list[ContextSection] = []
        running = 0

        for section in sections:
            if running + section.token_count <= available:
                keep.append(section)
                running += section.token_count
            else:
                remaining = available - running
                if remaining > 64 and section.token_count > remaining:
                    ratio = remaining / max(section.token_count, 1)
                    mid = int(len(section.content) * ratio)
                    section.content = section.content[:mid]
                    section.token_count = self._tokenizer.count(section.content)
                    section.truncated = True
                    keep.append(section)
                    running += section.token_count
                    break
                else:
                    self._logger.debug(
                        "Dropped section: %s (priority=%s, tokens=%s)",
                        section.source.value, section.priority.value, section.token_count,
                    )
                    section.content = ""
                    section.token_count = 0
                    section.truncated = True

        return keep

    def _sliding_window(
        self,
        sections: list[ContextSection],
        available: int,
    ) -> list[ContextSection]:
        conv_sections = [s for s in sections if s.source == ContextSource.CONVERSATION]
        others = [s for s in sections if s.source != ContextSource.CONVERSATION]

        other_tokens = sum(s.token_count for s in others)
        conv_budget = available - other_tokens

        for section in conv_sections:
            lines = section.content.split("\n")
            window = self.config.sliding_window_size
            if len(lines) > window:
                section.content = "\n".join(lines[-window:])
                section.token_count = self._tokenizer.count(section.content)
                section.truncated = True

            if section.token_count > conv_budget:
                ratio = conv_budget / max(section.token_count, 1)
                mid = int(len(section.content) * ratio)
                section.content = section.content[-mid:]
                section.token_count = self._tokenizer.count(section.content)
                section.truncated = True

        return others + conv_sections

    def _truncate_each(
        self,
        sections: list[ContextSection],
        available: int,
    ) -> list[ContextSection]:
        total = sum(s.token_count for s in sections)
        if total <= available:
            return sections

        ratio = available / total
        for section in sections:
            max_tok = int(section.token_count * ratio)
            if max_tok < 64:
                section.content = ""
                section.token_count = 0
            else:
                scale = max_tok / max(section.token_count, 1)
                mid = int(len(section.content) * scale)
                section.content = section.content[:mid]
                section.token_count = self._tokenizer.count(section.content)
            section.truncated = True
        return sections

    def _summarize_overflow(
        self,
        sections: list[ContextSection],
        available: int,
    ) -> list[ContextSection]:
        low_priority = [s for s in sections if s.priority.value < Priority.HIGH.value]
        high_priority = [s for s in sections if s.priority.value >= Priority.HIGH.value]

        hp_tokens = sum(s.token_count for s in high_priority)
        if hp_tokens > available:
            return self._drop_lowest_priority(high_priority, available)

        remaining = available - hp_tokens
        for section in low_priority:
            if section.token_count > remaining:
                words = section.content.split()
                summary = " ".join(words[:50])
                if len(words) > 50:
                    summary += "\n...[summarized]..."
                section.content = summary
                section.token_count = self._tokenizer.count(summary)
                section.truncated = True

        all_sections = high_priority + low_priority
        running = sum(s.token_count for s in all_sections)
        if running > available:
            return self._drop_lowest_priority(all_sections, available)

        return all_sections

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    @staticmethod
    def _summarize_memories(memories: list[str]) -> str:
        if not memories:
            return ""
        header = f"[{len(memories)} memories stored. Most recent:]\n"
        recent = memories[-5:]
        body = "\n".join(f"- {m[:200]}" for m in recent)
        return header + body

    def count_tokens(self, text: str) -> int:
        return self._tokenizer.count(text)


# ---------------------------------------------------------------------------
# Built Context
# ---------------------------------------------------------------------------

@dataclass
class BuiltContext:
    """The final assembled context ready for inference."""

    full_text: str
    token_count: int
    sections: list[ContextSection]
    truncated_sections: list[str]
    remaining_budget: int
    config: ContextConfig

    @property
    def utilization_pct(self) -> float:
        available = self.config.available_tokens
        if available == 0:
            return 0.0
        return (self.token_count / available) * 100

    @property
    def is_under_budget(self) -> bool:
        return self.remaining_budget >= 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_count": self.token_count,
            "available_tokens": self.config.available_tokens,
            "utilization_pct": round(self.utilization_pct, 1),
            "truncated_sections": self.truncated_sections,
            "section_count": len(self.sections),
            "remaining_budget": self.remaining_budget,
            "is_under_budget": self.is_under_budget,
        }

    def __str__(self) -> str:
        return (
            f"BuiltContext(tokens={self.token_count}/{self.config.available_tokens}, "
            f"util={self.utilization_pct:.0f}%, "
            f"truncated={self.truncated_sections or 'none'})"
        )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ContextOverflowError(Exception):
    """Raised when context exceeds limits and strategy is ``FAIL``."""

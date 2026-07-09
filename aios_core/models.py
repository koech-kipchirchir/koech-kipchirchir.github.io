from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ChatMessage:
    role: MessageRole | str
    content: str
    name: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "role": self.role.value if isinstance(self.role, MessageRole) else self.role,
            "content": self.content,
        }
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatMessage:
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            name=data.get("name", ""),
            tool_calls=data.get("tool_calls", []),
            tool_call_id=data.get("tool_call_id", ""),
        )


@dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ChatResponse:
    message: ChatMessage
    usage: UsageStats = field(default_factory=UsageStats)
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingChunk:
    content: str
    finish_reason: str | None = None
    usage: UsageStats | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    messages: list[ChatMessage] = field(default_factory=list)
    system_prompt: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    max_messages: int = 100

    def add_message(self, message: ChatMessage) -> None:
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def to_openai_format(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self.messages]


@dataclass
class EngineConfig:
    model: str = "gpt-4o"
    provider: str = "openai"
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_min_delay: float = 1.0
    retry_max_delay: float = 30.0
    max_context_length: int = 128000
    stream_chunk_size: int = 5
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    log_level: str = "INFO"

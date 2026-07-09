from __future__ import annotations

import logging
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Callable, Optional


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentMessage:
    role: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    success: bool
    output: str
    agent_name: str
    duration_ms: float = 0.0
    tokens_used: int = 0
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    name: str = "agent"
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str = ""
    memory_enabled: bool = True
    max_retries: int = 3
    timeout_seconds: int = 120
    tools: list[str] = field(default_factory=list)


StreamCallback = Callable[[str], None]


class BaseAgent(ABC):
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.status = AgentStatus.IDLE
        self.conversation_history: list[AgentMessage] = []
        self._lock = threading.Lock()
        self._logger = logging.getLogger(f"aios.agent.{self.config.name}")
        self._session_id: str = uuid.uuid4().hex[:12]
        self._start_time: float = 0.0

    @abstractmethod
    async def execute(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        ...

    async def stream(
        self, task: str, callback: StreamCallback, context: dict[str, Any] | None = None
    ) -> AgentResult:
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")

    def add_message(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        self.conversation_history.append(
            AgentMessage(role=role, content=content, metadata=metadata or {})
        )

    @property
    def session_id(self) -> str:
        return self._session_id

    def reset(self) -> None:
        self.conversation_history.clear()
        self.status = AgentStatus.IDLE

    def cancel(self) -> None:
        with self._lock:
            self.status = AgentStatus.CANCELLED

    @property
    def name(self) -> str:
        return self.config.name

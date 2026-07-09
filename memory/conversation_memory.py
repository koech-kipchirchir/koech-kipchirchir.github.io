from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from memory.storage import SQLiteStorage
from memory.utils import now_utc, setup_logger, thread_safe, timestamp_ms

logger = setup_logger("aios.memory.conversation")


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=now_utc)
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=MessageRole(data.get("role", "user")),
            content=data.get("content", ""),
            timestamp=datetime.fromisoformat(data.get("timestamp", now_utc().isoformat())),
            message_id=data.get("message_id", uuid.uuid4().hex[:16]),
            metadata=data.get("metadata", {}),
        )


class ConversationMemory:
    def __init__(
        self,
        storage: SQLiteStorage,
        session_id: str | None = None,
        max_turns: int = 50,
    ) -> None:
        self._storage = storage
        self._session_id = session_id or uuid.uuid4().hex[:16]
        self._max_turns = max_turns
        self._messages: list[Message] = []
        self._lock = threading.Lock()
        self._logger = setup_logger("aios.memory.conversation")
        self._storage.create_session(self._session_id)
        self._logger.info("Conversation session: %s (max_turns=%s)", self._session_id, max_turns)

    @property
    def session_id(self) -> str:
        return self._session_id

    @thread_safe
    def add_message(
        self,
        role: MessageRole | str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        if isinstance(role, str):
            role = MessageRole(role)
        msg = Message(role=role, content=content, metadata=metadata or {})
        self._messages.append(msg)
        self._storage.add_message(
            session_id=self._session_id,
            role=role.value,
            content=content,
            metadata={"message_id": msg.message_id, **(metadata or {})},
        )
        self._prune()
        return msg

    @thread_safe
    def get_recent(self, n: int = 10) -> list[Message]:
        return self._messages[-n:]

    @thread_safe
    def get_all(self) -> list[Message]:
        return list(self._messages)

    @thread_safe
    def get_by_role(self, role: MessageRole) -> list[Message]:
        return [m for m in self._messages if m.role == role]

    @thread_safe
    def clear(self) -> None:
        self._messages.clear()
        self._storage.delete_session(self._session_id)
        self._storage.create_session(self._session_id)

    @property
    def turn_count(self) -> int:
        return len([m for m in self._messages if m.role == MessageRole.USER])

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def _prune(self) -> None:
        while len(self._messages) > self._max_turns * 2:
            removed = self._messages.pop(0)
            self._logger.debug("Pruned message: %s", removed.message_id)

    def to_messages_list(self) -> list[dict[str, str]]:
        return [
            {"role": m.role.value, "content": m.content}
            for m in self._messages
        ]

    def __len__(self) -> int:
        return len(self._messages)

    def __getitem__(self, index: int) -> Message:
        return self._messages[index]

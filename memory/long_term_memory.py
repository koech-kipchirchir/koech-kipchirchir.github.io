from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from memory.storage import SQLiteStorage
from memory.utils import now_utc, setup_logger, thread_safe, timestamp_ms

logger = setup_logger("aios.memory.long_term")


@dataclass
class MemoryNode:
    content: str
    importance: float = 0.0
    summary: str = ""
    session_id: str = ""
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: datetime = field(default_factory=now_utc)
    accessed_at: datetime = field(default_factory=now_utc)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "session_id": self.session_id,
            "content": self.content,
            "summary": self.summary,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryNode:
        return cls(
            node_id=data.get("node_id", uuid.uuid4().hex[:16]),
            session_id=data.get("session_id", ""),
            content=data.get("content", ""),
            summary=data.get("summary", ""),
            importance=data.get("importance", 0.0),
            created_at=datetime.fromisoformat(data.get("created_at", now_utc().isoformat())),
            accessed_at=datetime.fromisoformat(data.get("accessed_at", now_utc().isoformat())),
            metadata=data.get("metadata", {}),
        )


SummarizerFn = Callable[[list[str]], str]


class LongTermMemory:
    def __init__(
        self,
        storage: SQLiteStorage,
        max_nodes: int = 1000,
        importance_threshold: float = 0.3,
        summarizer: SummarizerFn | None = None,
    ) -> None:
        self._storage = storage
        self._max_nodes = max_nodes
        self._importance_threshold = importance_threshold
        self._summarizer = summarizer
        self._nodes: dict[str, MemoryNode] = {}
        self._lock = threading.Lock()
        self._logger = setup_logger("aios.memory.long_term")

    @thread_safe
    def add(
        self,
        content: str,
        session_id: str = "",
        importance: float | None = None,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryNode:
        imp = importance if importance is not None else self._estimate_importance(content)
        if imp < self._importance_threshold:
            self._logger.debug("Skipping low-importance memory (%.2f < %.2f)", imp, self._importance_threshold)
            pass

        node = MemoryNode(
            content=content,
            importance=imp,
            session_id=session_id,
            embedding=embedding,
            metadata=metadata or {},
        )
        self._nodes[node.node_id] = node
        self._storage.add_memory_node(
            session_id=session_id,
            content=content,
            importance=imp,
            embedding=embedding,
            metadata=metadata,
        )
        self._prune()
        return node

    @thread_safe
    def get(self, node_id: str) -> MemoryNode | None:
        node = self._nodes.get(node_id)
        if node is not None:
            node.accessed_at = now_utc()
            self._storage.update_memory_node_access(int(node_id, 16) % 10**9)
        return node

    @thread_safe
    def get_all(self, session_id: str = "") -> list[MemoryNode]:
        if session_id:
            return [n for n in self._nodes.values() if n.session_id == session_id]
        return list(self._nodes.values())

    @thread_safe
    def search_by_importance(self, min_importance: float = 0.0, limit: int = 20) -> list[MemoryNode]:
        sorted_nodes = sorted(self._nodes.values(), key=lambda n: n.importance, reverse=True)
        return [n for n in sorted_nodes if n.importance >= min_importance][:limit]

    def summarize(self, session_id: str = "") -> str:
        nodes = self.get_all(session_id)
        if not nodes:
            return ""
        texts = [n.content for n in nodes[:20]]
        if self._summarizer:
            return self._summarizer(texts)
        return self._default_summarize(texts)

    def consolidate(self, session_id: str = "") -> None:
        nodes = self.get_all(session_id)
        if len(nodes) < 10:
            return
        summary = self.summarize(session_id)
        if summary:
            self.add(
                content=f"[Consolidated Summary] {summary}",
                session_id=session_id,
                importance=0.9,
                metadata={"type": "consolidation"},
            )

    @thread_safe
    def cleanup_expired(self) -> int:
        return self._storage.delete_expired_nodes()

    def _prune(self) -> None:
        if len(self._nodes) <= self._max_nodes:
            return

        sorted_nodes = sorted(self._nodes.values(), key=lambda n: (n.importance, n.accessed_at.timestamp()))
        to_remove = sorted_nodes[:len(self._nodes) - self._max_nodes]
        for node in to_remove:
            self._nodes.pop(node.node_id, None)
        self._logger.info("Pruned %s memory nodes", len(to_remove))

    @staticmethod
    def _estimate_importance(content: str) -> float:
        important_keywords = [
            "remember", "important", "critical", "key", "essential",
            "always", "never", "must", "required", "vital",
            "name", "birthday", "preference", "password", "address",
            "favorite", "love", "hate", "want", "need",
        ]
        cl = content.lower()
        score = sum(1 for kw in important_keywords if kw in cl)
        base = min(score * 0.15, 0.9)
        base += min(len(content) / 1000 * 0.1, 0.1)
        return min(base, 1.0)

    @staticmethod
    def _default_summarize(texts: list[str]) -> str:
        if not texts:
            return ""
        combined = " ".join(texts)
        words = combined.split()
        if len(words) <= 200:
            return combined
        return " ".join(words[:200]) + "..."

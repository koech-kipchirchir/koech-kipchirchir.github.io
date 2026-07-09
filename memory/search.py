from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from memory.conversation_memory import ConversationMemory, Message, MessageRole
from memory.embeddings import EmbeddingProvider
from memory.long_term_memory import LongTermMemory, MemoryNode
from memory.utils import setup_logger
from memory.vector_memory import VectorStore

logger = setup_logger("aios.memory.search")


@dataclass
class SearchResult:
    content: str
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    node_id: str = ""


class MemorySearch:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._lock = threading.Lock()
        self._logger = setup_logger("aios.memory.search")

    def search_conversation(
        self,
        memory: ConversationMemory,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored: list[tuple[Message, float]] = []
        for msg in memory.get_all():
            msg_lower = msg.content.lower()
            word_overlap = len(query_words & set(msg_lower.split()))
            if word_overlap > 0:
                score = word_overlap / max(len(query_words), 1)
                scored.append((msg, score))

        scored.sort(key=lambda x: -x[1])
        return [
            SearchResult(
                content=msg.content,
                score=score,
                source="conversation",
                metadata={"role": msg.role.value, "message_id": msg.message_id},
            )
            for msg, score in scored[:top_k]
        ]

    def search_long_term(
        self,
        memory: LongTermMemory,
        query: str,
        top_k: int = 5,
        session_id: str = "",
    ) -> list[SearchResult]:
        query_lower = query.lower()
        scored: list[tuple[MemoryNode, float]] = []

        for node in memory.get_all(session_id):
            text = f"{node.content} {node.summary}".lower()
            overlap = len(set(query_lower.split()) & set(text.split()))
            if overlap > 0 or node.importance > 0.5:
                score = (overlap * 0.4 + node.importance * 0.6)
                scored.append((node, score))

        scored.sort(key=lambda x: -x[1])
        return [
            SearchResult(
                content=node.content,
                score=score,
                source="long_term",
                node_id=node.node_id,
                metadata={"importance": node.importance, "session_id": node.session_id},
            )
            for node, score in scored[:top_k]
        ]

    def search_vector(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if self._vector_store is None:
            return []

        query_emb = self._embeddings.embed(query)
        results = self._vector_store.search(query_emb, top_k, filter)

        return [
            SearchResult(
                content=r["metadata"].get("content", ""),
                score=r["score"],
                source="vector",
                metadata=r["metadata"],
                node_id=r["id"],
            )
            for r in results
        ]

    def hybrid_search(
        self,
        query: str,
        conversation_memory: ConversationMemory | None = None,
        long_term_memory: LongTermMemory | None = None,
        top_k: int = 10,
        session_id: str = "",
        alpha: float = 0.5,
    ) -> list[SearchResult]:
        all_results: list[SearchResult] = []

        if conversation_memory is not None:
            conv_results = self.search_conversation(conversation_memory, query, top_k)
            all_results.extend(conv_results)

        if long_term_memory is not None:
            lt_results = self.search_long_term(long_term_memory, query, top_k, session_id)
            all_results.extend(lt_results)

        vec_results = self.search_vector(query, top_k)
        all_results.extend(vec_results)

        source_weights = {"conversation": 0.3, "long_term": 0.5, "vector": 0.7}
        for r in all_results:
            sw = source_weights.get(r.source, 0.5)
            r.score = alpha * r.score + (1 - alpha) * sw

        seen = set()
        unique: list[SearchResult] = []
        for r in sorted(all_results, key=lambda x: -x.score):
            dedup_key = r.content[:100]
            if dedup_key not in seen:
                seen.add(dedup_key)
                unique.append(r)

        return unique[:top_k]

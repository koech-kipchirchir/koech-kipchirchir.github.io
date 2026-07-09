from __future__ import annotations

import threading
import uuid
from typing import Any, Optional

from memory.conversation_memory import ConversationMemory, Message, MessageRole
from memory.embeddings import EmbeddingProvider, SentenceTransformerEmbeddings
from memory.long_term_memory import LongTermMemory, MemoryNode
from memory.search import MemorySearch, SearchResult
from memory.storage import SQLiteStorage
from memory.utils import MemoryConfig, setup_logger, timestamp_ms
from memory.vector_memory import ChromaDBStore, FAISSStore, VectorStore

logger = setup_logger("aios.memory.manager")


class MemoryManager:
    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()
        self._logger = setup_logger("aios.memory.manager")
        self._lock = threading.Lock()

        self._storage = SQLiteStorage(self.config.db_path)
        self._embeddings = SentenceTransformerEmbeddings(
            model_name=self.config.embedding_model,
            device=self.config.device,
        )
        self._vector_store = self._init_vector_store()
        self._conversation: ConversationMemory | None = None
        self._long_term = LongTermMemory(
            storage=self._storage,
            max_nodes=self.config.max_memory_nodes,
            importance_threshold=self.config.memory_importance_threshold,
        )
        self._search = MemorySearch(
            embeddings=self._embeddings,
            vector_store=self._vector_store,
        )

        self._logger.info(
            "MemoryManager initialized (embedding=%s, vector=%s, max_turns=%s)",
            self.config.embedding_model,
            "chromadb" if self.config.enable_chromadb else "faiss" if self.config.enable_faiss else "none",
            self.config.max_conversation_turns,
        )

    def _init_vector_store(self) -> VectorStore | None:
        if self.config.enable_chromadb:
            try:
                return ChromaDBStore(
                    collection_name="aios_memories",
                    persist_path=self.config.vector_db_path,
                )
            except Exception as exc:
                self._logger.warning("ChromaDB init failed: %s", exc)

        if self.config.enable_faiss:
            try:
                return FAISSStore(
                    dimension=self.config.embedding_dim,
                    index_path=str(self.config.vector_db_path / "faiss.index"),
                )
            except Exception as exc:
                self._logger.warning("FAISS init failed: %s", exc)

        return None

    def create_session(self, session_id: str | None = None) -> str:
        sid = session_id or uuid.uuid4().hex[:16]
        self._conversation = ConversationMemory(
            storage=self._storage,
            session_id=sid,
            max_turns=self.config.max_conversation_turns,
        )
        self._logger.info("Created session: %s", sid)
        return sid

    def add_message(
        self,
        role: str | MessageRole,
        content: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message | None:
        conv = self._get_or_create_conversation(session_id)
        msg = conv.add_message(role, content, metadata)

        if self._should_store_long_term(msg):
            embedding = self._embeddings.embed(content) if self._vector_store else None
            node = self._long_term.add(
                content=content,
                session_id=conv.session_id,
                embedding=embedding,
                metadata={"message_id": msg.message_id, **(metadata or {})},
            )
            if embedding and self._vector_store:
                try:
                    self._vector_store.add(
                        id=node.node_id,
                        embedding=embedding,
                        metadata={"content": content, "node_id": node.node_id, "session_id": conv.session_id},
                    )
                except Exception as exc:
                    self._logger.warning("Vector store add failed: %s", exc)

        if (
            self.config.auto_summarize
            and conv.turn_count > 0
            and conv.turn_count % self.config.summary_interval_turns == 0
        ):
            self._auto_summarize(conv.session_id)

        return msg

    def search(
        self,
        query: str,
        top_k: int = 10,
        session_id: str = "",
    ) -> list[SearchResult]:
        conv = self._conversation if session_id in ("", self._conversation.session_id if self._conversation else "") else None
        return self._search.hybrid_search(
            query=query,
            conversation_memory=conv,
            long_term_memory=self._long_term,
            top_k=top_k,
            session_id=session_id,
        )

    def get_conversation(self, session_id: str | None = None) -> ConversationMemory | None:
        conv = self._conversation
        if conv and (session_id is None or conv.session_id == session_id):
            return conv
        return None

    def get_long_term_memories(self, session_id: str = "") -> list[MemoryNode]:
        return self._long_term.get_all(session_id)

    def get_recent_messages(self, n: int = 10, session_id: str | None = None) -> list[Message]:
        conv = self._conversation
        if conv and (session_id is None or conv.session_id == session_id):
            return conv.get_recent(n)
        return []

    def summarize(self, session_id: str = "") -> str:
        return self._long_term.summarize(session_id)

    def cleanup(self) -> int:
        deleted = self._long_term.cleanup_expired()
        before = timestamp_ms() - self.config.cleanup_interval_hours * 3600_000
        deleted_messages = self._storage.delete_old_messages(before)
        self._storage.vacuum()
        total = deleted + deleted_messages
        if total:
            self._logger.info("Cleanup removed %s expired items", total)
        return total

    def clear_session(self, session_id: str | None = None) -> None:
        conv = self._conversation
        if conv:
            if session_id is None or conv.session_id == session_id:
                conv.clear()
                self._logger.info("Cleared session: %s", conv.session_id)

    def _get_or_create_conversation(self, session_id: str | None = None) -> ConversationMemory:
        conv = self._conversation
        if conv is None:
            sid = session_id or uuid.uuid4().hex[:16]
            conv = ConversationMemory(
                storage=self._storage,
                session_id=sid,
                max_turns=self.config.max_conversation_turns,
            )
            self._conversation = conv
        elif session_id and conv.session_id != session_id:
            conv = ConversationMemory(
                storage=self._storage,
                session_id=session_id,
                max_turns=self.config.max_conversation_turns,
            )
            self._conversation = conv
        return conv

    def _should_store_long_term(self, msg: Message) -> bool:
        if msg.role == MessageRole.SYSTEM:
            return False
        if len(msg.content.split()) < 3:
            return False
        return True

    def _auto_summarize(self, session_id: str) -> None:
        try:
            summary = self._long_term.summarize(session_id)
            if summary:
                self._logger.info("Auto-summary generated for session %s", session_id)
        except Exception as exc:
            self._logger.warning("Auto-summarize failed: %s", exc)

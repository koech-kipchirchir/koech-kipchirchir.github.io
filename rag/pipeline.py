from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from rag.chunker import Chunker, RecursiveChunker
from rag.document_parser import get_parser
from rag.embeddings import RAGEmbeddings, SentenceTransformerRAGEmbeddings
from rag.loader import Document, DocumentLoader
from rag.retriever import (
    CrossEncoderReranker,
    HybridRetriever,
    KeywordRetriever,
    Reranker,
    Retriever,
    VectorRetriever,
)
from rag.utils import RAGConfig, setup_logger
from rag.vector_store import ChromaDBVectorStore, FAISSVectorStore, VectorStore

logger = setup_logger("aios.rag.pipeline")


class RAGPipeline:
    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()
        self._logger = setup_logger("aios.rag.pipeline")
        self._lock = threading.Lock()

        self._embeddings = SentenceTransformerRAGEmbeddings(
            model_name=self.config.embedding_model,
            device=self.config.device,
        )
        self._vector_store = self._init_vector_store()
        self._chunker: Chunker = RecursiveChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
        )
        self._loader = DocumentLoader(self.config)
        self._documents: list[dict[str, Any]] = []
        self._retriever: Retriever | None = None
        self._reranker: Reranker | None = CrossEncoderReranker()

        self._logger.info(
            "RAGPipeline initialized (embedding=%s, chunk_size=%s, overlap=%s)",
            self.config.embedding_model,
            self.config.chunk_size,
            self.config.chunk_overlap,
        )

    def _init_vector_store(self) -> VectorStore | None:
        if self.config.enable_chromadb:
            try:
                return ChromaDBVectorStore(
                    collection_name="aios_rag",
                    persist_path=self.config.vector_db_path,
                )
            except Exception as exc:
                self._logger.warning("ChromaDB init failed: %s", exc)

        if self.config.enable_faiss:
            try:
                return FAISSVectorStore(
                    dimension=self.config.embedding_dim,
                    index_path=str(Path(self.config.vector_db_path) / "faiss.index"),
                )
            except Exception as exc:
                self._logger.warning("FAISS init failed: %s", exc)

        return None

    def index_file(self, path: str | Path, metadata: dict[str, Any] | None = None) -> int:
        doc = self._loader.load_file(path, metadata)
        return self._index_document(doc)

    def index_text(self, text: str, source: str = "", metadata: dict[str, Any] | None = None) -> int:
        doc = self._loader.load_text(text, source, metadata)
        return self._index_document(doc)

    def index_directory(
        self, path: str | Path, pattern: str = "*", metadata: dict[str, Any] | None = None
    ) -> int:
        docs = self._loader.load_directory(path, pattern, recursive=True, metadata=metadata)
        total = 0
        for doc in docs:
            total += self._index_document(doc)
        return total

    def _index_document(self, doc: Document) -> int:
        chunks = self._chunker.chunk(
            doc.content,
            metadata={"source": doc.source, "doc_id": doc.doc_id, **doc.metadata},
        )

        with self._lock:
            self._documents.extend(chunks)

            if self._vector_store is not None:
                texts = [c["text"] for c in chunks]
                embeddings = self._embeddings.embed_batch(texts)
                ids = [f"{doc.doc_id}_{i}" for i in range(len(chunks))]
                metadatas = [c["metadata"] for c in chunks]
                try:
                    self._vector_store.add_batch(ids, embeddings, metadatas)
                except Exception as exc:
                    self._logger.warning("Vector store add_batch failed: %s", exc)

        self._rebuild_retriever()
        self._logger.info("Indexed %s chunks from %s", len(chunks), doc.source)
        return len(chunks)

    def _rebuild_retriever(self) -> None:
        keyword = KeywordRetriever(self._documents)
        if self._vector_store is not None:
            vector = VectorRetriever(self._vector_store, self._embeddings)
            self._retriever = HybridRetriever(
                vector,
                keyword,
                alpha=self.config.hybrid_search_alpha,
            )
        else:
            self._retriever = keyword

    async def query(
        self,
        query: str,
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        if self._retriever is None:
            self._rebuild_retriever()

        k = top_k or self.config.similarity_top_k
        results = await self._retriever.retrieve(query, k, filter)

        if rerank and self._reranker and len(results) > 1:
            rerank_k = top_k or self.config.rerank_top_k
            results = await self._reranker.rerank(query, results, rerank_k)

        return results

    async def stream_query(self, query: str, top_k: int = 5) -> AsyncIterator[dict[str, Any]]:
        results = await self.query(query, top_k, rerank=False)
        for r in results:
            yield r

    def get_document_count(self) -> int:
        return len(self._documents)

    def clear(self) -> None:
        self._documents.clear()
        self._retriever = None
        self._loader.clear()
        self._logger.info("Pipeline cleared")

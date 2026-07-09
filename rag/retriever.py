from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional

from rag.utils import RAGConfig, setup_logger
from rag.vector_store import VectorStore

logger = setup_logger("aios.rag.retriever")


class Retriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5, filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        ...

    async def stream(self, query: str, top_k: int = 5) -> AsyncIterator[dict[str, Any]]:
        results = await self.retrieve(query, top_k)
        for r in results:
            yield r


class VectorRetriever(Retriever):
    def __init__(self, vector_store: VectorStore, embeddings: Any) -> None:
        self._vector_store = vector_store
        self._embeddings = embeddings
        self._logger = setup_logger("aios.rag.retriever.vector")

    async def retrieve(self, query: str, top_k: int = 5, filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        query_emb = self._embeddings.embed(query)
        results = self._vector_store.search(query_emb, top_k, filter)
        return [
            {"text": r["metadata"].get("text", ""), "score": r["score"], "source": "vector", **r}
            for r in results
        ]


class KeywordRetriever(Retriever):
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self._documents = documents
        self._logger = setup_logger("aios.rag.retriever.keyword")

    async def retrieve(self, query: str, top_k: int = 5, filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        query_terms = set(re.findall(r"\w+", query.lower()))
        if not query_terms:
            return []

        scored: list[dict[str, Any]] = []
        for doc in self._documents:
            text = doc.get("text", "").lower()
            terms_in_doc = set(re.findall(r"\w+", text))
            overlap = len(query_terms & terms_in_doc)
            if overlap > 0:
                score = overlap / math.sqrt(len(query_terms) * max(len(terms_in_doc), 1))
                scored.append({**doc, "score": score})

        scored.sort(key=lambda x: -x["score"])
        return scored[:top_k]

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        self._documents.extend(documents)


class HybridRetriever(Retriever):
    def __init__(
        self,
        vector_retriever: Retriever,
        keyword_retriever: Retriever,
        alpha: float = 0.7,
    ) -> None:
        self._vector = vector_retriever
        self._keyword = keyword_retriever
        self._alpha = alpha
        self._logger = setup_logger("aios.rag.retriever.hybrid")

    async def retrieve(self, query: str, top_k: int = 5, filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        vector_results = await self._vector.retrieve(query, top_k * 2, filter)
        keyword_results = await self._keyword.retrieve(query, top_k * 2, filter)

        combined: dict[str, dict[str, Any]] = {}
        for r in vector_results:
            r["_alpha_score"] = r["score"] * self._alpha
            combined[r.get("id", r.get("text", "")[:50])] = r

        for r in keyword_results:
            key = r.get("id", r.get("text", "")[:50])
            if key in combined:
                combined[key]["score"] = combined[key].get("_alpha_score", 0) + r["score"] * (1 - self._alpha)
                combined[key]["_is_hybrid"] = True
            else:
                r["score"] = r["score"] * (1 - self._alpha)
                combined[key] = r

        results = sorted(combined.values(), key=lambda x: -x["score"])[:top_k]
        for r in results:
            r.pop("_alpha_score", None)
        return results


class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, documents: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
        ...


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._logger = setup_logger("aios.rag.reranker")

    async def rerank(self, query: str, documents: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self._model_name)
            except ImportError:
                self._logger.warning("CrossEncoder unavailable; returning original order")
                return documents[:top_k]

        pairs = [(query, doc.get("text", "")) for doc in documents]
        scores = self._model.predict(pairs)
        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: -x[1])
        return [{**doc, "score": float(score), "rerank_score": float(score)} for doc, score in scored[:top_k]]

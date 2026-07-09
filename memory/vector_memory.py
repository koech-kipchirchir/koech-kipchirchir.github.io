from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np

from memory.utils import setup_logger

logger = setup_logger("aios.memory.vector")


class VectorStore(ABC):
    @abstractmethod
    def add(self, id: str, embedding: list[float], metadata: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def add_batch(self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict[str, Any]]) -> None:
        ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5, filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class FAISSStore(VectorStore):
    def __init__(self, dimension: int, index_path: str | Path | None = None) -> None:
        self._dim = dimension
        self._index_path = Path(index_path) if index_path else None
        self._index: Any = None
        self._id_map: dict[str, int] = {}
        self._reverse_map: dict[int, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        self._logger = setup_logger("aios.memory.vector.faiss")
        self._init_index()

    def _init_index(self) -> None:
        try:
            import faiss

            if self._index_path and self._index_path.exists():
                self._index = faiss.read_index(str(self._index_path))
                self._logger.info("Loaded FAISS index from %s", self._index_path)
                meta_path = self._index_path.with_suffix(".json")
                if meta_path.exists():
                    data = json.loads(meta_path.read_text(encoding="utf-8"))
                    self._id_map = data.get("id_map", {})
                    self._reverse_map = {v: k for k, v in self._id_map.items()}
                    self._metadata = data.get("metadata", {})
                    self._next_id = data.get("next_id", max(self._id_map.values(), default=-1) + 1)
            else:
                import faiss

                self._index = faiss.IndexFlatIP(self._dim)
                self._logger.info("Created FAISS index (dim=%s)", self._dim)
        except ImportError:
            self._logger.warning("faiss not installed; using in-memory fallback")
            self._index = None

    def add(self, id: str, embedding: list[float], metadata: dict[str, Any]) -> None:
        self.add_batch([id], [embedding], [metadata])

    def add_batch(self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict[str, Any]]) -> None:
        with self._lock:
            arr = np.array(embeddings, dtype=np.float32)
            if self._index is not None:
                self._index.add(arr)
            for i, fid in enumerate(ids):
                idx = self._next_id
                self._id_map[fid] = idx
                self._reverse_map[idx] = fid
                self._metadata[fid] = metadatas[i] if i < len(metadatas) else {}
                self._next_id += 1
            self._save()

    def search(self, query_embedding: list[float], top_k: int = 5, filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            import faiss

            query = np.array([query_embedding], dtype=np.float32)
            scores, indices = self._index.search(query, top_k)
            results: list[dict[str, Any]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    break
                fid = self._reverse_map.get(int(idx))
                if fid is None:
                    continue
                meta = self._metadata.get(fid, {})
                if filter and not self._matches_filter(meta, filter):
                    continue
                results.append({"id": fid, "score": float(score), "metadata": meta})
            return results

    def delete(self, ids: list[str]) -> None:
        with self._lock:
            for fid in ids:
                idx = self._id_map.pop(fid, None)
                if idx is not None:
                    self._reverse_map.pop(idx, None)
                    self._metadata.pop(fid, None)
            self._save()

    def count(self) -> int:
        with self._lock:
            return self._index.ntotal if self._index is not None else len(self._id_map)

    def close(self) -> None:
        self._save()

    def _save(self) -> None:
        if self._index_path is None:
            return
        try:
            import faiss

            faiss.write_index(self._index, str(self._index_path))
            meta_path = self._index_path.with_suffix(".json")
            meta_path.write_text(
                json.dumps({
                    "id_map": self._id_map,
                    "metadata": self._metadata,
                    "next_id": self._next_id,
                }),
                encoding="utf-8",
            )
        except Exception as exc:
            self._logger.warning("Failed to save FAISS index: %s", exc)

    @staticmethod
    def _matches_filter(metadata: dict[str, Any], filter: dict[str, Any]) -> bool:
        for key, value in filter.items():
            if key not in metadata:
                return False
            if isinstance(value, (list, tuple)):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True


class ChromaDBStore(VectorStore):
    def __init__(self, collection_name: str, persist_path: str | Path | None = None) -> None:
        self._collection_name = collection_name
        self._persist_path = Path(persist_path) if persist_path else None
        self._client: Any = None
        self._collection: Any = None
        self._lock = threading.Lock()
        self._logger = setup_logger("aios.memory.vector.chromadb")
        self._init_collection()

    def _init_collection(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings

            kwargs: dict[str, Any] = {"settings": Settings(anonymized_telemetry=False)}
            if self._persist_path:
                self._persist_path.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(self._persist_path), **kwargs)
            else:
                self._client = chromadb.Client(**kwargs)

            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._logger.info("Connected to ChromaDB collection: %s", self._collection_name)
        except ImportError:
            self._logger.warning("chromadb not installed; ChromaDB unavailable")

    def add(self, id: str, embedding: list[float], metadata: dict[str, Any]) -> None:
        self.add_batch([id], [embedding], [metadata])

    def add_batch(self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict[str, Any]]) -> None:
        if self._collection is None:
            return
        with self._lock:
            self._collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def search(self, query_embedding: list[float], top_k: int = 5, filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if self._collection is None:
            return []
        with self._lock:
            kwargs: dict[str, Any] = {"query_embeddings": [query_embedding], "n_results": top_k}
            if filter:
                kwargs["where"] = filter
            results = self._collection.query(**kwargs)
            output: list[dict[str, Any]] = []
            for i in range(len(results["ids"][0])):
                output.append({
                    "id": results["ids"][0][i],
                    "score": results["distances"][0][i] if results.get("distances") else 0.0,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                })
            return output

    def delete(self, ids: list[str]) -> None:
        if self._collection is None:
            return
        with self._lock:
            self._collection.delete(ids=ids)

    def count(self) -> int:
        return self._collection.count() if self._collection else 0

    def close(self) -> None:
        pass

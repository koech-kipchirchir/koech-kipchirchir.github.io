from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from memory.utils import setup_logger

logger = setup_logger("aios.memory.embeddings")


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class SentenceTransformerEmbeddings(EmbeddingProvider):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_size: int = 1024,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._dim: int = 0
        self._lock = threading.Lock()
        self._cache: dict[str, list[float]] = {}
        self._cache_size = cache_size
        self._logger = setup_logger("aios.memory.embeddings.sentence")

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
            self._dim = self._model.get_sentence_embedding_dimension() or 384
            self._logger.info("Loaded model: %s (dim=%s, device=%s)", self._model_name, self._dim, self._device)
        except ImportError:
            self._logger.warning("sentence-transformers not installed; using fallback embeddings")
            self._dim = 384

    def embed(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        self._lazy_load()
        if self._model is None:
            vec = self._fallback_embed(text)
        else:
            vec = self._model.encode(text, normalize_embeddings=True).tolist()
        self._cache[text] = vec
        if len(self._cache) > self._cache_size:
            self._cache.pop(next(iter(self._cache)), None)
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._lazy_load()
        if self._model is None:
            return [self._fallback_embed(t) for t in texts]
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        self._lazy_load()
        return self._dim

    def _fallback_embed(self, text: str) -> list[float]:
        rng = np.random.RandomState(hash(text) & 0xFFFFFFFF)
        vec = rng.randn(self._dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

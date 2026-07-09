from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from rag.utils import setup_logger

logger = setup_logger("aios.rag.chunker")


class ChunkingStrategy(Enum):
    RECURSIVE = "recursive"
    CHARACTER = "character"
    TOKEN = "token"
    SEMANTIC = "semantic"


class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        ...


class RecursiveChunker(Chunker):
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or ["\n\n", "\n", ".", " ", ""]
        self._logger = setup_logger("aios.rag.chunker.recursive")

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not text:
            return []
        meta = metadata or {}
        chunks = self._split_text(text)
        return [
            {
                "text": chunk,
                "metadata": {
                    **meta,
                    "chunk_index": i,
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks),
                },
            }
            for i, chunk in enumerate(chunks)
        ]

    def _split_text(self, text: str) -> list[str]:
        chunks: list[str] = []
        current = text

        while current:
            if len(current) <= self._chunk_size:
                chunks.append(current)
                break

            split_point = self._find_split(current)
            if split_point == 0 or split_point >= self._chunk_size:
                split_point = self._chunk_size

            chunks.append(current[:split_point].strip())
            overlap_start = max(0, split_point - self._chunk_overlap)
            current = current[overlap_start:]

        return [c for c in chunks if c]

    def _find_split(self, text: str) -> int:
        for sep in self._separators:
            pos = text.rfind(sep, 0, self._chunk_size)
            if pos > 0:
                return pos + len(sep)
        return self._chunk_size


class CharacterChunker(Chunker):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not text:
            return []
        meta = metadata or {}
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size
            chunks.append(text[start:end])
            start += self._chunk_size - self._chunk_overlap
        return [
            {
                "text": chunk,
                "metadata": {**meta, "chunk_index": i, "total_chunks": len(chunks)},
            }
            for i, chunk in enumerate(chunks)
        ]


class TokenChunker(Chunker):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not text:
            return []
        meta = metadata or {}
        words = text.split()
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = start + self._chunk_size
            chunks.append(" ".join(words[start:end]))
            start += self._chunk_size - self._chunk_overlap
        return [
            {
                "text": chunk,
                "metadata": {**meta, "chunk_index": i, "total_chunks": len(chunks)},
            }
            for i, chunk in enumerate(chunks)
        ]


class SemanticChunker(Chunker):
    def __init__(self, max_chunk_size: int = 512) -> None:
        self._max_chunk_size = max_chunk_size

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not text:
            return []
        meta = metadata or {}
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                chunks.append(current.strip())
                current = ""
            elif len(current) + len(para) <= self._max_chunk_size:
                current += "\n\n" + para if current else para
            else:
                if current:
                    chunks.append(current.strip())
                current = para
        if current:
            chunks.append(current.strip())
        return [
            {
                "text": chunk,
                "metadata": {**meta, "chunk_index": i, "total_chunks": len(chunks)},
            }
            for i, chunk in enumerate(chunks) if chunk
        ]


_CHUNKER_REGISTRY: dict[str, type[Chunker]] = {
    "recursive": RecursiveChunker,
    "character": CharacterChunker,
    "token": TokenChunker,
    "semantic": SemanticChunker,
}


def get_chunker(strategy: str, **kwargs: Any) -> Chunker:
    cls = _CHUNKER_REGISTRY.get(strategy)
    if cls is None:
        logger.warning("Unknown strategy: %s, using recursive", strategy)
        return RecursiveChunker(**kwargs)
    return cls(**kwargs)

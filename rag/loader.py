from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from rag.chunker import Chunker, get_chunker
from rag.document_parser import get_parser
from rag.utils import RAGConfig, setup_logger

logger = setup_logger("aios.rag.loader")


@dataclass
class Document:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    doc_id: str = ""

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = hashlib.md5(f"{self.source}{self.content[:100]}".encode()).hexdigest()[:16]


class DocumentLoader:
    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()
        self._logger = setup_logger("aios.rag.loader")
        self._loaded: dict[str, Document] = {}

    def load_file(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Document:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        parser = get_parser(ext)
        if parser is None:
            raise ValueError(f"Unsupported file type: {ext}")

        content_bytes = path.read_bytes()
        text = parser.parse(content_bytes, metadata)

        doc_meta = {
            "source": str(path.resolve()),
            "filename": path.name,
            "extension": ext,
            "size": path.stat().st_size,
            **(metadata or {}),
        }

        doc = Document(content=text, metadata=doc_meta, source=str(path))
        self._loaded[doc.doc_id] = doc
        self._logger.info("Loaded: %s (%s chars)", path.name, len(text))
        return doc

    def load_text(self, text: str, source: str = "", metadata: dict[str, Any] | None = None) -> Document:
        doc = Document(
            content=text,
            metadata={"source": source or "text", "extension": ".txt", **(metadata or {})},
            source=source or "text",
        )
        self._loaded[doc.doc_id] = doc
        return doc

    def load_directory(
        self,
        path: str | Path,
        pattern: str = "*",
        recursive: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        path = Path(path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        glob_pattern = "**/*" if recursive else "*"
        documents: list[Document] = []

        for file_path in sorted(path.glob(glob_pattern)):
            if not file_path.is_file():
                continue
            if pattern != "*" and not file_path.match(pattern):
                continue

            try:
                doc = self.load_file(file_path, metadata)
                documents.append(doc)
            except (ValueError, Exception) as exc:
                self._logger.warning("Skipping %s: %s", file_path.name, exc)

        self._logger.info("Loaded %s documents from %s", len(documents), path)
        return documents

    async def load_stream(self, path: str | Path, chunk_size: int = 8192) -> AsyncIterator[str]:
        path = Path(path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    def get_document(self, doc_id: str) -> Document | None:
        return self._loaded.get(doc_id)

    def clear(self) -> None:
        self._loaded.clear()

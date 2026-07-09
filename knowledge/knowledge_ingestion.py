"""
Document ingestion pipeline with multi-format parsing, chunking,
and entity extraction integration.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from knowledge.entity_extractor import EntityExtractor, ExtractionResult, RegexEntityExtractor
from knowledge.fact_store import Fact, FactStore, FactStatus
from knowledge.knowledge_graph import GraphEdge, GraphNode, KnowledgeGraph
from knowledge.utils import structured_log

logger = logging.getLogger("aios.knowledge.ingestion")


@dataclass
class Document:
    """A parsed document ready for ingestion."""

    id: str = ""
    title: str = ""
    content: str = ""
    content_type: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    chunks: list[DocumentChunk] = field(default_factory=list)
    created_at: str = ""


@dataclass
class DocumentChunk:
    """A chunk of a document with position tracking."""

    id: str = ""
    document_id: str = ""
    content: str = ""
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionConfig:
    """Configuration for the ingestion pipeline."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    extract_entities: bool = True
    extract_relations: bool = True
    store_facts: bool = True
    min_confidence: float = 0.3
    max_fact_length: int = 200
    supported_extensions: set[str] = field(default_factory=lambda: {
        ".txt", ".md", ".html", ".htm", ".csv", ".json", ".xml", ".yaml", ".yml",
        ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".rs", ".go", ".kt",
        ".sql", ".sh", ".bat", ".ps1",
    })


class DocumentParser(ABC):
    """Abstract document parser."""

    @abstractmethod
    async def parse(self, content: str | bytes, source: str = "", **kwargs: Any) -> Document:
        pass


class TextParser(DocumentParser):
    """Plain text parser."""

    async def parse(self, content: str | bytes, source: str = "", **kwargs: Any) -> Document:
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        return Document(
            id=_generate_id(),
            title=Path(source).stem if source else "Untitled",
            content=content,
            content_type="text",
            source=source,
            created_at=_now(),
        )


class MarkdownParser(DocumentParser):
    """Markdown parser that strips formatting."""

    async def parse(self, content: str | bytes, source: str = "", **kwargs: Any) -> Document:
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        # Strip markdown headings, bold, italic, code, links, images
        plain = re.sub(r"#{1,6}\s+", "", content)
        plain = re.sub(r"\*\*(.+?)\*\*", r"\1", plain)
        plain = re.sub(r"\*(.+?)\*", r"\1", plain)
        plain = re.sub(r"`{1,3}[^`]*`{1,3}", "", plain)
        plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", plain)
        plain = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", plain)
        plain = re.sub(r"^[-*+]\s+", "", plain, flags=re.MULTILINE)
        plain = re.sub(r"^\d+\.\s+", "", plain, flags=re.MULTILINE)
        plain = re.sub(r"^>\s+", "", plain, flags=re.MULTILINE)
        plain = re.sub(r"\|", " ", plain)
        lines = [l.strip() for l in plain.split("\n") if l.strip()]
        title = lines[0] if lines else Path(source).stem if source else "Untitled"
        return Document(
            id=_generate_id(),
            title=title,
            content="\n".join(lines),
            content_type="markdown",
            source=source,
            created_at=_now(),
        )


class HTMLParser(DocumentParser):
    """Basic HTML parser that strips tags."""

    async def parse(self, content: str | bytes, source: str = "", **kwargs: Any) -> Document:
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        plain = re.sub(r"<[^>]+>", "", content)
        plain = re.sub(r"&[a-zA-Z]+;", " ", plain)
        plain = re.sub(r"\s+", " ", plain).strip()
        title_match = re.search(r"<title[^>]*>(.+?)</title>", content, re.IGNORECASE)
        title = title_match.group(1) if title_match else Path(source).stem if source else "Untitled"
        return Document(
            id=_generate_id(),
            title=title,
            content=plain,
            content_type="html",
            source=source,
            created_at=_now(),
        )


class CSVParser(DocumentParser):
    """CSV parser that converts to a textual representation."""

    async def parse(self, content: str | bytes, source: str = "", **kwargs: Any) -> Document:
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        lines = [f"CSV File: {Path(source).name if source else 'data.csv'}"]
        if rows:
            headers = list(rows[0].keys())
            lines.append("Columns: " + ", ".join(headers))
            # Summarize up to 50 rows
            for i, row in enumerate(rows[:50]):
                parts = [f"{k}: {v}" for k, v in row.items() if v]
                lines.append(f"Row {i + 1}: " + "; ".join(parts))
        text = "\n".join(lines)
        return Document(
            id=_generate_id(),
            title=Path(source).stem if source else "Untitled",
            content=text,
            content_type="csv",
            metadata={"row_count": len(rows), "columns": list(rows[0].keys()) if rows else []},
            source=source,
            created_at=_now(),
        )


class JSONParser(DocumentParser):
    """JSON parser that flattens to readable text."""

    async def parse(self, content: str | bytes, source: str = "", **kwargs: Any) -> Document:
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        data = json.loads(content)
        text = json.dumps(data, indent=2, default=str)
        return Document(
            id=_generate_id(),
            title=Path(source).stem if source else "Untitled",
            content=text,
            content_type="json",
            metadata={"keys": list(data.keys()) if isinstance(data, dict) else [],
                      "type": type(data).__name__},
            source=source,
            created_at=_now(),
        )


class SQLParser(DocumentParser):
    """SQL script parser (basic keyword extraction)."""

    KEYWORDS = {"SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE",
                "CREATE", "ALTER", "DROP", "TABLE", "INDEX", "JOIN",
                "LEFT", "RIGHT", "INNER", "OUTER", "GROUP", "ORDER",
                "BY", "HAVING", "LIMIT", "OFFSET", "INTO", "VALUES",
                "SET", "AND", "OR", "NOT", "NULL", "AS", "ON", "USING"}

    async def parse(self, content: str | bytes, source: str = "", **kwargs: Any) -> Document:
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        tokens = re.findall(r"\b\w+\b", content)
        found_keywords = sorted(set(t for t in tokens if t.upper() in self.KEYWORDS))
        tables = re.findall(r"(?:FROM|INTO|TABLE|UPDATE|JOIN)\s+(\w+)", content, re.IGNORECASE)
        text = f"SQL File: {Path(source).name if source else 'script.sql'}\n"
        text += f"Keywords: {', '.join(found_keywords)}\n"
        text += f"Tables: {', '.join(set(tables))}\n"
        text += f"\nContent:\n{content}"
        return Document(
            id=_generate_id(),
            title=Path(source).stem if source else "Untitled",
            content=text,
            content_type="sql",
            metadata={"keywords": found_keywords, "tables": list(set(tables))},
            source=source,
            created_at=_now(),
        )


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------


class KnowledgeIngestion:
    """Orchestrates document parsing, chunking, entity extraction,
    and fact/knowledge-graph storage."""

    def __init__(
        self,
        config: IngestionConfig | None = None,
        entity_extractor: EntityExtractor | None = None,
        fact_store: FactStore | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
    ) -> None:
        self._config = config or IngestionConfig()
        self._entity_extractor = entity_extractor or RegexEntityExtractor()
        self._fact_store = fact_store
        self._knowledge_graph = knowledge_graph
        # Build parser registry
        self._parsers: dict[str, DocumentParser] = {
            ".txt": TextParser(),
            ".md": MarkdownParser(),
            ".markdown": MarkdownParser(),
            ".html": HTMLParser(),
            ".htm": HTMLParser(),
            ".csv": CSVParser(),
            ".json": JSONParser(),
            ".sql": SQLParser(),
        }

    @property
    def parsers(self) -> dict[str, DocumentParser]:
        return dict(self._parsers)

    def register_parser(self, extension: str, parser: DocumentParser) -> None:
        ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        self._parsers[ext] = parser

    def get_parser_for(self, filename: str) -> DocumentParser | None:
        ext = Path(filename).suffix.lower()
        return self._parsers.get(ext)

    async def ingest_file(self, filepath: str) -> Document | None:
        path = Path(filepath)
        if not path.exists():
            logger.warning("File not found: %s", filepath)
            return None
        ext = path.suffix.lower()
        parser = self._parsers.get(ext)
        if not parser and ext in self._config.supported_extensions:
            parser = TextParser()
        if not parser:
            logger.warning("No parser for extension: %s", ext)
            return None
        content: str | bytes
        try:
            if ext in (".json", ".csv", ".html", ".htm", ".xml"):
                content = path.read_bytes()
            else:
                content = path.read_text("utf-8", errors="replace")
        except Exception as e:
            logger.error("Error reading file %s: %s", filepath, e)
            return None

        doc = await parser.parse(content, source=str(path))
        return await self._process_document(doc, filepath)

    async def ingest_text(
        self,
        text: str,
        title: str = "Untitled",
        source: str = "",
        content_type: str = "text",
    ) -> Document:
        doc = Document(
            id=_generate_id(),
            title=title,
            content=text,
            content_type=content_type,
            source=source,
            created_at=_now(),
        )
        return await self._process_document(doc, source)

    async def _process_document(self, doc: Document, source: str) -> Document:
        doc.chunks = self._chunk_document(doc)
        structured_log(logging.DEBUG, "ingestion.chunked",
                       document_id=doc.id,
                       chunks=len(doc.chunks),
                       content_length=len(doc.content))

        if self._config.extract_entities and self._entity_extractor:
            extraction = await self._entity_extractor.extract(doc.content, doc.id)
            doc.metadata["extraction"] = {
                "entity_count": extraction.entity_count,
                "relation_count": extraction.relation_count,
                "duration_ms": round(extraction.duration_ms, 1),
            }

            if self._config.store_facts:
                await self._store_facts(doc, extraction)
                await self._store_graph(doc, extraction)

        return doc

    def _chunk_document(self, doc: Document) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        text = doc.content
        cs = self._config.chunk_size
        overlap = self._config.chunk_overlap

        if len(text) <= cs:
            chunks.append(DocumentChunk(
                id=_generate_id(),
                document_id=doc.id,
                content=text,
                chunk_index=0,
                start_char=0,
                end_char=len(text),
            ))
            return chunks

        start = 0
        idx = 0
        while start < len(text):
            end = min(start + cs, len(text))
            chunk_text = text[start:end]
            if end < len(text):
                # Try to break at a sentence boundary
                last_period = chunk_text.rfind(".")
                if last_period > cs // 2:
                    end = start + last_period + 1
                    chunk_text = text[start:end]
            chunks.append(DocumentChunk(
                id=_generate_id(),
                document_id=doc.id,
                content=chunk_text,
                chunk_index=idx,
                start_char=start,
                end_char=end,
            ))
            idx += 1
            start = end - overlap if end < len(text) else end

        return chunks

    async def _store_facts(self, doc: Document, extraction: ExtractionResult) -> None:
        if not self._fact_store:
            return
        for ee in extraction.entities:
            if ee.entity.confidence < self._config.min_confidence:
                continue
            fact = Fact(
                subject=doc.id,
                predicate="contains_entity",
                object=ee.entity.name,
                confidence=ee.entity.confidence,
                status=FactStatus.CONFIRMED if ee.entity.confidence > 0.6 else FactStatus.PROPOSED,
                source=doc.source,
                provenance={
                    "document_id": doc.id,
                    "entity_type": ee.entity.type.value,
                    "position": (ee.start, ee.end),
                },
            )
            self._fact_store.add_fact(fact)

        for er in extraction.relations:
            rel = er.relation
            if rel.confidence < self._config.min_confidence:
                continue
            fact = Fact(
                subject=rel.source_id,
                predicate=rel.type,
                object=rel.target_id,
                confidence=rel.confidence,
                status=FactStatus.CONFIRMED if rel.confidence > 0.6 else FactStatus.PROPOSED,
                source=doc.source,
                provenance={
                    "document_id": doc.id,
                    "relation_text": er.text,
                },
            )
            self._fact_store.add_fact(fact)

    async def _store_graph(self, doc: Document, extraction: ExtractionResult) -> None:
        if not self._knowledge_graph:
            return
        for ee in extraction.entities:
            if ee.entity.confidence < self._config.min_confidence:
                continue
            if not self._knowledge_graph.has_node(ee.entity.id):
                node = GraphNode(
                    id=ee.entity.id,
                    name=ee.entity.name,
                    type=ee.entity.type.value,
                    properties={
                        "confidence": ee.entity.confidence,
                        "source_document": doc.id,
                    },
                    created_at=_now(),
                )
                self._knowledge_graph.add_node(node)

        for er in extraction.relations:
            rel = er.relation
            if rel.confidence < self._config.min_confidence:
                continue
            try:
                edge = GraphEdge(
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    type=rel.type,
                    weight=rel.confidence,
                    properties={"source_document": doc.id},
                )
                self._knowledge_graph.add_edge(edge)
            except (KeyError, ValueError):
                pass


def _generate_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

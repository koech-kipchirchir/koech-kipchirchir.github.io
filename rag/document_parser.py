from __future__ import annotations

import csv
import io
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from rag.utils import setup_logger

logger = setup_logger("aios.rag.document_parser")


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, content: bytes | str, metadata: dict[str, Any] | None = None) -> str:
        ...

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        ...


class TXTParser(DocumentParser):
    def parse(self, content: bytes | str, metadata: dict[str, Any] | None = None) -> str:
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="replace")
        return content

    def supported_extensions(self) -> list[str]:
        return [".txt"]


class MarkdownParser(DocumentParser):
    def parse(self, content: bytes | str, metadata: dict[str, Any] | None = None) -> str:
        text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"#{1,6}\s+", "", text)
        text = re.sub(r"[*_~]{1,3}", "", text)
        text = re.sub(r"```.*?\n", "\n", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        return text.strip()

    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown"]


class JSONParser(DocumentParser):
    def parse(self, content: bytes | str, metadata: dict[str, Any] | None = None) -> str:
        text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
        try:
            data = json.loads(text)
            return self._flatten(data)
        except json.JSONDecodeError:
            return text

    def _flatten(self, data: Any, prefix: str = "") -> str:
        if isinstance(data, dict):
            parts = []
            for key, value in data.items():
                flattened = self._flatten(value, f"{prefix}{key}: ")
                if flattened:
                    parts.append(flattened)
            return "\n".join(parts)
        if isinstance(data, list):
            return "\n".join(f"- {self._flatten(item, prefix)}" for item in data if item)
        if isinstance(data, (str, int, float, bool)):
            return f"{prefix}{data}"
        return ""

    def supported_extensions(self) -> list[str]:
        return [".json"]


class CSVParser(DocumentParser):
    def parse(self, content: bytes | str, metadata: dict[str, Any] | None = None) -> str:
        text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
        reader = csv.DictReader(io.StringIO(text))
        rows = []
        for row in reader:
            rows.append(" | ".join(f"{k}: {v}" for k, v in row.items() if v))
        return "\n".join(rows)

    def supported_extensions(self) -> list[str]:
        return [".csv"]


class HTMLParser(DocumentParser):
    def parse(self, content: bytes | str, metadata: dict[str, Any] | None = None) -> str:
        text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def supported_extensions(self) -> list[str]:
        return [".html", ".htm"]


class PDFParser(DocumentParser):
    def parse(self, content: bytes | str, metadata: dict[str, Any] | None = None) -> str:
        if isinstance(content, str):
            return content
        try:
            import pdfminer.high_level

            text = pdfminer.high_level.extract_text(io.BytesIO(content))
            return text.strip()
        except ImportError:
            try:
                import PyPDF2

                reader = PyPDF2.PdfReader(io.BytesIO(content))
                pages = []
                for page in reader.pages:
                    pages.append(page.extract_text())
                return "\n\n".join(pages)
            except ImportError:
                logger.warning("No PDF library available. Install pdfminer.six or PyPDF2")
                return "[PDF content unavailable - install a PDF parser]"

    def supported_extensions(self) -> list[str]:
        return [".pdf"]


class DOCXParser(DocumentParser):
    def parse(self, content: bytes | str, metadata: dict[str, Any] | None = None) -> str:
        if isinstance(content, str):
            return content
        try:
            import docx

            doc = docx.Document(io.BytesIO(content))
            paragraphs = [p.text for p in doc.paragraphs]
            return "\n".join(paragraphs)
        except ImportError:
            logger.warning("python-docx not installed")
            return "[DOCX content unavailable - install python-docx]"

    def supported_extensions(self) -> list[str]:
        return [".docx"]


_PARSER_REGISTRY: dict[str, DocumentParser] = {
    ".txt": TXTParser(),
    ".md": MarkdownParser(),
    ".markdown": MarkdownParser(),
    ".json": JSONParser(),
    ".csv": CSVParser(),
    ".html": HTMLParser(),
    ".htm": HTMLParser(),
    ".pdf": PDFParser(),
    ".docx": DOCXParser(),
}


def get_parser(extension: str) -> DocumentParser | None:
    return _PARSER_REGISTRY.get(extension.lower())

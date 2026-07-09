"""
AIOS Knowledge System
=====================

Production-grade knowledge management with entity extraction,
knowledge graphs, semantic/hybrid search, document ingestion,
fact validation, reasoning, versioning, and metadata tracking.

Supported document formats: TXT, Markdown, HTML, CSV, JSON, SQL, and code files.
"""

from __future__ import annotations

from knowledge.entity_extractor import (
    Entity, EntityType, Relation, ExtractedEntity,
    ExtractedRelation, ExtractionResult, EntityExtractor,
    RegexEntityExtractor,
)
from knowledge.fact_store import (
    Fact, FactVersion, FactStore, FactStatus, FactQuery, FactVersion,
)
from knowledge.knowledge_base import KnowledgeBase, KnowledgeBaseConfig
from knowledge.knowledge_graph import (
    GraphNode, GraphEdge, KnowledgeGraph, QueryPath,
    NodeNotFoundError, EdgeNotFoundError,
)
from knowledge.knowledge_ingestion import (
    Document, DocumentChunk, IngestionConfig,
    KnowledgeIngestion, DocumentParser, TextParser,
    MarkdownParser, JSONParser, CSVParser, HTMLParser, SQLParser,
)
from knowledge.knowledge_search import (
    SearchQuery, SearchResult, SearchResponse, TextIndex,
    KnowledgeSearch,
)
from knowledge.reasoning import (
    Rule, InferenceResult, RuleBasedEngine,
    RuleType,
)
from knowledge.validator import (
    ValidationIssue, ValidationReport, ValidationSeverity,
    ValidationRule, KnowledgeValidator,
)

__all__ = [
    "CSVParser",
    "Document",
    "DocumentChunk",
    "DocumentParser",
    "EdgeNotFoundError",
    "Entity",
    "EntityExtractor",
    "EntityType",
    "ExtractedEntity",
    "ExtractedRelation",
    "ExtractionResult",
    "Fact",
    "FactQuery",
    "FactStatus",
    "FactStore",
    "FactVersion",
    "GraphEdge",
    "GraphNode",
    "HTMLParser",
    "InferenceResult",
    "IngestionConfig",
    "JSONParser",
    "KnowledgeBase",
    "KnowledgeBaseConfig",
    "KnowledgeGraph",
    "KnowledgeIngestion",
    "KnowledgeSearch",
    "KnowledgeValidator",
    "MarkdownParser",
    "NodeNotFoundError",
    "QueryPath",
    "RegexEntityExtractor",
    "Relation",
    "Rule",
    "RuleBasedEngine",
    "RuleType",
    "SQLParser",
    "SearchQuery",
    "SearchResult",
    "SearchResponse",
    "TextIndex",
    "TextParser",
    "ValidationIssue",
    "ValidationReport",
    "ValidationRule",
    "ValidationSeverity",
]

__version__ = "0.1.0"

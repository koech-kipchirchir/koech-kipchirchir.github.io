"""
KnowledgeBase orchestrator that wires together entity extraction,
knowledge graph, fact store, ingestion, search, validation, and
reasoning into a unified knowledge management system.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from knowledge.entity_extractor import (
    Entity,
    EntityExtractor,
    ExtractionResult,
    RegexEntityExtractor,
)
from knowledge.fact_store import (
    Fact,
    FactQuery,
    FactStatus,
    FactStore,
)
from knowledge.knowledge_graph import (
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    NodeNotFoundError,
)
from knowledge.knowledge_ingestion import (
    Document,
    IngestionConfig,
    KnowledgeIngestion,
)
from knowledge.knowledge_search import (
    KnowledgeSearch,
    SearchQuery,
    SearchResponse,
)
from knowledge.reasoning import (
    InferenceResult,
    Rule,
    RuleBasedEngine,
    RuleType,
)
from knowledge.utils import KnowledgeConfig, structured_log
from knowledge.validator import (
    KnowledgeValidator,
    ValidationReport,
    ValidationSeverity,
)

logger = logging.getLogger("aios.knowledge.base")


@dataclass
class KnowledgeBaseConfig:
    """Top-level configuration for the knowledge base."""

    knowledge_config: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    auto_extract_entities: bool = True
    auto_validate: bool = True
    auto_reason: bool = False
    export_path: str = ""
    max_import_size_mb: int = 50


class KnowledgeBase:
    """Unified knowledge management system.

    Provides a single entry point for:
    - Entity extraction (RegexEntityExtractor)
    - Knowledge graph storage and traversal (KnowledgeGraph)
    - Fact storage with versioning (FactStore)
    - Document ingestion (KnowledgeIngestion)
    - Hybrid search (KnowledgeSearch)
    - Fact validation (KnowledgeValidator)
    - Rule-based reasoning (RuleBasedEngine)
    """

    def __init__(self, config: KnowledgeBaseConfig | None = None) -> None:
        self._config = config or KnowledgeBaseConfig()
        self._graph = KnowledgeGraph()
        self._fact_store = FactStore()
        self._entity_extractor: EntityExtractor = RegexEntityExtractor()
        self._search = KnowledgeSearch(
            knowledge_graph=self._graph,
            fact_store=self._fact_store,
        )
        self._validator = KnowledgeValidator()
        self._reasoning = RuleBasedEngine()
        self._ingestion = KnowledgeIngestion(
            config=self._config.ingestion,
            entity_extractor=self._entity_extractor,
            fact_store=self._fact_store,
            knowledge_graph=self._graph,
        )
        self._initialized = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def graph(self) -> KnowledgeGraph:
        return self._graph

    @property
    def fact_store(self) -> FactStore:
        return self._fact_store

    @property
    def search(self) -> KnowledgeSearch:
        return self._search

    @property
    def validator(self) -> KnowledgeValidator:
        return self._validator

    @property
    def reasoning(self) -> RuleBasedEngine:
        return self._reasoning

    @property
    def ingestion(self) -> KnowledgeIngestion:
        return self._ingestion

    @property
    def entity_extractor(self) -> EntityExtractor:
        return self._entity_extractor

    @property
    def config(self) -> KnowledgeBaseConfig:
        return self._config

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    async def extract_entities(self, text: str, document_id: str = "") -> ExtractionResult:
        return await self._entity_extractor.extract(text, document_id=document_id)

    # ------------------------------------------------------------------
    # Graph operations
    # ------------------------------------------------------------------

    def add_node(self, node: GraphNode) -> str:
        return self._graph.add_node(node)

    def get_node(self, node_id: str) -> GraphNode:
        return self._graph.get_node(node_id)

    def update_node(self, node_id: str, **updates: Any) -> GraphNode:
        return self._graph.update_node(node_id, **updates)

    def delete_node(self, node_id: str) -> bool:
        return self._graph.delete_node(node_id)

    def find_nodes(self, name: str | None = None, type_filter: str | None = None) -> list[GraphNode]:
        return self._graph.find_nodes(name=name, type_filter=type_filter)

    def add_edge(self, edge: GraphEdge) -> str:
        return self._graph.add_edge(edge)

    def get_neighbors(
        self, node_id: str, edge_type: str | None = None, direction: str = "outgoing"
    ) -> list[tuple[GraphEdge, GraphNode]]:
        return self._graph.get_neighbors(node_id, edge_type=edge_type, direction=direction)

    def find_path(self, source_id: str, target_id: str, max_depth: int = 5) -> list:
        return self._graph.find_path(source_id, target_id, max_depth=max_depth)

    def traverse(self, start_id: str, max_depth: int = 3, edge_type: str | None = None) -> list[GraphNode]:
        return self._graph.traverse(start_id, max_depth=max_depth, edge_type=edge_type)

    # ------------------------------------------------------------------
    # Fact store operations
    # ------------------------------------------------------------------

    def add_fact(self, fact: Fact) -> str:
        return self._fact_store.add_fact(fact)

    def get_fact(self, fact_id: str) -> Fact | None:
        return self._fact_store.get_fact(fact_id)

    def update_fact(self, fact_id: str, **updates: Any) -> Fact | None:
        return self._fact_store.update_fact(fact_id, **updates)

    def query_facts(self, query: FactQuery | None = None) -> list[Fact]:
        return self._fact_store.query_facts(query)

    def confirm_fact(self, fact_id: str) -> Fact | None:
        return self._fact_store.confirm_fact(fact_id)

    def dispute_fact(self, fact_id: str) -> Fact | None:
        return self._fact_store.dispute_fact(fact_id)

    def retract_fact(self, fact_id: str) -> Fact | None:
        return self._fact_store.retract_fact(fact_id)

    def get_fact_versions(self, fact_id: str) -> list:
        return self._fact_store.get_versions(fact_id)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    async def validate_fact(self, fact: Fact) -> ValidationReport:
        return await self._validator.validate_fact(fact, store=self._fact_store)

    async def validate_all_facts(self) -> list[ValidationReport]:
        facts = self._fact_store.query_facts()
        reports = []
        for f in facts:
            report = await self._validator.validate_fact(f, store=self._fact_store)
            reports.append(report)
            # Adjust confidence
            self._fact_store.update_fact(f.id, confidence=report.adjusted_confidence)
        return reports

    # ------------------------------------------------------------------
    # Reasoning
    # ------------------------------------------------------------------

    async def reason(self) -> list[InferenceResult]:
        return await self._reasoning.infer(self._fact_store, self._graph)

    async def check_contradictions(self) -> list[tuple[Fact, Fact, str]]:
        return await self._reasoning.check_contradictions(self._fact_store)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def ingest_file(self, filepath: str) -> Document | None:
        doc = await self._ingestion.ingest_file(filepath)
        if doc:
            self._search.index_document(doc.id, doc.content, {
                "title": doc.title,
                "source": doc.source,
                "content_type": doc.content_type,
            })
        return doc

    async def ingest_text(
        self,
        text: str,
        title: str = "Untitled",
        source: str = "",
        content_type: str = "text",
    ) -> Document:
        doc = await self._ingestion.ingest_text(text, title=title, source=source, content_type=content_type)
        self._search.index_document(doc.id, doc.content, {
            "title": doc.title,
            "source": doc.source,
            "content_type": doc.content_type,
        })
        if self._config.auto_validate:
            for chunk in doc.chunks:
                pass  # validation is fact-specific
        if self._config.auto_reason:
            await self.reason()
        return doc

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(self, query: SearchQuery | str) -> SearchResponse:
        return await self._search.search(query)

    async def search_neighbors(self, node_id: str, max_depth: int = 1) -> SearchResponse:
        return await self._search.search_neighbors(node_id, max_depth=max_depth)

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def export_to_json(self, indent: int = 2) -> str:
        data = {
            "graph": self._graph.to_dict(),
            "facts": self._fact_store.export_facts(),
            "config": {
                "auto_extract_entities": self._config.auto_extract_entities,
                "auto_validate": self._config.auto_validate,
                "auto_reason": self._config.auto_reason,
            },
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        return json.dumps(data, indent=indent, default=str)

    def export_to_file(self, filepath: str) -> bool:
        try:
            data = self.export_to_json()
            Path(filepath).write_text(data, encoding="utf-8")
            structured_log(logging.INFO, "knowledge.exported",
                           path=filepath,
                           nodes=self._graph.node_count(),
                           facts=self._fact_store.count())
            return True
        except Exception as e:
            logger.error("Export failed: %s", e)
            return False

    def import_from_json(self, text: str) -> bool:
        try:
            data = json.loads(text)
            self._graph = KnowledgeGraph.from_dict(data.get("graph", {}))
            for fdata in data.get("facts", []):
                self._fact_store.add_fact(Fact.from_dict(fdata))
            # Rebuild search index
            self._search = KnowledgeSearch(
                knowledge_graph=self._graph,
                fact_store=self._fact_store,
            )
            structured_log(logging.INFO, "knowledge.imported",
                           nodes=self._graph.node_count(),
                           facts=self._fact_store.count())
            return True
        except Exception as e:
            logger.error("Import failed: %s", e)
            return False

    def import_from_file(self, filepath: str) -> bool:
        try:
            text = Path(filepath).read_text("utf-8")
            return self.import_from_json(text)
        except Exception as e:
            logger.error("Import from file failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "graph": self._graph.stats(),
            "facts": self._fact_store.stats(),
            "ingestion_chunk_size": self._config.ingestion.chunk_size,
            "auto_extract_entities": self._config.auto_extract_entities,
            "auto_validate": self._config.auto_validate,
            "auto_reason": self._config.auto_reason,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._graph.clear()
        self._fact_store.clear()
        self._search = KnowledgeSearch(
            knowledge_graph=self._graph,
            fact_store=self._fact_store,
        )
        self._initialized = False
        structured_log(logging.INFO, "knowledge.cleared")

    def __repr__(self) -> str:
        return (
            f"KnowledgeBase(nodes={self._graph.node_count()}, "
            f"edges={self._graph.edge_count()}, "
            f"facts={self._fact_store.count()})"
        )

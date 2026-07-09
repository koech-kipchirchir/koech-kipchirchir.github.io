"""
Hybrid search over knowledge graph, fact store, and document content.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from knowledge.fact_store import Fact, FactQuery, FactStore, FactStatus
from knowledge.knowledge_graph import GraphNode, KnowledgeGraph
from knowledge.utils import structured_log

logger = logging.getLogger("aios.knowledge.search")


@dataclass
class SearchQuery:
    """A structured search query with filters."""

    text: str = ""
    entity_types: list[str] | None = None
    relation_types: list[str] | None = None
    source: str | None = None
    min_confidence: float = 0.0
    max_results: int = 20
    offset: int = 0
    include_graph: bool = True
    include_facts: bool = True
    include_documents: bool = True
    boost_exact_match: bool = True
    weights: dict[str, float] = field(default_factory=lambda: {
        "keyword": 1.0,
        "semantic": 0.0,
        "graph": 1.5,
        "fact": 1.0,
    })


@dataclass
class SearchResult:
    """A single search result with score and source."""

    id: str = ""
    title: str = ""
    content: str = ""
    score: float = 0.0
    source_type: str = ""  # graph_node, fact, document, chunk
    source_subtype: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    matched_terms: list[str] = field(default_factory=list)


@dataclass
class SearchResponse:
    """Aggregated search response."""

    results: list[SearchResult] = field(default_factory=list)
    total_count: int = 0
    query: str = ""
    duration_ms: float = 0.0


class TextIndex:
    """Simple in-memory inverted index for keyword search."""

    def __init__(self) -> None:
        self._documents: dict[str, str] = {}
        self._inverted: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._doc_count: int = 0

    def add_document(self, doc_id: str, text: str) -> None:
        self._documents[doc_id] = text
        tokens = self._tokenize(text)
        for token in set(tokens):
            self._inverted[token][doc_id] = self._inverted[token].get(doc_id, 0) + 1
        self._doc_count = len(self._documents)

    def remove_document(self, doc_id: str) -> None:
        self._documents.pop(doc_id, None)
        for token in list(self._inverted.keys()):
            self._inverted[token].pop(doc_id, None)
            if not self._inverted[token]:
                del self._inverted[token]
        self._doc_count = len(self._documents)

    def search(self, query: str, max_results: int = 20) -> list[tuple[str, float, list[str]]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: dict[str, float] = defaultdict(float)
        matched_terms: dict[str, set[str]] = defaultdict(set)
        n = max(self._doc_count, 1)

        for qt in query_tokens:
            df = len(self._inverted.get(qt, {}))
            idf = math.log((n + 1) / (df + 1)) + 1
            for doc_id, tf in self._inverted.get(qt, {}).items():
                scores[doc_id] += (tf / (tf + 1)) * idf
                matched_terms[doc_id].add(qt)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in sorted_docs[:max_results]:
            results.append((doc_id, score, list(matched_terms.get(doc_id, []))))
        return results

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r"\b[a-z0-9_]+\b", text)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "out", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "just", "because",
            "but", "and", "or", "if", "while", "that", "this", "these",
            "those", "it", "its", "what", "which", "who", "whom",
        }
        return [t for t in tokens if t not in stopwords and len(t) > 1]


class KnowledgeSearch:
    """Hybrid search combining keyword, graph, and fact store queries."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph | None = None,
        fact_store: FactStore | None = None,
    ) -> None:
        self._knowledge_graph = knowledge_graph
        self._fact_store = fact_store
        self._text_index = TextIndex()
        self._document_index: dict[str, dict[str, Any]] = {}

    def index_document(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        self._text_index.add_document(doc_id, text)
        self._document_index[doc_id] = {
            "text": text,
            "metadata": metadata or {},
        }

    def remove_document(self, doc_id: str) -> None:
        self._text_index.remove_document(doc_id)
        self._document_index.pop(doc_id, None)

    async def search(self, query: SearchQuery | str) -> SearchResponse:
        start = time.perf_counter()
        if isinstance(query, str):
            query = SearchQuery(text=query)

        results: list[SearchResult] = []
        seen_ids: set[str] = set()

        if query.include_graph and self._knowledge_graph:
            graph_results = self._search_graph(query)
            for r in graph_results:
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    results.append(r)

        if query.include_facts and self._fact_store:
            fact_results = self._search_facts(query)
            for r in fact_results:
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    results.append(r)

        if query.include_documents:
            doc_results = self._search_documents(query)
            for r in doc_results:
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    results.append(r)

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        total = len(results)
        results = results[query.offset:query.offset + query.max_results]

        duration = (time.perf_counter() - start) * 1000
        structured_log(logging.DEBUG, "search.complete",
                       query=query.text,
                       results=total,
                       duration_ms=round(duration, 1))

        return SearchResponse(
            results=results,
            total_count=total,
            query=query.text,
            duration_ms=round(duration, 1),
        )

    def _search_graph(self, query: SearchQuery) -> list[SearchResult]:
        if not self._knowledge_graph:
            return []
        results: list[SearchResult] = []
        qt = query.text.lower()

        for node in self._knowledge_graph.all_nodes():
            score = self._score_node(node, qt, query)
            if score <= 0:
                continue
            neighbors = self._knowledge_graph.get_neighbors(node.id, direction="both")
            neighbor_names = [n.name for _, n in neighbors[:5]]
            results.append(SearchResult(
                id=f"graph:{node.id}",
                title=node.name,
                content=json.dumps({
                    "name": node.name,
                    "type": node.type,
                    "neighbors": neighbor_names,
                    "properties": node.properties,
                }, default=str),
                score=score,
                source_type="graph_node",
                source_subtype=node.type,
                metadata={
                    "node_id": node.id,
                    "type": node.type,
                    "neighbor_count": len(neighbors),
                    "properties": dict(node.properties),
                },
                matched_terms=[qt] if qt in node.name.lower() else [],
            ))

        return results

    def _score_node(self, node: GraphNode, qt: str, query: SearchQuery) -> float:
        score = 0.0
        name_lower = node.name.lower()

        # Exact match boost
        if query.boost_exact_match and qt == name_lower:
            score += 5.0
        elif qt in name_lower:
            score += 3.0

        # Type filter
        if query.entity_types:
            if node.type not in query.entity_types:
                score -= 2.0

        # Partial token match
        for token in qt.split():
            if len(token) > 2 and token in name_lower:
                score += 1.0

        # Properties match
        for val in node.properties.values():
            if isinstance(val, str) and qt in val.lower():
                score += 0.5

        score *= query.weights.get("graph", 1.5)
        return max(0.0, score)

    def _search_facts(self, query: SearchQuery) -> list[SearchResult]:
        if not self._fact_store:
            return []
        results: list[SearchResult] = []
        qt = query.text.lower()

        fact_query = FactQuery(
            subject=qt if qt else None,
            min_confidence=query.min_confidence,
            limit=100,
        )
        facts = self._fact_store.query_facts(fact_query)

        for fact in facts:
            score = 0.0
            combined = f"{fact.subject} {fact.predicate} {fact.object}".lower()
            if qt in combined:
                score += 2.0
            if qt == fact.subject.lower():
                score += 3.0
            if qt == fact.object.lower():
                score += 2.0

            score *= fact.confidence
            score *= query.weights.get("fact", 1.0)

            if score > 0:
                results.append(SearchResult(
                    id=f"fact:{fact.id}",
                    title=f"{fact.subject} {fact.predicate} {fact.object}",
                    content=json.dumps(fact.to_dict(), default=str),
                    score=score,
                    source_type="fact",
                    source_subtype=fact.status.value,
                    metadata={
                        "fact_id": fact.id,
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "object": fact.object,
                        "confidence": fact.confidence,
                        "status": fact.status.value,
                    },
                    matched_terms=[qt] if qt in combined else [],
                ))

        return results

    def _search_documents(self, query: SearchQuery) -> list[SearchResult]:
        if not self._document_index:
            return []
        results: list[SearchResult] = []

        kw_results = self._text_index.search(query.text, max_results=query.max_results)
        for doc_id, score, terms in kw_results:
            doc_info = self._document_index.get(doc_id, {})
            results.append(SearchResult(
                id=f"doc:{doc_id}",
                title=doc_info.get("metadata", {}).get("title", doc_id),
                content=doc_info.get("text", "")[:500],
                score=score * query.weights.get("keyword", 1.0),
                source_type="document",
                source_subtype=doc_info.get("metadata", {}).get("content_type", "text"),
                metadata=dict(doc_info.get("metadata", {})),
                matched_terms=terms,
            ))

        return results

    async def search_neighbors(
        self,
        node_id: str,
        max_depth: int = 1,
        max_results: int = 20,
    ) -> SearchResponse:
        start = time.perf_counter()
        if not self._knowledge_graph:
            return SearchResponse(query=f"neighbors:{node_id}")

        nodes = self._knowledge_graph.traverse(node_id, max_depth=max_depth)
        results: list[SearchResult] = []
        for node in nodes[:max_results]:
            neighbors = self._knowledge_graph.get_neighbors(node.id, direction="both")
            neighbor_names = [n.name for _, n in neighbors[:5]]
            results.append(SearchResult(
                id=f"graph:{node.id}",
                title=node.name,
                content=json.dumps({
                    "name": node.name,
                    "type": node.type,
                    "neighbors": neighbor_names,
                    "properties": node.properties,
                }, default=str),
                score=1.0 / (len(results) + 1),
                source_type="graph_node",
                source_subtype=node.type,
                metadata={"node_id": node.id},
            ))

        duration = (time.perf_counter() - start) * 1000
        return SearchResponse(
            results=results[:max_results],
            total_count=len(nodes),
            query=f"neighbors:{node_id}",
            duration_ms=round(duration, 1),
        )

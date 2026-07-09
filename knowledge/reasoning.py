"""
Rule-based reasoning engine for knowledge inference, contradiction
detection, and transitive closure.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from knowledge.fact_store import Fact, FactStore, FactStatus
from knowledge.knowledge_graph import GraphEdge, GraphNode, KnowledgeGraph
from knowledge.utils import structured_log

logger = logging.getLogger("aios.knowledge.reasoning")


class RuleType(Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    TRANSITIVE = "transitive"
    ABDUCTIVE = "abductive"
    DEFAULT = "default"


@dataclass
class Rule:
    """A logical rule for knowledge inference."""

    name: str = ""
    antecedent: str = ""        # Pattern or condition
    consequent: str = ""        # Inferred fact pattern
    rule_type: RuleType = RuleType.DEDUCTIVE
    confidence_multiplier: float = 0.8
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Result of applying a rule."""

    id: str = ""
    rule_name: str = ""
    rule_type: RuleType = RuleType.DEDUCTIVE
    source_facts: list[str] = field(default_factory=list)
    inferred_facts: list[Fact] = field(default_factory=list)
    confidence: float = 0.0
    duration_ms: float = 0.0
    success: bool = False


@dataclass
class ReasoningTrace:
    """Trace of a reasoning chain."""

    steps: list[str] = field(default_factory=list)
    facts_used: list[str] = field(default_factory=list)
    facts_derived: list[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------

TRANSITIVE_RULE = Rule(
    name="transitive_relation",
    antecedent="(?P<subj>.+) (?P<pred>.+) (?P<obj1>.+) AND (?P<obj1>) (?P<pred2>.+) (?P<obj2>.+)",
    consequent="{subj} {pred} {obj2}",
    rule_type=RuleType.TRANSITIVE,
    confidence_multiplier=0.7,
)

SAME_AS_RULE = Rule(
    name="same_entity",
    antecedent="(?P<a>.+) same_as (?P<b>.+) AND (?P<b>) (?P<rel>.+) (?P<c>.+)",
    consequent="{a} {rel} {c}",
    rule_type=RuleType.DEDUCTIVE,
    confidence_multiplier=0.6,
)

CONTRADICTION_RULE = Rule(
    name="contradiction",
    antecedent="(?P<subj>.+) (?P<pred>.+) (?P<obj1>.+) AND (?P<subj>) (?P<pred>.+) (?P<obj2>.+)",
    consequent="CONTRADICTION: {subj} {pred} {obj1} vs {obj2}",
    rule_type=RuleType.DEDUCTIVE,
    confidence_multiplier=0.9,
)

INVERSE_RULE = Rule(
    name="inverse_relation",
    antecedent="(?P<a>.+) (?P<rel>.+) (?P<b>.+)",
    consequent="{b} inverse_{rel} {a}",
    rule_type=RuleType.DEDUCTIVE,
    confidence_multiplier=0.8,
)


class RuleBasedEngine:
    """Rule-based reasoning engine over facts and graph.

    Supports transitive closure, deduction, induction, and
    contradiction detection using configurable rule sets.
    """

    def __init__(
        self,
        rules: list[Rule] | None = None,
        max_inference_depth: int = 3,
    ) -> None:
        self._rules: list[Rule] = rules or [
            TRANSITIVE_RULE,
            CONTRADICTION_RULE,
            INVERSE_RULE,
        ]
        self._max_depth = max_inference_depth

    def register_rule(self, rule: Rule) -> None:
        self._rules.append(rule)

    def unregister_rule(self, rule_name: str) -> bool:
        for i, r in enumerate(self._rules):
            if r.name == rule_name:
                self._rules.pop(i)
                return True
        return False

    # ------------------------------------------------------------------
    # Fact-level inference
    # ------------------------------------------------------------------

    async def infer(self, fact_store: FactStore, knowledge_graph: KnowledgeGraph | None = None) -> list[InferenceResult]:
        start = time.perf_counter()
        results: list[InferenceResult] = []
        all_facts = fact_store.query_facts()

        # Transitive closure
        trans_result = self._apply_transitive(all_facts, fact_store)
        if trans_result.success:
            results.append(trans_result)

        # Contradiction detection
        contra_result = self._detect_contradictions(all_facts, fact_store)
        if contra_result.success:
            results.append(contra_result)

        # Inverse relations
        inverse_result = self._apply_inverse(all_facts, fact_store)
        if inverse_result.success:
            results.append(inverse_result)

        structured_log(logging.DEBUG, "reasoning.infer.complete",
                       rules_applied=len(results),
                       duration_ms=round((time.perf_counter() - start) * 1000, 1))

        return results

    def _apply_transitive(self, facts: list[Fact], store: FactStore) -> InferenceResult:
        result = InferenceResult(
            id=_generate_id(),
            rule_name="transitive_relation",
            rule_type=RuleType.TRANSITIVE,
        )
        # Build adjacency: (subject, predicate) -> list of objects
        adj: dict[tuple[str, str], list[str]] = {}
        for f in facts:
            if f.is_active:
                key = (f.subject, f.predicate)
                if key not in adj:
                    adj[key] = []
                adj[key].append(f.object)

        inferred_count = 0
        for (subj, pred), objs in adj.items():
            for obj in objs:
                next_key = (obj, pred)
                if next_key in adj:
                    for target in adj[next_key]:
                        if target != subj:
                            # Create inferred fact
                            inferred = Fact(
                                subject=subj,
                                predicate=pred,
                                object=target,
                                confidence=0.6,
                                status=FactStatus.PROPOSED,
                                provenance={"inferred_from": "transitive_closure",
                                            "via": obj},
                            )
                            existing = store.add_fact(inferred)
                            result.inferred_facts.append(inferred)
                            result.source_facts.append(f"{subj} {pred} {obj}")
                            inferred_count += 1

        result.success = inferred_count > 0
        result.confidence = 0.7
        return result

    def _detect_contradictions(self, facts: list[Fact], store: FactStore) -> InferenceResult:
        result = InferenceResult(
            id=_generate_id(),
            rule_name="contradiction",
            rule_type=RuleType.DEDUCTIVE,
        )
        # Group facts by (subject, predicate)
        groups: dict[tuple[str, str], list[Fact]] = {}
        for f in facts:
            if f.is_active:
                key = (f.subject, f.predicate)
                if key not in groups:
                    groups[key] = []
                groups[key].append(f)

        for (subj, pred), group in groups.items():
            objects = set(f.object for f in group)
            if len(objects) > 1:
                # Mark all as disputed
                for f in group:
                    try:
                        store.update_fact(f.id, status=FactStatus.DISPUTED)
                    except (ValueError, KeyError):
                        pass
                    result.source_facts.append(f.id)
                result.inferred_facts.append(Fact(
                    subject=subj,
                    predicate="conflicts_with",
                    object=", ".join(sorted(objects)),
                    confidence=0.9,
                    status=FactStatus.CONFIRMED,
                ))
                continue

        result.success = bool(result.inferred_facts)
        result.confidence = 0.9
        return result

    def _apply_inverse(self, facts: list[Fact], store: FactStore) -> InferenceResult:
        result = InferenceResult(
            id=_generate_id(),
            rule_name="inverse_relation",
            rule_type=RuleType.DEDUCTIVE,
        )
        inverse_map = {
            "created_by": "created",
            "written_by": "wrote",
            "employed_by": "employs",
            "located_in": "contains",
            "part_of": "contains",
            "parent_of": "child_of",
            "depends_on": "depended_by",
        }
        for f in facts:
            if f.is_active:
                inverse_pred = inverse_map.get(f.predicate)
                if inverse_pred:
                    inferred = Fact(
                        subject=f.object,
                        predicate=inverse_pred,
                        object=f.subject,
                        confidence=f.confidence * 0.8,
                        status=FactStatus.PROPOSED,
                        provenance={"inferred_from": "inverse_relation",
                                    "source_fact": f.id},
                    )
                    store.add_fact(inferred)
                    result.inferred_facts.append(inferred)
                    result.source_facts.append(f.id)

        result.success = bool(result.inferred_facts)
        result.confidence = 0.8
        return result

    # ------------------------------------------------------------------
    # Graph-level reasoning
    # ------------------------------------------------------------------

    async def reason_graph(
        self,
        knowledge_graph: KnowledgeGraph,
        max_depth: int | None = None,
    ) -> list[InferenceResult]:
        start = time.perf_counter()
        results: list[InferenceResult] = []
        depth = max_depth or self._max_depth
        all_nodes = knowledge_graph.all_nodes()

        # Transitive edges
        trans_result = self._graph_transitive_closure(knowledge_graph, depth)
        if trans_result.success:
            results.append(trans_result)

        structured_log(logging.DEBUG, "reasoning.graph.complete",
                       results=len(results),
                       duration_ms=round((time.perf_counter() - start) * 1000, 1))

        return results

    def _graph_transitive_closure(self, graph: KnowledgeGraph, max_depth: int) -> InferenceResult:
        result = InferenceResult(
            id=_generate_id(),
            rule_name="graph_transitive",
            rule_type=RuleType.TRANSITIVE,
        )
        added = 0
        # For each pair of nodes, BFS to find indirect connections
        for node in graph.all_nodes():
            paths = []
            for other in graph.all_nodes():
                if other.id == node.id:
                    continue
                found = graph.find_path(node.id, other.id, max_depth=max_depth)
                paths.extend(found)
            for path in paths:
                if path.length > 1:
                    # Infer a direct edge between the two endpoints
                    direct_type = "indirectly_related"
                    try:
                        edge = GraphEdge(
                            source_id=node.id,
                            target_id=path.node_ids[-1],
                            type=direct_type,
                            weight=1.0 / path.length,
                            properties={"inferred": True, "path": path.node_ids},
                        )
                        graph.add_edge(edge)
                        added += 1
                    except (KeyError, ValueError):
                        pass
            if added > 100:
                break

        result.success = added > 0
        result.confidence = 0.5
        result.duration_ms = 0
        return result

    # ------------------------------------------------------------------
    # Custom inference
    # ------------------------------------------------------------------

    async def apply_rule(
        self,
        rule: Rule,
        fact_store: FactStore,
        knowledge_graph: KnowledgeGraph | None = None,
    ) -> InferenceResult:
        result = InferenceResult(
            id=_generate_id(),
            rule_name=rule.name,
            rule_type=rule.rule_type,
        )
        start = time.perf_counter()

        facts = fact_store.query_facts()
        for fact in facts:
            if not fact.is_active:
                continue
            # Simple pattern matching based on rule antecedent
            combined = f"{fact.subject} {fact.predicate} {fact.object}"
            if rule.antecedent and rule.antecedent.lower() in combined.lower():
                inferred = Fact(
                    subject=fact.subject,
                    predicate=f"inferred_{fact.predicate}",
                    object=fact.object,
                    confidence=fact.confidence * rule.confidence_multiplier,
                    status=FactStatus.PROPOSED,
                    provenance={
                        "rule": rule.name,
                        "source_fact": fact.id,
                    },
                )
                fact_store.add_fact(inferred)
                result.inferred_facts.append(inferred)
                result.source_facts.append(fact.id)

        result.confidence = rule.confidence_multiplier
        result.duration_ms = (time.perf_counter() - start) * 1000
        result.success = bool(result.inferred_facts)
        return result

    # ------------------------------------------------------------------
    # Contradiction check
    # ------------------------------------------------------------------

    async def check_contradictions(
        self,
        fact_store: FactStore,
    ) -> list[tuple[Fact, Fact, str]]:
        contradictions: list[tuple[Fact, Fact, str]] = []
        facts = fact_store.query_facts()

        groups: dict[tuple[str, str], list[Fact]] = {}
        for f in facts:
            if f.is_active:
                key = (f.subject, f.predicate)
                groups.setdefault(key, []).append(f)

        for (subj, pred), group in groups.items():
            objects = set(f.object for f in group)
            if len(objects) > 1:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        if group[i].object != group[j].object:
                            contradictions.append((
                                group[i],
                                group[j],
                                f"Conflicting {pred} for {subj}: "
                                f"'{group[i].object}' vs '{group[j].object}'",
                            ))

        return contradictions


def _generate_id() -> str:
    return uuid.uuid4().hex[:12]

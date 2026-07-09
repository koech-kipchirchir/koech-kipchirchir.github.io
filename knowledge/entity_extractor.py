"""
Entity extraction module for identifying and classifying entities
and relations in text.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from knowledge.utils import structured_log

logger = logging.getLogger("aios.knowledge.entity")


class EntityType(Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    CONCEPT = "concept"
    TECHNOLOGY = "technology"
    PRODUCT = "product"
    EVENT = "event"
    ROLE = "role"
    FILE = "file"
    CODE = "code"
    TERM = "term"
    METRIC = "metric"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """A typed entity extracted from text."""

    id: str = ""
    name: str = ""
    type: EntityType = EntityType.UNKNOWN
    canonical_name: str = ""
    aliases: list[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.name.lower())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.name.lower() == other.name.lower()


@dataclass
class Relation:
    """A typed relation between two entities."""

    id: str = ""
    source_id: str = ""
    target_id: str = ""
    type: str = "related_to"
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedEntity:
    """An entity occurrence at a specific text location."""

    entity: Entity = field(default_factory=Entity)
    text: str = ""
    start: int = 0
    end: int = 0
    sentence: str = ""


@dataclass
class ExtractedRelation:
    """A relation occurrence at a specific text location."""

    relation: Relation = field(default_factory=Relation)
    text: str = ""
    sentence: str = ""


@dataclass
class ExtractionResult:
    """Result of entity extraction from a document."""

    document_id: str = ""
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    duration_ms: float = 0.0
    entity_count: int = 0
    relation_count: int = 0


def _generate_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Default extraction patterns
# ---------------------------------------------------------------------------

PERSON_PATTERNS = [
    r"[A-Z][a-z]+ [A-Z][a-z]+",
    r"Dr\. [A-Z][a-z]+ [A-Z][a-z]+",
    r"Prof\. [A-Z][a-z]+ [A-Z][a-z]+",
]

ORG_PATTERNS = [
    r"[A-Z][a-zA-Z]+ (?:Inc|Corp|LLC|Ltd|GmbH|SA|PLC|Co)",
    r"[A-Z][a-z]+ [A-Z][a-z]+ (?:University|Institute|School|College|Laboratory|Lab)",
    r"the [A-Z][a-z]+ [A-Z][a-z]+",
]

LOCATION_PATTERNS = [
    r"(?:in|at|from|to) [A-Z][a-z]+(?: [A-Z][a-z]+)*",
]

TECH_PATTERNS = [
    r"[A-Z][a-z]+(?:Script|Lang|OS|DB|ML|AI|API|SDK|VM|CLI)",
    r"Python|JavaScript|TypeScript|Rust|Go|Kotlin|Swift|Java|C\+\+|C#",
    r"React|Angular|Vue|Django|Flask|FastAPI|TensorFlow|PyTorch",
    r"Linux|Windows|macOS|iOS|Android|Docker|Kubernetes|AWS|GCP|Azure",
    r"\b\w+FS\b",
]

CONCEPT_PATTERNS = [
    r"(?:the concept of|idea of|principle of) [a-zA-Z]+",
    r"[A-Z][a-z]+ [A-Z][a-z]+ (?:Theory|Law|Effect|Paradox|Hypothesis)",
    r"\b(?:API|REST|GraphQL|gRPC|SQL|NoSQL|ORM|SDK|CLI|GUI|TLS|SSL|SSH|HTTP|HTTPS|FTP|DNS|DHCP|JSON|XML|YAML|TOML|CSV)\b",
]

RELATION_PATTERNS: list[tuple[str, str, str]] = [
    (r"(.+?) (?:works at|is employed by|is a) (.+)", "employed_by", "is_a"),
    (r"(.+?) (?:created|developed|built|designed|invented|founded) (.+)", "created", "developed"),
    (r"(.+?) (?:located in|based in|situated in) (.+)", "located_in", "based_in"),
    (r"(.+?) (?:uses|utilizes|employs|runs on) (.+)", "uses", "employs"),
    (r"(.+?) (?:part of|member of|belongs to) (.+)", "part_of", "belongs_to"),
    (r"(.+?) (?:succeeded by|followed by|replaced by) (.+)", "succeeded_by", "followed_by"),
    (r"(.+?) (?:leads to|results in|causes) (.+)", "leads_to", "causes"),
    (r"(.+?) (?:depends on|requires|needs) (.+)", "depends_on", "requires"),
    (r"(.+?) (?:similar to|related to|analogous to) (.+)", "similar_to", "related_to"),
    (r"(.+?) (?:wrote|authored|published) (.+)", "authored", "wrote"),
]


# ---------------------------------------------------------------------------
# Entity extractor
# ---------------------------------------------------------------------------


class EntityExtractor:
    """Abstract entity extraction interface."""

    async def extract(self, text: str, document_id: str = "") -> ExtractionResult:
        raise NotImplementedError


class RegexEntityExtractor(EntityExtractor):
    """Rule-based entity extractor using regex patterns.

    Supports person, organization, location, technology, concept,
    and relation extraction via compiled patterns.
    """

    def __init__(
        self,
        person_patterns: list[str] | None = None,
        org_patterns: list[str] | None = None,
        location_patterns: list[str] | None = None,
        tech_patterns: list[str] | None = None,
        concept_patterns: list[str] | None = None,
        relation_patterns: list[tuple[str, str, str]] | None = None,
        min_confidence: float = 0.3,
    ) -> None:
        self._person_patterns = [
            re.compile(p) for p in (person_patterns or PERSON_PATTERNS)
        ]
        self._org_patterns = [
            re.compile(p) for p in (org_patterns or ORG_PATTERNS)
        ]
        self._location_patterns = [
            re.compile(p) for p in (location_patterns or LOCATION_PATTERNS)
        ]
        self._tech_patterns = [
            re.compile(p) for p in (tech_patterns or TECH_PATTERNS)
        ]
        self._concept_patterns = [
            re.compile(p) for p in (concept_patterns or CONCEPT_PATTERNS)
        ]
        self._relation_patterns = relation_patterns or RELATION_PATTERNS
        self._compiled_relations = [
            (re.compile(pattern), t1, t2)
            for pattern, t1, t2 in self._relation_patterns
        ]
        self._min_confidence = min_confidence

    async def extract(self, text: str, document_id: str = "") -> ExtractionResult:
        start = time.perf_counter()
        result = ExtractionResult(document_id=document_id)

        if not text:
            return result

        seen_entities: dict[str, Entity] = {}
        seen_entity_occurrences: list[ExtractedEntity] = []
        sents = self._split_sentences(text)

        for sent in sents:
            entities_in_sent = self._extract_entities_from_sentence(sent, text)
            for ee in entities_in_sent:
                key = ee.entity.name.lower()
                if key in seen_entities:
                    existing = seen_entities[key]
                    if ee.entity.confidence > existing.confidence:
                        seen_entities[key] = ee.entity
                else:
                    seen_entities[key] = ee.entity
                ee.entity.id = seen_entities[key].id or _generate_id()
                ee.entity.canonical_name = seen_entities[key].name
                seen_entity_occurrences.append(ee)

        # Deduplicate
        entity_map: dict[str, Entity] = {}
        for ee in seen_entity_occurrences:
            key = ee.entity.name.lower()
            if key not in entity_map:
                entity_map[key] = ee.entity
                entity_map[key].id = _generate_id()
                result.entities.append(ee)
            ee.entity.id = entity_map[key].id

        # Relation extraction
        for sent in sents:
            rels = self._extract_relations_from_sentence(sent, entity_map, text)
            for er in rels:
                er.relation.id = _generate_id()
                result.relations.append(er)

        result.entity_count = len(result.entities)
        result.relation_count = len(result.relations)
        result.duration_ms = (time.perf_counter() - start) * 1000
        structured_log(logging.DEBUG, "entity.extraction.complete",
                       document_id=document_id,
                       entities=result.entity_count,
                       relations=result.relation_count,
                       duration_ms=round(result.duration_ms, 1))
        return result

    def _split_sentences(self, text: str) -> list[str]:
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if len(s.strip()) > 5]

    def _extract_entities_from_sentence(self, sentence: str, full_text: str) -> list[ExtractedEntity]:
        entities: list[ExtractedEntity] = []

        for pattern, etype in [
            (self._person_patterns, EntityType.PERSON),
            (self._org_patterns, EntityType.ORGANIZATION),
            (self._location_patterns, EntityType.LOCATION),
            (self._tech_patterns, EntityType.TECHNOLOGY),
            (self._concept_patterns, EntityType.CONCEPT),
        ]:
            for pat in pattern:
                for m in pat.finditer(sentence):
                    name = m.group(0).strip()
                    if len(name) < 2 or len(name) > 80:
                        continue
                    # Clean location prefix
                    if etype == EntityType.LOCATION:
                        for prefix in ["in ", "at ", "from ", "to "]:
                            if name.lower().startswith(prefix):
                                name = name[len(prefix):].strip()
                    if not name:
                        continue
                    conf = self._compute_confidence(name, etype)
                    if conf < self._min_confidence:
                        continue
                    entity = Entity(
                        name=name,
                        type=etype,
                        confidence=conf,
                    )
                    ee = ExtractedEntity(
                        entity=entity,
                        text=name,
                        start=m.start(),
                        end=m.end(),
                        sentence=sentence,
                    )
                    entities.append(ee)

        return entities

    def _compute_confidence(self, name: str, etype: EntityType) -> float:
        conf = 0.7
        if len(name) <= 3:
            conf -= 0.2
        if any(c.isdigit() for c in name) and etype == EntityType.PERSON:
            conf -= 0.3
        if name[0].islower() and etype in (EntityType.PERSON, EntityType.ORGANIZATION):
            conf -= 0.2
        if etype == EntityType.LOCATION and name[0].islower():
            conf -= 0.3
        for kw in ["the ", "a ", "an "]:
            if name.lower().startswith(kw) and etype == EntityType.PERSON:
                conf -= 0.2
        return max(0.1, min(1.0, conf))

    def _extract_relations_from_sentence(
        self,
        sentence: str,
        entity_map: dict[str, Entity],
        full_text: str,
    ) -> list[ExtractedRelation]:
        rels: list[ExtractedRelation] = []
        for pat, type_a, type_b in self._compiled_relations:
            for m in pat.finditer(sentence):
                left = m.group(1).strip()
                right = m.group(2).strip()
                src = self._find_best_entity(left, entity_map)
                tgt = self._find_best_entity(right, entity_map)
                if src and tgt and src.id != tgt.id:
                    etype = type_a if "created" in pat.pattern or "authored" in pat.pattern else "related_to"
                    rel = Relation(
                        source_id=src.id,
                        target_id=tgt.id,
                        type=etype,
                        confidence=0.6,
                    )
                    er = ExtractedRelation(
                        relation=rel,
                        text=f"{left} -> {right}",
                        sentence=sentence,
                    )
                    rels.append(er)
                    break
        return rels

    def _find_best_entity(self, text: str, entity_map: dict[str, Entity]) -> Entity | None:
        tl = text.lower()
        best: Entity | None = None
        best_score = 0.0
        for key, ent in entity_map.items():
            if key in tl or tl in key:
                score = len(key) / max(len(tl), 1)
                if score > best_score:
                    best_score = score
                    best = ent
        return best

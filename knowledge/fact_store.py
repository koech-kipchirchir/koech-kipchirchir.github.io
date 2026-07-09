"""
Fact store with versioning, provenance tracking, and confidence scoring.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from knowledge.utils import structured_log

logger = logging.getLogger("aios.knowledge.fact")


class FactStatus(Enum):
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    DISPUTED = "disputed"
    RETRACTED = "retracted"
    EXPIRED = "expired"


@dataclass
class Fact:
    """A single fact with provenance and confidence."""

    id: str = ""
    subject: str = ""
    predicate: str = ""
    object: str = ""
    confidence: float = 0.5
    status: FactStatus = FactStatus.PROPOSED
    source: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    version: int = 1
    expires_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "status": self.status.value,
            "source": self.source,
            "provenance": self.provenance,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Fact:
        if isinstance(data.get("status"), str):
            data["status"] = FactStatus(data["status"])
        return cls(**data)

    @property
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        try:
            exp = datetime.fromisoformat(self.expires_at)
            return exp < datetime.now(timezone.utc)
        except (ValueError, TypeError):
            return False

    @property
    def is_active(self) -> bool:
        return self.status in (FactStatus.CONFIRMED, FactStatus.PROPOSED) and not self.is_expired


@dataclass
class FactVersion:
    """A snapshot of a fact at a point in time."""

    fact_id: str = ""
    version: int = 1
    fact_snapshot: dict[str, Any] = field(default_factory=dict)
    changed_fields: list[str] = field(default_factory=list)
    timestamp: str = ""
    change_reason: str = ""


@dataclass
class FactQuery:
    """Query filter for fact retrieval."""

    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    status: FactStatus | None = None
    min_confidence: float = 0.0
    source: str | None = None
    limit: int = 100
    offset: int = 0
    include_expired: bool = False
    sort_by: str = "created_at"
    sort_desc: bool = True


def _generate_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FactStore:
    """Fact storage with versioning and query support.

    Stores facts with provenance, supports CRUD operations,
    status transitions, confidence updates, and paginated queries.
    """

    def __init__(self) -> None:
        self._facts: dict[str, Fact] = {}
        self._versions: dict[str, list[FactVersion]] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_fact(self, fact: Fact) -> str:
        if not fact.id:
            fact.id = _generate_id()
        now = _now()
        fact.created_at = fact.created_at or now
        fact.updated_at = now
        self._facts[fact.id] = fact
        self._save_version(fact, ["*"], reason="created")
        structured_log(logging.DEBUG, "fact.added",
                       fact_id=fact.id,
                       triple=f"{fact.subject} {fact.predicate} {fact.object}")
        return fact.id

    def get_fact(self, fact_id: str) -> Fact | None:
        fact = self._facts.get(fact_id)
        if fact and not fact.is_active and fact.status in (FactStatus.RETRACTED, FactStatus.EXPIRED):
            pass
        return fact

    def update_fact(self, fact_id: str, **updates: Any) -> Fact | None:
        fact = self._facts.get(fact_id)
        if fact is None:
            return None
        changed = []
        for key, val in updates.items():
            if key == "status" and isinstance(val, str):
                val = FactStatus(val)
            if key == "status" and not self._valid_transition(fact.status, val):
                raise ValueError(f"Invalid status transition: {fact.status.value} -> {val.value}")
            if hasattr(fact, key) and getattr(fact, key) != val:
                changed.append(key)
                setattr(fact, key, val)
        fact.updated_at = _now()
        fact.version += 1
        if changed:
            self._save_version(fact, changed, reason="updated")
        structured_log(logging.DEBUG, "fact.updated",
                       fact_id=fact_id, changed=changed)
        return fact

    def delete_fact(self, fact_id: str) -> bool:
        fact = self._facts.pop(fact_id, None)
        self._versions.pop(fact_id, None)
        if fact:
            structured_log(logging.DEBUG, "fact.deleted", fact_id=fact_id)
            return True
        return False

    def count(self) -> int:
        return len(self._facts)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_facts(self, query: FactQuery | None = None) -> list[Fact]:
        facts = list(self._facts.values())
        if query is None:
            return facts

        if not query.include_expired:
            facts = [f for f in facts if not f.is_expired]

        if query.subject:
            facts = [f for f in facts if query.subject.lower() in f.subject.lower()]
        if query.predicate:
            facts = [f for f in facts if query.predicate.lower() in f.predicate.lower()]
        if query.object:
            facts = [f for f in facts if query.object.lower() in f.object.lower()]
        if query.status:
            facts = [f for f in facts if f.status == query.status]
        if query.source:
            facts = [f for f in facts if query.source.lower() in f.source.lower()]
        if query.min_confidence > 0:
            facts = [f for f in facts if f.confidence >= query.min_confidence]

        reverse = query.sort_desc
        if query.sort_by == "created_at":
            facts.sort(key=lambda f: f.created_at, reverse=reverse)
        elif query.sort_by == "confidence":
            facts.sort(key=lambda f: f.confidence, reverse=reverse)
        elif query.sort_by == "updated_at":
            facts.sort(key=lambda f: f.updated_at, reverse=reverse)

        return facts[query.offset:query.offset + query.limit]

    def get_facts_by_subject(self, subject: str) -> list[Fact]:
        return self.query_facts(FactQuery(subject=subject))

    def get_facts_by_object(self, object: str) -> list[Fact]:
        return self.query_facts(FactQuery(object=object))

    def get_facts_by_predicate(self, predicate: str) -> list[Fact]:
        return self.query_facts(FactQuery(predicate=predicate))

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def confirm_fact(self, fact_id: str) -> Fact | None:
        return self.update_fact(fact_id, status=FactStatus.CONFIRMED)

    def dispute_fact(self, fact_id: str) -> Fact | None:
        return self.update_fact(fact_id, status=FactStatus.DISPUTED)

    def retract_fact(self, fact_id: str) -> Fact | None:
        return self.update_fact(fact_id, status=FactStatus.RETRACTED)

    def _valid_transition(self, current: FactStatus, next: FactStatus) -> bool:
        transitions = {
            FactStatus.PROPOSED: [FactStatus.CONFIRMED, FactStatus.DISPUTED, FactStatus.RETRACTED],
            FactStatus.CONFIRMED: [FactStatus.DISPUTED, FactStatus.RETRACTED, FactStatus.EXPIRED],
            FactStatus.DISPUTED: [FactStatus.CONFIRMED, FactStatus.RETRACTED],
            FactStatus.RETRACTED: [],
            FactStatus.EXPIRED: [FactStatus.PROPOSED],
        }
        return next in transitions.get(current, [])

    # ------------------------------------------------------------------
    # Versions
    # ------------------------------------------------------------------

    def _save_version(self, fact: Fact, changed: list[str], reason: str = "") -> None:
        if fact.id not in self._versions:
            self._versions[fact.id] = []
        self._versions[fact.id].append(FactVersion(
            fact_id=fact.id,
            version=fact.version,
            fact_snapshot=fact.to_dict(),
            changed_fields=changed,
            timestamp=_now(),
            change_reason=reason,
        ))

    def get_versions(self, fact_id: str) -> list[FactVersion]:
        return list(self._versions.get(fact_id, []))

    def get_version(self, fact_id: str, version: int) -> FactVersion | None:
        versions = self._versions.get(fact_id, [])
        for v in versions:
            if v.version == version:
                return v
        return None

    # ------------------------------------------------------------------
    # Bulk import / export
    # ------------------------------------------------------------------

    def import_facts(self, facts: list[Fact]) -> list[str]:
        ids = []
        for fact in facts:
            ids.append(self.add_fact(fact))
        structured_log(logging.INFO, "fact.bulk_import", count=len(ids))
        return ids

    def export_facts(self, status: FactStatus | None = None) -> list[dict[str, Any]]:
        facts = self._facts.values()
        if status:
            facts = [f for f in facts if f.status == status]
        return [f.to_dict() for f in facts]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        for f in self._facts.values():
            key = f.status.value
            statuses[key] = statuses.get(key, 0) + 1
        return {
            "total_facts": len(self._facts),
            "total_versions": sum(len(v) for v in self._versions.values()),
            "by_status": statuses,
            "avg_confidence": round(
                sum(f.confidence for f in self._facts.values()) / max(len(self._facts), 1), 3
            ),
        }

    def clear(self) -> None:
        self._facts.clear()
        self._versions.clear()

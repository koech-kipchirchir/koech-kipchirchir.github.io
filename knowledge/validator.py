"""
Fact and entity validation with cross-referencing and confidence scoring.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from knowledge.fact_store import Fact, FactStore, FactStatus
from knowledge.knowledge_graph import KnowledgeGraph
from knowledge.utils import structured_log

logger = logging.getLogger("aios.knowledge.validator")


class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue found during validation."""

    id: str = ""
    rule: str = ""
    message: str = ""
    severity: ValidationSeverity = ValidationSeverity.WARNING
    confidence_delta: float = 0.0
    location: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation result for a fact or entity."""

    target_id: str = ""
    target_type: str = "fact"
    passed: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)
    original_confidence: float = 0.0
    adjusted_confidence: float = 0.0
    duration_ms: float = 0.0
    rule_count: int = 0
    passed_count: int = 0


class ValidationRule(ABC):
    """Abstract validation rule."""

    def __init__(self, name: str = "") -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def validate_fact(self, fact: Fact, store: FactStore | None = None) -> ValidationIssue | None:
        pass

    def _issue(
        self,
        rule: str,
        message: str,
        severity: ValidationSeverity = ValidationSeverity.WARNING,
        confidence_delta: float = -0.1,
        location: str = "",
        details: dict[str, Any] | None = None,
    ) -> ValidationIssue:
        return ValidationIssue(
            id=uuid.uuid4().hex[:8],
            rule=rule,
            message=message,
            severity=severity,
            confidence_delta=confidence_delta,
            location=location,
            details=details or {},
        )


class ConfidenceThresholdRule(ValidationRule):
    """Flag facts below a minimum confidence threshold."""

    def __init__(self, min_confidence: float = 0.2) -> None:
        super().__init__(name="confidence_threshold")
        self._min = min_confidence

    async def validate_fact(self, fact: Fact, store: FactStore | None = None) -> ValidationIssue | None:
        if fact.confidence < self._min:
            return self._issue(
                "confidence_threshold",
                f"Confidence ({fact.confidence:.2f}) below minimum ({self._min})",
                ValidationSeverity.WARNING,
                confidence_delta=0.0,
                location=f"fact:{fact.id}",
                details={"confidence": fact.confidence, "threshold": self._min},
            )
        return None


class EmptyFieldRule(ValidationRule):
    """Flag facts with empty subject, predicate, or object."""

    async def validate_fact(self, fact: Fact, store: FactStore | None = None) -> ValidationIssue | None:
        empty: list[str] = []
        if not fact.subject or not fact.subject.strip():
            empty.append("subject")
        if not fact.predicate or not fact.predicate.strip():
            empty.append("predicate")
        if not fact.object or not fact.object.strip():
            empty.append("object")
        if empty:
            return self._issue(
                "empty_field",
                f"Empty fields: {', '.join(empty)}",
                ValidationSeverity.ERROR,
                confidence_delta=-0.3,
                location=f"fact:{fact.id}",
                details={"empty_fields": empty},
            )
        return None


class MaxLengthRule(ValidationRule):
    """Flag facts with excessively long predicate/object values."""

    def __init__(self, max_length: int = 500) -> None:
        super().__init__(name="max_length")
        self._max = max_length

    async def validate_fact(self, fact: Fact, store: FactStore | None = None) -> ValidationIssue | None:
        if len(fact.predicate) > self._max:
            return self._issue(
                "max_length",
                f"Predicate length ({len(fact.predicate)}) exceeds maximum ({self._max})",
                ValidationSeverity.WARNING,
                confidence_delta=-0.1,
                details={"field": "predicate", "length": len(fact.predicate)},
            )
        if len(fact.object) > self._max:
            return self._issue(
                "max_length",
                f"Object length ({len(fact.object)}) exceeds maximum ({self._max})",
                ValidationSeverity.WARNING,
                confidence_delta=-0.1,
                details={"field": "object", "length": len(fact.object)},
            )
        return None


class CrossReferenceRule(ValidationRule):
    """Check for consistency across multiple facts about the same subject."""

    def __init__(self, contradiction_distance: int = 3) -> None:
        super().__init__(name="cross_reference")
        self._distance = contradiction_distance

    async def validate_fact(self, fact: Fact, store: FactStore | None = None) -> ValidationIssue | None:
        if store is None:
            return None
        # Look for conflicting facts with the same subject and predicate but different object
        similar = store.query_facts(FactQuery(
            subject=fact.subject,
            predicate=fact.predicate,
            min_confidence=0.5,
            include_expired=False,
        ))
        conflicts = [f for f in similar if f.id != fact.id and f.object != fact.object]
        if conflicts:
            diff = sum(abs(fact.confidence - c.confidence) for c in conflicts) / len(conflicts)
            return self._issue(
                "cross_reference",
                f"Found {len(conflicts)} conflicting fact(s) for {fact.subject} {fact.predicate}",
                ValidationSeverity.WARNING,
                confidence_delta=-min(0.2, diff * 0.5),
                location=f"fact:{fact.id}",
                details={
                    "conflict_count": len(conflicts),
                    "conflict_ids": [c.id for c in conflicts[:10]],
                    "conflict_objects": [c.object for c in conflicts[:5]],
                },
            )
        return None


class ProvenanceRule(ValidationRule):
    """Check that facts have adequate provenance information."""

    async def validate_fact(self, fact: Fact, store: FactStore | None = None) -> ValidationIssue | None:
        issues: list[str] = []
        if not fact.source:
            issues.append("no source")
        if not fact.provenance:
            issues.append("no provenance")
        elif not fact.provenance.get("document_id") and "document_id" not in str(fact.provenance):
            issues.append("provenance missing document_id")
        if issues:
            return self._issue(
                "provenance",
                f"Missing provenance: {', '.join(issues)}",
                ValidationSeverity.INFO,
                confidence_delta=-0.05,
                location=f"fact:{fact.id}",
                details={"missing": issues},
            )
        return None


class ExpiredFactRule(ValidationRule):
    """Flag expired facts."""

    async def validate_fact(self, fact: Fact, store: FactStore | None = None) -> ValidationIssue | None:
        if fact.is_expired:
            return self._issue(
                "expired_fact",
                f"Fact expired at {fact.expires_at}",
                ValidationSeverity.WARNING,
                confidence_delta=-0.2,
                location=f"fact:{fact.id}",
                details={"expires_at": fact.expires_at},
            )
        return None


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class KnowledgeValidator:
    """Runs validation rules against facts and entities."""

    def __init__(self, rules: list[ValidationRule] | None = None) -> None:
        self._rules = rules or [
            EmptyFieldRule(),
            MaxLengthRule(),
            ConfidenceThresholdRule(),
            CrossReferenceRule(),
            ProvenanceRule(),
            ExpiredFactRule(),
        ]

    def register_rule(self, rule: ValidationRule) -> None:
        self._rules.append(rule)

    async def validate_fact(
        self,
        fact: Fact,
        store: FactStore | None = None,
    ) -> ValidationReport:
        start = time.perf_counter()
        report = ValidationReport(
            target_id=fact.id,
            target_type="fact",
            original_confidence=fact.confidence,
            adjusted_confidence=fact.confidence,
        )

        for rule in self._rules:
            report.rule_count += 1
            try:
                issue = await rule.validate_fact(fact, store)
                if issue:
                    report.issues.append(issue)
                    report.adjusted_confidence += issue.confidence_delta
                else:
                    report.passed_count += 1
            except Exception as e:
                report.issues.append(ValidationIssue(
                    id=uuid.uuid4().hex[:8],
                    rule=rule.name,
                    message=f"Validation error: {e}",
                    severity=ValidationSeverity.WARNING,
                    details={"error": str(e)},
                ))

        report.adjusted_confidence = max(0.0, min(1.0, report.adjusted_confidence))
        report.passed = len(report.issues) == 0
        report.duration_ms = (time.perf_counter() - start) * 1000

        structured_log(logging.DEBUG, "validation.fact.complete",
                       fact_id=fact.id,
                       passed=report.passed,
                       issues=len(report.issues),
                       confidence_before=round(report.original_confidence, 3),
                       confidence_after=round(report.adjusted_confidence, 3))

        return report

    async def validate_fact_batch(
        self,
        facts: list[Fact],
        store: FactStore | None = None,
    ) -> list[ValidationReport]:
        reports = []
        for fact in facts:
            report = await self.validate_fact(fact, store)
            reports.append(report)
        return reports

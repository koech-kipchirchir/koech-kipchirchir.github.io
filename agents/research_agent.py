"""
AIOS Research Agent
===================

Production-grade research agent with capabilities for information retrieval,
document summarization, source comparison, report generation, citation
management, and research memory.

Actions (auto-detected from task description):

- ``retrieve`` --- Fetch information from web, files, or API sources
- ``summarize`` --- Condense a document or findings into a structured summary
- ``compare`` / ``cross-reference`` --- Compare multiple sources for agreement/contradiction
- ``report`` --- Build a structured research report with sections and citations
- ``cite`` / ``citation`` --- Format citations in MLA, APA, Chicago, or IEEE style
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from agents.base_agent import AgentConfig, AgentResult, BaseAgent

logger = logging.getLogger("aios.agent.research")

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CitationStyle(Enum):
    MLA = "mla"
    APA = "apa"
    CHICAGO = "chicago"
    IEEE = "ieee"


class SourceType(Enum):
    WEB = "web"
    PDF = "pdf"
    DOCUMENT = "document"
    API = "api"
    FILE = "file"
    BOOK = "book"
    ARTICLE = "article"
    REPORT = "report"
    UNKNOWN = "unknown"


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNVERIFIED = "unverified"


class ComparisonResult(Enum):
    AGREEMENT = "agreement"
    CONTRADICTION = "contradiction"
    PARTIAL_AGREEMENT = "partial_agreement"
    UNIQUE = "unique"
    UNRELATED = "unrelated"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CitationInfo:
    """Structured source citation."""

    id: str = ""
    title: str = ""
    author: str = ""
    publication: str = ""
    year: str = ""
    url: str = ""
    source_type: SourceType = SourceType.UNKNOWN
    accessed_date: str = ""
    publisher: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def format_mla(self) -> str:
        parts: list[str] = []
        if self.author:
            parts.append(f"{self.author}.")
        if self.title:
            parts.append(f'"{self.title}."')
        if self.publication:
            parts.append(f"{self.publication},")
        if self.publisher:
            parts.append(f"{self.publisher},")
        if self.year:
            parts.append(f"{self.year}.")
        if self.url:
            parts.append(self.url)
        return " ".join(parts)

    def format_apa(self) -> str:
        parts: list[str] = []
        if self.author:
            parts.append(f"{self.author}.")
        if self.year:
            parts.append(f"({self.year}).")
        if self.title:
            parts.append(f"{self.title}.")
        if self.publication:
            parts.append(f"*{self.publication}*,")
        if self.volume:
            parts.append(f"*{self.volume}*")
        if self.issue:
            parts.append(f"({self.issue}),")
        if self.pages:
            parts.append(f"{self.pages}.")
        if self.url:
            parts.append(self.url)
        return " ".join(parts)

    def format_chicago(self) -> str:
        parts: list[str] = []
        if self.author:
            parts.append(f"{self.author}.")
        if self.title:
            parts.append(f'"{self.title}."')
        if self.publication:
            parts.append(f"{self.publication}")
        if self.publisher:
            parts.append(f"({self.publisher}")
            if self.year:
                parts.append(f"{self.year})")
            else:
                parts[-1] += ")"
        elif self.year:
            parts.append(f"({self.year})")
        if self.url:
            parts.append(self.url)
        return " ".join(parts)

    def format_ieee(self) -> str:
        parts: list[str] = []
        if self.author:
            parts.append(f"{self.author},")
        if self.title:
            parts.append(f'"{self.title},"')
        if self.publication:
            parts.append(f"{self.publication},")
        if self.volume:
            parts.append(f"vol. {self.volume},")
        if self.issue:
            parts.append(f"no. {self.issue},")
        if self.pages:
            parts.append(f"pp. {self.pages},")
        if self.year:
            parts.append(f"{self.year}.")
        if self.url:
            parts.append(f"[Online]. Available: {self.url}")
        return " ".join(parts)

    def format(self, style: CitationStyle = CitationStyle.APA) -> str:
        fmt_map = {
            CitationStyle.MLA: self.format_mla,
            CitationStyle.APA: self.format_apa,
            CitationStyle.CHICAGO: self.format_chicago,
            CitationStyle.IEEE: self.format_ieee,
        }
        return fmt_map.get(style, self.format_apa)()


@dataclass
class SourceInfo:
    """Information about a research source."""

    id: str = ""
    title: str = ""
    content: str = ""
    source_type: SourceType = SourceType.UNKNOWN
    url: str = ""
    path: str = ""
    citation: CitationInfo = field(default_factory=CitationInfo)
    relevance_score: float = 0.0
    confidence: Confidence = Confidence.UNVERIFIED
    retrieved_at: str = ""
    word_count: int = 0
    language: str = "en"
    tags: list[str] = field(default_factory=list)
    summary: str = ""
    key_claims: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchFinding:
    """A single finding or claim gathered during research."""

    id: str = ""
    claim: str = ""
    source_ids: list[str] = field(default_factory=list)
    confidence: Confidence = Confidence.UNVERIFIED
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    counter_evidence: list[str] = field(default_factory=list)


@dataclass
class SourceComparison:
    """Result of comparing two or more sources."""

    source_a_id: str = ""
    source_b_id: str = ""
    result: ComparisonResult = ComparisonResult.UNRELATED
    agreement_points: list[str] = field(default_factory=list)
    contradiction_points: list[str] = field(default_factory=list)
    unique_points_a: list[str] = field(default_factory=list)
    unique_points_b: list[str] = field(default_factory=list)
    overlap_percentage: float = 0.0


@dataclass
class ResearchSession:
    """Persistent research session memory."""

    id: str = ""
    topic: str = ""
    created_at: str = ""
    updated_at: str = ""
    sources: list[SourceInfo] = field(default_factory=list)
    findings: list[ResearchFinding] = field(default_factory=list)
    comparisons: list[SourceComparison] = field(default_factory=list)
    report: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SummaryResult:
    """Structured summarization output."""

    title: str = ""
    original_word_count: int = 0
    summary_word_count: int = 0
    key_points: list[str] = field(default_factory=list)
    main_topic: str = ""
    summary: str = ""
    compression_ratio: float = 0.0


# ---------------------------------------------------------------------------
# In-memory research store
# ---------------------------------------------------------------------------

_research_memory: dict[str, ResearchSession] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Action detection
# ---------------------------------------------------------------------------


def _detect_research_action(task: str) -> str:
    """Map task description to a ResearchAgent action."""
    tl = task.lower()
    # Check more specific phrases first
    if any(w in tl for w in ["cross-reference", "cross reference"]):
        return "compare"
    if any(w in tl for w in ["write report", "build report", "research paper", "works cited"]):
        return "report"
    if any(w in tl for w in ["look up", "get information"]):
        return "retrieve"
    if any(w in tl for w in ["summarize", "summary", "condense", "brief", "tl;dr", "digest"]):
        return "summarize"
    if any(w in tl for w in ["compare", "contrast", "synthesize", "cross-ref"]):
        return "compare"
    if any(w in tl for w in ["report", "draft"]):
        return "report"
    if any(w in tl for w in ["cite", "citation", "reference", "bibliography"]):
        return "cite"
    # Single-word checks (word-boundary sensitive)
    if "retrieve" in tl or "fetch" in tl or "search" in tl or "gather" in tl:
        return "retrieve"
    # 'find' as a standalone word (not part of 'findings')
    for w in tl.split():
        if w in ("find", "finding"):
            return "retrieve"
    return "retrieve"


# ---------------------------------------------------------------------------
# Citation generation
# ---------------------------------------------------------------------------


def _build_citation(
    title: str = "",
    author: str = "",
    source_type: SourceType = SourceType.UNKNOWN,
    url: str = "",
    publication: str = "",
    year: str = "",
    publisher: str = "",
    volume: str = "",
    issue: str = "",
    pages: str = "",
    doi: str = "",
    extra: dict[str, Any] | None = None,
) -> CitationInfo:
    """Build a CitationInfo with auto-generated ID."""
    return CitationInfo(
        id=_generate_id(),
        title=title,
        author=author,
        source_type=source_type,
        url=url,
        publication=publication,
        year=year,
        publisher=publisher,
        volume=volume,
        issue=issue,
        pages=pages,
        doi=doi,
        accessed_date=_now()[:10],
        extra=extra or {},
    )


# ---------------------------------------------------------------------------
# Source text processing
# ---------------------------------------------------------------------------


def _count_words(text: str) -> int:
    return len(text.split())


def _extract_key_claims(text: str, max_claims: int = 10) -> list[str]:
    """Extract key claims from text using sentence-level heuristics."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if any(kw in s.lower() for kw in
               ["demonstrate", "show", "find", "suggest", "indicate", "reveal",
                "conclude", "propose", "argue", "claim", "evidence", "prove",
                "confirm", "establish", "identify"]):
            claims.append(s)
        if len(claims) >= max_claims:
            break
    if not claims and sentences:
        claims = [s for s in sentences if len(s) > 40][:max_claims]
    return claims


# ---------------------------------------------------------------------------
# Source comparisons
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Return a set of lowercased alphanumeric tokens."""
    return set(re.findall(r"[a-z0-9]\w+", text.lower()))


def _compare_sources(source_a: SourceInfo, source_b: SourceInfo) -> SourceComparison:
    """Compare two sources for agreement, contradiction, and unique claims."""
    comparison = SourceComparison(
        source_a_id=source_a.id,
        source_b_id=source_b.id,
    )

    tokens_a = _tokenize(source_a.content)
    tokens_b = _tokenize(source_b.content)

    if not tokens_a or not tokens_b:
        comparison.result = ComparisonResult.UNRELATED
        return comparison

    overlap = tokens_a & tokens_b
    union = tokens_a | tokens_b
    overlap_pct = len(overlap) / len(union) if union else 0.0
    comparison.overlap_percentage = round(overlap_pct * 100, 1)

    # Classify based on overlap
    if overlap_pct > 0.6:
        comparison.result = ComparisonResult.AGREEMENT
    elif overlap_pct > 0.3:
        comparison.result = ComparisonResult.PARTIAL_AGREEMENT
    elif overlap_pct > 0.05:
        comparison.result = ComparisonResult.CONTRADICTION
    else:
        comparison.result = ComparisonResult.UNIQUE

    # Extract shared claims
    shared_claims = []
    for claim_a in source_a.key_claims:
        claim_tokens = _tokenize(claim_a)
        if claim_tokens and len(claim_tokens & tokens_b) / len(claim_tokens) > 0.4:
            shared_claims.append(claim_a)
    comparison.agreement_points = shared_claims[:5]

    # Unique to A
    for claim_a in source_a.key_claims:
        if claim_a not in comparison.agreement_points:
            comparison.unique_points_a.append(claim_a)

    # Unique to B
    for claim_b in source_b.key_claims:
        claim_tokens = _tokenize(claim_b)
        if claim_tokens and len(claim_tokens & tokens_a) / len(claim_tokens) < 0.3:
            comparison.unique_points_b.append(claim_b)

    return comparison


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _build_report(
    topic: str,
    sources: list[SourceInfo],
    findings: list[ResearchFinding],
    comparisons: list[SourceComparison] | None = None,
    depth: str = "standard",
    citation_style: CitationStyle = CitationStyle.APA,
    include_toc: bool = True,
) -> str:
    """Build a structured markdown research report."""
    lines: list[str] = []
    lines.append(f"# Research Report: {topic}")
    lines.append("")
    lines.append(f"> Generated: {_now()}")
    lines.append(f"> Depth: {depth}")
    lines.append(f"> Sources: {len(sources)}")
    lines.append(f"> Findings: {len(findings)}")

    if include_toc:
        lines.append("")
        lines.append("## Table of Contents")
        lines.append("")
        lines.append("1. [Executive Summary](#executive-summary)")
        lines.append("2. [Key Findings](#key-findings)")
        lines.append("3. [Source Analysis](#source-analysis)")
        if comparisons:
            lines.append("4. [Source Comparisons](#source-comparisons)")
            lines.append("5. [Conclusions](#conclusions)")
            lines.append("6. [References](#references)")
        else:
            lines.append("4. [Conclusions](#conclusions)")
            lines.append("5. [References](#references)")

    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")

    # Build summary from findings
    high_conf = [f for f in findings if f.confidence == Confidence.HIGH]
    medium_conf = [f for f in findings if f.confidence in (Confidence.MEDIUM,)]

    if high_conf:
        lines.append("High-confidence findings:")
        for f in high_conf:
            lines.append(f"- {f.claim}")
    if medium_conf:
        lines.append("Medium-confidence findings:")
        for f in medium_conf:
            lines.append(f"- {f.claim}")
    if not findings:
        lines.append("No findings recorded for this research session.")

    lines.append("")
    lines.append("## Key Findings")
    lines.append("")

    if findings:
        for i, f in enumerate(findings, 1):
            confidence_badge = {
                Confidence.HIGH: "[GREEN] High",
                Confidence.MEDIUM: "[YELLOW] Medium",
                Confidence.LOW: "[RED] Low",
                Confidence.UNVERIFIED: "[GRAY] Unverified",
            }.get(f.confidence, "[GRAY] Unverified")
            lines.append(f"### Finding {i}: {f.claim}")
            lines.append("")
            lines.append(f"- **Confidence:** {confidence_badge}")
            lines.append(f"- **Category:** {f.category}")
            if f.tags:
                lines.append(f"- **Tags:** `{'`, `'.join(f.tags)}`")
            if f.source_ids:
                src_refs = [f"[{sid}](#source-{sid})" for sid in f.source_ids[:5]]
                sources_str = ", ".join(src_refs)
                lines.append(f"- **Sources:** {sources_str}")
            if f.supporting_evidence:
                lines.append("- **Supporting:**")
                for ev in f.supporting_evidence[:3]:
                    lines.append(f"  - {ev}")
            if f.counter_evidence:
                lines.append("- **Contradicting:**")
                for ev in f.counter_evidence[:3]:
                    lines.append(f"  - {ev}")
            lines.append("")
    else:
        lines.append("_No structured findings extracted._")

    lines.append("## Source Analysis")
    lines.append("")

    for src in sources:
        anchor = f"source-{src.id}"
        lines.append(f"### <a id='{anchor}'></a>{src.title or 'Untitled'}")
        lines.append("")
        lines.append(f"- **Type:** {src.source_type.value}")
        lines.append(f"- **Confidence:** {src.confidence.value}")
        lines.append(f"- **Relevance:** {src.relevance_score:.0%}")
        lines.append(f"- **Words:** {src.word_count:,}")
        if src.url:
            lines.append(f"- **URL:** {src.url}")
        if src.summary:
            lines.append("")
            lines.append(f"  > {src.summary}")
        if src.key_claims:
            lines.append("")
            lines.append("  Key claims:")
            for claim in src.key_claims[:5]:
                lines.append(f"  - {claim}")
        lines.append("")

    if comparisons:
        lines.append("## Source Comparisons")
        lines.append("")

        for comp in comparisons:
            src_a = next((s for s in sources if s.id == comp.source_a_id), None)
            src_b = next((s for s in sources if s.id == comp.source_b_id), None)
            name_a = src_a.title if src_a else comp.source_a_id
            name_b = src_b.title if src_b else comp.source_b_id

            result_labels = {
                ComparisonResult.AGREEMENT: "Agreement",
                ComparisonResult.CONTRADICTION: "Contradiction",
                ComparisonResult.PARTIAL_AGREEMENT: "Partial Agreement",
                ComparisonResult.UNIQUE: "Unique Perspectives",
                ComparisonResult.UNRELATED: "Unrelated",
            }
            badge = result_labels.get(comp.result, "Unknown")
            lines.append(f"### {name_a} vs {name_b}")
            lines.append("")
            lines.append(f"- **Result:** {badge}")
            lines.append(f"- **Content overlap:** {comp.overlap_percentage}%")
            if comp.agreement_points:
                lines.append("- **Agreements:**")
                for pt in comp.agreement_points[:3]:
                    lines.append(f"  - ✓ {pt}")
            if comp.contradiction_points:
                lines.append("- **Contradictions:**")
                for pt in comp.contradiction_points[:3]:
                    lines.append(f"  - ✗ {pt}")
            if comp.unique_points_a:
                lines.append(f"- **Unique to {name_a}:**")
                for pt in comp.unique_points_a[:3]:
                    lines.append(f"  - {pt}")
            if comp.unique_points_b:
                lines.append(f"- **Unique to {name_b}:**")
                for pt in comp.unique_points_b[:3]:
                    lines.append(f"  - {pt}")
            lines.append("")

    lines.append("## Conclusions")
    lines.append("")

    high_findings = [f for f in findings if f.confidence == Confidence.HIGH]
    if high_findings:
        lines.append("Based on the evidence gathered:")
        for f in high_findings:
            lines.append(f"1. **{f.claim}** --- supported by {len(f.source_ids)} source(s).")
    else:
        lines.append("Insufficient high-confidence evidence to draw firm conclusions.")
        if findings:
            lines.append("Further investigation is recommended.")

    lines.append("")
    lines.append("## References")
    lines.append("")

    if sources:
        for i, src in enumerate(sources, 1):
            citation_text = src.citation.format(citation_style) if src.citation else src.title
            lines.append(f"[{i}] {citation_text}")
    else:
        lines.append("_No sources cited._")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summarization engine
# ---------------------------------------------------------------------------


def _summarize_text(text: str, title: str = "", max_points: int = 8, max_summary_words: int = 200) -> SummaryResult:
    """Summarize text using sentence extraction heuristics."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    original_wc = _count_words(text)

    if not sentences:
        return SummaryResult(
            title=title, original_word_count=original_wc,
            summary="", summary_word_count=0,
        )

    # Score sentences by position, length, and keyword presence
    research_kw = {"find", "show", "result", "demonstrate", "evidence", "conclude",
                    "analysis", "suggest", "significant", "key", "important",
                    "discover", "reveal", "impact", "effect", "study", "research",
                    "data", "experiment", "observe", "measure"}
    scored: list[tuple[float, str]] = []

    for i, s in enumerate(sentences):
        score = 0.0
        # Position bonus: first and last sentences
        if i == 0:
            score += 3.0
        elif i == len(sentences) - 1:
            score += 2.0
        # Length penalty for very short/long
        wc = _count_words(s)
        if wc < 5:
            score -= 2.0
        elif wc > 60:
            score -= 0.5
        elif 10 <= wc <= 40:
            score += 1.0
        # Keyword bonus
        words_lower = set(s.lower().split())
        kw_matches = len(words_lower & research_kw)
        score += kw_matches * 0.5
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Select top sentences up to max_summary_words
    selected: list[str] = []
    wc_sofar = 0
    for _, s in scored:
        s_wc = _count_words(s)
        if wc_sofar + s_wc > max_summary_words:
            continue
        selected.append(s)
        wc_sofar += s_wc

    selected.sort(key=lambda s: sentences.index(s) if s in sentences else 0)
    summary = " ".join(selected)

    # Key points
    key_points = [s for _, s in scored[:max_points]]

    return SummaryResult(
        title=title,
        original_word_count=original_wc,
        summary_word_count=_count_words(summary),
        key_points=key_points,
        main_topic=title or "Untitled",
        summary=summary,
        compression_ratio=round(1 - (_count_words(summary) / original_wc), 2) if original_wc else 0.0,
    )


# ---------------------------------------------------------------------------
# Mock retrieval (expand to real HTTP fetch in production)
# ---------------------------------------------------------------------------


async def _retrieve_web(url: str, timeout: float = 10.0) -> str:
    """Simulate web retrieval. In production, replace with aiohttp."""
    logger.info("Retrieving from (mock): %s", url)
    await asyncio.sleep(0.1)
    return (
        f"Simulated content retrieved from {url}. "
        "In a production environment, replace this function with an actual "
        "HTTP client (e.g., aiohttp or httpx) to fetch real web pages. "
        "This mock returns placeholder text to enable offline testing and "
        "development without network dependencies."
    )


async def _retrieve_file(path: str) -> str:
    """Read content from a local file."""
    logger.info("Reading file: %s", path)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------


class ResearchAgent(BaseAgent):
    """Production-grade research agent with retrieval, summarization,
    comparison, citation, report generation, and research memory."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__(config or AgentConfig(
            name="research",
            system_prompt=(
                "You are a research assistant. You gather, synthesize, and present "
                "information from multiple sources. You provide properly formatted "
                "citations, assess confidence levels for each claim, compare sources "
                "for agreement and contradiction, and build comprehensive research "
                "reports. You maintain research memory across sessions."
            ),
        ))

    async def execute(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        start = time.perf_counter()
        ctx = context or {}

        action = _detect_research_action(task)

        try:
            action_map: dict[str, Any] = {
                "retrieve": self._action_retrieve,
                "summarize": self._action_summarize,
                "compare": self._action_compare,
                "report": self._action_report,
                "cite": self._action_cite,
            }

            handler = action_map.get(action)
            if handler is None:
                return AgentResult(
                    success=False, output="", agent_name=self.name,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    error=f"Unknown action: {action}",
                )

            output = await handler(task, ctx)
            duration = (time.perf_counter() - start) * 1000

            return AgentResult(
                success=True, output=output, agent_name=self.name,
                duration_ms=duration,
                metadata={
                    "action": action,
                    "session_id": ctx.get("session_id", ""),
                    "source_count": len(ctx.get("sources", [])),
                },
            )

        except Exception as exc:
            duration = (time.perf_counter() - start) * 1000
            self._logger.exception("Research action failed: %s", exc)
            return AgentResult(
                success=False, output="", agent_name=self.name,
                duration_ms=duration, error=str(exc),
            )

    # ------------------------------------------------------------------
    # Research memory
    # ------------------------------------------------------------------

    def create_session(self, topic: str, tags: list[str] | None = None) -> ResearchSession:
        session = ResearchSession(
            id=_generate_id(),
            topic=topic,
            created_at=_now(),
            updated_at=_now(),
            tags=tags or [],
        )
        _research_memory[session.id] = session
        return session

    def get_session(self, session_id: str) -> ResearchSession | None:
        return _research_memory.get(session_id)

    def list_sessions(self, topic_filter: str = "") -> list[ResearchSession]:
        sessions = list(_research_memory.values())
        if topic_filter:
            tl = topic_filter.lower()
            sessions = [s for s in sessions if tl in s.topic.lower()]
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        return _research_memory.pop(session_id, None) is not None

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    async def _action_retrieve(self, task: str, ctx: dict[str, Any]) -> str:
        """Retrieve information from sources and return structured findings."""
        sources_raw: list[dict[str, Any]] = ctx.get("sources", [])
        urls: list[str] = ctx.get("urls", [])
        paths: list[str] = ctx.get("paths", [])
        session_id: str | None = ctx.get("session_id")

        # Resolve sources from context
        resolved_sources: list[SourceInfo] = []

        # From explicit source dicts
        for src in sources_raw:
            s = SourceInfo(
                id=src.get("id", _generate_id()),
                title=src.get("title", ""),
                content=src.get("content", ""),
                url=src.get("url", ""),
                source_type=SourceType(src.get("type", "unknown")) if src.get("type") else SourceType.UNKNOWN,
            )
            s.word_count = _count_words(s.content)
            s.key_claims = _extract_key_claims(s.content)
            resolved_sources.append(s)

        # From URLs
        for url in urls:
            content = await _retrieve_web(url)
            s = SourceInfo(
                id=_generate_id(),
                title=url,
                content=content,
                url=url,
                source_type=SourceType.WEB,
                citation=_build_citation(title=url, url=url, source_type=SourceType.WEB),
            )
            s.word_count = _count_words(content)
            s.key_claims = _extract_key_claims(content)
            resolved_sources.append(s)

        # From file paths
        for path in paths:
            content = await _retrieve_file(path)
            title = Path(path).name
            s = SourceInfo(
                id=_generate_id(),
                title=title,
                content=content,
                path=path,
                source_type=SourceType.FILE,
            )
            s.word_count = _count_words(content)
            s.key_claims = _extract_key_claims(content)
            resolved_sources.append(s)

        if not resolved_sources:
            return "No sources provided. Pass `sources`, `urls`, or `paths` in context."

        # Build output
        lines: list[str] = []
        lines.append(f"## Retrieved Sources ({len(resolved_sources)})")
        lines.append("")

        for s in resolved_sources:
            lines.append(f"### {s.title or 'Untitled Source'} (`{s.id}`)")
            lines.append("")
            lines.append(f"- **Type:** {s.source_type.value}")
            lines.append(f"- **Words:** {s.word_count:,}")
            if s.url:
                lines.append(f"- **URL:** {s.url}")
            if s.path:
                lines.append(f"- **Path:** `{s.path}`")
            # Content preview
            if s.content:
                preview = s.content[:150].replace("\n", " ")
                lines.append(f"- **Preview:** {preview}...")
            if s.key_claims:
                lines.append("")
                lines.append("**Key claims:**")
                for claim in s.key_claims[:5]:
                    lines.append(f"- {claim}")
            lines.append("")

        # Store in session memory if provided
        if session_id:
            session = _research_memory.get(session_id)
            if session:
                session.sources.extend(resolved_sources)
                session.updated_at = _now()

        return "\n".join(lines)

    async def _action_summarize(self, task: str, ctx: dict[str, Any]) -> str:
        """Summarize a document or research findings."""
        content: str = ctx.get("content", "")
        source_id: str | None = ctx.get("source_id")
        session_id: str | None = ctx.get("session_id")

        # Try to get content from session memory
        resolved_title = ""
        if not content and source_id and session_id:
            session = _research_memory.get(session_id)
            if session:
                src = next((s for s in session.sources if s.id == source_id), None)
                if src:
                    content = src.content
                    resolved_title = src.title

        if not content:
            return "No content provided. Pass `content` or `source_id` with `session_id` in context."

        title = ctx.get("title", resolved_title or "Untitled")
        max_points = ctx.get("max_points", 8)
        max_words = ctx.get("max_summary_words", 200)

        result = _summarize_text(content, title=title, max_points=max_points, max_summary_words=max_words)

        lines: list[str] = []
        lines.append(f"## Summary: {result.title}")
        lines.append("")
        lines.append(f"- **Original words:** {result.original_word_count:,}")
        lines.append(f"- **Summary words:** {result.summary_word_count:,}")
        lines.append(f"- **Compression ratio:** {result.compression_ratio:.0%}")
        lines.append("")

        if result.key_points:
            lines.append("**Key Points:**")
            for i, kp in enumerate(result.key_points, 1):
                lines.append(f"{i}. {kp}")
            lines.append("")

        lines.append("**Summary:**")
        lines.append("")
        lines.append(result.summary)

        # Store in session memory
        if session_id:
            session = _research_memory.get(session_id)
            if session:
                # Update source summary
                for src in session.sources:
                    if src.title == title or src.id == source_id:
                        src.summary = result.summary
                session.updated_at = _now()

        return "\n".join(lines)

    async def _action_compare(self, task: str, ctx: dict[str, Any]) -> str:
        """Compare multiple sources and report agreement/contradiction."""
        source_ids: list[str] = ctx.get("source_ids", [])
        session_id: str | None = ctx.get("session_id")

        if not source_ids and not session_id:
            return "No sources to compare. Pass `source_ids` with `session_id` in context."

        # Resolve sources
        sources: list[SourceInfo] = []
        if session_id:
            session = _research_memory.get(session_id)
            if session:
                if source_ids:
                    sources = [s for s in session.sources if s.id in source_ids]
                else:
                    sources = session.sources

        if len(sources) < 2:
            return f"Need at least 2 sources to compare (found {len(sources)})."

        lines: list[str] = []
        lines.append(f"## Source Comparison ({len(sources)} sources)")
        lines.append("")

        comparisons: list[SourceComparison] = []
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                comp = _compare_sources(sources[i], sources[j])
                comparisons.append(comp)

                name_a = sources[i].title or sources[i].id
                name_b = sources[j].title or sources[j].id

                result_labels = {
                    ComparisonResult.AGREEMENT: "[OK] Agreement",
                    ComparisonResult.CONTRADICTION: "[FAIL] Contradiction",
                    ComparisonResult.PARTIAL_AGREEMENT: "[WARN] Partial Agreement",
                    ComparisonResult.UNIQUE: "[NEW] Unique Perspectives",
                    ComparisonResult.UNRELATED: "[SKIP] Unrelated",
                }

                lines.append(f"### {name_a} vs {name_b}")
                lines.append("")
                lines.append(f"- **Result:** {result_labels.get(comp.result, 'Unknown')}")
                lines.append(f"- **Content overlap:** {comp.overlap_percentage}%")
                if comp.agreement_points:
                    lines.append("- **Shared claims:**")
                    for pt in comp.agreement_points[:3]:
                        lines.append(f"  - ✓ {pt}")
                if comp.contradiction_points:
                    lines.append("- **Contradictions:**")
                    for pt in comp.contradiction_points[:3]:
                        lines.append(f"  - ✗ {pt}")
                if comp.unique_points_a:
                    lines.append(f"- **Unique to {name_a}:**")
                    for pt in comp.unique_points_a[:3]:
                        lines.append(f"  - {pt}")
                if comp.unique_points_b:
                    lines.append(f"- **Unique to {name_b}:**")
                    for pt in comp.unique_points_b[:3]:
                        lines.append(f"  - {pt}")
                lines.append("")

        # Store comparisons in session
        if session_id:
            session = _research_memory.get(session_id)
            if session:
                session.comparisons.extend(comparisons)
                session.updated_at = _now()

        return "\n".join(lines)

    async def _action_report(self, task: str, ctx: dict[str, Any]) -> str:
        """Generate a structured research report."""
        session_id: str | None = ctx.get("session_id")
        topic: str = ctx.get("topic", task)
        depth: str = ctx.get("depth", "standard")
        style_name: str = ctx.get("citation_style", "apa")
        include_toc: bool = ctx.get("include_toc", True)

        try:
            style = CitationStyle(style_name.lower())
        except ValueError:
            style = CitationStyle.APA

        # Resolve from session
        sources: list[SourceInfo] = []
        findings: list[ResearchFinding] = []
        comparisons: list[SourceComparison] = []

        if session_id:
            session = _research_memory.get(session_id)
            if session:
                sources = session.sources
                findings = session.findings
                comparisons = session.comparisons

        # Also accept inline overrides
        sources = ctx.get("sources", sources)
        findings = ctx.get("findings", findings)

        # If no session and no sources, build a placeholder
        if not sources and not findings:
            # Create a minimal exploration report
            content = ctx.get("content", "")
            if content:
                s = SourceInfo(
                    id=_generate_id(),
                    title=topic,
                    content=content,
                    word_count=_count_words(content),
                    key_claims=_extract_key_claims(content),
                )
                sources = [s]
            else:
                return (
                    "No research data available. Provide `session_id` or `sources` "
                    "in context to generate a report."
                )

        report = _build_report(
            topic=topic,
            sources=sources,
            findings=findings,
            comparisons=comparisons if comparisons else None,
            depth=depth,
            citation_style=style,
            include_toc=include_toc,
        )

        # Store report in session
        if session_id:
            session = _research_memory.get(session_id)
            if session:
                session.report = report
                session.updated_at = _now()

        return report

    async def _action_cite(self, task: str, ctx: dict[str, Any]) -> str:
        """Format citations in the requested style."""
        style_name: str = ctx.get("citation_style", "apa")
        source_ids: list[str] = ctx.get("source_ids", [])
        session_id: str | None = ctx.get("session_id")

        try:
            style = CitationStyle(style_name.lower())
        except ValueError:
            style = CitationStyle.APA

        # Build citations from context or session
        citations: list[CitationInfo] = []

        # From explicit citation dicts
        for cit_dict in ctx.get("citations", []):
            c = _build_citation(
                title=cit_dict.get("title", ""),
                author=cit_dict.get("author", ""),
                source_type=SourceType(cit_dict.get("type", "unknown")) if cit_dict.get("type") else SourceType.UNKNOWN,
                url=cit_dict.get("url", ""),
                publication=cit_dict.get("publication", ""),
                year=cit_dict.get("year", ""),
                publisher=cit_dict.get("publisher", ""),
                volume=cit_dict.get("volume", ""),
                issue=cit_dict.get("issue", ""),
                pages=cit_dict.get("pages", ""),
                doi=cit_dict.get("doi", ""),
            )
            citations.append(c)

        # From session sources
        if session_id:
            session = _research_memory.get(session_id)
            if session:
                srcs = session.sources
                if source_ids:
                    srcs = [s for s in srcs if s.id in source_ids]
                for src in srcs:
                    if src.citation and src.citation.title:
                        citations.append(src.citation)

        if not citations:
            return "No citations to format. Pass `citations`, `source_ids`, or `session_id` in context."

        lines: list[str] = []
        style_name_upper = style.name
        lines.append(f"## Citations ({style_name_upper})")
        lines.append("")

        for i, c in enumerate(citations, 1):
            formatted = c.format(style)
            lines.append(f"[{i}] {formatted}")

        lines.append("")
        lines.append(f"---")
        lines.append(f"Style: {style_name_upper} | Sources: {len(citations)} | Generated: {_now()[:10]}")

        return "\n".join(lines)

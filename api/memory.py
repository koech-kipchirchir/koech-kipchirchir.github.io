from __future__ import annotations

import logging
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, Field

from api.dependencies import get_config
from api.exceptions import EngineError, NotFoundError, ServiceUnavailableError

logger = logging.getLogger("aios.api.memory")

router = APIRouter(prefix="/memory", tags=["Memory"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class StoreMemoryRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Memory content")
    role: str = Field("user", description="Message role (user, assistant, system)")
    session_id: str | None = Field(None, description="Session ID (auto-generated if omitted)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class MemoryMessageSchema(BaseModel):
    message_id: str = Field("", description="Unique message ID")
    role: str = Field("", description="Message role")
    content: str = Field("", description="Message content")
    timestamp: str = Field("", description="ISO-8601 timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryNodeSchema(BaseModel):
    node_id: str = Field("", description="Unique node ID")
    content: str = Field("", description="Memory content")
    importance: float = Field(0.0, description="Importance score (0-1)")
    summary: str = Field("", description="Optional summary")
    session_id: str = Field("", description="Associated session")
    created_at: str = Field("", description="ISO-8601 timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchMemoryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    session_id: str | None = Field(None, description="Scope to session")
    top_k: int = Field(10, ge=1, le=100, description="Max results")
    filters: dict[str, Any] | None = Field(None, description="Metadata filters")


class SearchResultSchema(BaseModel):
    content: str = Field("", description="Matched content")
    score: float = Field(0.0, description="Relevance score")
    source: str = Field("", description="Source: conversation, long_term, vector")
    metadata: dict[str, Any] = Field(default_factory=dict)
    node_id: str = Field("", description="Node ID if from long_term/vector")


class MemoryListResponse(BaseModel):
    session_id: str = Field("", description="Session ID")
    messages: list[MemoryMessageSchema] = Field(default_factory=list, description="Conversation messages")
    long_term_memories: list[MemoryNodeSchema] = Field(default_factory=list, description="Long-term memory nodes")
    total_messages: int = Field(0, description="Number of conversation messages")
    total_long_term: int = Field(0, description="Number of long-term memories")


class MemoryStoreResponse(BaseModel):
    status: str = Field("ok", description="Operation status")
    session_id: str = Field("", description="Session ID")
    message: MemoryMessageSchema | None = Field(None, description="Stored message")


class MemoryDeleteResponse(BaseModel):
    status: str = Field("ok", description="Operation status")
    session_id: str = Field("", description="Session ID")
    deleted: bool = Field(True, description="Whether anything was deleted")


class SearchMemoryResponse(BaseModel):
    results: list[SearchResultSchema] = Field(default_factory=list, description="Search results")
    session_id: str = Field("", description="Session ID")
    total: int = Field(0, description="Number of results")


class SummarizeResponse(BaseModel):
    summary: str = Field("", description="Generated summary")
    session_id: str = Field("", description="Session ID")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_memory_manager(request: Request):
    mm = getattr(request.app.state, "memory_manager", None)
    if mm is None:
        raise ServiceUnavailableError(detail="Memory manager not initialized")
    return mm


def _resolve_session_id(
    session_id_body: str | None = None,
    session_id_header: str | None = None,
    session_id_query: str | None = None,
) -> str:
    sid = session_id_body or session_id_header or session_id_query
    if sid:
        return sid
    return uuid4().hex[:16]


def _message_to_schema(msg) -> MemoryMessageSchema:
    if hasattr(msg, "to_dict"):
        d = msg.to_dict()
    elif isinstance(msg, dict):
        d = msg
    else:
        d = {"message_id": "", "role": "", "content": str(msg), "timestamp": "", "metadata": {}}
    return MemoryMessageSchema(
        message_id=d.get("message_id", ""),
        role=d.get("role", ""),
        content=d.get("content", ""),
        timestamp=str(d.get("timestamp", "")),
        metadata=d.get("metadata", {}),
    )


def _node_to_schema(node) -> MemoryNodeSchema:
    if hasattr(node, "to_dict"):
        d = node.to_dict()
    elif isinstance(node, dict):
        d = node
    else:
        d = {"node_id": "", "content": "", "importance": 0.0, "summary": "", "session_id": "", "created_at": "", "metadata": {}}
    return MemoryNodeSchema(
        node_id=d.get("node_id", ""),
        content=d.get("content", ""),
        importance=float(d.get("importance", 0.0)),
        summary=d.get("summary", ""),
        session_id=d.get("session_id", ""),
        created_at=str(d.get("created_at", "")),
        metadata=d.get("metadata", {}),
    )


def _result_to_schema(r) -> SearchResultSchema:
    if hasattr(r, "__dataclass_fields__"):
        return SearchResultSchema(
            content=getattr(r, "content", ""),
            score=float(getattr(r, "score", 0.0)),
            source=getattr(r, "source", ""),
            metadata=getattr(r, "metadata", {}),
            node_id=getattr(r, "node_id", ""),
        )
    if isinstance(r, dict):
        return SearchResultSchema(
            content=r.get("content", ""),
            score=float(r.get("score", 0.0)),
            source=r.get("source", ""),
            metadata=r.get("metadata", {}),
            node_id=r.get("node_id", ""),
        )
    return SearchResultSchema(content=str(r), score=0.0, source="")


# ---------------------------------------------------------------------------
# POST /memory  — Store a memory
# ---------------------------------------------------------------------------

@router.post("", summary="Store a memory", status_code=201)
async def store_memory(
    body: StoreMemoryRequest,
    request: Request,
    x_session_id: Annotated[str | None, Header()] = None,
    session_id: Annotated[str | None, Query()] = None,
) -> MemoryStoreResponse:
    mm = _get_memory_manager(request)
    sid = _resolve_session_id(body.session_id, x_session_id, session_id)

    try:
        msg = mm.add_message(role=body.role, content=body.content, session_id=sid, metadata=body.metadata)
    except Exception as exc:
        logger.error("Failed to store memory: %s", exc)
        raise EngineError(detail=str(exc))

    return MemoryStoreResponse(
        status="ok",
        session_id=sid,
        message=_message_to_schema(msg) if msg else None,
    )


# ---------------------------------------------------------------------------
# GET /memory  — Retrieve memories
# ---------------------------------------------------------------------------

@router.get("", summary="Retrieve memories for a session")
async def list_memories(
    request: Request,
    x_session_id: Annotated[str | None, Header()] = None,
    session_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    include_long_term: Annotated[bool, Query()] = True,
) -> MemoryListResponse:
    mm = _get_memory_manager(request)
    sid = _resolve_session_id(None, x_session_id, session_id)

    try:
        messages = mm.get_recent_messages(n=limit, session_id=sid) or []
        long_term = mm.get_long_term_memories(session_id=sid) if include_long_term else []
    except Exception as exc:
        logger.error("Failed to retrieve memories: %s", exc)
        raise EngineError(detail=str(exc))

    return MemoryListResponse(
        session_id=sid,
        messages=[_message_to_schema(m) for m in messages],
        long_term_memories=[_node_to_schema(n) for n in (long_term or [])],
        total_messages=len(messages),
        total_long_term=len(long_term or []),
    )


# ---------------------------------------------------------------------------
# DELETE /memory  — Clear memories for a session
# ---------------------------------------------------------------------------

@router.delete("", summary="Clear memories for a session")
async def delete_memories(
    request: Request,
    x_session_id: Annotated[str | None, Header()] = None,
    session_id: Annotated[str | None, Query()] = None,
) -> MemoryDeleteResponse:
    mm = _get_memory_manager(request)
    sid = _resolve_session_id(None, x_session_id, session_id)

    try:
        mm.clear_session(session_id=sid)
    except Exception as exc:
        logger.error("Failed to clear memories: %s", exc)
        raise EngineError(detail=str(exc))

    return MemoryDeleteResponse(status="ok", session_id=sid, deleted=True)


# ---------------------------------------------------------------------------
# POST /memory/search  — Search across all memory stores
# ---------------------------------------------------------------------------

@router.post("/search", summary="Search memories")
async def search_memories(
    body: SearchMemoryRequest,
    request: Request,
    x_session_id: Annotated[str | None, Header()] = None,
) -> SearchMemoryResponse:
    mm = _get_memory_manager(request)
    sid = _resolve_session_id(body.session_id, x_session_id, None)

    try:
        results = mm.search(query=body.query, top_k=body.top_k, session_id=sid)
    except Exception as exc:
        logger.error("Memory search failed: %s", exc)
        raise EngineError(detail=str(exc))

    items = [_result_to_schema(r) for r in (results or [])]
    return SearchMemoryResponse(results=items, session_id=sid, total=len(items))


# ---------------------------------------------------------------------------
# POST /memory/summarize  — Summarize long-term memories
# ---------------------------------------------------------------------------

@router.post("/summarize", summary="Summarize memories")
async def summarize_memories(
    request: Request,
    x_session_id: Annotated[str | None, Header()] = None,
    session_id: Annotated[str | None, Query()] = None,
) -> SummarizeResponse:
    mm = _get_memory_manager(request)
    sid = _resolve_session_id(None, x_session_id, session_id)

    try:
        summary = mm.summarize(session_id=sid)
    except Exception as exc:
        logger.error("Memory summarization failed: %s", exc)
        raise EngineError(detail=str(exc))

    return SummarizeResponse(summary=summary or "", session_id=sid)

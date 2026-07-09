from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Annotated, Any, AsyncIterator

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from api.dependencies import get_config, get_engine, get_request_id
from api.exceptions import BadRequestError, EngineError, NotFoundError
from api.config import ApiConfig

logger = logging.getLogger("aios.api.chat")

router = APIRouter(prefix="/chat", tags=["Chat"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class MessageSchema(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")
    name: str | None = Field(None, description="Optional participant name")


class ChatRequest(BaseModel):
    messages: list[MessageSchema] = Field(
        ..., min_length=1,
        description="Conversation messages. At least one user message required.",
        examples=[[{"role": "user", "content": "Hello!"}]],
    )
    session_id: str | None = Field(
        None,
        description="Session identifier. Omit to auto-create a new session.",
    )
    system_prompt: str | None = Field(
        None,
        description="Optional system prompt. Ignored if session already has one.",
    )
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(None, ge=1, le=131072, description="Maximum tokens to generate")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    stream: bool = Field(False, description="Enable streaming response")


class ChatResponseSchema(BaseModel):
    message: MessageSchema = Field(..., description="Assistant response message")
    session_id: str = Field(..., description="Session identifier")
    finish_reason: str = Field("stop", description="Reason generation finished")
    latency_ms: float = Field(0.0, description="Request latency in milliseconds")
    usage: dict[str, int] = Field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        description="Token usage statistics",
    )


class StreamChunkSchema(BaseModel):
    content: str = Field("", description="Streamed content fragment")
    session_id: str = Field(..., description="Session identifier")
    finish_reason: str | None = Field(None, description="Set on the final chunk")
    usage: dict[str, int] | None = Field(None, description="Final token usage")


class ChatHistoryResponse(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    messages: list[MessageSchema] = Field(default_factory=list, description="Message history")
    total_messages: int = Field(0, description="Number of messages in history")
    total_tokens: int = Field(0, description="Estimated total tokens in history")
    system_prompt: str = Field("", description="System prompt for this session")


class ChatResetResponse(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    status: str = Field("ok", description="Reset status")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_messages_for_engine(messages: list[MessageSchema]) -> list[dict[str, Any]]:
    return [{"role": m.role, "content": m.content, **({"name": m.name} if m.name else {})} for m in messages]


def _build_engine_kwargs(request: ChatRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    return kwargs


async def _resolve_session(
    engine: Any,
    session_id: str | None,
    system_prompt: str | None,
) -> str:
    if session_id:
        session = engine.get_session(session_id)
        if session is None:
            session_id = engine.create_session(
                session_id=session_id,
                system_prompt=system_prompt or "",
            )
        elif system_prompt and not session.system_prompt:
            engine.set_system_prompt(session_id, system_prompt)
        return session_id
    return engine.create_session(system_prompt=system_prompt or "")


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

@router.post("", summary="Send a chat message")
async def chat(
    body: ChatRequest,
    request: Request,
    engine: Any = Depends(get_engine),
    config: ApiConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> ChatResponseSchema:
    start = time.perf_counter()

    session_id = await _resolve_session(engine, body.session_id, body.system_prompt)
    messages = _format_messages_for_engine(body.messages)
    kwargs = _build_engine_kwargs(body)

    try:
        response = await engine.achat(messages, session_id=session_id, **kwargs)
    except Exception as exc:
        logger.error("Chat error (session=%s): %s", session_id, exc)
        raise EngineError(detail=str(exc))

    latency = (time.perf_counter() - start) * 1000

    return ChatResponseSchema(
        message=MessageSchema(
            role=response.message.role.value if hasattr(response.message.role, "value") else response.message.role,
            content=response.message.content,
        ),
        session_id=session_id,
        finish_reason=response.finish_reason,
        latency_ms=latency,
        usage={
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    )


# ---------------------------------------------------------------------------
# POST /chat/stream
# ---------------------------------------------------------------------------

@router.post("/stream", summary="Stream a chat response")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    engine: Any = Depends(get_engine),
    config: ApiConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> StreamingResponse:
    session_id = await _resolve_session(engine, body.session_id, body.system_prompt)
    messages = _format_messages_for_engine(body.messages)
    kwargs = _build_engine_kwargs(body)

    async def event_stream() -> AsyncIterator[str]:
        try:
            full_content: list[str] = []
            async for chunk in engine.stream(messages, session_id=session_id, **kwargs):
                full_content.append(chunk.content)
                data = StreamChunkSchema(
                    content=chunk.content,
                    session_id=session_id,
                    finish_reason=chunk.finish_reason,
                    usage={
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    } if chunk.usage else None,
                )
                yield f"data: {data.model_dump_json()}\n\n"

            yield f"data: [DONE]\n\n"
        except Exception as exc:
            logger.error("Stream error (session=%s): %s", session_id, exc)
            error_data = {"error": {"code": "engine_error", "message": str(exc)}}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield f"data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "X-Request-ID": request_id,
            "X-Session-ID": session_id,
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# POST /chat/reset
# ---------------------------------------------------------------------------

@router.post("/reset", summary="Reset a session's conversation history")
async def chat_reset(
    session_id: str = Query(..., description="Session identifier to reset"),
    engine: Any = Depends(get_engine),
) -> ChatResetResponse:
    session = engine.get_session(session_id)
    if session is None:
        raise NotFoundError(detail=f"Session not found: {session_id}")
    system = session.system_prompt
    engine.delete_session(session_id)
    engine.create_session(session_id=session_id, system_prompt=system)
    logger.info("Session reset: %s", session_id)
    return ChatResetResponse(session_id=session_id, status="ok")


# ---------------------------------------------------------------------------
# GET /chat/history
# ---------------------------------------------------------------------------

@router.get("/history", summary="Get conversation history")
async def chat_history(
    session_id: str = Query(..., description="Session identifier"),
    engine: Any = Depends(get_engine),
) -> ChatHistoryResponse:
    session = engine.get_session(session_id)
    if session is None:
        raise NotFoundError(detail=f"Session not found: {session_id}")

    total_tokens = 0
    try:
        total_tokens = engine._token_manager.count_messages(
            [m.to_dict() for m in session.messages]
        )
    except Exception:
        pass

    return ChatHistoryResponse(
        session_id=session_id,
        messages=[
            MessageSchema(
                role=m.role.value if hasattr(m.role, "value") else m.role,
                content=m.content,
                name=m.name or None,
            )
            for m in session.messages
        ],
        total_messages=len(session.messages),
        total_tokens=total_tokens,
        system_prompt=session.system_prompt,
    )


# ---------------------------------------------------------------------------
# DELETE /chat/history
# ---------------------------------------------------------------------------

@router.delete("/history", summary="Delete conversation history")
async def chat_history_delete(
    session_id: str = Query(..., description="Session identifier"),
    engine: Any = Depends(get_engine),
) -> ChatResetResponse:
    session = engine.get_session(session_id)
    if session is None:
        raise NotFoundError(detail=f"Session not found: {session_id}")
    engine.delete_session(session_id)
    logger.info("Session deleted: %s", session_id)
    return ChatResetResponse(session_id=session_id, status="deleted")

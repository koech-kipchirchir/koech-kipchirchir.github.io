from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Annotated, Any, AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from api.dependencies import get_config, get_request_id
from api.exceptions import BadRequestError, EngineError, ServiceUnavailableError
from api.config import ApiConfig

logger = logging.getLogger("aios.api.rag")

router = APIRouter(prefix="/documents", tags=["RAG"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class IndexRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text content to index")
    source: str = Field("", description="Source identifier (filename, URL, etc.)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class IndexResponse(BaseModel):
    status: str = Field("ok", description="Operation status")
    doc_id: str = Field("", description="Document ID")
    filename: str = Field("", description="Source filename")
    chunks: int = Field(0, description="Number of chunks produced")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(5, ge=1, le=100, description="Max results to return")
    filters: dict[str, Any] | None = Field(None, description="Metadata filters")
    rerank: bool = Field(True, description="Apply cross-encoder reranking")


class SearchResultItem(BaseModel):
    text: str = Field("", description="Chunk text")
    score: float = Field(0.0, description="Relevance score")
    source: str = Field("", description="Source document identifier")
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_index: int = Field(0, description="Chunk position in source document")
    doc_id: str = Field("", description="Document ID")


class SearchResponse(BaseModel):
    results: list[SearchResultItem] = Field(default_factory=list, description="Search results")
    total: int = Field(0, description="Number of results returned")
    query: str = Field("", description="Original query")


class DocumentListItem(BaseModel):
    doc_id: str = Field("", description="Document ID")
    source: str = Field("", description="Source filename or identifier")
    chunks: int = Field(0, description="Number of chunks for this document")
    indexed_at: str = Field("", description="ISO-8601 index timestamp")


class DocumentsResponse(BaseModel):
    total_chunks: int = Field(0, description="Total indexed chunks")
    total_documents: int = Field(0, description="Total indexed documents")
    documents: list[DocumentListItem] = Field(default_factory=list, description="Per-document details")


class DeleteResponse(BaseModel):
    status: str = Field("ok", description="Operation status")
    deleted: bool = Field(True, description="Whether data was deleted")
    total_chunks_removed: int = Field(0, description="Number of chunks removed")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def _get_pipeline(request: Request):
    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if pipeline is None:
        raise ServiceUnavailableError(detail="RAG pipeline not initialized")
    return pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm", ".csv", ".json"}


def _check_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise BadRequestError(
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )
    return ext


async def _save_upload(tmp_dir: str, file: UploadFile) -> str:
    ext = _check_extension(file.filename or "unknown.bin")
    dest = Path(tmp_dir) / f"{uuid4().hex}{ext}"
    content = await file.read()
    dest.write_bytes(content)
    return str(dest)


def _result_to_item(r: dict[str, Any]) -> SearchResultItem:
    meta = r.get("metadata", {})
    return SearchResultItem(
        text=r.get("text", ""),
        score=float(r.get("score", r.get("rerank_score", 0.0))),
        source=r.get("source", meta.get("source", "")),
        metadata=meta,
        chunk_index=meta.get("chunk_index", 0),
        doc_id=meta.get("doc_id", r.get("id", "")),
    )


# ---------------------------------------------------------------------------
# POST /documents/upload  — Upload & index a file
# ---------------------------------------------------------------------------

@router.post("/upload", summary="Upload and index a document")
async def upload_document(
    request: Request,
    file: UploadFile,
    metadata: Annotated[str | None, Form()] = None,
    stream: Annotated[bool, Query()] = False,
    request_id: str = Depends(get_request_id),
):
    pipeline = _get_pipeline(request)
    meta_dict: dict[str, Any] = {}
    if metadata:
        try:
            meta_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise BadRequestError(detail="metadata must be valid JSON")

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = await _save_upload(tmp_dir, file)

        if stream:
            return await _upload_stream(pipeline, path, file.filename or "unknown", meta_dict, request_id)
        return await _upload_sync(pipeline, path, file.filename or "unknown", meta_dict)


async def _upload_sync(pipeline, path: str, filename: str, metadata: dict[str, Any]) -> IndexResponse:
    try:
        metadata["filename"] = filename
        chunks = pipeline.index_file(path, metadata=metadata)
    except Exception as exc:
        logger.error("Index error for '%s': %s", filename, exc)
        raise EngineError(detail=str(exc))

    doc_id = metadata.get("doc_id", "")
    return IndexResponse(status="ok", doc_id=doc_id, filename=filename, chunks=chunks)


async def _upload_stream(pipeline, path: str, filename: str, metadata: dict[str, Any], request_id: str):
    from rag.loader import DocumentLoader
    from rag.document_parser import get_parser

    ext = Path(path).suffix.lower()
    parser = get_parser(ext)
    if parser is None:
        return JSONResponse(
            status_code=400,
            content={"error": {"code": "bad_request", "message": f"No parser for '{ext}'"}},
        )

    async def event_stream() -> AsyncIterator[str]:
        try:
            yield f"data: {json.dumps({'event': 'parsing', 'filename': filename})}\n\n"

            content_bytes = Path(path).read_bytes()
            text = parser.parse(content_bytes, metadata=metadata)
            text_len = len(text)

            yield f"data: {json.dumps({'event': 'parsed', 'characters': text_len, 'filename': filename})}\n\n"

            yield f"data: {json.dumps({'event': 'chunking', 'filename': filename})}\n\n"
            chunks = pipeline._chunker.chunk(text, metadata={"source": filename, **metadata})
            total = len(chunks)

            yield f"data: {json.dumps({'event': 'chunked', 'total_chunks': total, 'filename': filename})}\n\n"

            doc_id = uuid4().hex[:16]
            texts = [c["text"] for c in chunks]
            embeddings = pipeline._embeddings.embed_batch(texts)
            ids = [f"{doc_id}_{i}" for i in range(total)]
            metadatas = [c["metadata"] for c in chunks]

            import asyncio
            for i in range(total):
                chunk_meta = dict(metadatas[i])
                chunk_meta["doc_id"] = doc_id
                chunk_meta["source"] = filename
                chunk_meta["chunk_index"] = i
                chunk_meta["total_chunks"] = total

                if pipeline._vector_store is not None:
                    try:
                        pipeline._vector_store.add(ids[i], embeddings[i], chunk_meta)
                    except Exception as exc:
                        logger.warning("Vector store add failed for chunk %d: %s", i, exc)

                yield f"data: {json.dumps({'event': 'progress', 'chunk': i + 1, 'total': total, 'filename': filename})}\n\n"
                await asyncio.sleep(0)

            pipeline._documents.extend([
                {"text": texts[i], "metadata": dict(metadatas[i])}
                for i in range(total)
            ])
            pipeline._rebuild_retriever()

            yield f"data: {json.dumps({'event': 'complete', 'doc_id': doc_id, 'chunks': total, 'filename': filename})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as exc:
            logger.error("Streaming index error for '%s': %s", filename, exc)
            yield f"data: {json.dumps({'event': 'error', 'message': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "X-Request-ID": request_id,
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# POST /documents/index  — Index text directly
# ---------------------------------------------------------------------------

@router.post("/index", summary="Index text content directly", status_code=201)
async def index_text(
    body: IndexRequest,
    request: Request,
) -> IndexResponse:
    pipeline = _get_pipeline(request)

    try:
        metadata = dict(body.metadata)
        if body.source:
            metadata["source"] = body.source
        chunks = pipeline.index_text(body.text, source=body.source, metadata=metadata)
    except Exception as exc:
        logger.error("Index text error: %s", exc)
        raise EngineError(detail=str(exc))

    doc_id = metadata.get("doc_id", "")
    return IndexResponse(status="ok", doc_id=doc_id, filename=body.source, chunks=chunks)


# ---------------------------------------------------------------------------
# POST /documents/search  — Search indexed documents
# ---------------------------------------------------------------------------

@router.post("/search", summary="Search indexed documents")
async def search_documents(
    body: SearchRequest,
    request: Request,
) -> SearchResponse:
    pipeline = _get_pipeline(request)

    try:
        results = await pipeline.query(
            query=body.query,
            top_k=body.top_k,
            filter=body.filters,
            rerank=body.rerank,
        )
    except Exception as exc:
        logger.error("Search error: %s", exc)
        raise EngineError(detail=str(exc))

    items = [_result_to_item(r) for r in (results or [])]
    return SearchResponse(results=items, total=len(items), query=body.query)


# ---------------------------------------------------------------------------
# GET /documents  — List indexed documents
# ---------------------------------------------------------------------------

@router.get("", summary="List indexed documents and stats")
async def list_documents(
    request: Request,
) -> DocumentsResponse:
    pipeline = _get_pipeline(request)

    try:
        total_chunks = pipeline.get_document_count()
        doc_map: dict[str, dict[str, Any]] = {}
        for chunk in pipeline._documents:
            meta = chunk.get("metadata", {})
            doc_id = meta.get("doc_id", meta.get("source", "unknown"))
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "doc_id": doc_id,
                    "source": meta.get("source", meta.get("filename", "")),
                    "chunks": 0,
                    "indexed_at": meta.get("indexed_at", ""),
                }
            doc_map[doc_id]["chunks"] += 1
    except Exception as exc:
        logger.error("List documents error: %s", exc)
        raise EngineError(detail=str(exc))

    docs = [DocumentListItem(**d) for d in doc_map.values()]
    return DocumentsResponse(
        total_chunks=total_chunks,
        total_documents=len(docs),
        documents=docs,
    )


# ---------------------------------------------------------------------------
# DELETE /documents  — Clear all indexed documents
# ---------------------------------------------------------------------------

@router.delete("", summary="Clear all indexed documents")
async def delete_documents(
    request: Request,
) -> DeleteResponse:
    pipeline = _get_pipeline(request)

    try:
        total = pipeline.get_document_count()
        pipeline.clear()
    except Exception as exc:
        logger.error("Clear documents error: %s", exc)
        raise EngineError(detail=str(exc))

    return DeleteResponse(status="ok", deleted=True, total_chunks_removed=total)

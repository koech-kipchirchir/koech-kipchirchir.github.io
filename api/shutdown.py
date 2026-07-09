from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI

logger = logging.getLogger("aios.api.shutdown")


async def shutdown_event(app: FastAPI) -> None:
    logger.info("Shutting down AIOS API...")

    app.state.ready = False

    engine = getattr(app.state, "engine", None)
    if engine is not None:
        try:
            logger.debug("Clearing engine sessions...")
            engine.clear_sessions()
            logger.debug("Engine sessions cleared")
        except Exception as exc:
            logger.warning("Error clearing engine sessions: %s", exc)

    memory_manager = getattr(app.state, "memory_manager", None)
    if memory_manager is not None:
        try:
            logger.debug("Running memory manager cleanup...")
            memory_manager.cleanup()
            logger.debug("Memory manager cleanup complete")
        except Exception as exc:
            logger.warning("Error cleaning memory manager: %s", exc)

    rag_pipeline = getattr(app.state, "rag_pipeline", None)
    if rag_pipeline is not None:
        try:
            logger.debug("Clearing RAG pipeline...")
            rag_pipeline.clear()
            logger.debug("RAG pipeline cleared")
        except Exception as exc:
            logger.warning("Error clearing RAG pipeline: %s", exc)

    model_manager = getattr(app.state, "model_manager", None)
    if model_manager is not None:
        try:
            logger.debug("Unloading all models...")
            for name in list(model_manager.loaded_models.keys()):
                await model_manager.unload(name)
            logger.debug("All models unloaded")
        except Exception as exc:
            logger.warning("Error unloading models: %s", exc)

    try:
        rate_limiter = getattr(app.state, "rate_limiter", None)
        if rate_limiter is not None:
            await rate_limiter.cleanup()
            logger.debug("Rate limiter cleaned up")
    except Exception as exc:
        logger.warning("Error cleaning rate limiter: %s", exc)

    # Flush pending tasks
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        logger.debug("Waiting for %d pending tasks...", len(pending))
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    logger.info("AIOS API shutdown complete")

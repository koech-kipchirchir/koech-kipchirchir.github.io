from __future__ import annotations

import asyncio
import logging
import time

from fastapi import FastAPI

from api.config import ApiConfig
from aios_core.config import AiosConfig, configure_logging
from aios_core.engine import AIOSEngine
from memory.memory_manager import MemoryManager, MemoryConfig
from aios_core.models import EngineConfig
from aios_core.model_manager import ModelManager
from aios_core.token_manager import TokenManager
from rag.pipeline import RAGPipeline
from rag.utils import RAGConfig
from tools.tool_manager import ToolManager

logger = logging.getLogger("aios.api.startup")


async def init_engine(config: ApiConfig) -> AIOSEngine:
    logger.info(
        "Initializing AIOSEngine (model=%s, provider=%s)",
        config.model, config.provider,
    )
    engine_config = EngineConfig(
        model=config.model,
        provider=config.provider,
        api_key=config.api_key,
        api_base=config.api_base,
        enable_caching=config.enable_caching,
    )
    engine = AIOSEngine(engine_config)
    engine._start_time = time.time()

    # Warm up with a simple health query
    try:
        async def warmup() -> None:
            try:
                await asyncio.wait_for(
                    engine.achat([{"role": "user", "content": "ping"}]),
                    timeout=10.0,
                )
                logger.info("Engine warm-up completed")
            except asyncio.TimeoutError:
                logger.warning("Engine warm-up timed out (continuing)")
            except Exception as exc:
                logger.warning("Engine warm-up failed: %s (continuing)", exc)

        await warmup()
    except Exception:
        pass

    return engine


async def startup_event(app: FastAPI, config: ApiConfig) -> None:
    logger.info("Starting AIOS API (v%s)", config.api_version)

    aios_cfg = AiosConfig()
    aios_cfg.setup_directories()
    configure_logging(aios_cfg)

    app.state.config = config
    app.state.ready = False
    app.state.start_time = time.time()

    try:
        engine = await init_engine(config)
        app.state.engine = engine
        app.state.token_manager = engine._token_manager

        model_manager = ModelManager()
        app.state.model_manager = model_manager
        logger.info("Model manager initialized")

        memory_config = MemoryConfig(
            db_path=config.memory_db_path,
            vector_db_path=config.memory_vector_db_path,
            embedding_model=config.memory_embedding_model,
            max_conversation_turns=config.memory_max_turns,
            auto_summarize=config.memory_auto_summarize,
        )
        memory_manager = MemoryManager(config=memory_config)
        app.state.memory_manager = memory_manager
        logger.info("Memory manager initialized")

        rag_config = RAGConfig(
            chunk_size=config.rag_chunk_size,
            chunk_overlap=config.rag_chunk_overlap,
            embedding_model=config.rag_embedding_model,
            vector_db_path=config.rag_vector_db_path,
            similarity_top_k=config.rag_similarity_top_k,
            hybrid_search_alpha=config.rag_hybrid_alpha,
            enable_chromadb=config.rag_enable_chromadb,
        )
        rag_pipeline = RAGPipeline(config=rag_config)
        app.state.rag_pipeline = rag_pipeline
        logger.info("RAG pipeline initialized")

        tool_manager = ToolManager()
        app.state.tool_manager = tool_manager
        app.state._builtin_tools = set(tool_manager.available_tools)
        logger.info("Tool manager initialized with %d built-in tools", len(app.state._builtin_tools))

        app.state.ready = True
        logger.info("AIOS API startup complete")
    except Exception as exc:
        logger.error("Failed to initialize engine: %s", exc)
        app.state.engine = None
        app.state.model_manager = None
        app.state.memory_manager = None
        app.state.rag_pipeline = None
        app.state.tool_manager = None
        app.state._builtin_tools = set()
        app.state.ready = False
        raise

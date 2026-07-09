from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.dependencies import get_config, get_engine, get_request_id
from api.exceptions import BadRequestError, EngineError, NotFoundError
from api.config import ApiConfig

logger = logging.getLogger("aios.api.model")

router = APIRouter(prefix="/models", tags=["Models"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ModelInfoSchema(BaseModel):
    name: str = Field("", description="Model name alias")
    model_id: str = Field("", description="Full model identifier")
    backend: str = Field("", description="Backend type")
    device: str = Field("", description="Compute device")
    dtype: str = Field("", description="Data type")
    quantization: str = Field("none", description="Quantization method")
    max_model_len: int = Field(0, description="Maximum context length")
    loaded: bool = Field(False, description="Whether the model is loaded")
    active: bool = Field(False, description="Whether this is the active model")


class GpuDeviceSchema(BaseModel):
    index: int = Field(..., description="GPU device index")
    name: str = Field("", description="GPU name")
    memory_total_gb: float = Field(0.0, description="Total GPU memory in GB")
    memory_used_gb: float = Field(0.0, description="Used GPU memory in GB")
    memory_free_gb: float = Field(0.0, description="Free GPU memory in GB")
    utilization_pct: float = Field(0.0, description="GPU utilization percentage")


class SystemStatusSchema(BaseModel):
    gpu_available: bool = Field(False, description="Whether GPU is available")
    gpu_count: int = Field(0, description="Number of GPUs")
    gpu_devices: list[GpuDeviceSchema] = Field(default_factory=list, description="GPU device details")
    cpu_count: int = Field(0, description="Number of CPU cores")
    memory_total_gb: float = Field(0.0, description="Total system memory in GB")
    memory_used_gb: float = Field(0.0, description="Used system memory in GB")
    memory_free_gb: float = Field(0.0, description="Free system memory in GB")
    platform: str = Field("", description="Operating system")
    python_version: str = Field("", description="Python version")


class LoadModelRequest(BaseModel):
    name: str = Field(..., description="Model name alias", examples=["gpt-4o"])
    model_id: str | None = Field(None, description="Full model identifier (defaults to name)")
    backend: str | None = Field(None, description="Backend type: huggingface, vllm, ollama, llama.cpp, openai")
    device: str = Field("auto", description="Device: auto, cpu, cuda")
    dtype: str = Field("auto", description="Data type: auto, float16, float32, bfloat16")
    quantization: str = Field("", description="Quantization: 4bit, 8bit, none")
    max_model_len: int = Field(8192, description="Maximum context length")


class UnloadModelRequest(BaseModel):
    name: str = Field(..., description="Model name alias to unload", examples=["gpt-4o"])


class SwitchModelRequest(BaseModel):
    name: str = Field(..., description="Model name alias to set as active", examples=["gpt-4o"])


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def _get_model_manager(request):
    mm = getattr(request.app.state, "model_manager", None)
    if mm is None:
        raise EngineError(detail="Model manager not initialized")
    return mm


def _get_engine(request):
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise EngineError(detail="AI engine not initialized")
    return engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gather_gpu_info() -> list[GpuDeviceSchema]:
    devices: list[GpuDeviceSchema] = []
    try:
        import subprocess
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    devices.append(GpuDeviceSchema(
                        index=int(parts[0]),
                        name=parts[1],
                        memory_total_gb=float(parts[2]) / 1024,
                        memory_used_gb=float(parts[3]) / 1024,
                        memory_free_gb=float(parts[4]) / 1024,
                        utilization_pct=float(parts[5]),
                    ))
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    total = props.total_memory / 1e9
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    devices.append(GpuDeviceSchema(
                        index=i,
                        name=torch.cuda.get_device_name(i),
                        memory_total_gb=round(total, 2),
                        memory_used_gb=round(allocated, 2),
                        memory_free_gb=round(total - allocated, 2),
                        utilization_pct=0.0,
                    ))
        except ImportError:
            pass
    return devices


def _gather_system_memory() -> dict[str, float]:
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / 1e9, 2),
            "used_gb": round(mem.used / 1e9, 2),
            "free_gb": round(mem.available / 1e9, 2),
        }
    except ImportError:
        return {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0}


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

@router.get("", summary="List available and loaded models")
async def list_models(
    request,
    engine: Any = Depends(_get_engine),
    config: ApiConfig = Depends(get_config),
) -> list[ModelInfoSchema]:
    models: list[ModelInfoSchema] = []

    mm = getattr(request.app.state, "model_manager", None)
    if mm is not None:
        for entry in mm.list_models():
            info = mm.get_info(entry["name"])
            models.append(ModelInfoSchema(
                name=entry["name"],
                model_id=info.model_id if info else entry["model_id"],
                backend=entry["backend"],
                device=info.device if info else "",
                dtype=info.dtype if info else "",
                quantization=info.quantization if info else "none",
                max_model_len=info.model_len if info else 0,
                loaded=entry["loaded"],
                active=entry["active"],
            ))

    if not models:
        models.append(ModelInfoSchema(
            name="default",
            model_id=config.model,
            backend=config.provider,
            device="cpu",
            loaded=True,
            active=True,
        ))

    return models


# ---------------------------------------------------------------------------
# GET /models/current
# ---------------------------------------------------------------------------

@router.get("/current", summary="Get currently active model details")
async def current_model(
    request,
    engine: Any = Depends(_get_engine),
    config: ApiConfig = Depends(get_config),
) -> ModelInfoSchema:
    mm = getattr(request.app.state, "model_manager", None)
    if mm is not None and mm.active_model:
        info = mm.get_info(mm.active_model)
        if info:
            return ModelInfoSchema(
                name=mm.active_model,
                model_id=info.model_id,
                backend=info.backend.value,
                device=info.device,
                dtype=info.dtype,
                quantization=info.quantization,
                max_model_len=info.model_len,
                loaded=True,
                active=True,
            )
        return ModelInfoSchema(
            name=mm.active_model,
            model_id=mm.active_model,
            backend="",
            loaded=True,
            active=True,
        )

    engine_config = getattr(engine, "config", None)
    return ModelInfoSchema(
        name="default",
        model_id=engine_config.model if engine_config else config.model,
        backend=engine_config.provider if engine_config else config.provider,
        device="cpu",
        loaded=True,
        active=True,
    )


# ---------------------------------------------------------------------------
# POST /models/load
# ---------------------------------------------------------------------------

@router.post("/load", summary="Load a model", status_code=201)
async def load_model(
    body: LoadModelRequest,
    request,
) -> ModelInfoSchema:
    mm = _get_model_manager(request)
    model_id = body.model_id or body.name

    from aios_core.model_manager import ModelConfig, BackendType, detect_backend

    if body.backend:
        backend = BackendType(body.backend.lower())
    else:
        backend = detect_backend(model_id)

    config = ModelConfig(
        model_id=model_id,
        backend=backend,
        device=body.device,
        dtype=body.dtype,
        quantization=body.quantization,
        max_model_len=body.max_model_len,
    )

    try:
        info = await mm.load(body.name, config)
    except Exception as exc:
        logger.error("Failed to load model '%s': %s", body.name, exc)
        raise EngineError(detail=str(exc))

    return ModelInfoSchema(
        name=body.name,
        model_id=info.model_id,
        backend=info.backend.value,
        device=info.device,
        dtype=info.dtype,
        quantization=info.quantization,
        max_model_len=info.model_len,
        loaded=True,
        active=body.name == mm.active_model,
    )


# ---------------------------------------------------------------------------
# POST /models/unload
# ---------------------------------------------------------------------------

@router.post("/unload", summary="Unload a model")
async def unload_model(
    body: UnloadModelRequest,
    request,
) -> dict[str, str]:
    mm = _get_model_manager(request)

    try:
        success = await mm.unload(body.name)
    except Exception as exc:
        logger.error("Failed to unload model '%s': %s", body.name, exc)
        raise EngineError(detail=str(exc))

    if not success:
        raise NotFoundError(detail=f"Model '{body.name}' not found")

    logger.info("Model unloaded: %s", body.name)
    return {"status": "ok", "name": body.name}


# ---------------------------------------------------------------------------
# POST /models/switch
# ---------------------------------------------------------------------------

@router.post("/switch", summary="Switch active model")
async def switch_model(
    body: SwitchModelRequest,
    request,
) -> dict[str, str]:
    mm = _get_model_manager(request)

    if body.name not in mm.loaded_models:
        raise NotFoundError(detail=f"Model '{body.name}' not loaded. Load it first.")

    mm.set_active(body.name)
    logger.info("Active model switched to: %s", body.name)
    return {"status": "ok", "active_model": body.name}


# ---------------------------------------------------------------------------
# GET /models/status
# ---------------------------------------------------------------------------

@router.get("/status", summary="Get system status (GPU, memory, platform)")
async def system_status(
    request,
    engine: Any = Depends(_get_engine),
) -> SystemStatusSchema:
    import platform as plat
    gpu_devices = _gather_gpu_info()
    mem = _gather_system_memory()

    cpu_count = 0
    try:
        cpu_count = len(plat.processor()) if hasattr(plat, "processor") else 0
    except Exception:
        pass
    try:
        import os
        cpu_count = os.cpu_count() or 0
    except Exception:
        pass

    return SystemStatusSchema(
        gpu_available=len(gpu_devices) > 0,
        gpu_count=len(gpu_devices),
        gpu_devices=gpu_devices,
        cpu_count=cpu_count,
        memory_total_gb=mem["total_gb"],
        memory_used_gb=mem["used_gb"],
        memory_free_gb=mem["free_gb"],
        platform=plat.system(),
        python_version=plat.python_version(),
    )

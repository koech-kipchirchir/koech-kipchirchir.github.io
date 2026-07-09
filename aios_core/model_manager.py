from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger("aios.core.model_manager")


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class BackendType(str, Enum):
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"
    OLLAMA = "ollama"
    LLAMA_CPP = "llama.cpp"
    OPENAI = "openai"


_MODEL_PATTERN_MAP: list[tuple[str, BackendType]] = [
    (r"^https?://api\.openai\.com", BackendType.OPENAI),
    (r"^https?://", BackendType.OPENAI),
    (r":\d{4,5}/v1", BackendType.OPENAI),
    (r"\.gguf$", BackendType.LLAMA_CPP),
    (r"\.ggml", BackendType.LLAMA_CPP),
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop: list[str] = field(default_factory=list)
    stream: bool = False


@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    backend: BackendType = BackendType.HUGGINGFACE
    device: str = "auto"
    dtype: str = "auto"
    quantization: str = ""
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = True
    use_flash_attn: bool = True
    seed: int = 42
    api_base: str = ""
    api_key: str = ""
    ollama_host: str = "http://localhost:11434"
    llm_cpp_path: str = ""
    n_gpu_layers: int = -1
    cache_dir: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    generation: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass
class ModelInfo:
    model_id: str
    backend: BackendType
    device: str
    dtype: str
    quantization: str
    model_len: int
    loaded_at: float
    memory_mb: float = 0.0
    num_parameters: int = 0


@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    time_to_first_token_ms: float = 0.0


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------

class ModelBackend(ABC):
    """Abstract base for all model backends."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._logger = logging.getLogger(f"aios.model.backend.{self.backend_type.value}")
        self._loaded = False
        self._lock = threading.Lock()

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        ...

    @abstractmethod
    async def load(self) -> None:
        ...

    @abstractmethod
    async def unload(self) -> None:
        ...

    @abstractmethod
    async def generate(
        self, prompt: str, gen_config: GenerationConfig | None = None, **kwargs: Any
    ) -> tuple[str, GenerationStats]:
        ...

    @abstractmethod
    async def generate_stream(
        self, prompt: str, gen_config: GenerationConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        ...

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        ...

    @abstractmethod
    async def health(self) -> bool:
        ...

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self.config.model_id,
            backend=self.backend_type,
            device=self._detect_device(),
            dtype=self.config.dtype,
            quantization=self.config.quantization or "none",
            model_len=self.config.max_model_len,
            loaded_at=time.time(),
        )

    def _detect_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @staticmethod
    def detect_gpu() -> dict[str, Any]:
        info: dict[str, Any] = {"available": False, "count": 0, "devices": []}
        try:
            import torch

            if torch.cuda.is_available():
                info["available"] = True
                info["count"] = torch.cuda.device_count()
                for i in range(info["count"]):
                    info["devices"].append({
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_total_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2),
                    })
        except ImportError:
            pass
        if not info["available"]:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    info["available"] = True
                    for line in result.stdout.strip().split("\n"):
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 3:
                            info["devices"].append({
                                "index": int(parts[0]),
                                "name": parts[1],
                                "memory_total_gb": float(parts[2].replace(" MiB", "")) / 1024,
                            })
                    info["count"] = len(info["devices"])
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        return info


# ---------------------------------------------------------------------------
# HuggingFace Backend
# ---------------------------------------------------------------------------

class HFBackend(ModelBackend):
    @property
    def backend_type(self) -> BackendType:
        return BackendType.HUGGINGFACE

    async def load(self) -> None:
        if self._loaded:
            return
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

            kwargs: dict[str, Any] = {
                "trust_remote_code": self.config.trust_remote_code,
                "device_map": self.config.device if self.config.device != "auto" else "auto",
            }

            quant = self.config.quantization.lower()
            if quant in ("4bit", "4-bit", "nf4"):
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self._get_torch_dtype(),
                    bnb_4bit_use_double_quant=True,
                )
            elif quant in ("8bit", "8-bit"):
                kwargs["load_in_8bit"] = True
            else:
                kwargs["torch_dtype"] = self._get_torch_dtype()

            if self.config.use_flash_attn:
                try:
                    import flash_attn

                    kwargs["attn_implementation"] = "flash_attention_2"
                except ImportError:
                    pass

            self._logger.info("Loading HF model: %s", self.config.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **kwargs)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id, trust_remote_code=self.config.trust_remote_code
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._loaded = True
            self._logger.info("HF model loaded: %s", self.config.model_id)
        except ImportError:
            raise ImportError("transformers package required for HuggingFace backend")

    async def unload(self) -> None:
        with self._lock:
            if hasattr(self, "_model") and self._model is not None:
                del self._model
            if hasattr(self, "_tokenizer") and self._tokenizer is not None:
                del self._tokenizer
            self._loaded = False
            import gc

            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass
            self._logger.info("HF model unloaded: %s", self.config.model_id)

    async def generate(
        self, prompt: str, gen_config: GenerationConfig | None = None, **kwargs: Any
    ) -> tuple[str, GenerationStats]:
        cfg = gen_config or self.config.generation
        stats = GenerationStats()
        start = time.perf_counter()

        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self._detect_device() == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        stats.prompt_tokens = inputs["input_ids"].shape[1]

        import torch

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                repetition_penalty=cfg.repeat_penalty,
                do_sample=cfg.temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                stop_strings=cfg.stop or None,
                tokenizer=self._tokenizer,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        stats.generated_tokens = len(generated)
        stats.total_time_ms = (time.perf_counter() - start) * 1000
        stats.tokens_per_second = stats.generated_tokens / (stats.total_time_ms / 1000) if stats.total_time_ms > 0 else 0

        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return text, stats

    async def generate_stream(
        self, prompt: str, gen_config: GenerationConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        cfg = gen_config or self.config.generation
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self._detect_device() == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        import torch

        with torch.no_grad():
            generated = inputs["input_ids"]
            for _ in range(cfg.max_tokens):
                outputs = self._model(**inputs if _ == 0 else {"input_ids": generated})
                logits = outputs.logits[:, -1, :]
                if cfg.temperature > 0:
                    probs = torch.softmax(logits / cfg.temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=-1)
                token_str = self._tokenizer.decode(next_token[0], skip_special_tokens=True)
                if token_str:
                    yield token_str

                if next_token.item() == self._tokenizer.eos_token_id:
                    break

    async def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    async def health(self) -> bool:
        return self._loaded and hasattr(self, "_model") and self._model is not None

    def _get_torch_dtype(self) -> Any:
        try:
            import torch

            return getattr(torch, self.config.dtype) if self.config.dtype != "auto" else torch.float16 if self._detect_device() == "cuda" else torch.float32
        except ImportError:
            import numpy as np

            return np.float32


# ---------------------------------------------------------------------------
# vLLM Backend
# ---------------------------------------------------------------------------

class VLLMBackend(ModelBackend):
    @property
    def backend_type(self) -> BackendType:
        return BackendType.VLLM

    async def load(self) -> None:
        if self._loaded:
            return
        try:
            from vllm import AsyncLLMEngine, SamplingParams
            from vllm.engine.arg_utils import AsyncEngineArgs

            args = AsyncEngineArgs(
                model=self.config.model_id,
                max_model_len=self.config.max_model_len,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=self.config.trust_remote_code,
                dtype=self.config.dtype if self.config.dtype != "auto" else "auto",
                seed=self.config.seed,
            )
            self._engine = AsyncLLMEngine.from_engine_args(args)
            self._loaded = True
            self._logger.info("vLLM model loaded: %s", self.config.model_id)
        except ImportError:
            raise ImportError("vllm package required for vLLM backend")

    async def unload(self) -> None:
        with self._lock:
            if hasattr(self, "_engine") and self._engine is not None:
                del self._engine
            self._loaded = False
            import gc

            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass
            self._logger.info("vLLM model unloaded: %s", self.config.model_id)

    async def generate(
        self, prompt: str, gen_config: GenerationConfig | None = None, **kwargs: Any
    ) -> tuple[str, GenerationStats]:
        from vllm import SamplingParams

        cfg = gen_config or self.config.generation
        stats = GenerationStats()
        start = time.perf_counter()

        params = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            max_tokens=cfg.max_tokens,
            stop=cfg.stop or None,
            repetition_penalty=cfg.repeat_penalty,
        )

        request_id = uuid.uuid4().hex[:16]
        full_text = ""
        first_token = True

        async for result in self._engine.generate(prompt, params, request_id):
            if first_token:
                stats.time_to_first_token_ms = (time.perf_counter() - start) * 1000
                first_token = False
            full_text = result.outputs[0].text

        stats.prompt_tokens = len(result.prompt_token_ids) if hasattr(result, "prompt_token_ids") else 0
        stats.generated_tokens = len(result.outputs[0].token_ids) if hasattr(result, "outputs") else 0
        stats.total_time_ms = (time.perf_counter() - start) * 1000
        stats.tokens_per_second = stats.generated_tokens / (stats.total_time_ms / 1000) if stats.total_time_ms > 0 else 0

        return full_text, stats

    async def generate_stream(
        self, prompt: str, gen_config: GenerationConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        from vllm import SamplingParams

        cfg = gen_config or self.config.generation
        params = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            max_tokens=cfg.max_tokens,
            stop=cfg.stop or None,
        )

        request_id = uuid.uuid4().hex[:16]
        async for result in self._engine.generate(prompt, params, request_id):
            yield result.outputs[0].text

    async def count_tokens(self, text: str) -> int:
        if hasattr(self, "_engine") and self._engine is not None:
            return len((await self._engine.get_tokenizer()).encode(text))
        return len(text) // 2

    async def health(self) -> bool:
        return self._loaded and hasattr(self, "_engine") and self._engine is not None


# ---------------------------------------------------------------------------
# Ollama Backend
# ---------------------------------------------------------------------------

class OllamaBackend(ModelBackend):
    @property
    def backend_type(self) -> BackendType:
        return BackendType.OLLAMA

    async def load(self) -> None:
        try:
            import httpx

            self._client = httpx.AsyncClient(base_url=self.config.ollama_host, timeout=60)
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            available = [m["name"] for m in models]

            if self.config.model_id not in available and ":" not in self.config.model_id:
                tagged = f"{self.config.model_id}:latest"
                if tagged not in available:
                    self._logger.warning(
                        "Model %s not found in Ollama. Pull it first: ollama pull %s",
                        self.config.model_id, self.config.model_id,
                    )

            self._loaded = True
            self._logger.info("Ollama backend ready (host=%s)", self.config.ollama_host)
        except ImportError:
            raise ImportError("httpx package required for Ollama backend")

    async def unload(self) -> None:
        with self._lock:
            if hasattr(self, "_client"):
                await self._client.aclose()
            self._loaded = False
            self._logger.info("Ollama backend disconnected")

    async def generate(
        self, prompt: str, gen_config: GenerationConfig | None = None, **kwargs: Any
    ) -> tuple[str, GenerationStats]:
        cfg = gen_config or self.config.generation
        stats = GenerationStats()
        start = time.perf_counter()

        payload = {
            "model": self.config.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "num_predict": cfg.max_tokens,
                "repeat_penalty": cfg.repeat_penalty,
                "stop": cfg.stop or [],
            },
        }

        resp = await self._client.post("/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()

        stats.prompt_tokens = data.get("prompt_eval_count", 0)
        stats.generated_tokens = data.get("eval_count", 0)
        stats.total_time_ms = (time.perf_counter() - start) * 1000
        stats.tokens_per_second = stats.generated_tokens / (stats.total_time_ms / 1000) if stats.total_time_ms > 0 else 0

        return data.get("response", ""), stats

    async def generate_stream(
        self, prompt: str, gen_config: GenerationConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        cfg = gen_config or self.config.generation
        payload = {
            "model": self.config.model_id,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "num_predict": cfg.max_tokens,
            },
        }

        async with self._client.stream("POST", "/api/generate", json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("response"):
                    yield data["response"]

    async def count_tokens(self, text: str) -> int:
        resp = await self._client.post("/api/embed", json={"model": self.config.model_id, "input": text})
        resp.raise_for_status()
        data = resp.json()
        return len(data.get("embeddings", [[]])[0]) if data.get("embeddings") else len(text) // 2

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/")
            return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# llama.cpp Backend
# ---------------------------------------------------------------------------

class LlamaCPPBackend(ModelBackend):
    @property
    def backend_type(self) -> BackendType:
        return BackendType.LLAMA_CPP

    async def load(self) -> None:
        if self._loaded:
            return
        try:
            from llama_cpp import Llama

            kwargs: dict[str, Any] = {
                "model_path": self.config.model_id,
                "n_ctx": self.config.max_model_len,
                "seed": self.config.seed,
                "verbose": False,
            }

            if self.config.n_gpu_layers > 0:
                kwargs["n_gpu_layers"] = self.config.n_gpu_layers

            self._llama = Llama(**kwargs)
            self._loaded = True
            self._logger.info("llama.cpp model loaded: %s", self.config.model_id)
        except ImportError:
            raise ImportError("llama-cpp-python package required for llama.cpp backend")

    async def unload(self) -> None:
        with self._lock:
            if hasattr(self, "_llama") and self._llama is not None:
                del self._llama
            self._loaded = False
            import gc

            gc.collect()
            self._logger.info("llama.cpp model unloaded: %s", self.config.model_id)

    async def generate(
        self, prompt: str, gen_config: GenerationConfig | None = None, **kwargs: Any
    ) -> tuple[str, GenerationStats]:
        cfg = gen_config or self.config.generation
        stats = GenerationStats()
        start = time.perf_counter()

        output = self._llama(
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stop=cfg.stop or [],
            echo=False,
        )

        stats.prompt_tokens = output.get("usage", {}).get("prompt_tokens", 0)
        stats.generated_tokens = output.get("usage", {}).get("completion_tokens", 0)
        stats.total_time_ms = (time.perf_counter() - start) * 1000
        stats.tokens_per_second = stats.generated_tokens / (stats.total_time_ms / 1000) if stats.total_time_ms > 0 else 0

        return output.get("choices", [{}])[0].get("text", ""), stats

    async def generate_stream(
        self, prompt: str, gen_config: GenerationConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        cfg = gen_config or self.config.generation

        for output in self._llama(
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stop=cfg.stop or [],
            echo=False,
            stream=True,
        ):
            text = output.get("choices", [{}])[0].get("text", "")
            if text:
                yield text

    async def count_tokens(self, text: str) -> int:
        if hasattr(self, "_llama") and self._llama is not None:
            return self._llama.tokenize(text.encode()).shape[0]
        return len(text) // 2

    async def health(self) -> bool:
        return self._loaded and hasattr(self, "_llama") and self._llama is not None


# ---------------------------------------------------------------------------
# OpenAI-compatible Backend
# ---------------------------------------------------------------------------

class OpenAIBackend(ModelBackend):
    @property
    def backend_type(self) -> BackendType:
        return BackendType.OPENAI

    async def load(self) -> None:
        try:
            import openai

            kwargs: dict[str, Any] = {}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            if self.config.api_base:
                kwargs["base_url"] = self.config.api_base

            self._client = openai.AsyncOpenAI(**kwargs)
            self._loaded = True
            self._logger.info("OpenAI backend ready (base=%s)", self.config.api_base or "https://api.openai.com")
        except ImportError:
            raise ImportError("openai package required for OpenAI backend")

    async def unload(self) -> None:
        with self._lock:
            if hasattr(self, "_client") and self._client is not None:
                await self._client.close()
            self._loaded = False
            self._logger.info("OpenAI backend disconnected")

    async def generate(
        self, prompt: str, gen_config: GenerationConfig | None = None, **kwargs: Any
    ) -> tuple[str, GenerationStats]:
        cfg = gen_config or self.config.generation
        stats = GenerationStats()
        start = time.perf_counter()

        resp = await self._client.chat.completions.create(
            model=self.config.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            stop=cfg.stop or None,
        )

        choice = resp.choices[0]
        stats.prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
        stats.generated_tokens = resp.usage.completion_tokens if resp.usage else 0
        stats.total_time_ms = (time.perf_counter() - start) * 1000
        stats.tokens_per_second = stats.generated_tokens / (stats.total_time_ms / 1000) if stats.total_time_ms > 0 else 0

        return choice.message.content or "", stats

    async def generate_stream(
        self, prompt: str, gen_config: GenerationConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        cfg = gen_config or self.config.generation

        stream = await self._client.chat.completions.create(
            model=self.config.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            stop=cfg.stop or None,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def count_tokens(self, text: str) -> int:
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(self.config.model_id)
            return len(enc.encode(text))
        except (ImportError, KeyError):
            return len(text) // 2

    async def health(self) -> bool:
        try:
            resp = await self._client.models.list()
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Backend Registry
# ---------------------------------------------------------------------------

_BACKEND_REGISTRY: dict[BackendType, type[ModelBackend]] = {
    BackendType.HUGGINGFACE: HFBackend,
    BackendType.VLLM: VLLMBackend,
    BackendType.OLLAMA: OllamaBackend,
    BackendType.LLAMA_CPP: LlamaCPPBackend,
    BackendType.OPENAI: OpenAIBackend,
}


def detect_backend(model_id: str) -> BackendType:
    """Auto-detect the appropriate backend for a model identifier."""
    for pattern, backend in _MODEL_PATTERN_MAP:
        if re.search(pattern, model_id, re.IGNORECASE):
            return backend

    if model_id.endswith(".gguf"):
        return BackendType.LLAMA_CPP
    if model_id.startswith("http"):
        return BackendType.OPENAI
    if ":" in model_id and not model_id.startswith("/"):
        return BackendType.OLLAMA
    if "/" in model_id and not model_id.startswith("."):
        return BackendType.HUGGINGFACE

    return BackendType.OLLAMA


# ---------------------------------------------------------------------------
# Model Manager
# ---------------------------------------------------------------------------

class ModelManager:
    """Production-grade model manager supporting multiple backends.

    Handles model loading/unloading/switching with automatic backend detection,
    GPU detection, CPU fallback, caching, health checks, and generation stats.

    Usage::

        from aios_core.model_manager import ModelManager, ModelConfig, GenerationConfig

        mgr = ModelManager()

        # Load models
        await mgr.load("gpt-4o", ModelConfig(model_id="gpt-4o", backend="openai"))
        await mgr.load("local-model", ModelConfig(model_id="Qwen/Qwen2.5-7B-Instruct"))

        # Generate
        text, stats = await mgr.generate("local-model", "What is AI?")
        print(text)

        # Stream
        async for chunk in mgr.generate_stream("local-model", "Tell me a story"):
            print(chunk, end="")

        # Switch active model
        mgr.set_active("gpt-4o")

        # Health check
        healthy = await mgr.health("gpt-4o")
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelBackend] = {}
        self._configs: dict[str, ModelConfig] = {}
        self._active: str | None = None
        self._lock = threading.RLock()
        self._logger = logging.getLogger("aios.core.model_manager")
        self._stats: dict[str, list[GenerationStats]] = {}

    # -- Loading / Unloading ------------------------------------------------

    async def load(
        self,
        name: str,
        config: ModelConfig | None = None,
    ) -> ModelInfo:
        """Load a model with the given name and optional config.

        If config is omitted, auto-detects the backend from the model name.
        """
        if config is None:
            backend_type = detect_backend(name)
            config = ModelConfig(model_id=name, backend=backend_type)

        if name in self._models:
            self._logger.info("Model '%s' already loaded; skipping", name)
            return self._models[name].model_info

        backend_cls = _BACKEND_REGISTRY.get(config.backend)
        if backend_cls is None:
            raise ValueError(f"Unsupported backend: {config.backend}")

        self._logger.info("Loading model '%s' via %s ...", name, config.backend.value)
        backend = backend_cls(config)

        try:
            await backend.load()
        except Exception:
            self._logger.error("Failed to load model '%s':", name, exc_info=True)
            raise

        with self._lock:
            self._models[name] = backend
            self._configs[name] = config
            self._stats[name] = []
            if self._active is None:
                self._active = name

        info = backend.model_info
        self._logger.info("Model '%s' loaded (%s, device=%s)", name, info.backend.value, info.device)
        return info

    async def unload(self, name: str) -> bool:
        """Unload a model and free its resources."""
        backend = self._models.pop(name, None)
        if backend is None:
            self._logger.warning("Model '%s' not found", name)
            return False

        self._configs.pop(name, None)
        self._stats.pop(name, None)

        if self._active == name:
            self._active = next(iter(self._models.keys())) if self._models else None

        await backend.unload()
        self._logger.info("Model '%s' unloaded", name)
        return True

    async def unload_all(self) -> int:
        """Unload all loaded models."""
        names = list(self._models.keys())
        for name in names:
            await self.unload(name)
        return len(names)

    async def reload(self, name: str) -> ModelInfo:
        """Reload a model (unload then load)."""
        config = self._configs.get(name)
        if config is None:
            raise ValueError(f"Model '{name}' not found")
        await self.unload(name)
        return await self.load(name, config)

    # -- Active model -------------------------------------------------------

    def set_active(self, name: str) -> None:
        """Set the active model for inference."""
        if name not in self._models:
            raise ValueError(f"Model '{name}' not loaded")
        self._active = name
        self._logger.info("Active model set to '%s'", name)

    @property
    def active_model(self) -> str | None:
        return self._active

    @property
    def active_backend(self) -> ModelBackend | None:
        if self._active is None:
            return None
        return self._models.get(self._active)

    # -- Generation ---------------------------------------------------------

    async def generate(
        self,
        model_name: str | None,
        prompt: str,
        gen_config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> tuple[str, GenerationStats]:
        """Generate a response from a model (non-streaming)."""
        backend = self._resolve(model_name)
        text, stats = await backend.generate(prompt, gen_config, **kwargs)
        self._record_stats(model_name or self._active or "", stats)
        return text, stats

    async def generate_stream(
        self,
        model_name: str | None,
        prompt: str,
        gen_config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response from a model."""
        backend = self._resolve(model_name)
        start = time.perf_counter()
        collected: list[str] = []

        async for chunk in backend.generate_stream(prompt, gen_config, **kwargs):
            collected.append(chunk)
            yield chunk

        elapsed = (time.perf_counter() - start) * 1000
        stats = GenerationStats(
            generated_tokens=len(collected),
            total_time_ms=elapsed,
            tokens_per_second=len(collected) / (elapsed / 1000) if elapsed > 0 else 0,
        )
        self._record_stats(model_name or self._active or "", stats)

    # -- Utilities ----------------------------------------------------------

    async def count_tokens(self, model_name: str | None, text: str) -> int:
        backend = self._resolve(model_name)
        return await backend.count_tokens(text)

    async def health(self, model_name: str | None = None) -> dict[str, bool]:
        """Check health of one or all models."""
        if model_name:
            backend = self._models.get(model_name)
            return {model_name: await backend.health() if backend else False}

        results: dict[str, bool] = {}
        for name, backend in self._models.items():
            try:
                results[name] = await backend.health()
            except Exception:
                results[name] = False
        return results

    def get_info(self, model_name: str) -> ModelInfo | None:
        backend = self._models.get(model_name)
        return backend.model_info if backend else None

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "backend": config.backend.value,
                "model_id": config.model_id,
                "active": name == self._active,
                "loaded": backend.is_loaded,
            }
            for name, (backend, config) in [(n, (self._models[n], self._configs[n])) for n in self._models]
        ]

    @property
    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    def get_stats(self, model_name: str | None = None) -> dict[str, Any]:
        if model_name:
            stats_list = self._stats.get(model_name, [])
            return self._aggregate_stats(model_name, stats_list)

        return {
            name: self._aggregate_stats(name, sl)
            for name, sl in self._stats.items()
        }

    @staticmethod
    def detect_gpu() -> dict[str, Any]:
        return ModelBackend.detect_gpu()

    # -- Internal -----------------------------------------------------------

    def _resolve(self, model_name: str | None) -> ModelBackend:
        key = model_name or self._active
        if key is None:
            raise ValueError("No model specified and no active model set")
        backend = self._models.get(key)
        if backend is None:
            raise ValueError(f"Model '{key}' not loaded")
        return backend

    def _record_stats(self, model_name: str, stats: GenerationStats) -> None:
        if model_name in self._stats:
            self._stats[model_name].append(stats)
            if len(self._stats[model_name]) > 1000:
                self._stats[model_name] = self._stats[model_name][-500:]

    @staticmethod
    def _aggregate_stats(name: str, stats_list: list[GenerationStats]) -> dict[str, Any]:
        if not stats_list:
            return {"model": name, "requests": 0}
        n = len(stats_list)
        total_tokens = sum(s.generated_tokens for s in stats_list)
        total_time = sum(s.total_time_ms for s in stats_list)
        return {
            "model": name,
            "requests": n,
            "total_tokens": total_tokens,
            "total_time_ms": round(total_time, 2),
            "avg_tokens_per_request": round(total_tokens / n, 1) if n else 0,
            "avg_latency_ms": round(total_time / n, 1) if n else 0,
            "avg_tokens_per_second": round(total_tokens / (total_time / 1000), 1) if total_time > 0 else 0,
            "prompt_tokens": sum(s.prompt_tokens for s in stats_list),
        }

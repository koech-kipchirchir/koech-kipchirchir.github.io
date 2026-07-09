"""
Image captioning providers with stubs for future VLM support
(Qwen VL, Llama Vision, Gemma Vision).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from vision.config import VisionConfig

logger = logging.getLogger("aios.vision.image_caption")


@dataclass
class CaptionResult:
    caption: str = ""
    confidence: float = 0.0
    latency_s: float = 0.0
    model: str = ""


class CaptionProvider(ABC):
    @abstractmethod
    async def caption(self, image: np.ndarray, prompt: Optional[str] = None) -> CaptionResult:
        pass

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# BLIP (Salesforce)
# ---------------------------------------------------------------------------

class BLIPProvider(CaptionProvider):
    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._processor: Any = None
        self._model: Any = None

    async def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor
        loop = asyncio.get_event_loop()
        device = self._config.caption_device
        model_name = self._config.caption_model or "Salesforce/blip-image-captioning-base"

        def _load() -> tuple[Any, Any]:
            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
            return processor, model

        self._processor, self._model = await loop.run_in_executor(None, _load)
        logger.info("BLIP model loaded: %s (device=%s)", model_name, device)

    async def caption(self, image: np.ndarray, prompt: Optional[str] = None) -> CaptionResult:
        t0 = time.perf_counter()
        await self._ensure_model()
        from PIL import Image
        import torch
        loop = asyncio.get_event_loop()
        pil_image = Image.fromarray(image)

        def _generate() -> str:
            inputs = self._processor(pil_image, return_tensors="pt").to(self._model.device)
            if prompt:
                inputs = self._processor(pil_image, prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(**inputs, max_length=50)
            return self._processor.decode(out[0], skip_special_tokens=True)

        caption = await loop.run_in_executor(None, _generate)
        return CaptionResult(
            caption=caption,
            confidence=1.0,
            latency_s=time.perf_counter() - t0,
            model=self._config.caption_model or "blip",
        )

    async def close(self) -> None:
        self._model = None
        self._processor = None


# ---------------------------------------------------------------------------
# Future VLM providers (stubs)
# ---------------------------------------------------------------------------

class QwenVLProvider(CaptionProvider):
    """Qwen Vision-Language model (future)."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config

    async def _ensure_model(self) -> None:
        raise NotImplementedError("Qwen VL support coming soon")

    async def caption(self, image: np.ndarray, prompt: Optional[str] = None) -> CaptionResult:
        raise NotImplementedError("Qwen VL support coming soon")


class LlamaVisionProvider(CaptionProvider):
    """Llama Vision model (future)."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config

    async def caption(self, image: np.ndarray, prompt: Optional[str] = None) -> CaptionResult:
        raise NotImplementedError("Llama Vision support coming soon")


class GemmaVisionProvider(CaptionProvider):
    """Gemma Vision model (future)."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config

    async def caption(self, image: np.ndarray, prompt: Optional[str] = None) -> CaptionResult:
        raise NotImplementedError("Gemma Vision support coming soon")


# ---------------------------------------------------------------------------
# Caption Engine
# ---------------------------------------------------------------------------

class CaptionEngine:
    """Image captioning engine."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._provider: Optional[CaptionProvider] = None
        self._provider_map: dict[str, type[CaptionProvider]] = {
            "blip": BLIPProvider,
            "qwen_vl": QwenVLProvider,
            "llama_vision": LlamaVisionProvider,
            "gemma_vision": GemmaVisionProvider,
        }

    async def _get_provider(self) -> CaptionProvider:
        if self._provider is not None:
            return self._provider
        model = self._config.caption_model.lower() if self._config.caption_model else "blip"
        cls = self._provider_map.get(model, BLIPProvider)
        self._provider = cls(self._config)
        return self._provider

    async def caption(self, image: np.ndarray, prompt: Optional[str] = None) -> CaptionResult:
        provider = await self._get_provider()
        return await provider.caption(image, prompt)

    async def close(self) -> None:
        if self._provider:
            await self._provider.close()

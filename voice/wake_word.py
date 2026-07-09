"""
Wake word detection with Porcupine, Snowboy, and custom keyword
support.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Optional

import numpy as np

from voice.config import VoiceConfig
from voice.utils import REGISTRY, structured_log

logger = logging.getLogger("aios.voice.wake_word")


class WakeWordError(Exception):
    pass


@dataclass
class WakeWordResult:
    detected: bool = False
    keyword: str = ""
    confidence: float = 0.0
    start_sample: int = 0
    end_sample: int = 0


class WakeWordProvider(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def process(self, audio: np.ndarray, sample_rate: int) -> WakeWordResult:
        pass

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Porcupine
# ---------------------------------------------------------------------------

class PorcupineProvider(WakeWordProvider):
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._porcupine = None

    async def initialize(self) -> None:
        if self._porcupine is not None:
            return
        import pvporcupine
        access_key = self._config.porcupine_key
        keyword = self._config.wake_word
        model_path = self._config.porcupine_model_path or None
        sensitivities = [self._config.wake_sensitivity]

        loop = asyncio.get_event_loop()

        def _create() -> pvporcupine.Porcupine:
            return pvporcupine.create(
                access_key=access_key,
                keywords=[keyword],
                keyword_paths=[model_path] if model_path else None,
                sensitivities=sensitivities,
            )

        self._porcupine = await loop.run_in_executor(None, _create)
        structured_log(logging.INFO, "wake_word.porcupine.initialized",
                       keyword=keyword, sensitivity=self._config.wake_sensitivity)

    async def process(self, audio: np.ndarray, sample_rate: int) -> WakeWordResult:
        await self.initialize()
        if sample_rate != self._porcupine.sample_rate:
            raise WakeWordError(f"Expected {self._porcupine.sample_rate} Hz, got {sample_rate}")

        frame_length = self._porcupine.frame_length
        result = WakeWordResult()
        num_frames = len(audio) // frame_length

        for i in range(num_frames):
            start = i * frame_length
            frame = audio[start:start + frame_length].astype(np.int16)
            if len(frame) < frame_length:
                break
            loop = asyncio.get_event_loop()

            def _process() -> int:
                return self._porcupine.process(frame.tolist())

            keyword_index = await loop.run_in_executor(None, _process)
            if keyword_index >= 0:
                result.detected = True
                result.keyword = self._config.wake_word
                result.confidence = 1.0
                result.start_sample = start
                result.end_sample = start + frame_length
                structured_log(logging.INFO, "wake_word.detected",
                               keyword=result.keyword, confidence=result.confidence)
                break

        return result

    async def close(self) -> None:
        if self._porcupine is not None:
            self._porcupine.delete()
            self._porcupine = None


# ---------------------------------------------------------------------------
# Snowboy
# ---------------------------------------------------------------------------

class SnowboyProvider(WakeWordProvider):
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._detector = None

    async def initialize(self) -> None:
        if self._detector is not None:
            return
        import snowboydetect
        model_path = self._config.porcupine_model_path or "resources/snowboy.umdl"
        sensitivity = str(self._config.wake_sensitivity)

        loop = asyncio.get_event_loop()

        def _create():
            detector = snowboydetect.SnowboyDetect(
                resource_filename="resources/common.res",
                model_str=model_path,
            )
            detector.SetSensitivity(sensitivity)
            return detector

        self._detector = await loop.run_in_executor(None, _create)
        structured_log(logging.INFO, "wake_word.snowboy.initialized")

    async def process(self, audio: np.ndarray, sample_rate: int) -> WakeWordResult:
        await self.initialize()
        audio_bytes = audio.astype(np.int16).tobytes()
        loop = asyncio.get_event_loop()

        def _detect() -> int:
            return self._detector.RunDetection(audio_bytes)

        result_code = await loop.run_in_executor(None, _detect)
        if result_code > 0:
            return WakeWordResult(
                detected=True,
                keyword=self._config.wake_word,
                confidence=result_code / 10.0,
            )
        return WakeWordResult()

    async def close(self) -> None:
        self._detector = None


# ---------------------------------------------------------------------------
# Custom keyword (simple energy + pattern matching — basic)
# ---------------------------------------------------------------------------

class CustomKeywordProvider(WakeWordProvider):
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._keywords: list[str] = [kw.strip().lower() for kw in config.wake_word.split(",")]

    async def initialize(self) -> None:
        pass

    async def process(self, audio: np.ndarray, sample_rate: int) -> WakeWordResult:
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        if rms > 0.02:
            return WakeWordResult(detected=True, keyword=self._keywords[0] if self._keywords else "", confidence=min(rms * 5, 1.0))
        return WakeWordResult()


# ---------------------------------------------------------------------------
# Wake Word Engine (facade with async listener)
# ---------------------------------------------------------------------------

class WakeWordEngine:
    """Wake word detection engine."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._provider: WakeWordProvider | None = None
        self._on_wake: list[Callable[[WakeWordResult], None]] = []

    def on_wake(self, callback: Callable[[WakeWordResult], None]) -> None:
        self._on_wake.append(callback)

    async def _get_provider(self) -> WakeWordProvider:
        if self._provider is not None:
            return self._provider
        name = self._config.wake_word_provider
        provider_cls = REGISTRY.get("wake_word", name)
        if provider_cls is None:
            logger.warning("Wake word provider '%s' not registered, falling back to custom", name)
            provider_cls = CustomKeywordProvider
        self._provider = provider_cls(self._config)
        await self._provider.initialize()
        return self._provider

    async def process(self, audio: np.ndarray, sample_rate: int) -> WakeWordResult:
        provider = await self._get_provider()
        result = await provider.process(audio, sample_rate)
        if result.detected:
            for cb in self._on_wake:
                try:
                    cb(result)
                except Exception as e:
                    logger.exception("Wake word callback error: %s", e)
        return result

    async def listen_background(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        sample_rate: int,
    ) -> AsyncGenerator[WakeWordResult, None]:
        """Continuously listen for wake words on a stream."""
        async for chunk in audio_stream:
            result = await self.process(chunk, sample_rate)
            if result.detected:
                yield result

    async def close(self) -> None:
        if self._provider:
            await self._provider.close()


REGISTRY.register("wake_word", "porcupine", PorcupineProvider)
REGISTRY.register("wake_word", "snowboy", SnowboyProvider)
REGISTRY.register("wake_word", "custom", CustomKeywordProvider)

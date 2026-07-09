"""
Noise reduction interface with multiple suppression algorithms.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from voice.config import VoiceConfig
from voice.utils import REGISTRY, int16_to_float32, float32_to_int16, structured_log

logger = logging.getLogger("aios.voice.noise_reduction")


class NoiseReductionProvider(ABC):
    @abstractmethod
    async def reduce(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        pass

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# NoiseReduce (noisereduce)
# ---------------------------------------------------------------------------

class NoiseReduceProvider(NoiseReductionProvider):
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._stationary = False

    async def reduce(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import noisereduce as nr
        strength = self._config.noise_reduction_strength
        audio_f32 = int16_to_float32(audio)
        loop = asyncio.get_event_loop()

        def _reduce() -> np.ndarray:
            return nr.reduce_noise(
                y=audio_f32,
                sr=sample_rate,
                stationary=self._stationary,
                prop_decrease=strength,
            )

        reduced = await loop.run_in_executor(None, _reduce)
        return float32_to_int16(reduced)


# ---------------------------------------------------------------------------
# SciPy / spectral gating (simple)
# ---------------------------------------------------------------------------

class SpectralGateProvider(NoiseReductionProvider):
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config

    async def reduce(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        from scipy import signal
        strength = self._config.noise_reduction_strength
        audio_f32 = int16_to_float32(audio)

        loop = asyncio.get_event_loop()

        def _reduce() -> np.ndarray:
            f, t, Zxx = signal.stft(audio_f32, fs=sample_rate, npershal=512)
            magnitude = np.abs(Zxx)
            threshold = np.median(magnitude, axis=1, keepdims=True) * (1.0 + (1.0 - strength) * 2)
            mask = magnitude > threshold
            Zxx_clean = Zxx * mask
            _, x_clean = signal.istft(Zxx_clean, fs=sample_rate)
            return x_clean

        reduced = await loop.run_in_executor(None, _reduce)
        if len(reduced) < len(audio_f32):
            reduced = np.pad(reduced, (0, len(audio_f32) - len(reduced)))
        elif len(reduced) > len(audio_f32):
            reduced = reduced[:len(audio_f32)]
        return float32_to_int16(reduced)


# ---------------------------------------------------------------------------
# Noise Reduction Engine
# ---------------------------------------------------------------------------

class NoiseReductionEngine:
    """Noise reduction engine."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._provider: NoiseReductionProvider | None = None

    async def _get_provider(self) -> NoiseReductionProvider:
        if self._provider is not None:
            return self._provider
        if hasattr(self, "_provider_cls"):
            provider_cls = self._provider_cls
        else:
            provider_cls = REGISTRY.get("noise_reduction", "noisereduce") or NoiseReduceProvider
        self._provider = provider_cls(self._config)
        return self._provider

    async def reduce(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self._config.noise_reduction_enabled:
            return audio
        provider = await self._get_provider()
        return await provider.reduce(audio, sample_rate)

    async def close(self) -> None:
        if self._provider:
            await self._provider.close()


REGISTRY.register("noise_reduction", "noisereduce", NoiseReduceProvider)
REGISTRY.register("noise_reduction", "spectralgate", SpectralGateProvider)

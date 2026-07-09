"""
Text-to-speech with multi-provider support: Piper, Coqui TTS, XTTS,
and Edge TTS.

Each provider implements the ``TTSProvider`` abstract base and
registers itself in the plugin registry.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import numpy as np

from voice.config import VoiceConfig
from voice.utils import REGISTRY, structured_log

logger = logging.getLogger("aios.voice.tts")


@dataclass
class TTSResult:
    audio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))
    sample_rate: int = 16000
    duration_s: float = 0.0
    text: str = ""


class TTSProvider(ABC):
    @abstractmethod
    async def synthesize(self, text: str, **kwargs: Any) -> TTSResult:
        pass

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Piper TTS
# ---------------------------------------------------------------------------

class PiperProvider(TTSProvider):
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._process: subprocess.Popen | None = None

    async def synthesize(self, text: str, **kwargs: Any) -> TTSResult:
        model_path = kwargs.get("model_path", self._config.piper_model_path)
        voice = kwargs.get("voice", self._config.piper_voice)
        speed = kwargs.get("speed", self._config.piper_speed)

        if not model_path:
            raise ValueError("Piper model path not configured (AIOS_VOICE_PIPER_MODEL_PATH)")

        cmd = [
            "piper",
            "--model", model_path,
            "--output-raw",
            "--length-scale", str(1.0 / speed),
        ]

        loop = asyncio.get_event_loop()

        def _run() -> tuple[np.ndarray, int]:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = proc.communicate(input=text.encode(), timeout=30)
            if proc.returncode != 0:
                raise RuntimeError(f"Piper error: {stderr.decode()}")
            audio = np.frombuffer(stdout, dtype=np.int16)
            sr = 16000  # Piper always outputs 16kHz
            return audio, sr

        audio, sr = await loop.run_in_executor(None, _run)
        return TTSResult(audio=audio, sample_rate=sr, duration_s=len(audio) / sr, text=text)


# ---------------------------------------------------------------------------
# Coqui TTS
# ---------------------------------------------------------------------------

class CoquiProvider(TTSProvider):
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._model = None

    async def _load_model(self) -> None:
        if self._model is not None:
            return
        from TTS.api import TTS
        model_name = self._config.coqui_model
        logger.info("Loading Coqui TTS model '%s'...", model_name)
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, lambda: TTS(model_name))
        structured_log(logging.INFO, "tts.coqui.loaded", model=model_name)

    async def synthesize(self, text: str, **kwargs: Any) -> TTSResult:
        await self._load_model()
        speaker = kwargs.get("speaker", self._config.coqui_voice_dir)
        loop = asyncio.get_event_loop()

        def _synth() -> tuple[np.ndarray, int]:
            wav = self._model.tts(text, speaker=speaker if speaker else None)
            audio = np.array(wav, dtype=np.float32)
            sr = self._model.synthesizer.output_sample_rate
            return (audio * 32767).astype(np.int16), sr

        audio, sr = await loop.run_in_executor(None, _synth)
        return TTSResult(audio=audio, sample_rate=sr, duration_s=len(audio) / sr, text=text)


# ---------------------------------------------------------------------------
# XTTS (Coqui XTTS)
# ---------------------------------------------------------------------------

class XTTSProvider(TTSProvider):
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._model = None

    async def _load_model(self) -> None:
        if self._model is not None:
            return
        from TTS.api import TTS
        model_name = self._config.xtts_model
        logger.info("Loading XTTS model '%s'...", model_name)
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, lambda: TTS(model_name))
        structured_log(logging.INFO, "tts.xtts.loaded", model=model_name)

    async def synthesize(self, text: str, **kwargs: Any) -> TTSResult:
        await self._load_model()
        speaker = kwargs.get("speaker", self._config.xtts_speaker)
        language = kwargs.get("language", self._config.language)
        loop = asyncio.get_event_loop()

        def _synth() -> tuple[np.ndarray, int]:
            wav = self._model.tts(text, speaker=speaker if speaker else None, language=language)
            audio = np.array(wav, dtype=np.float32)
            sr = self._model.synthesizer.output_sample_rate
            return (audio * 32767).astype(np.int16), sr

        audio, sr = await loop.run_in_executor(None, _synth)
        return TTSResult(audio=audio, sample_rate=sr, duration_s=len(audio) / sr, text=text)


# ---------------------------------------------------------------------------
# Edge TTS (Microsoft Edge online TTS)
# ---------------------------------------------------------------------------

class EdgeTTSProvider(TTSProvider):
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config

    async def synthesize(self, text: str, **kwargs: Any) -> TTSResult:
        import edge_tts
        voice = kwargs.get("voice", self._config.edge_voice)
        rate = kwargs.get("rate", self._config.edge_rate)
        volume = kwargs.get("volume", self._config.edge_volume)

        communicate = edge_tts.Communicate(text, voice=voice, rate=rate, volume=volume)
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])

        sr = 16000
        audio = np.frombuffer(audio_data, dtype=np.int16)
        return TTSResult(audio=audio, sample_rate=sr, duration_s=len(audio) / sr, text=text)


# ---------------------------------------------------------------------------
# TTS Engine (facade)
# ---------------------------------------------------------------------------

class TTSEngine:
    """Text-to-speech engine that delegates to a configured provider."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._provider: TTSProvider | None = None

    @property
    def provider_name(self) -> str:
        return self._config.tts_provider

    async def _get_provider(self) -> TTSProvider:
        if self._provider is not None:
            return self._provider
        name = self._config.tts_provider
        provider_cls = REGISTRY.get("tts", name)
        if provider_cls is None:
            logger.warning("TTS provider '%s' not registered, falling back to edge", name)
            provider_cls = EdgeTTSProvider
        self._provider = provider_cls(self._config)
        return self._provider

    async def synthesize(self, text: str, **kwargs: Any) -> TTSResult:
        provider = await self._get_provider()
        start = time.perf_counter()
        result = await provider.synthesize(text, **kwargs)
        structured_log(logging.DEBUG, "tts.synthesize",
                        provider=self._config.tts_provider,
                        duration_ms=round((time.perf_counter() - start) * 1000, 1),
                        audio_len=len(result.audio))
        return result

    async def close(self) -> None:
        if self._provider:
            await self._provider.close()


REGISTRY.register("tts", "piper", PiperProvider)
REGISTRY.register("tts", "coqui", CoquiProvider)
REGISTRY.register("tts", "xtts", XTTSProvider)
REGISTRY.register("tts", "edge", EdgeTTSProvider)

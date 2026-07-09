"""
Voice activity detection with Silero VAD and WebRTC VAD.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

import numpy as np

from voice.config import VoiceConfig
from voice.utils import REGISTRY, int16_to_float32, resample, structured_log

logger = logging.getLogger("aios.voice.vad")


@dataclass
class VADResult:
    is_speech: bool = False
    confidence: float = 0.0
    start_sample: int = 0
    end_sample: int = 0
    duration_s: float = 0.0


class VADProvider(ABC):
    @abstractmethod
    async def is_speech(self, audio: np.ndarray, sample_rate: int) -> VADResult:
        pass

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Silero VAD
# ---------------------------------------------------------------------------

class SileroVADProvider(VADProvider):
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._model = None
        self._get_speech_timestamps = None
        self._sr = 16000

    async def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        torch.set_num_threads(1)
        loop = asyncio.get_event_loop()

        def _load():
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,
            )
            return model, utils

        self._model, utils = await loop.run_in_executor(None, _load)
        self._get_speech_timestamps = utils[0]
        structured_log(logging.INFO, "vad.silero.loaded")

    async def is_speech(self, audio: np.ndarray, sample_rate: int) -> VADResult:
        await self._load_model()
        import torch
        audio_f32 = int16_to_float32(audio)
        if sample_rate != self._sr:
            audio_f32 = resample(audio_f32, sample_rate, self._sr)

        audio_tensor = torch.from_numpy(audio_f32).float()
        loop = asyncio.get_event_loop()

        def _detect() -> tuple[bool, float]:
            with torch.no_grad():
                speech_probs = self._model(audio_tensor, self._sr).item()
            return speech_probs > self._config.vad_threshold, speech_probs

        is_speech, confidence = await loop.run_in_executor(None, _detect)
        return VADResult(
            is_speech=is_speech,
            confidence=float(confidence),
            duration_s=len(audio) / max(sample_rate, 1),
        )


# ---------------------------------------------------------------------------
# WebRTC VAD
# ---------------------------------------------------------------------------

class WebRTCVADProvider(VADProvider):
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._vad = None

    async def _load_model(self) -> None:
        if self._vad is not None:
            return
        import webrtcvad
        loop = asyncio.get_event_loop()
        self._vad = await loop.run_in_executor(None, lambda: webrtcvad.Vad(2))

    async def is_speech(self, audio: np.ndarray, sample_rate: int) -> VADResult:
        await self._load_model()
        audio_int16 = audio.astype(np.int16)
        frame_ms = self._config.frame_duration_ms
        frame_bytes = int(sample_rate * frame_ms / 1000) * 2
        num_frames = len(audio_int16) // (frame_bytes // 2)
        speech_frames = 0
        total_frames = 0

        loop = asyncio.get_event_loop()

        for i in range(num_frames):
            start = i * (frame_bytes // 2)
            frame = audio_int16[start:start + (frame_bytes // 2)].tobytes()
            if len(frame) < frame_bytes:
                break

            def _check(f: bytes, sr: int) -> bool:
                return self._vad.is_speech(f, sr)

            is_speech = await loop.run_in_executor(None, _check, frame, sample_rate)
            total_frames += 1
            if is_speech:
                speech_frames += 1

        ratio = speech_frames / max(total_frames, 1)
        return VADResult(
            is_speech=ratio > self._config.vad_threshold,
            confidence=ratio,
            duration_s=len(audio) / max(sample_rate, 1),
        )


# ---------------------------------------------------------------------------
# VAD Engine (facade with speech/silence segmentation)
# ---------------------------------------------------------------------------

class VADEngine:
    """Voice activity detection engine."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._provider: VADProvider | None = None
        self._speech_buffer: list[np.ndarray] = []
        self._silence_frames = 0
        self._is_speaking = False

    async def _get_provider(self) -> VADProvider:
        if self._provider is not None:
            return self._provider
        name = self._config.vad_provider
        provider_cls = REGISTRY.get("vad", name)
        if provider_cls is None:
            logger.warning("VAD provider '%s' not registered, falling back to silero", name)
            provider_cls = SileroVADProvider
        self._provider = provider_cls(self._config)
        return self._provider

    async def is_speech(self, audio: np.ndarray, sample_rate: int) -> VADResult:
        provider = await self._get_provider()
        return await provider.is_speech(audio, sample_rate)

    async def process_stream(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        sample_rate: int,
    ) -> AsyncGenerator[tuple[np.ndarray, bool], None]:
        """Process audio stream and yield (audio_chunk, is_speech_ended) tuples."""
        min_speech = self._config.vad_min_speech_duration_ms / 1000
        min_silence = self._config.vad_min_silence_duration_ms / 1000
        buffer_ms = self._config.vad_buffer_ms / 1000
        frame_samples = int(sample_rate * self._config.frame_duration_ms / 1000)

        self._speech_buffer = []
        self._silence_frames = 0
        self._is_speaking = False
        speech_frames = 0
        onset_buffer: list[np.ndarray] = []

        async for chunk in audio_stream:
            result = await self.is_speech(chunk, sample_rate)

            if result.is_speech:
                speech_frames += 1
                self._silence_frames = 0
                onset_buffer.append(chunk)
                buffer_duration = len(onset_buffer) * frame_samples / sample_rate

                if not self._is_speaking and buffer_duration >= buffer_ms:
                    self._is_speaking = True
                    for buf_chunk in onset_buffer:
                        self._speech_buffer.append(buf_chunk)
                    onset_buffer = []
                elif self._is_speaking:
                    self._speech_buffer.append(chunk)

            else:
                if self._is_speaking:
                    self._silence_frames += 1
                    silence_duration = self._silence_frames * frame_samples / sample_rate
                    if silence_duration >= min_silence:
                        combined = np.concatenate(self._speech_buffer)
                        speech_duration = len(combined) / sample_rate
                        if speech_duration >= min_speech:
                            yield combined, True
                        self._speech_buffer = []
                        self._silence_frames = 0
                        self._is_speaking = False
                        speech_frames = 0
                        onset_buffer = []
                else:
                    onset_buffer = []
                    speech_frames = 0

        # Flush remaining buffer
        if self._speech_buffer:
            combined = np.concatenate(self._speech_buffer)
            if len(combined) / sample_rate >= min_speech:
                yield combined, True

    async def close(self) -> None:
        if self._provider:
            await self._provider.close()


REGISTRY.register("vad", "silero", SileroVADProvider)
REGISTRY.register("vad", "webrtcvad", WebRTCVADProvider)

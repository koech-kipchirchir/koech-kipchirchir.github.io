"""
Speech-to-text with multi-provider support: Whisper, Faster-Whisper,
Vosk, and ONNX Runtime.

Each provider implements the ``STTProvider`` abstract base and
registers itself in the plugin registry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import numpy as np

from voice.config import VoiceConfig
from voice.utils import (
    REGISTRY, int16_to_float32, resample, structured_log,
)

logger = logging.getLogger("aios.voice.stt")


@dataclass
class STTResult:
    text: str = ""
    segments: list[dict] = field(default_factory=list)
    language: str = ""
    duration_s: float = 0.0
    confidence: float = 0.0
    is_final: bool = True


class STTProvider(ABC):
    """Abstract base for speech-to-text providers."""

    @abstractmethod
    async def transcribe(self, audio: np.ndarray, sample_rate: int, **kwargs: Any) -> STTResult:
        pass

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        sample_rate: int,
        **kwargs: Any,
    ) -> AsyncGenerator[STTResult, None]:
        pass

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Whisper (openai-whisper)
# ---------------------------------------------------------------------------

class WhisperProvider(STTProvider):
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._model = None

    async def _load_model(self) -> None:
        if self._model is not None:
            return
        import whisper
        model_name = self._config.whisper_model
        device = self._config.whisper_device
        logger.info("Loading Whisper model '%s' (device=%s)...", model_name, device)
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None, lambda: whisper.load_model(model_name, device=device if device != "auto" else None),
        )
        structured_log(logging.INFO, "stt.whisper.loaded", model=model_name)

    async def transcribe(self, audio: np.ndarray, sample_rate: int, **kwargs: Any) -> STTResult:
        await self._load_model()
        audio_f32 = int16_to_float32(audio).astype(np.float32)
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe(audio_f32, language=kwargs.get("language", self._config.language)),
        )
        return STTResult(
            text=result.get("text", "").strip(),
            segments=[
                {"start": s.get("start", 0), "end": s.get("end", 0), "text": s.get("text", "")}
                for s in result.get("segments", [])
            ],
            language=result.get("language", ""),
            duration_s=result.get("duration", 0.0),
        )

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        sample_rate: int,
        **kwargs: Any,
    ) -> AsyncGenerator[STTResult, None]:
        await self._load_model()
        buffer: list[np.ndarray] = []
        async for chunk in audio_stream:
            buffer.append(chunk)
            combined = np.concatenate(buffer) if len(buffer) > 1 else buffer[0]
            result = await self.transcribe(combined, sample_rate, **kwargs)
            if result.text:
                yield result
                buffer = []


# ---------------------------------------------------------------------------
# Faster-Whisper
# ---------------------------------------------------------------------------

class FasterWhisperProvider(STTProvider):
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._model = None

    async def _load_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        model_name = self._config.whisper_model
        device = self._config.whisper_device
        compute = self._config.whisper_compute_type
        logger.info("Loading Faster-Whisper model '%s' (device=%s)...", model_name, device)
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(model_name, device=device if device != "auto" else "cpu",
                                 compute_type=compute),
        )
        structured_log(logging.INFO, "stt.faster_whisper.loaded", model=model_name)

    async def transcribe(self, audio: np.ndarray, sample_rate: int, **kwargs: Any) -> STTResult:
        await self._load_model()
        audio_f32 = int16_to_float32(audio).astype(np.float32)
        loop = asyncio.get_event_loop()
        lang = kwargs.get("language", self._config.language)

        segments, info = await loop.run_in_executor(
            None, lambda: self._model.transcribe(audio_f32, language=lang if lang != "auto" else None),
        )
        seg_list = list(segments)
        text = " ".join(s.text for s in seg_list)
        return STTResult(
            text=text.strip(),
            segments=[
                {"start": s.start, "end": s.end, "text": s.text}
                for s in seg_list
            ],
            language=getattr(info, "language", "") if info else "",
            duration_s=0.0,
            confidence=float(np.mean([s.avg_logprob for s in seg_list])) if seg_list else 0.0,
        )

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        sample_rate: int,
        **kwargs: Any,
    ) -> AsyncGenerator[STTResult, None]:
        await self._load_model()
        buffer: list[np.ndarray] = []
        async for chunk in audio_stream:
            buffer.append(chunk)
            combined = np.concatenate(buffer) if len(buffer) > 1 else buffer[0]
            if len(combined) >= sample_rate:  # process every ~1s of audio
                result = await self.transcribe(combined, sample_rate, **kwargs)
                if result.text:
                    yield result
                    buffer = []


# ---------------------------------------------------------------------------
# Vosk
# ---------------------------------------------------------------------------

class VoskProvider(STTProvider):
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._model = None
        self._rec = None

    async def _load_model(self) -> None:
        if self._model is not None:
            return
        import vosk
        model_path = self._config.vosk_model_path
        if not model_path:
            raise ValueError("Vosk model path not set (AIOS_VOICE_VOSK_MODEL_PATH)")
        logger.info("Loading Vosk model from '%s'...", model_path)
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, lambda: vosk.Model(model_path))
        self._rec = vosk.KaldiRecognizer(self._model, self._config.sample_rate)
        structured_log(logging.INFO, "stt.vosk.loaded", path=model_path)

    async def transcribe(self, audio: np.ndarray, sample_rate: int, **kwargs: Any) -> STTResult:
        await self._load_model()
        audio_bytes = audio.astype(np.int16).tobytes()
        loop = asyncio.get_event_loop()
        if self._rec.AcceptWaveform(audio_bytes):
            result_json = await loop.run_in_executor(None, self._rec.Result)
            result = json.loads(result_json)
            return STTResult(
                text=result.get("text", ""),
                confidence=result.get("confidence", 0.0),
            )
        return STTResult(text="")

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        sample_rate: int,
        **kwargs: Any,
    ) -> AsyncGenerator[STTResult, None]:
        await self._load_model()
        async for chunk in audio_stream:
            result = await self.transcribe(chunk, sample_rate, **kwargs)
            if result.text:
                yield result


# ---------------------------------------------------------------------------
# ONNX Runtime (e.g. sherpa-onnx, silero)
# ---------------------------------------------------------------------------

class ONNXProvider(STTProvider):
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._session = None

    async def _load_model(self) -> None:
        if self._session is not None:
            return
        import onnxruntime
        model_path = self._config.onnx_model_path
        if not model_path:
            raise ValueError("ONNX model path not set (AIOS_VOICE_ONNX_MODEL_PATH)")
        logger.info("Loading ONNX model from '%s'...", model_path)
        loop = asyncio.get_event_loop()
        self._session = await loop.run_in_executor(
            None, lambda: onnxruntime.InferenceSession(model_path),
        )
        structured_log(logging.INFO, "stt.onnx.loaded", path=model_path)

    async def transcribe(self, audio: np.ndarray, sample_rate: int, **kwargs: Any) -> STTResult:
        await self._load_model()
        audio_f32 = int16_to_float32(audio).astype(np.float32)
        input_name = self._session.get_inputs()[0].name
        loop = asyncio.get_event_loop()

        def _infer() -> Any:
            return self._session.run(None, {input_name: audio_f32.reshape(1, -1)})

        outputs = await loop.run_in_executor(None, _infer)
        text = outputs[0][0] if outputs else ""
        return STTResult(text=str(text).strip() if text else "")

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        sample_rate: int,
        **kwargs: Any,
    ) -> AsyncGenerator[STTResult, None]:
        buffer = b""
        async for chunk in audio_stream:
            buffer += chunk.tobytes()
            result = await self.transcribe(
                np.frombuffer(buffer, dtype=np.int16), sample_rate, **kwargs,
            )
            if result.text:
                yield result
                buffer = b""


# ---------------------------------------------------------------------------
# STT Engine (facade)
# ---------------------------------------------------------------------------

class STTEngine:
    """Speech-to-text engine that delegates to a configured provider."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._provider: STTProvider | None = None

    @property
    def provider_name(self) -> str:
        return self._config.stt_provider

    async def _get_provider(self) -> STTProvider:
        if self._provider is not None:
            return self._provider
        name = self._config.stt_provider
        provider_cls = REGISTRY.get("stt", name)
        if provider_cls is None:
            logger.warning("STT provider '%s' not registered, falling back to faster_whisper", name)
            provider_cls = FasterWhisperProvider
        self._provider = provider_cls(self._config)
        return self._provider

    async def transcribe(self, audio: np.ndarray, sample_rate: int, **kwargs: Any) -> STTResult:
        provider = await self._get_provider()
        start = time.perf_counter()
        result = await provider.transcribe(audio, sample_rate, **kwargs)
        structured_log(logging.DEBUG, "stt.transcribe",
                        provider=self._config.stt_provider,
                        duration_ms=round((time.perf_counter() - start) * 1000, 1),
                        text_len=len(result.text))
        return result

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        sample_rate: int,
        **kwargs: Any,
    ) -> AsyncGenerator[STTResult, None]:
        provider = await self._get_provider()
        async for result in provider.transcribe_stream(audio_stream, sample_rate, **kwargs):
            yield result

    async def close(self) -> None:
        if self._provider:
            await self._provider.close()


# Register providers
REGISTRY.register("stt", "whisper", WhisperProvider)
REGISTRY.register("stt", "faster_whisper", FasterWhisperProvider)
REGISTRY.register("stt", "vosk", VoskProvider)
REGISTRY.register("stt", "onnx", ONNXProvider)

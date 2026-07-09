"""
Voice assistant — orchestrator for the full audio pipeline.

Wakes on keyword, captures audio, runs VAD+STT, dispatches to LLM,
and streams TTS response back.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Optional

import numpy as np

from voice.config import VoiceConfig
from voice.utils import structured_log, int16_to_float32

logger = logging.getLogger("aios.voice.assistant")


class AssistantState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


@dataclass
class ConversationTurn:
    transcript: str = ""
    response: str = ""
    audio_duration_s: float = 0.0
    stt_latency_s: float = 0.0
    llm_latency_s: float = 0.0
    tts_latency_s: float = 0.0
    language: str = "en"
    wake_word_detected: bool = False


class VoiceAssistant:
    """Orchestrates the full voice pipeline."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._state = AssistantState.IDLE
        self._conversation_history: list[ConversationTurn] = []
        self._on_transcript: Callable[[str], None] | None = None
        self._on_response: Callable[[str], None] | None = None
        self._on_state_change: Callable[[AssistantState], None] | None = None
        self._llm_callback: Callable[[str], AsyncGenerator[str, None]] | None = None

        # Lazy-loaded engines
        self._wake_word = None
        self._vad = None
        self._stt = None
        self._tts = None
        self._noise_reduction = None
        self._language_detection = None
        self._mic = None
        self._speaker = None

        self._running = False

    # --- Configuration ---

    def on_transcript(self, callback: Callable[[str], None]) -> None:
        self._on_transcript = callback

    def on_response(self, callback: Callable[[str], None]) -> None:
        self._on_response = callback

    def on_state_change(self, callback: Callable[[AssistantState], None]) -> None:
        self._on_state_change = callback

    def set_llm_handler(self, handler: Callable[[str], AsyncGenerator[str, None]]) -> None:
        self._llm_callback = handler

    # --- Engines lazy init ---

    async def _ensure_engines(self) -> None:
        if self._wake_word is not None:
            return

        from voice.wake_word import WakeWordEngine
        from voice.voice_activity_detection import VADEngine
        from voice.speech_to_text import STTEngine
        from voice.text_to_speech import TTSEngine
        from voice.noise_reduction import NoiseReductionEngine
        from voice.language_detection import LanguageDetectionEngine
        from voice.microphone import Microphone
        from voice.speaker import Speaker

        self._wake_word = WakeWordEngine(self._config)
        self._vad = VADEngine(self._config)
        self._stt = STTEngine(self._config)
        self._tts = TTSEngine(self._config)
        self._noise_reduction = NoiseReductionEngine(self._config)
        self._language_detection = LanguageDetectionEngine(self._config)
        self._mic = Microphone(self._config)
        self._speaker = Speaker(self._config)

    # --- State management ---

    def _set_state(self, state: AssistantState) -> None:
        if self._state != state:
            old = self._state
            self._state = state
            structured_log(logging.INFO, "assistant.state", old=old, new=state)
            if self._on_state_change:
                self._on_state_change(state)

    # --- Main pipeline ---

    async def run(self) -> None:
        """Main assistant loop (blocking async)."""
        await self._ensure_engines()
        self._running = True

        await self._mic.start()
        await self._speaker.start()

        self._set_state(AssistantState.IDLE)

        try:
            await self._run_loop()
        finally:
            await self._cleanup()

    async def _run_loop(self) -> None:
        wake_required = self._config.wake_word_enabled
        audio_buffer = b""

        async for chunk in self._mic.stream():
            if not self._running:
                break

            # Noise reduction
            if self._config.noise_reduction_enabled and len(chunk) > 0:
                chunk = await self._noise_reduction.reduce(chunk, self._config.sample_rate)

            # Wake word detection
            if wake_required:
                is_wake, wake_audio = await self._wake_word.process_chunk(chunk)
                if is_wake:
                    self._set_state(AssistantState.LISTENING)
                    # Keep audio after wake word for processing
                    if wake_audio is not None and len(wake_audio) > 0:
                        audio_buffer = wake_audio.tobytes()
                    continue

            if self._state != AssistantState.LISTENING:
                continue

            # VAD: detect speech segments
            audio_buffer += chunk.tobytes()
            audio_np = np.frombuffer(audio_buffer, dtype=np.int16)

            is_speech = await self._vad.is_speech(audio_np, self._config.sample_rate)

            if is_speech:
                continue  # Keep accumulating

            # Silence detected — process the utterance
            if len(audio_buffer) < self._config.chunk_size * 2:
                continue  # Too short

            await self._process_utterance(audio_np)
            audio_buffer = b""
            self._set_state(AssistantState.IDLE)

    async def _process_utterance(self, audio: np.ndarray) -> None:
        self._set_state(AssistantState.PROCESSING)
        turn = ConversationTurn()
        turn.audio_duration_s = len(audio) / self._config.sample_rate

        # Language detection
        lang_result = await self._language_detection.detect("")
        turn.language = lang_result.language if lang_result.language != "en" else self._config.language

        # STT
        t0 = time.perf_counter()
        transcript = await self._stt.transcribe(audio)
        turn.stt_latency_s = time.perf_counter() - t0
        turn.transcript = transcript

        structured_log(logging.INFO, "assistant.transcript",
                       text=transcript, lang=turn.language, duration=turn.audio_duration_s)

        if self._on_transcript:
            self._on_transcript(transcript)

        # LLM
        if self._llm_callback:
            t0 = time.perf_counter()
            full_response = ""
            async for part in self._llm_callback(transcript):
                full_response += part
            turn.llm_latency_s = time.perf_counter() - t0
            turn.response = full_response

            if self._on_response:
                self._on_response(full_response)

            # TTS
            self._set_state(AssistantState.SPEAKING)
            t0 = time.perf_counter()
            tts_audio = await self._tts.synthesize(full_response)
            turn.tts_latency_s = time.perf_counter() - t0

            if tts_audio is not None and len(tts_audio) > 0:
                await self._speaker.play_and_wait(tts_audio)

        self._conversation_history.append(turn)
        structured_log(logging.INFO, "assistant.turn",
                       turn_idx=len(self._conversation_history),
                       transcript=transcript,
                       response=turn.response[:100] if turn.response else "")

    async def process_single(self, audio: np.ndarray) -> str:
        """Process a single audio chunk and return the transcript.

        Used for non-streaming / push-to-talk scenarios.
        """
        await self._ensure_engines()
        if self._config.noise_reduction_enabled:
            audio = await self._noise_reduction.reduce(audio, self._config.sample_rate)

        transcript = await self._stt.transcribe(audio)
        return transcript

    async def speak(self, text: str) -> None:
        """Synthesize and play text-to-speech."""
        await self._ensure_engines()
        self._set_state(AssistantState.SPEAKING)
        tts_audio = await self._tts.synthesize(text)
        if tts_audio is not None and len(tts_audio) > 0:
            await self._speaker.play_and_wait(tts_audio)
        self._set_state(AssistantState.IDLE)

    def stop(self) -> None:
        self._running = False
        if self._mic:
            self._mic.stop()
        if self._speaker:
            self._speaker.stop()

    async def close(self) -> None:
        self._running = False
        for component in [
            self._mic, self._speaker, self._stt, self._tts,
            self._wake_word, self._vad, self._noise_reduction,
        ]:
            if component is not None and hasattr(component, "close"):
                await component.close()

    @property
    def state(self) -> AssistantState:
        return self._state

    @property
    def conversation_history(self) -> list[ConversationTurn]:
        return list(self._conversation_history)

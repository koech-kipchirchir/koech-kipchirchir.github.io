"""
Streaming audio I/O: circular buffers, chunk concatenation, and
stream utilities.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, AsyncIterator, Optional

import numpy as np

from voice.config import VoiceConfig
from voice.utils import structured_log

logger = logging.getLogger("aios.voice.audio_stream")


class AudioBuffer:
    """Thread-safe circular buffer for audio data."""

    def __init__(self, max_seconds: float = 10.0, sample_rate: int = 16000) -> None:
        self._max_samples = int(max_seconds * sample_rate)
        self._buffer: list[np.ndarray] = []
        self._total_samples = 0

    def append(self, chunk: np.ndarray) -> None:
        self._buffer.append(chunk)
        self._total_samples += len(chunk)
        while self._total_samples > self._max_samples and self._buffer:
            removed = self._buffer.pop(0)
            self._total_samples -= len(removed)

    def get_all(self) -> np.ndarray:
        if not self._buffer:
            return np.array([], dtype=np.int16)
        return np.concatenate(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
        self._total_samples = 0

    @property
    def duration_s(self) -> float:
        return self._total_samples / 16000.0

    def __len__(self) -> int:
        return self._total_samples


class StreamConcatenator:
    """Combines multiple async audio streams into one."""

    def __init__(self) -> None:
        self._streams: list[AsyncGenerator[np.ndarray, None]] = []

    def add_stream(self, stream: AsyncGenerator[np.ndarray, None]) -> None:
        self._streams.append(stream)

    async def merged_stream(self) -> AsyncGenerator[np.ndarray, None]:
        """Merge multiple streams with simple interleaving."""
        if not self._streams:
            return
        iterators = [s.__aiter__() for s in self._streams]
        pending = list(range(len(iterators)))

        while pending:
            for idx in list(pending):
                try:
                    chunk = await iterators[idx].__anext__()
                    yield chunk
                except StopAsyncIteration:
                    pending.remove(idx)


class AudioStream:
    """High-level audio stream abstraction for voice processing.

    Chains microphone → noise reduction → VAD → wake word → STT
    into a single processing pipeline.
    """

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._buffer = AudioBuffer(
            max_seconds=30.0,
            sample_rate=config.sample_rate,
        )

    async def feed(self, audio: np.ndarray) -> None:
        self._buffer.append(audio)

    def get_buffered(self) -> np.ndarray:
        return self._buffer.get_all()

    def clear(self) -> None:
        self._buffer.clear()

    @property
    def buffered_seconds(self) -> float:
        return self._buffer.duration_s


async def chunk_generator(
    audio: np.ndarray,
    chunk_size: int,
) -> AsyncGenerator[np.ndarray, None]:
    """Split a numpy array into fixed-size chunks and yield asynchronously."""
    for i in range(0, len(audio), chunk_size):
        yield audio[i:i + chunk_size]


async def stream_from_queue(
    queue: asyncio.Queue[np.ndarray],
    timeout: float = 0.1,
) -> AsyncGenerator[np.ndarray, None]:
    """Convert an async queue into an async generator."""
    while True:
        try:
            chunk = await asyncio.wait_for(queue.get(), timeout=timeout)
            yield chunk
        except asyncio.TimeoutError:
            continue
        except Exception:
            break


class StreamSilenceDetector:
    """Detects silence in a stream and yields speech segments."""

    def __init__(
        self,
        silence_threshold: float = 0.02,
        min_silence_ms: int = 500,
        min_speech_ms: int = 200,
    ) -> None:
        self._threshold = silence_threshold
        self._min_silence = min_silence_ms / 1000.0
        self._min_speech = min_speech_ms / 1000.0
        self._speech_buffer: list[np.ndarray] = []
        self._silence_seconds = 0.0
        self._speech_seconds = 0.0

    async def process_stream(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        sample_rate: int,
    ) -> AsyncGenerator[np.ndarray, None]:
        chunk_duration = 0.0
        for chunk in audio_stream:
            chunk_duration = len(chunk) / sample_rate
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            is_speech = rms > self._threshold

            if is_speech:
                self._speech_buffer.append(chunk)
                self._speech_seconds += chunk_duration
                self._silence_seconds = 0.0
            else:
                self._silence_seconds += chunk_duration
                if self._silence_seconds >= self._min_silence and self._speech_buffer:
                    if self._speech_seconds >= self._min_speech:
                        yield np.concatenate(self._speech_buffer)
                    self._speech_buffer = []
                    self._speech_seconds = 0.0

        if self._speech_buffer and self._speech_seconds >= self._min_speech:
            yield np.concatenate(self._speech_buffer)

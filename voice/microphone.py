"""
Microphone capture with async audio streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional

import numpy as np

from voice.config import VoiceConfig
from voice.utils import structured_log

logger = logging.getLogger("aios.voice.microphone")


class MicError(Exception):
    pass


class Microphone:
    """Async microphone capture using PyAudio or sounddevice.

    Captures audio from a physical mic and yields it as an
    async generator of numpy int16 arrays.
    """

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._running = False
        self._thread: threading.Thread | None = None
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

    @property
    def sample_rate(self) -> int:
        return self._config.sample_rate

    @property
    def channels(self) -> int:
        return self._config.channels

    @property
    def chunk_size(self) -> int:
        return self._config.chunk_size

    async def start(self) -> None:
        """Start capturing from the microphone."""
        if self._running:
            return
        self._running = True
        self._queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        self._thread = threading.Thread(
            target=self._capture_thread,
            args=(loop,),
            daemon=True,
        )
        self._thread.start()
        structured_log(logging.INFO, "microphone.started",
                       device=self._config.mic_device_index,
                       sr=self.sample_rate)

    def _capture_thread(self, loop: asyncio.AbstractEventLoop) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            self._capture_pyaudio(loop)
            return

        device = self._config.mic_device_index if self._config.mic_device_index >= 0 else None
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=device,
                blocksize=self.chunk_size,
                dtype=np.int16,
            ) as stream:
                while self._running:
                    chunk, _ = stream.read(self.chunk_size)
                    if self._config.mic_gain != 1.0:
                        chunk = (chunk * self._config.mic_gain).astype(np.int16)
                    asyncio.run_coroutine_threadsafe(
                        self._queue.put(chunk), loop,
                    )
        except Exception as e:
            logger.error("Microphone capture error: %s", e)
            self._running = False

    def _capture_pyaudio(self, loop: asyncio.AbstractEventLoop) -> None:
        import pyaudio
        p = pyaudio.PyAudio()
        device = self._config.mic_device_index if self._config.mic_device_index >= 0 else None
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device,
                frames_per_buffer=self.chunk_size,
            )
            while self._running:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.int16)
                if self._config.mic_gain != 1.0:
                    chunk = (chunk * self._config.mic_gain).astype(np.int16)
                asyncio.run_coroutine_threadsafe(
                    self._queue.put(chunk), loop,
                )
            stream.close()
        except Exception as e:
            logger.error("PyAudio capture error: %s", e)
        finally:
            p.terminate()
            self._running = False

    async def read_chunk(self) -> np.ndarray:
        """Read a single audio chunk (blocking async)."""
        timeout = self._config.mic_timeout_s
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            raise MicError(f"No audio input within {timeout}s")

    async def stream(self) -> AsyncGenerator[np.ndarray, None]:
        """Async generator yielding audio chunks until stopped."""
        timeout = self._config.mic_timeout_s
        while self._running:
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                yield chunk
            except asyncio.TimeoutError:
                logger.debug("Microphone stream timeout (no audio)")
                if not self._running:
                    break

    def stop(self) -> None:
        self._running = False
        structured_log(logging.INFO, "microphone.stopped")

    async def close(self) -> None:
        self.stop()
        if self._thread:
            self._thread.join(timeout=3)

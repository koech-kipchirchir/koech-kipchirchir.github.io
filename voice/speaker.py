"""
Audio output via speaker with async playback support.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from voice.config import VoiceConfig
from voice.utils import structured_log

logger = logging.getLogger("aios.voice.speaker")


class SpeakerError(Exception):
    pass


class Speaker:
    """Async audio playback via sounddevice or PyAudio.

    Supports queue-based playback of numpy audio arrays with
    configurable volume and buffering.
    """

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._running = False
        self._thread: threading.Thread | None = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._playing = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._playback_thread, daemon=True)
        self._thread.start()
        structured_log(logging.INFO, "speaker.started",
                       device=self._config.speaker_device_index)

    def _playback_thread(self) -> None:
        try:
            import sounddevice as sd
            self._play_sounddevice(sd)
        except ImportError:
            self._play_pyaudio()

    def _play_sounddevice(self, sd: Any) -> None:
        device = self._config.speaker_device_index if self._config.speaker_device_index >= 0 else None
        buffer_ms = int(self._config.speaker_buffer_seconds * 1000)
        try:
            with sd.OutputStream(
                samplerate=self._config.sample_rate,
                channels=self._config.channels,
                device=device,
                dtype=np.int16,
            ) as stream:
                while self._running:
                    try:
                        audio = self._queue.get(timeout=0.5)
                        audio = self._apply_volume(audio)
                        stream.write(audio)
                        self._playing = True
                    except queue.Empty:
                        self._playing = False
                        continue
        except Exception as e:
            logger.error("SoundDevice playback error: %s", e)

    def _play_pyaudio(self) -> None:
        import pyaudio
        p = pyaudio.PyAudio()
        device = self._config.speaker_device_index if self._config.speaker_device_index >= 0 else None
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=self._config.channels,
                rate=self._config.sample_rate,
                output=True,
                output_device_index=device,
            )
            while self._running:
                try:
                    audio = self._queue.get(timeout=0.5)
                    audio = self._apply_volume(audio)
                    stream.write(audio.tobytes())
                    self._playing = True
                except queue.Empty:
                    self._playing = False
                    continue
            stream.close()
        except Exception as e:
            logger.error("PyAudio playback error: %s", e)
        finally:
            p.terminate()

    def _apply_volume(self, audio: np.ndarray) -> np.ndarray:
        vol = self._config.speaker_volume
        if vol != 1.0:
            return (audio * vol).astype(np.int16)
        return audio

    def play(self, audio: np.ndarray) -> None:
        """Queue audio for playback (non-blocking)."""
        if not self._running:
            raise SpeakerError("Speaker not started")
        self._queue.put(audio)

    async def play_async(self, audio: np.ndarray) -> None:
        """Queue audio in a thread-safe manner."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.play, audio)

    async def play_and_wait(self, audio: np.ndarray) -> None:
        """Play audio and block until it finishes."""
        self.play(audio)
        duration = len(audio) / self._config.sample_rate
        await asyncio.sleep(duration + self._config.speaker_buffer_seconds)

    def is_playing(self) -> bool:
        return self._playing

    def stop(self) -> None:
        self._running = False
        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        structured_log(logging.INFO, "speaker.stopped")

    async def close(self) -> None:
        self.stop()
        if self._thread:
            self._thread.join(timeout=3)

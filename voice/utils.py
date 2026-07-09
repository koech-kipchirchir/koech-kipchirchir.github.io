"""
Shared utilities for the voice module: audio helpers, logging, and
plugin registry.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import struct
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger("aios.voice")


def structured_log(level: int, event: str, **kwargs: Any) -> None:
    record: dict[str, Any] = {"event": event}
    record.update(kwargs)
    logger.log(level, "%s", json.dumps(record, default=str))


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16 array to float32 in [-1, 1]."""
    return audio.astype(np.float32) / 32768.0


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] to int16 array."""
    return np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to a target sample rate (simple linear)."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_len = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(audio.dtype)


def audio_to_bytes(audio: np.ndarray, sample_width: int = 2) -> bytes:
    """Convert numpy audio array to raw bytes."""
    if sample_width == 2:
        return audio.astype(np.int16).tobytes()
    return audio.astype(np.float32).tobytes()


def bytes_to_audio(data: bytes, sample_width: int = 2) -> np.ndarray:
    """Convert raw bytes to numpy audio array."""
    if sample_width == 2:
        return np.frombuffer(data, dtype=np.int16)
    return np.frombuffer(data, dtype=np.float32)


def save_wav(filepath: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Save a numpy audio array as a WAV file (stdlib only)."""
    import wave
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_int16 = float32_to_int16(audio) if audio.dtype == np.float32 else audio
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def load_wav(filepath: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV file. Returns (audio_int16, sample_rate)."""
    import wave
    with wave.open(str(filepath), "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
    return audio, sr


def rms_energy(audio: np.ndarray) -> float:
    """Compute RMS energy of an audio signal."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


def silence_detected(audio: np.ndarray, threshold: float = 0.02) -> bool:
    """Check if audio is below a silence threshold (normalized)."""
    return rms_energy(audio) < threshold


# ---------------------------------------------------------------------------
# Plugin registry
# ---------------------------------------------------------------------------

class PluginRegistry:
    """Simple registry for voice provider plugins."""

    def __init__(self) -> None:
        self._providers: dict[str, dict[str, type]] = {}

    def register(self, category: str, name: str, cls: type) -> None:
        if category not in self._providers:
            self._providers[category] = {}
        self._providers[category][name] = cls
        structured_log(logging.DEBUG, "plugin.registered",
                       category=category, name=name, cls=cls.__name__)

    def get(self, category: str, name: str) -> type | None:
        return self._providers.get(category, {}).get(name)

    def list(self, category: str) -> list[str]:
        return list(self._providers.get(category, {}).keys())

    def create(self, category: str, name: str, **kwargs: Any) -> Any:
        cls = self.get(category, name)
        if cls is None:
            raise ValueError(f"Unknown provider '{name}' for category '{category}'. "
                             f"Available: {self.list(category)}")
        return cls(**kwargs)


REGISTRY = PluginRegistry()

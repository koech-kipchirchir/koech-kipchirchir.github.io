"""
Voice module configuration with provider selection and tuning parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VoiceConfig:
    """Voice subsystem configuration.

    Loaded from AIOS_VOICE_* environment variables.
    """

    # --- Provider Selection ---
    stt_provider: str = "faster_whisper"   # whisper, faster_whisper, vosk, onnx
    tts_provider: str = "edge"             # piper, coqui, xtts, edge
    vad_provider: str = "silero"           # silero, webrtcvad
    wake_word_provider: str = "porcupine"  # porcupine, snowboy, custom

    # --- STT ---
    whisper_model: str = "base"            # tiny, base, small, medium, large-v3
    whisper_device: str = "auto"           # cpu, cuda, auto
    whisper_compute_type: str = "float16"
    vosk_model_path: str = ""
    onnx_model_path: str = ""

    # --- TTS ---
    piper_model_path: str = ""
    piper_voice: str = "en_US-lessac-medium"
    piper_speed: float = 1.0
    coqui_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    coqui_voice_dir: str = ""
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    xtts_speaker: str = ""
    edge_voice: str = "en-US-JennyNeural"
    edge_rate: str = "+0%"
    edge_volume: str = "+0%"

    # --- Audio ---
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2
    chunk_size: int = 1024
    frame_duration_ms: int = 30
    bytes_per_sample: int = 2

    # --- Microphone ---
    mic_device_index: int = -1             # -1 = default
    mic_gain: float = 1.0
    mic_timeout_s: float = 30.0
    mic_agc_enabled: bool = True

    # --- Speaker ---
    speaker_device_index: int = -1
    speaker_volume: float = 1.0
    speaker_buffer_seconds: float = 0.2

    # --- Wake Word ---
    wake_word: str = "hey aios"
    wake_sensitivity: float = 0.5
    wake_timeout_s: float = 0.0
    porcupine_key: str = ""
    porcupine_model_path: str = ""

    # --- VAD ---
    vad_threshold: float = 0.5
    vad_min_speech_duration_ms: int = 200
    vad_min_silence_duration_ms: int = 500
    vad_buffer_ms: int = 300

    # --- Noise Reduction ---
    noise_reduction_enabled: bool = True
    noise_reduction_strength: float = 0.8
    noise_profile_path: str = ""

    # --- Language ---
    language: str = "en"
    language_detection_enabled: bool = False
    supported_languages: list[str] = field(default_factory=lambda: [
        "en", "zh", "es", "fr", "de", "ja", "ko", "ru", "ar", "pt",
    ])

    # --- Streaming ---
    streaming_stt_enabled: bool = True
    streaming_tts_enabled: bool = True
    streaming_buffer_seconds: float = 0.5

    # --- Conversation ---
    conversation_timeout_s: float = 300.0
    conversation_max_turns: int = 50
    conversation_auto_end_silence_s: float = 2.0

    # --- Paths ---
    model_dir: str = "models/voice"
    temp_dir: str = "temp/audio"
    log_audio: bool = False
    log_audio_dir: str = "logs/audio"

    @classmethod
    def from_env(cls) -> VoiceConfig:
        prefix = "AIOS_VOICE_"
        kwargs: dict[str, Any] = {}
        for fname in cls.__dataclass_fields__:
            env_val = os.environ.get(f"{prefix}{fname.upper()}")
            if env_val is None:
                continue
            ft = cls.__dataclass_fields__[fname].type
            if ft is bool or ft == "bool":
                kwargs[fname] = env_val.lower() in ("1", "true", "yes", "on")
            elif ft is int or ft == "int":
                kwargs[fname] = int(env_val)
            elif ft is float or ft == "float":
                kwargs[fname] = float(env_val)
            elif "list" in str(ft):
                kwargs[fname] = [x.strip() for x in env_val.split(",") if x.strip()]
            else:
                kwargs[fname] = env_val
        return cls(**kwargs)

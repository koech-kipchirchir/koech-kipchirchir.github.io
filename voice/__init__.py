"""
AIOS Voice Module — real-time speech processing pipeline.

Provides wake-word detection, VAD, STT, TTS, noise reduction,
language detection, and a high-level VoiceAssistant orchestrator.
"""

from __future__ import annotations

from voice.audio_stream import AudioBuffer, AudioStream, StreamConcatenator, StreamSilenceDetector, chunk_generator, stream_from_queue
from voice.config import VoiceConfig
from voice.language_detection import FastTextProvider, LangDetectProvider, LanguageDetectionEngine, LanguageResult
from voice.microphone import MicError, Microphone
from voice.noise_reduction import NoiseReduceProvider, NoiseReductionEngine, NoiseReductionProvider, SpectralGateProvider
from voice.speaker import Speaker, SpeakerError
from voice.speech_to_text import (
    FasterWhisperProvider,
    ONNXProvider,
    STTEngine,
    STTProvider,
    VoskProvider,
    WhisperProvider,
)
from voice.text_to_speech import (
    CoquiProvider,
    EdgeTTSProvider,
    PiperProvider,
    TTSEngine,
    TTSProvider,
    XTTSProvider,
)
from voice.utils import REGISTRY, PluginRegistry, audio_to_bytes, bytes_to_audio, float32_to_int16, int16_to_float32, load_wav, resample, rms_energy, save_wav, silence_detected, structured_log
from voice.voice_activity_detection import SileroVADProvider, VADEngine, VADProvider, WebRTCVADProvider
from voice.voice_assistant import AssistantState, ConversationTurn, VoiceAssistant
from voice.wake_word import CustomKeywordProvider, PorcupineProvider, SnowboyProvider, WakeWordEngine, WakeWordProvider

__all__ = [
    "AssistantState",
    "AudioBuffer",
    "AudioStream",
    "CoquiProvider",
    "ConversationTurn",
    "CustomKeywordProvider",
    "EdgeTTSProvider",
    "FastTextProvider",
    "FasterWhisperProvider",
    "LanguageDetectionEngine",
    "LanguageResult",
    "LangDetectProvider",
    "MicError",
    "Microphone",
    "NoiseReduceProvider",
    "NoiseReductionEngine",
    "NoiseReductionProvider",
    "ONNXProvider",
    "PiperProvider",
    "PluginRegistry",
    "PorcupineProvider",
    "REGISTRY",
    "STTEngine",
    "STTProvider",
    "SileroVADProvider",
    "SnowboyProvider",
    "Speaker",
    "SpeakerError",
    "SpectralGateProvider",
    "StreamConcatenator",
    "StreamSilenceDetector",
    "TTSEngine",
    "TTSProvider",
    "VADEngine",
    "VADProvider",
    "VoiceAssistant",
    "VoiceConfig",
    "VoskProvider",
    "WakeWordEngine",
    "WakeWordProvider",
    "WebRTCVADProvider",
    "WhisperProvider",
    "XTTSProvider",
    "audio_to_bytes",
    "bytes_to_audio",
    "chunk_generator",
    "float32_to_int16",
    "int16_to_float32",
    "load_wav",
    "resample",
    "rms_energy",
    "save_wav",
    "silence_detected",
    "stream_from_queue",
    "structured_log",
]

"""
Language detection for multi-lingual voice support.

Uses ``fasttext`` or ``langdetect`` to identify the spoken language
from audio transcript text (not raw audio).
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from voice.config import VoiceConfig
from voice.utils import REGISTRY, structured_log

logger = logging.getLogger("aios.voice.language_detection")


@dataclass
class LanguageResult:
    language: str = "en"
    confidence: float = 0.0
    all_languages: list[tuple[str, float]] = field(default_factory=list)


class LanguageDetector(ABC):
    @abstractmethod
    async def detect(self, text: str) -> LanguageResult:
        pass

    @abstractmethod
    async def detect_batch(self, texts: list[str]) -> list[LanguageResult]:
        pass


# ---------------------------------------------------------------------------
# langdetect
# ---------------------------------------------------------------------------

class LangDetectProvider(LanguageDetector):
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config

    async def detect(self, text: str) -> LanguageResult:
        import langdetect
        loop = asyncio.get_event_loop()

        def _detect() -> LanguageResult:
            try:
                langs = langdetect.detect_langs(text)
                if langs:
                    top = langs[0]
                    return LanguageResult(
                        language=top.lang[:2],
                        confidence=top.prob,
                        all_languages=[(l.lang[:2], l.prob) for l in langs],
                    )
            except langdetect.lang_detect_exception.LangDetectException:
                pass
            return LanguageResult(language=self._config.language)

        return await loop.run_in_executor(None, _detect)

    async def detect_batch(self, texts: list[str]) -> list[LanguageResult]:
        results = []
        for t in texts:
            results.append(await self.detect(t))
        return results


# ---------------------------------------------------------------------------
# fastText
# ---------------------------------------------------------------------------

class FastTextProvider(LanguageDetector):
    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._model = None

    async def _load_model(self) -> None:
        if self._model is not None:
            return
        import fasttext
        import fasttext.util
        loop = asyncio.get_event_loop()

        def _load():
            fasttext.util.download_model("lid.176", if_exists="ignore")
            return fasttext.load_model("lid.176.bin")

        self._model = await loop.run_in_executor(None, _load)
        structured_log(logging.INFO, "language_detection.fasttext.loaded")

    async def detect(self, text: str) -> LanguageResult:
        await self._load_model()
        text_clean = text.replace("\n", " ").strip()
        if not text_clean:
            return LanguageResult(language=self._config.language)
        loop = asyncio.get_event_loop()

        def _predict() -> tuple[list[str], list[float]]:
            labels, scores = self._model.predict(text_clean, k=5)
            langs = [l.replace("__label__", "") for l in labels]
            return langs, scores

        langs, scores = await loop.run_in_executor(None, _predict)
        all_langs = [(langs[i], scores[i]) for i in range(len(langs))]
        return LanguageResult(
            language=langs[0][:2] if langs else self._config.language,
            confidence=scores[0] if scores else 0.0,
            all_languages=all_langs,
        )

    async def detect_batch(self, texts: list[str]) -> list[LanguageResult]:
        results = []
        for t in texts:
            results.append(await self.detect(t))
        return results


# ---------------------------------------------------------------------------
# Language Detection Engine
# ---------------------------------------------------------------------------

class LanguageDetectionEngine:
    """Language detection engine."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._detector: LanguageDetector | None = None

    async def _get_detector(self) -> LanguageDetector:
        if self._detector is not None:
            return self._detector
        # Prefer fasttext if available, fall back to langdetect
        try:
            import fasttext
            self._detector = FastTextProvider(self._config)
        except ImportError:
            self._detector = LangDetectProvider(self._config)

        await self._detector.detect("test")
        return self._detector

    async def detect(self, text: str) -> LanguageResult:
        if not self._config.language_detection_enabled:
            return LanguageResult(language=self._config.language, confidence=1.0)
        detector = await self._get_detector()
        return await detector.detect(text)

    def is_supported(self, lang: str) -> bool:
        return lang[:2] in self._config.supported_languages


REGISTRY.register("language_detection", "langdetect", LangDetectProvider)
REGISTRY.register("language_detection", "fasttext", FastTextProvider)

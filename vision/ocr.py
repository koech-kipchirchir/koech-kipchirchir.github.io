"""
OCR providers and engine supporting EasyOCR, Tesseract, and PaddleOCR.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from vision.config import VisionConfig

logger = logging.getLogger("aios.vision.ocr")


@dataclass
class OCRTextBlock:
    text: str
    confidence: float = 0.0
    bbox: Optional[list[tuple[int, int]]] = None


@dataclass
class OCRResult:
    text: str = ""
    blocks: list[OCRTextBlock] = field(default_factory=list)
    language: str = "en"
    confidence: float = 0.0
    latency_s: float = 0.0
    engine: str = ""

    @property
    def full_text(self) -> str:
        return self.text or "\n".join(b.text for b in self.blocks)


class OCRProvider(ABC):
    @abstractmethod
    async def recognize(self, image: np.ndarray, languages: list[str]) -> OCRResult:
        pass

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# EasyOCR
# ---------------------------------------------------------------------------

class EasyOCRProvider(OCRProvider):
    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._reader: Any = None

    async def _ensure_reader(self, languages: list[str]) -> None:
        if self._reader is not None:
            return
        import easyocr
        loop = asyncio.get_event_loop()

        def _load() -> Any:
            return easyocr.Reader(
                languages or ["en"],
                gpu=False,
            )

        self._reader = await loop.run_in_executor(None, _load)
        logger.info("EasyOCR reader loaded for %s", languages)

    async def recognize(self, image: np.ndarray, languages: list[str]) -> OCRResult:
        t0 = time.perf_counter()
        await self._ensure_reader(languages or self._config.ocr_languages)
        loop = asyncio.get_event_loop()

        def _ocr() -> list[Any]:
            return self._reader.readtext(
                image,
                paragraph=True,
                width_ths=0.7,
                height_ths=0.7,
            )

        raw = await loop.run_in_executor(None, _ocr)
        blocks = []
        texts = []
        confs = []
        for bbox, text, conf in raw:
            blocks.append(OCRTextBlock(text=text, confidence=conf, bbox=[(int(x), int(y)) for x, y in bbox]))
            texts.append(text)
            confs.append(conf)

        latency = time.perf_counter() - t0
        return OCRResult(
            text="\n".join(texts),
            blocks=blocks,
            confidence=sum(confs) / len(confs) if confs else 0.0,
            latency_s=latency,
            engine="easyocr",
        )


# ---------------------------------------------------------------------------
# Tesseract
# ---------------------------------------------------------------------------

class TesseractProvider(OCRProvider):
    def __init__(self, config: VisionConfig) -> None:
        self._config = config

    async def recognize(self, image: np.ndarray, languages: list[str]) -> OCRResult:
        import pytesseract
        from PIL import Image
        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()

        if self._config.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self._config.tesseract_path

        pil_image = Image.fromarray(image)
        lang_str = "+".join(languages or self._config.ocr_languages)

        def _ocr() -> tuple[str, Any]:
            data = pytesseract.image_to_data(pil_image, lang=lang_str, output_type=pytesseract.Output.DICT)
            text = pytesseract.image_to_string(pil_image, lang=lang_str)
            return text, data

        text, data = await loop.run_in_executor(None, _ocr)
        blocks = []
        confs = []
        for i in range(len(data.get("text", []))):
            txt = data["text"][i].strip()
            conf = int(data.get("conf", [0])[i]) / 100.0
            if txt and conf > 0:
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                bbox = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                blocks.append(OCRTextBlock(text=txt, confidence=conf, bbox=bbox))
                confs.append(conf)

        latency = time.perf_counter() - t0
        return OCRResult(
            text=text.strip(),
            blocks=blocks,
            confidence=sum(confs) / len(confs) if confs else 0.0,
            latency_s=latency,
            engine="tesseract",
        )


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------

class PaddleOCRProvider(OCRProvider):
    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._ocr: Any = None

    async def _ensure_ocr(self) -> None:
        if self._ocr is not None:
            return
        from paddleocr import PaddleOCR
        loop = asyncio.get_event_loop()

        def _load() -> Any:
            return PaddleOCR(
                use_angle_cls=True,
                lang=self._config.paddleocr_lang,
                use_gpu=self._config.paddleocr_use_gpu,
                show_log=False,
            )

        self._ocr = await loop.run_in_executor(None, _load)
        logger.info("PaddleOCR initialized (lang=%s, gpu=%s)", self._config.paddleocr_lang, self._config.paddleocr_use_gpu)

    async def recognize(self, image: np.ndarray, languages: list[str]) -> OCRResult:
        t0 = time.perf_counter()
        await self._ensure_ocr()
        loop = asyncio.get_event_loop()

        def _ocr() -> list[Any]:
            return self._ocr.ocr(image, cls=True)

        raw_list = await loop.run_in_executor(None, _ocr)
        blocks = []
        texts = []
        confs = []
        for page in raw_list:
            if page is None:
                continue
            for item in page:
                bbox_coords, (text, conf) = item
                blocks.append(OCRTextBlock(
                    text=text,
                    confidence=conf,
                    bbox=[(int(x), int(y)) for x, y in bbox_coords],
                ))
                texts.append(text)
                confs.append(conf)

        latency = time.perf_counter() - t0
        return OCRResult(
            text="\n".join(texts),
            blocks=blocks,
            confidence=sum(confs) / len(confs) if confs else 0.0,
            latency_s=latency,
            engine="paddleocr",
        )

    async def close(self) -> None:
        self._ocr = None


# ---------------------------------------------------------------------------
# OCR Engine
# ---------------------------------------------------------------------------

class OCREngine:
    """OCR engine facade with automatic provider selection."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._provider: Optional[OCRProvider] = None
        self._engine_map: dict[str, type[OCRProvider]] = {
            "easyocr": EasyOCRProvider,
            "tesseract": TesseractProvider,
            "paddleocr": PaddleOCRProvider,
        }

    async def _get_provider(self) -> OCRProvider:
        if self._provider is not None:
            return self._provider
        engine = self._config.ocr_engine
        cls = self._engine_map.get(engine)
        if cls is None:
            available = list(self._engine_map.keys())
            logger.warning("OCR engine '%s' not found, using easyocr. Available: %s", engine, available)
            cls = EasyOCRProvider
        self._provider = cls(self._config)
        return self._provider

    async def recognize(self, image: np.ndarray, languages: Optional[list[str]] = None) -> OCRResult:
        provider = await self._get_provider()
        return await provider.recognize(image, languages or self._config.ocr_languages)

    async def close(self) -> None:
        if self._provider:
            await self._provider.close()

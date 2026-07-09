"""
Barcode and QR code detection using pyzbar and OpenCV.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from vision.config import VisionConfig

logger = logging.getLogger("aios.vision.barcode")


class BarcodeType(str, Enum):
    QR = "QR"
    CODE128 = "CODE128"
    CODE39 = "CODE39"
    EAN13 = "EAN13"
    EAN8 = "EAN8"
    UPC_A = "UPC-A"
    UPC_E = "UPC-E"
    ITF = "ITF"
    DATAMATRIX = "DATAMATRIX"
    PDF417 = "PDF417"
    AZTEC = "AZTEC"
    UNKNOWN = "UNKNOWN"


@dataclass
class BarcodeResult:
    data: str = ""
    barcode_type: BarcodeType = BarcodeType.UNKNOWN
    confidence: float = 1.0
    bbox: Optional[list[tuple[int, int]]] = None
    poly_points: Optional[list[tuple[int, int]]] = None
    latency_s: float = 0.0
    engine: str = ""


class BarcodeProvider(ABC):
    @abstractmethod
    async def detect(self, image: np.ndarray) -> list[BarcodeResult]:
        pass


# ---------------------------------------------------------------------------
# Pyzbar
# ---------------------------------------------------------------------------

class PyzbarProvider(BarcodeProvider):
    async def detect(self, image: np.ndarray) -> list[BarcodeResult]:
        import cv2
        from PIL import Image
        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        pil = Image.fromarray(gray)

        def _decode() -> list[Any]:
            from pyzbar import pyzbar
            return pyzbar.decode(pil)

        decoded = await loop.run_in_executor(None, _decode)
        results = []
        for obj in decoded:
            btype = obj.type if isinstance(obj.type, str) else "UNKNOWN"
            try:
                bt = BarcodeType(btype)
            except ValueError:
                bt = BarcodeType.UNKNOWN
            poly = [(int(p.x), int(p.y)) for p in obj.polygon] if obj.polygon else None
            rect = obj.rect
            bbox = [(rect.left, rect.top), (rect.left + rect.width, rect.top),
                    (rect.left + rect.width, rect.top + rect.height), (rect.left, rect.top + rect.height)] if rect else None
            results.append(BarcodeResult(
                data=obj.data.decode("utf-8", errors="replace"),
                barcode_type=bt,
                confidence=1.0,
                bbox=bbox,
                poly_points=poly,
                engine="pyzbar",
            ))

        latency = time.perf_counter() - t0
        for r in results:
            r.latency_s = latency
        return results


# ---------------------------------------------------------------------------
# OpenCV QR
# ---------------------------------------------------------------------------

class OpenCVBarcodeProvider(BarcodeProvider):
    def __init__(self) -> None:
        self._qr_detector: Any = None

    async def detect(self, image: np.ndarray) -> list[BarcodeResult]:
        import cv2
        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()

        if self._qr_detector is None:
            self._qr_detector = cv2.QRCodeDetector()

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()

        def _detect_qr() -> tuple[bool, str, Any]:
            val, points, _ = self._qr_detector.detectAndDecode(gray)
            return bool(val), val, points

        has_qr, qr_data, points = await loop.run_in_executor(None, _detect_qr)
        results = []
        if has_qr and qr_data:
            bbox = [(int(points[0][0]), int(points[0][1])),
                    (int(points[1][0]), int(points[1][1])),
                    (int(points[2][0]), int(points[2][1])),
                    (int(points[3][0]), int(points[3][1]))] if points is not None else None
            results.append(BarcodeResult(
                data=qr_data,
                barcode_type=BarcodeType.QR,
                confidence=1.0,
                bbox=bbox,
                poly_points=bbox,
                latency_s=time.perf_counter() - t0,
                engine="opencv",
            ))
        return results


# ---------------------------------------------------------------------------
# Barcode Detector
# ---------------------------------------------------------------------------

class BarcodeDetector:
    """Unified barcode/QR code detector."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._providers: list[BarcodeProvider] = [PyzbarProvider(), OpenCVBarcodeProvider()]

    async def detect(self, image: np.ndarray) -> list[BarcodeResult]:
        all_results: list[BarcodeResult] = []
        seen_data: set[str] = set()
        for provider in self._providers:
            try:
                results = await provider.detect(image)
                for r in results:
                    if r.data and r.data not in seen_data:
                        all_results.append(r)
                        seen_data.add(r.data)
            except Exception as e:
                logger.debug("Barcode provider error: %s", e)
        return all_results

    async def detect_qr(self, image: np.ndarray) -> Optional[BarcodeResult]:
        results = await self.detect(image)
        for r in results:
            if r.barcode_type == BarcodeType.QR:
                return r
        return None

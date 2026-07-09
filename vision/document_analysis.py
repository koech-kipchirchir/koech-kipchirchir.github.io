"""
Document scanning, perspective correction, and PDF OCR pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from vision.config import VisionConfig
from vision.ocr import OCREngine, OCRResult

logger = logging.getLogger("aios.vision.document_analysis")


@dataclass
class PageResult:
    page_number: int = 0
    text: str = ""
    ocr_result: OCRResult = field(default_factory=OCRResult)
    image: Optional[np.ndarray] = None


@dataclass
class DocumentResult:
    pages: list[PageResult] = field(default_factory=list)
    full_text: str = ""
    page_count: int = 0
    latency_s: float = 0.0

    @property
    def text(self) -> str:
        return self.full_text or "\n\n".join(p.text for p in self.pages)


# ---------------------------------------------------------------------------
# Document Scanner (edge detection + perspective transform)
# ---------------------------------------------------------------------------

class DocumentScanner:
    """Detects document edges and applies perspective correction."""

    async def scan(self, image: np.ndarray) -> Optional[np.ndarray]:
        import cv2
        loop = asyncio.get_event_loop()

        def _scan() -> Optional[np.ndarray]:
            h, w = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 75, 200)

            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            doc_contour = None
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    doc_contour = approx
                    break

            if doc_contour is None:
                return None

            pts = doc_contour.reshape(4, 2).astype(np.float32)
            rect = self._order_points(pts)
            tw, th = self._get_target_dim(rect)
            dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(image, matrix, (tw, th))

        return await loop.run_in_executor(None, _scan)

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _get_target_dim(self, rect: np.ndarray) -> tuple[int, int]:
        (tl, tr, br, bl) = rect
        w1 = np.linalg.norm(br - bl)
        w2 = np.linalg.norm(tr - tl)
        h1 = np.linalg.norm(tr - br)
        h2 = np.linalg.norm(tl - bl)
        return int(max(w1, w2)), int(max(h1, h2))


# ---------------------------------------------------------------------------
# PDF OCR
# ---------------------------------------------------------------------------

class PDFOCR:
    """Extracts text from PDFs via image-based OCR."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._ocr_engine = OCREngine(config)

    async def process(self, pdf_path: str, start_page: int = 0, end_page: Optional[int] = None) -> DocumentResult:
        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()

        def _convert_to_images() -> list[np.ndarray]:
            try:
                import pdf2image
                images = pdf2image.convert_from_path(
                    pdf_path,
                    dpi=self._config.pdf_dpi,
                    first_page=start_page + 1,
                    last_page=end_page if end_page else None,
                )
                import cv2
                return [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]
            except ImportError:
                import fitz
                doc = fitz.open(pdf_path)
                pages = []
                s = start_page or 0
                e = end_page or len(doc)
                for i in range(s, min(e, len(doc))):
                    pix = doc[i].get_pixmap(dpi=self._config.pdf_dpi)
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                    if pix.n == 4:
                        import cv2
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                    pages.append(img)
                doc.close()
                return pages

        images = await loop.run_in_executor(None, _convert_to_images)
        logger.info("PDF converted: %d pages from %s", len(images), pdf_path)

        page_results: list[PageResult] = []
        for i, img in enumerate(images):
            ocr_result = await self._ocr_engine.recognize(img)
            page_results.append(PageResult(
                page_number=start_page + i,
                text=ocr_result.full_text,
                ocr_result=ocr_result,
                image=img,
            ))

        full_text = "\n\n".join(p.text for p in page_results)
        return DocumentResult(
            pages=page_results,
            full_text=full_text,
            page_count=len(page_results),
            latency_s=time.perf_counter() - t0,
        )

    async def close(self) -> None:
        await self._ocr_engine.close()


# ---------------------------------------------------------------------------
# Document Analyzer
# ---------------------------------------------------------------------------

class DocumentAnalyzer:
    """Unified document analysis combining scanning and OCR."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._scanner = DocumentScanner()
        self._pdf_ocr = PDFOCR(config)
        self._ocr_engine = OCREngine(config)

    async def scan_document(self, image: np.ndarray) -> tuple[Optional[np.ndarray], OCRResult]:
        corrected = await self._scanner.scan(image)
        if corrected is None:
            corrected = image
        ocr_result = await self._ocr_engine.recognize(corrected)
        return corrected, ocr_result

    async def ocr_pdf(self, pdf_path: str, start_page: int = 0, end_page: Optional[int] = None) -> DocumentResult:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        return await self._pdf_ocr.process(pdf_path, start_page, end_page)

    async def close(self) -> None:
        await self._pdf_ocr.close()
        await self._ocr_engine.close()

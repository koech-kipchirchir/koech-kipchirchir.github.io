"""
Vision Engine — unified orchestrator for all vision capabilities.

Provides a single entry point for OCR, object detection, face detection,
barcode reading, image captioning, scene analysis, document scanning,
and camera capture.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Optional, Union

import numpy as np

from vision.barcode import BarcodeDetector, BarcodeResult
from vision.camera import Camera, CameraInfo
from vision.config import VisionConfig
from vision.document_analysis import DocumentAnalyzer, DocumentResult, DocumentScanner
from vision.face_detection import FaceDetector, FaceResult
from vision.image_caption import CaptionEngine, CaptionResult
from vision.image_utils import load_image, save_image, get_image_info, preprocess_for_ocr
from vision.object_detection import DetectionResult, ObjectDetector
from vision.ocr import OCREngine, OCRResult
from vision.scene_analysis import SceneAnalyzer, SceneResult

logger = logging.getLogger("aios.vision.engine")

PathLike = Union[str, Path]


@dataclass
class VisionAnalysis:
    image_info: dict[str, Any] = field(default_factory=dict)
    ocr: Optional[OCRResult] = None
    objects: Optional[DetectionResult] = None
    faces: list[FaceResult] = field(default_factory=list)
    barcodes: list[BarcodeResult] = field(default_factory=list)
    caption: Optional[CaptionResult] = None
    scene: Optional[SceneResult] = None
    latency_s: float = 0.0


class VisionEngine:
    """Main orchestrator for all vision capabilities."""

    def __init__(self, config: Optional[VisionConfig] = None) -> None:
        self._config = config or VisionConfig.from_env()
        self._ocr_engine = OCREngine(self._config)
        self._object_detector = ObjectDetector(self._config)
        self._face_detector = FaceDetector(self._config)
        self._barcode_detector = BarcodeDetector(self._config)
        self._caption_engine = CaptionEngine(self._config)
        self._scene_analyzer = SceneAnalyzer(self._config)
        self._document_analyzer = DocumentAnalyzer(self._config)
        self._document_scanner = DocumentScanner()
        self._camera: Optional[Camera] = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        logger.info("Vision engine initialized")

    # ------------------------------------------------------------------ #
    # Image loading
    # ------------------------------------------------------------------ #

    async def load(self, source: Union[PathLike, bytes, np.ndarray], color_space: str = "RGB") -> np.ndarray:
        loop = asyncio.get_event_loop()

        def _load() -> np.ndarray:
            return load_image(source, color_space=color_space, max_size=self._config.max_image_size)

        return await loop.run_in_executor(None, _load)

    # ------------------------------------------------------------------ #
    # OCR
    # ------------------------------------------------------------------ #

    async def read_text(self, image: np.ndarray, languages: Optional[list[str]] = None) -> OCRResult:
        return await self._ocr_engine.recognize(image, languages)

    async def read_text_preprocessed(self, image: np.ndarray, languages: Optional[list[str]] = None) -> OCRResult:
        processed = preprocess_for_ocr(image)
        return await self._ocr_engine.recognize(processed, languages)

    # ------------------------------------------------------------------ #
    # Object Detection
    # ------------------------------------------------------------------ #

    async def detect_objects(self, image: np.ndarray) -> DetectionResult:
        return await self._object_detector.detect(image)

    # ------------------------------------------------------------------ #
    # Face Detection
    # ------------------------------------------------------------------ #

    async def detect_faces(self, image: np.ndarray) -> list[FaceResult]:
        return await self._face_detector.detect(image)

    async def count_faces(self, image: np.ndarray) -> int:
        return await self._face_detector.count(image)

    # ------------------------------------------------------------------ #
    # Barcode / QR
    # ------------------------------------------------------------------ #

    async def detect_barcodes(self, image: np.ndarray) -> list[BarcodeResult]:
        return await self._barcode_detector.detect(image)

    async def read_qr(self, image: np.ndarray) -> Optional[BarcodeResult]:
        return await self._barcode_detector.detect_qr(image)

    # ------------------------------------------------------------------ #
    # Caption
    # ------------------------------------------------------------------ #

    async def caption(self, image: np.ndarray, prompt: Optional[str] = None) -> CaptionResult:
        return await self._caption_engine.caption(image, prompt)

    # ------------------------------------------------------------------ #
    # Scene Analysis
    # ------------------------------------------------------------------ #

    async def analyze_scene(self, image: np.ndarray) -> SceneResult:
        return await self._scene_analyzer.analyze(image)

    # ------------------------------------------------------------------ #
    # Document
    # ------------------------------------------------------------------ #

    async def scan_document(self, image: np.ndarray) -> tuple[Optional[np.ndarray], OCRResult]:
        return await self._document_analyzer.scan_document(image)

    async def ocr_pdf(self, pdf_path: str, start_page: int = 0, end_page: Optional[int] = None) -> DocumentResult:
        return await self._document_analyzer.ocr_pdf(pdf_path, start_page, end_page)

    # ------------------------------------------------------------------ #
    # Full analysis
    # ------------------------------------------------------------------ #

    async def analyze(
        self,
        image: np.ndarray,
        enable_ocr: bool = True,
        enable_objects: bool = True,
        enable_faces: bool = True,
        enable_barcodes: bool = True,
        enable_caption: bool = False,
        enable_scene: bool = False,
    ) -> VisionAnalysis:
        t0 = time.perf_counter()
        info = get_image_info(image)
        tasks: dict[str, Any] = {}

        if enable_ocr:
            tasks["ocr"] = self._ocr_engine.recognize(image)
        if enable_objects:
            tasks["objects"] = self._object_detector.detect(image)
        if enable_faces:
            tasks["faces"] = self._face_detector.detect(image)
        if enable_barcodes:
            tasks["barcodes"] = self._barcode_detector.detect(image)
        if enable_caption:
            tasks["caption"] = self._caption_engine.caption(image)
        if enable_scene:
            tasks["scene"] = self._scene_analyzer.analyze(image)

        results: dict[str, Any] = {}
        for name, coro in tasks.items():
            try:
                results[name] = await coro
            except Exception as e:
                logger.warning("Analysis component '%s' failed: %s", name, e)

        return VisionAnalysis(
            image_info=info,
            ocr=results.get("ocr"),
            objects=results.get("objects"),
            faces=results.get("faces", []),
            barcodes=results.get("barcodes", []),
            caption=results.get("caption"),
            scene=results.get("scene"),
            latency_s=time.perf_counter() - t0,
        )

    # ------------------------------------------------------------------ #
    # Camera
    # ------------------------------------------------------------------ #

    async def start_camera(self) -> Camera:
        if self._camera is None:
            self._camera = Camera(self._config)
        await self._camera.start()
        return self._camera

    async def capture_frame(self) -> np.ndarray:
        if self._camera is None:
            self._camera = Camera(self._config)
            await self._camera.start()
        return await self._camera.read_frame()

    async def camera_stream(self) -> AsyncGenerator[np.ndarray, None]:
        if self._camera is None:
            self._camera = Camera(self._config)
            await self._camera.start()
        async for frame in self._camera.stream():
            yield frame

    async def stop_camera(self) -> None:
        if self._camera:
            await self._camera.close()
            self._camera = None

    # ------------------------------------------------------------------ #
    # Save processed image
    # ------------------------------------------------------------------ #

    async def save(self, image: np.ndarray, path: PathLike, fmt: Optional[str] = None) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, save_image, image, str(path), fmt)

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    async def close(self) -> None:
        await self.stop_camera()
        await self._ocr_engine.close()
        await self._object_detector.close()
        await self._face_detector.close()
        await self._caption_engine.close()
        await self._scene_analyzer.close()
        await self._document_analyzer.close()
        self._initialized = False
        logger.info("Vision engine closed")

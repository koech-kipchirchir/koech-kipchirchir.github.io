"""
Scene analysis combining object detection, face detection, OCR, and
image captioning into a unified understanding.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from vision.config import VisionConfig
from vision.face_detection import FaceDetector, FaceResult
from vision.image_caption import CaptionEngine, CaptionResult
from vision.object_detection import Detection, DetectionResult, ObjectDetector
from vision.ocr import OCREngine, OCRResult

logger = logging.getLogger("aios.vision.scene_analysis")


@dataclass
class SceneResult:
    caption: CaptionResult = field(default_factory=CaptionResult)
    objects: DetectionResult = field(default_factory=DetectionResult)
    faces: list[FaceResult] = field(default_factory=list)
    ocr: OCRResult = field(default_factory=OCRResult)
    scene_type: str = ""
    quality_score: float = 0.0
    analysis: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0


class SceneAnalyzer:
    """Combines vision pipelines for full scene understanding."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._object_detector = ObjectDetector(config)
        self._face_detector = FaceDetector(config)
        self._ocr_engine = OCREngine(config)
        self._caption_engine = CaptionEngine(config)

    async def analyze(self, image: np.ndarray, enable_caption: bool = True) -> SceneResult:
        t0 = time.perf_counter()
        h, w = image.shape[:2]

        tasks = {
            "objects": self._object_detector.detect(image),
            "faces": self._face_detector.detect(image),
        }
        if enable_caption:
            tasks["caption"] = self._caption_engine.caption(image)

        results: dict[str, Any] = {}
        for name, coro in tasks.items():
            try:
                results[name] = await coro
            except Exception as e:
                logger.warning("Scene analysis component %s failed: %s", name, e)
                results[name] = CaptionResult() if name == "caption" else DetectionResult() if name == "objects" else []

        caption = results.get("caption", CaptionResult())
        objects = results.get("objects", DetectionResult())
        faces = results.get("faces", [])

        scene_type = self._classify_scene(caption.caption, objects.detections, faces)
        quality = self._assess_quality(image)
        analysis = self._build_analysis(caption, objects, faces, image)

        return SceneResult(
            caption=caption,
            objects=objects,
            faces=faces,
            scene_type=scene_type,
            quality_score=quality,
            analysis=analysis,
            latency_s=time.perf_counter() - t0,
        )

    def _classify_scene(
        self,
        caption: str,
        detections: list[Detection],
        faces: list[FaceResult],
    ) -> str:
        labels = [d.label.lower() for d in detections]
        if faces:
            if len(faces) > 3:
                return "crowd"
            return "portrait"
        if any(l in labels for l in ("person", "people")):
            return "group_photo"
        if any(l in ("car", "truck", "bus", "motorcycle", "bicycle") for l in labels):
            return "outdoor_transport"
        if any(l in ("book", "laptop", "cell phone", "keyboard") for l in labels):
            return "indoor_office"
        if any(l in ("cat", "dog", "bird", "horse", "zebra") for l in labels):
            return "animal"
        if any(l in ("bottle", "cup", "bowl", "fork", "knife", "spoon") for l in labels):
            return "food"
        if any(l in ("tv", "remote") for l in labels):
            return "entertainment"
        return "general"

    def _assess_quality(self, image: np.ndarray) -> float:
        import cv2
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        size_score = min(1.0, (w * h) / (1920 * 1080))
        sharpness = min(1.0, laplacian_var / 500.0)
        return round((size_score * 0.3 + sharpness * 0.7), 4)

    def _build_analysis(
        self,
        caption: CaptionResult,
        objects: DetectionResult,
        faces: list[FaceResult],
        image: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "image_size": f"{image.shape[1]}x{image.shape[0]}",
            "object_count": len(objects.detections),
            "face_count": len(faces),
            "top_objects": [d.label for d in objects.detections[:5]],
            "average_object_confidence": round(
                sum(d.confidence for d in objects.detections) / len(objects.detections), 4
            ) if objects.detections else 0.0,
            "caption_text": caption.caption,
            "caption_model": caption.model,
        }

    async def close(self) -> None:
        await self._object_detector.close()
        await self._face_detector.close()
        await self._ocr_engine.close()
        await self._caption_engine.close()

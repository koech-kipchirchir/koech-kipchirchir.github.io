"""
Face detection providers using OpenCV Haar cascades, DNN, and MediaPipe.
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

logger = logging.getLogger("aios.vision.face_detection")


@dataclass
class FaceResult:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float = 0.0
    landmarks: Optional[list[tuple[int, int]]] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    embedding: Optional[np.ndarray] = None


class FaceDetectionProvider(ABC):
    @abstractmethod
    async def detect(self, image: np.ndarray) -> list[FaceResult]:
        pass

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# OpenCV Haar Cascade
# ---------------------------------------------------------------------------

class OpenCVHaarProvider(FaceDetectionProvider):
    def __init__(self) -> None:
        self._cascade: Any = None

    async def _ensure_cascade(self) -> None:
        if self._cascade is not None:
            return
        import cv2
        loop = asyncio.get_event_loop()

        def _load() -> Any:
            return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self._cascade = await loop.run_in_executor(None, _load)
        logger.info("OpenCV Haar cascade loaded")

    async def detect(self, image: np.ndarray) -> list[FaceResult]:
        import cv2
        await self._ensure_cascade()
        loop = asyncio.get_event_loop()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()

        def _detect() -> list[tuple[int, int, int, int]]:
            faces = self._cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
            )
            return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

        raw = await loop.run_in_executor(None, _detect)
        return [FaceResult(bbox=b, confidence=0.9) for b in raw]


# ---------------------------------------------------------------------------
# OpenCV DNN (Caffe/SSD)
# ---------------------------------------------------------------------------

class OpenCVDNNProvider(FaceDetectionProvider):
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None) -> None:
        self._model_path = model_path
        self._config_path = config_path
        self._net: Any = None

    async def _ensure_net(self) -> None:
        if self._net is not None:
            return
        import cv2
        loop = asyncio.get_event_loop()

        def _load() -> Any:
            model = self._model_path or cv2.data.findFile("opencv_face_detector_uint8.pb")
            config = self._config_path or cv2.data.findFile("opencv_face_detector.pbtxt")
            return cv2.dnn.readNetFromTensorflow(model, config)

        self._net = await loop.run_in_executor(None, _load)
        logger.info("OpenCV DNN face detector loaded")

    async def detect(self, image: np.ndarray) -> list[FaceResult]:
        import cv2
        await self._ensure_net()
        loop = asyncio.get_event_loop()

        def _detect() -> list[FaceResult]:
            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self._net.setInput(blob)
            detections = self._net.forward()
            results = []
            for i in range(detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence > 0.5:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    results.append(FaceResult(
                        bbox=(x1, y1, x2 - x1, y2 - y1),
                        confidence=confidence,
                    ))
            return results

        return await loop.run_in_executor(None, _detect)


# ---------------------------------------------------------------------------
# MediaPipe
# ---------------------------------------------------------------------------

class MediaPipeFaceProvider(FaceDetectionProvider):
    def __init__(self) -> None:
        self._face_detection: Any = None

    async def _ensure_model(self) -> None:
        if self._face_detection is not None:
            return
        import mediapipe as mp
        loop = asyncio.get_event_loop()

        def _load() -> Any:
            return mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

        self._face_detection = await loop.run_in_executor(None, _load)
        logger.info("MediaPipe face detection loaded")

    async def detect(self, image: np.ndarray) -> list[FaceResult]:
        import cv2
        await self._ensure_model()
        loop = asyncio.get_event_loop()
        rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 and image.shape[2] == 3 else image.copy()

        def _detect() -> list[FaceResult]:
            results = self._face_detection.process(rgb)
            faces = []
            if results.detections:
                h, w = rgb.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)
                    landmarks = []
                    if detection.location_data.relative_keypoints:
                        for kp in detection.location_data.relative_keypoints:
                            landmarks.append((int(kp.x * w), int(kp.y * h)))
                    faces.append(FaceResult(
                        bbox=(x, y, bw, bh),
                        confidence=detection.score[0],
                        landmarks=landmarks,
                    ))
            return faces

        return await loop.run_in_executor(None, _detect)

    async def close(self) -> None:
        if self._face_detection:
            self._face_detection.close()


# ---------------------------------------------------------------------------
# Face Detector
# ---------------------------------------------------------------------------

class FaceDetector:
    """Unified face detector."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._provider: Optional[FaceDetectionProvider] = None
        self._provider_map: dict[str, type[FaceDetectionProvider]] = {
            "opencv": OpenCVHaarProvider,
            "opencv_dnn": OpenCVDNNProvider,
            "mediapipe": MediaPipeFaceProvider,
        }

    async def _get_provider(self) -> FaceDetectionProvider:
        if self._provider is not None:
            return self._provider
        model = self._config.face_model
        cls = self._provider_map.get(model, OpenCVHaarProvider)
        kwargs = {}
        if model == "opencv_dnn" and self._config.face_model_path:
            kwargs["model_path"] = self._config.face_model_path
        self._provider = cls(**kwargs)
        return self._provider

    async def detect(self, image: np.ndarray) -> list[FaceResult]:
        provider = await self._get_provider()
        return await provider.detect(image)

    async def count(self, image: np.ndarray) -> int:
        faces = await self.detect(image)
        return len(faces)

    async def close(self) -> None:
        if self._provider:
            await self._provider.close()

"""
Object detection interface with OpenCV DNN and YOLO support.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from vision.config import VisionConfig

logger = logging.getLogger("aios.vision.object_detection")


COCO_NAMES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Detection:
    label: str = ""
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    class_id: int = 0


@dataclass
class DetectionResult:
    detections: list[Detection] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)
    latency_s: float = 0.0
    model: str = ""


class ObjectDetectionProvider(ABC):
    @abstractmethod
    async def detect(self, image: np.ndarray) -> DetectionResult:
        pass

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# OpenCV DNN (YOLO / SSD)
# ---------------------------------------------------------------------------

class OpenCVDNNObjectProvider(ObjectDetectionProvider):
    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._net: Any = None
        self._labels: list[str] = COCO_NAMES.copy()

    async def _ensure_net(self) -> None:
        if self._net is not None:
            return
        import cv2
        loop = asyncio.get_event_loop()

        model_path = self._config.detection_model_path
        labels_path = self._config.detection_labels_path

        if labels_path and os.path.exists(labels_path):
            with open(labels_path) as f:
                self._labels = [l.strip() for l in f if l.strip()]

        if model_path and os.path.exists(model_path):
            def _load_custom() -> Any:
                ext = Path(model_path).suffix.lower()
                if ext == ".pb":
                    config_path = self._config.detection_labels_path or ""
                    return cv2.dnn.readNetFromTensorflow(model_path, config_path if os.path.exists(config_path) else "")
                elif ext in (".onnx",):
                    return cv2.dnn.readNetFromONNX(model_path)
                elif ext in (".cfg",):
                    weights = model_path.replace(".cfg", ".weights")
                    return cv2.dnn.readNetFromDarknet(model_path, weights)
                return None
            self._net = await loop.run_in_executor(None, _load_custom)
            logger.info("Loaded custom detection model: %s", model_path)
        else:
            logger.warning("No detection model configured; detections will be empty")

    async def detect(self, image: np.ndarray) -> DetectionResult:
        import cv2
        t0 = time.perf_counter()
        await self._ensure_net()
        h, w = image.shape[:2]

        if self._net is None:
            return DetectionResult(image_size=(w, h), latency_s=time.perf_counter() - t0, model="opencv_dnn")

        loop = asyncio.get_event_loop()

        def _detect() -> DetectionResult:
            blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (416, 416), swapRB=True, crop=False)
            self._net.setInput(blob)
            layer_names = self._net.getUnconnectedOutLayersNames()
            outputs = self._net.forward(layer_names)
            boxes, confs, class_ids = [], [], []
            for out in outputs:
                for detection in out:
                    scores = detection[5:]
                    class_id = int(np.argmax(scores))
                    confidence = float(scores[class_id])
                    if confidence > self._config.detection_confidence:
                        cx, cy, bw, bh = (detection[0:4] * np.array([w, h, w, h])).astype(int)
                        x, y = int(cx - bw / 2), int(cy - bh / 2)
                        boxes.append((x, y, int(bw), int(bh)))
                        confs.append(confidence)
                        class_ids.append(class_id)

            indices = cv2.dnn.NMSBoxes(boxes, confs, self._config.detection_confidence, 0.4)
            detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    label = self._labels[class_ids[i]] if class_ids[i] < len(self._labels) else f"class_{class_ids[i]}"
                    detections.append(Detection(
                        label=label,
                        confidence=confs[i],
                        bbox=tuple(boxes[i]),
                        class_id=class_ids[i],
                    ))
            return DetectionResult(
                detections=detections,
                image_size=(w, h),
                latency_s=time.perf_counter() - t0,
                model="opencv_dnn",
            )

        return await loop.run_in_executor(None, _detect)


# ---------------------------------------------------------------------------
# YOLO (Ultralytics)
# ---------------------------------------------------------------------------

class YOLOProvider(ObjectDetectionProvider):
    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._model: Any = None

    async def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO
        loop = asyncio.get_event_loop()
        model_path = self._config.detection_model_path or "yolov8n.pt"

        def _load() -> Any:
            return YOLO(model_path)

        self._model = await loop.run_in_executor(None, _load)
        logger.info("YOLO model loaded: %s", model_path)

    async def detect(self, image: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()
        await self._ensure_model()
        loop = asyncio.get_event_loop()
        h, w = image.shape[:2]

        def _detect() -> DetectionResult:
            results = self._model(image, conf=self._config.detection_confidence, verbose=False)
            detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        xyxy = boxes.xyxy[i].tolist()
                        conf = float(boxes.conf[i])
                        cls_id = int(boxes.cls[i])
                        x1, y1, x2, y2 = map(int, xyxy)
                        detections.append(Detection(
                            label=results[0].names[cls_id],
                            confidence=conf,
                            bbox=(x1, y1, x2 - x1, y2 - y1),
                            class_id=cls_id,
                        ))
            return DetectionResult(
                detections=detections,
                image_size=(w, h),
                latency_s=time.perf_counter() - t0,
                model="yolo",
            )

        return await loop.run_in_executor(None, _detect)

    async def close(self) -> None:
        self._model = None


# ---------------------------------------------------------------------------
# Object Detector
# ---------------------------------------------------------------------------

class ObjectDetector:
    """Unified object detector."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._provider: Optional[ObjectDetectionProvider] = None
        self._provider_map: dict[str, type[ObjectDetectionProvider]] = {
            "opencv": OpenCVDNNObjectProvider,
            "yolo": YOLOProvider,
        }

    async def _get_provider(self) -> ObjectDetectionProvider:
        if self._provider is not None:
            return self._provider
        model = self._config.detection_model
        cls = self._provider_map.get(model, OpenCVDNNObjectProvider)
        self._provider = cls(self._config)
        return self._provider

    async def detect(self, image: np.ndarray) -> DetectionResult:
        provider = await self._get_provider()
        return await provider.detect(image)

    async def close(self) -> None:
        if self._provider:
            await self._provider.close()

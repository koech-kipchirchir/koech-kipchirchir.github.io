"""
Vision module configuration via ``VISION_*`` environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class VisionConfig:
    enabled: bool = True
    default_engine: str = "opencv"

    camera_device: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    ocr_engine: str = "easyocr"
    ocr_languages: list[str] = field(default_factory=lambda: ["en"])
    tesseract_path: Optional[str] = None
    paddleocr_use_gpu: bool = False
    paddleocr_lang: str = "en"

    detection_model: str = "opencv"
    detection_confidence: float = 0.5
    detection_model_path: Optional[str] = None
    detection_labels_path: Optional[str] = None

    face_model: str = "opencv"
    face_model_path: Optional[str] = None

    caption_model: str = ""
    caption_device: str = "cpu"

    max_image_size: int = 4096
    temp_dir: str = "temp"
    pdf_dpi: int = 200

    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> VisionConfig:
        return cls(
            enabled=os.environ.get("VISION_ENABLED", "true").lower() in ("true", "1", "yes"),
            default_engine=os.environ.get("VISION_DEFAULT_ENGINE", "opencv"),
            camera_device=int(os.environ.get("VISION_CAMERA_DEVICE", "0")),
            camera_width=int(os.environ.get("VISION_CAMERA_WIDTH", "640")),
            camera_height=int(os.environ.get("VISION_CAMERA_HEIGHT", "480")),
            camera_fps=int(os.environ.get("VISION_CAMERA_FPS", "30")),
            ocr_engine=os.environ.get("VISION_OCR_ENGINE", "easyocr"),
            ocr_languages=os.environ.get("VISION_OCR_LANGUAGES", "en").split(","),
            tesseract_path=os.environ.get("VISION_TESSERACT_PATH"),
            paddleocr_use_gpu=os.environ.get("VISION_PADDLE_OCR_USE_GPU", "false").lower() in ("true", "1"),
            paddleocr_lang=os.environ.get("VISION_PADDLE_OCR_LANG", "en"),
            detection_model=os.environ.get("VISION_DETECTION_MODEL", "opencv"),
            detection_confidence=float(os.environ.get("VISION_DETECTION_CONFIDENCE", "0.5")),
            detection_model_path=os.environ.get("VISION_DETECTION_MODEL_PATH"),
            detection_labels_path=os.environ.get("VISION_DETECTION_LABELS_PATH"),
            face_model=os.environ.get("VISION_FACE_MODEL", "opencv"),
            face_model_path=os.environ.get("VISION_FACE_MODEL_PATH"),
            caption_model=os.environ.get("VISION_CAPTION_MODEL", ""),
            caption_device=os.environ.get("VISION_CAPTION_DEVICE", "cpu"),
            max_image_size=int(os.environ.get("VISION_MAX_IMAGE_SIZE", "4096")),
            temp_dir=os.environ.get("VISION_TEMP_DIR", "temp"),
            pdf_dpi=int(os.environ.get("VISION_PDF_DPI", "200")),
            log_level=os.environ.get("VISION_LOG_LEVEL", "INFO"),
        )

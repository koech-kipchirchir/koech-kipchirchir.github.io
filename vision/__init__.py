"""
AIOS Vision Module — production-grade computer vision pipeline.

Provides OCR (EasyOCR, Tesseract, PaddleOCR), object detection,
face detection, barcode/QR reading, image captioning, scene analysis,
document scanning, PDF OCR, and camera capture.
"""

from __future__ import annotations

from vision.barcode import BarcodeDetector, BarcodeProvider, BarcodeResult, BarcodeType, OpenCVBarcodeProvider, PyzbarProvider
from vision.camera import Camera, CameraError, CameraInfo, CameraProvider, OpenCVCameraProvider
from vision.config import VisionConfig
from vision.document_analysis import DocumentAnalyzer, DocumentResult, DocumentScanner, PDFOCR, PageResult
from vision.face_detection import FaceDetectionProvider, FaceDetector, FaceResult, MediaPipeFaceProvider, OpenCVDNNProvider, OpenCVHaarProvider
from vision.image_caption import BLIPProvider, CaptionEngine, CaptionProvider, CaptionResult, GemmaVisionProvider, LlamaVisionProvider, QwenVLProvider
from vision.image_utils import (
    ImageColorSpace,
    ImageFormat,
    deskew,
    encode_image,
    enhance_contrast,
    from_pil,
    get_image_info,
    is_image_file,
    load_image,
    preprocess_for_ocr,
    resize,
    resize_to_max,
    save_image,
    to_pil,
)
from vision.object_detection import COCO_NAMES, Detection, DetectionResult, ObjectDetectionProvider, ObjectDetector, OpenCVDNNObjectProvider, YOLOProvider
from vision.ocr import EasyOCRProvider, OCRProvider, OCRResult, OCRTextBlock, OCREngine, PaddleOCRProvider, TesseractProvider
from vision.scene_analysis import SceneAnalyzer, SceneResult
from vision.vision_engine import VisionAnalysis, VisionEngine

__all__ = [
    "BLIPProvider",
    "BarcodeDetector",
    "BarcodeProvider",
    "BarcodeResult",
    "BarcodeType",
    "COCO_NAMES",
    "Camera",
    "CameraError",
    "CameraInfo",
    "CameraProvider",
    "CaptionEngine",
    "CaptionProvider",
    "CaptionResult",
    "Detection",
    "DetectionResult",
    "DocumentAnalyzer",
    "DocumentResult",
    "DocumentScanner",
    "EasyOCRProvider",
    "FaceDetectionProvider",
    "FaceDetector",
    "FaceResult",
    "GemmaVisionProvider",
    "ImageColorSpace",
    "ImageFormat",
    "LlamaVisionProvider",
    "MediaPipeFaceProvider",
    "OCRProvider",
    "OCRResult",
    "OCRTextBlock",
    "OCREngine",
    "ObjectDetectionProvider",
    "ObjectDetector",
    "OpenCVBarcodeProvider",
    "OpenCVCameraProvider",
    "OpenCVDNNObjectProvider",
    "OpenCVDNNProvider",
    "OpenCVHaarProvider",
    "PDFOCR",
    "PaddleOCRProvider",
    "PageResult",
    "PyzbarProvider",
    "QwenVLProvider",
    "SceneAnalyzer",
    "SceneResult",
    "TesseractProvider",
    "VisionAnalysis",
    "VisionConfig",
    "VisionEngine",
    "YOLOProvider",
    "deskew",
    "encode_image",
    "enhance_contrast",
    "from_pil",
    "get_image_info",
    "is_image_file",
    "load_image",
    "preprocess_for_ocr",
    "resize",
    "resize_to_max",
    "save_image",
    "to_pil",
]

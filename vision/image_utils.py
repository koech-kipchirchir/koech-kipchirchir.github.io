"""
Image loading, conversion, and preprocessing utilities.
"""

from __future__ import annotations

import io
import logging
import struct
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger("aios.vision.image_utils")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".ppm", ".pgm"}

PathLike = Union[str, Path]


class ImageFormat(str, Enum):
    JPEG = "jpeg"
    PNG = "png"
    BMP = "bmp"
    TIFF = "tiff"
    WEBP = "webp"
    PPM = "ppm"


class ImageColorSpace(str, Enum):
    BGR = "BGR"
    RGB = "RGB"
    GRAY = "GRAY"
    RGBA = "RGBA"


def is_image_file(path: PathLike) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def load_image(
    source: Union[PathLike, bytes, np.ndarray],
    color_space: str = "RGB",
    max_size: Optional[int] = None,
) -> np.ndarray:
    if isinstance(source, np.ndarray):
        img = source
    elif isinstance(source, bytes):
        arr = np.frombuffer(source, dtype=np.uint8)
        import cv2
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from bytes")
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")

    if max_size is not None:
        img = resize_to_max(img, max_size)

    if color_space.upper() == "RGB":
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color_space.upper() == "GRAY":
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    elif color_space.upper() == "BGR":
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    return img


def resize_to_max(img: np.ndarray, max_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    import cv2
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def resize(img: np.ndarray, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
    h, w = img.shape[:2]
    if width and height:
        import cv2
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    elif width:
        scale = width / w
        import cv2
        return cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_AREA)
    elif height:
        scale = height / h
        import cv2
        return cv2.resize(img, (int(w * scale), height), interpolation=cv2.INTER_AREA)
    return img


def to_pil(img: np.ndarray) -> Any:
    import cv2
    from PIL import Image
    if img.ndim == 3 and img.shape[2] == 3:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif img.ndim == 2:
        return Image.fromarray(img, mode="L")
    return Image.fromarray(img)


def from_pil(pil_image: Any) -> np.ndarray:
    import cv2
    arr = np.array(pil_image)
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def encode_image(img: np.ndarray, fmt: str = "jpeg", quality: int = 95) -> bytes:
    import cv2
    ext = fmt.lower().replace("jpg", "jpeg")
    params = []
    if ext == "jpeg":
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    success, encoded = cv2.imencode(f".{ext}", img, params)
    if not success:
        raise RuntimeError(f"Failed to encode image as {fmt}")
    return encoded.tobytes()


def save_image(img: np.ndarray, path: PathLike, fmt: Optional[str] = None) -> None:
    import cv2
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = fmt or path.suffix.lstrip(".")
    params = []
    if ext in ("jpg", "jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif ext == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    cv2.imwrite(str(path), img, params)


def get_image_info(img: np.ndarray) -> dict[str, Any]:
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    dtype = str(img.dtype)
    return {"width": w, "height": h, "channels": channels, "dtype": dtype, "size_pixels": h * w}


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    import cv2
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    import cv2
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def deskew(img: np.ndarray) -> np.ndarray:
    import cv2
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return img
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

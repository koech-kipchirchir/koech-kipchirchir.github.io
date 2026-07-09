"""
Camera capture with async streaming support.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

import numpy as np

from vision.config import VisionConfig

logger = logging.getLogger("aios.vision.camera")


class CameraError(Exception):
    pass


@dataclass
class CameraInfo:
    device: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    backend: str = "opencv"
    is_opened: bool = False


class CameraProvider(ABC):
    @abstractmethod
    def open(self, device: int, width: int, height: int, fps: int) -> None:
        pass

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def release(self) -> None:
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        pass

    @property
    @abstractmethod
    def info(self) -> CameraInfo:
        pass


class OpenCVCameraProvider(CameraProvider):
    def __init__(self) -> None:
        self._cap: Any = None
        self._info = CameraInfo()

    def open(self, device: int, width: int, height: int, fps: int) -> None:
        import cv2
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise CameraError(f"Failed to open camera device {device}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._info = CameraInfo(
            device=device, width=actual_w or width,
            height=actual_h or height, fps=int(actual_fps) or fps,
            backend="opencv", is_opened=True,
        )

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        import cv2
        ret, frame = self._cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._info.is_opened = False

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def info(self) -> CameraInfo:
        return self._info


class Camera:
    """Async camera capture wrapper."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._provider: CameraProvider = OpenCVCameraProvider()
        self._running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._frame_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=30)
        self._frame_count = 0
        self._fps_tracker: list[float] = []

    async def start(self) -> None:
        if self._running:
            return
        loop = asyncio.get_event_loop()
        self._loop = loop
        await loop.run_in_executor(
            None,
            lambda: self._provider.open(
                self._config.camera_device,
                self._config.camera_width,
                self._config.camera_height,
                self._config.camera_fps,
            ),
        )
        self._running = True
        self._frame_queue = asyncio.Queue(maxsize=30)
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Camera started: %s", self._provider.info)

    def _capture_loop(self) -> None:
        loop = self._loop
        while self._running:
            frame = self._provider.read()
            if frame is None:
                time.sleep(0.01)
                continue
            self._frame_count += 1
            self._fps_tracker.append(time.time())
            asyncio.run_coroutine_threadsafe(
                self._put_frame(frame), loop,
            )

    async def _put_frame(self, frame: np.ndarray) -> None:
        try:
            await asyncio.wait_for(self._frame_queue.put(frame), timeout=0.1)
        except asyncio.TimeoutError:
            pass

    async def read_frame(self) -> np.ndarray:
        if not self._running:
            raise CameraError("Camera not started")
        try:
            return await asyncio.wait_for(self._frame_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            raise CameraError("Timeout waiting for frame")

    async def stream(self) -> AsyncGenerator[np.ndarray, None]:
        while self._running:
            try:
                frame = await asyncio.wait_for(self._frame_queue.get(), timeout=1.0)
                yield frame
            except asyncio.TimeoutError:
                continue

    @property
    def fps(self) -> float:
        now = time.time()
        self._fps_tracker = [t for t in self._fps_tracker if now - t < 2.0]
        return len(self._fps_tracker) / 2.0 if self._fps_tracker else 0.0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def info(self) -> CameraInfo:
        return self._provider.info

    def stop(self) -> None:
        self._running = False
        logger.info("Camera stopped (frames captured: %d)", self._frame_count)

    async def close(self) -> None:
        self.stop()
        if self._thread:
            self._thread.join(timeout=3)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._provider.release)

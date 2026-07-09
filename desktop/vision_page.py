"""
Vision page — image analysis, OCR, object detection, and camera preview.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from desktop.config import DesktopConfig


class VisionPage(QWidget):
    def __init__(self, config: DesktopConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        title = QLabel("Vision & Image Analysis")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(title)

        btn_layout = QHBoxLayout()
        self._load_btn = QPushButton("Load Image")
        self._load_btn.clicked.connect(self._load_image)
        btn_layout.addWidget(self._load_btn)

        self._ocr_btn = QPushButton("Run OCR")
        self._ocr_btn.setObjectName("secondary")
        self._ocr_btn.clicked.connect(self._run_ocr)
        btn_layout.addWidget(self._ocr_btn)

        self._detect_btn = QPushButton("Detect Objects")
        self._detect_btn.setObjectName("secondary")
        self._detect_btn.clicked.connect(self._run_detection)
        btn_layout.addWidget(self._detect_btn)

        self._capture_btn = QPushButton("Capture Camera")
        self._capture_btn.setObjectName("secondary")
        self._capture_btn.clicked.connect(self._capture)
        btn_layout.addWidget(self._capture_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Image display
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._image_label = QLabel("Load an image to begin")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(400, 300)
        self._image_label.setStyleSheet("background: palette(window); border: 1px solid palette(mid); border-radius: 8px;")
        scroll.setWidget(self._image_label)
        splitter.addWidget(scroll)

        # Results
        self._results = QTextBrowser()
        self._results.setOpenExternalLinks(True)
        self._results.setPlaceholderText("Analysis results will appear here...")
        splitter.addWidget(self._results)

        splitter.setSizes([500, 400])
        layout.addWidget(splitter, 1)

        self._current_image: Optional[str] = None

    def _load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)",
        )
        if not path:
            return
        self._current_image = path
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._image_label.setText("Failed to load image")
            return
        scaled = pixmap.scaled(600, 500, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        self._image_label.setPixmap(scaled)
        self._results.setHtml(f"<p><b>Loaded:</b> {Path(path).name}</p>"
                              f"<p><b>Size:</b> {pixmap.width()} x {pixmap.height()}</p>")

    def _run_ocr(self) -> None:
        if not self._current_image:
            self._results.setHtml("<p style='color:red;'>Load an image first.</p>")
            return
        self._results.setHtml("<p>Running OCR... (requires EasyOCR/PaddleOCR backend)</p>")

    def _run_detection(self) -> None:
        if not self._current_image:
            self._results.setHtml("<p style='color:red;'>Load an image first.</p>")
            return
        self._results.setHtml("<p>Running object detection... (requires vision module)</p>")

    def _capture(self) -> None:
        self._results.setHtml("<p>Opening camera... (requires vision module)</p>")

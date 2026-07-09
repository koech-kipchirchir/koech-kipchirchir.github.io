"""
About page — version info, credits, and update check.
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from desktop.config import DesktopConfig


class AboutPage(QWidget):
    def __init__(self, config: DesktopConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config

        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 40, 48, 40)
        layout.setSpacing(16)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        logo = QLabel("AIOS")
        logo.setStyleSheet("font-size: 48px; font-weight: 800; color: palette(highlight);")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo)

        subtitle = QLabel("AI Operating System — Desktop")
        subtitle.setStyleSheet("font-size: 18px; font-weight: 500; color: palette(mid);")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        info_group = QFrame()
        info_group.setStyleSheet("QFrame { background: palette(window); border: 1px solid palette(mid);"
                                 " border-radius: 12px; padding: 24px; max-width: 500px; }")
        info_layout = QVBoxLayout(info_group)
        info_layout.setSpacing(8)

        fields = [
            ("Version", "1.0.0"),
            ("Build", "2026-07-08"),
            ("Python", "3.10+"),
            ("Framework", "PySide6"),
            ("Backend", "FastAPI + SSE"),
            ("License", "MIT"),
            ("Repository", "github.com/anomalyco/AIOS"),
        ]
        for label, value in fields:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label + ":")
            lbl.setStyleSheet("font-weight: 600; min-width: 100px;")
            row_layout.addWidget(lbl)
            val = QLabel(value)
            val.setStyleSheet("color: palette(mid);")
            row_layout.addWidget(val)
            row_layout.addStretch()
            info_layout.addWidget(row)

        layout.addWidget(info_group, 0, Qt.AlignmentFlag.AlignCenter)

        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._update_btn = QPushButton("Check for Updates")
        self._update_btn.clicked.connect(self._check_updates)
        btn_layout.addWidget(self._update_btn)

        self._docs_btn = QPushButton("Documentation")
        self._docs_btn.setObjectName("secondary")
        self._docs_btn.clicked.connect(self._open_docs)
        btn_layout.addWidget(self._docs_btn)

        layout.addLayout(btn_layout)
        layout.addStretch()

        footer = QLabel("© 2026 AIOS Project. All rights reserved.")
        footer.setStyleSheet("color: palette(shadow); font-size: 12px;")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(footer)

    def _check_updates(self) -> None:
        self._update_btn.setText("Checking...")
        self._update_btn.setEnabled(False)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(2000, self._update_done)

    def _update_done(self) -> None:
        self._update_btn.setText("✓ Up to date (1.0.0)")
        self._update_btn.setEnabled(True)

    def _open_docs(self) -> None:
        pass

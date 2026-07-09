"""
Memory page — view, search, and edit knowledge base entries.
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop.config import DesktopConfig


class MemoryPage(QWidget):
    def __init__(self, config: DesktopConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        title = QLabel("Memory & Knowledge")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(title)

        search = QLineEdit()
        search.setPlaceholderText("Search memories...")
        layout.addWidget(search)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._list = QListWidget()
        self._list.addItems([
            "User preferences",
            "Project context",
            "API credentials",
            "Custom instructions",
            "Frequently used tools",
        ])
        self._list.currentRowChanged.connect(self._show_memory)
        splitter.addWidget(self._list)

        self._editor = QTextEdit()
        self._editor.setPlaceholderText("Memory content...")
        splitter.addWidget(self._editor)

        splitter.setSizes([250, 550])
        layout.addWidget(splitter, 1)

        btn_layout = QHBoxLayout()
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._save)
        btn_layout.addWidget(self._save_btn)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._refresh)
        btn_layout.addWidget(self._refresh_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self._current_idx = -1

    def _show_memory(self, row: int) -> None:
        self._current_idx = row
        if row >= 0:
            self._editor.setPlainText(
                f"# {self._list.item(row).text()}\n\n"
                f"This is a placeholder for memory entry \"{self._list.item(row).text()}\".\n"
                f"Connect to the knowledge API to load real memory data."
            )

    def _save(self) -> None:
        pass

    def _refresh(self) -> None:
        pass

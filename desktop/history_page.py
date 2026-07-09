"""
History page — browse, search, and manage past conversations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from desktop.config import DesktopConfig
from desktop.theme import Theme


class HistoryPage(QWidget):
    def __init__(self, config: DesktopConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        title = QLabel("Chat History")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(title)

        search_layout = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search conversations...")
        self._search.textChanged.connect(self._filter)
        search_layout.addWidget(self._search)

        self._delete_btn = QPushButton("Delete Selected")
        self._delete_btn.setObjectName("danger")
        self._delete_btn.clicked.connect(self._delete_selected)
        search_layout.addWidget(self._delete_btn)

        layout.addLayout(search_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._show_chat)
        splitter.addWidget(self._list)

        self._viewer = QTextBrowser()
        self._viewer.setOpenExternalLinks(True)
        splitter.addWidget(self._viewer)

        splitter.setSizes([300, 500])
        layout.addWidget(splitter, 1)

        self._chats: list[dict] = []
        self._refresh()

    def _refresh(self) -> None:
        self._chats.clear()
        self._list.clear()
        self._viewer.clear()

        history_dir = Path(self._config.chat_history_dir)
        if history_dir.exists():
            for f in sorted(history_dir.glob("*.json"), reverse=True):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    title = data.get("title", f.stem)
                    self._chats.append({"path": str(f), "title": title, "data": data})
                    item = QListWidgetItem(f"{title}\n{f.stem[:16]}...")
                    self._list.addItem(item)
                except Exception:
                    pass

    def _filter(self, text: str) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setHidden(text.lower() not in item.text().lower() if text else False)

    def _show_chat(self, row: int) -> None:
        if row < 0 or row >= len(self._chats):
            self._viewer.clear()
            return
        chat = self._chats[row]
        msgs = chat.get("data", {}).get("messages", [])
        tokens = Theme.tokens(self._config.theme)
        html = []
        for m in msgs:
            role = m.get("role", "")
            content = m.get("content", "")
            bg = tokens.chat_user_bg if role == "user" else tokens.chat_assistant_bg
            html.append(f'<div style="background:{bg};border-radius:8px;padding:8px 12px;margin:4px 0;">'
                        f'<strong>{"You" if role=="user" else "AIOS"}:</strong><br>{content}</div>')
        self._viewer.setHtml(f"<html><body>{''.join(html)}</body></html>")

    def _delete_selected(self) -> None:
        row = self._list.currentRow()
        if row < 0 or row >= len(self._chats):
            return
        path = Path(self._chats[row]["path"])
        if path.exists():
            path.unlink()
        self._refresh()

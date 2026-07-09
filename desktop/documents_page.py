"""
Documents page — upload, manage, and view documents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from desktop.config import DesktopConfig


class DocumentsPage(QWidget):
    def __init__(self, config: DesktopConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        title = QLabel("Documents")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(title)

        btn_layout = QHBoxLayout()
        self._upload_btn = QPushButton("Upload Document")
        self._upload_btn.clicked.connect(self._upload)
        btn_layout.addWidget(self._upload_btn)

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setObjectName("danger")
        self._delete_btn.clicked.connect(self._delete)
        btn_layout.addWidget(self._delete_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._show_document)
        splitter.addWidget(self._list)

        self._viewer = QTextBrowser()
        self._viewer.setOpenExternalLinks(True)
        splitter.addWidget(self._viewer)

        splitter.setSizes([250, 550])
        layout.addWidget(splitter, 1)

        self._documents: list[Path] = []
        self._refresh()

    def _refresh(self) -> None:
        self._list.clear()
        self._viewer.clear()
        self._documents.clear()

        doc_dir = Path(self._config.chat_history_dir).parent / "documents"
        if doc_dir.exists():
            for f in sorted(doc_dir.iterdir()):
                if f.is_file():
                    self._documents.append(f)
                    self._list.addItem(f.name)

    def _upload(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Upload Documents",
            "",
            "All Files (*);;PDF (*.pdf);;Text (*.txt *.md);;Images (*.png *.jpg)",
        )
        doc_dir = Path(self._config.chat_history_dir).parent / "documents"
        doc_dir.mkdir(parents=True, exist_ok=True)
        for src in files:
            dst = doc_dir / Path(src).name
            import shutil
            shutil.copy2(src, dst)
        self._refresh()

    def _delete(self) -> None:
        row = self._list.currentRow()
        if 0 <= row < len(self._documents):
            self._documents[row].unlink()
            self._refresh()

    def _show_document(self, row: int) -> None:
        if row < 0 or row >= len(self._documents):
            self._viewer.clear()
            return
        path = self._documents[row]
        ext = path.suffix.lower()
        if ext in (".txt", ".md", ".py", ".json", ".csv", ".yaml", ".yml", ".xml", ".html", ".css", ".js"):
            self._viewer.setPlainText(path.read_text(encoding="utf-8", errors="replace"))
        elif ext == ".pdf":
            try:
                import fitz
                doc = fitz.open(str(path))
                text = "\n\n".join(page.get_text() for page in doc)
                doc.close()
                self._viewer.setPlainText(text)
            except ImportError:
                self._viewer.setPlainText(f"[PDF viewer requires PyMuPDF]\n{path}")
        else:
            self._viewer.setPlainText(f"[Preview not available for {ext} files]\n{path}")

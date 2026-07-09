"""
Plugins page — browse, install, enable/disable plugins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from desktop.config import DesktopConfig


PLUGIN_MANIFESTS: list[dict[str, Any]] = [
    {"name": "Code Interpreter", "version": "1.0.0", "description": "Execute Python code in sandboxed environment", "enabled": True, "builtin": True},
    {"name": "Web Search", "version": "1.0.0", "description": "Search the web and fetch page content", "enabled": True, "builtin": True},
    {"name": "File System", "version": "1.0.0", "description": "Read and write files in workspace", "enabled": True, "builtin": True},
    {"name": "Git Integration", "version": "0.9.0", "description": "Stage, commit, and push changes", "enabled": False, "builtin": True},
    {"name": "Database Viewer", "version": "0.8.0", "description": "Browse and query SQL databases", "enabled": False, "builtin": False},
    {"name": "Slack Connector", "version": "0.5.0", "description": "Send and receive Slack messages", "enabled": False, "builtin": False},
    {"name": "Email Client", "version": "0.4.0", "description": "Read and send emails", "enabled": False, "builtin": False},
    {"name": "API Tester", "version": "1.1.0", "description": "Test REST and GraphQL APIs", "enabled": False, "builtin": False},
]


class PluginItem(QFrame):
    def __init__(self, manifest: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._manifest = manifest
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        info_layout = QVBoxLayout()
        name = QLabel(manifest["name"])
        name.setStyleSheet("font-weight: 600;")
        info_layout.addWidget(name)

        desc = QLabel(manifest["description"])
        desc.setStyleSheet("color: palette(mid); font-size: 12px;")
        info_layout.addWidget(desc)

        meta = QLabel(f"v{manifest['version']}" + (" (built-in)" if manifest.get("builtin") else ""))
        meta.setStyleSheet("color: palette(shadow); font-size: 11px;")
        info_layout.addWidget(meta)

        layout.addLayout(info_layout, 1)

        self._toggle = QCheckBox()
        self._toggle.setChecked(manifest.get("enabled", False))
        self._toggle.setToolTip("Enable/disable plugin")
        layout.addWidget(self._toggle)

        self.setStyleSheet("PluginItem { background: palette(window); border: 1px solid palette(mid);"
                           " border-radius: 8px; margin: 2px; }")


class PluginsPage(QWidget):
    def __init__(self, config: DesktopConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        title = QLabel("Plugins")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(title)

        btn_layout = QHBoxLayout()
        self._install_btn = QPushButton("Install Plugin")
        self._install_btn.setObjectName("secondary")
        self._install_btn.clicked.connect(self._install)
        btn_layout.addWidget(self._install_btn)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._refresh)
        btn_layout.addWidget(self._refresh_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self._list = QListWidget()
        self._list.setStyleSheet("QListWidget { border: none; }")
        layout.addWidget(self._list, 1)

        self._refresh()

    def _refresh(self) -> None:
        self._list.clear()
        for manifest in PLUGIN_MANIFESTS:
            item = QListWidgetItem(self._list)
            widget = PluginItem(manifest)
            item.setSizeHint(widget.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, widget)

    def _install(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Install Plugin",
            "",
            "Plugin (*.zip *.tar.gz *.whl);;All Files (*)",
        )
        if path:
            pass

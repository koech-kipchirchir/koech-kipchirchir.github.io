"""
Application bootstrap — QApplication setup, splash screen, and system tray.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QFont, QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSplashScreen, QSystemTrayIcon, QWidget

from desktop.config import DesktopConfig
from desktop.theme import Theme


class AiosApplication(QApplication):
    """Custom QApplication with theme management."""

    def __init__(self, argv: list[str]) -> None:
        super().__init__(argv)
        self.setApplicationName("AIOS Desktop")
        self.setOrganizationName("AIOS")
        self.setApplicationVersion("1.0.0")
        self.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        self.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)

        self._config = DesktopConfig.load()
        self._theme = self._config.theme
        self.apply_theme(self._theme)

        font = QFont(self._config.font_family, self._config.font_size)
        self.setFont(font)

    @property
    def config(self) -> DesktopConfig:
        return self._config

    def apply_theme(self, name: str) -> None:
        self._theme = name
        ss = Theme.stylesheet(name)
        self.setStyleSheet(ss)
        self._config.update(theme=name)


def create_splash() -> QSplashScreen:
    pixmap = QPixmap(400, 300)
    pixmap.fill(Qt.GlobalColor.transparent)
    splash = QSplashScreen(pixmap)
    splash.showMessage(
        "Loading AIOS Desktop...",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
        Qt.GlobalColor.white,
    )
    return splash


def create_app(argv: list[str]) -> AiosApplication:
    app = AiosApplication(argv)
    app.setWindowIcon(QIcon())
    return app

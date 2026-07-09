"""
Main application window — sidebar navigation with stacked page container.
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from desktop.app import AiosApplication
from desktop.config import DesktopConfig
from desktop.icons import svg_icon
from desktop.theme import Theme

from desktop.chat_page import ChatPage
from desktop.settings_page import SettingsPage
from desktop.history_page import HistoryPage
from desktop.memory_page import MemoryPage
from desktop.documents_page import DocumentsPage
from desktop.voice_page import VoicePage
from desktop.vision_page import VisionPage
from desktop.plugins_page import PluginsPage
from desktop.about_page import AboutPage


PAGE_NAMES = [
    ("chat", "Chat"),
    ("history", "History"),
    ("memory", "Memory"),
    ("documents", "Documents"),
    ("voice", "Voice"),
    ("vision", "Vision"),
    ("plugins", "Plugins"),
    ("about", "About"),
]

SETTINGS_INDEX = len(PAGE_NAMES)  # appended after


class MainWindow(QMainWindow):
    def __init__(self, app: AiosApplication) -> None:
        super().__init__()
        self._app = app
        self._config = app.config
        self._pages: dict[str, QWidget] = {}
        self._nav_buttons: dict[str, QPushButton] = {}

        self.setWindowTitle("AIOS Desktop")
        self.setMinimumSize(800, 600)
        self._restore_geometry()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = self._build_sidebar()
        layout.addWidget(sidebar)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 1)

        self._build_pages()
        self._navigate("chat")

        self._build_menu()

    # ------------------------------------------------------------------ #
    # Sidebar
    # ------------------------------------------------------------------ #

    def _build_sidebar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("sidebar")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 12, 0, 12)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        for key, label in PAGE_NAMES:
            btn = QPushButton()
            btn.setToolTip(label)
            btn.setCheckable(True)
            btn.setFixedSize(44, 44)
            icon_svg = svg_icon(key, "currentColor", 22)
            btn.setStyleSheet(
                f"QPushButton {{ background: transparent; border: none; border-radius: 8px; }}"
                f"QPushButton:hover {{ background: palette(highlight); }}"
                f"QPushButton:checked {{ background: palette(highlight); }}"
            )
            self._set_button_icon(btn, icon_svg)
            btn.clicked.connect(lambda checked=False, k=key: self._navigate(k))
            layout.addWidget(btn, 0, Qt.AlignmentFlag.AlignHCenter)
            self._nav_buttons[key] = btn

        layout.addStretch()

        # Settings button at bottom
        self._settings_btn = QPushButton()
        self._settings_btn.setToolTip("Settings")
        self._settings_btn.setCheckable(True)
        self._settings_btn.setFixedSize(44, 44)
        self._set_button_icon(self._settings_btn, svg_icon("settings", "currentColor", 22))
        self._settings_btn.clicked.connect(lambda: self._navigate("settings"))
        layout.addWidget(self._settings_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        self._nav_buttons["settings"] = self._settings_btn

        return frame

    def _set_button_icon(self, btn: QPushButton, svg: str) -> None:
        import tempfile
        f = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
        f.write(svg.encode("utf-8"))
        f.close()
        icon = QIcon(f.name)
        btn.setIcon(icon)
        btn.setIconSize(btn.sizeHint())

    # ------------------------------------------------------------------ #
    # Pages
    # ------------------------------------------------------------------ #

    def _build_pages(self) -> None:
        page_classes: list[tuple[str, type[QWidget]]] = [
            ("chat", ChatPage),
            ("history", HistoryPage),
            ("memory", MemoryPage),
            ("documents", DocumentsPage),
            ("voice", VoicePage),
            ("vision", VisionPage),
            ("plugins", PluginsPage),
            ("about", AboutPage),
        ]
        for key, cls in page_classes:
            page = cls(self._config, self)
            self._pages[key] = page
            self._stack.addWidget(page)

        # Settings page last
        self._settings_page = SettingsPage(self._config, self)
        self._settings_page.theme_changed.connect(self._on_theme_changed)
        self._stack.addWidget(self._settings_page)

    def _navigate(self, key: str) -> None:
        for k, btn in self._nav_buttons.items():
            btn.setChecked(k == key)
        if key in self._pages:
            idx = list(self._pages.keys()).index(key)
        else:
            idx = self._stack.count() - 1  # settings is last
        self._stack.setCurrentIndex(idx)

    # ------------------------------------------------------------------ #
    # Menu bar
    # ------------------------------------------------------------------ #

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        new_action = QAction("New Chat", self)
        new_action.setShortcut(QKeySequence("Ctrl+N"))
        new_action.triggered.connect(self._new_chat)
        file_menu.addAction(new_action)

        settings_action = QAction("Settings", self)
        settings_action.setShortcut(QKeySequence("Ctrl+,"))
        settings_action.triggered.connect(lambda: self._navigate("settings"))
        file_menu.addAction(settings_action)

        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = menu.addMenu("&View")
        toggle_theme = QAction("Toggle Theme", self)
        toggle_theme.setShortcut(QKeySequence("Ctrl+T"))
        toggle_theme.triggered.connect(self._toggle_theme)
        view_menu.addAction(toggle_theme)

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #

    def _new_chat(self) -> None:
        page = self._pages.get("chat")
        if page and hasattr(page, "new_chat"):
            page.new_chat()
        self._navigate("chat")

    def _toggle_theme(self) -> None:
        new = Theme.LIGHT if self._app.config.theme == Theme.DARK else Theme.DARK
        self._app.apply_theme(new)
        self._app.config.update(theme=new)

    def _on_theme_changed(self, theme: str) -> None:
        self._app.apply_theme(theme)

    # ------------------------------------------------------------------ #
    # Geometry persistence
    # ------------------------------------------------------------------ #

    def _restore_geometry(self) -> None:
        cfg = self._config
        self.setGeometry(cfg.window_x, cfg.window_y, cfg.window_width, cfg.window_height)
        if cfg.window_maximized:
            self.showMaximized()

    def closeEvent(self, event: Any) -> None:
        cfg = self._config
        if not self.isMaximized():
            geo = self.geometry()
            cfg.update(window_x=geo.x(), window_y=geo.y(),
                       window_width=geo.width(), window_height=geo.height(),
                       window_maximized=self.isMaximized())
        else:
            cfg.update(window_maximized=True)
        super().closeEvent(event)

    @property
    def aios_app(self) -> AiosApplication:
        return self._app

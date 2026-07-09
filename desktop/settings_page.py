"""
Settings page — application preferences form.
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from desktop.config import DesktopConfig
from desktop.theme import Theme


class SettingsPage(QWidget):
    theme_changed = Signal(str)

    def __init__(self, config: DesktopConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        container = QWidget()
        scroll.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(16)

        title = QLabel("Settings")
        title.setStyleSheet("font-size: 24px; font-weight: 700; margin-bottom: 8px;")
        layout.addWidget(title)

        layout.addWidget(self._build_group("Appearance", [
            ("Theme", self._theme_combo()),
            ("Font Size", self._spin("font_size", 10, 24)),
            ("Font Family", self._text("font_family")),
        ]))

        layout.addWidget(self._build_group("API", [
            ("API URL", self._text("api_base_url")),
            ("API Key", self._password("api_key")),
            ("Model", self._text("model")),
            ("Temperature", self._slider("temperature", 0.0, 2.0, 0.1)),
            ("Max Tokens", self._spin("max_tokens", 256, 16384, 256)),
            ("Stream", self._check("stream")),
        ]))

        layout.addWidget(self._build_group("Chat", [
            ("Auto-save Chat", self._check("auto_save_chat")),
            ("Max History Items", self._spin("max_history_items", 10, 1000)),
        ]))

        layout.addWidget(self._build_group("Code", [
            ("Code Font Size", self._spin("code_font_size", 10, 24)),
            ("Code Font Family", self._text("code_font_family")),
        ]))

        layout.addWidget(self._build_group("Updates", [
            ("Check for Updates", self._check("check_updates")),
            ("Update Channel", self._combo("update_channel", ["stable", "beta", "nightly"])),
        ]))

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

        self._widgets: dict[str, Any] = {}

    def _build_group(self, title: str, fields: list[tuple[str, QWidget]]) -> QGroupBox:
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setSpacing(8)
        form.setContentsMargins(16, 24, 16, 16)
        for label, widget in fields:
            form.addRow(label, widget)
        return group

    def _text(self, key: str) -> QLineEdit:
        w = QLineEdit(str(getattr(self._config, key, "")))
        self._widgets[key] = w
        return w

    def _password(self, key: str) -> QLineEdit:
        w = QLineEdit(str(getattr(self._config, key, "")))
        w.setEchoMode(QLineEdit.EchoMode.Password)
        self._widgets[key] = w
        return w

    def _check(self, key: str) -> QCheckBox:
        w = QCheckBox()
        w.setChecked(bool(getattr(self._config, key, False)))
        self._widgets[key] = w
        return w

    def _spin(self, key: str, min_v: int, max_v: int, step: int = 1) -> QSpinBox:
        w = QSpinBox()
        w.setMinimum(min_v)
        w.setMaximum(max_v)
        w.setSingleStep(step)
        w.setValue(int(getattr(self._config, key, min_v)))
        self._widgets[key] = w
        return w

    def _slider(self, key: str, min_v: float, max_v: float, step: float) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(int(min_v / step))
        slider.setMaximum(int(max_v / step))
        slider.setValue(int(float(getattr(self._config, key, min_v)) / step))
        label = QLabel(str(slider.value() * step))
        slider.valueChanged.connect(lambda v: label.setText(f"{v * step:.1f}"))
        layout.addWidget(slider, 1)
        layout.addWidget(label)
        self._widgets[f"_{key}"] = container
        return container

    def _combo(self, key: str, items: list[str]) -> QComboBox:
        w = QComboBox()
        w.addItems(items)
        val = getattr(self._config, key, "")
        if val in items:
            w.setCurrentText(val)
        self._widgets[key] = w
        return w

    def _theme_combo(self) -> QComboBox:
        w = QComboBox()
        w.addItems([Theme.DARK, Theme.LIGHT])
        w.setCurrentText(self._config.theme)
        w.currentTextChanged.connect(self.theme_changed.emit)
        self._widgets["theme"] = w
        return w

    def _save(self) -> None:
        for key, widget in self._widgets.items():
            if isinstance(widget, QLineEdit):
                setattr(self._config, key, widget.text())
            elif isinstance(widget, QCheckBox):
                setattr(self._config, key, widget.isChecked())
            elif isinstance(widget, QSpinBox):
                setattr(self._config, key, widget.value())
            elif isinstance(widget, QComboBox):
                setattr(self._config, key, widget.currentText())
            elif key.startswith("_"):
                slider = widget.findChild(QSlider)
                if slider and "temperature" in key:
                    step = 0.1
                    setattr(self._config, "temperature", round(slider.value() * step, 1))
        self._config.save()

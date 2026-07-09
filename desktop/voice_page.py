"""
Voice page — voice settings, device selection, and voice chat controls.
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from desktop.config import DesktopConfig


class VoicePage(QWidget):
    def __init__(self, config: DesktopConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)

        title = QLabel("Voice Settings")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(title)

        desc = QLabel("Configure microphone, speaker, and voice chat preferences.")
        desc.setStyleSheet("color: palette(mid); font-size: 14px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        devices_group = QGroupBox("Audio Devices")
        dev_form = QFormLayout(devices_group)

        self._mic_combo = QComboBox()
        self._mic_combo.addItems(["Default", "Microphone (Realtek)", "Microphone (USB)"])
        dev_form.addRow("Input Device", self._mic_combo)

        self._speaker_combo = QComboBox()
        self._speaker_combo.addItems(["Default", "Speakers (Realtek)", "Headphones (USB)"])
        dev_form.addRow("Output Device", self._speaker_combo)

        self._camera_combo = QComboBox()
        self._camera_combo.addItems(["Default", "Integrated Webcam", "USB Camera"])
        dev_form.addRow("Camera", self._camera_combo)
        layout.addWidget(devices_group)

        stt_group = QGroupBox("Speech-to-Text")
        stt_form = QFormLayout(stt_group)
        self._stt_combo = QComboBox()
        self._stt_combo.addItems(["Whisper (Local)", "Whisper (API)", "Faster-Whisper", "Vosk"])
        stt_form.addRow("Engine", self._stt_combo)
        layout.addWidget(stt_group)

        tts_group = QGroupBox("Text-to-Speech")
        tts_form = QFormLayout(tts_group)
        self._tts_combo = QComboBox()
        self._tts_combo.addItems(["Edge TTS", "Piper TTS", "Coqui TTS", "System TTS"])
        tts_form.addRow("Engine", self._tts_combo)

        vol_layout = QHBoxLayout()
        vol_layout.addWidget(QLabel("Volume"))
        self._volume = QSlider(Qt.Orientation.Horizontal)
        self._volume.setMinimum(0)
        self._volume.setMaximum(100)
        self._volume.setValue(80)
        vol_layout.addWidget(self._volume)
        vol_layout.addWidget(QLabel("80%"))
        self._volume.valueChanged.connect(lambda v: vol_layout.itemAt(2).widget().setText(f"{v}%"))
        tts_form.addRow(vol_layout)
        layout.addWidget(tts_group)

        wake_group = QGroupBox("Wake Word")
        wake_form = QFormLayout(wake_group)
        self._wake_check = QCheckBox("Enable wake word detection")
        self._wake_check.setChecked(True)
        wake_form.addRow(self._wake_check)

        self._wake_combo = QComboBox()
        self._wake_combo.addItems(["Hey AIOS", "Computer", "Jarvis", "Custom"])
        wake_form.addRow("Wake Word", self._wake_combo)
        layout.addWidget(wake_group)

        test_layout = QHBoxLayout()
        test_layout.addStretch()
        self._test_btn = QPushButton("Test Microphone")
        self._test_btn.setObjectName("secondary")
        test_layout.addWidget(self._test_btn)
        self._start_btn = QPushButton("Start Voice Chat")
        self._start_btn.clicked.connect(self._toggle_voice)
        test_layout.addWidget(self._start_btn)
        layout.addLayout(test_layout)

        layout.addStretch()

    def _toggle_voice(self) -> None:
        if self._start_btn.text() == "Start Voice Chat":
            self._start_btn.setText("Stop Voice Chat")
            self._start_btn.setObjectName("danger")
            self._start_btn.style().unpolish(self._start_btn)
            self._start_btn.style().polish(self._start_btn)
        else:
            self._start_btn.setText("Start Voice Chat")
            self._start_btn.setObjectName("")
            self._start_btn.style().unpolish(self._start_btn)
            self._start_btn.style().polish(self._start_btn)

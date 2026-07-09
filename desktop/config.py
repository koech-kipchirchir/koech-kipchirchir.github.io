"""
Desktop application configuration, persisted as JSON in the user config directory.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


def _config_dir() -> Path:
    path = Path(os.environ.get("AIOS_CONFIG_DIR", Path.home() / ".aios"))
    path.mkdir(parents=True, exist_ok=True)
    return path


CONFIG_FILE = _config_dir() / "desktop.json"


@dataclass
class DesktopConfig:
    theme: str = "dark"
    language: str = "en"

    window_x: int = 100
    window_y: int = 100
    window_width: int = 1280
    window_height: int = 800
    window_maximized: bool = False

    api_base_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = True

    font_size: int = 14
    font_family: str = "Segoe UI"
    code_font_size: int = 13
    code_font_family: str = "Cascadia Code"

    chat_history_dir: str = str(_config_dir() / "chats")
    auto_save_chat: bool = True
    max_history_items: int = 100

    mic_device: str = ""
    speaker_device: str = ""
    camera_device: str = ""
    voice_enabled: bool = False
    vision_enabled: bool = False

    plugin_dir: str = str(_config_dir() / "plugins")
    enabled_plugins: list[str] = field(default_factory=list)

    check_updates: bool = True
    update_channel: str = "stable"
    last_update_check: str = ""

    @classmethod
    def load(cls) -> DesktopConfig:
        path = CONFIG_FILE
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except (json.JSONDecodeError, TypeError) as e:
            return cls()

    def save(self) -> None:
        data = asdict(self)
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def update(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self.__dataclass_fields__:
                setattr(self, k, v)
        self.save()

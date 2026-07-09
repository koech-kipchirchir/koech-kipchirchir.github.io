"""
Theme system — Dark and Light QSS stylesheets with semantic colour tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ColorTokens:
    bg_primary: str = "#1e1e2e"
    bg_secondary: str = "#181825"
    bg_surface: str = "#313244"
    bg_hover: str = "#45475a"
    bg_active: str = "#585b70"
    text_primary: str = "#cdd6f4"
    text_secondary: str = "#a6adc8"
    text_muted: str = "#6c7086"
    accent: str = "#89b4fa"
    accent_hover: str = "#74c7ec"
    accent_muted: str = "#45475a"
    success: str = "#a6e3a1"
    warning: str = "#f9e2af"
    error: str = "#f38ba8"
    info: str = "#89b4fa"
    border: str = "#45475a"
    scrollbar_bg: str = "#181825"
    scrollbar_fg: str = "#585b70"
    input_bg: str = "#313244"
    input_fg: str = "#cdd6f4"
    input_placeholder: str = "#6c7086"
    code_bg: str = "#11111b"
    code_text: str = "#cdd6f4"
    selection_bg: str = "#45475a"
    shadow: str = "rgba(0,0,0,0.3)"
    chat_user_bg: str = "#45475a"
    chat_assistant_bg: str = "#313244"
    sidebar_bg: str = "#181825"
    sidebar_hover: str = "#313244"
    sidebar_active: str = "#45475a"
    sidebar_text: str = "#a6adc8"
    sidebar_icon: str = "#a6adc8"
    header_bg: str = "#1e1e2e"
    tab_bg: str = "#1e1e2e"
    tab_active: str = "#313244"
    progress_bg: str = "#45475a"
    progress_fg: str = "#89b4fa"
    dialog_bg: str = "#313244"
    tooltip_bg: str = "#45475a"
    tooltip_text: str = "#cdd6f4"
    danger: str = "#f38ba8"
    danger_hover: str = "#eba0ac"


LIGHT_TOKENS = ColorTokens(
    bg_primary="#ffffff",
    bg_secondary="#f5f5f5",
    bg_surface="#e8e8e8",
    bg_hover="#dcdcdc",
    bg_active="#cccccc",
    text_primary="#1a1a2e",
    text_secondary="#4a4a6a",
    text_muted="#8888aa",
    accent="#4a6cf7",
    accent_hover="#3b5de7",
    accent_muted="#d0d5f7",
    success="#2ecc71",
    warning="#f39c12",
    error="#e74c3c",
    info="#3498db",
    border="#dcdcdc",
    scrollbar_bg="#f0f0f0",
    scrollbar_fg="#c0c0c0",
    input_bg="#ffffff",
    input_fg="#1a1a2e",
    input_placeholder="#8888aa",
    code_bg="#f8f8f8",
    code_text="#1a1a2e",
    selection_bg="#b0c0f0",
    shadow="rgba(0,0,0,0.1)",
    chat_user_bg="#e3e8ff",
    chat_assistant_bg="#f5f5f5",
    sidebar_bg="#f0f0f5",
    sidebar_hover="#e0e0ea",
    sidebar_active="#d0d0df",
    sidebar_text="#4a4a6a",
    sidebar_icon="#4a4a6a",
    header_bg="#ffffff",
    tab_bg="#f5f5f5",
    tab_active="#ffffff",
    progress_bg="#e0e0e0",
    progress_fg="#4a6cf7",
    dialog_bg="#ffffff",
    tooltip_bg="#4a4a6a",
    tooltip_text="#ffffff",
    danger="#e74c3c",
    danger_hover="#c0392b",
)


def build_stylesheet(tokens: ColorTokens) -> str:
    return f"""
QWidget {{
    background-color: {tokens.bg_primary};
    color: {tokens.text_primary};
    font-family: "Segoe UI", "SF Pro", "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
}}

QMainWindow {{
    background-color: {tokens.bg_primary};
}}

QFrame#sidebar {{
    background-color: {tokens.sidebar_bg};
    border-right: 1px solid {tokens.border};
    min-width: 56px;
    max-width: 56px;
}}

QFrame#sidebar QPushButton {{
    background-color: transparent;
    border: none;
    border-radius: 8px;
    padding: 10px;
    margin: 2px 6px;
    min-height: 36px;
    max-height: 36px;
}}

QFrame#sidebar QPushButton:hover {{
    background-color: {tokens.sidebar_hover};
}}

QFrame#sidebar QPushButton:checked {{
    background-color: {tokens.sidebar_active};
}}

QFrame#sidebar QPushButton#sidebar_settings {{
    margin-top: auto;
}}

QFrame#header {{
    background-color: {tokens.header_bg};
    border-bottom: 1px solid {tokens.border};
    min-height: 48px;
    max-height: 48px;
    padding: 0 16px;
}}

QLabel#header_title {{
    font-size: 16px;
    font-weight: 600;
    color: {tokens.text_primary};
}}

QTextEdit, QPlainTextEdit {{
    background-color: {tokens.input_bg};
    color: {tokens.input_fg};
    border: 1px solid {tokens.border};
    border-radius: 8px;
    padding: 8px 12px;
    selection-background-color: {tokens.selection_bg};
}}

QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {tokens.accent};
}}

QLineEdit {{
    background-color: {tokens.input_bg};
    color: {tokens.input_fg};
    border: 1px solid {tokens.border};
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: {tokens.selection_bg};
}}

QLineEdit:focus {{
    border-color: {tokens.accent};
}}

QPushButton {{
    background-color: {tokens.accent};
    color: {tokens.bg_primary};
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: {tokens.accent_hover};
}}

QPushButton:pressed {{
    background-color: {tokens.accent};
}}

QPushButton:disabled {{
    background-color: {tokens.bg_surface};
    color: {tokens.text_muted};
}}

QPushButton#secondary {{
    background-color: {tokens.bg_surface};
    color: {tokens.text_primary};
}}

QPushButton#secondary:hover {{
    background-color: {tokens.bg_hover};
}}

QPushButton#danger {{
    background-color: {tokens.danger};
    color: #ffffff;
}}

QPushButton#danger:hover {{
    background-color: {tokens.danger_hover};
}}

QComboBox {{
    background-color: {tokens.input_bg};
    color: {tokens.input_fg};
    border: 1px solid {tokens.border};
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 24px;
}}

QComboBox:hover {{
    border-color: {tokens.accent};
}}

QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}

QComboBox QAbstractItemView {{
    background-color: {tokens.bg_surface};
    color: {tokens.text_primary};
    border: 1px solid {tokens.border};
    selection-background-color: {tokens.accent};
    selection-color: {tokens.bg_primary};
}}

QCheckBox {{
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid {tokens.border};
}}

QCheckBox::indicator:checked {{
    background-color: {tokens.accent};
    border-color: {tokens.accent};
}}

QScrollBar:vertical {{
    background-color: {tokens.scrollbar_bg};
    width: 8px;
    border: none;
}}

QScrollBar::handle:vertical {{
    background-color: {tokens.scrollbar_fg};
    border-radius: 4px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {tokens.text_muted};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {tokens.scrollbar_bg};
    height: 8px;
    border: none;
}}

QScrollBar::handle:horizontal {{
    background-color: {tokens.scrollbar_fg};
    border-radius: 4px;
    min-width: 30px;
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QTabWidget::pane {{
    background-color: {tokens.bg_primary};
    border: 1px solid {tokens.border};
    border-top: none;
}}

QTabBar::tab {{
    background-color: {tokens.tab_bg};
    color: {tokens.text_secondary};
    border: none;
    padding: 8px 16px;
    min-width: 80px;
}}

QTabBar::tab:selected {{
    background-color: {tokens.tab_active};
    color: {tokens.text_primary};
    border-bottom: 2px solid {tokens.accent};
}}

QTabBar::tab:hover {{
    background-color: {tokens.bg_hover};
}}

QSplitter::handle {{
    background-color: {tokens.border};
    width: 1px;
}}

QProgressBar {{
    background-color: {tokens.progress_bg};
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {tokens.progress_fg};
    border-radius: 4px;
}}

QListWidget, QTreeWidget {{
    background-color: {tokens.bg_primary};
    border: 1px solid {tokens.border};
    border-radius: 8px;
    outline: none;
}}

QListWidget::item, QTreeWidget::item {{
    padding: 8px 12px;
    border-radius: 4px;
}}

QListWidget::item:hover, QTreeWidget::item:hover {{
    background-color: {tokens.bg_hover};
}}

QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {tokens.accent_muted};
    color: {tokens.text_primary};
}}

QDialog {{
    background-color: {tokens.dialog_bg};
}}

QLabel#message_user {{
    background-color: {tokens.chat_user_bg};
    border-radius: 12px;
    padding: 10px 14px;
    margin: 4px 0;
}}

QLabel#message_assistant {{
    background-color: {tokens.chat_assistant_bg};
    border-radius: 12px;
    padding: 10px 14px;
    margin: 4px 0;
}}

QToolTip {{
    background-color: {tokens.tooltip_bg};
    color: {tokens.tooltip_text};
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
}}

QMenu {{
    background-color: {tokens.bg_surface};
    color: {tokens.text_primary};
    border: 1px solid {tokens.border};
    border-radius: 8px;
    padding: 4px;
}}

QMenu::item {{
    padding: 6px 24px;
    border-radius: 4px;
}}

QMenu::item:selected {{
    background-color: {tokens.bg_hover};
}}

QMenu::separator {{
    height: 1px;
    background-color: {tokens.border};
    margin: 4px 8px;
}}
"""


class Theme:
    DARK = "dark"
    LIGHT = "light"

    _tokens: dict[str, ColorTokens] = {
        DARK: ColorTokens(),
        LIGHT: LIGHT_TOKENS,
    }

    @classmethod
    def stylesheet(cls, name: str = DARK) -> str:
        tokens = cls._tokens.get(name, cls._tokens[cls.DARK])
        return build_stylesheet(tokens)

    @classmethod
    def tokens(cls, name: str = DARK) -> ColorTokens:
        return cls._tokens.get(name, cls._tokens[cls.DARK])

"""
Chat page — streaming conversation with markdown rendering,
syntax highlighting, file and image upload.
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QKeySequence, QTextCursor, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from desktop.config import DesktopConfig
from desktop.theme import Theme, ColorTokens


# ---------------------------------------------------------------------------
# Markdown / code rendering helpers
# ---------------------------------------------------------------------------

def render_markdown(text: str, tokens: ColorTokens) -> str:
    try:
        import markdown as md
        html = md.markdown(
            text,
            extensions=["fenced_code", "codehilite", "tables", "nl2br"],
        )
    except ImportError:
        html = _simple_markdown(text)
    return _wrap_html(html, tokens)


def _simple_markdown(text: str) -> str:
    lines = []
    in_code = False
    for line in text.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
            if in_code:
                lines.append("<pre><code>")
            else:
                lines.append("</code></pre>")
        elif in_code:
            lines.append(line.replace("<", "&lt;").replace(">", "&gt;"))
        elif line.startswith("# "):
            lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("- ") or line.startswith("* "):
            lines.append(f"<li>{line[2:]}</li>")
        elif line.strip() == "":
            lines.append("<br>")
        else:
            lines.append(f"<p>{line}</p>")
    return "\n".join(lines)


def _wrap_html(body: str, tokens: ColorTokens) -> str:
    return f"""<!DOCTYPE html><html><head><style>
body {{ font-family: 'Segoe UI', sans-serif; font-size: 14px; line-height: 1.6;
       color: {tokens.text_primary}; background: transparent; margin: 0; padding: 0; }}
pre {{ background: {tokens.code_bg}; padding: 12px; border-radius: 8px; overflow-x: auto; }}
code {{ font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 13px; }}
p {{ margin: 6px 0; }}
h1, h2, h3 {{ margin: 12px 0 6px; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}
th, td {{ border: 1px solid {tokens.border}; padding: 6px 10px; text-align: left; }}
th {{ background: {tokens.bg_surface}; }}
blockquote {{ border-left: 3px solid {tokens.accent}; margin: 8px 0; padding: 4px 12px;
             background: {tokens.bg_surface}; }}
img {{ max-width: 100%; border-radius: 8px; }}
a {{ color: {tokens.accent}; }}
</style></head><body>{body}</body></html>"""


# ---------------------------------------------------------------------------
# Worker thread for streaming API calls
# ---------------------------------------------------------------------------

class ChatWorker(QThread):
    chunk_received = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, messages: list[dict], config: DesktopConfig) -> None:
        super().__init__()
        self._messages = messages
        self._config = config
        self._aborted = False

    def abort(self) -> None:
        self._aborted = True

    def run(self) -> None:
        import httpx
        url = f"{self._config.api_base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        payload = {
            "model": self._config.model,
            "messages": self._messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "stream": self._config.stream,
        }

        full = ""
        try:
            if self._config.stream:
                with httpx.stream("POST", url, json=payload, headers=headers, timeout=120) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if self._aborted:
                            break
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                break
                            try:
                                obj = json.loads(data)
                                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if delta:
                                    full += delta
                                    self.chunk_received.emit(delta)
                            except json.JSONDecodeError:
                                pass
            else:
                import httpx
                resp = httpx.post(url, json=payload, headers=headers, timeout=120)
                resp.raise_for_status()
                obj = resp.json()
                full = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
                if full:
                    self.chunk_received.emit(full)

            self.finished.emit(full)
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Message bubble
# ---------------------------------------------------------------------------

class MessageBubble(QFrame):
    def __init__(self, role: str, content: str, tokens: ColorTokens, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._role = role
        self._tokens = tokens

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        header = QLabel("You" if role == "user" else "AIOS")
        header.setStyleSheet(f"font-weight: 600; font-size: 12px; color: {tokens.text_secondary}; "
                             f"text-transform: uppercase; letter-spacing: 0.5px;")
        layout.addWidget(header)

        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(True)
        self._browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._browser.setStyleSheet("background: transparent; border: none;")
        self._browser.setHtml(render_markdown(content, tokens))
        layout.addWidget(self._browser)

        self.setStyleSheet(f"""
            MessageBubble {{
                background: {tokens.chat_user_bg if role == 'user' else tokens.chat_assistant_bg};
                border-radius: 12px;
                margin: 4px 40px 4px 8px;
            }}
        """)

    def append_content(self, delta: str) -> None:
        current = self._browser.toPlainText()
        self._browser.setHtml(render_markdown(current + delta, self._tokens))


# ---------------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------------

class ChatPage(QWidget):
    def __init__(self, config: DesktopConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._messages: list[dict] = []
        self._worker: Optional[ChatWorker] = None
        self._bubbles: list[MessageBubble] = []
        self._current_bubble: Optional[MessageBubble] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Scroll area for messages
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._messages_container = QWidget()
        self._messages_layout = QVBoxLayout(self._messages_container)
        self._messages_layout.setContentsMargins(16, 16, 16, 16)
        self._messages_layout.setSpacing(8)
        self._messages_layout.addStretch()

        scroll.setWidget(self._messages_container)
        layout.addWidget(scroll, 1)

        # Input area
        input_frame = QFrame()
        input_frame.setStyleSheet("QFrame { background: transparent; border-top: 1px solid palette(mid); }")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(16, 8, 16, 8)

        self._input = QTextEdit()
        self._input.setPlaceholderText("Type a message...")
        self._input.setMaximumHeight(120)
        self._input.setAcceptRichText(False)
        self._input.installEventFilter(self)
        input_layout.addWidget(self._input, 1)

        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(4)

        self._attach_btn = QPushButton("+")
        self._attach_btn.setFixedSize(36, 36)
        self._attach_btn.setToolTip("Attach file or image")
        self._attach_btn.clicked.connect(self._attach_file)
        btn_layout.addWidget(self._attach_btn)

        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedSize(72, 36)
        self._send_btn.clicked.connect(self._send_message)
        btn_layout.addWidget(self._send_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedSize(72, 36)
        self._stop_btn.setVisible(False)
        self._stop_btn.clicked.connect(self._stop_stream)
        btn_layout.addWidget(self._stop_btn)

        input_layout.addLayout(btn_layout)
        layout.addWidget(input_frame)

        self._attached_files: list[str] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def new_chat(self) -> None:
        self._messages.clear()
        self._bubbles.clear()
        self._current_bubble = None
        self._clear_layout(self._messages_layout)
        self._messages_layout.addStretch()
        self._input.clear()

    # ------------------------------------------------------------------ #
    # Send / Receive
    # ------------------------------------------------------------------ #

    def _send_message(self) -> None:
        text = self._input.toPlainText().strip()
        if not text and not self._attached_files:
            return

        tokens = Theme.tokens(self._config.theme)
        user_content: list[dict] = [{"type": "text", "text": text}]
        for fpath in self._attached_files:
            mime, _ = mimetypes.guess_type(fpath)
            if mime and mime.startswith("image/"):
                with open(fpath, "rb") as f:
                    import base64
                    b64 = base64.b64encode(f.read()).decode()
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
            else:
                with open(fpath) as f:
                    content = f.read()
                    user_content.append({"type": "text", "text": f"\n\n--- {Path(fpath).name} ---\n{content}"})

        self._attached_files.clear()
        self._input.clear()

        self._messages.append({"role": "user", "content": user_content})

        user_bubble = MessageBubble("user", text or "[Attached files]", tokens)
        self._bubbles.append(user_bubble)
        self._messages_layout.insertWidget(self._messages_layout.count() - 1, user_bubble)

        assistant_bubble = MessageBubble("assistant", "▌", tokens)
        self._bubbles.append(assistant_bubble)
        self._messages_layout.insertWidget(self._messages_layout.count() - 1, assistant_bubble)
        self._current_bubble = assistant_bubble

        self._send_btn.setVisible(False)
        self._stop_btn.setVisible(True)

        self._worker = ChatWorker(self._messages, self._config)
        self._worker.chunk_received.connect(self._on_chunk)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_chunk(self, delta: str) -> None:
        if self._current_bubble:
            self._current_bubble.append_content(delta)

    def _on_finished(self, full: str) -> None:
        self._messages.append({"role": "assistant", "content": full})
        self._send_btn.setVisible(True)
        self._stop_btn.setVisible(False)
        self._worker = None
        if self._current_bubble:
            self._current_bubble.append_content("")

    def _on_error(self, err: str) -> None:
        tokens = Theme.tokens(self._config.theme)
        bubble = MessageBubble("assistant", f"**Error:** {err}", tokens)
        self._bubbles.append(bubble)
        self._messages_layout.insertWidget(self._messages_layout.count() - 1, bubble)
        self._send_btn.setVisible(True)
        self._stop_btn.setVisible(False)
        self._worker = None

    def _stop_stream(self) -> None:
        if self._worker:
            self._worker.abort()
        self._send_btn.setVisible(True)
        self._stop_btn.setVisible(False)

    # ------------------------------------------------------------------ #
    # Attachments
    # ------------------------------------------------------------------ #

    def _attach_file(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Attach Files",
            "",
            "All Files (*);;Images (*.png *.jpg *.jpeg *.gif *.bmp);;Documents (*.pdf *.txt *.md *.csv *.json)",
        )
        self._attached_files.extend(files)
        if files:
            self._input.setPlaceholderText(f"{len(self._attached_files)} file(s) attached...")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

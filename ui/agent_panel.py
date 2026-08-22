"""Agent chat dock — type a request, watch the agent edit the project.

Requests are sent to the window, which runs the Claude tool-calling loop in a
background thread; tool execution is marshaled back to the UI thread so every
edit goes through the CommandBus like a normal edit.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui import theme

_COLORS = {"you": theme.CYAN, "agent": theme.PURPLE, "system": theme.ORANGE}


class AgentPanel(QDockWidget):
    send = Signal(str)
    cleared = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Agent", parent)
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.setMinimumWidth(220)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(6, 6, 6, 6)

        self.view = QTextEdit()
        self.view.setReadOnly(True)
        self.append("system", "Ask me to compose or edit — e.g. \"add a 4-bar drum beat\", "
                    "\"write a bassline on track 2\", \"make the pad warmer\".")

        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Message the agent…")
        self.btn = QPushButton("Send")
        self.btn_new = QPushButton("New")
        self.btn_new.setToolTip("Start a fresh conversation (clears history — cheaper next request)")
        row.addWidget(self.input, 1)
        row.addWidget(self.btn)
        row.addWidget(self.btn_new)

        self.usage_label = QLabel("session ≈ $0.0000")
        self.usage_label.setStyleSheet("color:#6b719e; font-size:10px; padding:2px;")

        layout.addWidget(self.view, 1)
        layout.addLayout(row)
        layout.addWidget(self.usage_label)
        self.setWidget(body)

        self.input.returnPressed.connect(self._send)
        self.btn.clicked.connect(self._send)
        self.btn_new.clicked.connect(self._new_chat)

    def _new_chat(self) -> None:
        self.view.clear()
        self.append("system", "New conversation — earlier history cleared.")
        self.cleared.emit()

    def _send(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.append("you", text)
        self.input.clear()
        self.send.emit(text)

    def append(self, role: str, text: str) -> None:
        color = _COLORS.get(role, "#cccccc")
        safe = text.replace("<", "&lt;").replace(">", "&gt;")
        self.view.append(f'<span style="color:{color}"><b>{role}:</b> {safe}</span>')

    def set_busy(self, busy: bool) -> None:
        self.btn.setEnabled(not busy)
        self.btn_new.setEnabled(not busy)
        self.input.setEnabled(not busy)
        self.btn.setText("…" if busy else "Send")

    def update_usage(self, info: dict) -> None:
        if not info:
            return
        cost = info.get("cumulative_cost", 0.0)
        toks = info.get("cumulative_tokens", 0)
        model = (info.get("model") or "").replace("claude-", "")
        cached = info.get("cache_read", 0)
        extra = f"  ·  {cached:,} cached" if cached else ""
        self.usage_label.setText(f"session ≈ ${cost:.4f}  ·  {toks:,} tok  ·  {model}{extra}")

"""Sound-search dock — describe a sound, get matches, drop one on the timeline.

Text query → CLAP text embedding → nearest sounds; "Similar to clip" embeds the
selected clip's audio and finds neighbours. Embedding runs in a background worker
(the panel just emits requests); double-click a result to add it as a clip.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui import theme


class SearchPanel(QDockWidget):
    search = Signal(str)          # text query
    similar = Signal()           # find sounds like the selected clip
    ingest = Signal()            # add a folder to the library
    activated = Signal(str, str, float)  # (path, name, duration) → add to timeline

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Sound Search", parent)
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.setMinimumWidth(300)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(6, 6, 6, 6)

        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Describe a sound — e.g. \"warm analog pad\"")
        self.btn = QPushButton("Search")
        row.addWidget(self.input, 1)
        row.addWidget(self.btn)
        layout.addLayout(row)

        self.results = QListWidget()
        self.results.setAlternatingRowColors(True)
        layout.addWidget(self.results, 1)

        actions = QHBoxLayout()
        self.btn_similar = QPushButton("Similar to clip")
        self.btn_ingest = QPushButton("Add folder…")
        actions.addWidget(self.btn_similar)
        actions.addWidget(self.btn_ingest)
        layout.addLayout(actions)

        self.status = QLabel("Double-click a result to add it to the selected track.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color:{theme.FG_DIM}; font-size:11px;")
        layout.addWidget(self.status)

        self.setWidget(body)

        self.input.returnPressed.connect(self._emit_search)
        self.btn.clicked.connect(self._emit_search)
        self.btn_similar.clicked.connect(self.similar.emit)
        self.btn_ingest.clicked.connect(self.ingest.emit)
        self.results.itemActivated.connect(self._activate)
        self.results.itemDoubleClicked.connect(self._activate)

    def _emit_search(self) -> None:
        q = self.input.text().strip()
        if q:
            self.search.emit(q)

    def _activate(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.UserRole)
        if data:
            self.activated.emit(data["path"], data["name"], float(data["duration"]))

    # ---- driven by the window -------------------------------------------
    def show_results(self, rows: List[dict]) -> None:
        self.results.clear()
        if not rows:
            self.set_status("No matches. Ingest a folder first, or try other words.")
            return
        for r in rows:
            score = r.get("score", 0.0)
            tags = r.get("tags", "")
            label = f"{r['name']}   ·  {r.get('duration', 0.0):.1f}s   ·  {score:+.2f}"
            if tags:
                label += f"\n    {tags}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, r)
            self.results.addItem(item)
        self.set_status(f"{len(rows)} result(s). Double-click to add to the selected track.")

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def set_busy(self, busy: bool, text: str = "") -> None:
        self.btn.setEnabled(not busy)
        self.btn_similar.setEnabled(not busy)
        self.input.setEnabled(not busy)
        if text:
            self.set_status(text)

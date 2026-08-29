"""Double-click a slider (or its readout) to type a value.

A frameless popup sits on the control; Enter commits, Escape / click-away
cancels. Cheaper and faster than a modal dialog, and stays on the cyberdeck
chrome instead of a system prompt.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLineEdit, QWidget

from ui import theme


class NumericPopup(QLineEdit):
    """Tiny type-in anchored to ``host``."""

    def __init__(
        self,
        host: QWidget,
        text: str,
        commit: Callable[[str], bool],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent or host.window())
        self._commit = commit
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setFont(QFont(theme.FONT_FAMILIES[0], 11, QFont.Bold))
        self.setStyleSheet(
            f"QLineEdit {{ background:{theme.BG_DEEP}; color:{theme.CYAN};"
            f" border:1px solid {theme.CYAN}; border-radius:2px;"
            f" padding:2px 6px; font-weight:700; }}"
        )
        self.setText(text)
        self.selectAll()
        rect = host.rect()
        top_left = host.mapToGlobal(rect.topLeft())
        self.setFixedSize(max(72, rect.width()), max(22, rect.height() + 4))
        self.move(top_left)
        self.returnPressed.connect(self._accept)
        self.show()
        self.setFocus()

    def _accept(self) -> None:
        if self._commit(self.text()):
            self.close()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)


class _DblClickFilter(QObject):
    def __init__(self, widget: QWidget, opener: Callable[[], None]) -> None:
        super().__init__(widget)
        self._open = opener

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
            self._open()
            return True
        return False


def bind_double_click_edit(
    widget: QWidget,
    *,
    getter: Callable[[], str],
    commit: Callable[[str], bool],
) -> None:
    """Open a numeric popup when ``widget`` is double-clicked."""

    def _open() -> None:
        NumericPopup(widget, getter(), commit)

    filt = _DblClickFilter(widget, _open)
    widget.installEventFilter(filt)
    widget._numeric_typein = filt  # noqa: SLF001 — keep the filter alive
    widget.setToolTip((widget.toolTip() + "\n" if widget.toolTip() else "") + "Double-click to type a value")


def parse_number(text: str) -> Optional[float]:
    raw = text.strip().replace("−", "-").replace("+", "")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_pan(text: str) -> Optional[float]:
    """``2L`` / ``3R`` / ``C`` / ``-25`` (percent) / ``-0.25`` (normalised)."""
    raw = text.strip().upper().replace(" ", "").replace("−", "-")
    if raw in ("C", "CTR", "CENTER", "0", "0.0"):
        return 0.0
    side = None
    if raw.endswith("L") or raw.startswith("L"):
        side = -1.0
        raw = raw[1:] if raw.startswith("L") else raw[:-1]
    elif raw.endswith("R") or raw.startswith("R"):
        side = 1.0
        raw = raw[1:] if raw.startswith("R") else raw[:-1]
    try:
        value = float(raw) if raw else 0.0
    except ValueError:
        return None
    if side is not None:
        return max(-1.0, min(1.0, side * abs(value) / 100.0))
    if abs(value) <= 1.0:
        return max(-1.0, min(1.0, value))
    return max(-1.0, min(1.0, value / 100.0))


def format_pan(pan: float) -> str:
    """``2L`` / ``3R`` / ``C`` from a -1..1 pan."""
    units = int(round(float(pan) * 100.0))
    if units == 0:
        return "C"
    if units < 0:
        return f"{abs(units)}L"
    return f"{units}R"


def solo_selection_states(
    arrangement_ids: list[str],
    selected_ids: list[str],
    soloed_ids: set[str],
) -> dict[str, bool]:
    """Desired solo flags after S / Solo on the current selection.

    If the selection is already exactly the solo set, unsolo that set.
    Otherwise solo exactly the selection (anything else goes down).
    """
    selected = [tid for tid in selected_ids if tid in arrangement_ids]
    if not selected:
        return {}
    if set(selected) == set(soloed_ids):
        return {tid: False for tid in selected}
    want = set(selected)
    return {tid: (tid in want) for tid in arrangement_ids}

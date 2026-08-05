"""Transport bar: play / stop / loop, tempo, and time readout.

M0: visual shell only. The buttons emit Qt signals so that the audio engine
(M3) can connect to them later without changing this widget.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)


class TransportBar(QWidget):
    """Top-of-window transport controls.

    Signals are declared now so M3 playback can wire to them; in M0 they simply
    fire and update the local play/stop button state.
    """

    play_requested = Signal()
    stop_requested = Signal()
    loop_toggled = Signal(bool)
    tempo_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build()

    def _build(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        self.play_btn = QPushButton("▶")  # ▶
        self.play_btn.setToolTip("Play (Space)")
        self.play_btn.setFixedWidth(40)
        self.play_btn.clicked.connect(self.play_requested.emit)

        self.stop_btn = QPushButton("■")  # ■
        self.stop_btn.setToolTip("Stop")
        self.stop_btn.setFixedWidth(40)
        self.stop_btn.clicked.connect(self.stop_requested.emit)

        self.loop_btn = QPushButton("↺")  # ↺
        self.loop_btn.setToolTip("Loop")
        self.loop_btn.setCheckable(True)
        self.loop_btn.setFixedWidth(40)
        self.loop_btn.toggled.connect(self.loop_toggled.emit)

        self.time_label = QLabel("00:00.000")
        self.time_label.setObjectName("timeReadout")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setMinimumWidth(96)

        tempo_label = QLabel("Tempo")
        self.tempo_spin = QDoubleSpinBox()
        self.tempo_spin.setRange(20.0, 300.0)
        self.tempo_spin.setValue(120.0)
        self.tempo_spin.setDecimals(1)
        self.tempo_spin.setSuffix(" BPM")
        self.tempo_spin.setFixedWidth(110)
        self.tempo_spin.valueChanged.connect(self.tempo_changed.emit)

        layout.addWidget(self.play_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.loop_btn)
        layout.addSpacing(12)
        layout.addWidget(self.time_label)
        layout.addStretch(1)
        layout.addWidget(tempo_label)
        layout.addWidget(self.tempo_spin)

    def set_tempo(self, bpm: float) -> None:
        """Update the tempo display without emitting tempo_changed (for syncing
        from the model on load / undo / agent edits)."""
        blocked = self.tempo_spin.blockSignals(True)
        self.tempo_spin.setValue(float(bpm))
        self.tempo_spin.blockSignals(blocked)

    def set_time(self, seconds: float) -> None:
        """Update the readout (called by playback in M3)."""
        minutes = int(seconds // 60)
        secs = seconds - minutes * 60
        self.time_label.setText(f"{minutes:02d}:{secs:06.3f}")

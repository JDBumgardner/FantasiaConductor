"""Main application window — DAW shell + command-driven editing + M3 audio.

Every mutation goes through the :class:`CommandBus` (undoable, agent-ready).
M3 adds an :class:`AudioPool` (decoded buffers + waveform peaks) and a
:class:`PlaybackEngine` (sounddevice output). The UI polls the engine's
playhead on a timer during playback; the engine never touches Qt.
"""

from __future__ import annotations

import math
import os
import pathlib
import threading
import uuid
from typing import Optional

import numpy as np

from PySide6.QtCore import QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from fantasia_core.commands import (
    AddClipCommand,
    AddTrackCommand,
    CommandBus,
    MakeMidiClipCommand,
    RemoveClipCommand,
    RemoveTrackCommand,
    SetClipAttrCommand,
    SetClipGeometryCommand,
    SetClipNotesCommand,
    SetClipSourceCommand,
    SetTempoCommand,
    SetTrackAttrCommand,
    SetTrackFxCommand,
    SetTrackSynthParamCommand,
    SplitClipCommand,
)
from fantasia_core.agent import AgentSession, AgentTools
from fantasia_core.document import (
    Note,
    Project,
    default_drum_pattern,
    default_midi_pattern,
)
from fantasia_core.document.serialize import load_project, save_project
from fantasia_core.engine import (
    AudioPool,
    MidiRenderer,
    PlaybackEngine,
    Recorder,
    SynthRenderer,
    bounce_to_file,
    bounce_track_to_file,
    default_soundfont,
    list_input_devices,
)
from fantasia_core.search import SearchService
from ui import theme
from ui.agent_panel import AgentPanel
from ui.search_panel import SearchPanel
from ui.gm_instruments import DRUM_KITS, gm_name
from ui.editor_dock import EditorDock
from ui.timeline_view import TimelineView
from ui.track_header import TrackHeaderPanel
from ui.transport_bar import TransportBar

_AUDIO_FILTER = "Audio (*.wav *.flac *.aiff *.aif *.ogg *.mp3)"
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEMO_TRACKS = [
    ("Drums", "drums.wav", theme.MAGENTA),
    ("Bass", "bass.wav", theme.GREEN),
    ("Keys", "keys.wav", theme.BLUE),
    ("Lead", "lead.wav", theme.PURPLE),
]

_STYLESHEET = f"""
* {{ font-family: "SF Pro Text", "Helvetica Neue", sans-serif; }}
QMainWindow, QWidget#central {{ background: {theme.BG_DEEP}; }}
QToolBar {{ background: {theme.BG_PANEL}; border: none; border-bottom: 1px solid {theme.BORDER_SOFT}; spacing: 3px; padding: 3px; }}
QToolBar QToolButton {{ color: {theme.FG}; padding: 4px 8px; border-radius: 4px; }}
QToolBar QToolButton:hover {{ background: {theme.BG_HOVER}; }}
QToolBar QToolButton:pressed {{ background: {theme.ACCENT_DIM}; }}
QMenuBar {{ background: {theme.BG_PANEL}; color: {theme.FG}; border-bottom: 1px solid {theme.BORDER_SOFT}; }}
QMenuBar::item {{ background: transparent; padding: 5px 10px; }}
QMenuBar::item:selected {{ background: {theme.BG_HOVER}; color: {theme.FG_BRIGHT}; }}

QWidget#trackHeader {{ background: {theme.BG_PANEL}; border-bottom: 1px solid {theme.BORDER_SOFT}; }}
QWidget#trackHeader[selected="true"] {{ background: {theme.BG_SELECTED};
    border-left: 3px solid {theme.ACCENT}; }}
QLineEdit {{ background: transparent; color: {theme.FG_BRIGHT}; font-weight: 600; }}
QLabel {{ color: {theme.FG}; }}
QPushButton {{ background: {theme.BG_ELEVATED}; color: {theme.FG}; border: 1px solid {theme.BORDER};
    border-radius: 4px; padding: 3px 6px; }}
QPushButton:hover {{ background: {theme.BG_HOVER}; border-color: {theme.PURPLE}; }}
QPushButton:checked {{ background: {theme.ACCENT}; color: #12030c; border-color: {theme.PINK};
    font-weight: 700; }}

QStatusBar {{ background: {theme.BG_PANEL}; color: {theme.FG_DIM}; }}
QStatusBar QLabel {{ color: {theme.CYAN}; }}

QDockWidget {{ color: {theme.FG}; titlebar-close-icon: none; }}
QDockWidget::title {{ background: {theme.BG_PANEL}; color: {theme.CYAN}; padding: 6px 10px;
    border-bottom: 1px solid {theme.BORDER_SOFT}; }}
QTabBar::tab {{ background: {theme.BG_PANEL}; color: {theme.FG_DIM}; padding: 6px 16px;
    border: 1px solid {theme.BORDER_SOFT}; border-bottom: none; }}
QTabBar::tab:selected {{ background: {theme.BG_SELECTED}; color: {theme.ACCENT};
    border-top: 2px solid {theme.ACCENT}; }}
QTabBar::tab:hover {{ background: {theme.BG_HOVER}; color: {theme.FG_BRIGHT}; }}

QMenu {{ background: {theme.BG_ELEVATED}; color: {theme.FG}; border: 1px solid {theme.BORDER}; }}
QMenu::item {{ padding: 5px 22px; }}
QMenu::item:selected {{ background: {theme.ACCENT}; color: #12030c; }}
QMenu::separator {{ height: 1px; background: {theme.BORDER}; margin: 4px 8px; }}

QTextEdit, QListWidget {{ background: {theme.TIMELINE_BG}; color: {theme.FG};
    border: 1px solid {theme.BORDER_SOFT}; selection-background-color: {theme.ACCENT};
    selection-color: #12030c; }}
QListWidget::item:selected {{ background: {theme.BG_SELECTED}; color: {theme.FG_BRIGHT}; }}
QListWidget::item:alternate {{ background: {theme.BG_PANEL}; }}

QComboBox, QSpinBox, QDoubleSpinBox {{ background: {theme.BG_ELEVATED}; color: {theme.FG};
    border: 1px solid {theme.BORDER}; border-radius: 4px; padding: 2px 6px; }}
QComboBox QAbstractItemView {{ background: {theme.BG_ELEVATED}; color: {theme.FG};
    selection-background-color: {theme.ACCENT}; selection-color: #12030c; }}

QSlider::groove:horizontal {{ height: 4px; background: {theme.BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {theme.CYAN}; width: 12px; margin: -5px 0;
    border-radius: 6px; }}
QSlider::sub-page:horizontal {{ background: {theme.ACCENT}; border-radius: 2px; }}

QScrollBar:vertical {{ background: {theme.BG_DEEP}; width: 12px; margin: 0; }}
QScrollBar:horizontal {{ background: {theme.BG_DEEP}; height: 12px; margin: 0; }}
QScrollBar::handle {{ background: {theme.BG_HOVER}; border-radius: 5px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {theme.PURPLE}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}

QDialog {{ background: {theme.BG_PANEL}; }}
QDialog QLabel {{ color: {theme.FG}; }}
QDialog QLineEdit, QDialog QSpinBox, QDialog QDoubleSpinBox, QInputDialog QLineEdit {{
    background: {theme.BG_ELEVATED}; color: {theme.FG_BRIGHT}; border: 1px solid {theme.BORDER};
    border-radius: 4px; padding: 5px; selection-background-color: {theme.ACCENT};
    selection-color: #12030c; }}
QDialogButtonBox QPushButton {{ min-width: 68px; padding: 5px 12px; }}

/* Drag handles. The defaults are ~4px and invisible on a dark theme, which
   makes panels feel unresizable — give them width and a hover highlight. */
QMainWindow::separator {{ background: {theme.BORDER}; width: 7px; height: 7px; }}
QMainWindow::separator:hover {{ background: {theme.ACCENT}; }}
QSplitter::handle {{ background: {theme.BORDER}; }}
QSplitter::handle:horizontal {{ width: 7px; }}
QSplitter::handle:vertical {{ height: 7px; }}
QSplitter::handle:hover {{ background: {theme.ACCENT}; }}
"""

_SECRETS_PATH = _REPO_ROOT / ".fantasia_cache" / "secrets.env"


def _load_secrets() -> None:
    """Load KEY=VALUE lines from the local (gitignored) secrets file into the
    environment, so a saved API key survives restarts. Existing env vars win."""
    try:
        if not _SECRETS_PATH.exists():
            return
        for line in _SECRETS_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:  # noqa: BLE001
        pass


def _save_secret(name: str, value: str) -> None:
    """Persist one KEY=VALUE to the local secrets file (chmod 600), replacing
    any existing line for that key."""
    _SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines, found = [], False
    if _SECRETS_PATH.exists():
        for line in _SECRETS_PATH.read_text().splitlines():
            if line.strip().startswith(f"{name}=") or line.strip().startswith(f"{name} ="):
                lines.append(f"{name}={value}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{name}={value}")
    _SECRETS_PATH.write_text("\n".join(lines) + "\n")
    try:
        os.chmod(_SECRETS_PATH, 0o600)
    except OSError:
        pass


_STOPWORDS = {"a", "an", "the", "of", "and", "with", "in", "on", "to", "for",
              "is", "that", "some", "very"}


def _prompt_tags(prompt: str, extra=("generated", "ai audio")) -> list:
    """Turn a generation prompt into a compact tag list for the sound library."""
    tags = []
    for raw in prompt.replace(",", " ").split():
        w = raw.strip(".,!?'\"()").lower()
        if w and w not in _STOPWORDS and w not in tags:
            tags.append(w)
    return tags[:12] + list(extra)


def _file_tags(path: str, extra=()) -> list:
    """Tags derived from a file name, for auto-ingesting imports/recordings."""
    base = os.path.splitext(os.path.basename(path))[0]
    words = []
    for w in base.replace("_", " ").replace("-", " ").split():
        wl = w.strip().lower()
        if wl and wl not in _STOPWORDS and wl not in words:
            words.append(wl)
    return words[:8] + list(extra)


class _GenerateDialog(QDialog):
    """Prompt + duration for MusicGen generation."""

    def __init__(self, parent=None, default_seconds: float = 4.0) -> None:
        super().__init__(parent)
        self.setWindowTitle("Generate Audio")
        form = QFormLayout(self)
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText("e.g. warm analog pad, lush and evolving")
        self.prompt.setMinimumWidth(320)
        self.seconds = QDoubleSpinBox()
        self.seconds.setRange(1.0, 30.0)
        self.seconds.setValue(default_seconds)
        self.seconds.setSingleStep(1.0)
        self.seconds.setSuffix(" s")
        self.quality = QComboBox()
        self.quality.addItem("Best (full guidance)", 3.0)
        self.quality.addItem("Draft (≈2× faster)", 1.0)
        form.addRow("Describe the sound (MusicGen):", self.prompt)
        form.addRow("Length:", self.seconds)
        form.addRow("Quality:", self.quality)
        note = QLabel("Runs on CPU (~30 s per second of audio). Draft halves that.")
        note.setStyleSheet("color:#8a8f96; font-size:11px;")
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def result_values(self):
        return (self.prompt.text().strip(), float(self.seconds.value()),
                float(self.quality.currentData()))


# format label -> (extension, {quality label: soundfile subtype})
_EXPORT_FORMATS = {
    "WAV — uncompressed": ("wav", {"16-bit": "PCM_16", "24-bit": "PCM_24", "32-bit float": "FLOAT"}),
    "FLAC — lossless": ("flac", {"16-bit": "PCM_16", "24-bit": "PCM_24"}),
    "AIFF — uncompressed": ("aiff", {"16-bit": "PCM_16", "24-bit": "PCM_24", "32-bit float": "FLOAT"}),
    "MP3 — compressed": ("mp3", {"Standard": "MPEG_LAYER_III"}),
    "OGG — compressed": ("ogg", {"Standard": "VORBIS"}),
}


class _FxDialog(QDialog):
    """Parameter editor for one effect. Spec: [(key, label, lo, hi, default, suffix)]."""

    def __init__(self, title: str, spec: list, hint: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        form = QFormLayout(self)
        self._spins = {}
        for key, label, lo, hi, default, suffix in spec:
            box = QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setDecimals(2 if hi <= 50 else 0)
            box.setSingleStep((hi - lo) / 100.0)
            box.setValue(default)
            if suffix:
                box.setSuffix(f" {suffix}")
            form.addRow(label, box)
            self._spins[key] = box
        if hint:
            note = QLabel(hint)
            note.setWordWrap(True)
            note.setStyleSheet("color:#8a8f96; font-size:11px;")
            form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def params(self) -> dict:
        return {k: b.value() for k, b in self._spins.items()}


# Effects that open a parameter dialog: action -> (fx type, title, spec, hint)
_FX_DIALOGS = {
    "add_eq_peak": ("eq_peak", "Bell / Peak EQ", [
        ("freq", "Frequency", 20.0, 20000.0, 1000.0, "Hz"),
        ("gain", "Gain", -24.0, 24.0, 0.0, "dB"),
        ("q", "Q (width)", 0.1, 10.0, 1.0, ""),
    ], "Boost or cut a band. Low Q is broad and musical; high Q is surgical. "
       "Cutting usually sounds more natural than boosting."),
    "add_eq_low_shelf": ("eq_low_shelf", "Low Shelf EQ", [
        ("freq", "Corner", 20.0, 2000.0, 200.0, "Hz"),
        ("gain", "Gain", -24.0, 24.0, 0.0, "dB"),
        ("q", "Q", 0.1, 4.0, 0.7, ""),
    ], "Tilts everything BELOW the corner — weight and body."),
    "add_eq_high_shelf": ("eq_high_shelf", "High Shelf EQ", [
        ("freq", "Corner", 1000.0, 20000.0, 6000.0, "Hz"),
        ("gain", "Gain", -24.0, 24.0, 0.0, "dB"),
        ("q", "Q", 0.1, 4.0, 0.7, ""),
    ], "Tilts everything ABOVE the corner — air and brightness."),
    "add_saturator": ("saturator", "Saturator", [
        ("drive", "Drive", 0.0, 30.0, 5.0, "dB"),
        ("output", "Output trim", -24.0, 6.0, -3.0, "dB"),
    ], "Adds harmonics so the track reads louder and warmer without more level. "
       "Output trim compensates for the drive — keep it roughly −0.6× drive."),
    "add_compressor": ("compressor", "Compressor", [
        ("threshold", "Threshold", -60.0, 0.0, -16.0, "dB"),
        ("ratio", "Ratio (n:1)", 1.0, 20.0, 4.0, ""),
        ("attack", "Attack", 0.1, 200.0, 10.0, "ms"),
        ("release", "Release", 5.0, 1000.0, 100.0, "ms"),
    ], "Turns down anything above the threshold. Fast attack tames transients "
       "(less punch); slow attack lets the hit through. 4:1 is a good default."),
}

# Effects added straight from the menu with sensible defaults.
_FX_PRESETS_EXTRA = {
    "add_limiter": {"type": "limiter", "params": {"threshold": -1.0, "release": 100.0}},
    "add_gate": {"type": "gate", "params": {"threshold": -45.0, "ratio": 4.0}},
    "add_distortion": {"type": "distortion", "params": {"drive": 12.0}},
}


class _MidiImportDialog(QDialog):
    """How to bring a .mid in — as-is, or translated into a real strum."""

    def __init__(self, filename: str, is_pattern: bool, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import MIDI")
        form = QFormLayout(self)
        self.mode = QComboBox()
        if is_pattern:
            self.mode.addItem("Play as a strum (recommended)", "strum")
        self.mode.addItem("Import notes exactly as written", "raw")
        if is_pattern:
            self.mode.addItem("Import notes, drop keyswitches", "raw_clean")
        self.chord = QComboBox()
        from fantasia_core.strum import CHORDS
        for name in CHORDS:
            self.chord.addItem(name)
        self.chord.setCurrentText("G")
        self.strum_ms = QDoubleSpinBox()
        self.strum_ms.setRange(0.0, 120.0)
        self.strum_ms.setValue(22.0)
        self.strum_ms.setSuffix(" ms")
        form.addRow(QLabel(f"<b>{filename}</b>"))
        form.addRow("Mode:", self.mode)
        form.addRow("Chord:", self.chord)
        form.addRow("Strum speed:", self.strum_ms)
        if is_pattern:
            hint = ("This file drives a sample library with keyswitches — its notes are "
                    "triggers, not music. 'Play as a strum' keeps the rhythm, accents and "
                    "stroke direction and plays them on the chord you pick.")
        else:
            hint = "Notes are placed against the project tempo, so they land on your grid."
        note = QLabel(hint)
        note.setWordWrap(True)
        note.setStyleSheet("color:#8a8f96; font-size:11px;")
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.mode.currentIndexChanged.connect(self._sync)
        self._sync()

    def _sync(self) -> None:
        strum = self.mode.currentData() == "strum"
        self.chord.setEnabled(strum)
        self.strum_ms.setEnabled(strum)

    def values(self):
        return (self.mode.currentData(), self.chord.currentText(),
                float(self.strum_ms.value()))


class _ExportDialog(QDialog):
    """Choose mix-vs-stems, format, and quality for audio export."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Audio")
        form = QFormLayout(self)
        self.scope = QComboBox()
        self.scope.addItem("Full mix (one file)", "mix")
        self.scope.addItem("Stems — one file per track", "stems")
        self.fmt = QComboBox()
        for label in _EXPORT_FORMATS:
            self.fmt.addItem(label)
        self.quality = QComboBox()
        self.fmt.currentTextChanged.connect(self._refresh_quality)
        self._refresh_quality(self.fmt.currentText())
        self.loudness = QComboBox()
        self.loudness.addItem("None (raw mix)", None)
        self.loudness.addItem("Normalize (−1 dB peak)", "normalize")
        self.loudness.addItem("Limiter (louder master)", "limiter")
        form.addRow("Export:", self.scope)
        form.addRow("Format:", self.fmt)
        form.addRow("Quality:", self.quality)
        form.addRow("Loudness:", self.loudness)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _refresh_quality(self, fmt_label: str) -> None:
        self.quality.clear()
        for q in _EXPORT_FORMATS[fmt_label][1]:
            self.quality.addItem(q)

    def result_values(self):
        ext, qmap = _EXPORT_FORMATS[self.fmt.currentText()]
        return (self.scope.currentData(), ext, qmap.get(self.quality.currentText()),
                self.loudness.currentData())


_CONTINUOUS_ATTRS = {"gain_db", "pan"}


class _TranscribeWorker(QThread):
    """Runs basic-pitch off the UI thread; emits the resulting notes."""

    done = Signal(str, list)  # (clip_id, list[Note])
    failed = Signal(str, str)  # (clip_id, error)

    def __init__(self, clip_id: str, samples, sr: int) -> None:  # noqa: ANN001
        super().__init__()
        self._clip_id = clip_id
        self._samples = samples
        self._sr = sr

    def run(self) -> None:
        try:
            from fantasia_core.transcribe import transcribe_audio

            notes = transcribe_audio(self._samples, self._sr)
            self.done.emit(self._clip_id, notes)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self._clip_id, str(exc))


class _HumWorker(QThread):
    """Monophonic hum → melody off the UI thread."""

    done = Signal(str, list)   # (clip_id, list[Note])
    failed = Signal(str, str)

    def __init__(self, clip_id: str, samples, sr: int, spb: float, bpb: int,
                 quantize: bool, key=None, scale: str = "major") -> None:
        super().__init__()
        self._clip_id = clip_id
        self._samples = samples
        self._sr = sr
        self._spb = spb
        self._bpb = bpb
        self._quantize = quantize
        self._key = key
        self._scale = scale

    def run(self) -> None:
        try:
            from fantasia_core.hum import transcribe_hum

            notes = transcribe_hum(self._samples, self._sr, spb=self._spb,
                                   bpb=self._bpb, quantize=self._quantize,
                                   key=self._key, scale=self._scale)
            self.done.emit(self._clip_id, notes)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self._clip_id, str(exc))


class _HumDialog(QDialog):
    """Options for turning a hum into notes."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Hum → Melody")
        form = QFormLayout(self)
        self.quantize = QComboBox()
        self.quantize.addItem("Snap to the grid (1/16)", True)
        self.quantize.addItem("Keep my exact timing", False)
        self.key = QComboBox()
        self.key.addItem("Any note (chromatic)", None)
        for k in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
            self.key.addItem(f"Snap to {k}", k)
        self.scale = QComboBox()
        for s in ("major", "minor", "pentatonic"):
            self.scale.addItem(s)
        form.addRow("Timing:", self.quantize)
        form.addRow("Key:", self.key)
        form.addRow("Scale:", self.scale)
        note = QLabel("Tracks a single voice — hum, whistle or sing one note at a "
                      "time. Slides between notes are ignored.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8a8f96; font-size:11px;")
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self):
        return (bool(self.quantize.currentData()), self.key.currentData(),
                self.scale.currentText())


class _GenerateWorker(QThread):
    """Runs MusicGen off the UI thread, writes a WAV, and (if given a search
    service) ingests it into the sound library so it can be reused/searched."""

    done = Signal(str, str, float)  # (clip_id, path, duration)
    failed = Signal(str, str)

    def __init__(self, clip_id: str, prompt: str, duration: float, path: str,
                 search=None, guidance: float = 3.0) -> None:
        super().__init__()
        self._clip_id = clip_id
        self._prompt = prompt
        self._duration = duration
        self._path = path
        self._search = search
        self._guidance = guidance

    def run(self) -> None:
        try:
            from fantasia_core.generate import generate_to_file

            dur = generate_to_file(self._prompt, self._duration, 44100, self._path,
                                   guidance=self._guidance)
            if self._search is not None:
                try:
                    self._search.ingest_tagged([{
                        "path": self._path, "name": self._prompt[:48],
                        "tags": _prompt_tags(self._prompt)}])
                except Exception:  # noqa: BLE001
                    pass
            self.done.emit(self._clip_id, self._path, dur)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self._clip_id, str(exc))


class _SeparateWorker(QThread):
    """Runs Demucs stem separation off the UI thread; writes one WAV per stem
    and (if given a search service) ingests each into the sound library."""

    done = Signal(str, list)  # (source_clip_id, [(name, path, duration), ...])
    failed = Signal(str, str)

    def __init__(self, clip_id: str, src_path: str, out_dir: str, base: str,
                 search=None) -> None:
        super().__init__()
        self._clip_id = clip_id
        self._src_path = src_path
        self._out_dir = out_dir
        self._base = base
        self._search = search

    def run(self) -> None:
        try:
            from fantasia_core.separate import separate_to_files

            stems = separate_to_files(self._src_path, self._out_dir, 44100)
            if self._search is not None:
                try:
                    self._search.ingest_tagged([
                        {"path": p, "name": f"{self._base} {n}",
                         "tags": [n, "stem", "isolated", "separated", self._base]}
                        for n, p, _ in stems])
                except Exception:  # noqa: BLE001
                    pass
            self.done.emit(self._clip_id, stems)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self._clip_id, str(exc))


class _IngestWorker(QThread):
    """Embeds + adds sounds to the library off the UI thread (CLAP is slow)."""

    done = Signal(int)  # count added

    def __init__(self, service, items: list) -> None:
        super().__init__()
        self._service = service
        self._items = items

    def run(self) -> None:
        try:
            self.done.emit(self._service.ingest_tagged(self._items))
        except Exception:  # noqa: BLE001
            self.done.emit(0)


class _SearchWorker(QThread):
    """Runs CLAP embedding + LanceDB query off the UI thread. On the first query
    against an empty library it seeds the demo samples so results aren't empty."""

    results = Signal(list)   # list[dict]
    ingested = Signal(int)   # count added
    failed = Signal(str)

    def __init__(self, service, kind: str, payload: str, seed: Optional[str] = None) -> None:
        super().__init__()
        self._service = service
        self._kind = kind
        self._payload = payload
        self._seed = seed

    def run(self) -> None:
        try:
            if self._kind in ("text", "audio") and self._seed and self._service.count() == 0:
                self._service.ingest_folder(self._seed)
            if self._kind == "text":
                self.results.emit(self._service.search_text(self._payload))
            elif self._kind == "audio":
                self.results.emit(self._service.search_audio_file(self._payload))
            elif self._kind == "ingest_folder":
                self.ingested.emit(self._service.ingest_folder(self._payload))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class _TTSWorker(QThread):
    """Runs Kokoro (MLX) text-to-speech off the UI thread; writes a WAV and
    optionally ingests it into the sound library."""

    done = Signal(str, str, float)  # (clip_id, path, duration)
    failed = Signal(str, str)

    def __init__(self, clip_id: str, text: str, voice: str, speed: float, path: str,
                 search=None) -> None:
        super().__init__()
        self._clip_id = clip_id
        self._text = text
        self._voice = voice
        self._speed = speed
        self._path = path
        self._search = search

    def run(self) -> None:
        try:
            from fantasia_core.tts import synthesize_to_file

            dur = synthesize_to_file(self._text, self._path, voice=self._voice,
                                     speed=self._speed, sr_out=44100)
            if self._search is not None:
                try:
                    name = f"voice: {self._text[:40]}"
                    self._search.replace_name(name)  # same text → replace the old take
                    self._search.ingest_tagged([{
                        "path": self._path, "name": name,
                        "tags": ["voice", "speech", "vocal", "tts", "spoken", self._voice]}])
                except Exception:  # noqa: BLE001
                    pass
            self.done.emit(self._clip_id, self._path, dur)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self._clip_id, str(exc))


class _TTSDialog(QDialog):
    """Text + voice + speed for Kokoro speech synthesis."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Text to Speech")
        form = QFormLayout(self)
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText("Type what the voice should say…")
        self.text.setMinimumSize(360, 90)
        self.voice = QComboBox()
        from fantasia_core.tts import VOICES

        for vid, label in VOICES:
            self.voice.addItem(label, vid)
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.5, 2.0)
        self.speed.setValue(1.0)
        self.speed.setSingleStep(0.1)
        self.speed.setSuffix("×")
        form.addRow("Text:", self.text)
        form.addRow("Voice:", self.voice)
        form.addRow("Speed:", self.speed)
        note = QLabel("Runs on the Apple GPU (MLX) — first use loads the model (~10s).")
        note.setStyleSheet("color:#8a8f96; font-size:11px;")
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def result_values(self):
        return (self.text.toPlainText().strip(), self.voice.currentData(),
                float(self.speed.value()))


class _SingWorker(QThread):
    """Renders a melody + lyrics to a sung vocal (Kokoro syllables mapped onto the
    notes) off the UI thread; writes a WAV and optionally ingests it."""

    done = Signal(str, str, float)  # (source_clip_id, path, duration)
    failed = Signal(str, str)

    def __init__(self, clip_id: str, notes: list, lyrics: str, voice: str, path: str,
                 search=None) -> None:
        super().__init__()
        self._clip_id = clip_id
        self._notes = notes
        self._lyrics = lyrics
        self._voice = voice
        self._path = path
        self._search = search

    def run(self) -> None:
        try:
            from fantasia_core.sing import sing_to_file

            dur = sing_to_file(self._notes, self._lyrics, self._path,
                               voice=self._voice, sr=44100)
            if self._search is not None:
                try:
                    self._search.ingest_tagged([{
                        "path": self._path, "name": f"vocal: {self._lyrics[:36]}",
                        "tags": ["singing", "vocal", "voice", "melody", "sung", self._voice]}])
                except Exception:  # noqa: BLE001
                    pass
            self.done.emit(self._clip_id, self._path, dur)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self._clip_id, str(exc))


class _SingDialog(QDialog):
    """Lyrics + voice for singing a MIDI melody."""

    def __init__(self, parent=None, note_count: int = 0) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sing Lyrics")
        form = QFormLayout(self)
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText("One syllable per note — split words with hyphens:\n"
                                     "fan-ta-si-a con-duc-tor")
        self.text.setMinimumSize(360, 90)
        self.voice = QComboBox()
        from fantasia_core.tts import VOICES

        for vid, label in VOICES:
            self.voice.addItem(label, vid)
        form.addRow(f"Lyrics ({note_count} notes):", self.text)
        form.addRow("Voice:", self.voice)
        note = QLabel("Vocoder-style singing — each syllable is pitched to its note. "
                      "Extra notes get a hummed 'la'.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8a8f96; font-size:11px;")
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def result_values(self):
        return self.text.toPlainText().strip(), self.voice.currentData()


class _AutotuneDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Auto-Tune")
        form = QFormLayout(self)
        self.key = QComboBox()
        for k in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
            self.key.addItem(k)
        self.key.setCurrentText("C")
        self.scale = QComboBox()
        for s in ["major", "minor", "harmonic_minor", "pentatonic", "chromatic"]:
            self.scale.addItem(s)
        self.strength = QDoubleSpinBox()
        self.strength.setRange(0.0, 1.0)
        self.strength.setValue(1.0)
        self.strength.setSingleStep(0.1)
        form.addRow("Key:", self.key)
        form.addRow("Scale:", self.scale)
        form.addRow("Strength:", self.strength)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def result_values(self):
        return {"key": self.key.currentText(), "scale": self.scale.currentText(),
                "strength": float(self.strength.value())}


class _VocalFxWorker(QThread):
    """Applies a WORLD-vocoder vocal effect off the UI thread."""

    done = Signal(str, str, str, float)  # (clip_id, op, out_path, duration)
    failed = Signal(str, str)

    def __init__(self, clip_id: str, op: str, params: dict, samples, sr: int, out_path: str) -> None:
        super().__init__()
        self._clip_id = clip_id
        self._op = op
        self._params = params
        self._samples = samples
        self._sr = sr
        self._out = out_path

    def run(self) -> None:
        try:
            import soundfile as sf

            from fantasia_core import vocalfx as vf

            x, sr, op, pr = self._samples, self._sr, self._op, self._params
            if op == "autotune":
                y = vf.autotune(x, sr, pr.get("key", "c"), pr.get("scale", "major"),
                                pr.get("strength", 1.0))
            elif op == "harmony3":
                y = vf.shift_pitch(x, sr, 4)
            elif op == "harmony5":
                y = vf.shift_pitch(x, sr, 7)
            elif op == "deess":
                y = vf.deess(x, sr)
            elif op == "double":
                y = vf.double(x, sr)
            elif op == "formant_up":
                y = vf.formant_shift(x, sr, 1.15)
            elif op == "formant_down":
                y = vf.formant_shift(x, sr, 0.87)
            else:
                self.failed.emit(self._clip_id, f"unknown vocal fx {op}")
                return
            sf.write(self._out, y, sr, subtype="PCM_16")
            self.done.emit(self._clip_id, op, self._out, len(y) / sr)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self._clip_id, str(exc))


class _StretchWorker(QThread):
    """Time-stretches a clip's audio (Rubber Band) off the UI thread."""

    done = Signal(str, str, float)  # (clip_id, out_path, duration)
    failed = Signal(str, str)

    def __init__(self, clip_id: str, samples, sr: int, factor: float, out_path: str) -> None:
        super().__init__()
        self._clip_id = clip_id
        self._samples = samples
        self._sr = sr
        self._factor = factor
        self._out = out_path

    def run(self) -> None:
        try:
            from fantasia_core.stretch import stretch_to_file

            dur = stretch_to_file(self._samples, self._sr, self._factor, self._out)
            self.done.emit(self._clip_id, self._out, dur)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self._clip_id, str(exc))


class _AgentWorker(QThread):
    """Runs the Claude tool-calling loop off the UI thread. Tool execution is
    marshaled back to the UI thread via the `tool` signal (it blocks on an event
    until the UI slot fills in the result), so bus edits stay on the UI thread."""

    text = Signal(str)
    note = Signal(str)  # status line for slow tools (shown as a system message)
    usage = Signal(object)  # per-API-call token/cost summary
    tool = Signal(object)  # (name, args, event, holder)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, session, message: str, search=None, seed: Optional[str] = None) -> None:
        super().__init__()
        self._session = session
        self._message = message
        self._search = search
        self._seed = seed

    def _execute(self, name: str, args: dict):
        if name == "generate_audio":
            return self._generate(args)  # heavy compute stays on this worker thread
        if name == "find_sound":
            return self._find(args)  # CLAP embedding stays on this worker thread
        if name == "separate_stems":
            return self._separate(args)  # Demucs stays on this worker thread
        if name == "speak":
            return self._speak(args)  # Kokoro TTS stays on this worker thread
        if name == "sing":
            return self._sing(args)  # singing synthesis stays on this worker thread
        if name == "sing_melody":
            return self._sing_melody(args)  # compose-in-time singing
        if name == "vocal_fx":
            return self._vocalfx(args)  # WORLD vocal fx stays on this worker thread
        if name in ("stretch_clip", "stretch_clip_to_bars"):
            return self._stretch(name, args)
        return self._marshal(name, args)

    def _stretch(self, name: str, args: dict):
        try:
            from fantasia_core.stretch import available, stretch_to_file
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        if not available():
            return {"error": "time-stretch unavailable — brew install rubberband + pip install pyrubberband"}
        req = {"clip_id": args.get("clip_id")}
        if name == "stretch_clip_to_bars":
            req["bars"] = args.get("bars")
        else:
            req["factor"] = args.get("factor")
        info = self._marshal("_prep_stretch", req) or {}
        if info.get("error"):
            return {"error": info["error"]}
        self.note.emit(f"⏱️ Time-stretching ×{info['factor']:.2f}…")
        try:
            cache = pathlib.Path.cwd() / ".fantasia_cache" / "stretch"
            cache.mkdir(parents=True, exist_ok=True)
            path = str(cache / f"str_{uuid.uuid4().hex[:8]}.wav")
            dur = stretch_to_file(info["samples"], info["sr"], info["factor"], path)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        res = dict(self._marshal("_apply_stretch",
                                 {"clip_id": args.get("clip_id"), "path": path, "duration": dur}) or {})
        res["new_duration"] = round(dur, 3)
        return res

    def _sing_melody(self, args: dict):
        try:
            from fantasia_core.sing import available, sing_to_file
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        if not available():
            return {"error": "singing unavailable — install the voice extra + pyworld"}
        info = self._marshal("_prep_sing_melody",
                             {"notes": args.get("notes"), "start_beat": args.get("start_beat", 0)}) or {}
        if info.get("error"):
            return {"error": info["error"]}
        self.note.emit("🎤 Composing and singing in time…")
        try:
            cache = pathlib.Path.cwd() / ".fantasia_cache" / "voice"
            cache.mkdir(parents=True, exist_ok=True)
            path = str(cache / f"sing_{uuid.uuid4().hex[:8]}.wav")
            dur = sing_to_file(info["notes"], str(args["lyrics"]), path,
                               voice=args.get("voice", "af_heart"), sr=44100)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        if self._search is not None:
            try:
                self._search.ingest_tagged([{
                    "path": path, "name": f"vocal: {str(args['lyrics'])[:36]}",
                    "tags": ["singing", "vocal", "voice", "sung", "in-time"]}])
            except Exception:  # noqa: BLE001
                pass
        res = dict(self._marshal("_add_vocal", {
            "path": path, "duration": dur, "start": info.get("start", 0.0),
            "base": "vocal"}) or {})
        res["sung_seconds"] = round(dur, 2)
        res["synced"] = True
        return res

    def _vocalfx(self, args: dict):
        try:
            import soundfile as sf

            from fantasia_core import vocalfx as vf
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        if not vf.available():
            return {"error": "vocal fx unavailable — pip install pyworld"}
        info = self._marshal("_prep_vocalfx", {"clip_id": args.get("clip_id")}) or {}
        if info.get("error"):
            return {"error": info["error"]}
        self.note.emit("🎚️ Applying vocal FX…")
        x, sr, eff = info["samples"], info["sr"], args.get("effect", "autotune")
        try:
            if eff == "autotune":
                y = vf.autotune(x, sr, args.get("key", "c"), args.get("scale", "major"),
                                float(args.get("strength", 1.0)))
            elif eff == "harmony":
                y = vf.shift_pitch(x, sr, float(args.get("semitones", 4)))
            elif eff == "formant_up":
                y = vf.formant_shift(x, sr, 1.15)
            elif eff == "formant_down":
                y = vf.formant_shift(x, sr, 0.87)
            elif eff == "deess":
                y = vf.deess(x, sr)
            elif eff == "double":
                y = vf.double(x, sr)
            else:
                return {"error": f"unknown effect {eff}"}
            cache = pathlib.Path.cwd() / ".fantasia_cache" / "voice"
            cache.mkdir(parents=True, exist_ok=True)
            path = str(cache / f"vfx_{uuid.uuid4().hex[:8]}.wav")
            sf.write(path, y, sr, subtype="PCM_16")
            dur = len(y) / sr
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        return dict(self._marshal("_apply_vocalfx_result", {
            "clip_id": args.get("clip_id"), "effect": eff, "path": path,
            "duration": dur, "base": info.get("base"), "start": info.get("start", 0.0)}) or {})

    def _sing(self, args: dict):
        try:
            from fantasia_core.sing import available, sing_to_file
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        if not available():
            return {"error": "singing unavailable — install the voice extra + librosa"}
        info = self._marshal("_prep_sing", {"clip_id": args.get("clip_id")}) or {}
        if info.get("error"):
            return {"error": info["error"]}
        self.note.emit("🎤 Singing the melody on the GPU…")
        try:
            cache = pathlib.Path.cwd() / ".fantasia_cache" / "voice"
            cache.mkdir(parents=True, exist_ok=True)
            path = str(cache / f"sing_{uuid.uuid4().hex[:8]}.wav")
            dur = sing_to_file(info["notes"], str(args["lyrics"]), path,
                               voice=args.get("voice", "af_heart"), sr=44100)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        if self._search is not None:
            try:
                self._search.ingest_tagged([{
                    "path": path, "name": f"vocal: {str(args['lyrics'])[:36]}",
                    "tags": ["singing", "vocal", "voice", "sung"]}])
            except Exception:  # noqa: BLE001
                pass
        res = dict(self._marshal("_add_vocal", {
            "path": path, "duration": dur, "start": info.get("start", 0.0),
            "base": info.get("base", "melody")}) or {})
        res["sung_seconds"] = round(dur, 2)
        return res

    def _speak(self, args: dict):
        try:
            from fantasia_core.tts import available, synthesize_to_file
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        if not available():
            return {"error": "text-to-speech unavailable — pip install mlx-audio 'misaki[en]'"}
        self.note.emit("🗣️ Synthesizing speech on the GPU…")
        try:
            cache = pathlib.Path.cwd() / ".fantasia_cache" / "voice"
            cache.mkdir(parents=True, exist_ok=True)
            path = str(cache / f"tts_{uuid.uuid4().hex[:8]}.wav")
            dur = synthesize_to_file(str(args["text"]), path,
                                     voice=args.get("voice", "af_heart"),
                                     speed=float(args.get("speed", 1.0)), sr_out=44100)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        if self._search is not None:
            try:
                name = f"voice: {str(args['text'])[:40]}"
                self._search.replace_name(name)  # same text → replace the old take
                self._search.ingest_tagged([{
                    "path": path, "name": name,
                    "tags": ["voice", "speech", "vocal", "tts", "spoken"]}])
            except Exception:  # noqa: BLE001
                pass
        fill = {"path": path, "duration": dur, "clip_id": args.get("clip_id"),
                "track_id": args.get("track_id"), "start": args.get("start", 0.0)}
        res = dict(self._marshal("_fill_generated", fill) or {})
        res["spoken_seconds"] = round(dur, 2)
        return res

    def _separate(self, args: dict):
        try:
            from fantasia_core.separate import available, separate_to_files
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        if not available():
            return {"error": "stem separation unavailable — pip install demucs"}
        self.note.emit("🎚️ Separating stems on CPU — this can take a bit…")
        info = self._marshal("_prep_separate", {"clip_id": args.get("clip_id")}) or {}
        if info.get("error"):
            return {"error": info["error"]}
        try:
            stems = separate_to_files(info["src_path"], info["out_dir"], 44100)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        if self._search is not None:
            try:
                self._search.ingest_tagged([
                    {"path": p, "name": f"{info['base']} {n}",
                     "tags": [n, "stem", "isolated", "separated", info["base"]]}
                    for n, p, _ in stems])
            except Exception:  # noqa: BLE001
                pass
        res = dict(self._marshal("_add_stems", {
            "stems": stems, "start": info.get("start", 0.0), "base": info["base"]}) or {})
        res["stem_names"] = [n for n, _, _ in stems]
        return res

    def _find(self, args: dict):
        if self._search is None or not self._search.available():
            return {"error": "sound search unavailable — install torch+transformers"}
        try:
            if self._seed and self._search.count() == 0:
                self._search.ingest_folder(self._seed)
            return {"results": self._search.search_text(args["query"], int(args.get("k", 8)))}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def _marshal(self, name: str, args: dict):
        """Run a tool on the UI thread (bus edits) and wait for its result."""
        holder: dict = {}
        event = threading.Event()
        self.tool.emit((name, args, event, holder))
        event.wait()
        return holder.get("result")

    def _generate(self, args: dict):
        try:
            from fantasia_core.generate import available, generate_to_file
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        if not available():
            return {"error": "audio generation unavailable — install torch+transformers"}
        secs = float(args.get("duration", 4.0))
        self.note.emit(f"🎵 Generating {secs:.0f}s of audio on CPU — this can take a minute or two…")
        try:
            cache = pathlib.Path.cwd() / ".fantasia_cache" / "generated"
            cache.mkdir(parents=True, exist_ok=True)
            path = str(cache / f"gen_{uuid.uuid4().hex[:8]}.wav")
            dur = generate_to_file(args["prompt"], float(args.get("duration", 4.0)), 44100, path)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        if self._search is not None:  # make agent-generated audio reusable/searchable
            try:
                self._search.ingest_tagged([{
                    "path": path, "name": str(args["prompt"])[:48],
                    "tags": _prompt_tags(str(args["prompt"]))}])
            except Exception:  # noqa: BLE001
                pass
        # Marshal only the bus mutation (fill the clip) onto the UI thread.
        fill = {"path": path, "duration": dur, "clip_id": args.get("clip_id"),
                "track_id": args.get("track_id"), "start": args.get("start", 0.0)}
        res = dict(self._marshal("_fill_generated", fill) or {})
        res["generated_seconds"] = round(dur, 2)
        return res

    def run(self) -> None:
        try:
            final = self._session.run(self._message, on_text=self.text.emit,
                                      execute_tool=self._execute, on_usage=self.usage.emit)
            self.done.emit(final or "")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        _load_secrets()  # pick up a saved ANTHROPIC_API_KEY before the agent inits
        self.setWindowTitle("Fantasia Conductor")
        self.resize(1280, 760)
        self.setStyleSheet(_STYLESHEET)

        self.project = Project(name="Untitled")
        self.project.add_track("Track 1")  # one-time setup (not undoable)
        self.bus = CommandBus(self.project)
        self.pool = AudioPool(self.project.sample_rate)
        self.midi = MidiRenderer(default_soundfont(), self.project.sample_rate)
        self.synth_engine = SynthRenderer(self.project.sample_rate)
        self.engine = PlaybackEngine(self.project, self.pool, self.project.sample_rate)
        self.engine.midi_renderer = self.midi
        self.engine.synth_renderer = self.synth_engine

        self.search_service = SearchService(
            str(_REPO_ROOT / ".fantasia_cache" / "soundlib.lancedb")
        )
        self._seed_folder = str(_REPO_ROOT / "assets" / "samples")
        self._search_worker = None

        self._preview_lock = threading.Lock()  # serialises note-preview playback
        self.recorder = Recorder(sample_rate=self.project.sample_rate)
        self.record_input_device = None
        self._rec_timer = QTimer(self)
        self._rec_timer.setInterval(200)
        self._rec_timer.timeout.connect(self._on_rec_tick)
        # Debounce tempo-lock re-stretching until the tempo stops changing.
        self._tempo_conform_timer = QTimer(self)
        self._tempo_conform_timer.setSingleShot(True)
        self._tempo_conform_timer.setInterval(450)
        self._tempo_conform_timer.timeout.connect(self._conform_locked_clips)

        self.agent_tools = AgentTools(
            self.bus, refresh=self._agent_refresh, search=self.search_service
        )
        self.agent = AgentSession(self.agent_tools)
        self._agent_busy = False
        self._agent_worker = None

        # Local control bridge (MCP / external frontends). The executor is an
        # UNSTARTED _AgentWorker: its _execute runs heavy tools on the calling
        # (HTTP) thread and marshals bus edits to the UI thread via the same
        # signal path as in-app agent calls.
        self._bridge_exec = _AgentWorker(self.agent, "", search=self.search_service,
                                         seed=str(_REPO_ROOT / "assets" / "samples"))
        self._bridge_exec.tool.connect(self._on_agent_tool)
        self._bridge_exec.note.connect(lambda s: self.statusBar().showMessage(s))
        from fantasia_core.bridge import DEFAULT_PORT, ControlBridge

        self.bridge = ControlBridge(
            self.agent_tools.definitions, self._bridge_exec._execute,
            port=int(os.environ.get("FANTASIA_BRIDGE_PORT", DEFAULT_PORT)))
        self._bridge_ok = self.bridge.start()  # False if another instance owns the port

        self.selected_track_id: Optional[str] = self.project.tracks[0].id
        self._current_path: Optional[str] = None
        self._project_label = "Untitled"
        self._dirty = False
        self._editing_clip_id: Optional[str] = None
        self._workers: list = []  # keep transcription threads alive
        self._clip_clipboard: Optional[dict] = None
        self._note_clipboard: list = []
        self._syncing = False

        self._play_timer = QTimer(self)
        self._play_timer.setInterval(30)  # ~33 fps playhead updates
        self._play_timer.timeout.connect(self._on_tick)

        self._build_central()
        self._build_actions()
        self._connect()

        # Always-visible target-track indicator (survives status messages).
        self._target_label = QLabel()
        self._target_label.setStyleSheet("color:#8fb7e0; font-weight:600;")
        self.statusBar().addPermanentWidget(self._target_label)

        self._rebuild_all()
        self._refresh_history_actions()
        self.statusBar().showMessage("Ready — M3 (audio). File ▸ Load Demo Arrangement to test.")

        self._restore_last_project()

    # ---- construction ----------------------------------------------------
    def _build_central(self) -> None:
        central = QWidget()
        central.setObjectName("central")
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.transport = TransportBar()
        self.header_panel = TrackHeaderPanel()
        self.timeline = TimelineView()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.header_panel)
        splitter.addWidget(self.timeline)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(7)          # grabbable, matches the dock separators
        splitter.setSizes([240, 900])       # header starts at its old fixed width

        layout.addWidget(self.transport)
        layout.addWidget(splitter, stretch=1)
        self.setCentralWidget(central)

        self.timeline.set_project(self.project)
        self.timeline.set_audio_pool(self.pool)

        # One bottom editor dock with a Piano Roll / Synth mode switch.
        self.editor = EditorDock(self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.editor)
        self.editor.hide()
        self.piano = self.editor.piano          # PianoRollPanel (edit_clip/reload/view/…)
        self.synth_panel = self.editor.synth    # SynthPanel (set_track/param_changed)

        self.agent_panel = AgentPanel(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.agent_panel)

        self.search_panel = SearchPanel(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.search_panel)
        self.tabifyDockWidget(self.agent_panel, self.search_panel)
        self.agent_panel.raise_()

        # Lock all docks in place — no floating/popping-out or closing (still
        # movable between areas so the Agent/Search tabs work).
        for dock in (self.editor, self.agent_panel, self.search_panel):
            dock.setFeatures(QDockWidget.DockWidgetMovable)

    def _build_actions(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        self.act_new = QAction("&New Project", self, shortcut=QKeySequence.New)
        self.act_open = QAction("&Open…", self, shortcut=QKeySequence.Open)
        self.act_save = QAction("&Save", self, shortcut=QKeySequence.Save)
        self.act_save_as = QAction("Save &As…", self, shortcut=QKeySequence.SaveAs)
        self.act_import = QAction("&Import Audio…", self, shortcut="Ctrl+I")
        self.act_import_midi = QAction("Import &MIDI…", self, shortcut="Ctrl+Shift+I")
        self.act_demo = QAction("Load &Demo Arrangement", self)
        self.act_export = QAction("&Export Audio…", self, shortcut="Ctrl+Shift+E")
        file_menu.addActions([self.act_new, self.act_open, self.act_save, self.act_save_as])
        file_menu.addSeparator()
        file_menu.addActions([self.act_import, self.act_import_midi, self.act_demo])
        file_menu.addSeparator()
        file_menu.addAction(self.act_export)

        edit_menu = menubar.addMenu("&Edit")
        self.act_undo = QAction("Undo", self, shortcut=QKeySequence.Undo)
        self.act_redo = QAction("Redo", self, shortcut=QKeySequence.Redo)
        self.act_undo.setEnabled(False)
        self.act_redo.setEnabled(False)
        edit_menu.addActions([self.act_undo, self.act_redo])
        edit_menu.addSeparator()
        self.act_add_track = QAction("Add &Track", self, shortcut="Ctrl+T")
        self.act_add_clip = QAction("Add &Clip", self, shortcut="Ctrl+K")
        self.act_split = QAction("Split at &Playhead", self, shortcut="Ctrl+E")
        self.act_delete = QAction("&Delete", self, shortcut=QKeySequence.Delete)
        edit_menu.addActions(
            [self.act_add_track, self.act_add_clip, self.act_split, self.act_delete]
        )

        transport_menu = menubar.addMenu("&Transport")
        self.act_play = QAction("Play/Pause", self, shortcut=Qt.Key_Space)
        self.act_stop = QAction("Stop", self)
        transport_menu.addActions([self.act_play, self.act_stop])
        self.act_record = QAction("● Record", self, shortcut="Ctrl+R")
        self.act_record.setToolTip("Record the microphone into a clip on the selected track")
        self.act_record.triggered.connect(self._toggle_record)
        transport_menu.addAction(self.act_record)
        transport_menu.addSeparator()
        self.menu_audio_out = transport_menu.addMenu("Audio &Output")
        self._output_group = None
        self._populate_output_devices()
        # Re-scan the OS device list every time the menu opens (catches hotplugs).
        self.menu_audio_out.aboutToShow.connect(self._populate_output_devices)
        self.menu_audio_in = transport_menu.addMenu("Audio &Input")
        self._input_group = None
        self._populate_input_devices()
        self.menu_audio_in.aboutToShow.connect(self._populate_input_devices)

        agent_menu = menubar.addMenu("&Agent")
        self.act_set_key = QAction("Set API &Key…", self)
        self.act_set_key.triggered.connect(self._on_set_api_key)
        agent_menu.addAction(self.act_set_key)
        model_menu = agent_menu.addMenu("&Model")
        self._model_group = QActionGroup(self)
        self._model_group.setExclusive(True)
        for label, mid in [("Haiku — fastest, cheapest", "claude-haiku-4-5"),
                           ("Sonnet — balanced", "claude-sonnet-5"),
                           ("Opus — most capable, priciest", "claude-opus-5")]:
            act = QAction(label, self, checkable=True)
            act.setChecked(self.agent.model == mid)
            act.triggered.connect(lambda _=False, m=mid, ll=label: self._on_select_model(m, ll))
            self._model_group.addAction(act)
            model_menu.addAction(act)

        view_menu = menubar.addMenu("&View")
        self.act_zoom_in = QAction("Zoom &In", self, shortcut=QKeySequence.ZoomIn)
        self.act_zoom_out = QAction("Zoom &Out", self, shortcut=QKeySequence.ZoomOut)
        self.act_snap = QAction("&Snap to grid", self, checkable=True)
        self.act_snap.setChecked(True)
        view_menu.addActions([self.act_zoom_in, self.act_zoom_out, self.act_snap])

        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.addActions([self.act_undo, self.act_redo])
        toolbar.addSeparator()
        toolbar.addActions([self.act_add_track, self.act_add_clip, self.act_delete])
        toolbar.addSeparator()
        toolbar.addAction(self.act_record)
        toolbar.addAction(self.act_import)
        toolbar.addSeparator()
        toolbar.addActions([self.act_zoom_in, self.act_zoom_out, self.act_snap])

    def _connect(self) -> None:
        self.bus.add_listener(self._on_bus_change)

        self.act_open.triggered.connect(self._on_open)
        self.act_new.triggered.connect(self._on_new)
        self.act_save.triggered.connect(self._on_save)
        self.act_save_as.triggered.connect(self._on_save_as)
        self.act_import.triggered.connect(self._on_import)
        self.act_import_midi.triggered.connect(self._on_import_midi)
        self.act_demo.triggered.connect(self._on_load_demo)
        self.act_export.triggered.connect(self._on_export)
        self.act_split.triggered.connect(self._on_split_selected)
        self.act_undo.triggered.connect(self._undo)
        self.act_redo.triggered.connect(self._redo)
        self.act_add_track.triggered.connect(self._on_add_track)
        self.act_add_clip.triggered.connect(self._on_add_clip)
        self.act_delete.triggered.connect(self._on_delete)
        self.act_play.triggered.connect(self._toggle_play)
        self.act_stop.triggered.connect(self._on_stop)
        self.act_zoom_in.triggered.connect(self.timeline.zoom_in)
        self.act_zoom_out.triggered.connect(self.timeline.zoom_out)
        self.act_snap.toggled.connect(self._on_snap_toggled)

        self.transport.play_requested.connect(self._toggle_play)
        self.transport.stop_requested.connect(self._on_stop)
        self.transport.loop_toggled.connect(self._on_loop_toggled)
        self.transport.tempo_changed.connect(self._on_tempo_changed)

        self.header_panel.header_clicked.connect(self._on_track_selected)
        self.header_panel.renamed.connect(
            lambda tid, name: self._dispatch_attr(tid, "name", name)
        )
        self.header_panel.mute_toggled.connect(
            lambda tid, on: self._dispatch_attr(tid, "mute", on)
        )
        self.header_panel.solo_toggled.connect(
            lambda tid, on: self._dispatch_attr(tid, "solo", on)
        )
        self.header_panel.gain_changed.connect(
            lambda tid, v: self._dispatch_attr(tid, "gain_db", v)
        )
        self.header_panel.pan_changed.connect(
            lambda tid, v: self._dispatch_attr(tid, "pan", v)
        )
        self.header_panel.fx_action.connect(self._on_fx_action)

        self.timeline.clip_selected.connect(self._on_clip_selected)
        self.timeline.track_selected.connect(self._on_track_selected)
        self.timeline.clip_double_clicked.connect(self._open_piano_roll)
        self.timeline.delete_requested.connect(self._on_delete)
        self.timeline.copy_requested.connect(self._on_copy_clip)
        self.timeline.paste_requested.connect(self._on_paste_clip)
        self.piano.notes_changed.connect(self._on_notes_changed)
        self.piano.copy_requested.connect(self._on_pr_copy)
        self.piano.paste_requested.connect(self._on_pr_paste)
        self.piano.view.preview.connect(self._preview_pitch)
        self.synth_panel.param_changed.connect(self._on_synth_param)
        self.timeline.clip_geometry_edited.connect(self._on_clip_geometry)
        self.timeline.import_into_clip_requested.connect(self._on_import_into_clip)
        self.timeline.clip_action_requested.connect(self._on_clip_action)
        self.timeline.playhead_moved.connect(self.transport.set_time)

        self.timeline.verticalScrollBar().valueChanged.connect(self._sync_from_timeline)
        self.header_panel.verticalScrollBar().valueChanged.connect(self._sync_from_headers)

        self.agent_panel.send.connect(self._on_agent_send)
        self.agent_panel.cleared.connect(self.agent.reset)

        self.search_panel.search.connect(self._on_search)
        self.search_panel.similar.connect(self._on_find_similar)
        self.search_panel.ingest.connect(self._on_ingest_folder)
        self.search_panel.activated.connect(self._on_add_search_result)

    # ---- command bus feedback -------------------------------------------
    def _on_bus_change(self, cmd) -> None:
        self._refresh_history_actions()
        if cmd is not None:
            self.statusBar().showMessage(cmd.label)
            self._set_dirty(True)

    # ---- unsaved-changes tracking ---------------------------------------
    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        self._update_window_title()

    def _update_window_title(self) -> None:
        star = "•  " if self._dirty else ""
        self.setWindowTitle(f"Fantasia Conductor — {star}{self._project_label}")

    def _maybe_discard(self) -> bool:
        """Prompt if there are unsaved changes. Return True to proceed, False to cancel."""
        if not self._dirty:
            return True
        choice = QMessageBox.warning(
            self, "Unsaved changes",
            f"Save changes to “{self._project_label}” before continuing?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save)
        if choice == QMessageBox.Cancel:
            return False
        if choice == QMessageBox.Save:
            self._on_save()
            return not self._dirty  # if the save dialog was cancelled, abort too
        return True  # Discard

    def _refresh_history_actions(self) -> None:
        self.act_undo.setEnabled(self.bus.can_undo)
        self.act_redo.setEnabled(self.bus.can_redo)
        self.act_undo.setText(f"Undo {self.bus.undo_label}" if self.bus.can_undo else "Undo")
        self.act_redo.setText(f"Redo {self.bus.redo_label}" if self.bus.can_redo else "Redo")

    def _undo(self) -> None:
        self.bus.undo()
        self._after_history_change()

    def _redo(self) -> None:
        self.bus.redo()
        self._after_history_change()

    def _after_history_change(self) -> None:
        if self.selected_track_id is not None and (
            self.project.track_by_id(self.selected_track_id) is None
        ):
            self.selected_track_id = (
                self.project.tracks[0].id if self.project.tracks else None
            )
        self.pool.preload(self.project)
        self._warm()  # re-cache MIDI after undo/redo of note/convert edits
        self._rebuild_all()
        self._sync_tempo_display()  # undo/redo of a tempo change updates the display
        self._conform_locked_clips()  # reconform tempo-locked clips (no-op if unchanged)
        if self._editing_clip_id is not None:  # keep the piano roll in sync
            _, clip = self.project.find_clip(self._editing_clip_id)
            self.piano.reload(clip)

    # ---- transport / playback -------------------------------------------
    def _toggle_play(self) -> None:
        if self.engine.is_playing:
            self._on_stop()
        else:
            self._on_play()

    def _on_play(self) -> None:
        self.engine.set_playhead_seconds(self.timeline.playhead)
        if self.engine.play():
            self._play_timer.start()
            self.statusBar().showMessage("Playing")
        else:
            self.statusBar().showMessage("No audio output device available")

    def _on_stop(self) -> None:
        self.engine.stop()
        self._play_timer.stop()
        self.statusBar().showMessage("Stopped")

    def _on_tick(self) -> None:
        self.timeline.set_playhead(self.engine.playhead)
        if not self.engine.is_playing:  # reached the end on its own
            self._play_timer.stop()
            self.statusBar().showMessage("Stopped")

    def _on_loop_toggled(self, on: bool) -> None:
        self.engine.loop = on

    def _on_tempo_changed(self, bpm: float) -> None:
        self.bus.dispatch(SetTempoCommand(bpm))  # undoable; slider drags coalesce
        self.timeline.viewport().update()  # grid depends on tempo
        self._tempo_conform_timer.start()  # re-stretch tempo-locked clips once settled

    # ---- tempo-lock (#25) -----------------------------------------------
    def _toggle_tempo_lock(self, clip) -> None:
        if clip.lock_tempo is not None:  # unlock (keep current audio)
            clip.lock_tempo = None
            clip.orig_source_path = None
            clip.lock_base_dur = 0.0
            self._set_dirty(True)
            self.statusBar().showMessage(f"Unlocked '{clip.name}' from tempo")
            return
        from fantasia_core import stretch as st

        if not st.available():
            self.statusBar().showMessage(
                "Tempo-lock needs Rubber Band (brew install rubberband)")
            return
        if not clip.source_path:
            self.statusBar().showMessage("Tempo-lock needs an audio clip")
            return
        try:
            data = self.pool.load(clip.source_path)
        except Exception:  # noqa: BLE001
            self.statusBar().showMessage("Couldn't read the clip audio")
            return
        sr = self.project.sample_rate
        s = int(clip.source_offset * sr)
        e = int((clip.source_offset + clip.duration) * sr)
        seg = data[s:e]
        if len(seg) == 0:
            return
        import soundfile as sf

        cache = _REPO_ROOT / ".fantasia_cache" / "lock"
        cache.mkdir(parents=True, exist_ok=True)
        orig = str(cache / f"orig_{uuid.uuid4().hex[:8]}.wav")
        sf.write(orig, seg, sr)
        clip.orig_source_path = orig
        clip.lock_tempo = self.project.tempo
        clip.lock_base_dur = clip.duration
        self._set_dirty(True)
        self.statusBar().showMessage(
            f"Locked '{clip.name}' to {self.project.tempo:.0f} BPM — it now follows tempo")

    def _conform_locked_clips(self) -> None:
        """Re-stretch every tempo-locked clip from its original so it matches the
        current project tempo. Derived from (orig, tempo), so undo/redo reconforms."""
        from fantasia_core import stretch as st

        if not st.available() or self.project.tempo <= 0:
            return
        import soundfile as sf

        cache = _REPO_ROOT / ".fantasia_cache" / "lock"
        cache.mkdir(parents=True, exist_ok=True)
        changed = False
        for track in self.project.tracks:
            for clip in track.clips:
                if clip.lock_tempo is None or not clip.orig_source_path or clip.lock_base_dur <= 0:
                    continue
                factor = clip.lock_tempo / self.project.tempo
                target = clip.lock_base_dur * factor
                if abs(clip.duration - target) < 0.005:
                    continue  # already conformed
                try:
                    orig = self.pool.load(clip.orig_source_path)
                    out = st.stretch(orig, self.project.sample_rate, factor)
                    wav = str(cache / f"lk_{uuid.uuid4().hex[:8]}.wav")
                    sf.write(wav, out, self.project.sample_rate)
                    clip.source_path = wav
                    clip.source_offset = 0.0
                    clip.duration = len(out) / self.project.sample_rate
                    changed = True
                except Exception:  # noqa: BLE001
                    pass
        if changed:
            self.pool.preload(self.project)
            self.timeline.rebuild()

    def _sync_tempo_display(self) -> None:
        """Reflect the model's tempo in the transport + grid (after load / undo /
        agent edit) without re-triggering a tempo change."""
        self.transport.set_tempo(self.project.tempo)
        self.timeline.viewport().update()

    # ---- audio output device --------------------------------------------
    def _populate_output_devices(self) -> None:
        from fantasia_core.engine.playback import default_output_device, list_output_devices

        self.menu_audio_out.clear()
        self._output_group = QActionGroup(self)
        self._output_group.setExclusive(True)

        # Re-initialising PortAudio with a live stream corrupts its state and
        # renumbers devices (that's what made a freshly-picked output silently
        # fail). Stop no longer closes the device — closing blocks the UI — so
        # release it here instead: opening this menu is rare and deliberate.
        if not self.engine.is_playing:
            self.engine.release_device()
        devices = list_output_devices(refresh=not self.engine.has_stream)
        valid = {i for i, _ in devices}
        # Drop a stale selection (e.g. headphones that were unplugged mid-session).
        if self.engine.output_device is not None and self.engine.output_device not in valid:
            self.engine.output_device = None
        current = self.engine.output_device
        if current is None:
            current = default_output_device()
        if not devices:
            act = self.menu_audio_out.addAction("No output devices found")
            act.setEnabled(False)
        for idx, name in devices:
            act = QAction(name, self, checkable=True)
            act.setChecked(idx == current)
            act.triggered.connect(lambda _=False, i=idx, n=name: self._on_select_output(i, n))
            self._output_group.addAction(act)
            self.menu_audio_out.addAction(act)
        self.menu_audio_out.addSeparator()
        refresh = self.menu_audio_out.addAction("Refresh Devices")
        refresh.triggered.connect(self._populate_output_devices)

    def _on_select_output(self, index: int, name: str) -> None:
        if self.engine.set_output_device(index):
            self.statusBar().showMessage(f"Audio output → {name}")
        else:
            # The device was verified at selection time, so this is a real
            # failure (in use elsewhere / disconnected), not a late surprise.
            self.statusBar().showMessage(
                f"Couldn't open {name} — using the system default instead")
            self._populate_output_devices()  # re-sync the checkmark

    # ---- audio input device + recording ---------------------------------
    def _populate_input_devices(self) -> None:
        self.menu_audio_in.clear()
        self._input_group = QActionGroup(self)
        self._input_group.setExclusive(True)
        devices = list_input_devices(refresh=not self.recorder.is_recording)
        valid = {i for i, _ in devices}
        if self.record_input_device is not None and self.record_input_device not in valid:
            self.record_input_device = None
        if not devices:
            act = self.menu_audio_in.addAction("No input devices found")
            act.setEnabled(False)
        for idx, name in devices:
            act = QAction(name, self, checkable=True)
            act.setChecked(idx == self.record_input_device)
            act.triggered.connect(lambda _=False, i=idx, n=name: self._on_select_input(i, n))
            self._input_group.addAction(act)
            self.menu_audio_in.addAction(act)
        self.menu_audio_in.addSeparator()
        self.menu_audio_in.addAction("Refresh Devices").triggered.connect(
            self._populate_input_devices)

    def _on_select_input(self, index: int, name: str) -> None:
        self.record_input_device = index
        self.statusBar().showMessage(f"Mic input → {name}")

    def _toggle_record(self) -> None:
        if self.recorder.is_recording:
            self._stop_record()
        else:
            self._start_record()

    def _start_record(self) -> None:
        if self.selected_track_id is None:
            if not self.project.tracks:
                self.selected_track_id = self.bus.dispatch(AddTrackCommand()).created_track.id
            else:
                self.selected_track_id = self.project.tracks[0].id
        self.recorder.input_device = self.record_input_device
        self.recorder.sr = self.project.sample_rate
        if not self.recorder.start():
            self.statusBar().showMessage(
                "Couldn't open the microphone — check the mic and macOS privacy permission.")
            return
        self.act_record.setText("■ Stop")
        self._rec_timer.start()
        self.statusBar().showMessage("● Recording…  (Ctrl+R to stop)")

    def _stop_record(self) -> None:
        self._rec_timer.stop()
        self.act_record.setText("● Record")
        audio = self.recorder.stop()
        if len(audio) == 0:
            self.statusBar().showMessage("Nothing recorded")
            return
        import soundfile as sf

        cache = _REPO_ROOT / ".fantasia_cache" / "recordings"
        cache.mkdir(parents=True, exist_ok=True)
        path = str(cache / f"rec_{uuid.uuid4().hex[:8]}.wav")
        sf.write(path, audio, self.project.sample_rate, subtype="PCM_16")
        dur = len(audio) / self.project.sample_rate
        start = self.timeline.playhead
        self.bus.dispatch(AddClipCommand(self.selected_track_id, start, dur,
                                         name="Recording", source_path=path))
        self.pool.preload(self.project)
        self.timeline.rebuild()
        self._ingest_to_library([{"path": path, "name": "Recording",
                                  "tags": ["recording", "mic", "take", "vocal", "audio"]}])
        if self.recorder.had_dropouts:
            lost = self.recorder.dropped_frames / max(self.project.sample_rate, 1)
            QMessageBox.warning(
                self, "Choppy recording",
                f"The microphone dropped {self.recorder.overflows} buffer(s) "
                f"(~{lost:.2f}s of audio), so this take will sound choppy.\n\n"
                "This happens when the machine is busy. Close heavy work "
                "(generation/rendering) and record again for a clean take.")
            self.statusBar().showMessage(
                f"Recorded {dur:.1f}s — WARNING: {self.recorder.overflows} dropout(s)")
        else:
            self.statusBar().showMessage(f"Recorded {dur:.1f}s onto the selected track")

    def _on_rec_tick(self) -> None:
        self.statusBar().showMessage(f"● Recording…  {self.recorder.elapsed_seconds:5.1f}s  (Ctrl+R to stop)")

    # ---- agent API key ---------------------------------------------------
    def _on_select_model(self, model_id: str, label: str) -> None:
        self.agent.model = model_id
        self.statusBar().showMessage(f"Agent model → {label}")
        self.agent_panel.append("system", f"Model set to {model_id}.")

    def _on_set_api_key(self) -> None:
        key, ok = QInputDialog.getText(
            self, "Set Anthropic API Key",
            "Paste your Anthropic API key.\nStored locally (plaintext) in "
            ".fantasia_cache/secrets.env and loaded on every launch:",
            QLineEdit.Password,
        )
        if not ok or not key.strip():
            return
        key = key.strip()
        _save_secret("ANTHROPIC_API_KEY", key)
        os.environ["ANTHROPIC_API_KEY"] = key
        self.agent._api_key = key
        self.agent._client = None  # force re-init with the new key
        self.agent_panel.append("system", "API key saved — the agent is ready.")
        self.statusBar().showMessage("Anthropic API key saved to .fantasia_cache/secrets.env")

    # ---- scroll sync -----------------------------------------------------
    def _sync_from_timeline(self, value: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        self.header_panel.verticalScrollBar().setValue(value)
        self._syncing = False

    def _sync_from_headers(self, value: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        self.timeline.verticalScrollBar().setValue(value)
        self._syncing = False

    # ---- rebuild ---------------------------------------------------------
    def _warm(self, project=None) -> None:
        """Re-render MIDI (FluidSynth) and synth-track buffers off the audio thread."""
        p = project if project is not None else self.project
        self.midi.warm(p)
        self.synth_engine.warm(p)

    def _rebuild_all(self) -> None:
        self.header_panel.rebuild(self.project)
        self.timeline.rebuild()
        self._apply_selection_highlight()

    # ---- editing actions (all via the bus) -------------------------------
    def _on_add_track(self) -> None:
        cmd = self.bus.dispatch(AddTrackCommand())
        self.selected_track_id = cmd.created_track.id
        self._rebuild_all()

    def _on_add_clip(self) -> None:
        if self.selected_track_id is None:
            if not self.project.tracks:
                return
            self.selected_track_id = self.project.tracks[0].id
        start = self.timeline.playhead
        duration = self.project.seconds_per_beat() * self.project.beats_per_bar
        self.bus.dispatch(AddClipCommand(self.selected_track_id, start, duration, name="Clip"))
        self.timeline.rebuild()

    def _on_delete(self) -> None:
        clip_id = self.timeline.selected_clip_id()
        if clip_id is not None:
            self.bus.dispatch(RemoveClipCommand(clip_id))
            self.timeline.rebuild()
            return
        if self.selected_track_id is not None and self.project.tracks:
            self.bus.dispatch(RemoveTrackCommand(self.selected_track_id))
            self.selected_track_id = (
                self.project.tracks[0].id if self.project.tracks else None
            )
            self._rebuild_all()

    def _on_snap_toggled(self, on: bool) -> None:
        self.timeline.snap_enabled = on

    def _dispatch_attr(self, track_id: str, attr: str, value) -> None:
        self.bus.dispatch(
            SetTrackAttrCommand(track_id, attr, value, mergeable=attr in _CONTINUOUS_ATTRS)
        )
        if attr == "name" and track_id == self.selected_track_id:
            self._update_target_label()

    def _on_clip_geometry(self, clip_id: str, start: float, duration: float) -> None:
        self.bus.dispatch(SetClipGeometryCommand(clip_id, start, duration))
        self.timeline.refresh_clip(clip_id)

    def _on_import_into_clip(self, clip_id: str) -> None:
        """Fill (or replace) a clip's audio content from a file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Audio into Clip", "", _AUDIO_FILTER
        )
        if not path:
            return
        try:
            dur = self.pool.duration(path)
        except Exception:  # noqa: BLE001
            self.statusBar().showMessage(f"Couldn't read {os.path.basename(path)}")
            return
        self.bus.dispatch(SetClipSourceCommand(clip_id, path, 0.0, dur))
        self.pool.preload(self.project)
        self.timeline.refresh_clip(clip_id)
        self.timeline.viewport().update()
        self._ingest_to_library([{"path": path, "name": os.path.basename(path),
                                  "tags": _file_tags(path, ["imported", "sample"])}])
        self.statusBar().showMessage(f"Imported {os.path.basename(path)} into clip")

    # ---- MIDI / piano roll (MID-3) --------------------------------------
    def _open_piano_roll(self, clip_id: str) -> None:
        track, clip = self.project.find_clip(clip_id)
        if clip is None or not clip.is_midi:
            return  # only MIDI clips have a piano roll
        self._editing_clip_id = clip_id
        bar_len = self.project.beats_per_bar * self.project.seconds_per_beat()
        clip_bar = int(round(clip.start / bar_len)) + 1 if bar_len > 0 else 1
        self.piano.edit_clip(
            clip, self.project.seconds_per_beat(), self.project.beats_per_bar,
            drum_mode=getattr(track, "is_drum", False), clip_bar=clip_bar,
        )
        self._update_piano_waveform(clip_id)
        self.editor.show_piano_roll()
        self.statusBar().showMessage(f"Editing notes in {clip.name}")

    def _update_piano_waveform(self, clip_id: str) -> None:
        """Render the clip's audio so the roll can draw it behind the notes."""
        track, clip = self.project.find_clip(clip_id)
        if clip is None or not clip.is_midi:
            self.piano.view.set_waveform(None, self.project.sample_rate)
            return
        try:
            if getattr(track, "is_synth", False):
                buf = self.synth_engine.render(clip, getattr(track, "synth", None) or {})
            else:
                buf = self.midi.render(clip, track.instrument,
                                       getattr(track, "is_drum", False))
            self.piano.view.set_waveform(buf, self.project.sample_rate)
        except Exception:  # noqa: BLE001
            self.piano.view.set_waveform(None, self.project.sample_rate)

    def _on_notes_changed(self, clip_id: str, notes: list) -> None:
        self.bus.dispatch(SetClipNotesCommand(clip_id, notes))
        self._warm()  # re-render this clip via whichever engine the track uses
        _, clip = self.project.find_clip(clip_id)
        self.timeline.viewport().update()  # refresh the timeline note preview
        self.piano.refresh_title(clip)
        self._update_piano_waveform(clip_id)  # keep the overlay in sync with edits

    def _preview_pitch(self, pitch: int) -> None:
        """Audition a note on the editing track's instrument (one-shot playback)."""
        if self._editing_clip_id is None:
            return
        track, _ = self.project.find_clip(self._editing_clip_id)
        if track is None:
            return
        try:
            import sounddevice as sd

            from fantasia_core.document.model import Clip

            sr = self.project.sample_rate
            note = Note(int(pitch), 0.0, 0.3, 112)
            clip = Clip(id="_preview", name="_", start=0.0, duration=0.4,
                        content_type="midi", notes=[note])
            # Render on this (UI) thread — FluidSynth is not thread-safe — but
            # hand playback to a worker: sd.play() opens a device and blocks the
            # caller ~125ms, which freezes the UI on every pitch change of a drag.
            if getattr(track, "is_synth", False):
                buf = self.synth_engine.render(clip, getattr(track, "synth", {}) or {})
            else:
                buf = self.midi.render(clip, track.instrument, getattr(track, "is_drum", False))
            device = self.engine.output_device

            def _play(buf=buf, sr=sr, device=device) -> None:
                # Drop the preview if one is still starting — stale auditions
                # are worse than a missed one, and it keeps sd's global stream
                # from being reopened concurrently.
                if not self._preview_lock.acquire(blocking=False):
                    return
                try:
                    sd.play(buf, sr, device=device)
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    self._preview_lock.release()

            threading.Thread(target=_play, daemon=True, name="note-preview").start()
        except Exception:  # noqa: BLE001
            pass

    # ---- copy / paste ---------------------------------------------------
    def _on_copy_clip(self) -> None:
        cid = self.timeline.selected_clip_id()
        if cid is None:
            self.statusBar().showMessage("Select a clip to copy")
            return
        _, clip = self.project.find_clip(cid)
        if clip is None:
            return
        self._clip_clipboard = {
            "name": clip.name, "duration": clip.duration, "content_type": clip.content_type,
            "source_path": clip.source_path, "source_offset": clip.source_offset,
            "notes": [Note(n.pitch, n.start, n.duration, n.velocity) for n in clip.notes],
            "gain_db": clip.gain_db, "fade_in": clip.fade_in, "fade_out": clip.fade_out,
            "reversed": clip.reversed, "pitch_semitones": clip.pitch_semitones,
        }
        self.statusBar().showMessage(f"Copied clip '{clip.name}'")

    def _on_paste_clip(self) -> None:
        cb = self._clip_clipboard
        if not cb:
            self.statusBar().showMessage("Nothing to paste")
            return
        if self.selected_track_id is None:
            if not self.project.tracks:
                return
            self.selected_track_id = self.project.tracks[0].id
        start = self.timeline.playhead
        notes = [Note(n.pitch, n.start, n.duration, n.velocity) for n in cb["notes"]]
        self.bus.dispatch(AddClipCommand(
            self.selected_track_id, start, cb["duration"], name=cb["name"],
            content_type=cb["content_type"], source_path=cb["source_path"],
            source_offset=cb["source_offset"], notes=notes, gain_db=cb["gain_db"],
            fade_in=cb["fade_in"], fade_out=cb["fade_out"], reversed=cb["reversed"],
            pitch_semitones=cb["pitch_semitones"],
        ))
        self.pool.preload(self.project)
        self._warm()
        self.timeline.rebuild()
        self.statusBar().showMessage(f"Pasted clip at {start:.2f}s")

    def _on_pr_copy(self) -> None:
        notes = self.piano.view.selected_notes()
        if not notes:
            self.statusBar().showMessage("Select notes to copy")
            return
        self._note_clipboard = notes
        self.statusBar().showMessage(f"Copied {len(notes)} note(s)")

    def _on_pr_paste(self) -> None:
        if not self._note_clipboard or self._editing_clip_id is None:
            return
        _, clip = self.project.find_clip(self._editing_clip_id)
        if clip is None:
            return
        anchor = max(0.0, min(self.timeline.playhead - clip.start, clip.duration))
        self.piano.view.paste_notes(anchor, self._note_clipboard)
        self.statusBar().showMessage(f"Pasted {len(self._note_clipboard)} note(s)")

    # ---- audio → MIDI transcription -------------------------------------
    def _transcribe_clip(self, clip) -> None:
        if not clip.source_path:
            self.statusBar().showMessage("Transcription needs an audio clip")
            return
        try:
            data = self.pool.load(clip.source_path)
        except Exception:  # noqa: BLE001
            self.statusBar().showMessage("Couldn't read the clip audio")
            return
        sr = self.project.sample_rate
        s = int(clip.source_offset * sr)
        e = int((clip.source_offset + clip.duration) * sr)
        seg = data[s:e]
        if len(seg) == 0:
            return
        self.statusBar().showMessage("Transcribing to MIDI… (first run loads the model)")
        worker = _TranscribeWorker(clip.id, seg.copy(), sr)
        worker.done.connect(self._on_transcribed)
        worker.failed.connect(self._on_transcribe_failed)
        worker.finished.connect(lambda w=worker: self._drop_worker(w))
        self._workers.append(worker)
        worker.start()

    def _drop_worker(self, worker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)

    def _hum_to_melody(self, clip) -> None:
        """Turn a hummed/sung recording into a monophonic MIDI melody."""
        from fantasia_core import hum as hum_mod

        if not hum_mod.available():
            self.statusBar().showMessage("Hum → Melody needs librosa")
            return
        if not clip.source_path:
            self.statusBar().showMessage("Hum → Melody needs an audio clip")
            return
        dlg = _HumDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        quantize, key, scale = dlg.values()
        try:
            data = self.pool.load(clip.source_path)
        except Exception:  # noqa: BLE001
            self.statusBar().showMessage("Couldn't read the clip audio")
            return
        sr = self.project.sample_rate
        s = int(clip.source_offset * sr)
        e = int((clip.source_offset + clip.duration) * sr)
        seg = data[s:e]
        if len(seg) == 0:
            return
        self.statusBar().showMessage("Tracking the hum…")
        worker = _HumWorker(clip.id, seg.copy(), sr, self.project.seconds_per_beat(),
                            self.project.beats_per_bar, quantize, key, scale)
        worker.done.connect(self._on_transcribed)   # same clip-fill path
        worker.failed.connect(self._on_transcribe_failed)
        worker.finished.connect(lambda w=worker: self._drop_worker(w))
        self._workers.append(worker)
        worker.start()

    def _on_transcribed(self, clip_id: str, notes: list) -> None:
        track, clip = self.project.find_clip(clip_id)
        if clip is None:
            return
        if not notes:
            self.statusBar().showMessage("No notes detected")
            return
        self.bus.dispatch(MakeMidiClipCommand(clip_id, notes))  # audio clip → MIDI (undoable)
        self._warm()
        self.timeline.rebuild()
        if track is not None:  # select it so picking an instrument is one right-click away
            self._set_selected_track(track.id)
        self.statusBar().showMessage(
            f"Transcribed {len(notes)} notes → MIDI. Right-click the header → Instrument to pick a sound."
        )

    def _on_transcribe_failed(self, clip_id: str, err: str) -> None:
        self.statusBar().showMessage(f"Transcription failed: {err}")

    # ---- text → audio generation (SND-2) --------------------------------
    def _generate_into_clip(self, clip) -> None:
        from fantasia_core import generate as gen

        if not gen.available():
            self.statusBar().showMessage(
                "Audio generation needs torch+transformers (pip install -e '.[generate]')")
            return
        dlg = _GenerateDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        prompt, seconds, guidance = dlg.result_values()
        if not prompt:
            return
        cache = _REPO_ROOT / ".fantasia_cache" / "generated"
        cache.mkdir(parents=True, exist_ok=True)
        path = str(cache / f"gen_{uuid.uuid4().hex[:8]}.wav")
        mode = "draft" if guidance < 2.0 else "best"
        self.statusBar().showMessage(
            f"Generating {seconds:.0f}s of audio on CPU ({mode})… (first run downloads the model)")
        worker = _GenerateWorker(clip.id, prompt, seconds, path,
                                 search=self.search_service, guidance=guidance)
        worker.done.connect(self._on_generated)
        worker.failed.connect(self._on_generate_failed)
        worker.finished.connect(lambda w=worker: self._drop_worker(w))
        self._workers.append(worker)
        worker.start()

    def _on_generated(self, clip_id: str, path: str, duration: float) -> None:
        self.bus.dispatch(SetClipSourceCommand(clip_id, path, 0.0, duration))
        self.pool.preload(self.project)
        self.timeline.refresh_clip(clip_id)
        self.timeline.viewport().update()
        self.statusBar().showMessage(
            f"Generated {duration:.1f}s of audio into clip (saved to sound library)")

    def _on_generate_failed(self, clip_id: str, err: str) -> None:
        self.statusBar().showMessage(f"Generation failed: {err}")

    # ---- stem separation (Demucs) ---------------------------------------
    _STEM_COLORS = theme.STEM_COLORS

    def _separate_clip(self, clip) -> None:
        from fantasia_core import separate

        if not separate.available():
            self.statusBar().showMessage(
                "Stem separation needs Demucs — run: pip install demucs")
            return
        if not clip.source_path:
            self.statusBar().showMessage("Separation needs an audio clip")
            return
        try:
            data = self.pool.load(clip.source_path)
        except Exception:  # noqa: BLE001
            self.statusBar().showMessage("Couldn't read the clip audio")
            return
        sr = self.project.sample_rate
        s = int(clip.source_offset * sr)
        e = int((clip.source_offset + clip.duration) * sr)
        seg = data[s:e]
        if len(seg) == 0:
            return
        import soundfile as sf

        cache = _REPO_ROOT / ".fantasia_cache" / "stems"
        cache.mkdir(parents=True, exist_ok=True)
        src_tmp = str(cache / f"src_{uuid.uuid4().hex[:6]}.wav")
        sf.write(src_tmp, seg, sr)
        self.statusBar().showMessage(
            "Separating stems on CPU… (first run downloads the Demucs model)")
        worker = _SeparateWorker(clip.id, src_tmp, str(cache), clip.name,
                                 search=self.search_service)
        worker.done.connect(self._on_separated)
        worker.failed.connect(self._on_separate_failed)
        worker.finished.connect(lambda w=worker: self._drop_worker(w))
        self._workers.append(worker)
        worker.start()

    def _on_separated(self, clip_id: str, stems: list) -> None:
        track, clip = self.project.find_clip(clip_id)
        start = clip.start if clip is not None else self.timeline.playhead
        base = clip.name if clip is not None else "clip"
        made = 0
        for name, path, dur in stems:
            tcmd = self.bus.dispatch(AddTrackCommand(f"{base} · {name}"))
            new_track = tcmd.created_track
            new_track.color = self._STEM_COLORS.get(name, "#4a90d9")
            self.bus.dispatch(AddClipCommand(new_track.id, start, dur, name=name, source_path=path))
            made += 1
        self.pool.preload(self.project)
        self._rebuild_all()
        self.statusBar().showMessage(f"Separated '{base}' into {made} stem tracks (added to library)")

    def _on_separate_failed(self, clip_id: str, err: str) -> None:
        self.statusBar().showMessage(f"Separation failed: {err}")

    # ---- text to speech (Kokoro / MLX) ----------------------------------
    def _tts_into_clip(self, clip) -> None:
        from fantasia_core import tts

        if not tts.available():
            self.statusBar().showMessage(
                "Text-to-speech needs mlx-audio + misaki (pip install mlx-audio 'misaki[en]')")
            return
        dlg = _TTSDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        text, voice, speed = dlg.result_values()
        if not text:
            return
        cache = _REPO_ROOT / ".fantasia_cache" / "voice"
        cache.mkdir(parents=True, exist_ok=True)
        path = str(cache / f"tts_{uuid.uuid4().hex[:8]}.wav")
        self.statusBar().showMessage("Synthesizing speech on the GPU… (first use loads the model)")
        worker = _TTSWorker(clip.id, text, voice, speed, path, search=self.search_service)
        worker.done.connect(self._on_generated)  # same fill-clip path as generation
        worker.failed.connect(self._on_generate_failed)
        worker.finished.connect(lambda w=worker: self._drop_worker(w))
        self._workers.append(worker)
        worker.start()

    # ---- singing synthesis (melody + lyrics) ----------------------------
    def _sing_clip(self, clip) -> None:
        from fantasia_core import sing

        if not sing.available():
            self.statusBar().showMessage(
                "Singing needs the voice extra (pip install mlx-audio 'misaki[en]') + librosa")
            return
        if not clip.is_midi or not clip.notes:
            self.statusBar().showMessage("Sing needs a MIDI clip with notes (draw a melody first)")
            return
        dlg = _SingDialog(self, note_count=len(clip.notes))
        if dlg.exec() != QDialog.Accepted:
            return
        lyrics, voice = dlg.result_values()
        if not lyrics:
            return
        notes = [Note(n.pitch, n.start, n.duration, n.velocity) for n in clip.notes]
        cache = _REPO_ROOT / ".fantasia_cache" / "voice"
        cache.mkdir(parents=True, exist_ok=True)
        path = str(cache / f"sing_{uuid.uuid4().hex[:8]}.wav")
        self.statusBar().showMessage("Singing the melody on the GPU… (a few seconds per note)")
        worker = _SingWorker(clip.id, notes, lyrics, voice, path, search=self.search_service)
        worker.done.connect(self._on_sung)
        worker.failed.connect(self._on_generate_failed)
        worker.finished.connect(lambda w=worker: self._drop_worker(w))
        self._workers.append(worker)
        worker.start()

    # ---- time stretch (Rubber Band) -------------------------------------
    def _stretch_clip_action(self, clip, which: str) -> None:
        from fantasia_core import stretch as st

        if not st.available():
            self.statusBar().showMessage(
                "Time-stretch needs Rubber Band (brew install rubberband) + pyrubberband")
            return
        if not clip.source_path:
            self.statusBar().showMessage("Time-stretch needs an audio clip")
            return
        if which == "2":
            factor = 2.0
        elif which == "0.5":
            factor = 0.5
        elif which == "custom":
            val, ok = QInputDialog.getDouble(
                self, "Time Stretch",
                "Duration multiplier  (2 = half speed, 0.5 = double speed):",
                1.0, 0.1, 10.0, 2)
            if not ok:
                return
            factor = val
        elif which == "bars":
            spb = self.project.seconds_per_beat()
            bar = self.project.beats_per_bar * spb
            cur = max(1, round(clip.duration / bar)) if bar > 0 else 1
            bars, ok = QInputDialog.getInt(
                self, "Fit to Bars", "Stretch the clip to span how many bars?",
                cur, 1, 256)
            if not ok or clip.duration <= 0:
                return
            factor = (bars * bar) / clip.duration
        else:
            return
        try:
            data = self.pool.load(clip.source_path)
        except Exception:  # noqa: BLE001
            self.statusBar().showMessage("Couldn't read the clip audio")
            return
        sr = self.project.sample_rate
        s = int(clip.source_offset * sr)
        e = int((clip.source_offset + clip.duration) * sr)
        seg = data[s:e]
        if len(seg) == 0:
            return
        cache = _REPO_ROOT / ".fantasia_cache" / "stretch"
        cache.mkdir(parents=True, exist_ok=True)
        out = str(cache / f"str_{uuid.uuid4().hex[:8]}.wav")
        self.statusBar().showMessage(f"Time-stretching ×{factor:.2f} (pitch preserved)…")
        worker = _StretchWorker(clip.id, seg.copy(), sr, factor, out)
        worker.done.connect(self._on_stretched)
        worker.failed.connect(self._on_generate_failed)
        worker.finished.connect(lambda w=worker: self._drop_worker(w))
        self._workers.append(worker)
        worker.start()

    def _on_stretched(self, clip_id: str, path: str, duration: float) -> None:
        self.bus.dispatch(SetClipSourceCommand(clip_id, path, 0.0, duration))
        self.pool.preload(self.project)
        self.timeline.refresh_clip(clip_id)
        self.timeline.viewport().update()
        self.statusBar().showMessage(f"Stretched clip to {duration:.2f}s (pitch unchanged)")

    # ---- vocal FX (WORLD) -----------------------------------------------
    def _apply_vocalfx(self, clip, op: str) -> None:
        from fantasia_core import vocalfx as vf

        if not vf.available():
            self.statusBar().showMessage("Vocal FX need pyworld (pip install pyworld)")
            return
        if not clip.source_path:
            self.statusBar().showMessage("Vocal FX need an audio clip")
            return
        params = {}
        if op == "autotune":
            dlg = _AutotuneDialog(self)
            if dlg.exec() != QDialog.Accepted:
                return
            params = dlg.result_values()
        try:
            data = self.pool.load(clip.source_path)
        except Exception:  # noqa: BLE001
            self.statusBar().showMessage("Couldn't read the clip audio")
            return
        sr = self.project.sample_rate
        s = int(clip.source_offset * sr)
        e = int((clip.source_offset + clip.duration) * sr)
        seg = data[s:e]
        if len(seg) == 0:
            return
        cache = _REPO_ROOT / ".fantasia_cache" / "voice"
        cache.mkdir(parents=True, exist_ok=True)
        out = str(cache / f"vfx_{uuid.uuid4().hex[:8]}.wav")
        self.statusBar().showMessage(f"Processing vocal FX ({op})…")
        worker = _VocalFxWorker(clip.id, op, params, seg.copy(), sr, out)
        worker.done.connect(self._on_vocalfx_done)
        worker.failed.connect(self._on_generate_failed)
        worker.finished.connect(lambda w=worker: self._drop_worker(w))
        self._workers.append(worker)
        worker.start()

    def _on_vocalfx_done(self, clip_id: str, op: str, path: str, duration: float) -> None:
        if op in ("harmony3", "harmony5"):
            track, clip = self.project.find_clip(clip_id)
            start = clip.start if clip is not None else self.timeline.playhead
            base = clip.name if clip is not None else "vocal"
            interval = "3rd" if op == "harmony3" else "5th"
            tcmd = self.bus.dispatch(AddTrackCommand(f"Harmony {interval} · {base}"))
            tcmd.created_track.color = theme.CYAN
            self.bus.dispatch(AddClipCommand(tcmd.created_track.id, start, duration,
                                             name=f"Harmony {interval}", source_path=path))
            self.pool.preload(self.project)
            self._rebuild_all()
            self.statusBar().showMessage(f"Added a {interval} harmony on a new track")
        else:
            self.bus.dispatch(SetClipSourceCommand(clip_id, path, 0.0, duration))
            self.pool.preload(self.project)
            self.timeline.refresh_clip(clip_id)
            self.timeline.viewport().update()
            self.statusBar().showMessage(f"Applied {op} to the clip")

    def _on_sung(self, clip_id: str, path: str, duration: float) -> None:
        # Place the vocal on its own new track, aligned with the source melody.
        track, clip = self.project.find_clip(clip_id)
        start = clip.start if clip is not None else self.timeline.playhead
        base = clip.name if clip is not None else "melody"
        tcmd = self.bus.dispatch(AddTrackCommand(f"Vocal · {base}"))
        tcmd.created_track.color = theme.PINK
        self.bus.dispatch(AddClipCommand(tcmd.created_track.id, start, duration,
                                         name="Vocal", source_path=path))
        self.pool.preload(self.project)
        self._rebuild_all()
        self.statusBar().showMessage(f"Sang {duration:.1f}s onto a new vocal track (saved to library)")

    # ---- agent (M6) -----------------------------------------------------
    def _agent_refresh(self) -> None:
        """Called after each agent edit — re-render and rebuild the UI."""
        self.pool.preload(self.project)
        self._warm()
        self._rebuild_all()
        self._sync_tempo_display()  # reflect any agent tempo change in the transport

    def _on_agent_send(self, message: str) -> None:
        if self._agent_busy:
            return
        if not self.agent.available():
            if not AgentSession.anthropic_available():
                msg = "Agent needs the anthropic SDK — run: pip install -e '.[agent]'."
            else:
                msg = ("No API key set. Use Agent ▸ Set API Key… to paste your Anthropic "
                       "API key (saved locally, loaded on every launch).")
            self.agent_panel.append("system", msg)
            return
        self._agent_busy = True
        self.agent_panel.set_busy(True)
        worker = _AgentWorker(self.agent, message, search=self.search_service, seed=self._seed_folder)
        worker.text.connect(lambda s: self.agent_panel.append("agent", s))
        worker.note.connect(lambda s: self.agent_panel.append("system", s))
        worker.usage.connect(self.agent_panel.update_usage)
        worker.tool.connect(self._on_agent_tool)  # queued → runs on the UI thread
        worker.done.connect(self._on_agent_done)
        worker.failed.connect(self._on_agent_failed)
        self._agent_worker = worker
        worker.start()

    def _on_agent_tool(self, req) -> None:
        name, args, event, holder = req
        try:
            if name == "_prep_separate":
                holder["result"] = self._agent_prep_separate(args)
            elif name == "_add_stems":
                holder["result"] = self._agent_add_stems(args)
            elif name == "_prep_sing":
                holder["result"] = self._agent_prep_sing(args)
            elif name == "_prep_sing_melody":
                holder["result"] = self._agent_prep_sing_melody(args)
            elif name == "_add_vocal":
                holder["result"] = self._agent_add_vocal(args)
            elif name == "_prep_vocalfx":
                holder["result"] = self._agent_prep_vocalfx(args)
            elif name == "_apply_vocalfx_result":
                holder["result"] = self._agent_apply_vocalfx(args)
            elif name == "_prep_stretch":
                holder["result"] = self._agent_prep_stretch(args)
            elif name == "_apply_stretch":
                holder["result"] = self._agent_apply_stretch(args)
            else:
                holder["result"] = self.agent_tools.execute(name, args)
        except Exception as exc:  # noqa: BLE001
            holder["result"] = {"error": str(exc)}
        event.set()

    def _agent_prep_separate(self, args: dict) -> dict:
        """UI thread: extract a clip's audio segment to a temp WAV for Demucs."""
        cid = args.get("clip_id")
        _, clip = self.project.find_clip(cid) if cid else (None, None)
        if clip is None or not clip.source_path:
            return {"error": "separate_stems needs an audio clip id"}
        try:
            data = self.pool.load(clip.source_path)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"couldn't read audio: {exc}"}
        sr = self.project.sample_rate
        s = int(clip.source_offset * sr)
        e = int((clip.source_offset + clip.duration) * sr)
        seg = data[s:e]
        if len(seg) == 0:
            return {"error": "empty clip"}
        import soundfile as sf

        cache = _REPO_ROOT / ".fantasia_cache" / "stems"
        cache.mkdir(parents=True, exist_ok=True)
        src = str(cache / f"src_{uuid.uuid4().hex[:6]}.wav")
        sf.write(src, seg, sr)
        return {"src_path": src, "out_dir": str(cache), "base": clip.name, "start": clip.start}

    def _agent_add_stems(self, args: dict) -> dict:
        """UI thread: create a track + clip per separated stem."""
        stems = args.get("stems") or []
        start = float(args.get("start", 0.0))
        base = args.get("base", "clip")
        made = []
        for name, path, dur in stems:
            tcmd = self.bus.dispatch(AddTrackCommand(f"{base} · {name}"))
            nt = tcmd.created_track
            nt.color = self._STEM_COLORS.get(name, "#4a90d9")
            self.bus.dispatch(AddClipCommand(nt.id, start, float(dur), name=name, source_path=path))
            made.append({"track_id": nt.id, "stem": name})
        self.pool.preload(self.project)
        self._rebuild_all()
        return {"ok": True, "stems": made}

    def _agent_prep_sing(self, args: dict) -> dict:
        """UI thread: pull a MIDI clip's notes for the singing worker."""
        cid = args.get("clip_id")
        _, clip = self.project.find_clip(cid) if cid else (None, None)
        if clip is None or not clip.is_midi or not clip.notes:
            return {"error": "sing needs a MIDI clip id with notes"}
        notes = [Note(n.pitch, n.start, n.duration, n.velocity) for n in clip.notes]
        return {"notes": notes, "base": clip.name, "start": clip.start}

    def _agent_prep_sing_melody(self, args: dict) -> dict:
        """UI thread: convert a beat-timed melody to seconds using the project
        tempo, so the sung vocal is locked to the song's grid."""
        spb = self.project.seconds_per_beat()
        start_beat = float(args.get("start_beat") or 0.0)
        raw = args.get("notes") or []
        if not raw:
            return {"error": "no notes given"}
        notes = []
        for n in raw:
            try:
                notes.append(Note(int(n["pitch"]), float(n["beat"]) * spb,
                                  max(float(n.get("beats", 1)) * spb, 0.05),
                                  int(n.get("velocity", 100))))
            except (KeyError, TypeError, ValueError):
                continue
        if not notes:
            return {"error": "no valid notes"}
        return {"notes": notes, "start": start_beat * spb, "spb": spb}

    def _agent_add_vocal(self, args: dict) -> dict:
        """UI thread: place the sung vocal on a new track."""
        tcmd = self.bus.dispatch(AddTrackCommand(f"Vocal · {args.get('base', 'melody')}"))
        tcmd.created_track.color = theme.PINK
        self.bus.dispatch(AddClipCommand(tcmd.created_track.id, float(args.get("start", 0.0)),
                                         float(args["duration"]), name="Vocal",
                                         source_path=args["path"]))
        self.pool.preload(self.project)
        self._rebuild_all()
        return {"ok": True, "track_id": tcmd.created_track.id}

    def _agent_prep_stretch(self, args: dict) -> dict:
        """UI thread: resolve the stretch factor (from factor or bars) and pull
        the clip's audio segment."""
        cid = args.get("clip_id")
        _, clip = self.project.find_clip(cid) if cid else (None, None)
        if clip is None or not clip.source_path:
            return {"error": "stretch needs an audio clip id"}
        if clip.duration <= 0:
            return {"error": "clip has no length"}
        if args.get("bars") is not None:
            bar = self.project.beats_per_bar * self.project.seconds_per_beat()
            factor = (float(args["bars"]) * bar) / clip.duration if bar > 0 else 1.0
        else:
            factor = float(args.get("factor") or 1.0)
        factor = max(0.1, min(10.0, factor))
        try:
            data = self.pool.load(clip.source_path)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"couldn't read audio: {exc}"}
        sr = self.project.sample_rate
        s = int(clip.source_offset * sr)
        e = int((clip.source_offset + clip.duration) * sr)
        seg = data[s:e]
        if len(seg) == 0:
            return {"error": "empty clip"}
        return {"samples": seg.copy(), "sr": sr, "factor": factor}

    def _agent_apply_stretch(self, args: dict) -> dict:
        """UI thread: replace the clip with its stretched audio."""
        self.bus.dispatch(SetClipSourceCommand(args["clip_id"], args["path"], 0.0,
                                               float(args["duration"])))
        self.pool.preload(self.project)
        self.timeline.refresh_clip(args["clip_id"])
        self.timeline.viewport().update()
        return {"ok": True, "clip_id": args["clip_id"]}

    def _agent_prep_vocalfx(self, args: dict) -> dict:
        """UI thread: pull a clip's audio segment for the vocal-fx worker."""
        cid = args.get("clip_id")
        _, clip = self.project.find_clip(cid) if cid else (None, None)
        if clip is None or not clip.source_path:
            return {"error": "vocal_fx needs an audio clip id"}
        try:
            data = self.pool.load(clip.source_path)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"couldn't read audio: {exc}"}
        sr = self.project.sample_rate
        s = int(clip.source_offset * sr)
        e = int((clip.source_offset + clip.duration) * sr)
        seg = data[s:e]
        if len(seg) == 0:
            return {"error": "empty clip"}
        return {"samples": seg.copy(), "sr": sr, "base": clip.name, "start": clip.start}

    def _agent_apply_vocalfx(self, args: dict) -> dict:
        """UI thread: replace the clip (or add a harmony track) with the result."""
        eff = args.get("effect")
        path, dur, cid = args["path"], float(args["duration"]), args.get("clip_id")
        if eff == "harmony":
            tcmd = self.bus.dispatch(AddTrackCommand(f"Harmony · {args.get('base', 'vocal')}"))
            tcmd.created_track.color = theme.CYAN
            self.bus.dispatch(AddClipCommand(tcmd.created_track.id, float(args.get("start", 0.0)),
                                             dur, name="Harmony", source_path=path))
            self.pool.preload(self.project)
            self._rebuild_all()
            return {"ok": True, "track_id": tcmd.created_track.id}
        self.bus.dispatch(SetClipSourceCommand(cid, path, 0.0, dur))
        self.pool.preload(self.project)
        self.timeline.refresh_clip(cid)
        self.timeline.viewport().update()
        return {"ok": True, "clip_id": cid}

    def _on_agent_done(self, final: str) -> None:
        self._agent_busy = False
        self.agent_panel.set_busy(False)

    def _on_agent_failed(self, err: str) -> None:
        self._agent_busy = False
        self.agent_panel.set_busy(False)
        self.agent_panel.append("system", f"Error: {err}")

    # ---- sound search (M5) ----------------------------------------------
    def _start_search(self, kind: str, payload: str) -> None:
        if not self.search_service.available():
            self.search_panel.set_status(
                "Sound search needs torch+transformers (pip install -e '.[search]')."
            )
            return
        if self._search_worker is not None:
            return
        busy = {
            "text": "Searching… (first use downloads the CLAP model, ~2GB)",
            "audio": "Finding similar sounds…",
            "ingest_folder": "Embedding folder into the library…",
        }.get(kind, "Working…")
        self.search_panel.set_busy(True, busy)
        worker = _SearchWorker(self.search_service, kind, payload, self._seed_folder)
        worker.results.connect(self.search_panel.show_results)
        worker.ingested.connect(self._on_ingested)
        worker.failed.connect(self._on_search_failed)
        worker.finished.connect(lambda w=worker: self._search_done(w))
        self._search_worker = worker
        self._workers.append(worker)
        worker.start()

    def _search_done(self, worker) -> None:
        self._drop_worker(worker)
        if self._search_worker is worker:
            self._search_worker = None
        self.search_panel.set_busy(False)

    def _on_search(self, query: str) -> None:
        self._start_search("text", query)

    def _on_find_similar(self) -> None:
        cid = self.timeline.selected_clip_id()
        if cid is None:
            self.search_panel.set_status("Select an audio clip first.")
            return
        _, clip = self.project.find_clip(cid)
        if clip is None or not clip.source_path:
            self.search_panel.set_status("Pick an audio clip (MIDI clips have no audio).")
            return
        self._start_search("audio", clip.source_path)

    def _on_ingest_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add Sound Folder to Library")
        if folder:
            self._start_search("ingest_folder", folder)

    def _on_ingested(self, count: int) -> None:
        self.search_panel.set_status(
            f"Added {count} sound(s) — {self.search_service.count()} in the library."
        )

    def _ingest_to_library(self, items: list) -> None:
        """Auto-add sounds (recordings/imports/generated) to the search DB off-thread."""
        if not items or not self.search_service.available():
            return
        worker = _IngestWorker(self.search_service, items)
        worker.done.connect(self._on_auto_ingested)
        worker.finished.connect(lambda w=worker: self._drop_worker(w))
        self._workers.append(worker)
        worker.start()

    def _on_auto_ingested(self, count: int) -> None:
        if count:
            self.statusBar().showMessage(
                f"Added {count} sound(s) to the library — {self.search_service.count()} total")

    def _on_search_failed(self, err: str) -> None:
        self.search_panel.set_busy(False)
        self.search_panel.set_status(f"Search failed: {err}")

    def _on_add_search_result(self, path: str, name: str, duration: float) -> None:
        if self.selected_track_id is None:
            if not self.project.tracks:
                return
            self.selected_track_id = self.project.tracks[0].id
        start = self.timeline.playhead
        self.bus.dispatch(
            AddClipCommand(self.selected_track_id, start, duration, name=name, source_path=path)
        )
        self.pool.preload(self.project)
        self.timeline.rebuild()
        track = self.project.track_by_id(self.selected_track_id)
        self.statusBar().showMessage(f"Added '{name}' to {track.name if track else 'track'}")

    # ---- clip editing (M4) ----------------------------------------------
    def _on_clip_action(self, clip_id: str, action: str) -> None:
        track, clip = self.project.find_clip(clip_id)
        if clip is None:
            return
        if action == "write_midi":
            if track is not None and track.is_drum:
                notes = default_drum_pattern(clip.duration, self.project.seconds_per_beat())
                msg = "Wrote a drum beat"
            else:
                notes = default_midi_pattern(clip.duration)
                msg = "Wrote a MIDI clip (C-major scale)"
            self.bus.dispatch(MakeMidiClipCommand(clip_id, notes))
            self._warm()  # render MIDI off the audio thread
            self.timeline.rebuild()
            self.statusBar().showMessage(msg)
        elif action == "transcribe":
            self._transcribe_clip(clip)
        elif action == "hum":
            self._hum_to_melody(clip)
        elif action == "generate":
            self._generate_into_clip(clip)
        elif action == "separate":
            self._separate_clip(clip)
        elif action == "tts":
            self._tts_into_clip(clip)
        elif action == "sing":
            self._sing_clip(clip)
        elif action.startswith("vfx_"):
            self._apply_vocalfx(clip, action[len("vfx_"):])
        elif action.startswith("stretch_"):
            self._stretch_clip_action(clip, action[len("stretch_"):])
        elif action == "tempo_lock":
            self._toggle_tempo_lock(clip)
        elif action == "split":
            self._split_clip(clip)
        elif action == "reverse":
            self.bus.dispatch(SetClipAttrCommand(clip_id, "reversed", not clip.reversed))
            self._warm_clip_buffer(clip)  # decode off the audio thread
        elif action in ("pitch_up", "pitch_down", "pitch_oct_up", "pitch_oct_down", "pitch_reset"):
            delta = {"pitch_up": 1.0, "pitch_down": -1.0, "pitch_oct_up": 12.0,
                     "pitch_oct_down": -12.0}.get(action)
            new = 0.0 if action == "pitch_reset" else clip.pitch_semitones + delta
            self.bus.dispatch(SetClipAttrCommand(clip_id, "pitch_semitones", new))
            self._warm_clip_buffer(clip)
            self.statusBar().showMessage(f"Pitch {new:+.0f} semitones")
        elif action == "normalize":
            gain = self._normalize_gain(clip)
            if gain is None:
                self.statusBar().showMessage("Nothing to normalize")
            else:
                self.bus.dispatch(SetClipAttrCommand(clip_id, "gain_db", gain))
        elif action == "fade_in":
            self.bus.dispatch(SetClipAttrCommand(clip_id, "fade_in", 0.25))
        elif action == "fade_out":
            self.bus.dispatch(SetClipAttrCommand(clip_id, "fade_out", 0.25))
        elif action == "clear_fades":
            self.bus.dispatch(SetClipAttrCommand(clip_id, "fade_in", 0.0))
            self.bus.dispatch(SetClipAttrCommand(clip_id, "fade_out", 0.0))
        elif action == "gain_up":
            self.bus.dispatch(SetClipAttrCommand(clip_id, "gain_db", clip.gain_db + 3.0))
        elif action == "gain_down":
            self.bus.dispatch(SetClipAttrCommand(clip_id, "gain_db", clip.gain_db - 3.0))
        # Any clip edit changes how the clip should look — always repaint.
        self.timeline.viewport().update()

    def _on_split_selected(self) -> None:
        clip_id = self.timeline.selected_clip_id()
        if clip_id is None:
            self.statusBar().showMessage("Select a clip to split")
            return
        _, clip = self.project.find_clip(clip_id)
        if clip is not None:
            self._split_clip(clip)

    def _split_clip(self, clip) -> None:
        ph = self.timeline.playhead
        if clip.start < ph < clip.end:
            self.bus.dispatch(SplitClipCommand(clip.id, ph))
            self.timeline.rebuild()
        else:
            self.statusBar().showMessage("Put the playhead inside the clip to split")

    def _normalize_gain(self, clip, target_db: float = -1.0) -> Optional[float]:
        if not clip.source_path:
            return None
        try:
            data = self.pool.load(clip.source_path)
        except Exception:  # noqa: BLE001
            return None
        sr = self.project.sample_rate
        s = int(clip.source_offset * sr)
        e = int((clip.source_offset + clip.duration) * sr)
        seg = data[s:e]
        if len(seg) == 0:
            return None
        peak = float(np.max(np.abs(seg)))
        if peak <= 1e-6:
            return None
        return target_db - 20.0 * math.log10(peak)

    def _warm_clip_buffer(self, clip) -> None:
        """Pre-decode reversed/pitched buffers so the audio callback never does
        heavy work (which would glitch playback)."""
        if not clip.source_path:
            return
        try:
            if clip.pitch_semitones:
                self.pool.load_pitched(clip.source_path, clip.pitch_semitones)
            if clip.reversed:
                self.pool.load_reversed(clip.source_path)
        except Exception:  # noqa: BLE001
            pass

    # ---- per-track FX (M4) ----------------------------------------------
    _FX_PRESETS = {
        "add_reverb": {"type": "reverb", "params": {"wet": 0.4}},
        "add_delay": {"type": "delay", "params": {"time": 0.3, "feedback": 0.35, "mix": 0.3}},
        "add_lowpass": {"type": "lowpass", "params": {"cutoff": 1200}},
        "add_highpass": {"type": "highpass", "params": {"cutoff": 120}},  # classic rumble cut
        **_FX_PRESETS_EXTRA,
    }

    def _on_fx_action(self, track_id: str, action: str) -> None:
        track = self.project.track_by_id(track_id)
        if track is None:
            return
        if action == "remove_track":
            self.bus.dispatch(RemoveTrackCommand(track_id))  # undoable (Cmd+Z restores it)
            if self.selected_track_id == track_id:
                self.selected_track_id = self.project.tracks[0].id if self.project.tracks else None
            self._rebuild_all()
            self.statusBar().showMessage(f"Removed {track.name} — Undo (Cmd+Z) to restore")
            return
        if action == "toggle_synth":
            self.bus.dispatch(SetTrackAttrCommand(track_id, "is_synth", not track.is_synth))
            self._warm()
            state = "on" if track.is_synth else "off"
            self.statusBar().showMessage(f"Synth voice {state} — {track.name}")
        elif action == "toggle_drum":
            self.bus.dispatch(SetTrackAttrCommand(track_id, "is_drum", not track.is_drum))
            self._warm()  # re-render this track's MIDI as drums/melodic
            state = "on" if track.is_drum else "off"
            self.statusBar().showMessage(f"Drum kit {state} — {track.name}")
        elif action.startswith("instrument:"):
            prog = int(action.split(":", 1)[1])
            self.bus.dispatch(SetTrackAttrCommand(track_id, "instrument", prog))
            self._warm()  # re-render this track's MIDI with the new preset/kit
            if track.is_drum:
                kit = dict(DRUM_KITS).get(prog, f"Kit {prog}")
                self.statusBar().showMessage(f"{track.name}: {kit} kit")
            else:
                self.statusBar().showMessage(f"{track.name}: {gm_name(prog)}")
        elif action == "clear_fx":
            self.bus.dispatch(SetTrackFxCommand(track_id, [], label="Clear track FX"))
            self.statusBar().showMessage(f"Cleared FX on {track.name}")
        elif action in _FX_DIALOGS:
            fx_type, title, spec, hint = _FX_DIALOGS[action]
            dlg = _FxDialog(title, spec, hint, self)
            if dlg.exec() != QDialog.Accepted:
                return
            entry = {"type": fx_type, "params": dlg.params()}
            self.bus.dispatch(SetTrackFxCommand(
                track_id, list(track.fx) + [entry], label=f"Add {fx_type}"))
            self.statusBar().showMessage(f"Added {title.lower()} to {track.name}")
            if fx_type.startswith("eq_"):   # show what the curve now looks like
                self.editor.show_eq(list(track.fx), self.project.sample_rate,
                                    f"EQ — {track.name}")
        elif action in self._FX_PRESETS:
            spec = self._FX_PRESETS[action]
            self.bus.dispatch(
                SetTrackFxCommand(
                    track_id, list(track.fx) + [spec], label=f"Add {spec['type']}"
                )
            )
            self.statusBar().showMessage(f"Added {spec['type']} to {track.name}")
        else:
            return
        self._rebuild_all()  # refresh the FX badge on the header
        self._refresh_eq_curve()  # and the curve, if the chain changed

    def _on_export(self) -> None:
        if self.project.duration <= 0:
            self.statusBar().showMessage("Nothing to export")
            return
        dlg = _ExportDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        scope, ext, subtype, loudness = dlg.result_values()
        self._on_stop()
        sr = self.project.sample_rate
        try:
            if scope == "stems":
                self._export_stems(ext, subtype, loudness)
            else:
                path, _ = QFileDialog.getSaveFileName(
                    self, "Export Mix", f"{self.project.name or 'mix'}.{ext}",
                    f"{ext.upper()} (*.{ext})")
                if not path:
                    return
                self.statusBar().showMessage("Rendering mix…")
                dur = bounce_to_file(self.project, self.pool, sr, path,
                                     midi_renderer=self.midi, synth_renderer=self.synth_engine,
                                     subtype=subtype, loudness=loudness)
                self.statusBar().showMessage(f"Exported {dur:.1f}s mix → {os.path.basename(path)}")
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Export failed: {exc}")

    def _export_stems(self, ext: str, subtype, loudness=None) -> None:
        if not self.project.tracks:
            self.statusBar().showMessage("No tracks to export")
            return
        folder = QFileDialog.getExistingDirectory(self, "Export Stems to Folder")
        if not folder:
            return
        sr = self.project.sample_rate
        used = set()
        count = 0
        for i, track in enumerate(self.project.tracks):
            safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in track.name).strip() or f"track{i+1}"
            name = safe
            n = 2
            while name in used:  # avoid collisions on duplicate track names
                name = f"{safe}_{n}"
                n += 1
            used.add(name)
            path = os.path.join(folder, f"{i+1:02d}_{name}.{ext}")
            self.statusBar().showMessage(f"Rendering stem {i+1}/{len(self.project.tracks)}: {track.name}…")
            self.statusBar().repaint()
            bounce_track_to_file(self.project, self.pool, sr, path, track.id,
                                 midi_renderer=self.midi, synth_renderer=self.synth_engine,
                                 subtype=subtype, loudness=loudness)
            count += 1
        self.statusBar().showMessage(f"Exported {count} stems → {folder}")

    # ---- selection -------------------------------------------------------
    def _on_track_selected(self, track_id: str) -> None:
        self._set_selected_track(track_id)

    def _on_clip_selected(self, clip_id: str) -> None:
        if clip_id:
            track, _ = self.project.find_clip(clip_id)
            if track is not None:
                self._set_selected_track(track.id)

    def _set_selected_track(self, track_id: Optional[str]) -> None:
        self.selected_track_id = track_id
        self._apply_selection_highlight()

    def _apply_selection_highlight(self) -> None:
        self.header_panel.set_selected(self.selected_track_id)
        self.timeline.set_selected_track(self.selected_track_id)
        self._update_target_label()
        # Switch the editor to Synth mode for a synth track; else flip back to
        # Piano Roll mode (without forcing the editor open).
        track = (
            self.project.track_by_id(self.selected_track_id)
            if self.selected_track_id
            else None
        )
        self._refresh_eq_curve()
        if self.editor.stack.currentIndex() == 2:
            pass  # keep the EQ view up while stepping through tracks
        elif track is not None and getattr(track, "is_synth", False):
            self.editor.show_synth(track)
        else:
            self.editor.switch_to_piano_mode()

    def _refresh_eq_curve(self) -> None:
        """Keep the EQ plot showing the selected track's filter chain."""
        track = (self.project.track_by_id(self.selected_track_id)
                 if self.selected_track_id else None)
        if track is None:
            self.editor.eq.set_chain([], self.project.sample_rate, "No track selected")
        else:
            self.editor.eq.set_chain(list(track.fx), self.project.sample_rate,
                                     f"EQ — {track.name}")

    def _on_synth_param(self, track_id: str, key: str, value) -> None:
        self.bus.dispatch(SetTrackSynthParamCommand(track_id, key, value))
        self._warm()  # re-render this synth track's clips
        self.timeline.viewport().update()

    def _update_target_label(self) -> None:
        track = (
            self.project.track_by_id(self.selected_track_id)
            if self.selected_track_id
            else None
        )
        self._target_label.setText(f"Target: {track.name}   " if track else "Target: —   ")

    # ---- import / demo ---------------------------------------------------
    def _on_import(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Import Audio", "", _AUDIO_FILTER)
        if not paths:
            return
        if self.selected_track_id is None and self.project.tracks:
            self.selected_track_id = self.project.tracks[0].id
        if self.selected_track_id is None:
            self.selected_track_id = self.bus.dispatch(AddTrackCommand()).created_track.id
        start = self.timeline.playhead
        imported = 0
        ingest_items = []
        for path in paths:
            try:
                dur = self.pool.duration(path)
            except Exception:  # noqa: BLE001
                self.statusBar().showMessage(f"Couldn't read {os.path.basename(path)}")
                continue
            self.bus.dispatch(
                AddClipCommand(
                    self.selected_track_id, start, dur,
                    name=os.path.basename(path), source_path=path,
                )
            )
            ingest_items.append({"path": path, "name": os.path.basename(path),
                                 "tags": _file_tags(path, ["imported", "sample"])})
            start += dur
            imported += 1
        self.timeline.rebuild()
        self._ingest_to_library(ingest_items)  # make imports searchable too
        self.statusBar().showMessage(f"Imported {imported} clip(s)")

    def _on_import_midi(self) -> None:
        """Import a .mid — either verbatim, or translated into a real strum."""
        from fantasia_core import midi_io

        if not midi_io.available():
            self.statusBar().showMessage("MIDI import needs mido (pip install mido)")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import MIDI", "", "MIDI (*.mid *.midi)")
        if not path:
            return
        try:
            is_pattern = midi_io.has_keyswitches(path)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Couldn't read that MIDI file: {exc}")
            return
        dlg = _MidiImportDialog(os.path.basename(path), is_pattern, self)
        if dlg.exec() != QDialog.Accepted:
            return
        mode, chord, strum_ms = dlg.values()
        spb = self.project.seconds_per_beat()
        try:
            if mode == "strum":
                from fantasia_core.strum import import_strum
                notes = import_strum(path, spb, chord, strum_ms=strum_ms)
                label = f"{os.path.splitext(os.path.basename(path))[0]} ({chord})"
            else:
                notes = midi_io.import_notes(
                    path, spb, drop_keyswitches=(mode == "raw_clean"))
                label = os.path.splitext(os.path.basename(path))[0]
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"MIDI import failed: {exc}")
            return
        if not notes:
            self.statusBar().showMessage("That MIDI file has no notes")
            return

        if self.selected_track_id is None:
            if not self.project.tracks:
                self.selected_track_id = self.bus.dispatch(AddTrackCommand()).created_track.id
            else:
                self.selected_track_id = self.project.tracks[0].id
        # Notes come back relative to the file's start; place the clip at the
        # playhead and keep them clip-relative.
        start = self.timeline.playhead
        dur = max(n.start + n.duration for n in notes)
        cmd = self.bus.dispatch(AddClipCommand(self.selected_track_id, start, dur, name=label))
        clip = cmd.created_clip
        if clip is None:
            return
        self.bus.dispatch(MakeMidiClipCommand(clip.id, notes))
        self._warm()
        self.timeline.rebuild()
        self.statusBar().showMessage(
            f"Imported {len(notes)} notes from {os.path.basename(path)}"
            + (f" as a {chord} strum" if mode == "strum" else ""))

    def _on_load_demo(self) -> None:
        if not self._maybe_discard():
            return
        sdir = _REPO_ROOT / "assets" / "samples"
        proj = Project(name="Demo", tempo=120.0)
        for tname, fname, color in _DEMO_TRACKS:
            fpath = sdir / fname
            if not fpath.exists():
                continue
            track = proj.add_track(tname)
            track.color = color
            track.gain_db = -6.0  # headroom so the 4-track sum doesn't clip
            dur = self.pool.duration(str(fpath))
            proj.add_clip(track.id, 0.0, dur, name=fname, source_path=str(fpath))
            proj.add_clip(track.id, dur, dur, name=fname, source_path=str(fpath))
        if not proj.tracks:
            self.statusBar().showMessage("Demo samples not found — run tools/make_demo_audio.py")
            return
        self._load_project(proj, path=None)
        self.statusBar().showMessage("Loaded demo arrangement — press Space to play")

    # ---- file ------------------------------------------------------------
    def _load_project(self, project: Project, path: Optional[str]) -> None:
        self._on_stop()
        self.project = project
        self._current_path = path
        self.bus.set_project(project)
        self.engine.set_project(project)
        self.selected_track_id = project.tracks[0].id if project.tracks else None
        self.timeline.set_project(project)
        self.pool.preload(project)
        self._warm(project)
        self._rebuild_all()
        self._sync_tempo_display()  # show the loaded project's tempo
        self._conform_locked_clips()  # conform tempo-locked clips to the loaded tempo
        self._project_label = os.path.basename(path) if path else (project.name or "Untitled")
        self._set_dirty(False)
        # Whatever is loaded becomes the project we reopen next launch; New and
        # Load Demo pass path=None, which clears it.
        self._remember_path(path)

    # ---- last-project memory ---------------------------------------------
    _SETTINGS_LAST_PATH = "session/last_project"

    def _remember_path(self, path: Optional[str]) -> None:
        """Persist (or clear) the last-used file so the next launch reopens it."""
        settings = QSettings("FantasiaConductor", "FantasiaConductor")
        if path:
            settings.setValue(self._SETTINGS_LAST_PATH, path)
        else:
            settings.remove(self._SETTINGS_LAST_PATH)

    def _restore_last_project(self) -> None:
        """Reopen the last-saved project at startup, so ⌘S updates it directly.

        Silently falls back to the empty Untitled project if the file is gone,
        unreadable, or written by an incompatible version.
        """
        settings = QSettings("FantasiaConductor", "FantasiaConductor")
        path = settings.value(self._SETTINGS_LAST_PATH, "", type=str)
        if not path or not os.path.isfile(path):
            return
        try:
            project = load_project(path)
        except Exception as exc:  # noqa: BLE001 — never block startup on a bad file
            self.statusBar().showMessage(f"Couldn't reopen {os.path.basename(path)}: {exc}")
            self._remember_path(None)
            return
        self._load_project(project, path=path)
        self.statusBar().showMessage(f"Reopened {os.path.basename(path)} — ⌘S saves back to it")

    def _on_save(self) -> None:
        """Save to the current file; if the project was never saved, prompt (Save As)."""
        if not self._current_path:
            self._on_save_as()
            return
        save_project(self.project, self._current_path)
        self._set_dirty(False)
        self._remember_path(self._current_path)
        self.statusBar().showMessage(f"Saved {os.path.basename(self._current_path)}")

    def _on_save_as(self) -> None:
        """Always prompt for a new file, then save to it and make it current."""
        default = self._current_path or f"{self.project.name or 'untitled'}.fcp"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", default, "Fantasia Project (*.fcp)")
        if not path:
            return
        save_project(self.project, path)
        self._current_path = path
        self._project_label = os.path.basename(path)
        self._set_dirty(False)
        self._remember_path(path)
        self.statusBar().showMessage(f"Saved {path}")

    def _on_new(self) -> None:
        if not self._maybe_discard():
            return
        proj = Project(name="Untitled")
        proj.add_track("Track 1")  # start with one empty track, like launch
        self._load_project(proj, path=None)
        self.statusBar().showMessage("New project")

    def _on_open(self) -> None:
        if not self._maybe_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "Fantasia Project (*.fcp)"
        )
        if not path:
            return
        self._load_project(load_project(path), path=path)
        self.statusBar().showMessage(f"Opened {path}")

    # ---- lifecycle -------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        if not self._maybe_discard():
            event.ignore()
            return
        self.bridge.stop()
        self.engine.close()
        self.recorder.close()
        self.midi.close()
        super().closeEvent(event)

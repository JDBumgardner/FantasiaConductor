"""Audio engine — buffers/peaks, mixing, and playback (headless, no Qt)."""

from fantasia_core.engine.bounce import bounce_to_array, bounce_to_file, bounce_track_to_file
from fantasia_core.engine.buffers import AudioPool
from fantasia_core.engine.fx import FxHost, build_board
from fantasia_core.engine.midi_render import MidiRenderer, default_soundfont
from fantasia_core.engine.metronome import render_metronome_block
from fantasia_core.engine.mixer import render_block, render_track_block
from fantasia_core.engine.playback import PlaybackEngine
from fantasia_core.engine.record import Recorder, list_input_devices
from fantasia_core.engine.synth import DEFAULT_PATCH, WAVEFORMS, SynthRenderer

__all__ = [
    "AudioPool",
    "render_block",
    "render_track_block",
    "render_metronome_block",
    "PlaybackEngine",
    "Recorder",
    "list_input_devices",
    "bounce_to_array",
    "bounce_to_file",
    "bounce_track_to_file",
    "FxHost",
    "build_board",
    "MidiRenderer",
    "default_soundfont",
    "SynthRenderer",
    "DEFAULT_PATCH",
    "WAVEFORMS",
]

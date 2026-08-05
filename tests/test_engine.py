"""Audio engine: buffer pool, peaks, and the mixer (M3, no audio device)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from fantasia_core.document import Project
from fantasia_core.engine import AudioPool, render_block

SAMPLES = pathlib.Path(__file__).resolve().parent.parent / "assets" / "samples"
DRUMS = str(SAMPLES / "drums.wav")
BASS = str(SAMPLES / "bass.wav")

pytestmark = pytest.mark.skipif(
    not (SAMPLES / "drums.wav").exists(),
    reason="demo audio not generated (run tools/make_demo_audio.py)",
)


def _project_with(path: str, start: float = 0.0, dur: float = 2.0) -> tuple[Project, AudioPool]:
    p = Project(sample_rate=44100)
    t = p.add_track("A")
    p.add_clip(t.id, start=start, duration=dur, name="c", source_path=path)
    pool = AudioPool(p.sample_rate)
    pool.preload(p)
    return p, pool


def test_load_and_duration():
    pool = AudioPool(44100)
    data = pool.load(DRUMS)
    assert data.dtype == np.float32 and data.ndim == 2
    assert abs(pool.duration(DRUMS) - 4.0) < 0.01


def test_render_block_has_audio():
    p, pool = _project_with(DRUMS)
    block = render_block(p, pool, start_frame=0, num_frames=2048, sr=44100)
    assert block.shape == (2048, 2)
    assert np.max(np.abs(block)) > 0.0  # first kick is at t=0


def test_block_before_clip_is_silent():
    p, pool = _project_with(DRUMS, start=5.0, dur=2.0)  # clip starts at 5s
    block = render_block(p, pool, start_frame=0, num_frames=2048, sr=44100)
    assert np.max(np.abs(block)) == 0.0


def test_mute_silences_track():
    p, pool = _project_with(DRUMS)
    p.tracks[0].mute = True
    block = render_block(p, pool, 0, 2048, 44100)
    assert np.max(np.abs(block)) == 0.0


def test_solo_isolates():
    p, pool = _project_with(DRUMS)
    t2 = p.add_track("B")
    p.add_clip(t2.id, 0.0, 2.0, "c2", source_path=BASS)
    pool.preload(p)
    full = render_block(p, pool, 0, 2048, 44100)
    p.tracks[0].solo = True  # solo drums only
    soloed = render_block(p, pool, 0, 2048, 44100)
    # Soloed mix should differ from the full mix (bass removed).
    assert not np.allclose(full, soloed)


def test_gain_scales_amplitude():
    p, pool = _project_with(DRUMS)
    loud = render_block(p, pool, 0, 2048, 44100)
    p.tracks[0].gain_db = -20.0
    quiet = render_block(p, pool, 0, 2048, 44100)
    assert np.max(np.abs(quiet)) < np.max(np.abs(loud))


def test_reverse_flips_audio():
    p, pool = _project_with(DRUMS, start=0.0, dur=4.0)
    fwd = render_block(p, pool, 0, 4096, 44100)
    p.tracks[0].clips[0].reversed = True
    rev = render_block(p, pool, 0, 4096, 44100)
    # Reversed playback differs from forward (kick is at the very start forward).
    assert not np.allclose(fwd, rev)


def test_bounce_matches_length():
    from fantasia_core.engine import bounce_to_array

    p, pool = _project_with(DRUMS, start=0.0, dur=2.0)
    t2 = p.add_track("B")
    p.add_clip(t2.id, 1.0, 3.0, "c2", source_path=BASS)
    pool.preload(p)
    mix = bounce_to_array(p, pool, 44100)
    assert mix.shape == (int(p.duration * 44100), 2)  # spans to the last clip end
    assert np.max(np.abs(mix)) > 0.0


def test_track_fx_changes_output():
    from fantasia_core.engine import FxHost

    p, pool = _project_with(DRUMS, start=0.0, dur=2.0)
    dry = render_block(p, pool, 0, 8192, 44100)
    p.tracks[0].fx = [{"type": "reverb", "params": {"wet": 0.7}}]
    wet = render_block(p, pool, 0, 8192, 44100, fx_host=FxHost())
    assert not np.allclose(dry, wet)


def test_pitch_shift_changes_output_same_length():
    p, pool = _project_with(BASS, start=0.0, dur=2.0)
    dry = render_block(p, pool, 0, 8192, 44100)
    p.tracks[0].clips[0].pitch_semitones = 5.0
    shifted = render_block(p, pool, 0, 8192, 44100)
    assert not np.allclose(dry, shifted)
    # Pitch shift preserves buffer length (so timeline timing is unchanged).
    assert len(pool.load_pitched(BASS, 5.0)) == len(pool.load(BASS))


def test_peaks_shape():
    pool = AudioPool(44100)
    mins, maxs = pool.peaks(DRUMS, 0.0, 2.0, buckets=200)
    assert len(mins) == len(maxs) == 200
    assert np.any(maxs > 0)

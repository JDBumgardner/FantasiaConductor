"""Built-in subtractive synth: default trio + render."""

from __future__ import annotations

import numpy as np

from fantasia_core.document.model import Clip, Note
from fantasia_core.engine.synth import DEFAULT_PATCH, render_clip, render_note


def test_default_patch_is_filtered_detuned_saws():
    assert DEFAULT_PATCH["osc1"] == DEFAULT_PATCH["osc2"] == DEFAULT_PATCH["osc3"] == "saw"
    assert DEFAULT_PATCH["mix"] == 1.0
    assert 0.05 <= float(DEFAULT_PATCH["detune"]) <= 0.25
    assert float(DEFAULT_PATCH["cutoff"]) <= 2500.0


def test_render_note_is_audible():
    buf = render_note(DEFAULT_PATCH, 60, 0.3, 22050)
    assert buf.dtype == np.float32
    assert len(buf) > 100
    assert float(np.max(np.abs(buf))) > 0.01


def test_empty_patch_fills_defaults():
    clip = Clip(id="c", name="n", start=0.0, duration=0.5, content_type="midi",
                notes=[Note(60, 0.0, 0.2, 100)])
    out = render_clip({}, clip, 22050)
    assert out.shape[0] > 0
    assert float(np.max(np.abs(out))) > 0.0

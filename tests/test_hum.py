"""Hum → melody: monophonic pitch tracking and note segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from fantasia_core.hum import available, transcribe_hum

pytestmark = pytest.mark.skipif(not available(), reason="librosa not installed")
SR = 44100


def _tone(pitch, dur, sr=SR, glide_from=None, harmonics=True):
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = np.full(n, 440.0 * 2 ** ((pitch - 69) / 12))
    if glide_from is not None:
        # Portamento, capped at a quarter of the note: singers slide briefly on
        # fast notes. (A slide covering most of a short note genuinely *is*
        # ambiguous, and gets read as a glide — which is the desired behaviour.)
        g = min(int(0.09 * sr), max(1, int(0.25 * n)))
        f[:g] = np.linspace(440.0 * 2 ** ((glide_from - 69) / 12), f[0], g)
    f *= 1 + 0.008 * np.sin(2 * np.pi * 5.0 * t)   # vibrato
    ph = 2 * np.pi * np.cumsum(f) / sr
    env = np.minimum(1, t / 0.04) * np.minimum(1, (dur - t) / 0.08)
    v = np.sin(ph) + (0.3 * np.sin(2 * ph) if harmonics else 0.0)
    return (0.4 * v * env).astype(np.float32)


def _render(seq, total, sr=SR):
    y = np.zeros(int(total * sr), dtype=np.float32)
    prev = None
    for pitch, start, dur in seq:
        seg = _tone(pitch, dur, sr, glide_from=prev)
        i = int(start * sr)
        y[i:i + len(seg)] += seg
        prev = pitch
    return y


def test_hum_with_portamento_gives_one_note_per_hum():
    seq = [(62, 0.0, 0.5), (64, 0.55, 0.45), (65, 1.05, 0.5),
           (67, 1.6, 0.75), (64, 2.45, 0.85)]
    notes = transcribe_hum(_render(seq, 3.6), SR, quantize=False)
    assert [n.pitch for n in notes] == [62, 64, 65, 67, 64]
    for n, (_, start, _) in zip(notes, seq):     # timing lands close
        assert abs(n.start - start) < 0.15


def test_fast_stepwise_notes_are_not_mistaken_for_glides():
    """A quick C-E-G-E-C looks like a slide by shape alone — the pitch-movement
    test is what keeps these real notes."""
    seq = [(60, 0.0, 0.14), (64, 0.15, 0.14), (67, 0.30, 0.14),
           (64, 0.45, 0.14), (60, 0.60, 0.14)]
    notes = transcribe_hum(_render(seq, 2.0), SR, quantize=False)
    assert [n.pitch for n in notes] == [60, 64, 67, 64, 60]


def test_quantize_snaps_to_the_grid():
    seq = [(60, 0.03, 0.45), (62, 0.53, 0.45)]
    spb = 0.5                                   # 120 BPM -> 1/16 grid = 0.125s
    notes = transcribe_hum(_render(seq, 1.2), SR, spb=spb, quantize=True)
    for n in notes:
        assert abs((n.start / (spb / 4)) - round(n.start / (spb / 4))) < 1e-6


def test_key_snapping_forces_scale_tones():
    seq = [(61, 0.0, 0.5), (66, 0.55, 0.5)]     # C#4 and F#4: both out of C major
    notes = transcribe_hum(_render(seq, 1.2), SR, quantize=False,
                           key="c", scale="major")
    assert notes, "expected notes"
    assert all(n.pitch % 12 in (0, 2, 4, 5, 7, 9, 11) for n in notes)


def test_silence_gives_no_notes():
    assert transcribe_hum(np.zeros(SR, dtype=np.float32), SR) == []

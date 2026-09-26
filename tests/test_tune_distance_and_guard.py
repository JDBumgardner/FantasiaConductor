"""The two faults that made tune_toward remove processing when asked to add it.

1. The distance metric floored at an absolute 1e-9, so a silent frame was not cheap — it was the most expensive
   thing in the window. The same reverb measured 0.008 on a dense part and 1.62 on one 60% silent, and the
   two-sided constraint then dragged the search inward.
2. Nothing rejected a proposal that scored worse than the untouched sound, so a stop that lost to doing nothing
   was still offered as a suggestion.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "experiments" / "text2fx"))


def _tone(seconds=4.0, sr=48000):
    """A few harmonics with an amplitude envelope — something with real spectral structure."""
    t = torch.arange(int(seconds * sr)) / sr
    x = sum(torch.sin(2 * torch.pi * f * t) / (i + 1) for i, f in enumerate((220.0, 440.0, 880.0, 1760.0)))
    return (x * (0.5 + 0.5 * torch.sin(2 * torch.pi * 1.7 * t))).float() * 0.2


def _blank(x, frac, sr=48000):
    y = x.clone()
    chunk = int(0.2 * sr)
    for i in range(int(frac * len(x) / chunk)):
        st = int((i + 0.5) * len(x) / max(int(frac * len(x) / chunk), 1))
        y[st:st + chunk] = 0.0
    return y


def _quiet_tail(x, sr=48000, db=-30.0):
    """A decaying tail — the change that used to be swamped by whatever silence surrounded it."""
    n = int(0.3 * sr)
    g = torch.Generator().manual_seed(0)
    ir = torch.randn(n, generator=g) * torch.exp(-torch.arange(n) / (0.08 * sr))
    ir = ir / ir.norm() * (10 ** (db / 20))
    L = len(x) + n
    y = torch.fft.irfft(torch.fft.rfft(x, n=L) * torch.fft.rfft(ir, n=L), n=L)[:len(x)]
    return x + y


def test_the_same_change_measures_the_same_however_much_silence_surrounds_it():
    import common as C

    x = _tone()
    dense = float(C.band_energy_loss(_quiet_tail(x), x))
    sparse_x = _blank(x, 0.6)
    sparse = float(C.band_energy_loss(_quiet_tail(sparse_x), sparse_x))
    assert dense > 0, "the change must register at all"
    ratio = sparse / dense
    assert ratio < 4.0, (f"silence still dominates the metric: {ratio:.1f}x "
                         f"(dense {dense:.4f}, 60% silent {sparse:.4f})")


def test_a_real_spectral_change_still_outweighs_a_quiet_tail():
    """The floor must not be so deep that it throws away audible content — the failure of a single global floor,
    which discarded 60% of a +4 dB shelf."""
    import common as C

    x = _tone()
    f = torch.fft.rfftfreq(x.numel(), 1 / 48000)
    tilt = torch.fft.irfft(torch.fft.rfft(x) * torch.where(f > 700.0, 10 ** (4 / 20), 1.0), n=x.numel())
    assert float(C.band_energy_loss(tilt, x)) > 3 * float(C.band_energy_loss(_quiet_tail(x), x))


def _job_with(stops, objective="dir"):
    from fantasia_core import tune as T

    j = T.Job("t", {}, "/tmp")
    j.status, j.result = "done", {"stops": stops, "objective": objective, "frozen": []}
    return j


def test_a_stop_that_lost_to_doing_nothing_is_never_suggested():
    base = {"stop": 0, "word": 0.20, "direction": 0.0, "distance": 0.0, "identity": 0.5, "flags": []}
    stops = [base,
             {"stop": 1, "word": 0.25, "direction": -0.05, "flags": [], "worse_than_start": True},
             {"stop": 2, "word": 0.28, "direction": +0.21, "flags": [], "worse_than_start": False},
             {"stop": 3, "word": 0.31, "direction": +0.18, "flags": [], "worse_than_start": False}]
    assert 1 not in _job_with(stops).summary()["suggested_stops"]


def test_a_regression_on_the_other_measure_is_reported_but_not_hidden():
    """A pad scored below baseline on the word at every stop while direction was strongly positive. The word is
    not what the directional search maximises, so the stop stays usable — but the disagreement must be visible."""
    base = {"stop": 0, "word": 0.27, "direction": 0.0, "distance": 0.0, "identity": 0.5, "flags": []}
    stops = [base,
             {"stop": 1, "word": 0.24, "direction": +0.33, "flags": [],
              "regressed": ["word"], "worse_than_start": False},
             {"stop": 2, "word": 0.25, "direction": +0.18, "flags": [],
              "regressed": ["word"], "worse_than_start": False}]
    out = _job_with(stops).summary()
    assert out["suggested_stops"], "a word regression alone must not empty the suggestions"
    assert all("word" in s.get("regressed", []) for s in out["stops"][1:]), "the disagreement must be reported"

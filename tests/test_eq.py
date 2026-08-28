"""Stock 8-band EQ: band schema, chain packing, analytic response, spectrum tap."""

from __future__ import annotations

import numpy as np
import pytest

from fantasia_core.document import MASTER_ID, Project
from fantasia_core.engine.eq import (
    MAX_BANDS,
    bands_from_fx,
    default_bands,
    fx_with_eq,
    normalize_band,
    response_db,
    struct_sig,
)
from fantasia_core.engine.spectrum import SpectrumTap, spectrum_db


def test_default_layout_is_eight_bands():
    bands = default_bands()
    assert len(bands) == MAX_BANDS
    assert bands[0]["type"] == "low_cut" and bands[0]["enabled"] is False
    assert bands[-1]["type"] == "high_cut" and bands[-1]["enabled"] is False
    assert all(b["type"] for b in bands)


def test_empty_chain_presents_the_default_layout():
    assert [b["type"] for b in bands_from_fx([])] == [b["type"] for b in default_bands()]


def test_legacy_filters_are_lifted_into_bands():
    fx = [
        {"type": "highpass", "params": {"cutoff": 80}},
        {"type": "reverb", "params": {"wet": 0.4}},
        {"type": "eq_peak", "params": {"freq": 1000, "gain": 3, "q": 1.2}},
    ]
    bands = bands_from_fx(fx)
    assert bands[0]["type"] == "low_cut" and bands[0]["freq"] == 80
    assert bands[1]["type"] == "bell" and bands[1]["gain"] == 3
    assert bands[2]["enabled"] is False


def test_writing_eq_replaces_legacy_filters_and_keeps_other_fx():
    fx = [
        {"type": "highpass", "params": {"cutoff": 80}},
        {"type": "reverb", "params": {"wet": 0.4}},
        {"type": "eq_peak", "params": {"freq": 1000, "gain": 3, "q": 1}},
    ]
    bands = default_bands()
    bands[2]["gain"] = 4.0
    out = fx_with_eq(fx, bands)
    kinds = [e.type for e in out]
    assert kinds == ["eq", "reverb"]
    assert out[0].params["bands"][2]["gain"] == 4.0


def test_writing_eq_updates_an_existing_insert_in_place():
    fx = [
        {"type": "delay", "params": {}},
        {"type": "eq", "params": {"bands": default_bands()}},
        {"type": "reverb", "params": {}},
    ]
    bands = default_bands()
    bands[1]["gain"] = -3.0
    out = fx_with_eq(fx, bands)
    assert [e.type for e in out] == ["delay", "eq", "reverb"]
    assert out[1].params["bands"][1]["gain"] == -3.0


def test_writing_eq_preserves_insert_id_and_bypass():
    from fantasia_core.document.fx_insert import FxInsert

    bands = default_bands()
    original = FxInsert(
        id="fx9", type="eq", params={"bands": bands}, bypassed=True)
    bands[1]["gain"] = -3.0
    out = fx_with_eq([original], bands)
    assert len(out) == 1
    assert out[0].id == "fx9" and out[0].bypassed is True
    assert out[0].params["bands"][1]["gain"] == -3.0


def test_struct_sig_ignores_knob_moves():
    a = [{"type": "eq", "params": {"bands": default_bands()}}]
    b = [{"type": "eq", "params": {"bands": default_bands()}}]
    b[0]["params"]["bands"][2]["gain"] = 6.0
    b[0]["params"]["bands"][2]["freq"] = 900.0
    assert struct_sig(a) == struct_sig(b)
    b[0]["params"]["bands"][2]["type"] = "notch"
    assert struct_sig(a) != struct_sig(b)
    bypassed = [{"id": "fx1", "type": "reverb", "params": {}, "bypassed": True}]
    live = [{"id": "fx1", "type": "reverb", "params": {}}]
    assert struct_sig(bypassed) != struct_sig(live)


def test_q_from_vertical_drag_is_relative_and_clipped():
    from fantasia_core.engine.eq import q_from_vertical_drag

    assert q_from_vertical_drag(1.0, 0.0) == pytest.approx(1.0)
    assert q_from_vertical_drag(2.0, 0.4) > 2.0
    assert q_from_vertical_drag(2.0, -0.4) < 2.0
    assert q_from_vertical_drag(1.0, 20.0) == 18.0
    assert q_from_vertical_drag(0.5, -20.0) == 0.1


def test_peak_response_centres_on_the_band():
    bands = [normalize_band({"type": "bell", "freq": 1000, "gain": 12, "q": 2, "enabled": True})]
    freqs = np.array([100.0, 1000.0, 10000.0])
    db = response_db(bands, freqs, 44100)
    assert db[1] > db[0] and db[1] > db[2]
    assert 10.0 < db[1] < 13.0


def test_disabled_band_is_flat():
    bands = [normalize_band({"type": "bell", "freq": 1000, "gain": 12, "q": 1, "enabled": False})]
    db = response_db(bands, np.array([1000.0]), 44100)
    assert abs(float(db[0])) < 0.2


def test_spectrum_tap_does_not_grow_and_preserves_energy():
    tap = SpectrumTap(1024)
    block = np.ones((256, 2), dtype=np.float32)
    for _ in range(8):
        tap.write(block)
    snap = tap.snapshot()
    assert snap.shape == (1024, 2)
    assert float(np.mean(snap[-256:])) == pytest.approx(1.0)


def test_spectrum_db_peaks_at_the_tone():
    sr = 44100
    t = np.arange(2048) / sr
    tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    stereo = np.stack([tone, tone], axis=1)
    freqs, db = spectrum_db(stereo, sr, n_fft=2048)
    peak = freqs[int(np.argmax(db))]
    assert 800 < peak < 1200


def test_master_is_addressable_and_survives_round_trip():
    from fantasia_core.document.serialize import project_from_dict, project_to_dict

    p = Project()
    assert p.master.id == MASTER_ID
    p.master.fx = [{"type": "eq", "params": {"bands": default_bands()}}]
    p.master.gain_db = -2.0
    restored = project_from_dict(project_to_dict(p))
    assert restored.master.id == MASTER_ID
    assert restored.master.gain_db == -2.0
    assert restored.master.fx[0].type == "eq"
    assert restored.master.fx[0].id
    assert restored.track_by_id(MASTER_ID) is restored.master
    assert p.remove_track(MASTER_ID) is None
    assert p.track_by_id(MASTER_ID) is p.master


def _fx_run(spec, x, sr=44100):
    from fantasia_core.engine.fx import build_board
    board = build_board([spec])
    assert board is not None, f"{spec['type']} failed to build"
    return board(x.T.astype(np.float32), sr).T


def _band(x, lo, hi, sr=44100):
    S = np.abs(np.fft.rfft(x[:, 0]))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return float(np.sum(S[(f >= lo) & (f < hi)]))


def test_stock_eq_insert_matches_legacy_peak():
    pytest.importorskip("pedalboard")
    sr = 44100
    t = np.arange(sr) / sr
    sig = 0.3 * np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 5000 * t)
    x = np.stack([sig] * 2, axis=1).astype(np.float32)
    high0 = _band(x, 4000, 6000)
    legacy = _fx_run({"type": "eq_peak", "params": {"freq": 5000, "gain": -24, "q": 2.0}}, x)
    stock = _fx_run({"type": "eq", "params": {"bands": [
        {"type": "bell", "freq": 5000, "gain": -24, "q": 2.0, "enabled": True},
    ]}}, x)
    assert _band(legacy, 4000, 6000) / high0 < 0.4
    assert _band(stock, 4000, 6000) / high0 < 0.4

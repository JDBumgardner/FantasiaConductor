"""Hosting VST3/AU plugins, and the surface the agent drives them through.

No plugin is installed in CI, and loading one is the part pedalboard owns
anyway. What is worth pinning is the layer around it: how notes become MIDI
bytes, how a parameter is found and set, and that a big synth's parameter list
gets searched rather than dumped.
"""

from __future__ import annotations

import numpy as np
import pytest

from fantasia_core import plugins


class _Param:
    """Stands in for pedalboard's parameter object."""

    def __init__(self, name, raw=0.5, text="440 Hz", label="Hz",
                 boolean=False, discrete=False, steps=0):
        self.name, self.raw_value, self.label = name, raw, label
        self._text = text
        self.is_boolean, self.is_discrete, self.num_steps = boolean, discrete, steps

    @property
    def string_value(self):
        return self._text

    def get_raw_value_for_text(self, text):
        if text.endswith(" Hz"):
            return float(text[:-3]) / 20000.0
        raise ValueError("unparseable")


class _Plugin:
    def __init__(self, params):
        self.parameters = params
        self.calls = []

    def __call__(self, midi, duration, sample_rate):
        self.calls.append((midi, duration, sample_rate))
        return np.zeros((2, int(duration * sample_rate)), dtype=np.float32)


class _Note:
    def __init__(self, pitch, start, duration, velocity=100):
        self.pitch, self.start, self.duration, self.velocity = pitch, start, duration, velocity


# ---- notes -> MIDI ------------------------------------------------------
def test_notes_become_paired_on_off_messages():
    msgs = plugins.notes_to_midi([_Note(60, 0.0, 0.5, 90)])
    assert len(msgs) == 2
    (on, t_on), (off, t_off) = msgs
    assert on[0] & 0xF0 == 0x90 and on[1] == 60 and on[2] == 90
    assert off[0] & 0xF0 == 0x80 and off[1] == 60
    assert t_on == 0.0 and t_off == 0.5


def test_notes_are_sorted_and_offset():
    msgs = plugins.notes_to_midi([_Note(64, 1.0, 0.5), _Note(60, 0.0, 0.5)], offset=2.0)
    assert [m[1] for m in msgs] == [2.0, 2.5, 3.0, 3.5]


def test_velocity_and_pitch_are_clamped_to_midi_range():
    msgs = plugins.notes_to_midi([_Note(200, 0.0, 0.1, 0), _Note(-5, 0.1, 0.1, 999)])
    for data, _t in msgs:
        assert 0 <= data[1] <= 127
        assert 0 <= data[2] <= 127


def test_render_leaves_room_for_the_release():
    """Cutting at the last note-off chops the ending off anything with a slow
    release."""
    p = _Plugin({})
    plugins.render_notes(p, [_Note(60, 0.0, 1.0)], duration=1.0, sr=100, tail=0.5)
    _midi, duration, _sr = p.calls[0]
    assert duration == pytest.approx(1.5)


def test_render_with_no_notes_is_empty():
    assert len(plugins.render_notes(_Plugin({}), [], duration=1.0)) == 0


# ---- parameters ---------------------------------------------------------
def test_params_are_searched_not_dumped():
    """Vital exposes hundreds; handing an agent all of them is neither useful
    nor cheap."""
    p = _Plugin({f"p{i}": _Param(f"Osc {i} Level") for i in range(200)})
    assert len(plugins.describe(p, limit=40)) == 40
    hits = plugins.describe(p, "osc 7")
    assert hits and all("7" in h["name"] for h in hits)


def test_describe_reports_the_displayed_value():
    p = _Plugin({"cut": _Param("Filter Cutoff", raw=0.3, text="880 Hz")})
    row = plugins.describe(p, "cutoff")[0]
    assert row["value"] == "880 Hz"
    assert row["unit"] == "Hz"
    assert row["raw"] == 0.3


def test_describe_marks_switches_and_choices():
    p = _Plugin({"a": _Param("Sync", boolean=True),
                 "b": _Param("Mode", discrete=True, steps=4)})
    kinds = {r["name"]: r.get("type") for r in plugins.describe(p)}
    assert kinds["Sync"] == "switch"
    assert kinds["Mode"] == "choice"


def test_set_by_text_uses_the_plugins_own_conversion():
    """Mapping a displayed value onto the internal 0-1 range is the plugin's
    business; guessing it from outside is how you land on the wrong value."""
    prm = _Param("Filter Cutoff")
    plugins.set_param(_Plugin({"cut": prm}), "Filter Cutoff", "1000 Hz")
    assert prm.raw_value == pytest.approx(0.05)


def test_set_accepts_a_raw_number():
    prm = _Param("Filter Cutoff")
    plugins.set_param(_Plugin({"cut": prm}), "cut", 0.75)
    assert prm.raw_value == 0.75


def test_set_falls_back_to_a_number_the_plugin_cannot_parse():
    prm = _Param("Filter Cutoff")
    plugins.set_param(_Plugin({"cut": prm}), "cut", "0.25")
    assert prm.raw_value == pytest.approx(0.25)


def test_set_matches_a_parameter_loosely():
    prm = _Param("Filter 1 Cutoff")
    out = plugins.set_param(_Plugin({"f1c": prm}), "filter 1 cutoff", 0.4)
    assert out["key"] == "f1c"


def test_unknown_parameter_is_named_in_the_error():
    with pytest.raises(KeyError, match="nonsense"):
        plugins.set_param(_Plugin({"a": _Param("A")}), "nonsense", 0.5)


def test_unparseable_text_is_reported():
    with pytest.raises(ValueError):
        plugins.set_param(_Plugin({"a": _Param("A")}), "A", "wide open")


# ---- discovery ----------------------------------------------------------
def test_scan_is_cached_until_refreshed(monkeypatch):
    calls = []
    monkeypatch.setattr(plugins, "search_paths", lambda: (calls.append(1), [])[1])
    plugins.scan(refresh=True)
    plugins.scan()
    plugins.scan()
    assert len(calls) == 1


def test_extra_search_path_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("FANTASIA_PLUGIN_PATH", str(tmp_path))
    assert tmp_path in plugins.search_paths()


def test_loading_an_unknown_plugin_lists_what_is_installed(monkeypatch):
    monkeypatch.setattr(plugins, "scan", lambda refresh=False: [])
    with pytest.raises(FileNotFoundError, match="none found"):
        plugins.load("Vital")


def test_a_huge_step_count_is_not_a_choice():
    """Plugins are loose with this flag — Vital marks all 903 of its parameters
    discrete, 557 of them with 2**31-1 steps, which plainly means continuous."""
    p = _Plugin({"a": _Param("Cutoff", discrete=True, steps=2147483647),
                 "b": _Param("Mode", discrete=True, steps=6)})
    kinds = {r["name"]: r.get("type") for r in plugins.describe(p)}
    assert kinds["Cutoff"] is None          # continuous
    assert kinds["Mode"] == "choice"


def test_choice_threshold_is_a_count_a_human_could_pick_from():
    p = _Plugin({"a": _Param("X", discrete=True, steps=plugins._MAX_CHOICE_STEPS + 1)})
    assert plugins.describe(p)[0].get("type") is None

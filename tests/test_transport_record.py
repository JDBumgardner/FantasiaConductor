"""Regression guards for three faults that were invisible in the running app.

None of them showed a symptom a user could act on: a Qt slot that raises prints
to stderr and the menu item simply does nothing; a take recorded while the
transport ran landed a whole take-length late; and the mic kept running after
transport Stop with nothing on screen saying so.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np

from fantasia_core.document.model import Project
from fantasia_core.engine.record import Recorder

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCES = [p for p in list(ROOT.glob("ui/*.py")) + list(ROOT.rglob("fantasia_core/**/*.py"))]


def test_duration_is_a_property_and_is_never_called():
    """`Project.duration` is a property; `self.project.duration()` raises
    TypeError, which Qt swallows — Go to End and Zoom Fit were dead for that
    reason alone."""
    assert isinstance(type(Project()).__dict__["duration"], property)
    offenders = []
    for path in SOURCES:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "duration"):
                continue
            owner = node.func.value          # AudioPool.duration(path) is a real method
            named = (owner.attr if isinstance(owner, ast.Attribute)
                     else owner.id if isinstance(owner, ast.Name) else "")
            if named in ("project", "proj"):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, f"duration is a property, not a method: {offenders}"


def test_recorder_stamps_when_the_first_block_arrived():
    """`start()` cannot say where a take belongs: the device takes time to open
    and then buffers. The first block's arrival is the anchor correction."""
    r = Recorder(sample_rate=48000, channels=1)
    assert r.first_block_time == 0.0 and r.input_latency == 0.0
    r._recording = True
    block = np.zeros((256, 1), dtype="float32")
    r._callback(block, 256, None, None)
    first = r.first_block_time
    assert first > 0.0
    r._callback(block, 256, None, None)
    assert r.first_block_time == first, "only the FIRST block stamps the time"


def _source(name: str) -> str:
    return (ROOT / name).read_text()


def test_transport_stop_ends_a_running_take_but_export_and_load_do_not():
    """Stop is the only path that may end a take: `_on_export` would fold it
    into the export and `_load_project` would dispatch a clip into a project
    about to be discarded."""
    mw = _source("ui/main_window.py")
    tree = ast.parse(mw)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "_halt_transport" in fns, "the take-preserving halt is gone"
    on_stop = ast.unparse(fns["_on_stop"])
    assert "_halt_transport" in on_stop and "_stop_record" in on_stop
    # and it must run the take AFTER the engine is down
    assert on_stop.index("_halt_transport") < on_stop.index("self._stop_record")
    for caller in ("_on_export", "_load_project"):
        body = ast.unparse(fns[caller])
        assert "_halt_transport" in body, f"{caller} must not end a take"
        assert "self._on_stop()" not in body, f"{caller} must not end a take"


def test_the_take_is_anchored_where_recording_started():
    """Reading `timeline.playhead` in `_stop_record` yields the playhead at the
    moment recording STOPPED — `_on_tick` keeps it live — so the clip landed one
    take-length late."""
    mw = _source("ui/main_window.py")
    tree = ast.parse(mw)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "_rec_anchor_pos" in ast.unparse(fns["_start_record"])
    stop = ast.unparse(fns["_stop_record"])
    assert "_rec_anchor_pos" in stop
    assert "start = self.timeline.playhead" not in stop


def test_the_input_menu_cannot_reinitialise_portaudio_under_a_live_stream():
    """`sd._terminate(); sd._initialize()` with a stream open renumbers devices,
    which is what made a freshly-picked device silently do nothing."""
    fns = {n.name: n for n in ast.walk(ast.parse(_source("ui/main_window.py")))
           if isinstance(n, ast.FunctionDef)}
    body = ast.unparse(fns["_populate_input_devices"])
    assert "has_stream" in body and "release_device" in body


def test_tune_toward_forwards_the_devices_the_caller_chose():
    """`add` and `use` are read by the runner and sent by the dialog; leaving
    them out of the spec silently searched the existing inserts every time."""
    fns = {n.name: n for n in ast.walk(ast.parse(_source("ui/main_window.py")))
           if isinstance(n, ast.FunctionDef)}
    body = ast.unparse(fns["_agent_tune_toward"])
    assert "'add'" in body and "'use'" in body, "the dialog's device choice is dropped"

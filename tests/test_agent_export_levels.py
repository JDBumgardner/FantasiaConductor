"""The two tools that building an actual song showed were missing.

Neither existed while diagnosing a clipping complaint: rendering anything to a file meant hand-writing an
offline bounce script, and reading levels meant solo-bouncing twelve tracks by hand to discover they summed to
+3.8 dBFS. These guard the wiring and, more importantly, the failure that produced today's other bug — a schema
that advertises arguments the handler never reads.
"""

from __future__ import annotations

import ast
import pathlib

from fantasia_core.agent.tools import AgentTools

ROOT = pathlib.Path(__file__).resolve().parent.parent
MW = (ROOT / "ui" / "main_window.py").read_text()
TREE = ast.parse(MW)
FNS = {n.name: n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef)}
NEW = {"export_audio": "_agent_export_audio", "track_levels": "_agent_track_levels"}


def _schema(name):
    for d in AgentTools(None).definitions():          # definitions() touches nothing on self
        if d["name"] == name:
            return d
    raise AssertionError(f"{name} is not in AgentTools.definitions()")


def test_both_tools_are_declared():
    for name in NEW:
        d = _schema(name)
        assert d["description"].strip(), f"{name} needs a description — it is all the caller gets"
        assert d["input_schema"]["type"] == "object"


def test_every_documented_argument_is_actually_read():
    """The bug this is here for: `add` and `use` were in the tune_toward schema, sent by the dialog, read by
    the runner — and silently dropped when the job spec was built, so the choice did nothing for weeks."""
    for tool, handler in NEW.items():
        props = _schema(tool)["input_schema"].get("properties") or {}
        body = ast.unparse(FNS[handler])
        for arg in props:
            assert f"'{arg}'" in body or f'"{arg}"' in body, \
                f"{tool} documents {arg!r} but {handler} never reads it"


def test_both_are_dispatched_on_the_ui_thread():
    body = ast.unparse(FNS["_on_agent_tool"])
    for tool, handler in NEW.items():
        assert f"'{tool}'" in body and handler in body, f"{tool} is declared but never dispatched"


def test_export_pairs_the_format_with_a_sample_format_the_writer_accepts():
    """soundfile takes the container from the extension; mp3 only accepts MPEG_LAYER_III and ogg only VORBIS,
    which is why the export dialog pairs them rather than offering two free lists."""
    body = ast.unparse(FNS["_agent_export_audio"])
    assert "_EXPORT_FORMATS" in body, "the ext/subtype pairing must come from the dialog's own table"


def test_neither_tool_runs_over_a_live_take():
    """Exporting halts the transport, and _halt_transport deliberately leaves a take running — so an export
    mid-take would orphan the mic. Measuring would fight the recorder for the CPU."""
    for handler in NEW.values():
        assert "is_recording" in ast.unparse(FNS[handler]), f"{handler} must refuse during a take"


def test_measuring_always_restores_mute_and_solo():
    """It solos every track in turn; leaving that behind would silently mute the user's song."""
    fn = FNS["_agent_track_levels"]
    tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try) and n.finalbody]
    assert tries, "the solo loop must sit in try/finally"
    assert any("x.mute" in ast.unparse(t) or "t.mute" in ast.unparse(t) for t in tries)
    finallies = ["\n".join(ast.unparse(st) for st in t.finalbody) for t in tries]
    assert any("t.mute" in f and "t.solo" in f and "saved" in f for f in finallies), \
        "mute/solo must be restored in the finally, not after the loop"


def test_levels_indexes_the_worst_moment_by_frame_not_by_flat_index():
    """np.argmax over an (n, 2) array is a flat index: it would report a time up to 2x wrong and pull each
    track's contribution from the wrong sample."""
    body = ast.unparse(FNS["_agent_track_levels"])
    assert "max(axis=1)" in body, "the envelope must collapse the channels before argmax"


def test_dbfs_is_computed_directly_rather_than_through_the_meter_helper():
    """levels.amp_to_db floors at -60 and caps at +12 for a meter widget; a silent stem would read '-60.0' and
    an over-full-scale sum would read '+12.0'."""
    for handler in NEW.values():
        body = ast.unparse(FNS[handler])
        assert "amp_to_db" not in body
        assert "math.log10" in body

"""`tune_toward`: push a track's sound toward a description, as a ladder of proposals.

The search (differentiable twins of the track's inserts and, for a Vital track the twin can model, of the patch;
CLAP as the ear; fixed-distance stops with artefact penalties) runs in its own process — torch and CLAP do not
belong in the app's process on an 8 GB machine — via ``experiments/text2fx/tune.py``. This module owns the jobs:
start one, read its progress, hand back the result. Nothing is applied here; the agent or the user picks a stop and
``apply`` turns its exported parameters into ordinary undoable edits.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Dict, Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNNER = ROOT / "experiments" / "text2fx" / "tune.py"


def _python() -> str:
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


class Job:
    def __init__(self, job_id: str, spec: dict, out_dir: str) -> None:
        self.id, self.spec, self.out_dir = job_id, spec, out_dir
        self.events: list = []
        self.status = "starting"        # starting | running | done | error
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.started = time.time()
        self.proc: Optional[subprocess.Popen] = None

    def summary(self) -> dict:
        prog = [e["message"] for e in self.events if e.get("event") == "progress"]
        route = next((e for e in self.events if e.get("event") == "route"), {})
        out = {"job_id": self.id, "status": self.status, "text": self.spec.get("text"),
               "track_id": self.spec.get("track_id"), "elapsed_s": round(time.time() - self.started),
               "route": route.get("route"), "coverage": route.get("coverage", []), "stops_done": len(prog), "progress": prog[-3:]}
        if self.error:
            out["error"] = self.error
        if self.result:
            out["stops"] = [{k: v for k, v in s.items() if k not in ("export", "describe")} for s in self.result["stops"]]
            out["frozen"] = self.result.get("frozen", [])
        return out


_JOBS: Dict[str, Job] = {}
_LOCK = threading.Lock()


def start(spec: dict) -> Job:
    """spec: the runner's job (see experiments/text2fx/tune.py) plus track_id. Returns immediately; poll ``status``."""
    job_id = uuid.uuid4().hex[:8]
    out_dir = spec.get("out_dir") or os.path.join(tempfile.gettempdir(), "fantasia_tune", job_id)
    os.makedirs(out_dir, exist_ok=True)
    spec = {**spec, "out_dir": out_dir}
    job = Job(job_id, spec, out_dir)
    with open(os.path.join(out_dir, "job.json"), "w") as f:
        json.dump(spec, f, indent=1)
    if not RUNNER.exists():
        job.status, job.error = "error", f"runner not found at {RUNNER}"
        return job
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    job.proc = subprocess.Popen([_python(), str(RUNNER), os.path.join(out_dir, "job.json")], stdout=subprocess.PIPE,
                                stderr=open(os.path.join(out_dir, "stderr.log"), "w"), text=True, cwd=str(ROOT), env=env)
    with _LOCK:
        _JOBS[job_id] = job
    threading.Thread(target=_pump, args=(job,), daemon=True).start()
    return job


def _pump(job: Job) -> None:
    job.status = "running"
    for line in job.proc.stdout:            # one JSON event per line
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        job.events.append(ev)
        if ev.get("event") == "done":
            try:
                job.result = json.load(open(ev["result"]))
            except Exception as exc:  # noqa: BLE001
                job.error = f"result unreadable: {exc}"
        elif ev.get("event") == "error":
            job.error = ev.get("message", "error") + (" — " + "; ".join(ev.get("coverage", [])) if ev.get("coverage") else "")
    code = job.proc.wait()
    if job.result is not None and not job.error:
        job.status = "done"
    else:
        job.status = "error"
        if not job.error:
            tail = ""
            try:
                tail = open(os.path.join(job.out_dir, "stderr.log")).read()[-800:]
            except OSError:
                pass
            job.error = f"runner exited with {code}: {tail.strip()[-400:]}"


def get(job_id: str) -> Optional[Job]:
    with _LOCK:
        return _JOBS.get(job_id)


def stop_export(job_id: str, stop: int) -> Optional[dict]:
    """The exported parameters of one stop: {insert_id: params, ..., "vital": {...}} or None."""
    job = get(job_id)
    if job is None or job.result is None:
        return None
    rows = [s for s in job.result["stops"] if s["stop"] == int(stop)]
    return rows[0].get("export") if rows else None

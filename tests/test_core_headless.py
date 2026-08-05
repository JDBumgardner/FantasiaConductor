"""Guardrail: the headless core must never import Qt.

Keeping ``fantasia_core`` UI-agnostic is what lets a future TS/localhost
frontend bind to the same core over a local API without a rewrite.
"""

from __future__ import annotations

import pathlib

CORE = pathlib.Path(__file__).resolve().parent.parent / "fantasia_core"


def test_core_has_no_qt_imports():
    offenders = []
    for path in CORE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "PySide6" in text or "import PyQt" in text:
            offenders.append(str(path.relative_to(CORE.parent)))
    assert not offenders, f"Qt imports found in headless core: {offenders}"

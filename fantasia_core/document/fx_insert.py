"""FX inserts — one device in a channel's signal chain.

The numbered EQ *bands* live inside an ``eq`` insert's params. The chain itself
is an ordered list of :class:`FxInsert` on :class:`Track.fx` (and on Master).
Identity is ``id`` (stable across save/load, reorder, bypass) so commands, the
UI, and agents can address "this compressor" rather than "the third dict".

Stored as a dataclass in memory; JSON is the ``to_dict`` form. Older project
files that only have ``{type, params}`` are upgraded on load (ids minted from
the project's monotonic counter).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FxInsert:
    """One device in a channel's insert graph."""

    id: str
    type: str
    params: dict = field(default_factory=dict)
    bypassed: bool = False

    def to_dict(self) -> dict[str, Any]:
        params = dict(self.params)
        bands = params.get("bands")
        if isinstance(bands, list):
            params["bands"] = [dict(b) for b in bands]
        return {
            "id": self.id,
            "type": self.type,
            "bypassed": bool(self.bypassed),
            "params": params,
        }


def as_insert(raw: Any, new_id: str = "") -> FxInsert:
    """Coerce a dict or :class:`FxInsert` into an insert. Empty id is filled later."""
    if isinstance(raw, FxInsert):
        if not raw.id and new_id:
            raw.id = new_id
        if isinstance(raw.params, dict):
            bands = raw.params.get("bands")
            if isinstance(bands, list):
                raw.params = {**raw.params, "bands": [dict(b) for b in bands]}
        return raw
    data = dict(raw or {})
    params = dict(data.get("params") or {})
    bands = params.get("bands")
    if isinstance(bands, list):
        params["bands"] = [dict(b) for b in bands]
    return FxInsert(
        id=str(data.get("id") or new_id or ""),
        type=str(data.get("type") or "gain"),
        params=params,
        bypassed=bool(data.get("bypassed", False)),
    )


def as_dict(raw: Any) -> dict[str, Any]:
    """DSP / JSON view of an insert. Always has type, params, bypassed, id."""
    if isinstance(raw, FxInsert):
        return raw.to_dict()
    data = dict(raw or {})
    params = dict(data.get("params") or {})
    bands = params.get("bands")
    if isinstance(bands, list):
        params["bands"] = [dict(b) for b in bands]
    return {
        "id": str(data.get("id") or ""),
        "type": str(data.get("type") or ""),
        "bypassed": bool(data.get("bypassed", False)),
        "params": params,
    }


def copy_insert(raw: Any) -> FxInsert:
    """Deep-ish copy so undo / merge never alias nested EQ bands."""
    src = as_insert(raw)
    params = dict(src.params)
    bands = params.get("bands")
    if isinstance(bands, list):
        params["bands"] = [dict(b) for b in bands]
    return FxInsert(id=src.id, type=src.type, params=params, bypassed=src.bypassed)


def insert_type(raw: Any) -> str:
    if isinstance(raw, FxInsert):
        return raw.type
    return str((raw or {}).get("type") or "")


def insert_id(raw: Any) -> str:
    if isinstance(raw, FxInsert):
        return raw.id
    return str((raw or {}).get("id") or "")


def insert_bypassed(raw: Any) -> bool:
    if isinstance(raw, FxInsert):
        return bool(raw.bypassed)
    return bool((raw or {}).get("bypassed", False))


def mint_missing_ids(chain: list, new_id: Callable[[], str]) -> list[FxInsert]:
    """Ensure every insert is an :class:`FxInsert` with a non-empty id."""
    out: list[FxInsert] = []
    for raw in chain or []:
        ins = copy_insert(raw)
        if not ins.id:
            ins.id = new_id()
        out.append(ins)
    return out

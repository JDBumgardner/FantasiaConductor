"""FX inserts — devices in a channel's directed signal graph.

The numbered EQ *bands* live inside an ``eq`` insert's params. The chain itself
is an ordered list of :class:`FxInsert` on :class:`Track.fx` (and on Master),
plus an optional list of :class:`FxWire` on ``Track.fx_wires``.

Identity is ``id`` (stable across save/load, reorder, bypass) so commands, the
UI, and agents can address "this compressor" rather than "the third dict".

An empty ``fx_wires`` list means the implicit serial graph
``in → fx[0] → fx[1] → … → out``. Explicit wires allow branching and merging.
Removing a node reconnects its predecessors to its successors so the graph
never has a hole.

Stored as dataclasses in memory; JSON is the ``to_dict`` form. Older project
files that only have ``{type, params}`` are upgraded on load (ids minted from
the project's monotonic counter).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence


# Sentinel node ids. Insert ids are ``fxN``, so these never collide.
SOURCE = "in"   # clip audio / instrument output
OUT = "out"     # channel fader → mixer / master


@dataclass
class FxInsert:
    """One device in a channel's insert graph."""

    id: str
    type: str
    params: dict = field(default_factory=dict)
    bypassed: bool = False
    x: float = 0.0  # node-editor position; 0,0 = auto-layout
    y: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        params = dict(self.params)
        bands = params.get("bands")
        if isinstance(bands, list):
            params["bands"] = [dict(b) for b in bands]
        data = {
            "id": self.id,
            "type": self.type,
            "bypassed": bool(self.bypassed),
            "params": params,
        }
        if self.x or self.y:
            data["x"] = float(self.x)
            data["y"] = float(self.y)
        return data


@dataclass
class FxWire:
    """Directed audio connection: ``src`` output feeds ``dst`` input."""

    src: str
    dst: str

    def to_dict(self) -> dict[str, str]:
        return {"src": self.src, "dst": self.dst}

    def key(self) -> tuple[str, str]:
        return (self.src, self.dst)


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
        x=float(data.get("x") or 0.0),
        y=float(data.get("y") or 0.0),
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
    out = {
        "id": str(data.get("id") or ""),
        "type": str(data.get("type") or ""),
        "bypassed": bool(data.get("bypassed", False)),
        "params": params,
    }
    if data.get("x") or data.get("y"):
        out["x"] = float(data.get("x") or 0.0)
        out["y"] = float(data.get("y") or 0.0)
    return out


def copy_insert(raw: Any) -> FxInsert:
    """Deep-ish copy so undo / merge never alias nested EQ bands."""
    src = as_insert(raw)
    params = dict(src.params)
    bands = params.get("bands")
    if isinstance(bands, list):
        params["bands"] = [dict(b) for b in bands]
    return FxInsert(
        id=src.id, type=src.type, params=params, bypassed=src.bypassed,
        x=src.x, y=src.y,
    )


def as_wire(raw: Any) -> FxWire:
    if isinstance(raw, FxWire):
        return FxWire(src=raw.src, dst=raw.dst)
    data = dict(raw or {})
    return FxWire(src=str(data.get("src") or ""), dst=str(data.get("dst") or ""))


def copy_wires(wires: Optional[Sequence] = None) -> list[FxWire]:
    return [as_wire(w) for w in (wires or []) if as_wire(w).src and as_wire(w).dst]


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


def serial_wires(chain: Sequence) -> list[FxWire]:
    """``in → n0 → n1 → … → out`` for an ordered insert list."""
    ids = [insert_id(n) for n in (chain or []) if insert_id(n)]
    if not ids:
        return [FxWire(SOURCE, OUT)]
    wires = [FxWire(SOURCE, ids[0])]
    for a, b in zip(ids, ids[1:]):
        wires.append(FxWire(a, b))
    wires.append(FxWire(ids[-1], OUT))
    return wires


def effective_wires(chain: Sequence, wires: Optional[Sequence] = None) -> list[FxWire]:
    """Explicit wires, or the implicit serial graph when none are stored."""
    stored = copy_wires(wires)
    if stored:
        return stored
    return serial_wires(chain)


def sanitize_wires(chain: Sequence, wires: Sequence) -> list[FxWire]:
    """Drop dangling ends; keep SOURCE/OUT and live insert ids."""
    live = {insert_id(n) for n in (chain or []) if insert_id(n)}
    live.add(SOURCE)
    live.add(OUT)
    out: list[FxWire] = []
    seen: set[tuple[str, str]] = set()
    for raw in wires or []:
        w = as_wire(raw)
        if w.src not in live or w.dst not in live or w.src == w.dst:
            continue
        key = w.key()
        if key in seen:
            continue
        seen.add(key)
        out.append(w)
    return out


def reachable(wires: Sequence[FxWire], start: str) -> set[str]:
    adj: dict[str, list[str]] = {}
    for w in wires:
        adj.setdefault(w.src, []).append(w.dst)
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def would_cycle(wires: Sequence[FxWire], src: str, dst: str) -> bool:
    """True if adding src→dst would close a loop (dst can already reach src)."""
    if src == dst:
        return True
    return src in reachable(wires, dst)


def connect_wire(wires: Sequence[FxWire], src: str, dst: str) -> Optional[list[FxWire]]:
    """Return a new wire list with src→dst, or None if the link is illegal."""
    if not src or not dst or src == dst:
        return None
    current = copy_wires(wires)
    if any(w.src == src and w.dst == dst for w in current):
        return current
    if would_cycle(current, src, dst):
        return None
    current.append(FxWire(src, dst))
    return current


def disconnect_wire(wires: Sequence[FxWire], src: str, dst: str) -> list[FxWire]:
    return [w for w in copy_wires(wires) if not (w.src == src and w.dst == dst)]


def rewire_remove(wires: Sequence[FxWire], node_id: str) -> list[FxWire]:
    """Drop ``node_id`` and bridge each predecessor to each successor."""
    current = copy_wires(wires)
    preds = [w.src for w in current if w.dst == node_id]
    succs = [w.dst for w in current if w.src == node_id]
    kept = [w for w in current if w.src != node_id and w.dst != node_id]
    seen = {w.key() for w in kept}
    for p in preds:
        for s in succs:
            if p == s or p == node_id or s == node_id:
                continue
            key = (p, s)
            if key in seen:
                continue
            if would_cycle(kept, p, s):
                continue
            kept.append(FxWire(p, s))
            seen.add(key)
    return kept


def splice_before_out(wires: Sequence[FxWire], new_id: str) -> list[FxWire]:
    """Append ``new_id`` just before OUT (serial add on an explicit graph)."""
    current = copy_wires(wires)
    hooked = False
    out: list[FxWire] = []
    for w in current:
        if w.dst == OUT:
            out.append(FxWire(w.src, new_id))
            hooked = True
        else:
            out.append(w)
    if not hooked:
        out.append(FxWire(SOURCE, new_id))
    out.append(FxWire(new_id, OUT))
    return out


def is_serial(chain: Sequence, wires: Optional[Sequence] = None) -> bool:
    """True when the effective graph is a single path matching list order."""
    ids = [insert_id(n) for n in (chain or []) if insert_id(n)]
    effective = effective_wires(chain, wires)
    expected = {w.key() for w in serial_wires(chain)}
    got = {w.key() for w in effective}
    if got != expected:
        return False
    # Each insert has exactly one in and one out.
    for nid in ids:
        ins = sum(1 for w in effective if w.dst == nid)
        outs = sum(1 for w in effective if w.src == nid)
        if ins != 1 or outs != 1:
            return False
    return True


def topo_order(chain: Sequence, wires: Optional[Sequence] = None) -> list[str]:
    """Kahn topo of insert ids (SOURCE/OUT omitted). Falls back to list order."""
    ids = [insert_id(n) for n in (chain or []) if insert_id(n)]
    idset = set(ids)
    effective = sanitize_wires(chain, effective_wires(chain, wires))
    indeg = {i: 0 for i in idset}
    adj: dict[str, list[str]] = {i: [] for i in idset}
    for w in effective:
        if w.src in idset and w.dst in idset:
            adj[w.src].append(w.dst)
            indeg[w.dst] = indeg.get(w.dst, 0) + 1
        elif w.src == SOURCE and w.dst in idset:
            pass
    ready = [i for i in ids if indeg.get(i, 0) == 0]
    order: list[str] = []
    seen: set[str] = set()
    while ready:
        nid = ready.pop(0)
        if nid in seen:
            continue
        seen.add(nid)
        order.append(nid)
        for nxt in adj.get(nid, ()):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
    if len(order) != len(ids):
        return ids
    return order


def linear_order(chain: Sequence, wires: Optional[Sequence] = None) -> list[str]:
    """Left-to-right device order for the rack view (topo, SOURCE→OUT)."""
    return topo_order(chain, wires)


STOCK_FX = (
    ("eq", "Stock EQ"),
    ("reverb", "Reverb"),
    ("delay", "Delay"),
    ("compressor", "Compressor"),
    ("limiter", "Limiter"),
    ("gate", "Gate"),
    ("saturator", "Saturator"),
    ("distortion", "Distortion"),
    ("chorus", "Chorus"),
    ("lowpass", "Low Pass"),
    ("highpass", "High Pass"),
    ("gain", "Utility Gain"),
)

_STOCK_LABELS = {k: v for k, v in STOCK_FX}


def device_label(raw: Any) -> str:
    kind = insert_type(raw)
    if kind == "vst":
        params = raw.params if isinstance(raw, FxInsert) else (raw or {}).get("params") or {}
        return str(params.get("name") or params.get("path") or "Plugin")
    return _STOCK_LABELS.get(kind, kind.replace("_", " ").title() or "FX")

"""Preset catalog — saved plugin patches, recallable by name.

A plugin's *parameters* are only half of its state. Vital's modulation routings,
for instance, are not automatable parameters at all: they live inside the patch
and are created by dragging in the plugin's own window. So an agent can tune a
knob but cannot invent a routing.

This module closes that gap from the other side. A patch built by hand once is
snapshotted here as its opaque state blob and can then be recalled onto any
track by name. The agent does not need to understand the blob — only to name it.

Stored under ``.fantasia_cache/presets/`` as ``<slug>.vstpreset`` beside a JSON
sidecar, mirroring :mod:`fantasia_core.voices`. Headless: no Qt in here.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import time
from typing import List, Optional


def catalog_dir() -> pathlib.Path:
    d = (pathlib.Path(os.environ["FANTASIA_PRESETS"])
         if os.environ.get("FANTASIA_PRESETS")
         else pathlib.Path.cwd() / ".fantasia_cache" / "presets")
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclasses.dataclass
class Preset:
    slug: str
    name: str
    plugin: str
    path: str
    bytes: int = 0
    created: float = 0.0
    note: str = ""


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return s or "preset"


def list_presets(plugin: Optional[str] = None) -> List[Preset]:
    """Every saved preset, newest first; optionally only one plugin's."""
    out = []
    for js in catalog_dir().glob("*.json"):
        try:
            d = json.loads(js.read_text())
            blob = catalog_dir() / f"{js.stem}.vstpreset"
            if not blob.exists():
                continue                      # sidecar without payload: skip
            if plugin and d.get("plugin") != plugin:
                continue
            out.append(Preset(slug=js.stem, name=d.get("name", js.stem),
                              plugin=d.get("plugin", ""), path=str(blob),
                              bytes=int(d.get("bytes", 0)),
                              created=float(d.get("created", 0.0)),
                              note=d.get("note", "")))
        except Exception:                     # noqa: BLE001 — one bad sidecar
            continue                          # shouldn't hide the rest
    return sorted(out, key=lambda p: -p.created)


def get(slug: str) -> Optional[Preset]:
    return next((p for p in list_presets() if p.slug == slug), None)


def save(plugin: str, name: str, data: bytes, *, note: str = "") -> Preset:
    """Write ``data`` (the plugin's state blob) as a named preset."""
    if not data:
        raise ValueError("plugin returned no state to save")
    slug = slugify(name)
    d = catalog_dir()
    (d / f"{slug}.vstpreset").write_bytes(data)
    meta = {"name": str(name), "plugin": str(plugin), "bytes": len(data),
            "created": time.time(), "note": str(note)}
    (d / f"{slug}.json").write_text(json.dumps(meta, indent=2))
    return Preset(slug=slug, name=str(name), plugin=str(plugin),
                  path=str(d / f"{slug}.vstpreset"), bytes=len(data),
                  created=meta["created"], note=str(note))


def read_bytes(slug: str) -> Optional[bytes]:
    p = get(slug)
    return pathlib.Path(p.path).read_bytes() if p else None


def delete(slug: str) -> bool:
    d = catalog_dir()
    hit = False
    for suffix in (".vstpreset", ".json"):
        f = d / f"{slug}{suffix}"
        if f.exists():
            f.unlink()
            hit = True
    return hit


# ---- Vital .vital -> .vstpreset ----------------------------------------
JSON_START = 224  # where the JSON begins inside a JUCE VST3 container


def splice_vital(template: bytes, patch: dict) -> bytes:
    """Rebuild a VST3 preset blob around a different Vital patch.

    Vital ships presets as ``.vital`` files, which are plain JSON, but pedalboard
    only loads ``.vstpreset``. A JUCE VST3 preset is a small header, one
    component chunk holding that same JSON, and a trailing chunk list::

        [0:40]    'VST3', version, class id
        [40:48]   int64 offset of the chunk list
        [48:...]  component chunk: JUCE preamble, the JSON, 32 bytes of padding
        [...:]    'List', count, ('Comp', offset, size), ('Cont', offset, size)

    Three integers depend on the JSON's length, so swapping in a patch of any
    size means shifting them by the delta. That is all this does — no padding,
    so a patch may be larger or smaller than the one already loaded.

    ``template`` is any preset blob from the same plugin; its current
    ``preset_data`` does fine.
    """
    import struct

    if template[:4] != b"VST3":
        raise ValueError("template is not a VST3 preset blob")
    list_off = struct.unpack("<q", template[40:48])[0]
    dec = json.JSONDecoder()
    _, span = dec.raw_decode(template[JSON_START:].decode("utf-8", "replace"))
    json_end = JSON_START + span
    pad = template[json_end:list_off]          # component-chunk tail padding

    entries = []
    cur = list_off + 4
    count = struct.unpack("<i", template[cur:cur + 4])[0]
    cur += 4
    for _ in range(count):
        cid = template[cur:cur + 4]
        off, size = struct.unpack("<qq", template[cur + 4:cur + 20])
        entries.append([cid, off, size])
        cur += 20

    body = json.dumps(patch, separators=(",", ":")).encode()
    delta = len(body) - span

    for e in entries:                          # shift anything past the JSON
        if e[0] == b"Comp":
            e[2] += delta                      # the chunk that holds it grows
        elif e[1] >= list_off:
            e[1] += delta
    chunk_list = b"List" + struct.pack("<i", count)
    for cid, off, size in entries:
        chunk_list += cid + struct.pack("<qq", off, size)

    # The component chunk also carries a VST2-style FXB header, whose two length
    # fields are BIG-endian and both span the JSON. Miss these and the plugin
    # accepts the blob and silently keeps its old patch.
    preamble = bytearray(template[48:JSON_START])
    ccnk = preamble.find(b"CcnK")
    if ccnk < 0:
        raise ValueError("no CcnK header in the preset preamble")
    struct.pack_into(">i", preamble, ccnk + 4,
                     struct.unpack_from(">i", preamble, ccnk + 4)[0] + delta)
    struct.pack_into(">i", preamble, len(preamble) - 4,
                     struct.unpack_from(">i", preamble, len(preamble) - 4)[0] + delta)

    return (template[:40] + struct.pack("<q", list_off + delta)
            + bytes(preamble) + body + pad + chunk_list)


def read_vital_file(path: str) -> dict:
    return json.loads(pathlib.Path(path).expanduser().read_text())

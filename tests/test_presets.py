"""Preset catalog and the .vital -> .vstpreset splice.

The splice exists because pedalboard only loads .vstpreset while Vital ships
.vital JSON, and the JUCE container turns out to wrap that JSON verbatim. Five
length/offset fields depend on the JSON's size, two of them BIG-endian and
buried in a VST2 FXB header. Getting those wrong is not loud: the plugin accepts
the blob, reports success, and silently keeps its old patch. So these tests
assert on the rebuilt bytes rather than on a return value.
"""

from __future__ import annotations

import json
import struct

import pytest

from fantasia_core import presets


# ---- a synthetic container, same shape as Vital's -----------------------
def build_container(patch: dict) -> bytes:
    body = json.dumps(patch, separators=(",", ":")).encode()
    preamble = bytearray(176)
    preamble[0:4] = b"VstW"
    preamble[16:20] = b"CcnK"
    struct.pack_into(">i", preamble, 20, len(body) + 184)      # FXB byteSize
    preamble[24:28] = b"FBCh"
    preamble[32:36] = b"Vita"
    struct.pack_into(">i", preamble, 172, len(body) + 32)      # FXB chunkSize
    pad = b"\x00" * 17 + b"JUCEPrivateData"                    # 32 bytes
    list_off = 48 + len(preamble) + len(body) + len(pad)
    head = b"VST3" + struct.pack("<i", 1) + b"V" * 32 + struct.pack("<q", list_off)
    chunks = (b"List" + struct.pack("<i", 2)
              + b"Comp" + struct.pack("<qq", 48, list_off - 48)
              + b"Cont" + struct.pack("<qq", list_off, 0))
    return head + bytes(preamble) + body + pad + chunks


def parse(blob: bytes) -> dict:
    list_off = struct.unpack("<q", blob[40:48])[0]
    _, span = json.JSONDecoder().raw_decode(blob[224:].decode("utf-8", "replace"))
    out = {"list_off": list_off, "json_len": span,
           "ccnk": struct.unpack(">i", blob[68:72])[0],
           "chunksz": struct.unpack(">i", blob[220:224])[0]}
    cur = list_off + 8
    for _ in range(2):
        cid = blob[cur:cur + 4]
        off, size = struct.unpack("<qq", blob[cur + 4:cur + 20])
        out[cid.decode()] = (off, size)
        cur += 20
    return out


SMALL = {"author": "t", "settings": {"osc_1_transpose": -27.0}}
BIG = {"author": "t", "settings": {"osc_1_transpose": 0.0, "blob": "x" * 5000}}


def test_catalog_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("FANTASIA_PRESETS", str(tmp_path))
    p = presets.save("Vital", "DT Kick", b"VST3payload", note="pitch drop on mod 1")
    assert p.slug == "dt_kick"
    assert presets.read_bytes("dt_kick") == b"VST3payload"
    assert [x.slug for x in presets.list_presets()] == ["dt_kick"]
    assert presets.list_presets(plugin="Other") == []      # filters by plugin
    assert presets.get("dt_kick").note.startswith("pitch drop")
    assert presets.delete("dt_kick") and presets.list_presets() == []


def test_save_rejects_empty_state(tmp_path, monkeypatch):
    monkeypatch.setenv("FANTASIA_PRESETS", str(tmp_path))
    with pytest.raises(ValueError):
        presets.save("Vital", "nope", b"")


@pytest.mark.parametrize("target", [SMALL, BIG], ids=["shrink", "grow"])
def test_splice_updates_every_size_field(target):
    """A patch of any size must leave a self-consistent container."""
    tpl = build_container({"author": "t", "settings": {"blob": "y" * 2000}})
    out = presets.splice_vital(tpl, target)
    got = parse(out)
    body_len = len(json.dumps(target, separators=(",", ":")).encode())

    assert got["json_len"] == body_len            # JSON swapped, not padded
    assert got["list_off"] == len(out) - 48       # chunk list still at the end
    assert got["ccnk"] == body_len + 184          # BIG-endian FXB byteSize
    assert got["chunksz"] == body_len + 32        # BIG-endian FXB chunkSize
    assert got["Comp"] == (48, got["list_off"] - 48)
    assert got["Cont"] == (got["list_off"], 0)


def test_spliced_json_is_the_new_patch():
    tpl = build_container({"author": "t", "settings": {"blob": "y" * 2000}})
    out = presets.splice_vital(tpl, SMALL)
    _, span = json.JSONDecoder().raw_decode(out[224:].decode("utf-8", "replace"))
    assert json.loads(out[224:224 + span]) == SMALL


def test_splice_rejects_foreign_blob():
    with pytest.raises(ValueError):
        presets.splice_vital(b"NOTVST3" + b"\x00" * 400, SMALL)


def test_read_vital_file(tmp_path):
    f = tmp_path / "Kick.vital"
    f.write_text(json.dumps(SMALL))
    assert presets.read_vital_file(str(f)) == SMALL

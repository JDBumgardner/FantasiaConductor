"""Sound-search plumbing — ingest, dedup, text→audio, audio→audio.

CLAP is stubbed with a deterministic fake embedder so the test proves the store
and service wiring without downloading the ~2GB model. The real model is verified
manually / in a background run.
"""

from __future__ import annotations

import numpy as np
import pytest

import fantasia_core.search.embed as embed
from fantasia_core.search import SearchService

lancedb = pytest.importorskip("lancedb")


def _fake_vec(key: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(key)) % (2**32))
    v = rng.standard_normal(embed.DIM).astype("float32")
    return v / (np.linalg.norm(v) or 1.0)


@pytest.fixture
def stub_clap(monkeypatch):
    monkeypatch.setattr(embed, "available", lambda: True)
    monkeypatch.setattr(embed, "embed_texts",
                        lambda texts: np.stack([_fake_vec("T:" + t) for t in texts]))
    monkeypatch.setattr(embed, "embed_audio_file",
                        lambda path: _fake_vec("A:" + path))


def _wavs(dirpath, n=3):
    import soundfile as sf

    paths = []
    for i in range(n):
        p = str(dirpath / f"snd_{i}.wav")
        sf.write(p, (np.random.default_rng(i).standard_normal(4410) * 0.1).astype("float32"), 44100)
        paths.append(p)
    return paths


def test_ingest_dedup_and_search(tmp_path, stub_clap):
    paths = _wavs(tmp_path)
    svc = SearchService(str(tmp_path / "lib.lancedb"))
    assert svc.available() and svc.count() == 0

    added = svc.ingest_paths(paths)
    assert added == len(paths)
    assert svc.count() == len(paths)

    # Re-ingesting the same files adds nothing.
    assert svc.ingest_paths(paths) == 0

    rows = svc.search_text("anything", k=len(paths))
    assert len(rows) == len(paths)
    assert all("path" in r and "score" in r and "tags" in r for r in rows)
    # Every ingested sound got auto-tags from the tag vocabulary.
    assert all(r["tags"] for r in rows)


def test_replace_name_swaps_old_take(tmp_path, stub_clap):
    paths = _wavs(tmp_path, n=2)
    svc = SearchService(str(tmp_path / "lib.lancedb"))
    name = "voice: hello there"
    # take 1, then a re-generation of the same text under a new file
    svc.ingest_tagged([{"path": paths[0], "name": name, "tags": ["voice"]}])
    svc.replace_name(name)
    svc.ingest_tagged([{"path": paths[1], "name": name, "tags": ["voice"]}])
    rows = svc.store.audio_names_paths()
    matches = [p for n, p in rows if n == name]
    assert len(matches) == 1
    assert matches[0].endswith("snd_1.wav")  # the newest take won


def test_audio_to_audio_self_match(tmp_path, stub_clap):
    paths = _wavs(tmp_path)
    svc = SearchService(str(tmp_path / "lib.lancedb"))
    svc.ingest_paths(paths)

    rows = svc.search_audio_file(paths[0], k=len(paths))
    # A clip is most similar to itself (cosine ~1.0 → score ~1.0).
    assert rows[0]["path"] == paths[0]
    assert rows[0]["score"] == pytest.approx(1.0, abs=1e-3)

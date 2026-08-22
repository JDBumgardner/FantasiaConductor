"""Reference-voice catalog: import, storage format, and clone/kokoro routing."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from fantasia_core import tts, voices


@pytest.fixture()
def catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("FANTASIA_VOICES", str(tmp_path / "voices"))
    return tmp_path


def _clip(path, seconds=3.0, sr=22050):
    t = np.arange(int(seconds * sr)) / sr
    y = (0.5 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    sf.write(str(path), y, sr, subtype="PCM_16")
    return str(path)


def test_add_voice_normalizes_to_the_codec_rate(catalog, tmp_path):
    """The catalog stores one format — 44.1k mono — whatever comes in."""
    v = voices.add_voice("Test", _clip(tmp_path / "in.wav", sr=22050), "hello there")
    audio, sr = sf.read(v.path, always_2d=True)
    assert sr == voices.REF_SR
    assert audio.shape[1] == 1
    assert abs(np.max(np.abs(audio)) - 0.95) < 0.02      # normalized


def test_long_clip_is_trimmed(catalog, tmp_path):
    v = voices.add_voice("Long", _clip(tmp_path / "l.wav", seconds=40.0), "words")
    assert v.seconds <= voices.MAX_REF_SECONDS + 0.01


def test_transcript_is_optional(catalog, tmp_path):
    """Chatterbox clones from audio alone, so requiring a transcript would be
    friction for nothing — but the field still round-trips for engines that use it."""
    v = voices.add_voice("No text", _clip(tmp_path / "n.wav"))
    assert v.ref_text == ""
    assert voices.get("no_text") is not None


def test_roundtrip_through_the_catalog(catalog, tmp_path):
    voices.add_voice("Alice Example", _clip(tmp_path / "a.wav"), "one two three",
                     tags=["alto"])
    got = voices.get("alice_example")
    assert got is not None
    assert got.name == "Alice Example"
    assert got.ref_text == "one two three"
    assert got.tags == ["alto"]
    assert len(voices.load_ref(got)) > 0
    assert voices.remove("alice_example")
    assert voices.get("alice_example") is None


def test_sidecar_without_audio_is_skipped(catalog, tmp_path):
    voices.add_voice("Ghost", _clip(tmp_path / "g.wav"), "boo")
    (voices.catalog_dir() / "ghost.wav").unlink()
    assert voices.get("ghost") is None      # not a crash, just absent


def test_ref_voice_selects_the_clone_backend(monkeypatch):
    """Passing a reference voice must route to the clone backend even with
    backend unset — Kokoro cannot clone, so the intent is unambiguous."""
    seen = {}

    def fake_clone(text, ref_voice, exaggeration, temperature):
        seen["clone"] = (text, ref_voice)
        return np.zeros(10, dtype=np.float32), tts.CLONE_SR

    def fake_kokoro(text, voice, speed):
        seen["kokoro"] = text
        return np.zeros(10, dtype=np.float32), tts.KOKORO_SR

    monkeypatch.setattr(tts, "_synth_clone", fake_clone)
    monkeypatch.setattr(tts, "_synth_kokoro", fake_kokoro)

    tts.synthesize("hi", ref_voice="someone")
    assert seen == {"clone": ("hi", "someone")}

    seen.clear()
    tts.synthesize("hi")
    assert "kokoro" in seen and "clone" not in seen


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        tts.synthesize("hi", backend="tortoise")


def test_clone_without_a_reference_voice_explains_itself(catalog):
    """The clone backend has no built-in voices at all — the error should say so
    rather than surfacing a None deref from deep in mlx-audio."""
    with pytest.raises(ValueError, match="no built-in voices"):
        tts._synth_clone("hi", "nope", 0.0, 0.8)


def test_syllable_cache_avoids_a_second_synthesis(tmp_path, monkeypatch):
    """Singing calls synthesize once per syllable; at cloning speeds the
    repeats have to be lookups."""
    monkeypatch.chdir(tmp_path)
    tts._MEM.clear()
    calls = []

    def fake_kokoro(text, voice, speed):
        calls.append(text)
        return np.full(64, 0.5, dtype=np.float32), tts.KOKORO_SR

    monkeypatch.setattr(tts, "_synth_kokoro", fake_kokoro)
    a, _ = tts.synthesize("la", cache=True)
    b, _ = tts.synthesize("la", cache=True)
    assert calls == ["la"]                       # second call never synthesized
    assert np.allclose(a, b)

    tts._MEM.clear()                             # now force the disk path
    c, sr = tts.synthesize("la", cache=True)
    assert calls == ["la"]
    assert np.allclose(a, c) and sr == tts.KOKORO_SR

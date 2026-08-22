"""Voice conversion: argument contract, chunking/crossfade, and catalog lookup.

The model itself is ~500MB and runs at ~10x slower than real time, so these
stub seed_vc's API and assert on what we pass it — which is where the bugs
actually were (see the module docstring in fantasia_core.voiceconv).
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from fantasia_core import voiceconv, voices


@pytest.fixture()
def catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("FANTASIA_VOICES", str(tmp_path / "voices"))
    import soundfile as sf

    sr = voices.REF_SR
    y = (0.4 * np.sin(2 * np.pi * 150 * np.arange(sr) / sr)).astype(np.float32)
    sf.write(str(tmp_path / "ref.wav"), y, sr)
    voices.add_voice("Ref Voice", str(tmp_path / "ref.wav"), "hello")
    return tmp_path


class _FakeAudioData:
    def __init__(self, samples, mel, dur, count, sr, meta):
        self.samples, self.sample_rate = samples, sr


@pytest.fixture()
def fake_api(monkeypatch):
    """Stub the seed_vc API and record every inference() call."""
    calls = []

    def inference(**kw):
        calls.append(kw)
        n = len(np.asarray(kw["source"].samples))
        return _FakeAudioData(np.zeros(n), None, 0, n, voiceconv._MODEL_SR, None)

    api = types.SimpleNamespace(
        AudioData=_FakeAudioData,
        inference=inference,
        create_v1_stream_state=lambda **kw: ("state", kw),
        get_audio_numpy=lambda ad: np.asarray(ad.samples, dtype=np.float32) / 32767.0,
    )
    monkeypatch.setattr(voiceconv, "_api", lambda: api)
    voiceconv.unload()
    return calls


def test_melody_is_not_transposed_by_default(catalog, fake_api):
    """auto_f0_adjust moves a scored vocal into the target's range — measured at
    more than an octave down. It must be opt-in."""
    voiceconv.convert(np.zeros(1000, dtype=np.float32), 44100, "ref_voice")
    assert fake_api[0]["auto_f0_adjust"] is False
    assert fake_api[0]["semi_tone_shift"] == 0


def test_fit_range_is_forwarded_when_asked(catalog, fake_api):
    voiceconv.convert(np.zeros(1000, dtype=np.float32), 44100, "ref_voice",
                      fit_range=True, semitones=-2)
    assert fake_api[0]["auto_f0_adjust"] is True
    assert fake_api[0]["semi_tone_shift"] == -2


def test_uses_the_only_working_inference_path(catalog, fake_api):
    """Without streaming=True/realtime=False the library loads a model whose f0
    extractor is None, and pitch-conditioned conversion raises."""
    voiceconv.convert(np.zeros(1000, dtype=np.float32), 44100, "ref_voice")
    kw = fake_api[0]
    assert kw["streaming"] is True and kw["realtime"] is False
    assert kw["f0_condition"] is True


def test_long_input_is_chunked_and_rejoined(catalog, fake_api):
    sr = 44100
    audio = np.zeros(int(25 * sr), dtype=np.float32)     # > CHUNK_SECONDS
    out, out_sr = voiceconv.convert(audio, sr, "ref_voice")
    assert len(fake_api) == 3                            # 25s / 10s -> 3 windows
    # Crossfades overlap, so the result must not grow with the window count.
    assert abs(len(out) - len(audio)) < sr
    assert out_sr == voiceconv._MODEL_SR


def test_short_input_is_a_single_pass(catalog, fake_api):
    voiceconv.convert(np.zeros(4410, dtype=np.float32), 44100, "ref_voice")
    assert len(fake_api) == 1


def test_stereo_is_mixed_to_mono(catalog, fake_api):
    voiceconv.convert(np.zeros((2000, 2), dtype=np.float32), 44100, "ref_voice")
    assert np.asarray(fake_api[0]["source"].samples).ndim == 1


def test_empty_audio_returns_empty(catalog, fake_api):
    out, _ = voiceconv.convert(np.zeros(0, dtype=np.float32), 44100, "ref_voice")
    assert len(out) == 0
    assert not fake_api


def test_unknown_voice_is_named_in_the_error(catalog, fake_api):
    with pytest.raises(ValueError, match="nope"):
        voiceconv.convert(np.zeros(100, dtype=np.float32), 44100, "nope")


def test_model_state_is_reused_across_calls(catalog, fake_api):
    """Rebuilding it per call cost ~85s, which dominated the whole conversion."""
    built = []
    api = voiceconv._api()
    orig = api.create_v1_stream_state
    api.create_v1_stream_state = lambda **kw: (built.append(1), orig(**kw))[1]
    for _ in range(3):
        voiceconv.convert(np.zeros(1000, dtype=np.float32), 44100, "ref_voice")
    assert len(built) == 1

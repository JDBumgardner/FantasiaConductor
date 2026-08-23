"""Singing voice synthesis via DiffSinger (ONNX) — real singing, not stretched speech.

:mod:`fantasia_core.sing` makes a voice sing by speaking a syllable and pushing
it to the note's pitch with a vocoder. That is why held notes sound synthetic:
a spoken vowel stretched four times its length has none of the movement a real
sustained note has, and no amount of better alignment invents it.

DiffSinger synthesizes *at* the target pitch instead, from a voicebank trained on
singing. Nothing is stretched, so the problem does not arise. It is also faster
here than the older path — measured at roughly real time on CPU, including
loading all seven models.

A voicebank is an OpenUtau-format directory under ``.fantasia_cache/voicebanks``:

    <bank>/configs/dsconfig.yaml     rates, feature flags, speaker list
    <bank>/files/*.onnx              linguistic / dur / pitch / variance / acoustic
    <bank>/configs/embeds/*.emb      speaker embeddings (384 float32)
    <bank>/tgm_hifigan/*.onnx        the vocoder, mel + f0 -> waveform

The chain, all ONNX, no PyTorch::

    words -> phonemes           CMUdict (the bank ships phoneme maps, not words)
    linguistic-dur + dur     -> how long each phoneme lasts
    linguistic-pitch + pitch -> the f0 curve, conditioned on the actual notes
    linguistic-variance + variance -> tension
    acoustic                 -> mel
    vocoder                  -> audio

Headless: no Qt in here.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HOP = 512
SR = 44100
# Diffusion steps. The acoustic model carries the quality; the variance and
# pitch predictors are cheap and converge early.
ACOUSTIC_STEPS = 30
VARIANCE_STEPS = 10
REST = "SP"          # the phoneme every bank uses for silence


def banks_dir() -> pathlib.Path:
    d = pathlib.Path(os.environ.get("FANTASIA_VOICEBANKS", "")) if os.environ.get(
        "FANTASIA_VOICEBANKS") else pathlib.Path.cwd() / ".fantasia_cache" / "voicebanks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def available() -> bool:
    try:
        import cmudict  # noqa: F401
        import onnxruntime  # noqa: F401
        import yaml  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


@dataclasses.dataclass
class VoicebankInfo:
    slug: str
    name: str
    path: str
    speakers: List[str]
    ready: bool          # False when the vocoder is missing
    note: str = ""


def _read_yaml(path: pathlib.Path):
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def list_voicebanks() -> List[VoicebankInfo]:
    """Every installed voicebank, whether or not it can render yet."""
    out = []
    for d in sorted(banks_dir().iterdir()):
        cfg = d / "configs" / "dsconfig.yaml"
        if not cfg.is_dir() and cfg.exists():
            try:
                c = _read_yaml(cfg) or {}
            except Exception:  # noqa: BLE001
                continue
            spk = [str(s).split("/")[-1] for s in (c.get("speakers") or [])] or ["default"]
            voc = _vocoder_path(d)
            name = d.name
            char = d / "configs" / "character.yaml"
            if char.exists():
                try:
                    name = (_read_yaml(char) or {}).get("name") or name
                except Exception:  # noqa: BLE001
                    pass
            out.append(VoicebankInfo(
                d.name, str(name), str(d), spk, voc is not None,
                "" if voc else "vocoder missing — see the bank's vocoder.yaml"))
    return out


def _vocoder_path(root: pathlib.Path) -> Optional[pathlib.Path]:
    """The bank's vocoder, which OpenUtau installs alongside rather than inside."""
    try:
        vc = _read_yaml(root / "configs" / "dsvocoder" / "vocoder.yaml") or {}
        rel = vc.get("model")
        if rel:
            p = (root / "configs" / "dsvocoder" / rel).resolve()
            if p.exists():
                return p
    except Exception:  # noqa: BLE001
        pass
    hits = sorted(root.glob("**/*hifigan*.onnx"))
    return hits[0] if hits else None


class Voicebank:
    """A loaded voicebank. Sessions are opened lazily and kept."""

    def __init__(self, root: str) -> None:
        self.root = pathlib.Path(root)
        self.config = _read_yaml(self.root / "configs" / "dsconfig.yaml") or {}
        files = self.root / "files"
        self.phonemes: Dict[str, int] = json.load(open(files / "phonemes.json"))
        langs = files / "languages.json"
        self.languages: Dict[str, int] = json.load(open(langs)) if langs.exists() else {}
        self.lang_en = int(self.languages.get("en", 0))
        self.sr = int(self.config.get("sample_rate", SR))
        self.hop = int(self.config.get("hop_size", HOP))
        self.max_depth = float(self.config.get("max_depth", 1.0))
        self.use_tension = bool(self.config.get("use_tension_embed", False))
        self._sessions: Dict[str, object] = {}
        self._embeds: Dict[str, np.ndarray] = {}
        self.speakers = [str(s).split("/")[-1] for s in (self.config.get("speakers") or [])]

    # ---- resources ----------------------------------------------------
    def session(self, name: str):
        import onnxruntime as ort

        if name not in self._sessions:
            path = (_vocoder_path(self.root) if name == "vocoder"
                    else self.root / "files" / f"{name}.onnx")
            if path is None or not pathlib.Path(path).exists():
                raise FileNotFoundError(f"{name}.onnx missing from {self.root.name}")
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self._sessions[name] = ort.InferenceSession(
                str(path), opts, providers=["CPUExecutionProvider"])
        return self._sessions[name]

    def embed(self, speaker: Optional[str] = None) -> np.ndarray:
        speaker = speaker or (self.speakers[0] if self.speakers else "")
        if speaker not in self._embeds:
            p = self.root / "configs" / "embeds" / f"{speaker}.emb"
            if not p.exists():
                cands = sorted((self.root / "configs" / "embeds").glob("*.emb"))
                if not cands:
                    raise FileNotFoundError("no speaker embeddings in this voicebank")
                p = cands[0]
            self._embeds[speaker] = np.fromfile(p, dtype="<f4")
        return self._embeds[speaker]

    def token(self, phoneme: str) -> int:
        """Phoneme name to id, preferring the language-tagged form."""
        return int(self.phonemes.get(f"en/{phoneme}",
                                     self.phonemes.get(phoneme,
                                                       self.phonemes.get(REST, 2))))


_BANKS: Dict[str, Voicebank] = {}
_CMU = None


def load(slug_or_path: Optional[str] = None) -> Voicebank:
    """Load a voicebank by slug, path, or (default) the first installed one."""
    if slug_or_path and os.path.isdir(slug_or_path):
        root = slug_or_path
    else:
        banks = list_voicebanks()
        if not banks:
            raise RuntimeError(
                f"no voicebanks installed — unpack an OpenUtau DiffSinger bank "
                f"into {banks_dir()}")
        hit = next((b for b in banks if b.slug == slug_or_path), None) if slug_or_path else None
        if slug_or_path and hit is None:
            raise ValueError(f"no voicebank named {slug_or_path!r}")
        root = (hit or banks[0]).path
    if root not in _BANKS:
        _BANKS[root] = Voicebank(root)
    return _BANKS[root]


def unload() -> None:
    _BANKS.clear()


def g2p(word: str) -> Optional[List[str]]:
    """English word to ARPAbet phonemes, lowercased and stress-stripped.

    The banks ship phoneme maps rather than word dictionaries — OpenUtau does
    this conversion itself — but their English inventory is plain ARPAbet, so
    CMUdict lines up exactly.
    """
    global _CMU
    if _CMU is None:
        import cmudict

        _CMU = cmudict.dict()
    pron = _CMU.get(word.lower().strip("'\".,!?;:"))
    if not pron:
        return None
    return [p.rstrip("012").lower() for p in pron[0]]


# --- turning a melody into what the models want -------------------------
def _frames(seconds: float, hop: int, sr: int) -> int:
    return max(1, int(round(seconds * sr / hop)))


@dataclasses.dataclass
class _Plan:
    """The note/word sequence, with rests made explicit."""
    phones: List[str]
    word_div: List[int]        # phonemes per word (a rest is a one-phoneme word)
    word_dur: List[int]        # frames per word
    note_midi: List[float]
    note_dur: List[int]        # frames per note
    ph_midi: List[int]         # the note each phoneme belongs to, for the dur model


def plan(notes: Sequence, lyrics: str, hop: int = HOP, sr: int = SR,
         gap_frames: int = 3) -> _Plan:
    """Lay a melody + lyrics out as words, notes and rests.

    A word may span several notes ("to-day" over two), which the models handle
    natively: word_dur covers the whole word while note_dur stays per note.
    Silence between notes becomes an explicit rest, because a gap that is simply
    left out shifts everything after it.
    """
    from fantasia_core.sing import split_lyrics_joined

    ordered = sorted(notes, key=lambda n: n.start)
    tokens = split_lyrics_joined(lyrics, len(ordered))

    # group note indices by the word they belong to
    words: List[List[int]] = []
    for i, (_tok, cont) in enumerate(tokens):
        if cont and words:
            words[-1].append(i)
        else:
            words.append([i])

    p = _Plan([], [], [], [], [], [])
    cursor = float(ordered[0].start)
    if cursor > 1e-6:                       # leading silence
        _add_rest(p, _frames(cursor, hop, sr), ordered[0].pitch)
    for widx in words:
        first, last = ordered[widx[0]], ordered[widx[-1]]
        gap = float(first.start) - cursor
        if gap * sr / hop >= gap_frames:
            _add_rest(p, _frames(gap, hop, sr), first.pitch)
        text = "".join(tokens[i][0] for i in widx)
        phones = g2p(text)
        if not phones:                      # unknown word: one phone per syllable
            phones = [tokens[i][0][:2] or REST for i in widx]
        p.phones += phones
        p.word_div.append(len(phones))
        span = sum(_frames(ordered[i].duration, hop, sr) for i in widx)
        p.word_dur.append(span)
        p.ph_midi += [int(first.pitch)] * len(phones)
        for i in widx:
            p.note_midi.append(float(ordered[i].pitch))
            p.note_dur.append(_frames(ordered[i].duration, hop, sr))
        cursor = float(last.start) + float(last.duration)
    return p


def _add_rest(p: _Plan, frames: int, pitch: float) -> None:
    p.phones.append(REST)
    p.word_div.append(1)
    p.word_dur.append(frames)
    p.ph_midi.append(int(pitch))
    p.note_midi.append(0.0)                 # a rest carries no pitch
    p.note_dur.append(frames)


def _fit(raw: np.ndarray, div: Sequence[int], target: Sequence[int]) -> np.ndarray:
    """Scale predicted phoneme durations so each word fills its notes exactly.

    The duration model returns a shape, not a schedule. Left unscaled the
    phonemes drift out of time with the notes within a couple of bars.
    """
    out, k = [], 0
    for n, want in zip(div, target):
        part = np.maximum(raw[k:k + n], 0.05)
        k += n
        got = np.maximum(1, np.round(part / part.sum() * want).astype(int))
        while got.sum() > want and got.max() > 1:
            got[int(np.argmax(got))] -= 1
        while got.sum() < want:
            got[int(np.argmax(got))] += 1
        out += list(got)
    return np.array([out], dtype=np.int64)


def sing_notes(notes: Sequence, lyrics: str, voicebank: Optional[str] = None,
               speaker: Optional[str] = None, sr: int = SR,
               steps: int = ACOUSTIC_STEPS, gender: float = 0.0,
               progress=None) -> np.ndarray:
    """Render a melody + lyrics as sung audio. Returns mono float32 at ``sr``."""
    vb = load(voicebank)
    ordered = sorted(notes, key=lambda n: n.start)
    if not ordered:
        return np.zeros((0,), dtype=np.float32)

    pl = plan(ordered, lyrics, vb.hop, vb.sr)
    tokens = np.array([[vb.token(p) for p in pl.phones]], dtype=np.int64)
    langs = np.array([[vb.lang_en] * len(pl.phones)], dtype=np.int64)
    word_div = np.array([pl.word_div], dtype=np.int64)
    word_dur = np.array([pl.word_dur], dtype=np.int64)
    emb = vb.embed(speaker)
    spk = lambda n: np.tile(emb, (1, n, 1)).astype(np.float32)  # noqa: E731

    if progress:
        progress("durations")
    enc, masks = vb.session("linguistic-dur").run(None, {
        "tokens": tokens, "languages": langs,
        "word_div": word_div, "word_dur": word_dur})
    raw = vb.session("dur").run(None, {
        "encoder_out": enc, "x_masks": masks,
        "ph_midi": np.array([pl.ph_midi], dtype=np.int64),
        "spk_embed": spk(tokens.shape[1])})[0][0]
    ph_dur = _fit(raw, pl.word_div, pl.word_dur)
    n = int(ph_dur.sum())

    if progress:
        progress("pitch")
    enc_p, _ = vb.session("linguistic-pitch").run(None, {
        "tokens": tokens, "languages": langs, "ph_dur": ph_dur})
    base = np.concatenate([np.full(d, m if m > 0 else pl.note_midi[max(i - 1, 0)] or 60.0)
                           for i, (d, m) in enumerate(zip(pl.note_dur, pl.note_midi))])
    base = np.resize(base, n)[None].astype(np.float32)
    pitch = vb.session("pitch").run(None, {
        "encoder_out": enc_p, "ph_dur": ph_dur,
        "note_midi": np.array([pl.note_midi], dtype=np.float32),
        "note_dur": np.array([pl.note_dur], dtype=np.int64),
        "pitch": base, "expr": np.ones((1, n), np.float32),
        "retake": np.ones((1, n), bool), "spk_embed": spk(n),
        "steps": np.array(VARIANCE_STEPS, dtype=np.int64)})[0]
    f0 = (440.0 * 2 ** ((pitch - 69) / 12)).astype(np.float32)

    feed = {"tokens": tokens, "languages": langs, "durations": ph_dur, "f0": f0,
            "gender": np.full((1, n), float(gender), np.float32),
            "velocity": np.ones((1, n), np.float32),
            "spk_embed": spk(n),
            "depth": np.array(vb.max_depth, dtype=np.float32),
            "steps": np.array(int(steps), dtype=np.int64)}
    if vb.use_tension:
        if progress:
            progress("expression")
        enc_v, _ = vb.session("linguistic-variance").run(None, {
            "tokens": tokens, "languages": langs, "ph_dur": ph_dur})
        feed["tension"] = vb.session("variance").run(None, {
            "encoder_out": enc_v, "ph_dur": ph_dur, "pitch": pitch,
            "tension": np.zeros((1, n), np.float32),
            "retake": np.ones((1, n, 1), bool), "spk_embed": spk(n),
            "steps": np.array(VARIANCE_STEPS, dtype=np.int64)})[0]

    if progress:
        progress("voice")
    mel = vb.session("acoustic").run(None, feed)[0]
    audio = vb.session("vocoder").run(None, {"mel": mel, "f0": f0})[0][0]
    audio = np.asarray(audio, dtype=np.float32)

    if sr != vb.sr and len(audio):
        import librosa

        audio = librosa.resample(audio, orig_sr=vb.sr, target_sr=sr).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio = (audio / peak * 0.95).astype(np.float32)
    return audio


def sing_to_file(notes: Sequence, lyrics: str, path: str, voicebank: Optional[str] = None,
                 speaker: Optional[str] = None, sr: int = SR, steps: int = ACOUSTIC_STEPS,
                 gender: float = 0.0, progress=None) -> float:
    """Render to a mono WAV; returns duration in seconds."""
    import soundfile as sf

    audio = sing_notes(notes, lyrics, voicebank, speaker, sr, steps, gender, progress)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")
    return len(audio) / sr

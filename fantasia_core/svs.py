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
# How far the predicted f0 may wander from the written note, in semitones.
# Banks differ in how freely their pitch models sing: Peiton drifts about four
# semitones below the note across a half-note, which reads as out of tune rather
# than expressive. Wide enough to keep scoops, vibrato and glides between notes;
# narrow enough that the melody stays the melody.
PITCH_LEEWAY = 2.5


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


def _read_yaml(path):
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def _root_config(bank: pathlib.Path) -> Optional[pathlib.Path]:
    """The bank's dsconfig.yaml. Layout differs between banks: LUNAI keeps it in
    configs/ with the models in files/, Peiton puts both at the top level."""
    for cand in (bank / "dsconfig.yaml", bank / "configs" / "dsconfig.yaml"):
        if cand.is_file():
            return cand
    hits = [p for p in bank.glob("*/dsconfig.yaml") if p.is_file()]
    return hits[0] if hits else None


def list_voicebanks() -> List[VoicebankInfo]:
    """Every installed voicebank, whether or not it can render yet."""
    out = []
    for d in sorted(banks_dir().iterdir()):
        if not d.is_dir():
            continue
        cfg = _root_config(d)
        if cfg is None:
            continue
        try:
            c = _read_yaml(cfg) or {}
        except Exception:  # noqa: BLE001
            continue
        spk = [str(s).split("/")[-1] for s in (c.get("speakers") or [])] or ["default"]
        name = d.name
        for ch in (cfg.parent / "character.yaml", d / "character.yaml"):
            if ch.exists():
                try:
                    name = (_read_yaml(ch) or {}).get("name") or name
                    break
                except Exception:  # noqa: BLE001
                    pass
        voc = _vocoder_path(cfg.parent)
        out.append(VoicebankInfo(d.name, str(name), str(d), spk, voc is not None,
                                 "" if voc else "vocoder missing"))
    return out


def _vocoder_path(base: pathlib.Path) -> Optional[pathlib.Path]:
    """The vocoder for a bank. Some ship it inside; LUNAI banks expect it
    installed alongside, which is why a bank alone can yield a mel but no audio."""
    for vy in sorted(base.glob("dsvocoder/*.yaml")) + sorted(base.glob("**/dsvocoder/*.yaml")):
        try:
            vc = _read_yaml(vy) or {}
        except Exception:  # noqa: BLE001
            continue
        rel = vc.get("model")
        if rel:
            p = (vy.parent / rel).resolve()
            if p.exists():
                return p
    hits = sorted(base.glob("**/*gan*.onnx")) + sorted(base.parent.glob("**/*gan*.onnx"))
    return hits[0] if hits else None


class _Module:
    """One stage of the chain: its models, its own phoneme table, its lang flag.

    Each stage carries a separate phoneme map — Katyusha's variance stages use
    phonemes-variance.json rather than the acoustic table — so token ids must be
    looked up per stage, not once for the bank.
    """

    def __init__(self, cfg_path: pathlib.Path) -> None:
        self.dir = cfg_path.parent
        self.config = _read_yaml(cfg_path) or {}
        self.use_lang = bool(self.config.get("use_lang_id", False))
        self.phonemes = self._table("phonemes") or {}
        self.languages = self._table("languages") or {}

    def _table(self, key):
        """Phoneme/language tables ship either as JSON name->id or as a plain
        text list where the line number is the id (TIGER does the latter)."""
        rel = self.config.get(key)
        if not rel:
            return None
        p = (self.dir / rel).resolve()
        if not p.exists():
            return None
        if p.suffix.lower() == ".json":
            try:
                return json.load(open(p))
            except Exception:  # noqa: BLE001
                return None
        names = [ln.strip() for ln in open(p, encoding="utf-8") if ln.strip()]
        return {n: i for i, n in enumerate(names)}

    def model(self, key) -> Optional[pathlib.Path]:
        rel = self.config.get(key)
        if not rel:
            return None
        p = (self.dir / rel).resolve()
        return p if p.exists() else None

    def token(self, phoneme: str) -> int:
        return int(self.phonemes.get(f"en/{phoneme}",
                                     self.phonemes.get(phoneme,
                                                       self.phonemes.get(REST, 2))))

    def lang_ids(self, n: int) -> np.ndarray:
        return np.array([[int(self.languages.get("en", 0))] * n], dtype=np.int64)


class Voicebank:
    """A loaded voicebank. Sessions open lazily and are kept."""

    def __init__(self, root: str) -> None:
        self.root = pathlib.Path(root)
        cfg = _root_config(self.root)
        if cfg is None:
            raise RuntimeError(f"{self.root.name} has no dsconfig.yaml")
        self.base = cfg.parent
        self.acoustic = _Module(cfg)
        self.config = self.acoustic.config
        self.sr = int(self.config.get("sample_rate", SR))
        self.hop = int(self.config.get("hop_size", HOP))
        self.max_depth = float(self.config.get("max_depth", 1.0))
        self.use_tension = bool(self.config.get("use_tension_embed", False))
        self.speakers = [str(s).split("/")[-1] for s in (self.config.get("speakers") or [])]
        self._spk_paths = {str(s).split("/")[-1]: str(s) for s in (self.config.get("speakers") or [])}
        self._mods: Dict[str, Optional[_Module]] = {}
        self._sessions: Dict[str, object] = {}
        self._embeds: Dict[str, np.ndarray] = {}

    def module(self, kind: str) -> Optional[_Module]:
        """kind is 'dur' | 'pitch' | 'variance'."""
        if kind not in self._mods:
            cand = self.base / f"ds{kind}" / "dsconfig.yaml"
            self._mods[kind] = _Module(cand) if cand.exists() else None
        return self._mods[kind]

    def session(self, path) -> object:
        import onnxruntime as ort

        key = str(path)
        if key not in self._sessions:
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self._sessions[key] = ort.InferenceSession(
                key, opts, providers=["CPUExecutionProvider"])
        return self._sessions[key]

    def vocoder(self):
        p = _vocoder_path(self.base)
        if p is None:
            raise FileNotFoundError(
                f"{self.root.name} has no vocoder — some banks expect it installed "
                f"alongside; see configs/dsvocoder/vocoder.yaml for the name")
        return self.session(p)

    def embed(self, speaker: Optional[str] = None) -> np.ndarray:
        speaker = speaker or (self.speakers[0] if self.speakers else "")
        if speaker not in self._embeds:
            rel = self._spk_paths.get(speaker, speaker)
            cands = [self.base / f"{rel}.emb", self.base / "embeds" / f"{speaker}.emb",
                     self.root / f"{rel}.emb"]
            p = next((c for c in cands if c.exists()), None)
            if p is None:
                found = sorted(self.base.glob("**/*.emb"))
                if not found:
                    raise FileNotFoundError("no speaker embeddings in this voicebank")
                p = found[0]
            self._embeds[speaker] = np.fromfile(p, dtype="<f4")
        return self._embeds[speaker]


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

        syls = [tokens[i][0] for i in widx]
        durs = [_frames(ordered[i].duration, hop, sr) for i in widx]
        phones = g2p("".join(syls))
        if phones:
            # A known word: it spans its notes as one unit, and the models
            # spread its phonemes across them.
            p.phones += phones
            p.word_div.append(len(phones))
            p.word_dur.append(sum(durs))
            p.ph_midi += [int(first.pitch)] * len(phones)
        else:
            # Hyphenation that does not rejoin into a dictionary word — either a
            # made-up split ("be-ter") or a melisma stretching one word over more
            # notes than it has syllables ("a-lo-one"). Phonemize each syllable
            # on its own and give it its own note, which is what the singer is
            # actually doing.
            for i, syl in zip(widx, syls):
                ph = g2p(syl) or [syl[:2] or REST]
                p.phones += ph
                p.word_div.append(len(ph))
                p.word_dur.append(_frames(ordered[i].duration, hop, sr))
                p.ph_midi += [int(ordered[i].pitch)] * len(ph)
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


_ONNX_DTYPE = {"tensor(float)": np.float32, "tensor(double)": np.float64,
               "tensor(int64)": np.int64, "tensor(int32)": np.int32,
               "tensor(bool)": np.bool_}


def _feed(session, args: Dict[str, object]) -> Dict[str, object]:
    """Keep only the inputs a model declares, cast to the types it declares.

    Voicebanks are exported by different DiffSinger versions and disagree on
    both: Peiton's duration model takes no speaker embedding at all, TIGER wants
    note_rest and a "speedup" divisor where newer banks take a step count, and
    TIGER's depth is an int where everyone else's is a float. Reading the graph
    is the only reliable way to know.
    """
    out = {}
    for i in session.get_inputs():
        if i.name not in args:
            continue
        v = args[i.name]
        want = _ONNX_DTYPE.get(i.type)
        out[i.name] = np.asarray(v, dtype=want) if want is not None else v
    return out


def _hold_the_tune(pitch: np.ndarray, base: np.ndarray, pl: "_Plan",
                   leeway: float = PITCH_LEEWAY) -> np.ndarray:
    """Keep a predicted f0 curve within ``leeway`` semitones of the written note.

    The target is smoothed across note boundaries first, so a leap still glides
    instead of being clipped — the bound is on drifting away from a note, not on
    moving between them.
    """
    if leeway <= 0 or pitch.shape != base.shape:
        return pitch
    # Ramp the target over ~60ms at each change so transitions stay free.
    win = max(1, int(0.06 * 1000.0 / 5.0))
    k = np.ones(win, dtype=np.float64) / win
    tgt = np.convolve(np.pad(base[0], (win, win), mode="edge"), k, mode="same")[win:-win]
    lo, hi = tgt - leeway, tgt + leeway
    out = np.clip(pitch[0], lo, hi)
    # Rests carry no pitch, so leave whatever the model produced there.
    voiced = np.concatenate([np.full(d, m > 0) for d, m in
                             zip(pl.note_dur, pl.note_midi)])
    voiced = np.resize(voiced, len(out))
    return np.where(voiced, out, pitch[0])[None].astype(np.float32)


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
    emb = vb.embed(speaker)
    n_ph = len(pl.phones)

    def spk(n):
        return np.tile(emb, (1, n, 1)).astype(np.float32)

    def stage(mod):
        """tokens + languages in this stage's own vocabulary."""
        toks = np.array([[mod.token(x) for x in pl.phones]], dtype=np.int64)
        return toks, (mod.lang_ids(n_ph) if mod.use_lang else None)

    note_midi = np.array([pl.note_midi], dtype=np.float32)
    note_dur = np.array([pl.note_dur], dtype=np.int64)
    note_rest = np.array([[m <= 0 for m in pl.note_midi]], dtype=bool)

    # --- 1. phoneme durations -----------------------------------------
    dur_m = vb.module("dur")
    if dur_m is None or not dur_m.model("dur"):
        raise RuntimeError(f"{vb.root.name} has no duration model")
    if progress:
        progress("durations")
    toks, langs = stage(dur_m)
    ling = vb.session(dur_m.model("linguistic"))
    enc, masks = ling.run(None, _feed(ling, {
        "tokens": toks, "languages": langs,
        "word_div": np.array([pl.word_div], dtype=np.int64),
        "word_dur": np.array([pl.word_dur], dtype=np.int64)}))
    ds = vb.session(dur_m.model("dur"))
    raw = ds.run(None, _feed(ds, {
        "encoder_out": enc, "x_masks": masks,
        "ph_midi": np.array([pl.ph_midi], dtype=np.int64),
        "spk_embed": spk(n_ph)}))[0][0]
    ph_dur = _fit(raw, pl.word_div, pl.word_dur)
    n = int(ph_dur.sum())

    # --- 2. the f0 curve, conditioned on the written notes -------------
    base = np.concatenate([
        np.full(d, m if m > 0 else (pl.note_midi[max(i - 1, 0)] or 60.0))
        for i, (d, m) in enumerate(zip(pl.note_dur, pl.note_midi))])
    base = np.resize(base, n)[None].astype(np.float32)
    pitch_m = vb.module("pitch")
    if pitch_m is not None and pitch_m.model("pitch"):
        if progress:
            progress("pitch")
        toks_p, langs_p = stage(pitch_m)
        lp = vb.session(pitch_m.model("linguistic"))
        enc_p, _ = lp.run(None, _feed(lp, {
            "tokens": toks_p, "languages": langs_p, "ph_dur": ph_dur}))
        ps = vb.session(pitch_m.model("pitch"))
        pitch = ps.run(None, _feed(ps, {
            "encoder_out": enc_p, "ph_dur": ph_dur, "note_midi": note_midi,
            "note_dur": note_dur, "note_rest": note_rest, "pitch": base,
            "expr": np.ones((1, n), np.float32),
            "retake": np.ones((1, n), bool), "spk_embed": spk(n),
            "steps": np.array(VARIANCE_STEPS), "speedup": np.array(5)}))[0]
    else:
        pitch = base                      # no pitch model: sing the notes flat
    pitch = _hold_the_tune(pitch, base, pl)
    f0 = (440.0 * 2 ** ((pitch - 69) / 12)).astype(np.float32)

    # --- 3. expression -------------------------------------------------
    feed = {"tokens": None, "durations": ph_dur, "f0": f0,
            "gender": np.full((1, n), float(gender), np.float32),
            "velocity": np.ones((1, n), np.float32), "spk_embed": spk(n),
            "depth": np.array(vb.max_depth), "steps": np.array(int(steps)),
            "speedup": np.array(max(1, 1000 // max(int(steps), 1)))}
    var_m = vb.module("variance")
    if vb.use_tension and var_m is not None and var_m.model("variance"):
        if progress:
            progress("expression")
        toks_v, langs_v = stage(var_m)
        lv = vb.session(var_m.model("linguistic"))
        enc_v, _ = lv.run(None, _feed(lv, {
            "tokens": toks_v, "languages": langs_v, "ph_dur": ph_dur}))
        vs = vb.session(var_m.model("variance"))
        feed["tension"] = vs.run(None, _feed(vs, {
            "encoder_out": enc_v, "ph_dur": ph_dur, "pitch": pitch,
            "tension": np.zeros((1, n), np.float32),
            "retake": np.ones((1, n, 1), bool), "spk_embed": spk(n),
            "steps": np.array(VARIANCE_STEPS), "speedup": np.array(5)}))[0]

    # --- 4. mel, then the vocoder --------------------------------------
    if progress:
        progress("voice")
    toks_a, langs_a = stage(vb.acoustic)
    feed["tokens"] = toks_a
    feed["languages"] = langs_a
    ac = vb.session(vb.acoustic.model("acoustic"))
    mel = ac.run(None, _feed(ac, feed))[0]

    voc = vb.vocoder()
    audio = np.asarray(voc.run(None, _feed(voc, {"mel": mel, "f0": f0}))[0][0],
                       dtype=np.float32)

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

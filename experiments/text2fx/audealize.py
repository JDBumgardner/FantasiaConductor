"""Audealize (Seetharaman & Pardo, JAES 2016) — crowdsourced word -> EQ curve.

394 English words, each a 40-band graphic EQ curve, log-spaced 20 Hz-20 kHz.
Applied exactly as the reference plugin does: curve normalised to span [-1, 1],
gain_dB = curve * 5 * amount, 40 peaking biquads at Q 4.31.

Data: interactiveaudiolab/audealize_api  static/data/eqdescriptors.json
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FREQS = [20, 50, 83, 120, 161, 208, 259, 318, 383, 455, 537, 628, 729, 843, 971,
         1114, 1273, 1452, 1652, 1875, 2126, 2406, 2719, 3070, 3462, 3901, 4392,
         4941, 5556, 6244, 7014, 7875, 8839, 9917, 11124, 12474, 13984, 15675,
         17566, 19682]
Q = 4.31
_DATA = None

def table() -> dict:
    global _DATA
    if _DATA is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audealize_eq.json")
        _DATA = {e["word"].lower(): np.asarray(e["settings"], dtype=np.float64)
                 for e in json.load(open(p)) if e.get("lang") == "English"}
    return _DATA

def curve_db(word: str, amount: float = 1.0) -> np.ndarray:
    """40 gains in dB for `word`, as the plugin computes them."""
    c = table()[word.lower()]
    lo, hi = min(c.min(), 0.0), max(c.max(), 0.0)
    if hi != lo: c = (c - lo) / (hi - lo) * 2 - 1
    return c * 5.0 * amount

def apply(y: np.ndarray, sr: int, word: str, amount: float = 1.0) -> np.ndarray:
    from pedalboard import Pedalboard, PeakFilter
    g = curve_db(word, amount)
    board = Pedalboard([PeakFilter(cutoff_frequency_hz=f, gain_db=float(d), q=Q)
                        for f, d in zip(FREQS, g) if f < sr / 2])
    return board(y.copy(), sr)

if __name__ == "__main__":
    import soundfile as sf, torch, common as C
    y = C.kalimba(); SRC = torch.tensor(y)
    T = C.text_emb("this sound is warm, dark and mellow")
    print(f"  {len(table())} words.  e.g. " + ", ".join(sorted(table())[:12]))
    print(f"\n  'warm' curve (dB):")
    g = curve_db("warm")
    for f, d in zip(FREQS[::4], g[::4]): print(f"     {f:6d} Hz  {d:+5.2f}")
    print(f"  max |gain| = {np.abs(g).max():.2f} dB")
    print(f"\n  dry score={C.score(SRC, T, SRC):+.4f}   hand shelf={C.score(C.hand_shelf(y), T, SRC):+.4f}")
    for amt in (1.0, 1.5):
        out = apply(y, C.SR, "warm", amt)
        sc = C.score(torch.tensor(out), T, SRC)
        print(f"  audealize 'warm' amount={amt}: score={sc:+.4f}")
        sf.write(os.path.join(C.HERE, f"audealize_warm_{amt}.wav"),
                 np.clip(C.level_match(torch.tensor(out), SRC).numpy(), -1, 1), C.SR)
    sf.write(os.path.join(C.HERE, "dry.wav"), y, C.SR)

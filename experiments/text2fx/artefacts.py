"""Signal-level 'is this sound broken?' metrics, each relative to a reference where one exists.

  clip      fraction of samples within 1 % of full scale after peak normalisation to the reference's crest -- true clipping
  jump      largest sample-to-sample step relative to the local RMS (50 ms), beyond what the reference has -- clicks / breaking
  wobble    amplitude-modulation depth at 0.3-12 Hz on sustained notes (dB peak-to-peak of the 20 ms envelope, band-passed),
            minus the reference's -- chorus / tremolo / pumping
  sweep     spectral-centroid modulation at 0.3-12 Hz (semitones p-p), minus the reference's -- flanger / chorus comb sweeps
  crest     peak / RMS ratio in dB relative to the reference -- negative = squashed, positive = spiky
  stutter   energy drops > 12 dB within a note that recover within 200 ms -- gating / tremolo chops
All in torch so the differentiable ones (wobble, sweep, crest, jump) can become loss terms; `report()` prints a table."""
import math, torch

def _env(x, win):
    return torch.nn.functional.avg_pool1d(x[None, None] ** 2, win, win)[0, 0].add(1e-10).sqrt()

def clip_fraction(x, thresh=0.99):
    return float((x.abs() >= thresh * x.abs().max()).float().mean()) if x.abs().max() > 0 else 0.0

def jump_ratio(x, sr=48000):
    """max |x[n]-x[n-1]| over the local RMS (50 ms), 99.9th percentile to ignore a single onset sample."""
    d = (x[1:] - x[:-1]).abs(); win = int(0.05 * sr); n = (len(d) // win) * win
    loc = _env(x[:n], win).repeat_interleave(win); r = d[:n] / (loc + 1e-4)
    return float(torch.quantile(r, 0.999))

def _bandpass_mod(e, rate_hz, lo=0.3, hi=12.0):
    """Keep the 0.3-12 Hz part of a slow signal e sampled at rate_hz (FFT mask)."""
    E = torch.fft.rfft(e - e.mean()); f = torch.fft.rfftfreq(len(e), 1 / rate_hz); E = E * ((f >= lo) & (f <= hi)).to(E.dtype)
    return torch.fft.irfft(E, n=len(e))

def wobble_db(x, sr=48000, win_ms=20):
    """Peak-to-peak (2nd-98th percentile) of the 0.3-12 Hz amplitude modulation, in dB, over frames with signal."""
    win = int(win_ms / 1000 * sr); e = _env(x, win); rate = sr / win
    lvl = 20 * torch.log10(e + 1e-6); on = lvl > (lvl.max() - 40)
    m = _bandpass_mod(lvl.masked_fill(~on, lvl[on].mean() if on.any() else 0.0), rate)
    m = m[on] if on.any() else m
    return float(torch.quantile(m, 0.98) - torch.quantile(m, 0.02))

def sweep_semitones(x, sr=48000, n_fft=2048, hop=960):
    """0.3-12 Hz modulation of the spectral centroid, in semitones peak-to-peak, over frames with signal."""
    win = torch.hann_window(n_fft); S = torch.stft(x, n_fft, hop, window=win, return_complex=True).abs() ** 2
    f = torch.fft.rfftfreq(n_fft, 1 / sr)[:, None]; cen = (S * f).sum(0) / (S.sum(0) + 1e-9)
    lvl = 10 * torch.log10(S.sum(0) + 1e-9); on = lvl > (lvl.max() - 40)
    semis = 12 * torch.log2(cen.clamp(min=20) / 440); rate = sr / hop
    m = _bandpass_mod(semis.masked_fill(~on, semis[on].mean() if on.any() else 0.0), rate); m = m[on] if on.any() else m
    return float(torch.quantile(m, 0.98) - torch.quantile(m, 0.02))

def crest_db(x): return float(20 * torch.log10(x.abs().max() / (x.pow(2).mean().sqrt() + 1e-9)))

def stutter_count(x, sr=48000, win_ms=10, drop_db=12.0, recover_ms=200):
    """Energy drops > drop_db within a note that recover within recover_ms: gated chops, tremolo holes."""
    win = int(win_ms / 1000 * sr); lvl = 20 * torch.log10(_env(x, win) + 1e-6); n = 0; i = 0; R = int(recover_ms / win_ms)
    while i < len(lvl) - R:
        if lvl[i] > lvl.max() - 30 and lvl[i + 1:i + 4].min() < lvl[i] - drop_db and lvl[i + 1:i + R].max() > lvl[i] - 3: n += 1; i += R
        else: i += 1
    return n

def measure(x, ref=None, sr=48000):
    x = torch.as_tensor(x, dtype=torch.float32); m = dict(clip=clip_fraction(x), jump=jump_ratio(x, sr), wobble=wobble_db(x, sr), sweep=sweep_semitones(x, sr), crest=crest_db(x), stutter=stutter_count(x, sr))
    if ref is not None:
        r = measure(torch.as_tensor(ref, dtype=torch.float32), None, sr)
        m.update(d_jump=m["jump"] - r["jump"], d_wobble=m["wobble"] - r["wobble"], d_sweep=m["sweep"] - r["sweep"], d_crest=m["crest"] - r["crest"], d_stutter=m["stutter"] - r["stutter"])
    return m

FLAGS = dict(clip=(0.001, "clipping"), d_jump=(3.0, "clicks/breaking"), d_wobble=(4.0, "wobble (chorus/tremolo/pumping)"), d_sweep=(1.5, "comb sweep (flanger/chorus)"), d_crest=(-6.0, "squashed"), d_stutter=(3, "stutter (gate)"))
def flags(m):
    out = []
    for k, (t, name) in FLAGS.items():
        if k not in m: continue
        v = m[k]
        if (k == "d_crest" and v < t) or (k != "d_crest" and v > t): out.append(name)
    return out


# ---- note-aware versions: measure inside the sustained part of each note, and as a delta between a result and the
# ---- same notes without the effects (the twin's dry render, or the soundfont for the FX-on-recording route). The
# ---- first pass above flagged 119/168 results for "sweep" because a melody moves the spectral centroid at note rate.
def _sustained_mask(n, notes, sr, skip_ms=100, hop=1):
    m = torch.zeros(n, dtype=torch.bool)
    for _, start, dur, _ in notes:
        a = int((start + skip_ms / 1000) * sr / hop); b = int((start + dur) * sr / hop)
        if b > a: m[a:min(b, n)] = True
    return m

def within_note_flux(x, notes, sr=48000, n_fft=2048, hop=480):
    """Median frame-to-frame change of the log spectrum inside sustained note segments (dB per frame). A steady
    tone is ~0; chorus, flanger, tremolo, pumping and stutter all raise it."""
    S = torch.stft(x, n_fft, hop, window=torch.hann_window(n_fft), return_complex=True).abs()
    Ld = 20 * torch.log10(S + 1e-5); flux = (Ld[:, 1:] - Ld[:, :-1]).abs().mean(0)
    lvl = 20 * torch.log10(S.pow(2).sum(0).sqrt() + 1e-6)[1:]; m = _sustained_mask(len(flux), notes, sr, hop=hop) & (lvl > lvl.max() - 40)
    return float(flux[m].median()) if m.any() else 0.0

def within_note_tremolo_db(x, notes, sr=48000, win_ms=10, lo=2.0, hi=14.0):
    """Peak-to-peak level modulation at 2-14 Hz inside sustained segments (dB): tremolo, pumping, gate chatter."""
    win = int(win_ms / 1000 * sr); lvl = 20 * torch.log10(_env(x, win) + 1e-6); rate = sr / win
    m = _sustained_mask(len(lvl), notes, sr, hop=win)
    if m.sum() < 8: return 0.0
    seg = lvl.clone(); seg[~m] = lvl[m].mean(); mod = _bandpass_mod(seg, rate, lo, hi)[m]
    return float(torch.quantile(mod, 0.98) - torch.quantile(mod, 0.02))

def measure_notes(fx, dry, notes, sr=48000):
    fx = torch.as_tensor(fx, dtype=torch.float32); dry = torch.as_tensor(dry, dtype=torch.float32)
    m = dict(clip=clip_fraction(fx), d_jump=jump_ratio(fx, sr) - jump_ratio(dry, sr), d_flux=within_note_flux(fx, notes, sr) - within_note_flux(dry, notes, sr),
             d_trem=within_note_tremolo_db(fx, notes, sr) - within_note_tremolo_db(dry, notes, sr), d_crest=crest_db(fx) - crest_db(dry), d_stutter=stutter_count(fx, sr) - stutter_count(dry, sr))
    return m
FLAGS_NOTES = dict(clip=(0.001, "clipping"), d_jump=(3.0, "clicks/breaking"), d_flux=(1.5, "modulation (chorus/flanger/pumping)"), d_trem=(4.0, "tremolo/pumping"), d_crest=(-6.0, "squashed"), d_stutter=(3, "stutter (gate)"))
def flags_notes(m): return [name for k, (t, name) in FLAGS_NOTES.items() if k in m and ((m[k] < t) if k == "d_crest" else (m[k] > t))]

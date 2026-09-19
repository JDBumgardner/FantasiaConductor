"""Audiobox-Aesthetics (Meta, 2025) as a reranker/gate: production quality (PQ), production complexity (PC),
content enjoyment (CE), content usefulness (CU), each 1-10, learned from ~97k human ratings. Never inside the
gradient loop -- a frozen judge in the objective gets gamed like CLAP -- only to rank finalists and to flag a stop
whose PQ fell against the original. Weights are CC-BY-NC: fine for research and an internal gate, check before
shipping. Loads audio with soundfile (the package's loader wants torchcodec)."""
import torch, soundfile as sf
_pred = None
def predictor():
    global _pred
    if _pred is None:
        from audiobox_aesthetics.infer import initialize_predictor
        _pred = initialize_predictor()
    return _pred
def score(items):
    """items: file paths or (waveform (L,), sr). -> list of dicts {PQ, PC, CE, CU}."""
    batch = []
    for it in items:
        if isinstance(it, str): w, sr = sf.read(it, dtype="float32"); w = torch.as_tensor(w).T if w.ndim == 2 else torch.as_tensor(w)[None]
        else: w, sr = it; w = torch.as_tensor(w, dtype=torch.float32); w = w[None] if w.ndim == 1 else w
        batch.append({"path": w, "sample_rate": sr})
    return predictor().forward(batch)

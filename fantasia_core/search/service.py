"""Sound-search service — ingest + query, over the CLAP + LanceDB dual store.

One service, two callers: the search panel and (via a tool) the agent. Query
paths: text→audio ("warm analog pad"), audio→audio ("more like this clip"), and
semantic tags. Ingest auto-tags each sound from the tag vocabulary.
"""

from __future__ import annotations

import glob
import os
from typing import Callable, List, Optional

# Vocabulary for semantic auto-tagging (CLAP text embeddings).
_TAG_VOCAB = [
    "kick drum", "snare drum", "hi-hat", "cymbal", "clap", "tom", "808 bass",
    "sub bass", "bass guitar", "warm pad", "bright synth lead", "pluck synth",
    "acoustic piano", "electric piano", "strings", "brass", "organ", "guitar",
    "vocal", "choir", "ambient texture", "drone", "riser", "impact", "percussion",
    "bell", "arpeggio", "white noise", "field recording", "rain", "ocean waves", "wind",
]
_AUDIO_EXTS = ("*.wav", "*.flac", "*.aiff", "*.aif", "*.mp3", "*.ogg")


class SearchService:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._store = None

    def available(self) -> bool:
        from fantasia_core.search import embed

        return embed.available()

    @property
    def store(self):
        if self._store is None:
            from fantasia_core.search.store import SoundLibrary

            self._store = SoundLibrary(self._db_path)
        return self._store

    def count(self) -> int:
        try:
            return self.store.audio_count()
        except Exception:  # noqa: BLE001
            return 0

    # ---- ingest ----------------------------------------------------------
    def ensure_tags(self) -> None:
        if self.store.has_tags():
            return
        from fantasia_core.search.embed import embed_texts

        vecs = embed_texts(_TAG_VOCAB)
        self.store.add_tags(
            [{"vector": vecs[i].tolist(), "tag": t} for i, t in enumerate(_TAG_VOCAB)]
        )

    def _auto_tags(self, audio_vec, k: int = 3) -> str:
        rows = self.store.search_tags(audio_vec, k)
        return ", ".join(r["tag"] for r in rows)

    def ingest_paths(self, paths: List[str], progress: Optional[Callable[[int, int], None]] = None) -> int:
        import soundfile as sf

        from fantasia_core.search.embed import embed_audio_file

        self.ensure_tags()
        existing = self._existing_norm()
        todo, seen = [], set()
        for p in paths:
            key = self._norm(p)
            if key not in existing and key not in seen:
                seen.add(key)
                todo.append(key)
        added = 0
        for path in todo:
            try:
                vec = embed_audio_file(path)
                info = sf.info(path)
                dur = info.frames / info.samplerate
            except Exception:  # noqa: BLE001
                continue
            self.store.add_audio([{
                "vector": vec.tolist(), "path": path, "name": os.path.basename(path),
                "duration": float(dur), "tags": self._auto_tags(vec),
            }])
            added += 1
            if progress:
                progress(added, len(todo))
        return added

    @staticmethod
    def _norm(path: str) -> str:
        """Canonical absolute path — the identity used for dedup. (Absolute vs
        relative spellings of the same file once caused duplicate rows.)"""
        return os.path.abspath(path)

    def _existing_norm(self) -> set:
        return {self._norm(p) for p in self.store.audio_paths()}

    def ingest_tagged(self, items: List[dict], progress: Optional[Callable[[int, int], None]] = None) -> int:
        """Ingest sounds with explicit, pre-authored tags (a curated library).

        Each item is ``{"path", "tags": [...], "name"?}``. Unlike ingest_paths this
        stores the given tags verbatim instead of inferring them."""
        import soundfile as sf

        from fantasia_core.search.embed import embed_audio_file

        existing = self._existing_norm()
        todo, seen = [], set()
        for it in items:  # dedup within the batch and against the store
            key = self._norm(it["path"])
            if key not in existing and key not in seen:
                seen.add(key)
                todo.append(it)
        added = 0
        for it in todo:
            path = self._norm(it["path"])
            try:
                vec = embed_audio_file(path)
                info = sf.info(path)
                dur = info.frames / info.samplerate
            except Exception:  # noqa: BLE001
                continue
            tags = it.get("tags")
            tag_str = ", ".join(tags) if isinstance(tags, (list, tuple)) else (tags or "")
            self.store.add_audio([{
                "vector": vec.tolist(), "path": path,
                "name": it.get("name") or os.path.basename(path),
                "duration": float(dur), "tags": tag_str,
            }])
            added += 1
            if progress:
                progress(added, len(todo))
        return added

    def ingest_folder(self, folder: str, progress=None) -> int:
        paths: List[str] = []
        for ext in _AUDIO_EXTS:
            paths.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
        return self.ingest_paths(sorted(set(paths)), progress)

    # ---- query -----------------------------------------------------------
    def search_text(self, query: str, k: int = 12) -> List[dict]:
        from fantasia_core.search.embed import embed_texts

        return self._results(self.store.search_audio(embed_texts([query])[0], k))

    def search_audio_file(self, path: str, k: int = 12) -> List[dict]:
        from fantasia_core.search.embed import embed_audio_file

        return self._results(self.store.search_audio(embed_audio_file(path), k))

    def _results(self, rows: List[dict]) -> List[dict]:
        out, seen = [], set()
        for r in rows:
            key = self._norm(r.get("path") or "")
            if key in seen:  # same file under two path spellings → one result
                continue
            seen.add(key)
            out.append({
                "name": r.get("name"), "path": r.get("path"), "tags": r.get("tags", ""),
                "duration": float(r.get("duration", 0.0)),
                "score": round(1.0 - float(r.get("_distance", 0.0)), 3),
            })
        return out

    # ---- maintenance -----------------------------------------------------
    def replace_name(self, name: str) -> int:
        """Delete rows whose display name matches exactly — used so a re-generated
        take (same text → same name) replaces the old one instead of piling up.
        Only DB rows are removed; the old WAV stays for clips that reference it."""
        doomed = [p for n, p in self.store.audio_names_paths() if n == name]
        self.store.delete_paths(doomed)
        return len(doomed)

    def dedupe(self) -> int:
        """Remove rows whose normalized path duplicates another row (keeps the
        absolute-path spelling). Returns the number of rows removed."""
        if not self.store.has_audio():
            return 0
        paths = self.store.audio_paths()
        keep: dict = {}
        for p in paths:
            key = self._norm(p)
            if key not in keep or (not os.path.isabs(keep[key]) and os.path.isabs(p)):
                keep[key] = p
        doomed = [p for p in paths if p != keep[self._norm(p)]]
        self.store.delete_paths(doomed)
        return len(doomed)

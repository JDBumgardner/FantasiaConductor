"""LanceDB store — two tables (the "dual vector DB").

* ``audio`` — one row per sound: CLAP audio embedding + path/name/duration/tags.
  Powers text→audio and audio→audio search.
* ``tags``  — one row per vocabulary tag: CLAP text embedding + label.
  Powers semantic tag search and audio→tag auto-tagging (embed a sound, find its
  nearest tags).

Embedded, on-disk (no server). LanceDB is imported lazily.
"""

from __future__ import annotations

from typing import List


class SoundLibrary:
    def __init__(self, db_path: str) -> None:
        import lancedb

        self._db = lancedb.connect(db_path)

    def _tables(self) -> set:
        lister = getattr(self._db, "list_tables", None)
        if lister is not None:
            resp = lister()
            return set(getattr(resp, "tables", resp))  # newer API returns a response obj
        return set(self._db.table_names())

    def has_audio(self) -> bool:
        return "audio" in self._tables()

    def has_tags(self) -> bool:
        return "tags" in self._tables()

    def audio_count(self) -> int:
        return self._db.open_table("audio").count_rows() if self.has_audio() else 0

    def audio_paths(self) -> set:
        if not self.has_audio():
            return set()
        return set(self._db.open_table("audio").to_arrow().column("path").to_pylist())

    # ---- writes ----------------------------------------------------------
    def add_audio(self, rows: List[dict]) -> None:
        if not rows:
            return
        if self.has_audio():
            self._db.open_table("audio").add(rows)
        else:
            self._db.create_table("audio", data=rows)

    def add_tags(self, rows: List[dict]) -> None:
        if not rows:
            return
        if self.has_tags():
            self._db.open_table("tags").add(rows)
        else:
            self._db.create_table("tags", data=rows)

    def audio_names_paths(self) -> List[tuple]:
        """All audio rows as ``(name, path)`` pairs."""
        if not self.has_audio():
            return []
        tbl = self._db.open_table("audio").to_arrow()
        return list(zip(tbl.column("name").to_pylist(), tbl.column("path").to_pylist()))

    def delete_paths(self, paths: List[str]) -> None:
        """Delete audio rows whose stored ``path`` exactly matches any given."""
        if not self.has_audio() or not paths:
            return
        quoted = ", ".join("'" + p.replace("'", "''") + "'" for p in paths)
        self._db.open_table("audio").delete(f"path IN ({quoted})")

    # ---- searches --------------------------------------------------------
    def search_audio(self, vector, k: int = 12) -> List[dict]:
        if not self.has_audio():
            return []
        return self._db.open_table("audio").search(vector).metric("cosine").limit(k).to_list()

    def search_tags(self, vector, k: int = 5) -> List[dict]:
        if not self.has_tags():
            return []
        return self._db.open_table("tags").search(vector).metric("cosine").limit(k).to_list()

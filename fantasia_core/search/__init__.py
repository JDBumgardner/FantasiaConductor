"""Sound search — CLAP embeddings + LanceDB stores, one service, two callers.

Headless (no Qt). Heavy deps (torch/transformers/lancedb) and the CLAP model are
loaded lazily; embedding runs off the UI thread.
"""

from fantasia_core.search.service import SearchService

__all__ = ["SearchService"]

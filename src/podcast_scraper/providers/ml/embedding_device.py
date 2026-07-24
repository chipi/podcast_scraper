"""Resolve a STABLE torch device for local sentence-transformers embedding.

macOS **MPS** (Apple GPU) intermittently crashes the process with a native SIGSEGV during the
GI ``all-MiniLM`` embedding (``sentence_transformers … using mps`` → segfault mid-``encode``).
Left to auto-detect, sentence-transformers prefers ``cuda > mps > cpu`` and picks MPS on a Mac.
So we resolve explicitly and **skip MPS**: prefer CUDA (DGX / prod, where it is stable and fast),
otherwise CPU (stable, a little slower). ``PODCAST_EMBED_DEVICE`` overrides for special cases.
"""

from __future__ import annotations

import logging
import os

_LOGGER = logging.getLogger(__name__)


def resolve_embedding_device() -> str:
    """Return the device for a local SentenceTransformer encoder — never ``mps``.

    ``PODCAST_EMBED_DEVICE`` (e.g. ``cuda`` / ``cpu`` / ``mps`` if you really want it) wins; else
    ``cuda`` when available, else ``cpu``.
    """
    override = os.environ.get("PODCAST_EMBED_DEVICE", "").strip()
    if override:
        return override
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 — torch missing / probe failure → fall back to CPU
        _LOGGER.debug("cuda probe failed; using cpu for embedding", exc_info=True)
    return "cpu"  # deliberately NOT mps — it flaky-segfaults the GI embedding on macOS

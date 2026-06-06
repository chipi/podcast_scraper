"""HTTP embedding client for Ollama-served models (#897 supersedes RFC-089 §D4).

Posts to ``{base_url}/api/embed`` (newer Ollama, batch) and falls back to the
older ``/api/embeddings`` (one-text-at-a-time) on 404. Ollama does not L2-normalize
by default — we do it client-side when requested, to match the in-process
sentence-transformers and shim contracts.

Used by ``embedding_loader.encode`` when ``vector_embedding_provider == "ollama"``.
"""

from __future__ import annotations

import logging
import math
from typing import List, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


def _l2_normalize(row: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in row))
    if norm == 0.0:
        return row
    return [x / norm for x in row]


def encode_via_ollama(
    texts: Union[str, List[str]],
    base_url: str,
    *,
    model_id: str,
    normalize: bool = True,
    timeout_sec: float = 120.0,
) -> List[List[float]]:
    """POST texts to an Ollama server; return embedding rows.

    Args:
        texts: Single string or list.
        base_url: Ollama base URL (e.g. ``http://dgx:11434``). Path is appended
            internally — do not include ``/api/embed``.
        model_id: Ollama model tag (e.g. ``nomic-embed-text``). NOT a HuggingFace id —
            Ollama has its own model registry.
        normalize: If True, L2-normalize each row client-side (Ollama embeddings
            are NOT pre-normalized).
        timeout_sec: HTTP timeout — embed calls on cold model can take 10s+.

    Returns:
        List of embedding rows, one per input text, in the same order.
    """
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx required for Ollama embedding provider") from exc

    if isinstance(texts, str):
        batch = [texts]
    else:
        batch = list(texts)
    if not batch:
        return []
    base = base_url.strip().rstrip("/")
    if not base:
        raise ValueError("Ollama base_url is empty")

    with httpx.Client(timeout=timeout_sec) as client:
        # Prefer the batch endpoint (Ollama ≥ 0.1.30).
        url = f"{base}/api/embed"
        resp = client.post(url, json={"model": model_id, "input": batch})
        if resp.status_code == 404:
            # Older Ollama: one-call-per-text via /api/embeddings.
            return _encode_legacy(client, base, model_id, batch, normalize)
        resp.raise_for_status()
        data = resp.json()

    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list):
        raise ValueError("Ollama /api/embed response missing 'embeddings' list")
    rows: List[List[float]] = []
    for row in embeddings:
        if not isinstance(row, list):
            raise ValueError("invalid embedding row from Ollama")
        vec = [float(x) for x in row]
        rows.append(_l2_normalize(vec) if normalize else vec)
    if len(rows) != len(batch):
        raise ValueError(f"Ollama row count mismatch: got {len(rows)} for {len(batch)} inputs")
    return rows


def _encode_legacy(
    client: httpx.Client,
    base: str,
    model_id: str,
    batch: List[str],
    normalize: bool,
) -> List[List[float]]:
    """Fallback for Ollama < 0.1.30 — POST /api/embeddings once per text."""
    rows: List[List[float]] = []
    for text in batch:
        resp = client.post(f"{base}/api/embeddings", json={"model": model_id, "prompt": text})
        resp.raise_for_status()
        data = resp.json()
        vec_raw = data.get("embedding")
        if not isinstance(vec_raw, list):
            raise ValueError("Ollama /api/embeddings response missing 'embedding' list")
        vec = [float(x) for x in vec_raw]
        rows.append(_l2_normalize(vec) if normalize else vec)
    return rows

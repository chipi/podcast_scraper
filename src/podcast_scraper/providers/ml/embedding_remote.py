"""HTTP embedding client for DGX embedding shim (RFC-089)."""

from __future__ import annotations

import logging
from typing import List, Union

logger = logging.getLogger(__name__)


def encode_via_endpoint(
    texts: Union[str, List[str]],
    endpoint_url: str,
    *,
    model_id: str,
    normalize: bool = True,
    timeout_sec: float = 120.0,
) -> List[List[float]]:
    """POST texts to a remote /embed endpoint; return embedding rows."""
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx required for remote embedding endpoint") from exc

    if isinstance(texts, str):
        batch = [texts]
    else:
        batch = list(texts)
    url = endpoint_url.strip()
    if not url:
        raise ValueError("embedding endpoint URL is empty")
    if not url.endswith("/embed"):
        url = url.rstrip("/") + "/embed"

    with httpx.Client(timeout=timeout_sec) as client:
        resp = client.post(
            url,
            json={"texts": batch, "model": model_id, "normalize": normalize},
        )
    resp.raise_for_status()
    data = resp.json()
    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list):
        raise ValueError("remote embed response missing embeddings list")
    rows: List[List[float]] = []
    for row in embeddings:
        if not isinstance(row, list):
            raise ValueError("invalid embedding row")
        rows.append([float(x) for x in row])
    if len(rows) != len(batch):
        raise ValueError("remote embed row count mismatch")
    return rows

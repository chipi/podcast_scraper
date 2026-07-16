"""Semantic ranking of insight->topic ABOUT edges.

Replaces the legacy all-to-all (insights × topics) cross-product with a
top-K semantic filter: for each insight, emit ABOUT edges only to the
topics with highest cosine similarity between insight text and topic
label, subject to a confidence floor.

Defaults (``ABOUT_EDGE_DEFAULT_TOP_K = 2``, ``ABOUT_EDGE_DEFAULT_FLOOR =
0.25``) were chosen from the sweep in
``scripts/validate/sweep_about_edges.py`` against the
``my-manual-run4`` corpus: ~73% reduction in edge count with kept-edge
median cosine 0.442 (vs. raw cross-product median 0.262). See #664.

Insights whose best topic match is below the floor emit no ABOUT
edges — they remain in the graph as Insight nodes, just untagged.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ABOUT_EDGE_DEFAULT_TOP_K = 2
ABOUT_EDGE_DEFAULT_FLOOR = 0.25
ABOUT_EDGE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_encoder_cache: Dict[str, Any] = {}
# The GI stage runs across worker threads. Without this lock two threads miss the cache together,
# both construct a SentenceTransformer, and torch's lazy meta-device init races — raising
# "Cannot copy out of meta tensor; no data!". The model loads perfectly well on its own, so the
# failure looks like a broken install and is really a missing lock.
#
# It does not crash loudly enough to notice: `_rank_about_edges_for_insights` throws, the GI
# artifact build for that episode collapses, and the episode lands with 1 insight instead of 10
# while the run reports success. Measured on the DGX pilot — one episode in three, every run.
#
# Every sibling loader (nli_loader, embedding_loader, extractive_qa) already guards its cache this
# way; this one was simply missed.
_encoder_cache_lock = threading.Lock()


def _get_encoder(model_id: str) -> Any:
    encoder = _encoder_cache.get(model_id)
    if encoder is not None:
        return encoder

    with _encoder_cache_lock:
        # Re-check under the lock: another thread may have loaded it while we waited.
        encoder = _encoder_cache.get(model_id)
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(model_id)
            _encoder_cache[model_id] = encoder
    return encoder


def rank_about_edges(
    insight_texts: List[str],
    topic_specs: List[Tuple[str, str]],
    *,
    top_k: int = ABOUT_EDGE_DEFAULT_TOP_K,
    floor: float = ABOUT_EDGE_DEFAULT_FLOOR,
    encoder: Optional[Any] = None,
    embedding_model: str = ABOUT_EDGE_EMBEDDING_MODEL,
) -> List[List[Tuple[str, float]]]:
    """Rank (topic_id, cosine) pairs per insight; return top-k above floor.

    Args:
        insight_texts: Per-insight text (order preserved in result).
        topic_specs: ``[(topic_node_id, display_label), ...]`` — the same
            structure produced by ``_dedupe_topic_node_specs`` in
            ``pipeline.py``.
        top_k: Keep at most this many topics per insight.
        floor: Discard pairs with cosine below this threshold.
        encoder: Optional pre-loaded SentenceTransformer-like object with
            ``encode(texts, normalize_embeddings=True)``. When ``None``,
            loads and caches the default model.
        embedding_model: Model id for the default encoder.

    Returns:
        One list per insight (same order as ``insight_texts``) containing
        up to ``top_k`` ``(topic_id, cosine)`` tuples sorted by cosine
        descending. Insights with no topic above the floor return ``[]``.
    """
    if not insight_texts or not topic_specs:
        return [[] for _ in insight_texts]

    import numpy as np

    topic_ids = [tid for tid, _ in topic_specs]
    topic_texts = [label or tid for tid, label in topic_specs]
    safe_insights = [(t or "") for t in insight_texts]

    enc = encoder if encoder is not None else _get_encoder(embedding_model)
    emb_i = enc.encode(safe_insights, normalize_embeddings=True)
    emb_t = enc.encode(topic_texts, normalize_embeddings=True)
    sim = np.asarray(emb_i) @ np.asarray(emb_t).T

    results: List[List[Tuple[str, float]]] = []
    for row in sim:
        scored = [(topic_ids[j], float(row[j])) for j in range(len(row)) if float(row[j]) >= floor]
        scored.sort(key=lambda p: p[1], reverse=True)
        results.append(scored[:top_k])
    return results

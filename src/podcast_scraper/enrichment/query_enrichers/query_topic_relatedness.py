"""``query_topic_relatedness`` — decorate hits with topic_similarity Top-K.

Reads the chunk-3 corpus-scope ``enrichments/topic_similarity.json``
output and annotates each search hit with the precomputed Top-K
similar topics for any ``topic_id`` the hit mentions.

Per-hit shape after enrichment::

    {
        "topic_id": "topic:foo",
        "related_topics": [
            {"topic_id": "topic:bar", "topic_label": "Bar", "similarity": 0.83},
            ...
        ]
    }

If the corpus has no ``topic_similarity.json`` (e.g. ``airgapped_thin``
where topic_similarity is disabled), the enricher passes the envelope
through unmodified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from podcast_scraper.enrichment.protocol import EnricherTier
from podcast_scraper.enrichment.query_protocol import (
    QueryEnricherManifest,
    QueryResultEnvelope,
)


def _load_topic_similarity(corpus_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Read enrichments/topic_similarity.json and return topic_id → top_k."""
    path = corpus_root / "enrichments" / "topic_similarity.json"
    if not path.is_file():
        return {}
    try:
        env = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    data = env.get("data") if isinstance(env, dict) else None
    topics = data.get("topics") if isinstance(data, dict) else None
    if not isinstance(topics, list):
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for row in topics:
        if not isinstance(row, dict):
            continue
        tid = row.get("topic_id")
        top_k = row.get("top_k")
        if isinstance(tid, str) and isinstance(top_k, list):
            out[tid] = top_k
    return out


def _topic_id_of(hit: dict[str, Any]) -> str | None:
    """Best-effort extraction of a topic_id from a hit."""
    tid = hit.get("topic_id")
    if isinstance(tid, str):
        return tid
    meta = hit.get("metadata") or {}
    if isinstance(meta, dict):
        meta_tid = meta.get("topic_id")
        if isinstance(meta_tid, str):
            return meta_tid
    return None


class QueryTopicRelatednessEnricher:
    """Deterministic per-request query enricher: adds Top-K related topics."""

    manifest = QueryEnricherManifest(
        id="query_topic_relatedness",
        version="1.0.0",
        tier=EnricherTier.DETERMINISTIC,
        description="Decorate search hits with precomputed topic_similarity ranks.",
    )

    def __init__(self, *, corpus_root_provider: Callable[[], Path], max_per_hit: int = 5) -> None:
        if max_per_hit < 1:
            raise ValueError("max_per_hit must be >= 1")
        self._corpus_root_provider = corpus_root_provider
        self._max_per_hit = max_per_hit

    async def enrich_query_result(
        self,
        *,
        envelope: QueryResultEnvelope,
        config: dict[str, Any],
        ctx: Any,  # RunContext — kept loose to avoid circular import
    ) -> QueryResultEnvelope:
        """QueryEnricher impl — decorate hits with topic_similarity Top-K."""
        max_per_hit = int(config.get("max_per_hit", self._max_per_hit))
        if max_per_hit < 1:
            max_per_hit = self._max_per_hit
        corpus_root = self._corpus_root_provider()
        similarity = _load_topic_similarity(corpus_root)
        if not similarity:
            envelope.annotations.setdefault(
                "query_topic_relatedness",
                {"available": False, "reason": "no topic_similarity output"},
            )
            return envelope
        decorated = 0
        for hit in envelope.hits:
            tid = _topic_id_of(hit)
            if not tid:
                continue
            top_k = similarity.get(tid)
            if not top_k:
                continue
            hit["related_topics"] = top_k[:max_per_hit]
            decorated += 1
        envelope.annotations["query_topic_relatedness"] = {
            "available": True,
            "decorated_hits": decorated,
            "max_per_hit": max_per_hit,
        }
        return envelope


__all__ = ["QueryTopicRelatednessEnricher"]

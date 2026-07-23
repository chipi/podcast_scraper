"""Server-side result-set operators for /api/search — Search v3 §S4b.

Both operators run **Python-side AFTER** the hybrid retrieval pipeline
(``rrf_fuse`` → filters → enrich_lift_and_slice → optional dedupe) returns
its ``top_k`` page. There is **no** new native combine site — the whole
S4b surface is a pure post-processing pass over hit metadata + JSON
lookup, keeping the #1205 SIGSEGV class of bug out of scope. The
``make lint-search-v3`` guard covers this file the same way it covers
every other module under ``search/``.

Two operators today, both additive on the existing ``CorpusSearchApiResponse``:

* ``operator=cluster`` — group the hit page by topic-cluster (from the
  shipped ``enrichments/topic_clusters.json`` join in
  ``_attach_topic_cluster_metadata``) with fallback to theme-cluster
  (via ``theme_clusters.consumer_theme_cluster_map``) and finally to a
  single-topic anchor. Hits with no resolvable cluster surface land in
  an ungrouped bucket (``cluster_id=null``).
* ``operator=consensus`` — read ``enrichments/topic_consensus.json``
  (produced by the shipped ``topic_consensus`` enricher, ADR-108,
  precision ~0.91 on prod-v2) and filter pairs to topics surfaced in
  the current hit page.

Callers are expected to hint the server with ``top_k * 3`` over-fetch
when requesting an operator so the aggregation has a meaningful
sample to group / filter over — the callers own that policy; this
module aggregates whatever slice it gets.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from podcast_scraper.search.theme_clusters import (
    consumer_theme_cluster_map,
    THEME_CLUSTERS_REL,
)

_logger = logging.getLogger(__name__)

_TOPIC_CONSENSUS_REL = "enrichments/topic_consensus.json"


# --------------------------------------------------------------------------
# Cluster operator
# --------------------------------------------------------------------------


def _topic_id_from_hit(hit_meta: dict[str, Any]) -> str | None:
    """Best-effort topic id per hit.

    * ``kg_topic`` hits carry ``source_id`` = ``topic:…`` directly.
    * Insight / quote / summary hits carry ``about_topic_id`` when the
      GI ABOUT edge was joined. Otherwise fall back to any single
      ``topic_ids`` list carried by the metadata (first entry).
    """
    doc_type = str(hit_meta.get("doc_type") or "")
    if doc_type == "kg_topic":
        sid = hit_meta.get("source_id")
        if isinstance(sid, str) and sid.strip():
            return sid.strip()
    about = hit_meta.get("about_topic_id")
    if isinstance(about, str) and about.strip():
        return about.strip()
    ids = hit_meta.get("topic_ids")
    if isinstance(ids, list) and ids:
        first = ids[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    return None


def _topic_label_from_hit(hit_meta: dict[str, Any]) -> str | None:
    """Best-effort topic label per hit — kg_topic hits carry ``topic_label``."""
    lbl = hit_meta.get("topic_label")
    if isinstance(lbl, str) and lbl.strip():
        return lbl.strip()
    return None


def _hit_cluster_key(
    hit_meta: dict[str, Any],
    theme_map: dict[str, dict[str, Any]],
) -> tuple[str, str, str] | None:
    """Return ``(kind, id, label)`` for the hit's best cluster surface, or None.

    Preference order:
      1. Topic cluster (``metadata.topic_cluster.topic_cluster_compound_id``)
         — attached by ``_attach_topic_cluster_metadata`` for ``kg_topic`` hits.
      2. Theme cluster — look up the hit's topic id in the theme cluster map.
      3. Fallback: bare topic id (single-topic pseudo-group so tightly-related
         hits still merge without a cluster surface).
    """
    tc = hit_meta.get("topic_cluster")
    if isinstance(tc, dict):
        cid = tc.get("topic_cluster_compound_id") or tc.get("cluster_id")
        clabel = tc.get("label") or tc.get("topic_cluster_label")
        if isinstance(cid, str) and cid.strip():
            return ("topic_cluster", cid.strip(), str(clabel or cid).strip())

    topic_id = _topic_id_from_hit(hit_meta)
    if topic_id and theme_map:
        theme = theme_map.get(topic_id)
        if theme:
            thc = theme.get("theme_cluster_id")
            tlabel = theme.get("theme_cluster_label") or thc
            if isinstance(thc, str) and thc.strip():
                return ("theme_cluster", thc.strip(), str(tlabel or thc).strip())

    if topic_id:
        label = _topic_label_from_hit(hit_meta) or topic_id
        return ("topic", topic_id, label)
    return None


def cluster_hits(
    hits: list[dict[str, Any]],
    corpus_root: Path,
) -> list[dict[str, Any]]:
    """Group ``hits`` (dicts with a ``metadata`` sub-dict) by cluster.

    Returns a list of dicts matching ``SearchClusterGroupModel``, ordered
    by descending group size, with an ``ungrouped`` bucket last when
    non-empty. Hit indices point back into the caller's ``hits`` list in
    original order so the client can render groups without re-sorting.
    """
    theme_map = consumer_theme_cluster_map(corpus_root)
    # (kind, cluster_id) -> {label, indices}
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    ungrouped: list[int] = []
    for idx, hit in enumerate(hits):
        meta = hit.get("metadata")
        if not isinstance(meta, dict):
            ungrouped.append(idx)
            continue
        key = _hit_cluster_key(meta, theme_map)
        if key is None:
            ungrouped.append(idx)
            continue
        kind, cid, label = key
        bucket = groups.setdefault((kind, cid), {"label": label, "indices": []})
        # Keep the first non-fallback label if a better one shows up later.
        if bucket["label"] == cid and label != cid:
            bucket["label"] = label
        bucket["indices"].append(idx)

    out: list[dict[str, Any]] = []
    # Rank real clusters by descending size (ties: preserve insertion order
    # via Python's stable sort).
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]["indices"]), kv[0][0], kv[0][1]))
    for (kind, cid), bucket in ordered:
        out.append(
            {
                "cluster_id": cid,
                "cluster_kind": kind,
                "label": bucket["label"],
                "size": len(bucket["indices"]),
                "hit_indices": bucket["indices"],
            }
        )
    if ungrouped:
        out.append(
            {
                "cluster_id": None,
                "cluster_kind": "ungrouped",
                "label": "Ungrouped",
                "size": len(ungrouped),
                "hit_indices": ungrouped,
            }
        )
    return out


# --------------------------------------------------------------------------
# Consensus operator
# --------------------------------------------------------------------------


def _load_consensus_payload(corpus_root: Path) -> dict[str, Any] | None:
    """Read ``enrichments/topic_consensus.json`` — returns None on any I/O error."""
    path = corpus_root / _TOPIC_CONSENSUS_REL
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _logger.warning("topic_consensus: failed to read %s: %s", path, exc)
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def _relevant_topic_ids(
    hits: list[dict[str, Any]],
    top_n_fallback: int = 5,
) -> list[str]:
    """Topic ids most-relevant to the hit set — direct kg_topic hits win, otherwise
    the most-referenced topics via metadata (``about_topic_id`` / ``topic_ids``)."""
    direct: list[str] = []
    seen: set[str] = set()
    counter: Counter[str] = Counter()
    for hit in hits:
        meta = hit.get("metadata")
        if not isinstance(meta, dict):
            continue
        doc_type = str(meta.get("doc_type") or "")
        sid = meta.get("source_id")
        if doc_type == "kg_topic" and isinstance(sid, str) and sid.strip():
            tid = sid.strip()
            if tid not in seen:
                seen.add(tid)
                direct.append(tid)
        # Also count references for the fallback path.
        for referenced in _iter_referenced_topics(meta):
            counter[referenced] += 1
    if direct:
        return direct
    return [tid for tid, _ in counter.most_common(top_n_fallback)]


def _iter_referenced_topics(meta: dict[str, Any]):
    """Yield topic ids referenced by an insight/quote/summary hit's metadata."""
    about = meta.get("about_topic_id")
    if isinstance(about, str) and about.strip():
        yield about.strip()
    ids = meta.get("topic_ids")
    if isinstance(ids, list):
        for x in ids:
            if isinstance(x, str) and x.strip():
                yield x.strip()


def consensus_pairs_for_hits(
    hits: list[dict[str, Any]],
    corpus_root: Path,
    *,
    max_pairs: int = 20,
) -> list[dict[str, Any]]:
    """Filter ``topic_consensus.json`` pairs to topics surfaced in the hit set.

    Returns ``[]`` when the enricher output is missing / empty / unreadable.
    Never raises — errors are logged and the caller degrades gracefully to
    "no consensus pairs for this query".
    """
    payload = _load_consensus_payload(corpus_root)
    if not payload:
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    pairs_raw = data.get("consensus")
    if not isinstance(pairs_raw, list) or not pairs_raw:
        return []

    relevant = set(_relevant_topic_ids(hits))
    if not relevant:
        # Degrade gracefully: no way to filter → return the strongest few
        # pairs by (lowest contradiction, highest cosine) up to max_pairs.
        candidates: list[dict[str, Any]] = [p for p in pairs_raw if isinstance(p, dict)]
        candidates.sort(
            key=lambda p: (
                float(p.get("contradiction_score") or 1.0),
                -float(p.get("cosine_similarity") or 0.0),
            ),
        )
        return _pair_dicts(candidates[:max_pairs])

    matched: list[dict[str, Any]] = []
    for pair in pairs_raw:
        if not isinstance(pair, dict):
            continue
        tid = pair.get("topic_id")
        if isinstance(tid, str) and tid.strip() and tid.strip() in relevant:
            matched.append(pair)
        if len(matched) >= max_pairs:
            break
    return _pair_dicts(matched)


def _pair_dicts(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize enricher-pair dicts into the response shape.

    Never fails on unexpected shapes; missing fields fall back to safe
    defaults (empty strings, ``None`` for optional labels, ``0.0`` /
    ``None`` for scores).
    """
    out: list[dict[str, Any]] = []
    for p in pairs:
        try:
            contradiction = float(p.get("contradiction_score") or 0.0)
        except (TypeError, ValueError):
            contradiction = 0.0
        cos = p.get("cosine_similarity")
        try:
            cos_val = float(cos) if cos is not None else None
        except (TypeError, ValueError):
            cos_val = None
        out.append(
            {
                "topic_id": str(p.get("topic_id") or ""),
                "topic_label": _maybe_str(p.get("topic_label")),
                "person_a_id": str(p.get("person_a_id") or ""),
                "person_a_label": _maybe_str(p.get("person_a_label")),
                "person_b_id": str(p.get("person_b_id") or ""),
                "person_b_label": _maybe_str(p.get("person_b_label")),
                "insight_a_id": str(p.get("insight_a_id") or ""),
                "insight_b_id": str(p.get("insight_b_id") or ""),
                "insight_a_text": str(p.get("insight_a_text") or ""),
                "insight_b_text": str(p.get("insight_b_text") or ""),
                "contradiction_score": contradiction,
                "cosine_similarity": cos_val,
            }
        )
    return out


def _maybe_str(v: Any) -> str | None:
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


__all__ = [
    "THEME_CLUSTERS_REL",  # re-export for callers that need the path constant
    "cluster_hits",
    "consensus_pairs_for_hits",
]

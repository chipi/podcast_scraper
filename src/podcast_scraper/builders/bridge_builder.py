"""RFC-072 per-episode ``bridge.json`` — canonical CIL identities across GI and KG."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Set

logger = logging.getLogger(__name__)

_CIL_PREFIXES = ("person:", "org:", "topic:")

# Fuzzy reconciliation: cosine similarity threshold for merging single-layer
# identities whose exact IDs don't match.  Only applied to identities of the
# same CIL type (person↔person, topic↔topic) where one is GI-only and the
# other is KG-only.
_FUZZY_MERGE_THRESHOLD = 0.85


def _strip_layer_prefixes(raw_id: str) -> str:
    s = str(raw_id).strip()
    if s.startswith("g:"):
        s = s[2:]
    if s.startswith("k:") and not s.startswith("kg:"):
        s = s[2:]
    if s.startswith("kg:"):
        s = s[3:]
    return s


def _node_aliases(props: Mapping[str, Any]) -> List[str]:
    raw = props.get("aliases")
    if raw is None:
        return []
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    if isinstance(raw, list):
        out: List[str] = []
        for x in raw:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out
    return []


def _merge_display_name(existing: str, incoming: str) -> str:
    a = (existing or "").strip()
    b = (incoming or "").strip()
    if not a:
        return b
    if not b:
        return a
    if a == b:
        return a
    return a if len(a) >= len(b) else b


def _collect_identity(
    identities: MutableMapping[str, Dict[str, Any]],
    node: Mapping[str, Any],
    *,
    source: str,
) -> None:
    nid_raw = node.get("id")
    if nid_raw is None:
        return
    nid = _strip_layer_prefixes(str(nid_raw))
    if not nid.startswith(_CIL_PREFIXES):
        return
    kind = nid.split(":", 1)[0]
    props_any = node.get("properties")
    props: Dict[str, Any] = props_any if isinstance(props_any, dict) else {}
    node_aliases = _node_aliases(props)
    name_prop = props.get("name")
    label_prop = props.get("label")
    display = ""
    if isinstance(name_prop, str) and name_prop.strip():
        display = name_prop.strip()
    elif isinstance(label_prop, str) and label_prop.strip():
        display = label_prop.strip()

    if nid in identities:
        ex = identities[nid]
        ex["sources"][source] = True
        merged_aliases: Set[str] = set(ex.get("aliases") or [])
        merged_aliases.update(node_aliases)
        ex["aliases"] = sorted(merged_aliases)
        ex["display_name"] = _merge_display_name(str(ex.get("display_name", "")), display)
    else:
        identities[nid] = {
            "id": nid,
            "type": kind,
            "display_name": display,
            "aliases": list(node_aliases),
            "sources": {"gi": source == "gi", "kg": source == "kg"},
        }


def _fuzzy_reconcile(
    identities: MutableMapping[str, Dict[str, Any]],
    threshold: float = _FUZZY_MERGE_THRESHOLD,
    embedder: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Post-pass: merge single-layer identities with high display-name similarity.

    After exact-ID matching, some identities remain GI-only or KG-only because
    the input names differ (e.g., "John Smith" vs "Dr. John Smith").  This pass
    finds unmatched pairs of the same CIL type, computes embedding cosine
    similarity on their display names, and merges those above *threshold*.

    Returns a list of ``{"gi_id": ..., "kg_id": ..., "similarity": ...}`` dicts
    describing which identities were fuzzy-merged (for logging/audit).
    """
    if embedder is None:
        try:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            logger.debug("sentence-transformers not available; skipping fuzzy reconciliation")
            return []

    # Partition single-layer identities by CIL type
    gi_only: Dict[str, Dict[str, Any]] = {}
    kg_only: Dict[str, Dict[str, Any]] = {}
    for nid, rec in identities.items():
        src = rec.get("sources", {})
        if src.get("gi") and not src.get("kg"):
            gi_only[nid] = rec
        elif src.get("kg") and not src.get("gi"):
            kg_only[nid] = rec

    if not gi_only or not kg_only:
        return []

    merges: List[Dict[str, Any]] = []

    for cil_type in ("person", "org", "topic"):
        gi_items = [(nid, rec) for nid, rec in gi_only.items() if rec.get("type") == cil_type]
        kg_items = [(nid, rec) for nid, rec in kg_only.items() if rec.get("type") == cil_type]
        if not gi_items or not kg_items:
            continue

        gi_names = [rec.get("display_name", "") for _, rec in gi_items]
        kg_names = [rec.get("display_name", "") for _, rec in kg_items]

        # Skip if any display names are empty
        if not all(gi_names) or not all(kg_names):
            continue

        import numpy as np

        gi_embs = embedder.encode(gi_names, normalize_embeddings=True)
        kg_embs = embedder.encode(kg_names, normalize_embeddings=True)
        sim = np.dot(gi_embs, kg_embs.T)

        # Greedy best-match: for each GI identity, find best KG match
        used_kg: Set[int] = set()
        for gi_idx in range(len(gi_items)):
            best_kg_idx = -1
            best_sim = -1.0
            for kg_idx in range(len(kg_items)):
                if kg_idx in used_kg:
                    continue
                s = float(sim[gi_idx, kg_idx])
                if s > best_sim:
                    best_sim = s
                    best_kg_idx = kg_idx
            if best_kg_idx >= 0 and best_sim >= threshold:
                used_kg.add(best_kg_idx)
                gi_nid, gi_rec = gi_items[gi_idx]
                kg_nid, kg_rec = kg_items[best_kg_idx]

                # Merge KG identity into GI identity (keep GI's id as canonical)
                gi_rec["sources"]["kg"] = True
                merged_aliases: Set[str] = set(gi_rec.get("aliases") or [])
                merged_aliases.update(kg_rec.get("aliases") or [])
                # Add the KG display name as an alias if different
                kg_display = kg_rec.get("display_name", "")
                gi_display = gi_rec.get("display_name", "")
                if kg_display and kg_display != gi_display:
                    merged_aliases.add(kg_display)
                # Also add the KG id as an alias for traceability
                merged_aliases.add(kg_nid)
                gi_rec["aliases"] = sorted(merged_aliases)
                gi_rec["display_name"] = _merge_display_name(gi_display, kg_display)

                # Remove KG identity (now merged into GI)
                del identities[kg_nid]

                merges.append(
                    {
                        "gi_id": gi_nid,
                        "kg_id": kg_nid,
                        "gi_name": gi_display,
                        "kg_name": kg_display,
                        "similarity": round(best_sim, 4),
                    }
                )
                logger.info(
                    "Fuzzy merge: %s (%s) ← %s (%s) [sim=%.3f]",
                    gi_nid,
                    gi_display,
                    kg_nid,
                    kg_display,
                    best_sim,
                )

    return merges


def build_bridge(
    episode_id: str,
    gi_artifact: Mapping[str, Any] | None,
    kg_artifact: Mapping[str, Any] | None,
    *,
    emitted_at: datetime | None = None,
    fuzzy_reconcile: bool = True,
    fuzzy_threshold: float = _FUZZY_MERGE_THRESHOLD,
    embedder: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build a ``bridge.json`` payload (RFC-072 §4).

    Args:
        episode_id: Episode identifier.
        gi_artifact: GI artifact dict (or None).
        kg_artifact: KG artifact dict (or None).
        emitted_at: Optional timestamp override.
        fuzzy_reconcile: If True (default), run a post-pass that merges
            single-layer identities with high display-name embedding similarity.
            Requires ``sentence-transformers``; degrades gracefully if unavailable.
        fuzzy_threshold: Cosine similarity threshold for fuzzy merging (default 0.85).
        embedder: Optional pre-loaded SentenceTransformer instance (avoids re-loading
            per episode in batch runs).

    Returns:
        bridge.json payload dict.
    """
    identities: Dict[str, Dict[str, Any]] = {}
    gi = gi_artifact or {}
    kg = kg_artifact or {}
    for node in gi.get("nodes") or []:
        if isinstance(node, dict):
            _collect_identity(identities, node, source="gi")
    for node in kg.get("nodes") or []:
        if isinstance(node, dict):
            _collect_identity(identities, node, source="kg")

    # Post-pass: fuzzy reconciliation for single-layer identities
    fuzzy_merges: List[Dict[str, Any]] = []
    if fuzzy_reconcile and identities:
        fuzzy_merges = _fuzzy_reconcile(identities, threshold=fuzzy_threshold, embedder=embedder)

    when = emitted_at or datetime.now(timezone.utc)
    ts = when.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    id_list = sorted(identities.values(), key=lambda x: str(x["id"]))

    result: Dict[str, Any] = {
        "schema_version": "1.0",
        "episode_id": str(episode_id),
        "emitted_at": ts,
        "identities": id_list,
    }
    if fuzzy_merges:
        result["fuzzy_merges"] = fuzzy_merges
    return result

"""Corpus-wide entity canonicalization — cross-episode spelling drift (#852).

The same person/org appears under different slugs across episodes of a show
(`Cargil`/`Cargill`, `Data Bricks`/`Databricks`, `Tracy`/`Tracey Alloway`) — ASR
proper-noun drift the within-episode fix (#851) and the per-episode bridge can't
catch. This builds a corpus-wide ``variant_id → canonical_id`` map, the missing
analog to topic clustering (RFC-075) for entities.

Design (evidence: ~95% of drift is same-show, frequency-dominant canonical):

- **Conservative, string-based** — names drift by *spelling*, not semantics, so we
  cluster by string similarity, not embeddings.
- **Same-show required** — two variants merge only if they share a podcast (cuts
  false merges; 95% of drift is same-show anyway).
- **Canonical = highest frequency** — the spelling in the most episodes wins
  (`Odd Lots` ×13 over the 1-episode garbles).
- **Guards (the balanced check):** acronym guard (`UPS` ≠ `USPS`); version /
  distinguishing-token guard (`Claude` ≠ `Claude 3`); a differing *content* token
  that is not itself a spelling variant blocks the merge (`Bloomberg Audio` ≠
  `Bloomberg Media Studios`).

The map is applied at read-time via ``CorpusGraph.build(identity_map=…)`` —
reversible, no artifact rewrite. Threshold precision tuning is deferred to
autoresearch. See issue #852.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from ..builders.bridge_builder import strip_layer_prefixes
from .filters import _clean_entity_name, _is_acronymish

logger = logging.getLogger(__name__)

ENTITY_CLUSTERS_SCHEMA_VERSION = "1.0"

# Conservative thresholds (tunable → autoresearch).
_TOKEN_RATIO = 0.78  # per-aligned-token spelling-variant floor
_OVERALL_RATIO = 0.85  # whole-string floor
_VERSION_TOKEN_RE = re.compile(r"\d")  # a differing token containing a digit blocks merge


@dataclass
class EntityCandidate:
    """One canonical entity id observed across the corpus."""

    id: str
    kind: str  # "person" | "org"
    name: str  # representative display name (most frequent)
    episodes: Set[str] = field(default_factory=set)
    shows: Set[str] = field(default_factory=set)
    _name_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def freq(self) -> int:
        return len(self.episodes)


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _are_xep_variants(name_a: str, name_b: str, kind: str) -> bool:
    """Conservative cross-episode variant test (spelling drift, not distinct names)."""
    a, b = _clean_entity_name(name_a), _clean_entity_name(name_b)
    if not a or not b:
        return False
    if a == b:
        return True
    # Acronyms never fuzzy-merge (UPS != USPS).
    if _is_acronymish(name_a, a) or _is_acronymish(name_b, b):
        return False
    # Spacing variants: "Data Bricks" == "Databricks", "Chat GPT" == "ChatGPT".
    if a.replace(" ", "") == b.replace(" ", "") and a.replace(" ", ""):
        return True
    ta, tb = a.split(), b.split()
    # Different token counts → only the despaced case above may merge; otherwise a
    # version/extra token (Claude vs Claude 3) → do not merge.
    if len(ta) != len(tb):
        return False
    if _ratio(a, b) < _OVERALL_RATIO:
        return False
    # Token-aligned: every differing token pair must be a spelling variant, not a
    # distinct content word (audio vs media) or a version token (3, v2).
    for x, y in zip(ta, tb):
        if x == y:
            continue
        if _VERSION_TOKEN_RE.search(x) or _VERSION_TOKEN_RE.search(y):
            return False  # numeric/version distinction
        if _ratio(x, y) < _TOKEN_RATIO:
            return False  # distinct words, not a spelling variant
    return True


def collect_entity_candidates(corpus_dir: Path | str) -> Dict[str, EntityCandidate]:
    """Aggregate person/org entities corpus-wide with episode frequency + shows."""
    from .corpus import load_kg_artifacts, scan_kg_artifact_paths

    out: Dict[str, EntityCandidate] = {}
    for _path, data in load_kg_artifacts(scan_kg_artifact_paths(Path(corpus_dir))):
        episode_id = str(data.get("episode_id") or "")
        show = ""
        for node in data.get("nodes") or []:
            if isinstance(node, dict) and node.get("type") == "Episode":
                props = node.get("properties") or {}
                show = str(props.get("podcast_id") or props.get("feed_id") or "")
                break
        for node in data.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            nid = strip_layer_prefixes(str(node.get("id") or ""))
            kind = nid.split(":", 1)[0]
            if kind not in ("person", "org"):
                continue
            props = node.get("properties") or {}
            name = str(props.get("name") or props.get("label") or "").strip()
            if not name:
                continue
            cand = out.get(nid)
            if cand is None:
                cand = EntityCandidate(id=nid, kind=kind, name=name)
                out[nid] = cand
            if episode_id:
                cand.episodes.add(episode_id)
            if show:
                cand.shows.add(show)
            cand._name_counts[name] = cand._name_counts.get(name, 0) + 1
    # Representative display name = most frequent surface form.
    for cand in out.values():
        if cand._name_counts:
            cand.name = max(cand._name_counts.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
    return out


def _pick_canonical(members: List[EntityCandidate]) -> EntityCandidate:
    """Highest frequency wins; tie → longest name, then lexical."""
    return sorted(members, key=lambda c: (-c.freq, -len(c.name), c.name.lower()))[0]


def build_entity_canonical_map(
    candidates: Dict[str, EntityCandidate],
    *,
    same_show_required: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Cluster variants and return ``(entity_clusters_payload, variant→canonical map)``."""
    by_kind: Dict[str, List[EntityCandidate]] = {}
    for cand in candidates.values():
        by_kind.setdefault(cand.kind, []).append(cand)

    id_map: Dict[str, str] = {}
    clusters_out: List[Dict[str, Any]] = []

    for kind, items in sorted(by_kind.items()):
        # High-frequency first so the dominant spelling seeds (and wins) each cluster.
        items_sorted = sorted(items, key=lambda c: (-c.freq, c.name.lower()))
        clusters: List[List[EntityCandidate]] = []
        for cand in items_sorted:
            for cluster in clusters:
                if same_show_required and not any(cand.shows & m.shows for m in cluster):
                    continue
                if any(_are_xep_variants(cand.name, m.name, kind) for m in cluster):
                    cluster.append(cand)
                    break
            else:
                clusters.append([cand])

        for cluster in clusters:
            if len(cluster) < 2:
                continue
            canonical = _pick_canonical(cluster)
            members_json = []
            for m in sorted(cluster, key=lambda c: (-c.freq, c.id)):
                if m.id != canonical.id:
                    id_map[m.id] = canonical.id
                members_json.append({"id": m.id, "name": m.name, "episode_count": m.freq})
            clusters_out.append(
                {
                    "canonical_id": canonical.id,
                    "canonical_name": canonical.name,
                    "member_count": len(cluster),
                    "members": members_json,
                }
            )

    payload = {
        "schema_version": ENTITY_CLUSTERS_SCHEMA_VERSION,
        "same_show_required": same_show_required,
        "entity_count": len(candidates),
        "cluster_count": len(clusters_out),
        "merged_variants": len(id_map),
        "clusters": clusters_out,
    }
    return payload, id_map


def build_entity_id_map(
    corpus_dir: Path | str, *, same_show_required: bool = True
) -> Dict[str, str]:
    """Convenience: corpus → ``variant_id → canonical_id`` map for ``CorpusGraph``."""
    candidates = collect_entity_candidates(corpus_dir)
    _payload, id_map = build_entity_canonical_map(candidates, same_show_required=same_show_required)
    return id_map


def id_map_from_clusters_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    """Reconstruct the ``variant_id → canonical_id`` map from a saved payload."""
    out: Dict[str, str] = {}
    for cluster in payload.get("clusters") or []:
        canonical = cluster.get("canonical_id")
        if not canonical:
            continue
        for member in cluster.get("members") or []:
            mid = member.get("id")
            if mid and mid != canonical:
                out[mid] = canonical
    return out


__all__ = [
    "EntityCandidate",
    "build_entity_canonical_map",
    "build_entity_id_map",
    "collect_entity_candidates",
    "id_map_from_clusters_payload",
]

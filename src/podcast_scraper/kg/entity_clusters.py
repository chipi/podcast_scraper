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

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from ..builders.bridge_builder import strip_layer_prefixes
from ..identity.person_match import cluster_variants as _are_xep_variants
from ..perf_cache import cache_peek, corpus_mtime, get_or_compute

logger = logging.getLogger(__name__)

ENTITY_CLUSTERS_SCHEMA_VERSION = "1.0"


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


def collect_entity_candidates(corpus_dir: Path | str) -> Dict[str, EntityCandidate]:
    """Aggregate person/org entities corpus-wide with episode frequency + shows."""
    from .corpus import load_kg_artifacts, newest_run_artifact_paths, scan_kg_artifact_paths

    out: Dict[str, EntityCandidate] = {}
    # NEWEST RUN PER EPISODE — the same membership rule the catalog uses. A reprocessed
    # episode's superseded run was still voting on who a person is (310 of 2,257 kg files on
    # the production snapshot), which is exactly backwards after a `relabel_only` repair.
    _root = Path(corpus_dir)
    _paths = newest_run_artifact_paths(_root, scan_kg_artifact_paths(_root), ".kg.json")
    for _path, data in load_kg_artifacts(_paths):
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
        if not same_show_required:
            for cand in items_sorted:
                for cluster in clusters:
                    if any(_are_xep_variants(cand.name, m.name, kind) for m in cluster):
                        cluster.append(cand)
                        break
                else:
                    clusters.append([cand])
        else:
            # THE SHOW INDEX. A cluster can only take a candidate it shares a show with, and the
            # old loop found that out by intersecting the candidate's shows with every member of
            # EVERY cluster — quadratic, minutes on the prod corpus, and the cold start every
            # restart waits on (entity search, share cards and trending block on this map).
            # `clusters_by_show` holds, per show, the clusters any member of which appeared on it:
            # exactly the clusters the old check let through. Visited in ascending index — the old
            # loop's order — so the first match, and the map, are identical.
            clusters_by_show: Dict[str, Set[int]] = {}
            for cand in items_sorted:
                eligible = sorted({i for s in cand.shows for i in clusters_by_show.get(s, ())})
                for i in eligible:
                    if any(_are_xep_variants(cand.name, m.name, kind) for m in clusters[i]):
                        clusters[i].append(cand)
                        break
                else:
                    clusters.append([cand])
                    i = len(clusters) - 1
                for s in cand.shows:
                    clusters_by_show.setdefault(s, set()).add(i)

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


#: One shared cache for the corpus-wide id map, and one lock in front of it.
_ID_MAP_NS = "kg.entity_id_map"
_ID_MAP_LOCK = threading.Lock()


_ISOLATED_BUILD_TIMEOUT_S = 1800.0


def _build_entity_id_map_isolated(root: Path, same_show_required: bool) -> Dict[str, str]:
    """:func:`build_entity_id_map` in a child process; in-process if that cannot run.

    The build is minutes of pure-Python string comparison (difflib) that never waits on IO, so in
    the api process it holds the GIL against every request thread. Measured on prod after the
    2026-10-06 deploys: while the cache warmer built it, anything that walks the corpus stalled --
    search 186-195 s (alone ~4 s), /api/index/stats 84-173 s, /api/artifacts up to 300 s -- and
    every post-deploy smoke failed in that window. A directory walk hands the GIL back on each
    filesystem call and then queues behind the builder for it. In a child process the build has
    its own interpreter; this thread only waits on the result, which holds no GIL.
    """
    import json
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "podcast_scraper.kg.entity_id_map_worker",
                str(root),
                "1" if same_show_required else "0",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=_ISOLATED_BUILD_TIMEOUT_S,
        )
        id_map = json.loads(proc.stdout)
        if not isinstance(id_map, dict):
            raise ValueError(f"worker returned {type(id_map).__name__}, not a map")
        return {str(k): str(v) for k, v in id_map.items()}
    except Exception as exc:  # noqa: BLE001 - a child that cannot run must not cost the map
        stderr = getattr(exc, "stderr", None) or ""
        logger.warning(
            "entity id map: isolated build failed (%s: %s%s); building in-process instead",
            type(exc).__name__,
            exc,
            f" | {stderr.strip()[-500:]}" if stderr else "",
        )
        return build_entity_id_map(root, same_show_required=same_show_required)


def cached_entity_id_map(
    corpus_dir: Path | str, *, same_show_required: bool = True, isolate_process: bool = False
) -> Dict[str, str]:
    """:func:`build_entity_id_map`, computed at most once per corpus change.

    WHY THIS EXISTS RATHER THAN THREE CALLERS EACH CALLING THE BUILDER. The map is a full corpus
    scan — **90 seconds** over the 2,257-episode production snapshot — and three surfaces need it:
    the app KG index, ``server/cil_queries`` and ``search/corpus_graph``. Each kept its own cache
    on the same token, so every ingest, migration or ``enrich-edges`` invalidated all three and
    paid for the same scan three times.

    ``perf_cache.get_or_compute`` deliberately runs ``compute()`` OUTSIDE its lock, which is right
    for the cheap projections it was built for and wrong here: on a miss, every arriving request
    thread starts its own 90-second scan. The lock below makes it single-flight — the first caller
    builds, the rest wait and take the cached value — and the double-check inside means a caller
    that waited does no work at all.

    Warmed by ``server/app_cache_warm``, so a reader normally never sees the miss. The warmer
    passes ``isolate_process=True`` (see :func:`_build_entity_id_map_isolated`): the build then
    runs in a child process, and readers arriving meanwhile wait on the lock without holding the
    GIL.
    """
    root = Path(corpus_dir)
    key = (str(root.resolve()), bool(same_show_required))
    token = corpus_mtime(root)
    hit = cache_peek(_ID_MAP_NS, key, token)
    if hit is not None:
        return dict(hit)
    with _ID_MAP_LOCK:
        # Re-check: another thread may have built it while we waited for the lock.
        return dict(
            get_or_compute(
                _ID_MAP_NS,
                key,
                token,
                (
                    (lambda: _build_entity_id_map_isolated(root, same_show_required))
                    if isolate_process
                    else (lambda: build_entity_id_map(root, same_show_required=same_show_required))
                ),
            )
        )


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
    "cached_entity_id_map",
    "collect_entity_candidates",
    "id_map_from_clusters_payload",
]

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

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Optional, Set, Tuple

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
    #: Primary language subtags of its episodes (``und`` for an episode whose metadata declares
    #: none); empty when no episode's metadata was found (RFC-131 S5b).
    languages: Set[str] = field(default_factory=set)
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
        language = _artifact_language(Path(_path))
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
            # The episode's own spelling when 0041 respelled the node for display (record 4g).
            name = str(
                props.get("respelled_from") or props.get("name") or props.get("label") or ""
            ).strip()
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
            if language is not None:
                cand.languages.add(language)
            cand._name_counts[name] = cand._name_counts.get(name, 0) + 1
    # Representative display name = most frequent surface form.
    for cand in out.values():
        if cand._name_counts:
            cand.name = max(cand._name_counts.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
    return out


def _artifact_language(kg_path: Path) -> Optional[str]:
    """The KG artifact's episode language (primary subtag), ``und`` when its metadata declares
    none, None when there is no readable metadata beside it. The catalog's rule
    (``languages.episode_language``)."""
    from ..languages import episode_language, primary_language

    name = kg_path.name
    if not name.endswith(".kg.json"):
        return None
    meta = kg_path.with_name(name[: -len(".kg.json")] + ".metadata.json")
    try:
        doc = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    lang = episode_language(doc)
    return primary_language(lang) if lang else "und"


def _pair_language(a: EntityCandidate, b: EntityCandidate) -> Optional[str]:
    """The one language two candidates share, ``und`` when they share none or several, None when
    neither has any (no metadata read: today's rule)."""
    if not a.languages and not b.languages:
        return None
    shared = a.languages & b.languages
    return next(iter(shared)) if len(shared) == 1 else "und"


def _words(name: str) -> int:
    from ..identity.person_match import name_words

    return name_words(name)


def _scoped(entity_id: str) -> bool:
    from ..identity.bare_name_scope import is_scoped_person_id

    return is_scoped_person_id(entity_id)


def _pick_canonical(members: List[EntityCandidate], keep: Collection[str] = ()) -> EntityCandidate:
    """A member already serving as a canonical id (``keep``) first; then highest frequency, the
    longest name, the name, and the id, so the pick never rests on the input order."""
    return sorted(
        members, key=lambda c: (c.id not in keep, -c.freq, -len(c.name), c.name.lower(), c.id)
    )[0]


_Guard = Callable[[EntityCandidate, EntityCandidate], Optional[Tuple[str, str]]]
_PairRule = Callable[[EntityCandidate, EntityCandidate], Optional[str]]
_LINKAGES = ("single", "no_one_word_bridge", "complete")


class _JoinTest:
    """May a candidate join a cluster? ``build_entity_canonical_map``'s join test, refusals kept."""

    def __init__(
        self,
        guard: Optional[_Guard],
        person_mode: str,
        linkage: str,
        pair_rule: Optional[_PairRule] = None,
    ) -> None:
        if linkage not in _LINKAGES:
            raise ValueError(f"unknown linkage: {linkage!r} (known: {list(_LINKAGES)})")
        self.guard, self.person_mode, self.linkage = guard, person_mode, linkage
        self.pair_rule = pair_rule
        self.blocked: List[List[str]] = []

    def _match(self, a: EntityCandidate, b: EntityCandidate, kind: str) -> bool:
        if kind == "person" and self.person_mode != "cluster":
            from ..identity.person_match import same_person

            return same_person(a.name, b.name, mode=self.person_mode, language=_pair_language(a, b))
        return _are_xep_variants(a.name, b.name, kind)

    def _matches(self, cand: EntityCandidate, cluster: List[EntityCandidate], kind: str) -> bool:
        if (
            kind == "person"
            and self.pair_rule is not None
            and any(self.pair_rule(cand, m) == "join" for m in cluster)
        ):
            # An evidence rule's join (record l_publisher_verified_mode, V3); the guards still
            # run on it in __call__.
            return True
        if kind == "person" and self.linkage == "complete":
            return all(self._match(cand, m, kind) for m in cluster)
        pool = cluster
        if kind == "person" and self.guard is not None and _scoped(cand.id):
            # Two bare names scoped to their episodes are not evidence of one person: a scoped id
            # joins only through a member with a name of its own (operator decision 1B,
            # 2026-11-01; record s5c_scoped_names_not_pooled). The guarded ledger build only.
            pool = [m for m in cluster if not _scoped(m.id)]
        if kind == "person" and self.linkage == "no_one_word_bridge" and _words(cand.name) > 1:
            pool = [m for m in pool if _words(m.name) > 1] or pool
        if kind == "person" and self.pair_rule is not None:
            # A near spelling of two one-episode mentions is not evidence (record s5d, rule 1).
            pool = [m for m in pool if self.pair_rule(cand, m) != "ignore"]
        return any(self._match(cand, m, kind) for m in pool)

    def _apart(self, cand: EntityCandidate, cluster: List[EntityCandidate]) -> bool:
        """Record s5d, rule 2: a reviewed two-people pair, over the whole cluster; recorded."""
        assert self.pair_rule is not None
        for m in cluster:
            if self.pair_rule(cand, m) == "apart" and self._match(cand, m, "person"):
                episode = sorted(cand.episodes)[0] if cand.episodes else ""
                self.blocked.append([cand.id, m.id, "reviewed_apart", episode])
                return True
        return False

    def __call__(self, cand: EntityCandidate, cluster: List[EntityCandidate], kind: str) -> bool:
        if kind == "person" and self.pair_rule is not None and self._apart(cand, cluster):
            return False
        if not self._matches(cand, cluster, kind):
            return False
        if self.guard is None or kind != "person":
            return True
        for m in cluster:
            fired = self.guard(cand, m)
            if fired is not None:
                self.blocked.append([cand.id, m.id, fired[0], fired[1]])
                return False
        return True


def build_entity_canonical_map(
    candidates: Dict[str, EntityCandidate],
    *,
    same_show_required: bool = True,
    guard: Optional[_Guard] = None,
    person_mode: str = "cluster",
    linkage: str = "single",
    keep_canonical: Collection[str] = (),
    pair_rule: Optional[_PairRule] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Cluster variants and return ``(entity_clusters_payload, variant→canonical map)``.

    ``guard(candidate, member)`` (ADR-167), for persons: a candidate joins a cluster only if it
    matches some member and the guard fires against no member; otherwise the next eligible cluster
    is tried. Each refusal is recorded in the payload's ``blocked_joins`` as
    ``[candidate_id, member_id, guard, episode_id]``. Without a guard the build is unchanged.

    ``person_mode`` names the identity module's rule mode for people (``cluster``, today's, or
    ``cluster_person``). ``linkage`` (people only) is how a candidate must match a cluster:

    * ``single`` (today's): some member;
    * ``no_one_word_bridge``: some member, and when the candidate has two or more words and the
      cluster holds a member that does too, one of THOSE. A bare surname then cannot bridge two
      full names (``Ilse Tolvane`` ~ ``Tolvane`` ~ ``Mara Tolvane``, two relatives), while
      spelling drift between full names still chains (``Brannick Holt`` ~ ``Brannick Hollt``);
    * ``complete``: every member (measured, not used: it splits that drift too).

    ``keep_canonical``: ids already serving as a canonical (the served ledger's). A cluster that
    holds one keeps it, so a person's pages and links do not move when frequencies shift or
    every member comes to carry one spelling (RFC-131 0041).

    ``pair_rule(candidate, member)`` (record s5d), for persons: ``"ignore"`` drops that member
    from the ones the candidate may match through; ``"apart"`` refuses the cluster, recorded in
    ``blocked_joins`` as guard ``reviewed_apart``; ``"join"`` (record l_publisher_verified_mode)
    matches it to the cluster, the guards still deciding. The ledger build passes one.

    The defaults keep today's build.
    """
    joins = _JoinTest(guard, person_mode, linkage, pair_rule)
    blocked = joins.blocked

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
                    if joins(cand, cluster, kind):
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
                    if joins(cand, clusters[i], kind):
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
            canonical = _pick_canonical(cluster, keep_canonical)
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
    if guard is not None or pair_rule is not None:
        payload["blocked_joins"] = blocked
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


def cached_entity_id_map(
    corpus_dir: Path | str, *, same_show_required: bool = True
) -> Dict[str, str]:
    """The identity map the server applies: the identity ledger's, cached per corpus change.

    ADR-166: the map is no longer built in the api process. A deterministic step writes it, with
    its evidence, to the identity ledger (``identity.ledger``), and this serves the ledger's map.
    A missing, unreadable or version-mismatched ledger means NO joins; the unguarded in-memory
    build is never a serving fallback. The ledger file is part of ``corpus_mtime``, so a rebuilt
    ledger is picked up without a restart.

    ``same_show_required=False`` (no serving caller) still builds in-process, for analysis.
    """
    root = Path(corpus_dir)
    key = (str(root.resolve()), bool(same_show_required))
    token = corpus_mtime(root)
    hit = cache_peek(_ID_MAP_NS, key, token)
    if hit is not None:
        return dict(hit)
    with _ID_MAP_LOCK:
        # Re-check: another thread may have read it while we waited for the lock.
        return dict(
            get_or_compute(
                _ID_MAP_NS,
                key,
                token,
                (
                    (lambda: _served_id_map(root))
                    if same_show_required
                    else (lambda: build_entity_id_map(root, same_show_required=False))
                ),
            )
        )


def _served_id_map(root: Path) -> Dict[str, str]:
    from ..identity.ledger import served_id_map

    return served_id_map(root)


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

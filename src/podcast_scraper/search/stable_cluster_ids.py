"""Keep theme (``tc:``) and storyline (``thc:``) ids across re-clustering (#2345).

Both clusterings name a cluster after its canonical label, so a re-run that picks another label,
or that merges two clusters, gave the same group of topics a new id. Every listener reference
stored by id (follows, favorites, "Less like this", paths, boards) then pointed at nothing.

:func:`carry_cluster_ids` runs after a clustering, against the previous artifact:

- a new cluster INHERITS a previous id when their member sets overlap by Jaccard >= 0.5, one to
  one, best overlap first;
- a previous id that was not inherited gets a SUCCESSOR when one new cluster holds at least half
  of its members (a rename that moved, or an absorption), else it is GONE;
- the previous ``id_history`` is carried forward through the new map, so a reference several
  generations old still resolves.

The result is published in the artifact as ``id_history: {successors: {old: new}, gone: [...]}``
for the reference-integrity pass (``server/reference_integrity.py``).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

#: Member overlap (Jaccard) at which a new cluster keeps a previous cluster's id (operator,
#: 2026-10-13). On three days of prod snapshots it kept 80 of 251 renamed themes and 3 of 10
#: renamed storylines.
MIN_JACCARD = 0.5

#: Share of a vanished cluster's members one new cluster must hold to be its successor.
MIN_SUCCESSOR_SHARE = 0.5

ID_KEY = "graph_compound_parent_id"


def _members(cluster: Mapping[str, Any]) -> frozenset[str]:
    out = []
    for m in cluster.get("members") or []:
        tid = m.get("topic_id") if isinstance(m, Mapping) else m
        if isinstance(tid, str) and tid:
            out.append(tid)
    return frozenset(out)


def _cluster_id(cluster: Mapping[str, Any]) -> str:
    return str(cluster.get(ID_KEY) or cluster.get("cluster_id") or "")


def _unique(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    k = 1
    while f"{base}-{k}" in taken:
        k += 1
    return f"{base}-{k}"


def carry_cluster_ids(
    clusters: list[dict[str, Any]],
    prior_clusters: Iterable[Mapping[str, Any]],
    *,
    prior_history: Mapping[str, Any] | None = None,
    min_jaccard: float = MIN_JACCARD,
) -> dict[str, Any]:
    """Rewrite ``clusters``' ids in place to keep previous ids, and return the ``id_history``."""
    prior = [(c_id, _members(c)) for c in prior_clusters if (c_id := _cluster_id(c))]
    new = [(i, _members(c)) for i, c in enumerate(clusters)]

    pairs: list[tuple[float, str, int]] = []
    for old_id, old_m in prior:
        for i, new_m in new:
            union = len(old_m | new_m)
            jac = len(old_m & new_m) / union if union else 0.0
            if jac >= min_jaccard:
                pairs.append((jac, old_id, i))
    # Best overlap first; ties broken by ids, so the result never depends on input order.
    pairs.sort(key=lambda p: (-p[0], p[1], _cluster_id(clusters[p[2]])))

    inherited: dict[int, str] = {}
    used_old: set[str] = set()
    for _jac, old_id, i in pairs:
        if i in inherited or old_id in used_old:
            continue
        inherited[i] = old_id
        used_old.add(old_id)

    taken = set(inherited.values())
    for i, cluster in enumerate(clusters):
        if i in inherited:
            cluster[ID_KEY] = inherited[i]
        else:
            cluster[ID_KEY] = _unique(_cluster_id(cluster), taken)
            taken.add(cluster[ID_KEY])

    current = {_cluster_id(c) for c in clusters}
    successors: dict[str, str] = {}
    gone: set[str] = set()
    for old_id, old_m in prior:
        if old_id in current:
            continue
        best_id, best_share = "", 0.0
        for c in clusters:
            share = len(old_m & _members(c)) / len(old_m) if old_m else 0.0
            if share > best_share or (share == best_share and _cluster_id(c) < best_id):
                best_id, best_share = _cluster_id(c), share
        if best_id and best_share >= MIN_SUCCESSOR_SHARE:
            successors[old_id] = best_id
        else:
            gone.add(old_id)

    # Carry the older history forward through this run's map.
    history = prior_history or {}
    for old_id, target in dict(history.get("successors") or {}).items():
        if old_id in current or old_id in successors:
            continue
        hop = str(target)
        seen = {old_id}
        while hop not in current and hop in successors and hop not in seen:
            seen.add(hop)
            hop = successors[hop]
        if hop in current:
            successors[old_id] = hop
        else:
            gone.add(old_id)
    for old_id in history.get("gone") or []:
        if old_id not in current and old_id not in successors:
            gone.add(str(old_id))
    gone -= current
    return {"successors": dict(sorted(successors.items())), "gone": sorted(gone)}


def map_prior_ids(
    prior_clusters: Iterable[Mapping[str, Any]],
    current_clusters: Iterable[Mapping[str, Any]],
    *,
    min_jaccard: float = MIN_JACCARD,
) -> dict[str, Any]:
    """``id_history`` from an OLD artifact to the CURRENT one, without changing either.

    For state written before ids were kept (the one-time backfill): an old id that a current
    cluster would have inherited, or whose members one current cluster mostly holds, maps to that
    cluster's current id; the rest are gone.
    """
    current = [dict(c) for c in current_clusters]
    original = [_cluster_id(c) for c in current]
    probe = [dict(c) for c in current]
    carried = carry_cluster_ids(probe, prior_clusters, min_jaccard=min_jaccard)
    successors: dict[str, str] = {}
    for orig, c in zip(original, probe):
        inherited = _cluster_id(c)
        if inherited != orig and inherited not in original:
            successors[inherited] = orig
    to_current = dict(zip((_cluster_id(c) for c in probe), original))
    for old, new in carried["successors"].items():
        successors.setdefault(old, to_current.get(new, new))
    current_ids = set(original)
    gone = sorted(g for g in carried["gone"] if g not in current_ids and g not in successors)
    return {"successors": dict(sorted(successors.items())), "gone": gone}


def resolve(ref: str, history: Mapping[str, Any]) -> str | None:
    """The id ``ref`` stands for now: its successor, itself when not in the history, or ``None``
    when it is gone."""
    successors = history.get("successors") or {}
    if ref in successors:
        return str(successors[ref])
    if ref in set(history.get("gone") or []):
        return None
    return ref

"""The ledger's joins that rest on a near spelling alone and no read has reviewed (RFC-131 D2).

A string test cannot tell a misspelling from a different person who shares a word and a few letters
(eval-data ``d2i9/threshold_sweep.py``: on the read pairs the wrong and right joins overlap in both
similarity ratios). Until voice evidence decides such joins, each new one is listed here for a
read, and the read's verdict goes to the reviewed list (``upgrade/lists/s5d_reviewed_fuzzy_joins``):
``groups`` when one person, ``apart`` when two.

A member is listed when every name match it has inside its group is a near spelling
(:func:`person_match.near_spelling_only`), its id is not in a reviewed group with the member it
matches, and the pair is not reviewed apart.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


def unreviewed_near_joins(
    ledger: Mapping[str, Any],
    candidates: Mapping[str, Any],
    evidence: Mapping[str, Any],
    reviewed: Mapping[str, list],
) -> List[Dict[str, Any]]:
    """One row per listed member: its group, id, name, episodes, voice and publisher flags, and the
    members it matches only by a near spelling."""
    from podcast_scraper.identity.bare_name_scope import is_scoped_person_id
    from podcast_scraper.identity.ledger import PERSON_LINKAGE, PERSON_MODE
    from podcast_scraper.identity.person_match import near_spelling_only
    from podcast_scraper.kg.entity_clusters import _JoinTest, _pair_language

    group_of = {str(i): n for n, g in enumerate(reviewed.get("groups") or []) for i in g}
    apart = {frozenset((str(p[0]), str(p[1]))) for p in reviewed.get("apart") or []}
    voiced = {p for e in evidence.values() for p, vs in e.voice_of.items() if vs}
    named = {p for e in evidence.values() for p in e.publisher_named}
    match = _JoinTest(None, PERSON_MODE, PERSON_LINKAGE)._match
    rows: List[Dict[str, Any]] = []
    for cluster in (ledger.get("clusters_payload") or {}).get("clusters") or []:
        canonical = str(cluster.get("canonical_id") or "")
        if not canonical.startswith("person:"):
            continue
        members = [
            candidates[m["id"]] for m in cluster.get("members") or [] if m["id"] in candidates
        ]
        for a in members:
            if is_scoped_person_id(a.id):
                continue
            matches = [b for b in members if b.id != a.id and match(a, b, "person")]
            if not matches or not all(
                near_spelling_only(a.name, b.name, _pair_language(a, b)) for b in matches
            ):
                continue
            open_ = [
                b
                for b in matches
                if frozenset((a.id, b.id)) not in apart
                and not (
                    group_of.get(a.id) is not None and group_of.get(a.id) == group_of.get(b.id)
                )
            ]
            if not open_:
                continue
            rows.append(
                {
                    "group": canonical,
                    "id": a.id,
                    "name": a.name,
                    "episodes": a.freq,
                    "voice": a.id in voiced,
                    "publisher": a.id in named,
                    "near": [{"id": b.id, "name": b.name, "episodes": b.freq} for b in open_],
                }
            )
    return rows


def pairs(rows: Iterable[Mapping[str, Any]]) -> List[tuple]:
    """The listed rows as unordered id pairs, each once."""
    seen = set()
    for r in rows:
        for b in r["near"]:
            seen.add(tuple(sorted((r["id"], b["id"]))))
    return sorted(seen)

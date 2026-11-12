"""Unit tests for cross-episode entity canonicalization (kg/entity_clusters.py, #852)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _cand(cid, kind, name, eps, shows):
    from podcast_scraper.kg.entity_clusters import EntityCandidate

    return EntityCandidate(id=cid, kind=kind, name=name, episodes=set(eps), shows=set(shows))


# --- variant rule: MERGE real drift ---------------------------------------------


# --- variant rule: REJECT the landmines -----------------------------------------


# --- #904 predicate redesign: nickname + token-count tolerance ------------------


# --- canonical map: frequency + same-show ---------------------------------------


# --- collect + end-to-end on a synthetic corpus ---------------------------------


def _write_kg(path: Path, episode_id: str, show: str, entities):
    nodes = [{"id": f"episode:{episode_id}", "type": "Episode", "properties": {"podcast_id": show}}]
    for eid, name in entities:
        nodes.append({"id": eid, "type": "Entity", "properties": {"name": name}})
    path.write_text(json.dumps({"episode_id": episode_id, "nodes": nodes, "edges": []}))


class TestDiacriticsAreFoldedBeforeComparing:
    """``Björk`` and ``Bjork`` are one person (#2056).

    Transcripts and feed metadata disagree about diacritics constantly — ASR emits unaccented
    ASCII, the show notes carry the real spelling. Multi-token names already survived this by
    accident, because one differing character barely moves the similarity ratio:

        'Jürgen Schmidhuber' == 'Jurgen Schmidhuber'   -> already True (fuzzy)

    A SINGLE-token name does not, because `_is_acronymish` refuses to fuzzy-match short single
    tokens at all — the UPS/USPS guard. So the mononym case fell through the one gap where the
    ratio test cannot rescue it.

    Folding also makes the multi-token cases match EXACTLY rather than by ratio, which is a
    precision gain, not just a recall one: an exact hit short-circuits before any threshold is
    consulted.
    """

    def test_folding_is_exact_not_fuzzy(self) -> None:
        from podcast_scraper.kg.filters import _clean_entity_name

        assert _clean_entity_name("François Chollet") == _clean_entity_name("Francois Chollet")


def _reference_clusters(candidates, same_show_required=True):
    """The pre-index loop, verbatim in behaviour: every cluster scanned, show overlap per member."""
    from podcast_scraper.kg.entity_clusters import _are_xep_variants

    out = {}
    by_kind = {}
    for c in candidates.values():
        by_kind.setdefault(c.kind, []).append(c)
    for kind, items in sorted(by_kind.items()):
        clusters = []
        for cand in sorted(items, key=lambda c: (-c.freq, c.name.lower())):
            for cluster in clusters:
                if same_show_required and not any(cand.shows & m.shows for m in cluster):
                    continue
                if any(_are_xep_variants(cand.name, m.name, kind) for m in cluster):
                    cluster.append(cand)
                    break
            else:
                clusters.append([cand])
        out[kind] = [[m.id for m in cl] for cl in clusters]
    return out

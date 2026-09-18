"""Unit tests for :func:`podcast_scraper.search.theme_clusters.top_theme_clusters_by_member_count`.

The storyline enumerator feeds the Home rail + interests picker. It mirrors the semantic
``top_clusters_by_member_count`` (ranks by member count, limits) but reads the envelope-wrapped
``enrichments/topic_theme_clusters.json`` and additionally resolves each cluster's
``anchor_topic_id`` (most-central member) so the client can open a representative topic card.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.search.theme_clusters import (
    STORYLINE_DOC_TYPE,
    storyline_index_rows,
    top_theme_clusters_by_member_count,
)

pytestmark = [pytest.mark.unit]


def _write(root: Path, body: dict) -> None:
    (root / "enrichments").mkdir(parents=True, exist_ok=True)
    (root / "enrichments" / "topic_theme_clusters.json").write_text(
        json.dumps(body), encoding="utf-8"
    )


def _cluster(gpid: str, label: str, members: list[dict], member_count: int | None = None) -> dict:
    cl: dict = {"graph_compound_parent_id": gpid, "canonical_label": label, "members": members}
    if member_count is not None:
        cl["member_count"] = member_count
    return cl


def test_empty_without_artifact(tmp_path: Path) -> None:
    assert top_theme_clusters_by_member_count(tmp_path) == []


def test_ranks_by_member_count_and_limits(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "data": {
                "clusters": [
                    _cluster("thc:small", "Small", [{"topic_id": "topic:a"}], member_count=2),
                    _cluster("thc:big", "Big", [{"topic_id": "topic:b"}], member_count=9),
                    _cluster(
                        "thc:mid",
                        "Mid",
                        [{"topic_id": "topic:c"}, {"topic_id": "topic:d"}, {"topic_id": "topic:e"}],
                    ),  # no member_count → len(members)=3
                ]
            }
        },
    )
    top = top_theme_clusters_by_member_count(tmp_path, top_n=2, min_members=1)
    assert [c["id"] for c in top] == ["thc:big", "thc:mid"]  # 9, then 3 (len fallback); small=2 cut
    assert top[1]["size"] == 3  # len(members) fallback when member_count absent


def test_anchor_is_highest_lift_member(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "data": {
                "clusters": [
                    _cluster(
                        "thc:x",
                        "X",
                        [
                            {"topic_id": "topic:low", "lift_to_cluster": 1.1},
                            {"topic_id": "topic:high", "lift_to_cluster": 3.4},
                            {"topic_id": "topic:mid", "lift_to_cluster": 2.0},
                        ],
                    )
                ]
            }
        },
    )
    (only,) = top_theme_clusters_by_member_count(tmp_path, min_members=1)
    assert only == {"id": "thc:x", "label": "X", "size": 3, "anchor_topic_id": "topic:high"}


def test_anchor_falls_back_to_first_topic_id_without_lifts(tmp_path: Path) -> None:
    members = [{"topic_id": "topic:b"}, {"topic_id": "topic:a"}]
    _write(tmp_path, {"data": {"clusters": [_cluster("thc:y", "Y", members)]}})
    (only,) = top_theme_clusters_by_member_count(tmp_path, min_members=1)
    # No lifts → all tie at 0.0; the tie-break keeps the smallest topic_id ("topic:a").
    assert only["anchor_topic_id"] == "topic:a"


def test_skips_clusters_with_no_valid_member(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "data": {
                "clusters": [
                    _cluster("thc:empty", "Empty", []),  # no members → no anchor → skipped
                    _cluster("thc:ok", "Ok", [{"topic_id": "topic:z"}]),
                ]
            }
        },
    )
    got = top_theme_clusters_by_member_count(tmp_path, min_members=1)
    assert [c["id"] for c in got] == ["thc:ok"]


def test_reads_unwrapped_payload_too(tmp_path: Path) -> None:
    # Tolerates an already-unwrapped file (no `data` envelope) — parity with the loader.
    _write(tmp_path, {"clusters": [_cluster("thc:u", "U", [{"topic_id": "topic:u"}])]})
    got = top_theme_clusters_by_member_count(tmp_path, min_members=1)
    assert [c["id"] for c in got] == ["thc:u"]


# --- the navigation floor (#1932) -------------------------------------------------------------


def test_small_themes_are_withheld_by_default(tmp_path: Path) -> None:
    """A theme is a place a listener is SENT, not merely a fact the corpus contains.

    Both theme surfaces — the operator overlay and the player's Storylines rail — apply this
    floor, and they must share it: it was added to the operator route alone at first, so the
    consumer rail (the surface the floor exists for) stayed unfiltered. The default lives on this
    function precisely so a new caller inherits it instead of having to remember.
    """
    _write(
        tmp_path,
        {
            "data": {
                "clusters": [
                    _cluster("thc:pair", "Pair", [{"topic_id": "topic:a"}], member_count=2),
                    _cluster("thc:real", "Real", [{"topic_id": "topic:b"}], member_count=6),
                ]
            }
        },
    )
    assert [c["id"] for c in top_theme_clusters_by_member_count(tmp_path)] == ["thc:real"]


def test_the_floor_is_overridable_for_callers_that_want_everything(tmp_path: Path) -> None:
    """Diagnostics and cluster-count checks need the full set; the artifact keeps every theme."""
    _write(
        tmp_path,
        {
            "data": {
                "clusters": [
                    _cluster("thc:pair", "Pair", [{"topic_id": "topic:a"}], member_count=2),
                    _cluster("thc:real", "Real", [{"topic_id": "topic:b"}], member_count=6),
                ]
            }
        },
    )
    ids = [c["id"] for c in top_theme_clusters_by_member_count(tmp_path, min_members=1)]
    assert ids == ["thc:real", "thc:pair"]


def test_withholding_everything_returns_empty_rather_than_falling_back(tmp_path: Path) -> None:
    """No silent "well, show the small ones anyway" — the caller must be able to see the gap."""
    _write(
        tmp_path,
        {
            "data": {
                "clusters": [
                    _cluster("thc:pair", "Pair", [{"topic_id": "topic:a"}], member_count=2)
                ]
            }
        },
    )
    assert top_theme_clusters_by_member_count(tmp_path) == []


# --- storyline_index_rows: the search-index rows (#2114 / operator 2026-09-17) ------------------


def test_storyline_index_rows_empty_without_artifact(tmp_path: Path) -> None:
    assert storyline_index_rows(tmp_path) == []


def test_storyline_index_rows_one_row_per_cluster_with_no_episode_id(tmp_path: Path) -> None:
    """ONE row per storyline, corpus-scoped — not one per episode.

    A storyline spans episodes; it is not a property of any one of them. Emitting it per episode and
    de-duplicating on read (what following the ``kg_topic`` pattern literally would mean) multiplies
    rows for an object with a single identity. ``episode_id`` is nullable in the aux schema, so the
    corpus-level row needs no schema change.
    """
    _write(
        tmp_path,
        {
            "data": {
                "clusters": [
                    _cluster(
                        "thc:risk",
                        "Managing risk across domains",
                        [
                            {"topic_id": "topic:risk-management", "label": "risk management"},
                            {"topic_id": "topic:systems", "label": "systems thinking"},
                        ],
                        member_count=4,
                    ),
                    _cluster(
                        "thc:learning",
                        "How people learn",
                        [{"topic_id": "topic:learning", "label": "lifelong learning"}],
                        member_count=2,
                    ),
                ]
            }
        },
    )
    rows = storyline_index_rows(tmp_path)
    assert len(rows) == 2, "expected exactly one row per cluster"
    by_id = {rid: (text, meta) for rid, text, meta in rows}
    assert set(by_id) == {"storyline:thc:risk", "storyline:thc:learning"}
    text, meta = by_id["storyline:thc:risk"]
    assert meta["doc_type"] == STORYLINE_DOC_TYPE
    assert meta["source_id"] == "thc:risk"
    assert meta["episode_id"] is None, "a storyline is not scoped to an episode"
    assert meta["storyline_label"] == "Managing risk across domains"
    assert meta["storyline_size"] == 4
    assert meta["anchor_topic_id"]


def test_storyline_embed_text_carries_member_labels(tmp_path: Path) -> None:
    """The label AND its members are embedded — that is what exact-name resolution cannot do.

    Indexing only the canonical label would make the index no better than the name resolver. With
    the members in the text, a query naming a MEMBER ("risk management") can reach the storyline.
    """
    _write(
        tmp_path,
        {
            "data": {
                "clusters": [
                    _cluster(
                        "thc:risk",
                        "Managing risk across domains",
                        [
                            {"topic_id": "topic:risk-management", "label": "risk management"},
                            {"topic_id": "topic:safety", "label": "safety practices"},
                        ],
                        member_count=4,
                    )
                ]
            }
        },
    )
    _rid, text, _meta = storyline_index_rows(tmp_path)[0]
    assert text.startswith("Managing risk across domains"), "the label should lead the embed text"
    assert "risk management" in text
    assert "safety practices" in text


def test_storyline_index_rows_ignore_the_surfacing_floor(tmp_path: Path) -> None:
    """A 2-member storyline is INDEXED even though the Home rail would not show it.

    The /4 minimum is a surfacing rule about where a listener is sent. Applying it here would make a
    small storyline unfindable by any means, which is a different and worse thing.
    """
    _write(
        tmp_path,
        {
            "data": {
                "clusters": [
                    _cluster(
                        "thc:tiny",
                        "Tiny pairing",
                        [
                            {"topic_id": "topic:a", "label": "a"},
                            {"topic_id": "topic:b", "label": "b"},
                        ],
                        member_count=2,
                    )
                ]
            }
        },
    )
    rows = storyline_index_rows(tmp_path)
    assert [r[2]["source_id"] for r in rows] == ["thc:tiny"]


def test_storyline_index_rows_skip_an_anchorless_cluster(tmp_path: Path) -> None:
    """No resolvable anchor → not indexed, as the rail also skips an unopenable cluster."""
    _write(tmp_path, {"data": {"clusters": [_cluster("thc:broken", "Broken", [], member_count=4)]}})
    assert storyline_index_rows(tmp_path) == []

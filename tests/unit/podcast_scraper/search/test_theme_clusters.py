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

from podcast_scraper.search.theme_clusters import top_theme_clusters_by_member_count

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

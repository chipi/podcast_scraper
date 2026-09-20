"""A storyline's anchor topic, resolved where the ranking is (operator 2026-09-19).

A storyline has no endpoint of its own — it is read as its most-central member topic's card — so a
trending row without an anchor cannot be opened at all. That was the shipped bug: the client
derived the anchor by joining trending rows against ``/storylines`` on ``thc:`` id, and those
two lists cannot cover the same set. ``/storylines`` floors at four members and returns a
top-N by SIZE; momentum ranks every cluster carrying a series, by MOMENTUM, with no floor. Misses
were routine, and the client's fallback handed the ``thc:`` id to a TOPIC lookup, which resolves
nothing.

It is resolved in the MOMENTUM layer rather than hydrated in the route, and that is the property
worth pinning. The route version read the theme-cluster artifact a SECOND time under a different
cache token, which reintroduces the same class of bug one level down: a re-enrichment that rewrites
only that file leaves the ranking reading a stale cluster list while the anchors read the fresh one,
and every disagreement renders as a row that will not open. One snapshot, one loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.search.storylines import DEFAULT_MIN_STORYLINE_MEMBERS
from podcast_scraper.server.app_momentum import _storyline_anchors

pytestmark = [pytest.mark.unit]


def _write_clusters(root: Path, clusters: list[dict]) -> None:
    (root / "enrichments").mkdir(parents=True, exist_ok=True)
    (root / "enrichments" / "topic_theme_clusters.json").write_text(
        json.dumps({"data": {"clusters": clusters}}), encoding="utf-8"
    )


def _cluster(gpid: str, members: list[dict], member_count: int | None = None) -> dict:
    cl: dict = {"graph_compound_parent_id": gpid, "canonical_label": gpid, "members": members}
    if member_count is not None:
        cl["member_count"] = member_count
    return cl


def test_a_cluster_below_the_surfacing_floor_still_gets_an_anchor(tmp_path: Path) -> None:
    """The bug in one assertion.

    ``/storylines`` withholds a 2-member theme — a storyline is somewhere a listener is SENT,
    and a single co-occurrence pair is not a destination. Momentum ranks it anyway. Anchoring only
    what some other surface considers worth showing would leave exactly the rows THIS ranking chose
    unopenable, which is what shipped.
    """
    assert (
        DEFAULT_MIN_STORYLINE_MEMBERS == 4
    ), "the floor moved; this test's premise needs re-reading"
    _write_clusters(
        tmp_path,
        [
            _cluster("thc:pair", [{"topic_id": "topic:a"}], member_count=2),
            _cluster("thc:real", [{"topic_id": "topic:b"}], member_count=6),
        ],
    )
    assert _storyline_anchors(tmp_path) == {"thc:pair": "topic:a", "thc:real": "topic:b"}


def test_no_top_n_either(tmp_path: Path) -> None:
    # The second way the join missed: big enough to clear the floor, too far down by SIZE to make
    # the top-N, while near the top by MOMENTUM — which is what trending ranks on.
    _write_clusters(
        tmp_path,
        [
            _cluster(f"thc:{i}", [{"topic_id": f"topic:{i}"}], member_count=n)
            for i, n in enumerate((30, 20, 10, 5, 4))
        ],
    )
    assert len(_storyline_anchors(tmp_path)) == 5


def test_the_anchor_is_the_most_central_member(tmp_path: Path) -> None:
    # Not merely "a member": every member's card shows the same discussed-together set, so the
    # anchor picks the most representative entry rather than whichever sorted first.
    _write_clusters(
        tmp_path,
        [
            _cluster(
                "thc:x",
                [
                    {"topic_id": "topic:low", "lift_to_cluster": 1.1},
                    {"topic_id": "topic:high", "lift_to_cluster": 3.4},
                ],
            )
        ],
    )
    assert _storyline_anchors(tmp_path) == {"thc:x": "topic:high"}


def test_a_cluster_with_no_resolvable_anchor_is_absent_not_self_referential(
    tmp_path: Path,
) -> None:
    # The contract the client leans on: a MISSING key means "not openable". If this ever returned
    # {"thc:broken": "thc:broken"} the dead tap would be back, wearing a new coat.
    _write_clusters(tmp_path, [_cluster("thc:broken", [], member_count=4)])
    assert _storyline_anchors(tmp_path) == {}


def test_no_artifact_is_empty_rather_than_an_error(tmp_path: Path) -> None:
    assert _storyline_anchors(tmp_path) == {}


def test_the_ranking_and_the_anchor_read_ONE_snapshot(tmp_path: Path, monkeypatch) -> None:
    """The reason this lives here and not in the route.

    Counted rather than reasoned about: the route version read the artifact once for the ranking
    and again for the anchors, under two different cache tokens. Pinned as a count so the second
    read cannot creep back — that divergence is invisible until a re-enrichment lands between the
    two reads, at which point rows are ranked from one file and anchored from another.
    """
    from podcast_scraper.server import app_momentum

    _write_clusters(tmp_path, [_cluster("thc:a", [{"topic_id": "topic:a"}], member_count=6)])

    reads: list[str] = []
    real = app_momentum.cached_json_artifact

    def counting(root: Path, rel: str):  # type: ignore[no-untyped-def]
        reads.append(rel)
        return real(root, rel)

    monkeypatch.setattr(app_momentum, "cached_json_artifact", counting)
    _storyline_anchors(tmp_path)

    theme_reads = [r for r in reads if "theme" in r]
    assert len(theme_reads) == 1, f"the artifact was read {len(theme_reads)} times, not once"

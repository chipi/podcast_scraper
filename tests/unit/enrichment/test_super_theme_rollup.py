"""The super-theme rollup is the TOP of the navigation, so it must not invent groups.

It previously called ``_average_linkage_to_target``, which merges the best-scoring pair each round
until the count reaches a target "regardless of edge sparsity". On the 1,066-episode corpus the
cross-cluster lift graph is 96% empty (57 of 1,431 theme-pairs carry any lift), so "best pair" is
mostly a tie at zero and greedy merging chains everything together. It produced exactly 6
super-themes — one holding **49 of 54 themes**, filing ``federal reserve policy`` and ``transition
metal catalysis`` under ``agentic ai systems``. It hit the target, and the target was the problem.

A threshold floor alone does not fix it either: sweeping the floor jumps straight from one blob
(t=0) to 32 fragments with 21 singletons (t=1), because the signal is sparse rather than
mis-scaled. So the rollup now merges only on real evidence, keeps the largest groups as a bounded
legend, and puts the remainder in an EXPLICIT long-tail bucket — a bucket that says "everything
else" is honest; one mislabelled ``agentic ai systems`` is not.
"""

from __future__ import annotations

import pytest

from podcast_scraper.enrichment.enrichers.topic_theme_clusters import (
    _assign_super_themes,
    _LONG_TAIL_ID,
    _LONG_TAIL_LABEL,
    _SUPER_THEME_MAX,
    _SUPER_THEME_MERGE_FLOOR,
)

pytestmark = pytest.mark.unit


def _clusters(n: int) -> tuple[list[dict], list[set[int]]]:
    out = [
        {"canonical_label": f"theme-{i}", "member_count": n - i, "cluster_type": "theme"}
        for i in range(n)
    ]
    return out, [{i} for i in range(n)]


def _biggest_group(out: list[dict]) -> int:
    ids = {c["super_theme_id"] for c in out}
    return max(sum(1 for c in out if c["super_theme_id"] == g) for g in ids)


def test_unrelated_themes_are_not_merged_into_one_group() -> None:
    """The regression: zero cross-lift must mean zero merging, however many themes there are."""
    out, sets = _clusters(20)
    _assign_super_themes(out, sets, lambda i, j: 0.0)
    assert _biggest_group(out) < 19, "themes with no shared lift were chained into one super-theme"


def test_the_remainder_lands_in_an_explicit_long_tail() -> None:
    out, sets = _clusters(20)
    _assign_super_themes(out, sets, lambda i, j: 0.0)
    tail = [c for c in out if c["super_theme_id"] == _LONG_TAIL_ID]
    assert tail, "with no lift anywhere, most themes belong in the catch-all"
    assert all(c["super_theme_label"] == _LONG_TAIL_LABEL for c in tail)
    assert all(c["super_theme_is_long_tail"] is True for c in tail)


def test_a_real_group_is_never_labelled_long_tail() -> None:
    """Themes that DO share lift must form a named group, not fall into the catch-all."""

    def w(i: int, j: int) -> float:
        # 0, 1, 2 genuinely co-occur; everything else is unrelated.
        return 50.0 if {i, j} <= {0, 1, 2} and i != j else 0.0

    out, sets = _clusters(20)
    _assign_super_themes(out, sets, w)
    ids = {out[i]["super_theme_id"] for i in (0, 1, 2)}
    assert len(ids) == 1, "the three related themes should share one super-theme"
    assert ids != {_LONG_TAIL_ID}
    assert all(out[i]["super_theme_is_long_tail"] is False for i in (0, 1, 2))


def test_legend_stays_bounded() -> None:
    """The legend is a browse surface (7±2). The long-tail bucket occupies one of its slots."""
    out, sets = _clusters(40)
    _assign_super_themes(out, sets, lambda i, j: 0.0)
    assert len({c["super_theme_id"] for c in out}) <= _SUPER_THEME_MAX


def test_small_corpora_skip_the_rollup_entirely() -> None:
    out, sets = _clusters(3)
    _assign_super_themes(out, sets, lambda i, j: 0.0)
    assert len({c["super_theme_id"] for c in out}) == 3
    assert not any(c["super_theme_is_long_tail"] for c in out)


def test_merge_floor_rejects_chance_cooccurrence() -> None:
    """Lift of 1.0 IS chance. Merging on it is how noise becomes a storyline."""
    assert _SUPER_THEME_MERGE_FLOOR >= 1.0

    def at_chance(i: int, j: int) -> float:
        return 0.99 if i != j else 0.0

    out, sets = _clusters(12)
    _assign_super_themes(out, sets, at_chance)
    assert _biggest_group(out) < 12, "below-chance lift must not merge everything"


def test_every_cluster_carries_the_long_tail_flag() -> None:
    """Consumers style the catch-all differently; they must not infer it from the label string."""
    out, sets = _clusters(20)
    _assign_super_themes(out, sets, lambda i, j: 0.0)
    assert all("super_theme_is_long_tail" in c for c in out)
    assert all(isinstance(c["super_theme_is_long_tail"], bool) for c in out)

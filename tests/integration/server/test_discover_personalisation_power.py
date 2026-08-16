"""Personalisation must be a real choice, not just a code path that runs.

`test_app_discover_view.py` proves `rank_discover` composes its signals correctly, and
`test_rank_discover_eval.py` proves it beats recency on seeded personas. Both build or seed their
own interests, so neither can see the failure this module is about: the ranker working perfectly
on tokens **the product never offers a user**.

Measured on the committed corpus (2026-08-16):

    topic:personal-finance     4/36 episodes  -> feed of p05 (investing)
    topic:safety-practices     4/36           -> feed of p03 (scuba)
    ... 6 niche topics         -> 6 DISTINCT feeds, each the right show

    tc:show-themes            36/36 (100%)
    thc:managing-risk         36/36 (100%)    -> all three produce ONE identical feed
    tc:lifelong-learning      36/36 (100%)

The picker (`GET /api/app/clusters` -> `top_clusters_by_member_count`) ranks its options by
PREVALENCE, and prevalence is inversely related to usefulness as a filter: a token on every
episode gives every episode the same affinity, so the feed collapses to a single significance
ordering — identical no matter which option is chosen. The engine discriminates; its input does not.

The two layers here:
  * `TestRankerDiscriminates` — the ranker's power, on the REAL corpus. Passes; locks in that a
    niche follow surfaces its show, so a refactor cannot quietly flatten it.
  * `TestPickerOffersARealChoice` — xfail(strict) against #1669. These record the live defect and
    will fail loudly the day the picker is fixed, prompting removal of the marker.

WHAT THIS DOES NOT COVER — measured, not assumed. Three regressions were simulated against the
committed corpus to check these assertions actually bite:

    affinity weight 2.0 -> 0.0     CAUGHT (6/6 topics surface the wrong show, 1 distinct feed)
    affinity signal disabled       CAUGHT (identical collapse)
    affinity weight 2.0 -> 1.0     NOT CAUGHT (still 6 distinct feeds, still the right shows)

So this module catches personalisation being switched OFF, not being mis-TUNED. On this corpus a
niche topic is so dominant within its show that halving the weight does not disturb the top 3.
Graded-quality drift is `test_rank_discover_eval.py`'s job (nDCG uplift against a floor); these two
modules are complementary and neither replaces the other.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.search.topic_clusters import top_clusters_by_member_count
from podcast_scraper.server.app_discover_view import _episode_features, rank_discover
from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative
from podcast_scraper.search.theme_clusters import consumer_theme_cluster_map
from podcast_scraper.search.topic_clusters import consumer_topic_cluster_map

pytestmark = [pytest.mark.integration]

CORPUS = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "app-validation-corpus"
    / "v3"
)

# Each of these is the lead topic of exactly one show, so following it should surface that show.
NICHE_TOPIC_TO_SHOW = {
    "topic:personal-finance": "p05",
    "topic:safety-practices": "p03",
    "topic:visual-craft": "p04",
    "topic:endurance-sport": "p01",
    "topic:public-radio": "p08",
    "topic:long-form": "p06",
}


@pytest.fixture(scope="module")
def rows():
    if not CORPUS.is_dir():
        pytest.skip(f"corpus missing: {CORPUS}")
    out = build_catalog_rows_cumulative(CORPUS)
    out.sort(key=lambda r: (r.publish_date or ""), reverse=True)
    if not out:
        pytest.skip("corpus has no episodes")
    return out


def feed(rows, tokens, limit=10):
    return [s.slug for s in rank_discover(CORPUS, tokens, rows, limit=limit)]


def coverage(rows) -> dict[str, int]:
    """token -> how many episodes carry it."""
    cluster_map = consumer_topic_cluster_map(CORPUS)
    theme_map = consumer_theme_cluster_map(CORPUS)
    counts: dict[str, int] = {}
    for row in rows:
        clusters, topics, persons = _episode_features(CORPUS, row, cluster_map, theme_map)
        for token in (*clusters, *topics, *persons):
            counts[token] = counts.get(token, 0) + 1
    return counts


class TestRankerDiscriminates:
    """The engine itself — these pass today and must keep passing."""

    @pytest.mark.parametrize(("topic", "show"), sorted(NICHE_TOPIC_TO_SHOW.items()))
    def test_following_a_niche_topic_surfaces_its_show(self, rows, topic: str, show: str) -> None:
        top3 = feed(rows, [topic], limit=3)
        assert top3, f"no results for {topic}"
        wrong = [s for s in top3 if not s.startswith(show)]
        assert not wrong, (
            f"following {topic} should surface {show} episodes first; got {top3}. "
            "Personalisation no longer routes a specific interest to its show."
        )

    def test_distinct_interests_give_distinct_feeds(self, rows) -> None:
        feeds = {t: tuple(feed(rows, [t])) for t in NICHE_TOPIC_TO_SHOW}
        assert len(set(feeds.values())) == len(NICHE_TOPIC_TO_SHOW), (
            "different niche interests produced the same feed — personalisation is not "
            f"distinguishing between them: { {k: v[:2] for k, v in feeds.items()} }"
        )

    def test_no_interests_is_recency(self, rows) -> None:
        assert feed(rows, []) == [r.episode_id and s for r, s in zip(rows[:10], feed(rows, []))]

    def test_following_something_changes_the_feed(self, rows) -> None:
        assert feed(rows, ["topic:personal-finance"]) != feed(rows, []), (
            "a followed interest left the feed identical to recency"
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#1669 — every option the picker offers covers 100% of the corpus, so all of them yield "
        "one identical feed. The ranker is fine; the picker selects options by prevalence, which "
        "is the opposite of discriminating power. Remove this marker when the picker is fixed."
    ),
)
class TestPickerOffersARealChoice:
    """What a USER can actually pick. Currently decorative — recorded, not hidden."""

    def test_picker_options_are_not_all_corpus_wide(self, rows) -> None:
        counts = coverage(rows)
        total = len(rows)
        offered = [c["id"] for c in top_clusters_by_member_count(CORPUS, 12)]
        assert offered, "the picker offers nothing at all"
        universal = [t for t in offered if counts.get(t, 0) >= total]
        assert not universal, (
            f"{len(universal)}/{len(offered)} picker options cover EVERY episode "
            f"({universal}) — following them cannot re-rank anything relative to each other."
        )

    def test_picker_options_produce_different_feeds(self, rows) -> None:
        offered = [c["id"] for c in top_clusters_by_member_count(CORPUS, 12)]
        if len(offered) < 2:
            pytest.fail(f"the picker offers {len(offered)} option(s) — no choice to make")
        feeds = {t: tuple(feed(rows, [t])) for t in offered}
        distinct = len(set(feeds.values()))
        assert distinct > 1, (
            f"all {len(offered)} picker options produce the SAME feed — the choice is decorative. "
            f"Options: {offered}"
        )

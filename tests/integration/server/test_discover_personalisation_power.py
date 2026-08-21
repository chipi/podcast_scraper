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

from podcast_scraper.search.theme_clusters import consumer_theme_cluster_map
from podcast_scraper.search.topic_clusters import (
    consumer_topic_cluster_map,
    top_clusters_by_member_count,
)
from podcast_scraper.server.app_discover_view import (
    _episode_features,
    build_discover_pool,
    interest_episode_index,
    rank_discover,
)
from podcast_scraper.server.app_slugs import slug_for_row
from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

pytestmark = [pytest.mark.integration]


def _recording(seen: list):
    """Stand in for ``_episode_entities``, recording which rows the deriver visited."""

    def _episode_entities(root, row):
        seen.append(str(row))
        return [("topic", "topic:x", "X")]

    return _episode_entities


CORPUS = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "app-validation-corpus" / "v3"

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
            # noqa: E201,E202 — the spaces inside `{ {` are load-bearing: without them the
            # f-string reads `{{` as an escaped literal brace, not a nested comprehension.
            f"distinguishing between them: { {k: v[:2] for k, v in feeds.items()} }"  # noqa: E201,E202
        )

    def test_no_interests_is_recency(self, rows) -> None:
        """Empty interests must be a byte-identical recency passthrough.

        This is the shipped default for every user while APP_PERSONALIZED_RANKING is off, so it is
        the one property a scoring refactor must not disturb. (The first version of this assertion
        was `r.episode_id and s`, which evaluates to `s` — it compared the feed to itself and could
        never fail. Caught in review.)
        """
        got = feed(rows, [])
        expected = [slug_for_row(r) for r in rows[: len(got)]]
        assert got == expected, (
            "no-interest discovery is no longer plain recency order — this is the signed-out / "
            "flag-off default path"
        )

    def test_following_something_changes_the_feed(self, rows) -> None:
        assert feed(rows, ["topic:personal-finance"]) != feed(
            rows, []
        ), "a followed interest left the feed identical to recency"


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


class TestPoolPolicyIsExplicit:
    """The candidate window is a product policy, so it is asserted rather than left implicit.

    /discover ranks only the newest `4 * limit` episodes (build_discover_pool). An episode outside
    that window cannot be surfaced however well it matches, so at corpus scale discovery is "the
    newest N, re-ranked" — not "the best match in the corpus".

    This matters because the offline eval used to rank the FULL catalog while the route ranked the
    window, so the eval scored a system production never runs. Both now call build_discover_pool.
    On this 36-episode fixture the two coincide (4*10 >= 36), which is exactly why nothing noticed;
    these tests use a small limit to force the divergence the fixture otherwise hides.
    """

    def test_pool_is_bounded_to_four_times_the_limit(self, rows) -> None:
        assert len(build_discover_pool(rows, limit=2)) == 8
        assert len(build_discover_pool(rows, limit=1)) == 4
        # Never smaller than the page itself, even if the multiple would round below it.
        assert len(build_discover_pool(rows[:3], limit=5)) == 3

    def test_a_strong_match_outside_the_window_is_not_surfaced(self, rows) -> None:
        """Documents the cost of the bound, with a real interest and a real corpus."""
        topic = "topic:personal-finance"
        full = [s.slug for s in rank_discover(CORPUS, [topic], rows, limit=3)]
        assert full, "expected the unbounded ranking to surface something"
        assert all(s.startswith("p05") for s in full), full

        # The same interest, but only the newest 4*1 episodes are candidates.
        pool = build_discover_pool(rows, limit=1)
        bounded = [s.slug for s in rank_discover(CORPUS, [topic], pool, limit=1)]
        pool_slugs = {slug_for_row(r) for r in pool}

        if not any(s.startswith("p05") for s in pool_slugs):
            assert not any(s.startswith("p05") for s in bounded), (
                "a p05 episode was surfaced from a pool that contains none — the pool bound is "
                "not being applied"
            )
        else:
            # The fixture happens to carry a matching episode in the window; the bound still holds.
            assert set(bounded) <= pool_slugs

    def test_route_and_eval_share_one_pool_policy(self) -> None:
        """Guards the specific drift that made the eval reassuring about the wrong thing."""
        import inspect

        from podcast_scraper.server.routes import app_discover as route_mod
        from scripts.eval.score import rank_discover_v1 as eval_mod

        route_src = inspect.getsource(route_mod.discover)
        eval_src = inspect.getsource(eval_mod._score_user)
        assert "build_discover_pool" in route_src, "the route stopped using the shared pool policy"
        assert "build_discover_pool" in eval_src, (
            "the eval stopped using the shared pool policy — it is again scoring a system "
            "production does not run"
        )


class TestPoolIsInterestAware:
    """An old-but-matching episode must be able to reach the ranker at all.

    Recency is a proxy for relevance that fails exactly where personalisation matters: a user who
    follows scuba, whose best episodes are four years old on a feed that stopped publishing, would
    match none of the newest 4*limit. With a recency-only pool that user got NO personalisation,
    silently, while telemetry still recorded the feed as `personalized`.

    The pool is now recency UNION interest-matching, both legs bounded. These tests use a small
    limit to force the situation the 36-episode fixture otherwise hides.

    The tests that consult the real index take ``app_validation_search_index`` — ``search/
    metadata.json`` is gitignored and BUILT at test time, so a test that reads it without asking
    for it is racing whichever other module happens to build it. That is not hypothetical: on
    2026-08-21 nightly ran the builder on gw0 and ``test_index_maps_tokens_to_episodes`` on gw1 at
    the same moment, the reader got ``{}``, and the suite went red on code that had not changed.
    """

    def test_index_maps_tokens_to_episodes(self, app_validation_search_index) -> None:
        index = interest_episode_index(CORPUS)
        assert index, "no interest index — is search/metadata.json missing?"
        # Same id space as interests, and coverage matches an independent count.
        assert len(index["topic:personal-finance"]) == 4
        assert len(index["topic:expert-interviews"]) == 36
        assert all(t.startswith(("topic:", "person:")) for t in index)

    def test_a_matching_episode_outside_the_window_joins_the_pool(
        self, rows, app_validation_search_index
    ) -> None:
        topic = "topic:personal-finance"
        index = interest_episode_index(CORPUS)
        matching = index[topic]

        recency_only = build_discover_pool(rows, limit=1)
        in_window = sum(1 for r in recency_only if r.metadata_relative_path in matching)

        union = build_discover_pool(rows, limit=1, interests=[topic], root=CORPUS)
        in_union = sum(1 for r in union if r.metadata_relative_path in matching)

        assert in_union >= in_window
        assert in_union == len(matching), (
            f"the union pool carries {in_union}/{len(matching)} matching episodes — an interest's "
            "episodes are still being starved out by recency"
        )

    def test_the_union_actually_changes_what_is_surfaced(
        self, rows, app_validation_search_index
    ) -> None:
        topic = "topic:personal-finance"
        bounded = [
            s.slug
            for s in rank_discover(CORPUS, [topic], build_discover_pool(rows, limit=1), limit=3)
        ]
        union = [
            s.slug
            for s in rank_discover(
                CORPUS,
                [topic],
                build_discover_pool(rows, limit=1, interests=[topic], root=CORPUS),
                limit=3,
            )
        ]
        assert any(s.startswith("p05") for s in union), union
        assert union != bounded or all(s.startswith("p05") for s in bounded)

    def test_both_legs_stay_bounded(self, rows, app_validation_search_index) -> None:
        """Cost control: the union is at most two windows, never the whole catalog."""
        union = build_discover_pool(
            rows, limit=2, interests=["topic:expert-interviews"], root=CORPUS
        )
        assert len(union) <= 2 * 8
        assert len(union) < len(rows), "the pool grew to the entire corpus"

    def test_no_interests_is_unchanged(self, rows) -> None:
        assert list(build_discover_pool(rows, limit=3, interests=[], root=CORPUS)) == list(
            build_discover_pool(rows, limit=3)
        )

    def test_missing_search_sidecar_falls_back_to_recency(self, rows, tmp_path) -> None:
        """A corpus without a search index must still serve a feed, not error."""
        assert interest_episode_index(tmp_path) == {}
        assert list(
            build_discover_pool(rows, limit=3, interests=["topic:personal-finance"], root=tmp_path)
        ) == list(build_discover_pool(rows, limit=3))


class TestDerivationTracksRecentListening:
    """`derive_interests` is bounded to 40 episodes — the bound must drop the OLDEST, not the
    alphabetically-latest.

    Slugs are ``{feed-slug}-{hash}``, so the old ``sorted(slugs)[:max_episodes]`` grouped by SHOW.
    Past 40 episodes a heavy listener's derived profile froze on whichever shows happen to be
    spelled early, and nothing they played afterwards could ever move it. That is a silent,
    permanent personalisation failure — the profile keeps being *used*, it just stops being *true*.
    """

    OLD_SHOW = "aaa-legacy-show"
    NEW_SHOW = "zzz-new-obsession"

    def _persona(
        self, monkeypatch, tmp_path: Path, *, old_episodes: int = 40, new_episodes: int = 5
    ):
        """N old episodes on an alphabetically-early show + M recent ones on a late-sorting show."""
        from podcast_scraper.server import app_user_corpus as uc, app_user_state

        uid = "u_0123456789abcdef01234567"
        old = [f"{self.OLD_SHOW}-{i:04d}" for i in range(old_episodes)]
        new = [f"{self.NEW_SHOW}-{i:04d}" for i in range(new_episodes)]

        # Real per-user writes: the recency signal has to survive the actual store round-trip.
        for n, slug in enumerate(old):
            app_user_state.set_playback(tmp_path, uid, slug, 900.0, 1_700_000_000 + n)
        for n, slug in enumerate(new):
            app_user_state.set_playback(tmp_path, uid, slug, 900.0, 1_800_000_000 + n)

        monkeypatch.setattr(uc, "slug_durations", lambda root: {s: 1800.0 for s in old + new})
        monkeypatch.setattr(uc, "build_catalog_rows_cumulative", lambda root: old + new)
        monkeypatch.setattr(uc, "slug_for_row", lambda r: r)
        monkeypatch.setattr(
            uc,
            "_episode_entities",
            lambda root, row: (
                [("topic", "topic:fresh", "Fresh")]
                if str(row).startswith(self.NEW_SHOW)
                else [("topic", "topic:legacy", "Legacy")]
            ),
        )
        return uc, tmp_path, uid

    def test_new_listening_still_moves_the_profile(self, monkeypatch, tmp_path) -> None:
        uc, data_dir, uid = self._persona(monkeypatch, tmp_path)
        got = uc.derive_interests(data_dir, data_dir, uid, k=8)
        # The regression: with lexicographic selection this is ["topic:legacy"] and the five
        # episodes the user actually just listened to are invisible.
        assert "topic:fresh" in got, got

    def test_the_bound_still_holds(self, monkeypatch, tmp_path) -> None:
        """Recency ordering must not smuggle in more than `max_episodes` KG loads."""
        uc, data_dir, uid = self._persona(monkeypatch, tmp_path)
        seen: list[str] = []
        monkeypatch.setattr(
            uc,
            "_episode_entities",
            _recording(seen),
        )
        uc.derive_interests(data_dir, data_dir, uid, k=8, max_episodes=12)
        assert len(seen) == 12, len(seen)
        assert all(s.startswith(self.NEW_SHOW) for s in seen[:5]), seen[:5]

    def test_episodes_without_a_recency_signal_are_last_but_eligible(
        self, monkeypatch, tmp_path
    ) -> None:
        """A saved-insight-only episode carries no timestamp `_most_recently_engaged` can read.

        `_captured_slugs` puts it in the episode set via `get_favorites`, but favorites are not one
        of the three recency sources — so it scores 0. It must still be *eligible* (a corpus with
        no engagement metadata degrades to a bounded subset, not to nothing), just ranked last.
        """
        from podcast_scraper.server import app_user_state

        captured = "mmm-saved-insight-0001"
        uc, data_dir, uid = self._persona(monkeypatch, tmp_path, old_episodes=0, new_episodes=2)
        app_user_state.add_favorite(
            data_dir, uid, {"kind": "insight", "ref": "ins-1", "slug": captured}
        )
        heard = [f"{self.NEW_SHOW}-{i:04d}" for i in range(2)]
        monkeypatch.setattr(uc, "build_catalog_rows_cumulative", lambda root: heard + [captured])
        monkeypatch.setattr(
            uc,
            "_episode_entities",
            lambda root, row: (
                [("topic", "topic:captured", "Captured")]
                if row == captured
                else [("topic", "topic:fresh", "Fresh")]
            ),
        )
        # Room for all three: the signal-less episode is reached.
        assert "topic:captured" in uc.derive_interests(data_dir, data_dir, uid, k=8)
        # Room for two: the timestamped ones win, and it is the one dropped.
        assert uc.derive_interests(data_dir, data_dir, uid, k=8, max_episodes=2) == ["topic:fresh"]

    def test_derivation_is_deterministic(self, monkeypatch, tmp_path) -> None:
        uc, data_dir, uid = self._persona(monkeypatch, tmp_path)
        first = uc.derive_interests(data_dir, data_dir, uid, k=8)
        assert all(uc.derive_interests(data_dir, data_dir, uid, k=8) == first for _ in range(3))


class TestAFollowDoesNotFlattenTheFeedsSenseOfTime:
    """Following one topic must not reshuffle the episodes that have nothing to do with it (#22).

    Before recency became a graded signal, any non-empty interest set sorted the WHOLE pool by
    score, with recency surviving only as the `-idx` tie-break. So a single follow turned the
    other ~90% of the feed from newest-first into enrichment-depth-first — an old episode could
    lead today's purely because it carried a richer KG.

    Measured here rather than asserted by feel: take the episodes that do NOT match the interest
    and count how many pairs are still in publish order.
    """

    TOPIC = "topic:personal-finance"
    SHOW = "p05"

    @staticmethod
    def _config(recency_weight: float):
        from podcast_scraper.server.app_ranking_config import (
            DEFAULT_RANKING_CONFIG,
            RankingSignal,
            SIGNAL_RECENCY,
        )

        return DEFAULT_RANKING_CONFIG.__class__(
            signals=tuple(
                (
                    RankingSignal(
                        SIGNAL_RECENCY,
                        enabled=recency_weight > 0,
                        weight=recency_weight,
                        params=DEFAULT_RANKING_CONFIG.params_of(SIGNAL_RECENCY),
                    )
                    if s.name == SIGNAL_RECENCY
                    else s
                )
                for s in DEFAULT_RANKING_CONFIG.signals
            )
        )

    def _time_order_agreement(self, rows, config) -> float:
        """Fraction of non-matching episode PAIRS still in publish order (1.0 = perfect)."""
        pool = build_discover_pool(rows, limit=10, interests=[self.TOPIC], root=CORPUS)
        got = [s.slug for s in rank_discover(CORPUS, [self.TOPIC], pool, limit=36, config=config)]
        by_recency = [slug_for_row(r) for r in rows]
        unrelated = [s for s in got if not s.startswith(self.SHOW)]
        ideal = [s for s in by_recency if s in set(unrelated)]
        pos = {s: i for i, s in enumerate(ideal)}
        pairs = [(i, j) for i in range(len(unrelated)) for j in range(i + 1, len(unrelated))]
        if not pairs:
            pytest.skip("corpus too small to measure pairwise order")
        ordered = sum(1 for i, j in pairs if pos[unrelated[i]] < pos[unrelated[j]])
        return ordered / len(pairs)

    def test_recency_restores_time_order_among_unrelated_episodes(self, rows) -> None:
        off = self._time_order_agreement(rows, self._config(0.0))
        on = self._time_order_agreement(rows, self._config(0.5))
        assert on > off, (
            f"the recency signal changed nothing ({on:.1%} vs {off:.1%}). On this corpus a 30-day "
            "half-life produced exactly this — the second-newest episode is already months old, so "
            "its boost rounds to zero and the signal is inert. Re-measure the half-life against "
            "the corpus span before assuming it works."
        )
        # Measured on the committed corpus WITH per-feed normalisation (#23): 76.0% off -> 96.2%
        # on. Normalisation made this signal matter far MORE, not less: un-normalised, significance
        # correlated incidentally with recency and propped the baseline up to 94.4%, so recency
        # looked like it was worth ~4 points. Once coverage bias stopped ordering the feed, the
        # real number showed up — recency is doing ~20 points of work.
        #
        # 96.2% -> 94.0% when the half-life moved 365 -> 730 (2026-08-19). That is a real cost,
        # recorded rather than hidden: a longer half-life flattens the freshness gradient, so
        # significance re-orders slightly more of the unrelated tail. It was accepted deliberately
        # — the half-life now encodes WHEN CONTENT GOES STALE (target window 2-4 years) instead of
        # being fitted to whatever corpus we happened to have, and a four-year-old episode scoring
        # 0.06 was the worse problem. See app_ranking_config.py for the full table.
        #
        # The floor tracks the measurement; `on > off` above is the assertion that actually
        # protects the behaviour, and it is untouched. If this drops much below 0.93 something
        # else has changed and it is worth understanding rather than re-baselining again.
        assert on >= 0.93, f"unrelated episodes are only {on:.1%} in publish order"

    def test_the_interest_still_wins_the_top_slots(self, rows) -> None:
        """Recency must not buy freshness at the cost of personalisation — affinity outranks it."""
        pool = build_discover_pool(rows, limit=10, interests=[self.TOPIC], root=CORPUS)
        top3 = [
            s.slug
            for s in rank_discover(CORPUS, [self.TOPIC], pool, limit=3, config=self._config(0.5))
        ]
        assert all(s.startswith(self.SHOW) for s in top3), top3

    def test_the_shipped_default_is_the_measured_one(self, rows) -> None:
        """Guards the config values themselves: the default must behave like the tuned setting."""
        from podcast_scraper.server.app_ranking_config import DEFAULT_RANKING_CONFIG

        assert self._time_order_agreement(rows, DEFAULT_RANKING_CONFIG) >= 0.93

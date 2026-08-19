"""The discover pool must not become a keyhole as the corpus grows (#1682, measured 2026-08-19).

`DISCOVER_POOL_MULTIPLE = 4` made the window a fixed `4 * limit` — 48 at the default page size —
regardless of how many episodes existed. Measured against the production corpus (678 episodes,
14 feeds) via `inspect-prod-corpus.yml -f checks=capability_audit`:

    recency leg reaches 48/678 (7.1%)
    630 episodes cannot reach the ranker without matching a followed interest

So discovery got NARROWER exactly as there was more to discover: at 1,500 episodes the same 48 is
3%. The relevance leg was capped at the same fixed number, so a followed interest could rescue at
most 48 older episodes too.

WHY NO EXISTING TEST CAUGHT IT
`app-validation-corpus/v3` holds 36 episodes and `4 * 12 = 48 > 36`, so the pool was the entire
corpus in every test we own. The window was never a constraint, the union never ran, and the
truncation was invisible. This file therefore builds its own corpora at sizes the fixture cannot
reach — that is the point of it.
"""

from __future__ import annotations

import pytest

from podcast_scraper.server.app_discover_view import (
    _pool_window,
    DISCOVER_POOL_CORPUS_SHARE,
    DISCOVER_POOL_MAX,
    DISCOVER_POOL_MIN_LIMIT_FOR_SHARE,
    DISCOVER_POOL_MULTIPLE,
)

pytestmark = [pytest.mark.unit]

#: The production reading this was tuned against.
PROD_EPISODES = 678
DEFAULT_LIMIT = 12


class TestTheWindowScalesWithTheCorpus:
    def test_the_production_corpus_is_no_longer_a_keyhole(self) -> None:
        """678 episodes: 48 (7.1%) before, ~102 (15%) now."""
        window = _pool_window(DEFAULT_LIMIT, PROD_EPISODES)
        assert window == int(PROD_EPISODES * DISCOVER_POOL_CORPUS_SHARE)
        assert window / PROD_EPISODES == pytest.approx(DISCOVER_POOL_CORPUS_SHARE, abs=0.01)
        # The regression this exists to prevent: the old fixed window.
        assert window > DEFAULT_LIMIT * DISCOVER_POOL_MULTIPLE

    def test_the_share_holds_as_the_corpus_grows(self) -> None:
        """The failure mode was a share that SHRANK with growth. It must not, below the cap."""
        for size in (700, 1_000, 2_000):
            window = _pool_window(DEFAULT_LIMIT, size)
            if window >= DISCOVER_POOL_MAX:
                continue
            assert window / size == pytest.approx(DISCOVER_POOL_CORPUS_SHARE, abs=0.01), size

    def test_growth_never_shrinks_the_window(self) -> None:
        """Monotonicity: a bigger corpus can never see less of itself than a smaller one."""
        windows = [_pool_window(DEFAULT_LIMIT, n) for n in range(50, 3_000, 137)]
        assert windows == sorted(windows)


class TestItStaysBounded:
    def test_a_large_corpus_is_capped(self) -> None:
        """Ranking is one KG artifact load per candidate, so the share cannot be unbounded."""
        assert _pool_window(DEFAULT_LIMIT, 100_000) == DISCOVER_POOL_MAX

    def test_the_cap_is_reached_where_expected(self) -> None:
        at_cap = int(DISCOVER_POOL_MAX / DISCOVER_POOL_CORPUS_SHARE)
        assert _pool_window(DEFAULT_LIMIT, at_cap * 2) == DISCOVER_POOL_MAX


class TestSmallCorporaAreUnchanged:
    """The fixture, and any small deployment, must behave exactly as before.

    This is what makes the change safe to ship without re-baselining every existing expectation:
    below the crossover the page-size term still wins, so the pool is what it always was.
    """

    def test_the_36_episode_fixture_is_untouched(self) -> None:
        assert _pool_window(DEFAULT_LIMIT, 36) == DEFAULT_LIMIT * DISCOVER_POOL_MULTIPLE

    @pytest.mark.parametrize("size", [0, 1, 10, 36, 100, 300])
    def test_below_the_crossover_the_page_size_term_wins(self, size: int) -> None:
        crossover = (DEFAULT_LIMIT * DISCOVER_POOL_MULTIPLE) / DISCOVER_POOL_CORPUS_SHARE
        if size >= crossover:
            pytest.skip(f"{size} is above the crossover ({crossover:.0f})")
        assert _pool_window(DEFAULT_LIMIT, size) == DEFAULT_LIMIT * DISCOVER_POOL_MULTIPLE

    def test_an_empty_corpus_does_not_produce_a_zero_window(self) -> None:
        """A zero window would rank nothing and read as 'no results' rather than as a bug."""
        assert _pool_window(DEFAULT_LIMIT, 0) > 0


class TestAProbeSizedLimitIsLeftAlone:
    """My first attempt at this change broke `TestPoolPolicyIsExplicit`, and that test was right.

    It calls `build_discover_pool(rows, limit=1)` and `limit=2` deliberately, to force the bound
    that a 36-episode fixture otherwise hides — they are the only tests that demonstrate
    truncation at all. Applying a corpus share to those calls widened the pool and destroyed the
    demonstration. Worse, my own test asserted "small corpora are unchanged" only at limit=12, so
    it did not catch the overreach; the older test did.

    A request for one or two episodes is a probe or a widget, not a discovery feed.
    """

    @pytest.mark.parametrize("limit", [1, 2, 4])
    def test_below_the_threshold_only_the_page_term_applies(self, limit: int) -> None:
        assert limit < DISCOVER_POOL_MIN_LIMIT_FOR_SHARE
        for corpus in (36, PROD_EPISODES, 5_000):
            assert _pool_window(limit, corpus) == limit * DISCOVER_POOL_MULTIPLE, corpus

    def test_the_exact_values_the_older_test_pins(self) -> None:
        """Same numbers as TestPoolPolicyIsExplicit, asserted here so a future change to the
        threshold breaks BOTH files rather than silently only that one."""
        assert _pool_window(2, 36) == 8
        assert _pool_window(1, 36) == 4

    def test_at_the_threshold_the_share_applies_again(self) -> None:
        limit = DISCOVER_POOL_MIN_LIMIT_FOR_SHARE
        assert _pool_window(limit, PROD_EPISODES) > limit * DISCOVER_POOL_MULTIPLE


class TestThePageSizeStillMatters:
    def test_a_larger_page_asks_for_more_candidates(self) -> None:
        small = _pool_window(5, 200)
        large = _pool_window(50, 200)
        assert large > small

    def test_the_window_is_never_smaller_than_the_page(self) -> None:
        """Fewer candidates than the page size would truncate the feed itself."""
        for limit in (1, 5, 12, 50):
            assert _pool_window(limit, 10) >= limit


class TestThePoolIsTunableLikeEverySignal:
    """Admission must be swappable from config, or an autoresearch sweep cannot vary it (#1795).

    Every scoring weight already flows through `ranking_config_from_dict`. The pool did not: it
    was module constants, so the ONE parameter no weight can compensate for — an episode the pool
    excluded cannot be promoted by any amount of affinity — was the only one nothing could
    override. That is backwards, and it is how the window stayed a fixed 48 while the corpus grew.
    """

    @staticmethod
    def _config(**params):
        from podcast_scraper.server.app_ranking_config import (
            DEFAULT_RANKING_CONFIG,
            RankingSignal,
            SIGNAL_DISCOVER_POOL,
        )

        merged = {**DEFAULT_RANKING_CONFIG.params_of(SIGNAL_DISCOVER_POOL), **params}
        return DEFAULT_RANKING_CONFIG.__class__(
            signals=tuple(
                (
                    RankingSignal(SIGNAL_DISCOVER_POOL, enabled=True, weight=0.0, params=merged)
                    if s.name == SIGNAL_DISCOVER_POOL
                    else s
                )
                for s in DEFAULT_RANKING_CONFIG.signals
            )
        )

    def test_the_shipped_default_matches_the_module_constants(self) -> None:
        """The config and the fallbacks must not drift apart into two different policies."""
        from podcast_scraper.server.app_ranking_config import (
            DEFAULT_RANKING_CONFIG,
            SIGNAL_DISCOVER_POOL,
        )

        p = DEFAULT_RANKING_CONFIG.params_of(SIGNAL_DISCOVER_POOL)
        assert p["corpus_share"] == DISCOVER_POOL_CORPUS_SHARE
        assert p["page_multiple"] == DISCOVER_POOL_MULTIPLE
        assert p["max_candidates"] == DISCOVER_POOL_MAX
        assert p["min_limit_for_share"] == DISCOVER_POOL_MIN_LIMIT_FOR_SHARE

    def test_a_sweep_can_widen_the_share(self) -> None:
        wide = _pool_window(DEFAULT_LIMIT, PROD_EPISODES, self._config(corpus_share=0.40))
        assert wide == int(PROD_EPISODES * 0.40)
        assert wide > _pool_window(DEFAULT_LIMIT, PROD_EPISODES)

    def test_a_sweep_can_narrow_the_share(self) -> None:
        narrow = _pool_window(DEFAULT_LIMIT, PROD_EPISODES, self._config(corpus_share=0.05))
        assert narrow < _pool_window(DEFAULT_LIMIT, PROD_EPISODES)

    def test_a_sweep_can_move_the_ceiling(self) -> None:
        assert _pool_window(DEFAULT_LIMIT, 100_000, self._config(max_candidates=1_000)) == 1_000

    def test_omitting_the_signal_falls_back_to_the_constants(self) -> None:
        assert _pool_window(DEFAULT_LIMIT, PROD_EPISODES, None) == _pool_window(
            DEFAULT_LIMIT, PROD_EPISODES
        )


class TestAMalformedOverrideCannotEmptyThePool:
    """A bad override must degrade to the default, never to zero candidates.

    A zero-width pool ranks nothing and renders as "no episodes" — indistinguishable from an empty
    corpus, and silent. Sweeps generate parameter sets programmatically, so this is the failure
    mode that would actually occur.
    """

    @pytest.mark.parametrize("bad", [0, -1, "abc", None, float("nan"), float("inf"), True, {}])
    def test_a_bad_share_falls_back(self, bad) -> None:
        window = _pool_window(
            DEFAULT_LIMIT,
            PROD_EPISODES,
            TestThePoolIsTunableLikeEverySignal._config(corpus_share=bad),
        )
        assert window == _pool_window(DEFAULT_LIMIT, PROD_EPISODES), bad

    @pytest.mark.parametrize("bad", [0, -5, "x", None, True])
    def test_a_bad_ceiling_falls_back(self, bad) -> None:
        window = _pool_window(
            DEFAULT_LIMIT,
            100_000,
            TestThePoolIsTunableLikeEverySignal._config(max_candidates=bad),
        )
        assert window == DISCOVER_POOL_MAX, bad

    def test_no_override_can_produce_a_zero_window(self) -> None:
        for params in (
            {"corpus_share": 0},
            {"page_multiple": 0},
            {"max_candidates": 0},
            {"corpus_share": -1, "page_multiple": -1, "max_candidates": -1},
        ):
            cfg = TestThePoolIsTunableLikeEverySignal._config(**params)
            assert _pool_window(DEFAULT_LIMIT, PROD_EPISODES, cfg) > 0, params

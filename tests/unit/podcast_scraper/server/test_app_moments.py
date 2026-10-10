"""Moments (operator 2026-10-10): which insights play in the quick-listening reel, and how many.

Pure dict-in / list-out over a hand-built GI artifact — no HTTP, no disk.
"""

from __future__ import annotations

from typing import Any

import pytest

from podcast_scraper.server.app_moments import (
    build_clip,
    moment_count,
    MomentsConfig,
    pick_moments,
    score_candidates,
)
from podcast_scraper.server.schemas import AppQuote

pytestmark = [pytest.mark.unit]


class _Gi:
    """Builds a GI artifact: insights, their timed quotes, and the topics they are ABOUT."""

    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = [
            {"id": "person:ann", "type": "Person", "properties": {"name": "Ann"}}
        ]
        self.edges: list[dict[str, Any]] = []

    def insight(
        self,
        iid: str,
        *,
        kind: str = "observation",
        quotes: list[tuple[float, float]] = ((60.0, 80.0),),  # type: ignore[assignment]
        topics: tuple[str, ...] = (),
        **props: Any,
    ) -> "_Gi":
        self.nodes.append(
            {
                "id": iid,
                "type": "Insight",
                "properties": {
                    "text": f"text of {iid}",
                    "grounded": True,
                    "insight_type": kind,
                    "routing_tag": "surface",
                    **props,
                },
            }
        )
        for n, (start, end) in enumerate(quotes):
            qid = f"quote:{iid}:{n}"
            self.nodes.append(
                {
                    "id": qid,
                    "type": "Quote",
                    "properties": {
                        "text": f"quote {n} of {iid}",
                        "timestamp_start_ms": int(start * 1000),
                        "timestamp_end_ms": int(end * 1000),
                    },
                }
            )
            self.edges.append({"type": "SUPPORTED_BY", "from": iid, "to": qid})
            self.edges.append({"type": "SPOKEN_BY", "from": qid, "to": "person:ann"})
        for t in topics:
            self.edges.append({"type": "ABOUT", "from": iid, "to": t})
        return self

    def build(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges}


class TestCount:
    @pytest.mark.parametrize(
        ("minutes", "expected"), [(30, 5), (48, 8), (90, 15), (180, 15), (10, 5)]
    )
    def test_one_per_six_minutes_between_five_and_fifteen(
        self, minutes: int, expected: int
    ) -> None:
        assert moment_count(minutes * 60, 100, MomentsConfig()) == expected

    def test_never_more_than_the_episode_has(self) -> None:
        assert moment_count(48 * 60, 3, MomentsConfig()) == 3
        assert moment_count(48 * 60, 0, MomentsConfig()) == 0


class TestScore:
    def test_a_central_claim_backed_three_times_beats_a_passing_observation(self) -> None:
        gi = (
            _Gi()
            .insight(
                "insight:strong",
                kind="claim",
                quotes=[(100, 120), (400, 420), (900, 915)],
                topics=("topic:risk",),
            )
            .insight("insight:other", kind="observation", topics=("topic:risk",))
            .insight("insight:weak", kind="observation", quotes=[(1500, 1520)])
            .build()
        )
        scored = {m.insight_id: m for m in score_candidates(gi, MomentsConfig())}
        assert scored["insight:strong"].score > scored["insight:weak"].score
        assert scored["insight:strong"].components["depth"] == 1.0
        assert scored["insight:strong"].components["centrality"] == 1.0
        assert scored["insight:weak"].components["centrality"] == 0.0

    def test_first_quote_rule_is_the_earliest_quote_capped_at_thirty_seconds(self) -> None:
        gi = _Gi().insight("insight:a", quotes=[(500, 600), (200, 290)]).build()
        (m,) = score_candidates(gi, MomentsConfig(clip_rule="first_quote"))
        assert m.start_ms == 200_000
        assert m.end_ms == 230_000  # 90 s quote, cut at the 30 s cap
        assert m.quote_text == "quote 1 of insight:a"

    def test_only_insights_the_player_shows_and_that_have_a_timed_quote(self) -> None:
        gi = (
            _Gi()
            .insight("insight:ok")
            .insight("insight:dropped", routing_tag="drop")
            .insight("insight:unnamed", surfaceable=False)
            .insight("insight:untimed", quotes=[])
            .build()
        )
        assert [m.insight_id for m in score_candidates(gi, MomentsConfig())] == ["insight:ok"]

    def test_weights_change_the_order(self) -> None:
        gi = (
            _Gi()
            .insight("insight:deep", kind="observation", quotes=[(60, 80), (300, 320), (600, 620)])
            .insight("insight:claim", kind="claim", quotes=[(900, 920)])
            .build()
        )
        by_depth = MomentsConfig.from_dict({"weights": {"depth": 1.0, "kind": 0.0}})
        by_kind = MomentsConfig.from_dict({"weights": {"depth": 0.0, "kind": 1.0}})

        def top(cfg: MomentsConfig) -> str:
            return max(score_candidates(gi, cfg), key=lambda m: m.score).insight_id

        assert top(by_depth) == "insight:deep"
        assert top(by_kind) == "insight:claim"


class TestPick:
    def test_spaced_apart_and_returned_in_timeline_order(self) -> None:
        gi = _Gi()
        # Two strong claims 60 s apart: the spacing rule keeps only one in the first pass.
        gi.insight("insight:a", kind="claim", quotes=[(1000, 1020), (1100, 1120), (1200, 1220)])
        gi.insight("insight:b", kind="claim", quotes=[(1060, 1080), (1160, 1180), (1260, 1280)])
        for n in range(6):
            gi.insight(f"insight:o{n}", quotes=[(n * 400 + 10, n * 400 + 30)])
        picked = pick_moments(gi.build(), 30 * 60)  # 30 min → 5 moments
        assert len(picked) == 5
        assert [m.start_ms for m in picked] == sorted(m.start_ms for m in picked)
        ids = [m.insight_id for m in picked]
        assert ("insight:a" in ids) != ("insight:b" in ids)

    def test_unknown_duration_falls_back_to_the_last_quote(self) -> None:
        gi = _Gi()
        for n in range(20):
            gi.insight(f"insight:{n}", quotes=[(n * 300, n * 300 + 20)])
        # Last quote ends at ~95 min → 16 → capped at 15.
        assert len(pick_moments(gi.build(), None)) == 15

    def test_no_candidates_no_moments(self) -> None:
        assert pick_moments({"nodes": [], "edges": []}, 3600) == []
        assert pick_moments(None, 3600) == []


class TestConfig:
    def test_env_json_overrides_and_merges_weights(self) -> None:
        cfg = MomentsConfig.from_env(
            {"APP_MOMENTS_CONFIG": '{"max_count": 10, "weights": {"clip": 0.5}, "bogus": 1}'}
        )
        assert cfg.max_count == 10
        assert cfg.weights["clip"] == 0.5
        assert cfg.weights["depth"] == MomentsConfig().weights["depth"]

    def test_unset_or_broken_json_is_the_default(self) -> None:
        assert MomentsConfig.from_env({}) == MomentsConfig()
        assert MomentsConfig.from_env({"APP_MOMENTS_CONFIG": "{not json"}) == MomentsConfig()


def _q(start: float, end: float, text: str = "q") -> AppQuote:
    return AppQuote(text=text, start_ms=int(start * 1000), end_ms=int(end * 1000))


def _clip(quotes: list[AppQuote], segs: Any, cfg: MomentsConfig) -> tuple[int, int, str]:
    out = build_clip(quotes, segs, cfg)
    assert out is not None
    return out


class TestClip:
    """The segments rule (default): start at a sentence, cover nearby quotes, at least 12 s."""

    SEGS = [
        (90_000, 100_000, "Before."),
        (100_000, 104_000, "So here is the thing."),
        (104_000, 109_000, "Risk is about surviving."),
        (109_000, 118_000, "You size for the worst week."),
        (118_000, 140_000, "A long tangent about something else entirely."),
    ]

    def test_starts_at_the_segment_and_extends_to_the_minimum(self) -> None:
        start, end, text = _clip([_q(102, 106)], self.SEGS, MomentsConfig())
        assert start == 100_000  # back to the segment the quote starts in (2 s lead)
        assert end == 118_000  # whole segments until >= 12 s
        assert text == "So here is the thing. Risk is about surviving. You size for the worst week."

    def test_never_past_the_thirty_second_cap(self) -> None:
        start, end, _ = _clip([_q(110, 112)], self.SEGS, MomentsConfig())
        assert start == 109_000
        assert end - start <= 30_000

    def test_a_segment_too_far_back_is_not_a_lead_in(self) -> None:
        start, _, _ = _clip([_q(98, 99)], self.SEGS, MomentsConfig(clip_lead_max_seconds=5))
        assert start == 98_000  # the holding segment starts 8 s earlier

    def test_close_quotes_of_the_same_insight_join_the_clip(self) -> None:
        start, end, text = _clip([_q(10, 14, "one"), _q(18, 22, "two")], None, MomentsConfig())
        assert (start, end) == (10_000, 22_000)
        assert text == "one two"  # no segments: the quotes it covers

    def test_a_far_quote_does_not(self) -> None:
        _, end, _ = _clip([_q(10, 14), _q(200, 210)], None, MomentsConfig())
        assert end == 14_000

    def test_no_timed_quote_no_clip(self) -> None:
        assert build_clip([AppQuote(text="x")], self.SEGS, MomentsConfig()) is None


class TestRanking:
    def test_default_is_the_player_order_and_score_is_opt_in(self) -> None:
        gi = (
            _Gi()
            .insight("insight:first", kind="observation", quotes=[(60, 80)])
            .insight("insight:deep", kind="claim", quotes=[(900, 920), (1200, 1220), (1500, 1520)])
            .build()
        )
        one = MomentsConfig(min_count=1, max_count=1)
        assert [m.insight_id for m in pick_moments(gi, 600, one)] == ["insight:first"]
        by_score = MomentsConfig(min_count=1, max_count=1, ranking="score")
        assert [m.insight_id for m in pick_moments(gi, 600, by_score)] == ["insight:deep"]


class TestShortEpisodes:
    def test_the_gap_shrinks_so_a_short_episode_still_gets_its_moments(self) -> None:
        gi = _Gi()
        # A 6-minute episode with candidates every 50 s: a fixed 3-minute gap allowed two.
        for n in range(7):
            gi.insight(f"insight:{n}", quotes=[(10 + n * 50, 20 + n * 50)])
        picked = pick_moments(gi.build(), 6 * 60)
        assert len(picked) == 5

    def test_long_episodes_keep_the_full_gap(self) -> None:
        gi = _Gi()
        for n in range(40):
            gi.insight(f"insight:{n}", quotes=[(n * 60, n * 60 + 15)])
        picked = pick_moments(gi.build(), 40 * 60)  # 40 min → 7 moments
        starts = [m.start_ms for m in picked]
        assert all(b - a >= 180_000 for a, b in zip(starts, starts[1:]))

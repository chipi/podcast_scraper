"""Moments (operator 2026-10-10): which insights play in the quick-listening reel, and how many.

Pure dict-in / list-out over a hand-built GI artifact — no HTTP, no disk.
"""

from __future__ import annotations

from typing import Any

import pytest

from podcast_scraper.server.app_moments import build_clip, moment_count, MomentsConfig, pick_moments
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


class TestPick:

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


class TestClipNeverPlaysAnAd:
    """A short listen ran on into "we'll be right back after a word from our sponsors" (operator
    2026-10-12): the extension to the minimum length took the next segment, which was the break."""

    SEGS = [
        (90_000, 100_000, "Before."),
        (100_000, 104_000, "So here is the thing."),
        (104_000, 109_000, "Risk is about surviving."),
        (109_000, 115_000, "We'll be right back after a quick word from our sponsors."),
        (115_000, 125_000, "Today's episode is sponsored by Linear."),
        (125_000, 140_000, "And we're back."),
    ]
    BREAK = [(109_000, 125_000)]

    def test_the_extension_stops_at_the_break(self) -> None:
        start, end, text = _clip_ads([_q(102, 106)], self.SEGS, self.BREAK)
        assert (start, end) == (100_000, 109_000)
        assert text == "So here is the thing. Risk is about surviving."
        # Without the ads it is the old leak, which is what this guards.
        old = build_clip([_q(102, 106)], self.SEGS, MomentsConfig())
        assert old is not None and old[1] == 115_000

    def test_the_lead_in_does_not_reach_back_into_an_ad(self) -> None:
        segs = [
            (0, 18_000, "This episode is brought to you by Strava."),
            (18_000, 40_000, "Point."),
        ]
        start, _, text = _clip_ads([_q(16, 22)], segs, [(0, 18_000)])
        assert start == 18_000 and "Strava" not in text

    def test_a_quote_inside_an_ad_is_no_moment(self) -> None:
        assert build_clip([_q(110, 114)], self.SEGS, MomentsConfig(), self.BREAK) is None

    def test_an_ad_inside_the_clip_leaves_no_clip(self) -> None:
        quotes = [_q(100, 105, "one"), _q(112, 118, "two")]  # 7 s apart: joined into one clip
        ads = [(106_000, 110_000)]
        assert build_clip(quotes, None, MomentsConfig(), ads) is None

    def test_the_first_quote_rule_is_cut_clear_too(self) -> None:
        cfg = MomentsConfig(clip_rule="first_quote")
        start, end, _ = _clip_ads([_q(100, 112)], self.SEGS, self.BREAK, cfg)
        assert (start, end) == (100_000, 109_000)

    def test_the_extension_stops_where_the_speaker_changes(self) -> None:
        """The next voice was a host's sponsor credit or native ad as often as a question."""
        segs = [
            (96_000, 102_000, "Suspension service intervals are a risk-management decision."),
            (
                102_000,
                108_000,
                "Y'know what, I'll just say it: Linear runs my whole engineering ops.",
            ),
        ]
        start, end, text = _clip_ads([_q(96, 100)], segs, (), turn_starts=frozenset({102_000}))
        assert (start, end) == (96_000, 102_000) and "Linear" not in text
        assert build_clip([_q(96, 100)], segs, MomentsConfig())[1] == 108_000  # type: ignore[index]

    def test_no_ads_is_the_old_clip(self) -> None:
        assert build_clip([_q(102, 106)], self.SEGS, MomentsConfig(), ()) == build_clip(
            [_q(102, 106)], self.SEGS, MomentsConfig()
        )


def _clip_ads(
    quotes: list[AppQuote],
    segs: Any,
    ads: Any,
    cfg: MomentsConfig | None = None,
    turn_starts: frozenset[int] = frozenset(),
) -> tuple[int, int, str]:
    out = build_clip(quotes, segs, cfg or MomentsConfig(), ads, turn_starts)
    assert out is not None
    return out


class TestShortEpisodes:

    def test_long_episodes_keep_the_full_gap(self) -> None:
        gi = _Gi()
        for n in range(40):
            gi.insight(f"insight:{n}", quotes=[(n * 60, n * 60 + 15)])
        picked = pick_moments(gi.build(), 40 * 60)  # 40 min → 7 moments
        starts = [m.start_ms for m in picked]
        assert all(b - a >= 180_000 for a, b in zip(starts, starts[1:]))

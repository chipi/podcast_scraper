"""Trending episodes (operator 2026-10-10): rising-topic cards interleaved with most-engaged ones.

The data sources (topic momentum, speaker takes, the engagement series, GI insights) are replaced
with small fakes: what is under test is the SELECTION — interleaving, one card per episode, the
engagement window and floor, and "mine" keeping to the listener's world (ADR-162) — and each card's
reason (ADR-163): a real quote and where it is said, never the insight's own text.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

import podcast_scraper.server.app_trending_episodes as te

pytestmark = [pytest.mark.unit]

_NOW = "2026-07-20T00:00:00Z"  # ISO week 2026-W30


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    *,
    topics: list[tuple[str, str]],
    takes: dict[str, list[tuple[str, str]]],
    engagement: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """topics: [(id, label)]; takes: topic -> [(speaker, slug)]; engagement: slug -> {week: n}."""
    seen: dict[str, Any] = {}

    def fake_trending(root: Path, data_dir: Path | None, **kw: Any) -> list[Any]:
        seen["trending"] = kw
        return [NS(entity_id=i, label=lbl) for i, lbl in topics]

    def fake_perspectives(root: Path, topic_id: str, *, mine_slugs: set[str] | None = None) -> Any:
        rows = [
            (spk, slug)
            for spk, slug in takes.get(topic_id, [])
            if mine_slugs is None or slug in mine_slugs
        ]
        if not rows:
            return None
        return NS(
            perspectives=[
                NS(
                    person_name=spk,
                    insights=[
                        NS(
                            id=f"insight:{slug}",
                            episode_slug=slug,
                            start_ms=60_000,
                            text=f"take on {topic_id}",
                            quotes=[
                                NS(text=f'"{spk} on {topic_id}"', speaker=None, start_ms=60_000)
                            ],
                        )
                    ],
                )
                for spk, slug in rows
            ]
        )

    def fake_engagement(
        data_dir: Path, *, user_id: str | None = None, since_week: str | None = None, **kw: Any
    ) -> dict[str, Any]:
        """``engagement`` is slug -> {week: distinct listeners that week}; the window is applied
        like the real series (weeks >= since_week), listeners summed (the fakes never overlap)."""
        seen["since_week"] = since_week
        out = []
        for slug, wc in engagement.items():
            inside = {w: n for w, n in wc.items() if since_week is None or w >= since_week}
            if inside:
                n = sum(inside.values())
                out.append({"kind": "episode", "entity_id": slug, "listeners": n, "total": n * 3})
        return {"entities": out}

    monkeypatch.setattr(te, "trending", fake_trending)
    monkeypatch.setattr(te, "build_topic_perspectives", fake_perspectives)
    monkeypatch.setattr(te, "engagement_series", fake_engagement)
    monkeypatch.setattr(
        te, "resolve_slug", lambda root, slug: NS(has_gi=True, gi_relative_path=f"{slug}.gi.json")
    )
    monkeypatch.setattr(te, "load_json_artifact", lambda root, rel: {"rel": rel})
    monkeypatch.setattr(
        te,
        "insights_from_gi",
        lambda art, limit=None, unattributed_fallback=False: [
            NS(
                id="insight:1",
                text="insight",
                quotes=[NS(text=f"quote from {art['rel']}", speaker="Host", start_ms=5_000)],
            )
        ],
    )
    return seen


def test_interleaves_rising_and_engaged_one_card_per_episode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _wire(
        monkeypatch,
        topics=[("topic:ai", "AI"), ("topic:pegs", "Currency pegs")],
        takes={"topic:ai": [("Ann", "ep-ai")], "topic:pegs": [("Bo", "ep-peg")]},
        # ep-ai is ALSO most engaged: it must appear once, as its rising-topic card.
        engagement={"ep-hot": {"2026-W29": 5}, "ep-ai": {"2026-W30": 4}},
    )
    cards = te.trending_episodes(tmp_path, tmp_path, now=_NOW)
    assert [(c.slug, c.why.kind) for c in cards] == [
        ("ep-ai", "rising_topic"),
        ("ep-hot", "most_engaged"),
        ("ep-peg", "rising_topic"),
    ]
    first = cards[0].why
    assert first.subject is not None and first.subject.label == "AI"
    assert first.subject.kind == "topic" and first.subject.id == "topic:ai"
    assert first.quote is not None
    assert (first.quote.speaker, first.quote.start_ms) == ("Ann", 60_000)
    assert first.quote.text == '"Ann on topic:ai"' and first.quote.episode_slug == "ep-ai"
    engaged = cards[1].why
    assert engaged.events == 5 and engaged.quote is not None
    assert engaged.quote.text == "quote from ep-hot.gi.json" and engaged.quote.start_ms == 5_000


def test_most_engaged_counts_distinct_listeners_over_four_weeks_and_drops_a_lone_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _wire(
        monkeypatch,
        topics=[],
        takes={},
        engagement={
            "ep-old": {"2026-W20": 50},  # outside the window
            "ep-once": {"2026-W30": 1},  # one listener: below the floor of 2
            "ep-two": {"2026-W27": 1, "2026-W30": 1},  # 2 inside the window
            "ep-many": {"2026-W28": 3},
        },
    )
    cards = te.trending_episodes(tmp_path, tmp_path, now=_NOW)
    assert [(c.slug, c.why.events) for c in cards] == [("ep-many", 3), ("ep-two", 2)]
    assert seen["since_week"] == "2026-W27"  # the four ISO weeks ending in W30


def test_mine_keeps_both_halves_to_the_listeners_world(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _wire(
        monkeypatch,
        topics=[("topic:ai", "AI")],
        takes={"topic:ai": [("Ann", "ep-not-mine"), ("Cy", "ep-mine")]},
        engagement={"ep-hot": {"2026-W30": 9}, "ep-mine-hot": {"2026-W30": 3}},
    )
    world = {"ep-mine", "ep-mine-hot"}
    cards = te.trending_episodes(
        tmp_path,
        tmp_path,
        scope="mine",
        user_id="u1",
        restrict_entities={"topic:ai"},
        world=world,
        now=_NOW,
    )
    assert {c.slug for c in cards} == {"ep-mine", "ep-mine-hot"}
    # Rising topics are the listener's own ones too.
    assert seen["trending"]["restrict_to"] == {"topic:ai"}
    assert seen["trending"]["scope"] == "mine"


def test_an_episode_without_insights_has_no_quote_and_no_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _wire(monkeypatch, topics=[], takes={}, engagement={"ep-bare": {"2026-W30": 6}})
    monkeypatch.setattr(te, "insights_from_gi", lambda art, limit=None, **kw: [])
    assert te.trending_episodes(tmp_path, tmp_path, now=_NOW) == []


def test_the_insights_own_text_never_stands_in_for_a_quote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ADR-163 §4: an insight's text is an LLM paraphrase. With no quote, the next insight that has
    one leads the card; with none at all, there is no card — not a card quoting the paraphrase."""
    _wire(monkeypatch, topics=[], takes={}, engagement={"ep-a": {"2026-W30": 6}})
    monkeypatch.setattr(
        te,
        "insights_from_gi",
        lambda art, limit=None, **kw: [
            NS(id="insight:paraphrase-only", text="A paraphrase", quotes=[]),
            NS(
                id="insight:quoted",
                text="x",
                quotes=[NS(text="Said it.", speaker=None, start_ms=9)],
            ),
        ],
    )
    [card] = te.trending_episodes(tmp_path, tmp_path, now=_NOW)
    assert card.why.quote is not None
    assert (card.why.quote.text, card.why.quote.insight_id) == ("Said it.", "insight:quoted")
    monkeypatch.setattr(
        te, "insights_from_gi", lambda art, limit=None, **kw: [NS(id="i", text="A", quotes=[])]
    )
    assert te.trending_episodes(tmp_path, tmp_path, now=_NOW) == []


def test_mine_gets_the_labelled_outside_share_from_everyones_cards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ADR-163 §7: under Mine every 5th card is one of Everyone's from outside the listener's
    habits, labelled; Everyone's cards from the listener's own shows or topics never fill it."""
    from podcast_scraper.server.app_reasons import Reason
    from podcast_scraper.server.app_serendipity import Habits

    feeds = {"far-1": "feed:far", "far-2": "feed:far", "theirs": "feed:mine", "topical": "feed:x"}
    monkeypatch.setattr(
        te,
        "resolve_slug",
        lambda root, slug: NS(feed_id=feeds.get(slug, "feed:mine"), kg_relative_path=slug),
    )
    monkeypatch.setattr(
        te, "episode_entity_ids", lambda root, rel: {"topic:rates"} if rel == "topical" else set()
    )
    habits = Habits(shows=frozenset({"feed:mine"}), entities=frozenset({"topic:rates"}))

    def card(slug: str) -> te.TrendingEpisodeCard:
        return te.TrendingEpisodeCard(slug=slug, why=Reason(kind="rising_topic"))

    mine = [card(f"m{i}") for i in range(8)]
    everyone = [card("m0"), card("theirs"), card("topical"), card("far-1"), card("far-2")]
    rail = te.mix_outside_habits(tmp_path, mine, everyone, habits, limit=10)
    assert [c.slug for c in rail] == [
        "m0",
        "m1",
        "m2",
        "m3",
        "far-1",
        "m4",
        "m5",
        "m6",
        "m7",
        "far-2",
    ]
    assert [c.why.outside_habits for c in rail].count(True) == 2
    assert all(c.why.outside_habits for c in rail if c.slug.startswith("far"))


def test_no_habits_means_no_outside_share_and_nothing_labelled_outside(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A listener with no history has no "usual" to step outside of: Everyone's cards must not fill
    their Mine labelled "Outside your usual" (cold start falls back to Everyone, labelled as such).
    """
    from podcast_scraper.server.app_reasons import Reason
    from podcast_scraper.server.app_serendipity import Habits

    monkeypatch.setattr(
        te, "resolve_slug", lambda root, slug: NS(feed_id="feed:x", kg_relative_path=slug)
    )
    monkeypatch.setattr(te, "episode_entity_ids", lambda root, rel: set())
    everyone = [
        te.TrendingEpisodeCard(slug=f"e{i}", why=Reason(kind="rising_topic")) for i in range(6)
    ]
    none = Habits(shows=frozenset(), entities=frozenset())
    assert te.mix_outside_habits(tmp_path, [], everyone, none, limit=6) == []
    mine = [te.TrendingEpisodeCard(slug="m0", why=Reason(kind="rising_topic"))]
    assert [c.slug for c in te.mix_outside_habits(tmp_path, mine, everyone, none, limit=6)] == [
        "m0"
    ]

"""Trending episodes (operator 2026-10-10): rising-topic cards interleaved with most-engaged ones.

The data sources (topic momentum, speaker takes, the engagement series, GI insights) are replaced
with small fakes: what is under test is the SELECTION — interleaving, one card per episode, the
engagement window and floor, and "mine" keeping to the listener's world (ADR-162).
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
                            episode_slug=slug,
                            start_ms=60_000,
                            text=f"take on {topic_id}",
                            quotes=[NS(text=f'"{spk} on {topic_id}"')],
                        )
                    ],
                )
                for spk, slug in rows
            ]
        )

    def fake_engagement(data_dir: Path, *, user_id: str | None = None) -> dict[str, Any]:
        return {
            "entities": [
                {"kind": "episode", "entity_id": slug, "weekly_counts": wc}
                for slug, wc in engagement.items()
            ]
        }

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
        lambda art, limit=None: [
            NS(
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
    assert [(c.slug, c.reason) for c in cards] == [
        ("ep-ai", "rising_topic"),
        ("ep-hot", "most_engaged"),
        ("ep-peg", "rising_topic"),
    ]
    first = cards[0]
    assert first.topic_label == "AI" and first.speaker == "Ann" and first.start_ms == 60_000
    assert first.quote == '"Ann on topic:ai"'
    assert cards[1].events == 5 and cards[1].quote == "quote from ep-hot.gi.json"


def test_most_engaged_counts_four_weeks_and_drops_one_off_activity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _wire(
        monkeypatch,
        topics=[],
        takes={},
        engagement={
            "ep-old": {"2026-W20": 50},  # outside the window
            "ep-once": {"2026-W30": 1},  # below the floor
            "ep-two": {"2026-W27": 1, "2026-W30": 1},  # 2 inside the window
            "ep-many": {"2026-W28": 3},
        },
    )
    cards = te.trending_episodes(tmp_path, tmp_path, now=_NOW)
    assert [(c.slug, c.events) for c in cards] == [("ep-many", 3), ("ep-two", 2)]


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
    monkeypatch.setattr(te, "insights_from_gi", lambda art, limit=None: [])
    assert te.trending_episodes(tmp_path, tmp_path, now=_NOW) == []


def test_long_quotes_are_cut_for_the_card() -> None:
    long = "word " * 200
    cut = te._cut(long)
    assert len(cut) == te.QUOTE_MAX_CHARS and cut.endswith("…")

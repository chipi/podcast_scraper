"""Unit tests for spaced-resurfacing selection + interest derivation (P3 #1123)."""

from __future__ import annotations

from podcast_scraper.server.app_resurfacing import (
    DAY,
    derive_interest_signals,
    LADDER_SECONDS,
    reflection_prompt,
    REFLECTION_PROMPTS,
    select_due,
)

NOW = 1_000_000_000


def _hl(hid: str, created_at: int) -> dict:
    return {"id": hid, "created_at": created_at, "kind": "moment"}


def test_due_after_first_interval_when_never_surfaced() -> None:
    fresh = _hl("h1", NOW - DAY)  # 1 day old < 2-day first step → not due
    due = _hl("h2", NOW - 3 * DAY)  # 3 days old ≥ 2-day step → due
    got = select_due([fresh, due], {}, NOW)
    assert [h["id"] for h in got] == ["h2"]


def test_surface_count_lengthens_the_interval() -> None:
    h = _hl("h1", NOW - 10 * DAY)
    # surfaced once 3 days ago → next step is 1 week (604800s); 3 days < 1 week → not due
    state = {"h1": {"count": 1, "last_surfaced": NOW - 3 * DAY}}
    assert select_due([h], state, NOW) == []
    # surfaced once 8 days ago → 8 days ≥ 1 week → due
    state = {"h1": {"count": 1, "last_surfaced": NOW - 8 * DAY}}
    assert [h["id"] for h in select_due([h], state, NOW)] == ["h1"]


def test_most_overdue_first() -> None:
    a = _hl("a", NOW - 5 * DAY)
    b = _hl("b", NOW - 30 * DAY)  # far more overdue
    assert [h["id"] for h in select_due([a, b], {}, NOW)] == ["b", "a"]


def test_paused_returns_nothing() -> None:
    h = _hl("h1", NOW - 100 * DAY)
    assert select_due([h], {}, NOW, paused=True) == []


def test_skips_malformed_highlights() -> None:
    assert select_due([{"id": "", "created_at": NOW}, {"id": "x"}], {}, NOW) == []


def test_ladder_caps_at_last_step() -> None:
    h = _hl("h1", NOW - 200 * DAY)
    state = {"h1": {"count": 99, "last_surfaced": NOW - 100 * DAY}}  # count beyond ladder
    # last step is 90 days; 100 days ≥ 90 → due (no IndexError)
    assert [h["id"] for h in select_due([h], state, NOW)] == ["h1"]
    assert LADDER_SECONDS[-1] == 90 * DAY


def test_reflection_prompt_is_stable() -> None:
    assert reflection_prompt("h1") == reflection_prompt("h1")
    assert reflection_prompt("h1") in REFLECTION_PROMPTS


def test_derive_interest_signals_ranks_by_frequency() -> None:
    entities = [
        ("person", "p:jane", "Jane"),
        ("person", "p:jane", "Jane"),  # heard in 2 episodes
        ("topic", "t:ai", "AI"),
        ("person", "p:bob", "Bob"),
        ("org", "o:acme", "Acme"),  # non person/topic → dropped
        ("topic", "", "blank"),  # blank id → dropped
    ]
    got = derive_interest_signals(entities)
    assert got[0] == {"token": "person:p:jane", "kind": "person", "label": "Jane", "count": 2}
    tokens = {g["token"] for g in got}
    assert tokens == {"person:p:jane", "topic:t:ai", "person:p:bob"}


def test_derive_interest_signals_min_count() -> None:
    entities = [("topic", "t:ai", "AI"), ("topic", "t:ml", "ML"), ("topic", "t:ai", "AI")]
    got = derive_interest_signals(entities, min_count=2)
    assert [g["token"] for g in got] == ["topic:t:ai"]

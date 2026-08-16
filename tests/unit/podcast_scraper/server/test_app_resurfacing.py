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


# --- 2026-08-16: the token must be usable, not merely well-shaped -------------------------------
#
# The two tests above pass ids like "p:jane" / "t:ai". No caller produces that shape: both
# (routes/app_consolidation.py and routes/app_corpus.py) hand over ids straight from
# entities_from_kg, which already carry their "person:" / "topic:" prefix. Prepending `kind`
# unconditionally therefore emitted "topic:topic:systems-thinking" and "person:person:sam" against
# the real corpus — tokens outside the id space the ranker matches and POST /interests/{token}
# stores, so acting on a derived interest would have silently done nothing.


class TestDerivedTokensAreUsableInterestTokens:
    def test_real_kg_ids_are_not_double_prefixed(self) -> None:
        entities = [
            ("topic", "topic:systems-thinking", "systems thinking"),
            ("topic", "topic:systems-thinking", "systems thinking"),
            ("person", "person:sam", "Sam"),
        ]
        tokens = [g["token"] for g in derive_interest_signals(entities)]
        assert tokens == ["topic:systems-thinking", "person:sam"], tokens
        assert not [t for t in tokens if t.startswith(("topic:topic:", "person:person:"))]

    def test_token_matches_the_id_space_the_ranker_uses(self) -> None:
        """A derived token must be exactly the entity id the ranker compares against."""
        kg_ids = ["topic:risk-management", "person:dr-elena-fischer"]
        entities = [
            ("topic", kg_ids[0], "risk management"),
            ("person", kg_ids[1], "Dr. Elena Fischer"),
        ]
        assert {g["token"] for g in derive_interest_signals(entities)} == set(kg_ids)

    def test_unprefixed_ids_still_get_their_kind(self) -> None:
        """Back-compat: an id without the prefix is still namespaced, as before."""
        got = derive_interest_signals([("topic", "ai", "AI"), ("person", "jane", "Jane")])
        assert {g["token"] for g in got} == {"topic:ai", "person:jane"}

    def test_kind_and_label_are_unaffected(self) -> None:
        got = derive_interest_signals([("topic", "topic:long-form", "long form")])
        assert got[0]["kind"] == "topic"
        assert got[0]["label"] == "long form"
        assert got[0]["count"] == 1

"""Unit tests for spaced-resurfacing selection + interest derivation (P3 #1123)."""

from __future__ import annotations

from typing import Any

from podcast_scraper.server.app_resurfacing import (
    DAY,
    LADDER_SECONDS,
    reflection_prompt,
    REFLECTION_PROMPTS,
    select_due,
)

NOW = 1_000_000_000


def _hl(hid: str, created_at: int, episode_slug: str = "ep") -> dict:
    return {
        "id": hid,
        "created_at": created_at,
        "kind": "moment",
        "episode_slug": episode_slug,
    }


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


def test_most_recently_engaged_episode_first_not_most_overdue() -> None:
    """The ordering that replaced most-overdue-first (operator 2026-09-18).

    Overdue-ness sounded right and measured worst: an unreviewed capture never moves `last_seen`,
    so it grows more overdue for ever AND sits on the 90-day rung, returning fast and re-occupying
    the top. The surface recirculated its oldest items instead of draining. Simulated over a year,
    coverage went 44% -> 71% and median age at review 118 days -> 4 days.
    """
    old = _hl("old", NOW - 30 * DAY, "ep-old")  # far MORE overdue
    new_ = _hl("new", NOW - 5 * DAY, "ep-new")
    assert [h["id"] for h in select_due([old, new_], {}, NOW)] == ["new", "old"]


def test_replaying_an_episode_lifts_its_captures() -> None:
    """`listened_at` is playback[slug].updated_at — replaying is renewed interest.

    Without it, an old episode you deliberately went back to stays buried under everything newer.
    """
    old = _hl("old", NOW - 300 * DAY, "ep-old")
    new_ = _hl("new", NOW - 5 * DAY, "ep-new")
    plain = [h["id"] for h in select_due([old, new_], {}, NOW)]
    assert plain == ["new", "old"]
    lifted = [
        h["id"] for h in select_due([old, new_], {}, NOW, listened_at={"ep-old": NOW - 1 * DAY})
    ]
    assert lifted == ["old", "new"], "a re-listened episode did not come back up"


def test_an_episodes_captures_stay_together() -> None:
    """Captures are made while listening, so an episode's captures are one session's thinking."""
    rows = [
        _hl("a1", NOW - 20 * DAY, "ep-a"),
        _hl("b1", NOW - 10 * DAY, "ep-b"),
        _hl("a2", NOW - 19 * DAY, "ep-a"),
        _hl("b2", NOW - 9 * DAY, "ep-b"),
    ]
    got = [h["episode_slug"] for h in select_due(rows, {}, NOW)]
    assert got == ["ep-b", "ep-b", "ep-a", "ep-a"], "an episode's captures were interleaved"


def test_publish_date_is_not_what_orders_this() -> None:
    """Publishing is not listening.

    Keying on publish date collapsed coverage to 49% for a half-back-catalogue listener: an old
    episode played today sank to the bottom of the queue. Nothing here reads a publish date, and
    an episode carrying one changes nothing.
    """
    old_show = _hl("x", NOW - 2 * DAY, "ep-2019")
    old_show["publish_date"] = "2019-01-01"
    fresh = _hl("y", NOW - 3 * DAY, "ep-2026")
    fresh["publish_date"] = "2026-09-01"
    # Ordered by CAPTURE recency, so the 2019 show captured more recently leads.
    assert [h["id"] for h in select_due([fresh, old_show], {}, NOW)] == ["x", "y"]


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


# --- select_due must survive the state it READS, not just the state it writes (#39) ------------
#
# `state` is a per-user JSON file on disk: hand-editable, possibly half-written, possibly left by
# an older build. `mark_surfaced` clamps what it writes, but the clamp on the WRITE side does
# nothing for a value that is already there.


def test_a_non_numeric_count_does_not_take_down_the_route() -> None:
    """A string count raised ValueError out of select_due — a 500 on /resurfacing AND /your-week.

    Two whole surfaces down because one key in one user's file was the wrong type.
    """
    hl = _hl("h1", NOW - 3 * DAY)
    got = select_due([hl], {"h1": {"count": "many", "last_surfaced": NOW - 3 * DAY}}, NOW)
    assert [h["id"] for h in got] == ["h1"]  # treated as never surfaced, still due


def test_a_negative_count_does_not_index_the_ladder_backwards() -> None:
    """The quiet one, and the worse one. `ladder[-1]` is the 90-DAY step, so a negative count
    silently scheduled a brand-new highlight on the longest rung — no error, no log, just a
    capture that stops resurfacing for three months."""
    hl = _hl("h1", NOW - 3 * DAY)  # 3 days old: due on the 2-day step, NOT on the 90-day one
    got = select_due([hl], {"h1": {"count": -1, "last_surfaced": NOW - 3 * DAY}}, NOW)
    assert [h["id"] for h in got] == [
        "h1"
    ], "a negative count indexed the ladder from the end and picked the 90-day interval"


def test_a_non_numeric_last_surfaced_falls_back_to_created_at() -> None:
    hl = _hl("h1", NOW - 3 * DAY)
    got = select_due([hl], {"h1": {"count": 0, "last_surfaced": "yesterday"}}, NOW)
    assert [h["id"] for h in got] == ["h1"]


def test_a_non_mapping_state_entry_is_ignored_rather_than_fatal() -> None:
    hl = _hl("h1", NOW - 3 * DAY)
    corrupt: dict[str, Any] = {"h1": "corrupt"}  # deliberately the wrong shape
    assert [h["id"] for h in select_due([hl], corrupt, NOW)] == ["h1"]


def test_an_orphan_state_entry_changes_nothing() -> None:
    """Selection iterates HIGHLIGHTS, so a schedule entry for a deleted capture is inert.

    That is precisely why the missing delete-cascade was growth and not a wrong answer — worth
    pinning, because a future rewrite that iterated `state` instead would resurrect dead captures.
    """
    hl = _hl("h1", NOW - 3 * DAY)
    state = {"h1": {"count": 0, "last_surfaced": NOW - 3 * DAY}, "h_deleted": {"count": 0}}
    assert [h["id"] for h in select_due([hl], state, NOW)] == ["h1"]


# --- retiring: kept, but never asked about again (operator 2026-09-18) --------------------------


def test_a_retired_highlight_is_never_due() -> None:
    """The ladder's only exit.

    Reviewing tops out at the 90-day rung and repeats for ever; ignoring leaves `last_seen` at the
    capture date, so the item is permanently overdue AND sorts first. Both loop. Retiring is how a
    capture leaves this surface without being deleted.
    """
    hl = _hl("h1", NOW - 400 * DAY)  # wildly overdue by any rung
    assert [h["id"] for h in select_due([hl], {"h1": {"retired": True}}, NOW)] == []


def test_retiring_beats_every_other_signal() -> None:
    """Checked before the date maths, so no amount of overdue-ness resurrects it."""
    hl = _hl("h1", NOW - 10_000 * DAY)
    state: dict[str, Any] = {
        "h1": {"retired": True, "count": 0, "last_surfaced": NOW - 9_000 * DAY}
    }
    assert select_due([hl], state, NOW) == []


def test_an_unretired_highlight_is_due_again() -> None:
    """`retired` is removed rather than written False, so absence is the only resurfacing state."""
    hl = _hl("h1", NOW - 3 * DAY)
    assert [h["id"] for h in select_due([hl], {"h1": {"count": 0}}, NOW)] == ["h1"]


def test_retiring_one_highlight_leaves_the_others_due() -> None:
    a, b = _hl("a", NOW - 3 * DAY), _hl("b", NOW - 3 * DAY)
    state: dict[str, Any] = {"a": {"retired": True}}
    assert [h["id"] for h in select_due([a, b], state, NOW)] == ["b"]


def test_a_non_boolean_retired_value_is_honoured_when_truthy() -> None:
    """The file can be hand-edited; `retired: 1` means what a human meant by it."""
    hl = _hl("h1", NOW - 3 * DAY)
    assert select_due([hl], {"h1": {"retired": 1}}, NOW) == []
    assert [h["id"] for h in select_due([hl], {"h1": {"retired": 0}}, NOW)] == ["h1"]

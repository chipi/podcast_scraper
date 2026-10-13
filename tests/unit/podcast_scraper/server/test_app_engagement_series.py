"""Unit tests for the RFC-103 Phase 2 engagement aggregator.

Seeds per-user engagement via the real writers (so on-disk formats are exercised) and asserts the
aggregator buckets every kind's events into per-entity weekly counts — corpus-wide and per-user.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from podcast_scraper.server import app_user_state
from podcast_scraper.server.app_engagement_series import engagement_series

pytestmark = [pytest.mark.unit]


def _ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp())


def _wk(iso: str) -> str:
    c = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).isocalendar()
    return f"{c.year:04d}-W{c.week:02d}"


def _by_entity(data: dict) -> dict[tuple[str, str], dict]:
    return {(e["kind"], e["entity_id"]): e for e in data["entities"]}


def test_engagement_series_covers_every_kind(tmp_path: Path) -> None:
    d = _wk("2026-01-19")  # a single ISO week for all the events below
    t = "2026-01-19T00:00:00"
    app_user_state.append_listen_event(tmp_path, "u1", "ep-index-investing", "p05", _ts(t))
    app_user_state.add_subscription(tmp_path, "u1", {"feed_id": "p05", "added_at": _ts(t)})
    app_user_state.add_favorite(
        tmp_path, "u1", {"kind": "insight", "ref": "ep#i1", "added_at": _ts(t)}
    )
    app_user_state.add_favorite(
        tmp_path, "u1", {"kind": "topic", "ref": "topic:ai", "added_at": _ts(t)}
    )
    app_user_state.record_interest_follow(tmp_path, "u1", "thc:managing-risk", _ts(t))
    app_user_state.record_interest_follow(tmp_path, "u1", "person:jane", _ts(t))

    e = _by_entity(engagement_series(tmp_path))
    # Weighted per DISTINCT signal (DEFAULT_WEIGHTS): opened 1, subscribed 1, followed 1, saved 2.
    assert e[("episode", "ep-index-investing")]["weekly_counts"] == {d: 1}  # opened
    assert e[("show", "p05")]["total"] == 2  # opened (feed of the listen) + subscribed
    assert e[("insight", "ep#i1")]["total"] == 2  # saved
    assert e[("topic", "topic:ai")]["total"] == 2  # saved
    assert e[("storyline", "thc:managing-risk")]["total"] == 1  # followed
    assert e[("person", "person:jane")]["total"] == 1  # followed
    assert all(row["listeners"] == 1 for row in e.values())


def test_one_listener_counts_once_per_signal_however_often_they_act(tmp_path: Path) -> None:
    """ADR-163 §5: three highlights by one listener are one "captured", not three — a small user
    base cannot inflate an episode by repeating an action."""
    t = _ts("2026-01-19T00:00:00")
    for i in range(3):
        app_user_state.add_highlight(
            tmp_path, "u1", {"id": f"h{i}", "episode_slug": "ep", "kind": "moment", "created_at": t}
        )
    for _ in range(4):
        app_user_state.append_listen_event(tmp_path, "u1", "ep", "p01", t)
    row = _by_entity(engagement_series(tmp_path))[("episode", "ep")]
    assert row["total"] == 3 + 1  # captured once (3) + opened once (1)
    assert row["listeners"] == 1


def test_finishing_and_capturing_weigh_more_than_opening(tmp_path: Path) -> None:
    t = _ts("2026-01-19T00:00:00")
    app_user_state.append_listen_event(tmp_path, "u1", "opened-only", "p01", t)
    app_user_state.append_playback_progress(tmp_path, "u2", "finished", "p01", 95, t)
    app_user_state.append_playback_progress(tmp_path, "u2", "halfway", "p01", 50, t)
    app_user_state.add_note(
        tmp_path,
        "u3",
        {"id": "n1", "target": "episode", "target_id": "noted", "text": "x", "created_at": t},
    )
    e = _by_entity(engagement_series(tmp_path))
    assert e[("episode", "opened-only")]["total"] == 1
    assert e[("episode", "finished")]["total"] == 3
    assert ("episode", "halfway") not in e  # 50% is not finishing
    assert e[("episode", "noted")]["total"] == 3


def test_listeners_are_distinct_within_the_window(tmp_path: Path) -> None:
    for uid, day in (("u1", "2026-01-05"), ("u2", "2026-01-19"), ("u2", "2026-01-26")):
        app_user_state.append_listen_event(tmp_path, uid, "ep", "p01", _ts(f"{day}T00:00:00"))
    everything = _by_entity(engagement_series(tmp_path))[("episode", "ep")]
    assert everything["listeners"] == 2
    recent = _by_entity(engagement_series(tmp_path, since_week=_wk("2026-01-19")))[
        ("episode", "ep")
    ]
    assert recent["listeners"] == 1 and recent["total"] == 2  # u2, two weeks


def test_engagement_series_aggregates_corpus_wide_and_scopes_to_user(tmp_path: Path) -> None:
    for uid in ("u1", "u2"):
        app_user_state.append_listen_event(tmp_path, uid, "ep-a", "p01", _ts("2026-02-02T00:00:00"))
    # corpus-wide: both users' opens sum
    corpus = _by_entity(engagement_series(tmp_path))
    assert corpus[("episode", "ep-a")]["total"] == 2
    # scope=mine (single user): just that user's opens
    mine = _by_entity(engagement_series(tmp_path, user_id="u1"))
    assert mine[("episode", "ep-a")]["total"] == 1


def test_engagement_series_weekly_axis_is_contiguous(tmp_path: Path) -> None:
    app_user_state.append_listen_event(tmp_path, "u1", "ep-a", "p01", _ts("2026-01-05T00:00:00"))
    app_user_state.append_listen_event(tmp_path, "u1", "ep-a", "p01", _ts("2026-02-16T00:00:00"))
    data = engagement_series(tmp_path)
    axis = data["window_weeks"]
    assert axis == sorted(axis)  # contiguous, oldest→newest
    assert _wk("2026-01-05") in axis and _wk("2026-02-16") in axis
    assert len(axis) >= 6  # ~6 weeks between the two events (a real gap, zero-filled at read)


def test_engagement_series_empty_without_users(tmp_path: Path) -> None:
    assert engagement_series(tmp_path) == {"window_weeks": [], "entities": []}

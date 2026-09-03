"""Unit tests for listening analytics aggregation (UXS-014)."""

from __future__ import annotations

from pathlib import Path

from podcast_scraper.server import app_stats, app_user_state as st

DAY = 86_400
# A fixed "now" (2023-11-14 UTC) so streak/series math is deterministic.
NOW = 1_700_000_000


def test_user_stats_scores_and_streak(tmp_path: Path) -> None:
    # alice opens ep1 today + yesterday, ep2 today; two distinct shows; a playback position.
    st.append_listen_event(tmp_path, "alice", "ep1", "feedX", NOW)
    st.append_listen_event(tmp_path, "alice", "ep1", "feedX", NOW - DAY)
    st.append_listen_event(tmp_path, "alice", "ep2", "feedY", NOW)
    st.set_playback(tmp_path, "alice", "ep1", 120.0, NOW)

    s = app_stats.compute_user_stats(tmp_path, "alice", now=NOW)
    assert s["episodes"] == 2
    assert s["shows"] == 2
    assert s["listening_seconds"] == 120.0
    assert s["active_days"] == 2
    assert s["day_streak"] == 2
    assert len(s["daily"]) == app_stats.SERIES_DAYS
    assert s["daily"][-1] == {"date": "2023-11-14", "count": 2}  # two opens today
    assert s["daily"][-2]["count"] == 1  # one open yesterday


def test_user_stats_streak_breaks_on_gap(tmp_path: Path) -> None:
    # Active today and 3 days ago — the streak is just today (the gap breaks the run).
    st.append_listen_event(tmp_path, "u", "ep1", "f", NOW)
    st.append_listen_event(tmp_path, "u", "ep1", "f", NOW - 3 * DAY)
    s = app_stats.compute_user_stats(tmp_path, "u", now=NOW)
    assert s["day_streak"] == 1


def test_user_stats_empty(tmp_path: Path) -> None:
    s = app_stats.compute_user_stats(tmp_path, "nobody", now=NOW)
    assert s["episodes"] == 0 and s["shows"] == 0 and s["day_streak"] == 0
    assert sum(p["count"] for p in s["daily"]) == 0


def test_episode_stats_counts_distinct_listeners(tmp_path: Path) -> None:
    # ep1 opened by five people so it clears the k-anonymity floor; alice twice, so 5 listeners
    # and 6 opens.
    for who in ("alice", "bob", "carol", "dave", "erin"):
        st.append_listen_event(tmp_path, who, "ep1", "f", NOW)
    st.append_listen_event(tmp_path, "alice", "ep1", "f", NOW - DAY)
    st.append_listen_event(tmp_path, "carol", "ep2", "f", NOW)

    s = app_stats.compute_episode_stats(tmp_path, "ep1", now=NOW)
    assert s["listeners"] == 5
    assert s["opens"] == 6
    assert s["daily"][-1] == {"date": "2023-11-14", "count": 5}  # everyone today
    # Alice alone yesterday is BELOW the floor, so that bucket reports 0 (advisor 2.4): clearing
    # the episode-level floor does not make a single-person day safe to publish.
    assert s["daily"][-2]["count"] == 0


def test_a_small_audience_is_withheld_not_reported(tmp_path: Path) -> None:
    """The endpoint is PUBLIC, so an exact small count re-identifies (#1923).

    With a handful of users, "listeners: 1" says that one user listened to this — and the
    catalogue can then be walked to reconstruct their history. Null means "not enough people";
    zero would be a lie, and a rounded number still leaks by changing.
    """
    st.append_listen_event(tmp_path, "alice", "ep1", "f", NOW)
    st.append_listen_event(tmp_path, "bob", "ep1", "f", NOW)

    s = app_stats.compute_episode_stats(tmp_path, "ep1", now=NOW)
    assert s["listeners"] is None
    # The whole shape is withheld: opens and the daily series describe one person's week just as
    # identifiably as the headline does.
    assert s["opens"] is None
    assert s["daily"] == []


def test_episode_stats_unknown_episode_is_withheld(tmp_path: Path) -> None:
    st.append_listen_event(tmp_path, "alice", "ep1", "f", NOW)
    s = app_stats.compute_episode_stats(tmp_path, "ghost", now=NOW)
    # Zero listeners is below the floor, so it reads the same as any other small audience — which
    # is the point: the response must not distinguish "nobody" from "a few".
    assert s["listeners"] is None and s["opens"] is None


def test_user_stats_skips_uncoercible_timestamps(tmp_path: Path) -> None:
    # A non-numeric ts can't become a date → it's dropped from the day buckets/streak math, but
    # the episode/show counts (which key off slug/feed_id) still include the event.
    events_path = tmp_path / "users" / "u" / "listen_events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(
        '{"slug": "epbad", "feed_id": "f", "ts": "not-a-number"}\n', encoding="utf-8"
    )
    s = app_stats.compute_user_stats(tmp_path, "u", now=NOW)
    assert s["episodes"] == 1  # slug still counted
    assert s["active_days"] == 0  # no parseable date
    assert s["day_streak"] == 0
    assert sum(p["count"] for p in s["daily"]) == 0


def test_user_stats_streak_anchors_on_yesterday_when_idle_today(tmp_path: Path) -> None:
    # Last activity was yesterday (nothing today): the streak still counts from yesterday back.
    st.append_listen_event(tmp_path, "u", "ep1", "f", NOW - DAY)
    st.append_listen_event(tmp_path, "u", "ep1", "f", NOW - 2 * DAY)
    s = app_stats.compute_user_stats(tmp_path, "u", now=NOW)
    assert s["day_streak"] == 2  # yesterday + the day before
    assert s["daily"][-1]["count"] == 0  # nothing logged today


def test_a_single_person_day_is_suppressed_even_on_a_popular_episode(tmp_path: Path) -> None:
    """Episode-level k-anonymity does not make a per-DAY count of one safe (advisor 2.4).

    A count of 1 on a known date, on a public endpoint, is the same per-event leak the floor
    exists to stop — one level down.
    """
    for who in ("a", "b", "c", "d", "e", "f"):
        st.append_listen_event(tmp_path, who, "ep1", "f", NOW)
    st.append_listen_event(tmp_path, "a", "ep1", "f", NOW - DAY)

    s = app_stats.compute_episode_stats(tmp_path, "ep1", now=NOW)
    assert s["listeners"] == 6  # the episode itself clears the floor
    assert s["daily"][-1]["count"] == 6  # a crowded day is reported
    assert s["daily"][-2]["count"] == 0  # a one-person day is not
    # The series keeps its length and shape, so a sparkline still renders.
    assert len(s["daily"]) == app_stats.SERIES_DAYS

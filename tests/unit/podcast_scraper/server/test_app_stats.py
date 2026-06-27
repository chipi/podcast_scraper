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
    # ep1 opened by alice (x2) and bob (x1); ep2 only by carol — so ep1 has 2 listeners, 3 opens.
    st.append_listen_event(tmp_path, "alice", "ep1", "f", NOW)
    st.append_listen_event(tmp_path, "alice", "ep1", "f", NOW - DAY)
    st.append_listen_event(tmp_path, "bob", "ep1", "f", NOW)
    st.append_listen_event(tmp_path, "carol", "ep2", "f", NOW)

    s = app_stats.compute_episode_stats(tmp_path, "ep1", now=NOW)
    assert s["listeners"] == 2
    assert s["opens"] == 3
    assert s["daily"][-1] == {"date": "2023-11-14", "count": 2}  # alice + bob today
    assert s["daily"][-2]["count"] == 1  # alice yesterday


def test_episode_stats_unknown_episode_is_zero(tmp_path: Path) -> None:
    st.append_listen_event(tmp_path, "alice", "ep1", "f", NOW)
    s = app_stats.compute_episode_stats(tmp_path, "ghost", now=NOW)
    assert s["listeners"] == 0 and s["opens"] == 0
    assert len(s["daily"]) == app_stats.SERIES_DAYS

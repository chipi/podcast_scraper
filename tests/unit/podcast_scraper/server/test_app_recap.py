"""Windowed listening recaps (#1914).

``build_recap`` is pure aggregation over what Phase 0 records, so the whole of the interesting
behaviour — window edges, the listener's day boundaries, and the honesty fields — is testable with
no HTTP and no corpus.
"""

from pathlib import Path

from podcast_scraper.server import app_user_state as st
from podcast_scraper.server.app_recap import build_recap

UID = "u_test"
# 2026-09-03T10:00:00Z
NOON = 1788429600
DAY = "2026-09-03"


def _listen(tmp_path: Path, slug: str, ts: int) -> None:
    st.append_listen_event(tmp_path, UID, slug, "p06", ts)


def test_an_empty_account_still_answers(tmp_path: Path) -> None:
    # A recap for someone who has listened to nothing must render, not 500.
    recap = build_recap(tmp_path, UID, "week", NOON)
    assert recap["listening_seconds"] == 0.0
    assert recap["days_recorded"] == 0
    assert len(recap["by_day"]) == 7
    assert recap["to_day"] == DAY


def test_it_sums_only_the_days_inside_the_window(tmp_path: Path) -> None:
    st.set_playback(tmp_path, UID, "ep", 0.0, NOON)
    st.set_playback(tmp_path, UID, "ep", 20.0, NOON)  # +20s today
    st.set_playback(tmp_path, UID, "ep2", 0.0, NOON - 3 * 86_400)
    st.set_playback(tmp_path, UID, "ep2", 10.0, NOON - 3 * 86_400)  # +10s, 3 days ago
    st.set_playback(tmp_path, UID, "ep3", 0.0, NOON - 20 * 86_400)
    st.set_playback(tmp_path, UID, "ep3", 15.0, NOON - 20 * 86_400)  # +15s, OUTSIDE a week

    week = build_recap(tmp_path, UID, "week", NOON)
    assert week["listening_seconds"] == 30.0
    month = build_recap(tmp_path, UID, "month", NOON)
    assert month["listening_seconds"] == 45.0


def test_every_day_in_the_window_is_present_so_a_chart_has_no_holes(tmp_path: Path) -> None:
    st.set_playback(tmp_path, UID, "ep", 0.0, NOON)
    st.set_playback(tmp_path, UID, "ep", 20.0, NOON)
    recap = build_recap(tmp_path, UID, "week", NOON)
    assert list(recap["by_day"])[-1] == DAY
    assert sum(1 for v in recap["by_day"].values() if v == 0.0) == 6


def test_starts_come_from_the_listen_log_so_a_relisten_counts(tmp_path: Path) -> None:
    # Playback holds ONE row per episode ever, so it cannot answer "what happened this week" and a
    # second listen would be invisible.
    _listen(tmp_path, "a", NOON)
    _listen(tmp_path, "a", NOON + 60)
    _listen(tmp_path, "b", NOON)
    _listen(tmp_path, "old", NOON - 30 * 86_400)

    recap = build_recap(tmp_path, UID, "week", NOON)
    assert recap["episodes_started"] == 3
    assert recap["distinct_episodes"] == 2
    assert recap["top_episodes"][0] == {"slug": "a", "starts": 2}


def test_finished_counts_only_what_was_finished_in_the_window(tmp_path: Path) -> None:
    # `finished` is a bool with no date, so this is only answerable via finished_at (Phase 0).
    st.set_playback(tmp_path, UID, "old", 10.0, NOON - 40 * 86_400, finished=True)
    st.set_playback(tmp_path, UID, "recent", 10.0, NOON, finished=True)

    assert build_recap(tmp_path, UID, "week", NOON)["episodes_finished"] == 1
    assert build_recap(tmp_path, UID, "year", NOON)["episodes_finished"] == 2


def test_coverage_says_how_much_of_the_window_we_actually_have(tmp_path: Path) -> None:
    """The anti-fabrication field. Recording started recently, so a 'year' covers days."""
    st.set_playback(tmp_path, UID, "ep", 0.0, NOON)
    st.set_playback(tmp_path, UID, "ep", 20.0, NOON)

    year = build_recap(tmp_path, UID, "year", NOON)
    assert year["days_in_window"] == 365
    assert year["days_recorded"] == 1
    assert year["coverage_from"] == DAY
    assert year["first_listened_at"] == NOON


def test_the_window_is_cut_on_the_LISTENER_day(tmp_path: Path) -> None:
    """23:00 in New York is 03:00 UTC tomorrow — it belongs to the listener's today.

    The window and the recording must agree about when a day starts, or a Sunday evening falls
    outside the week it belongs to.
    """
    late_utc = NOON + 17 * 3600  # 2026-09-04T03:00Z
    offset = -4 * 60
    st.set_playback(tmp_path, UID, "ep", 0.0, late_utc, tz_offset_minutes=offset)
    st.set_playback(tmp_path, UID, "ep", 20.0, late_utc, tz_offset_minutes=offset)

    recap = build_recap(tmp_path, UID, "week", late_utc, tz_offset_minutes=offset)
    assert recap["to_day"] == DAY
    assert recap["by_day"][DAY] == 20.0


def test_a_legacy_iso_listen_event_still_buckets(tmp_path: Path) -> None:
    # The log stores ISO-8601 (and older rows epoch); dropping either would silently lose history.
    _listen(tmp_path, "a", NOON)
    rows = st.list_listen_events(tmp_path, UID)
    assert rows and isinstance(rows[0]["ts"], str)
    assert build_recap(tmp_path, UID, "week", NOON)["episodes_started"] == 1

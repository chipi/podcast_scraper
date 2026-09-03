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


# --- the half a play-count cannot give you (#1914 slice 3) ---


def test_a_missing_corpus_still_produces_a_recap(tmp_path: Path) -> None:
    """The degrade rule: totals, the day series and the saved line survive without a corpus.

    Only themes and the strength ranking go empty. A recap must not fail because the corpus is
    briefly unavailable — the same rule capture already follows.
    """
    st.set_playback(tmp_path, UID, "ep", 0.0, NOON)
    st.set_playback(tmp_path, UID, "ep", 20.0, NOON)
    recap = build_recap(tmp_path, UID, "week", NOON, root=tmp_path / "nope")
    assert recap["listening_seconds"] == 20.0
    assert recap["topics"] == [] and recap["people"] == []
    assert recap["top_by_strength"] == []


def test_the_saved_line_is_the_most_substantial_one_in_the_window(tmp_path: Path) -> None:
    # Longest, not newest: there is no quality signal, and "newest" would make the headline depend
    # on when the recap happens to be opened.
    st.add_highlight(
        tmp_path,
        UID,
        {
            "id": "h1",
            "episode_slug": "a",
            "kind": "span",
            "quote_text": "short",
            "created_at": NOON,
        },
    )
    st.add_highlight(
        tmp_path,
        UID,
        {
            "id": "h2",
            "episode_slug": "b",
            "kind": "span",
            "quote_text": "a considerably longer line worth keeping",
            "created_at": NOON + 10,
            "start_ms": 42_000,
        },
    )
    line = build_recap(tmp_path, UID, "week", NOON)["best_line"]
    assert line["quote_text"] == "a considerably longer line worth keeping"
    # Carries its anchor, so the UI can open the episode AT the moment rather than at the start.
    assert line["episode_slug"] == "b" and line["start_ms"] == 42_000


def test_a_line_saved_outside_the_window_is_not_this_window_line(tmp_path: Path) -> None:
    st.add_highlight(
        tmp_path,
        UID,
        {
            "id": "h1",
            "episode_slug": "a",
            "kind": "span",
            "quote_text": "x" * 50,
            "created_at": NOON - 40 * 86_400,
        },
    )
    assert build_recap(tmp_path, UID, "week", NOON)["best_line"] is None
    assert build_recap(tmp_path, UID, "year", NOON)["best_line"] is not None


def test_a_highlight_with_no_quote_is_not_a_line(tmp_path: Path) -> None:
    # A "moment" capture has no text; it is a bookmark, not something the listener kept words from.
    st.add_highlight(
        tmp_path, UID, {"id": "h1", "episode_slug": "a", "kind": "moment", "created_at": NOON}
    )
    assert build_recap(tmp_path, UID, "week", NOON)["best_line"] is None


# --- the exposure log and what changed (#1923) ---


def _expose(tmp_path: Path, slug: str, ts: int, *entities: tuple[str, str, str]) -> None:
    st.append_topic_exposure(tmp_path, UID, slug, list(entities), ts)


TOPIC = ("topic", "topic:indexing", "Index investing")
OTHER = ("topic", "topic:systems", "Systems thinking")
PERSON = ("person", "person:cho", "Daniel Cho")


def test_themes_come_from_the_recorded_log_without_a_corpus(tmp_path: Path) -> None:
    """Recorded, not re-derived: a re-enrichment must not rewrite the listener's history."""
    _expose(tmp_path, "a", NOON, TOPIC, PERSON)
    _expose(tmp_path, "b", NOON, TOPIC)

    recap = build_recap(tmp_path, UID, "week", NOON)  # note: no root
    assert recap["topics"][0]["label"] == "Index investing"
    assert recap["topics"][0]["episodes"] == 2
    assert recap["people"][0]["label"] == "Daniel Cho"


def test_a_topic_counts_once_per_episode_however_often_it_was_written(tmp_path: Path) -> None:
    # A re-listen appends again; the count is "episodes it appeared in", not "rows in the log".
    _expose(tmp_path, "a", NOON, TOPIC)
    _expose(tmp_path, "a", NOON + 60, TOPIC)
    assert build_recap(tmp_path, UID, "week", NOON)["topics"][0]["episodes"] == 1


def test_each_theme_carries_its_change_against_the_previous_window(tmp_path: Path) -> None:
    # Previous week: indexing in two episodes. This week: indexing in one, systems arrives.
    _expose(tmp_path, "old1", NOON - 8 * 86_400, TOPIC)
    _expose(tmp_path, "old2", NOON - 9 * 86_400, TOPIC)
    _expose(tmp_path, "new1", NOON, TOPIC)
    _expose(tmp_path, "new2", NOON, OTHER)

    by_label = {t["label"]: t for t in build_recap(tmp_path, UID, "week", NOON)["topics"]}
    assert by_label["Index investing"]["delta"] == -1
    assert by_label["Index investing"]["is_new"] is False
    # "New" is not the same as "+1": arriving from nothing reads differently from growing.
    assert by_label["Systems thinking"]["delta"] == 1
    assert by_label["Systems thinking"]["is_new"] is True


def test_exposure_is_bucketed_on_the_listeners_day(tmp_path: Path) -> None:
    late_utc = NOON + 17 * 3600  # 03:00Z, which is still "yesterday evening" in New York
    _expose(tmp_path, "a", late_utc, TOPIC)
    recap = build_recap(tmp_path, UID, "week", late_utc, tz_offset_minutes=-4 * 60)
    assert recap["topics"] and recap["topics"][0]["episodes"] == 1


def test_a_corrupt_exposure_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    _expose(tmp_path, "a", NOON, TOPIC)
    path = tmp_path / "users" / UID / "topic_exposure.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
    assert build_recap(tmp_path, UID, "week", NOON)["topics"][0]["episodes"] == 1


# --- "your year so far" (#1914) ---


def test_ytd_runs_from_january_first_to_today(tmp_path: Path) -> None:
    """A complete year cannot be reported before it happens; the year SO FAR can, today."""
    recap = build_recap(tmp_path, UID, "ytd", NOON)
    assert recap["from_day"] == "2026-01-01"
    assert recap["to_day"] == DAY
    # 31+28+31+30+31+30+31+31 = 243 days to 31 Aug, +3 = 246 through 3 September.
    assert recap["days_in_window"] == 246
    assert len(recap["by_day"]) == 246


def test_ytd_on_new_years_day_is_one_day_not_a_year(tmp_path: Path) -> None:
    jan_first = 1798761600  # 2027-01-01T00:00:00Z
    recap = build_recap(tmp_path, UID, "ytd", jan_first)
    assert recap["days_in_window"] == 1
    assert recap["from_day"] == recap["to_day"] == "2027-01-01"


def test_ytd_is_cut_on_the_listeners_new_year(tmp_path: Path) -> None:
    # 2027-01-01T03:00Z is still New Year's EVE in New York — the year boundary is the listener's,
    # which is the whole reason recaps bucket on local days.
    ny_eve_utc = 1798761600 + 3 * 3600
    recap = build_recap(tmp_path, UID, "ytd", ny_eve_utc, tz_offset_minutes=-5 * 60)
    assert recap["to_day"] == "2026-12-31"
    assert recap["from_day"] == "2026-01-01"


def test_ytd_still_reports_how_little_of_it_was_recorded(tmp_path: Path) -> None:
    st.set_playback(tmp_path, UID, "ep", 0.0, NOON)
    st.set_playback(tmp_path, UID, "ep", 20.0, NOON)
    recap = build_recap(tmp_path, UID, "ytd", NOON)
    assert recap["listening_seconds"] == 20.0
    # The number is real and covers one day of 246 — saying so is what makes it honest.
    assert recap["days_recorded"] == 1
    assert recap["coverage_from"] == DAY

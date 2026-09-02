"""Listening time, recorded per day (#1914 Phase 0).

"Hours listened" does not exist yet: ``app_stats`` computes it as ``sum(position_seconds)``, a
lifetime snapshot of furthest-position-reached. It cannot be windowed, does not grow on a
re-listen, and inflates when you seek forward — so a recap led by it would be fabricated.

``accrue_listening`` is PURE, which is the point: the arithmetic that makes the number honest is
tested here with no files, no clock and no HTTP, and the persistence around it is tested once.
"""

from pathlib import Path

from podcast_scraper.server import app_user_state as st
from podcast_scraper.server.app_user_state import MAX_LISTEN_DELTA_SECONDS, accrue_listening

UID = "u_test"
# 2026-09-03T10:00:00Z
NOON = 1788429600
DAY = "2026-09-03"


def test_a_first_save_accrues_nothing() -> None:
    # We know WHERE the listener is, not how they got there. Resuming at 12:00 is not twelve
    # minutes of listening.
    state = accrue_listening({}, "ep", None, 720.0, NOON)
    assert state["days"] == {}
    assert state["first_listened_at"] == NOON


def test_a_normal_save_accrues_the_delta() -> None:
    state = accrue_listening({}, "ep", 100.0, 110.0, NOON)
    assert state["days"] == {DAY: 10.0}


def test_a_forward_seek_is_clamped_to_the_ceiling() -> None:
    # The whole reason sum(position_seconds) is unusable: skipping forward 20 minutes must not
    # book 20 minutes of listening.
    state = accrue_listening({}, "ep", 100.0, 1_300.0, NOON)
    assert state["days"] == {DAY: float(MAX_LISTEN_DELTA_SECONDS)}


def test_a_rewind_accrues_nothing_rather_than_subtracting() -> None:
    # Rewinding IS listening; it just cannot be measured this way. Subtracting it would let
    # someone scrub backwards into negative time.
    state = accrue_listening({}, "ep", 500.0, 100.0, NOON)
    assert state["days"] == {}


def test_deltas_accumulate_within_a_day_and_split_across_days() -> None:
    state = accrue_listening({}, "ep", 0.0, 10.0, NOON)
    state = accrue_listening(state, "ep", 10.0, 20.0, NOON)
    state = accrue_listening(state, "ep", 20.0, 25.0, NOON + 86_400)
    assert state["days"] == {DAY: 20.0, "2026-09-04": 5.0}


def test_first_listened_at_moves_only_backwards() -> None:
    # It is the anchor a recap leans on when there is no account creation date, so a later save
    # must not overwrite it — and an offline event flushed late must be able to pull it earlier.
    state = accrue_listening({}, "ep", 0.0, 5.0, NOON)
    state = accrue_listening(state, "ep", 5.0, 10.0, NOON + 999)
    assert state["first_listened_at"] == NOON
    state = accrue_listening(state, "ep", 0.0, 5.0, NOON - 999)
    assert state["first_listened_at"] == NOON - 999


def test_finished_at_records_when_and_keeps_the_first_time() -> None:
    # `finished` is a bool with no date, so "you finished 14 episodes in March" is unsayable.
    state = accrue_listening({}, "ep", 0.0, 5.0, NOON, finished=True)
    state = accrue_listening(state, "ep", 5.0, 10.0, NOON + 86_400, finished=True)
    assert state["finished_at"] == {"ep": NOON}


def test_an_offline_flush_lands_on_the_day_it_happened(tmp_path: Path) -> None:
    """The collision #1914 names: a week of offline listening flushed on Monday must not spike.

    ``set_playback`` is handed the CLAMPED client timestamp, so the bucket is the day the listener
    was actually there rather than the day their device reconnected.
    """
    st.set_playback(tmp_path, UID, "ep", 0.0, NOON - 3 * 86_400)
    st.set_playback(tmp_path, UID, "ep", 10.0, NOON - 3 * 86_400)
    st.set_playback(tmp_path, UID, "ep", 20.0, NOON)

    days = st.get_listening(tmp_path, UID)["days"]
    assert days == {"2026-08-31": 10.0, DAY: 10.0}


def test_it_persists_through_the_position_save(tmp_path: Path) -> None:
    st.set_playback(tmp_path, UID, "ep", 100.0, NOON)
    st.set_playback(tmp_path, UID, "ep", 112.0, NOON)
    record = st.get_listening(tmp_path, UID)
    assert record["days"] == {DAY: 12.0}
    assert record["first_listened_at"] == NOON


def test_a_broken_record_reads_as_empty_rather_than_raising(tmp_path: Path) -> None:
    (tmp_path / "users" / UID).mkdir(parents=True)
    (tmp_path / "users" / UID / "listening_daily.json").write_text("{not json", encoding="utf-8")
    assert st.get_listening(tmp_path, UID)["days"] == {}


def test_accrual_failure_never_costs_the_resume_point(tmp_path: Path, monkeypatch) -> None:
    # It rides along with the highest-frequency writer in the subsystem; a statistic must never
    # break playback persistence.
    monkeypatch.setattr(st, "get_listening", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    rec = st.set_playback(tmp_path, UID, "ep", 42.0, NOON)
    assert rec["position_seconds"] == 42.0
    stored = st.get_playback(tmp_path, UID, "ep")
    assert stored is not None and stored["position_seconds"] == 42.0

"""A client-supplied timestamp is advisory and must never be trusted verbatim (#1924/#1913).

Offline listening is recorded on the device and flushed on reconnect, so the server has to accept
"this happened earlier" — but a device with a wrong clock, or a hostile one, must not be able to
write events into the distant past (poisoning windowed stats) or the future (parking a record
nothing later can beat).
"""

from pathlib import Path

from podcast_scraper.server import app_user_state
from podcast_scraper.server.app_user_state import (
    CLIENT_TS_MAX_AGE_SECONDS,
    CLIENT_TS_MAX_SKEW_SECONDS,
    clamp_client_ts,
)

NOW = 1_800_000_000


def test_absent_timestamp_means_now() -> None:
    assert clamp_client_ts(None, NOW) == NOW


def test_a_plausible_past_timestamp_is_kept() -> None:
    # The whole point: a listen from three days ago lands on the day it happened.
    three_days_ago = NOW - 3 * 24 * 3600
    assert clamp_client_ts(three_days_ago, NOW) == three_days_ago


def test_a_future_timestamp_is_clamped_to_the_skew_ceiling() -> None:
    # Clamped, not collapsed to `now`: playback.updated_at is what cross-device conflict
    # resolution compares, so the bound must be predictable in both directions.
    assert clamp_client_ts(NOW + 3600, NOW) == NOW + CLIENT_TS_MAX_SKEW_SECONDS


def test_small_forward_skew_is_tolerated() -> None:
    # Device clocks drift; a couple of minutes ahead is not an attack.
    slightly_ahead = NOW + CLIENT_TS_MAX_SKEW_SECONDS - 1
    assert clamp_client_ts(slightly_ahead, NOW) == slightly_ahead


def test_an_ancient_timestamp_is_pulled_to_the_floor() -> None:
    # Clamped rather than dropped: the event is real, only its date is implausible.
    assert clamp_client_ts(NOW - 10 * 365 * 24 * 3600, NOW) == NOW - CLIENT_TS_MAX_AGE_SECONDS


def test_a_redelivered_listen_event_is_not_appended_twice(tmp_path: Path) -> None:
    """A lost 204 makes the client replay with the SAME client_ts; that must not double-count."""
    app_user_state.append_listen_event(tmp_path, "u1", "show/ep", "p05", NOW)
    app_user_state.append_listen_event(tmp_path, "u1", "show/ep", "p05", NOW)
    assert len(app_user_state.list_listen_events(tmp_path, "u1")) == 1


def test_two_genuine_opens_at_different_times_both_record(tmp_path: Path) -> None:
    app_user_state.append_listen_event(tmp_path, "u1", "show/ep", "p05", NOW)
    app_user_state.append_listen_event(tmp_path, "u1", "show/ep", "p05", NOW + 1)
    assert len(app_user_state.list_listen_events(tmp_path, "u1")) == 2


def test_the_same_moment_on_a_different_episode_still_records(tmp_path: Path) -> None:
    app_user_state.append_listen_event(tmp_path, "u1", "show/a", "p05", NOW)
    app_user_state.append_listen_event(tmp_path, "u1", "show/b", "p05", NOW)
    assert len(app_user_state.list_listen_events(tmp_path, "u1")) == 2

"""A client-supplied timestamp is advisory and must never be trusted verbatim (#1924/#1913).

Offline listening is recorded on the device and flushed on reconnect, so the server has to accept
"this happened earlier" — but a device with a wrong clock, or a hostile one, must not be able to
write events into the distant past (poisoning windowed stats) or the future (parking a record
nothing later can beat).
"""

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


def test_a_future_timestamp_falls_back_to_now() -> None:
    assert clamp_client_ts(NOW + 3600, NOW) == NOW


def test_small_forward_skew_is_tolerated() -> None:
    # Device clocks drift; a couple of minutes ahead is not an attack.
    slightly_ahead = NOW + CLIENT_TS_MAX_SKEW_SECONDS - 1
    assert clamp_client_ts(slightly_ahead, NOW) == slightly_ahead


def test_an_ancient_timestamp_is_pulled_to_the_floor() -> None:
    # Clamped rather than dropped: the event is real, only its date is implausible.
    assert clamp_client_ts(NOW - 10 * 365 * 24 * 3600, NOW) == NOW - CLIENT_TS_MAX_AGE_SECONDS

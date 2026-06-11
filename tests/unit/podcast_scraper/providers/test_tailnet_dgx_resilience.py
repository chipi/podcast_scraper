"""Unit tests for the shared DGX resilience primitives (#946 / #954).

Pure-logic coverage: circuit-breaker state machine, the hard wall-clock watchdog,
and the duration-scaled timeout math. No network, no ML deps — ``soundfile`` is
only exercised on its graceful-None path so these stay ``[dev]``-only.
"""

from __future__ import annotations

import time

import pytest

from podcast_scraper.providers.tailnet_dgx import resilience
from podcast_scraper.providers.tailnet_dgx.resilience import (
    CircuitBreaker,
    effective_timeout_sec,
    probe_audio_duration_sec,
    run_with_watchdog,
)


class _FakeClock:
    """Monotonic clock stub so breaker cooldowns are tested without real sleeps."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# --------------------------------------------------------------------------- #
# effective_timeout_sec — pure math                                            #
# --------------------------------------------------------------------------- #
class TestEffectiveTimeout:
    def test_no_duration_returns_base(self):
        assert effective_timeout_sec(180.0, 6.0, None) == 180.0

    def test_zero_per_minute_disables_scaling(self):
        assert effective_timeout_sec(180.0, 0.0, 3600.0) == 180.0

    def test_scales_with_audio_minutes(self):
        # 45 min audio * 6 s/min = 270 added to a 180 base.
        assert effective_timeout_sec(180.0, 6.0, 45 * 60) == pytest.approx(450.0)

    def test_nonpositive_duration_ignored(self):
        assert effective_timeout_sec(120.0, 10.0, 0) == 120.0
        assert effective_timeout_sec(120.0, 10.0, -5) == 120.0


# --------------------------------------------------------------------------- #
# probe_audio_duration_sec — graceful None (no ML dep in unit tier)            #
# --------------------------------------------------------------------------- #
class TestProbeDuration:
    def test_missing_file_returns_none(self):
        assert probe_audio_duration_sec("/no/such/file.wav") is None

    def test_non_audio_bytes_returns_none(self, tmp_path):
        f = tmp_path / "not-audio.bin"
        f.write_bytes(b"deadbeef" * 4)
        assert probe_audio_duration_sec(str(f)) is None

    def test_soundfile_absent_returns_none(self, monkeypatch):
        # Simulate ``soundfile`` not installed (the [dev]-only CI case).
        import builtins

        real_import = builtins.__import__

        def _fail(name, *a, **k):
            if name == "soundfile":
                raise ImportError("no soundfile in [dev]")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _fail)
        assert probe_audio_duration_sec("/whatever.wav") is None


# --------------------------------------------------------------------------- #
# run_with_watchdog — hard wall-clock deadline                                 #
# --------------------------------------------------------------------------- #
class TestWatchdog:
    def test_returns_fast_result(self):
        assert run_with_watchdog(lambda: 42, 5.0, label="t") == 42

    def test_propagates_exception(self):
        def boom():
            raise ValueError("kaboom")

        with pytest.raises(ValueError, match="kaboom"):
            run_with_watchdog(boom, 5.0, label="t")

    def test_raises_timeout_when_overrunning(self):
        def hang():
            time.sleep(1.0)  # daemon worker is abandoned; doesn't block the test
            return "never"

        started = time.monotonic()
        with pytest.raises(TimeoutError, match="hard wall-clock deadline"):
            run_with_watchdog(hang, 0.1, label="slow-call")
        # Bailed at ~0.1s, did NOT wait for the 1s sleep.
        assert time.monotonic() - started < 0.8


# --------------------------------------------------------------------------- #
# CircuitBreaker — state machine                                               #
# --------------------------------------------------------------------------- #
class TestCircuitBreaker:
    def test_closed_allows(self):
        cb = CircuitBreaker(failure_threshold=2, window_sec=60, cooldown_sec=60)
        assert cb.allow() is True
        assert cb.state == "closed"

    def test_hard_failure_trips_immediately(self):
        cb = CircuitBreaker(failure_threshold=5, window_sec=60, cooldown_sec=60)
        cb.record_failure(hard=True)
        assert cb.state == "open"
        assert cb.allow() is False  # within cooldown

    def test_threshold_trips_after_n_soft_failures(self):
        cb = CircuitBreaker(failure_threshold=2, window_sec=60, cooldown_sec=60)
        cb.record_failure()
        assert cb.state == "closed"  # one is not enough
        cb.record_failure()
        assert cb.state == "open"

    def test_cooldown_then_half_open_probe(self, monkeypatch):
        clock = _FakeClock()
        monkeypatch.setattr(resilience.time, "monotonic", clock)
        cb = CircuitBreaker(failure_threshold=1, window_sec=60, cooldown_sec=60)
        cb.record_failure(hard=True)
        assert cb.allow() is False  # still cooling down
        clock.advance(61)
        assert cb.allow() is True  # cooldown elapsed → one probe allowed
        assert cb.state == "half_open"

    def test_half_open_success_closes(self, monkeypatch):
        clock = _FakeClock()
        monkeypatch.setattr(resilience.time, "monotonic", clock)
        cb = CircuitBreaker(failure_threshold=1, window_sec=60, cooldown_sec=60)
        cb.record_failure(hard=True)
        clock.advance(61)
        cb.allow()  # → half_open
        cb.record_success()
        assert cb.state == "closed"
        assert cb.allow() is True

    def test_half_open_failure_reopens(self, monkeypatch):
        clock = _FakeClock()
        monkeypatch.setattr(resilience.time, "monotonic", clock)
        cb = CircuitBreaker(failure_threshold=5, window_sec=60, cooldown_sec=60)
        cb.record_failure(hard=True)
        clock.advance(61)
        cb.allow()  # → half_open
        cb.record_failure()  # probe failed → straight back to open (even soft)
        assert cb.state == "open"
        assert cb.allow() is False

    def test_success_clears_failure_window(self):
        cb = CircuitBreaker(failure_threshold=2, window_sec=60, cooldown_sec=60)
        cb.record_failure()
        cb.record_success()  # clears the single accrued failure
        cb.record_failure()
        assert cb.state == "closed"  # only one failure since the success

    def test_reset(self):
        cb = CircuitBreaker(failure_threshold=1, window_sec=60, cooldown_sec=60)
        cb.record_failure(hard=True)
        assert cb.state == "open"
        cb.reset()
        assert cb.state == "closed"
        assert cb.allow() is True

    def test_name_is_stored(self):
        cb = CircuitBreaker(1, 60, 60, name="dgx-whisper")
        assert cb._name == "dgx-whisper"


def test_timeout_like_includes_builtin_timeouterror():
    assert TimeoutError in resilience.TimeoutLike


# --------------------------------------------------------------------------- #
# dgx_http_client / keepalive_socket_options — #956 transport hardening        #
# --------------------------------------------------------------------------- #
class TestDgxHttpClient:
    def test_keepalive_options_enable_so_keepalive(self):
        import socket

        opts = resilience.keepalive_socket_options()
        # SO_KEEPALIVE on is the one option present on every platform.
        assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in opts

    def test_keepalive_options_set_an_idle_probe_time(self):
        import socket

        opts = resilience.keepalive_socket_options(idle_sec=30)
        # Either the Linux (TCP_KEEPIDLE) or macOS (TCP_KEEPALIVE) idle constant,
        # whichever this platform exposes, must carry the idle seconds.
        idle_names = [n for n in ("TCP_KEEPIDLE", "TCP_KEEPALIVE") if hasattr(socket, n)]
        assert idle_names, "platform exposes no TCP idle-keepalive constant"
        idle_consts = {getattr(socket, n) for n in idle_names}
        assert any(o in idle_consts and v == 30 for (_lvl, o, v) in opts)

    def test_keepalive_options_are_all_int_triples(self):
        opts = resilience.keepalive_socket_options()
        assert opts and all(
            isinstance(t, tuple) and len(t) == 3 and all(isinstance(x, int) for x in t)
            for t in opts
        )

    def test_client_sets_connection_close_and_is_closeable(self):
        client = resilience.dgx_http_client(30.0)
        try:
            assert client.headers.get("connection") == "close"
        finally:
            client.close()

    def test_client_merges_extra_headers(self):
        client = resilience.dgx_http_client(30.0, headers={"X-Test": "1"})
        try:
            assert client.headers.get("connection") == "close"
            assert client.headers.get("x-test") == "1"
        finally:
            client.close()

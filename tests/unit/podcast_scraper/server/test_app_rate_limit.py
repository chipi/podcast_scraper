"""Unit tests for the in-process per-principal rate limiter (RFC-112 T-13)."""

from __future__ import annotations

import pytest

from podcast_scraper.server import app_rate_limit as rl

pytestmark = pytest.mark.unit


def setup_function() -> None:
    rl.reset()


def test_allows_up_to_limit_then_denies() -> None:
    allowed = [rl.allow("k", limit=3, window_s=60.0) for _ in range(5)]
    assert allowed == [True, True, True, False, False]


def test_keys_are_independent() -> None:
    assert rl.allow("a", limit=1, window_s=60.0) is True
    assert rl.allow("a", limit=1, window_s=60.0) is False
    # a different principal has its own budget
    assert rl.allow("b", limit=1, window_s=60.0) is True


def test_key_flood_evicts_oldest_not_live_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    # H3: a flood of distinct (spoofable) keys must evict only the OLDEST — it must NOT wipe a
    # still-active key's counter (the old clear-all reset attack). We keep "victim" recently-touched
    # throughout the flood, so LRU never evicts it; its limit must stay enforced (never reset).
    monkeypatch.setattr(rl, "_MAX_KEYS", 5)
    assert rl.allow("victim", limit=1, window_s=60.0) is True
    assert rl.allow("victim", limit=1, window_s=60.0) is False  # victim is now at its limit
    for i in range(30):  # flood distinct keys far past _MAX_KEYS
        rl.allow(f"flood-{i}", limit=1, window_s=60.0)
        # re-touch the victim so it stays recent; it must remain AT its limit, never reset
        assert rl.allow("victim", limit=1, window_s=60.0) is False
    assert len(rl._HITS) <= 5  # table is bounded, not grown unbounded — and never cleared wholesale


def test_window_expiry_frees_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"t": 1000.0}
    monkeypatch.setattr(rl.time, "monotonic", lambda: clock["t"])
    assert rl.allow("k", limit=1, window_s=10.0) is True
    assert rl.allow("k", limit=1, window_s=10.0) is False
    clock["t"] += 11.0  # advance past the window
    assert rl.allow("k", limit=1, window_s=10.0) is True

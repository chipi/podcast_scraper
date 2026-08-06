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


def test_window_expiry_frees_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"t": 1000.0}
    monkeypatch.setattr(rl.time, "monotonic", lambda: clock["t"])
    assert rl.allow("k", limit=1, window_s=10.0) is True
    assert rl.allow("k", limit=1, window_s=10.0) is False
    clock["t"] += 11.0  # advance past the window
    assert rl.allow("k", limit=1, window_s=10.0) is True

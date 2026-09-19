"""A small structured call must not inherit the summarization budget (#1323).

MEASURED. On the 2026-09-19 harness (profile `dev_dgx_full`, detector `vllm`, NVFP4/Qwen3-30B on
`dgx-llm-1:8003`) a speaker-detection request is ~870 prompt tokens and ~30-60 completion tokens,
and answers in **0.1 s**. It was being allowed `summarization_timeout / 3` = **400 s**.

That is not academic. A full thread dump taken while a feed was wedged shows the process blocked
in `httpcore ... _receive_response_headers` inside `openai/_base_client._send_request` — a plain
socket read waiting for headers that never arrive — while `/v1/models` and a real chat completion
against the SAME server answered in 6 ms and 116 ms. Every feed in that run wedged this way after
its last episode; across 48 feeds that is hours of dead waiting, and it is the real content of
issue #1323 (which had attributed the stall to the HF evidence backend — the dump shows that
component is not involved at all).
"""

from typing import Any

import pytest

from podcast_scraper.utils.timeout_config import (
    get_single_chat_call_timeout,
    get_small_chat_call_timeout,
    SMALL_CALL_TIMEOUT_SEC,
)


def _Cfg(summarization_timeout: float = 1200.0) -> Any:
    """The only attribute these helpers read is `summarization_timeout` (via getattr), so a
    stub is sufficient and keeps the test independent of Config's constructor."""

    class _C:
        pass

    c = _C()
    c.summarization_timeout = summarization_timeout  # type: ignore[attr-defined]
    return c


def test_a_small_call_is_not_sized_by_the_summarization_deadline() -> None:
    """The defect, stated directly: 400 s for a 0.1 s call."""
    cfg = _Cfg(1200.0)
    big = get_single_chat_call_timeout(cfg)
    small = get_small_chat_call_timeout(cfg)
    assert big == pytest.approx(400.0), "the general per-call bound is a third of the deadline"
    assert small == SMALL_CALL_TIMEOUT_SEC == 60.0
    assert small < big, "a small structured call must be bounded far tighter than a summary"


def test_it_still_clears_a_healthy_call_by_a_wide_margin() -> None:
    """60 s against a measured 0.1 s is ~600x headroom, so this cannot fire on a slow model —
    the failure mode a timeout must never introduce."""
    measured_healthy_seconds = 0.116  # a real chat completion against the production server
    assert get_small_chat_call_timeout(_Cfg()) / measured_healthy_seconds > 400


def test_a_shorter_operator_deadline_still_wins() -> None:
    """Never LONGER than the general bound. An operator who configures an aggressive deadline
    must not find small calls quietly exempted from it."""
    cfg = _Cfg(60.0)  # -> general bound floors at MIN_SINGLE_CALL_TIMEOUT_SEC = 120 s
    assert get_small_chat_call_timeout(cfg) <= get_single_chat_call_timeout(cfg)
    cfg_tiny = _Cfg(1.0)
    assert get_small_chat_call_timeout(cfg_tiny) <= get_single_chat_call_timeout(cfg_tiny)


def test_speaker_detection_passes_the_small_bound() -> None:
    """Pin the wiring, not just the helper: the detector's chat call must carry it explicitly,
    or it silently falls back to `_chat_create`'s 400 s default via setdefault."""
    import inspect

    from podcast_scraper.providers.openai import openai_provider

    src = inspect.getsource(openai_provider.OpenAICompatibleProvider.detect_speakers)
    assert "get_small_chat_call_timeout" in src, (
        "detect_speakers must pass timeout=get_small_chat_call_timeout(self.cfg); without it "
        "_chat_create.setdefault applies the 400 s per-call bound (#1323)"
    )

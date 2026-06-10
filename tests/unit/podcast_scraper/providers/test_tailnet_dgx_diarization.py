"""Unit tests for TailnetDgxDiarizationProvider (#926, resilience #954)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper import Config
from podcast_scraper.providers.ml.diarization.base import (
    DiarizationResult,
    DiarizationSegment,
)
from podcast_scraper.providers.tailnet_dgx import diarization_provider as dp
from podcast_scraper.providers.tailnet_dgx.diarization_provider import (
    TailnetDgxDiarizationProvider,
)


@pytest.fixture(autouse=True)
def _reset_breaker():
    """The DGX diarize circuit breaker is process-wide; isolate every test."""
    dp._diarize_breaker.reset()
    yield
    dp._diarize_breaker.reset()


def _dgx_cfg() -> Config:
    return Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "diarization_provider": "tailnet_dgx",
            "diarize": True,
            "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            "hf_token": "hf_test",
        }
    )


def _local_fallback(model_name: str) -> MagicMock:
    fb = MagicMock()
    fb.diarize.return_value = DiarizationResult(
        segments=[DiarizationSegment(start=0.0, end=4.5, speaker="SPEAKER_00")],
        num_speakers=1,
        model_name=model_name,
    )
    return fb


def test_config_requires_host_for_tailnet_dgx() -> None:
    provider = TailnetDgxDiarizationProvider(
        Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "diarization_provider": "tailnet_dgx",
                "diarize": True,
                "hf_token": "hf_test",
                # dgx_tailnet_host intentionally omitted
            }
        )
    )
    with pytest.raises(ValueError, match="dgx_tailnet_host"):
        provider.initialize()


def test_provider_defaults_port_and_model() -> None:
    provider = TailnetDgxDiarizationProvider(_dgx_cfg())
    assert provider._port == 8001
    assert "speaker-diarization" in provider._model


@patch("podcast_scraper.providers.tailnet_dgx.diarization_provider.emit_dgx_fallback_breadcrumb")
@patch(
    "podcast_scraper.providers.tailnet_dgx.diarization_provider.check_pyannote_diarize_health",
    return_value=False,
)
@patch("podcast_scraper.providers.tailnet_dgx.diarization_provider.time.sleep")
def test_falls_back_to_local_when_dgx_unhealthy(
    mock_sleep: MagicMock,
    _health: MagicMock,
    _breadcrumb: MagicMock,
    tmp_path,
) -> None:
    audio = tmp_path / "ep.wav"
    audio.write_bytes(b"\x00\x00")

    provider = TailnetDgxDiarizationProvider(_dgx_cfg())
    provider.initialize()

    local_fallback = _local_fallback("local-fallback")
    with patch.object(provider, "_get_local_fallback", return_value=local_fallback):
        result = provider.diarize(str(audio))

    assert result.model_name == "local-fallback"
    local_fallback.diarize.assert_called_once()
    _breadcrumb.assert_called_once()
    # Unhealthy → retried each attempt then fell back: max_attempts-1 backoff sleeps.
    assert mock_sleep.call_count == provider._max_attempts - 1


@patch(
    "podcast_scraper.providers.tailnet_dgx.diarization_provider.check_pyannote_diarize_health",
    return_value=True,
)
@patch.object(TailnetDgxDiarizationProvider, "_diarize_dgx")
def test_healthy_dgx_path(
    mock_dgx_call: MagicMock,
    _health: MagicMock,
    tmp_path,
) -> None:
    audio = tmp_path / "ep.wav"
    audio.write_bytes(b"\x00\x00")
    mock_dgx_call.return_value = DiarizationResult(
        segments=[
            DiarizationSegment(start=0.0, end=4.5, speaker="SPEAKER_00"),
            DiarizationSegment(start=4.5, end=10.0, speaker="SPEAKER_01"),
        ],
        num_speakers=2,
        model_name="pyannote/speaker-diarization-3.1",
    )

    provider = TailnetDgxDiarizationProvider(_dgx_cfg())
    provider.initialize()

    result = provider.diarize(str(audio), min_speakers=2, max_speakers=5)
    assert result.num_speakers == 2
    mock_dgx_call.assert_called_once()  # runs once, under the watchdog
    assert dp._diarize_breaker.state == "closed"  # success keeps the breaker closed


@patch("podcast_scraper.providers.tailnet_dgx.diarization_provider.emit_dgx_fallback_breadcrumb")
@patch(
    "podcast_scraper.providers.tailnet_dgx.diarization_provider.check_pyannote_diarize_health",
    return_value=True,
)
@patch.object(TailnetDgxDiarizationProvider, "_diarize_dgx")
@patch("podcast_scraper.providers.tailnet_dgx.diarization_provider.time.sleep")
def test_dgx_raises_then_falls_back(
    mock_sleep: MagicMock,
    mock_dgx_call: MagicMock,
    _health: MagicMock,
    _breadcrumb: MagicMock,
    tmp_path,
) -> None:
    audio = tmp_path / "ep.wav"
    audio.write_bytes(b"\x00\x00")
    mock_dgx_call.side_effect = ValueError("empty diarization result from DGX pyannote service")

    provider = TailnetDgxDiarizationProvider(_dgx_cfg())
    provider.initialize()

    local_fallback = _local_fallback("local-after-dgx-error")
    with patch.object(provider, "_get_local_fallback", return_value=local_fallback):
        result = provider.diarize(str(audio))

    assert result.model_name == "local-after-dgx-error"
    _breadcrumb.assert_called_once()
    # A non-timeout error is retried each attempt before falling back.
    assert mock_dgx_call.call_count == provider._max_attempts


@patch("podcast_scraper.providers.tailnet_dgx.diarization_provider.emit_dgx_fallback_breadcrumb")
@patch(
    "podcast_scraper.providers.tailnet_dgx.diarization_provider.check_pyannote_diarize_health",
    return_value=True,
)
@patch.object(TailnetDgxDiarizationProvider, "_diarize_dgx")
@patch("podcast_scraper.providers.tailnet_dgx.diarization_provider.time.sleep")
def test_timeout_does_not_requeue_and_falls_back(
    mock_sleep: MagicMock,
    mock_dgx_call: MagicMock,
    _health: MagicMock,
    _breadcrumb: MagicMock,
    tmp_path,
) -> None:
    """A genuine timeout fails over immediately (no duplicate request) and trips
    the breaker so the next call skips DGX."""
    audio = tmp_path / "ep.wav"
    audio.write_bytes(b"\x00\x00")
    mock_dgx_call.side_effect = TimeoutError("read timed out")

    provider = TailnetDgxDiarizationProvider(_dgx_cfg())
    provider.initialize()

    local_fallback = _local_fallback("local-after-timeout")
    with patch.object(provider, "_get_local_fallback", return_value=local_fallback):
        result = provider.diarize(str(audio))

    assert result.model_name == "local-after-timeout"
    mock_dgx_call.assert_called_once()  # no re-queue on timeout
    mock_sleep.assert_not_called()  # broke immediately, no backoff
    assert dp._diarize_breaker.state == "open"  # hard timeout trips immediately


@patch("podcast_scraper.providers.tailnet_dgx.diarization_provider.emit_dgx_fallback_breadcrumb")
@patch.object(TailnetDgxDiarizationProvider, "_diarize_dgx")
@patch("podcast_scraper.providers.tailnet_dgx.diarization_provider.check_pyannote_diarize_health")
def test_open_breaker_skips_dgx_entirely(
    mock_health: MagicMock,
    mock_dgx_call: MagicMock,
    _breadcrumb: MagicMock,
    tmp_path,
) -> None:
    """While the breaker is open, diarize() must not probe health or hit DGX."""
    audio = tmp_path / "ep.wav"
    audio.write_bytes(b"\x00\x00")
    dp._diarize_breaker.record_failure(hard=True)  # force open

    provider = TailnetDgxDiarizationProvider(_dgx_cfg())
    provider.initialize()

    local_fallback = _local_fallback("local-circuit-open")
    with patch.object(provider, "_get_local_fallback", return_value=local_fallback):
        result = provider.diarize(str(audio))

    assert result.model_name == "local-circuit-open"
    mock_health.assert_not_called()
    mock_dgx_call.assert_not_called()
    assert _breadcrumb.call_args.kwargs["failure_reason"] == "dgx_diarize_circuit_open"


@patch("podcast_scraper.providers.tailnet_dgx.diarization_provider.emit_dgx_fallback_breadcrumb")
@patch(
    "podcast_scraper.providers.tailnet_dgx.diarization_provider.check_pyannote_diarize_health",
    return_value=True,
)
def test_watchdog_hard_deadline_forces_fallback(
    _health: MagicMock,
    _breadcrumb: MagicMock,
    tmp_path,
) -> None:
    """A request that hangs past the hard deadline (httpx timeout never fires) is
    abandoned by the watchdog and fails over to local pyannote."""
    audio = tmp_path / "ep.wav"
    audio.write_bytes(b"\x00\x00")

    provider = TailnetDgxDiarizationProvider(_dgx_cfg())
    provider.initialize()

    def _hang(*_a, **_k):
        time.sleep(3.0)  # longer than the (shrunk) deadline; daemon thread is abandoned
        return DiarizationResult(segments=[], num_speakers=0, model_name="never")

    local_fallback = _local_fallback("local-after-watchdog")
    started = time.monotonic()
    with (
        patch.object(provider, "_diarize_dgx", side_effect=_hang),
        patch.object(provider, "_effective_timeout_sec", return_value=0.05),
        patch.object(dp.resilience, "WATCHDOG_GRACE_SEC", 0.1),
        patch.object(provider, "_get_local_fallback", return_value=local_fallback),
    ):
        result = provider.diarize(str(audio))
    elapsed = time.monotonic() - started

    assert result.model_name == "local-after-watchdog"
    assert elapsed < 2.0  # bailed at ~0.15s, did NOT wait for the 3s hang
    assert dp._diarize_breaker.state == "open"  # watchdog timeout trips the breaker

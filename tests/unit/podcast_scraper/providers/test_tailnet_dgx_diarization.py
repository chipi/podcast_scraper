"""Unit tests for TailnetDgxDiarizationProvider (#926)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper import Config
from podcast_scraper.providers.ml.diarization.base import (
    DiarizationResult,
    DiarizationSegment,
)
from podcast_scraper.providers.tailnet_dgx.diarization_provider import (
    TailnetDgxDiarizationProvider,
)


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

    cfg = _dgx_cfg()
    provider = TailnetDgxDiarizationProvider(cfg)
    provider.initialize()

    local_fallback = MagicMock()
    local_fallback.diarize.return_value = DiarizationResult(
        segments=[DiarizationSegment(start=0.0, end=4.5, speaker="SPEAKER_00")],
        num_speakers=1,
        model_name="local-fallback",
    )

    with patch.object(provider, "_get_local_fallback", return_value=local_fallback):
        result = provider.diarize(str(audio))

    assert result.model_name == "local-fallback"
    local_fallback.diarize.assert_called_once()
    _breadcrumb.assert_called_once()
    mock_sleep.assert_called_once()


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
    mock_dgx_call.assert_called_once()


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

    cfg = _dgx_cfg()
    provider = TailnetDgxDiarizationProvider(cfg)
    provider.initialize()

    local_fallback = MagicMock()
    local_fallback.diarize.return_value = DiarizationResult(
        segments=[DiarizationSegment(start=0.0, end=4.5, speaker="SPEAKER_00")],
        num_speakers=1,
        model_name="local-after-dgx-error",
    )

    with patch.object(provider, "_get_local_fallback", return_value=local_fallback):
        result = provider.diarize(str(audio))

    assert result.model_name == "local-after-dgx-error"
    _breadcrumb.assert_called_once()
    assert mock_dgx_call.call_count == 2  # retried before fallback

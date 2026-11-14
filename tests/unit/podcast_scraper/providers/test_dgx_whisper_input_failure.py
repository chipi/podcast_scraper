"""DGX Whisper, reprocess strategy: a 500 on one input from a server that is up fails that input.

Prod 2026-11-11 to 11-14: faster-whisper answered 500 (cudaErrorInvalidDevice in word alignment)
to Colombia Calling 639's audio on every attempt and every probe, while it served 130-188 other
episodes a night. The probe resent the same audio, so each night held transcription for 900 s and
alerted "endpoint did not recover". A server that is down is still held for.
"""

from __future__ import annotations

from typing import Any, List
from unittest.mock import patch

import httpx
import pytest

from podcast_scraper import Config
from podcast_scraper.providers.resilience import (
    ResilienceFuseOpenError,
    ResilienceInputFailedError,
)
from podcast_scraper.providers.tailnet_dgx import whisper_provider as wp
from podcast_scraper.providers.tailnet_dgx.whisper_provider import (
    TailnetDgxWhisperTranscriptionProvider,
)

pytestmark = pytest.mark.unit

URL = "http://dgx-llm-1:8000/v1/audio/transcriptions"


@pytest.fixture(autouse=True)
def _isolate():
    wp._whisper_breaker.reset()
    with patch(
        "podcast_scraper.providers.resilience.sockets._duration_via_ffprobe", return_value=None
    ):
        yield
    wp._whisper_breaker.reset()


def _provider() -> TailnetDgxWhisperTranscriptionProvider:
    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "transcription_provider": "tailnet_dgx_whisper",
            "transcription_fallback_provider": "openai",
            "dgx_tailnet_host": "dgx-llm-1",
            "openai_api_key": "sk-test",
            "resilience_run_context": "reprocess",
            "resilience_backoff_schedule_sec": [0.0, 0.0, 0.0],
            "resilience_on_open_max_wait_sec": 0.002,
            "resilience_probe_interval_sec": 0.001,
        }
    )
    p = TailnetDgxWhisperTranscriptionProvider(cfg)
    p._initialized = True
    return p


def _answers_500(calls: List[Any], status: int = 500):
    def post(audio_path, language, timeout_sec=None, model_override=None, prompt=None):
        calls.append(audio_path)
        request = httpx.Request("POST", URL)
        raise httpx.HTTPStatusError(
            f"Server error '{status}'",
            request=request,
            response=httpx.Response(status, request=request),
        )

    return post


def _transcribe(provider, tmp_path, post):
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"\x00\x01")
    with patch.object(provider, "_transcribe_dgx", side_effect=post):
        return provider.transcribe_with_segments(str(audio), "en")


def test_a_500_from_a_server_that_is_up_fails_the_input_without_holding(tmp_path) -> None:
    calls: List[Any] = []
    with patch.object(wp, "check_faster_whisper_health", return_value=True):
        with pytest.raises(ResilienceInputFailedError) as exc_info:
            _transcribe(_provider(), tmp_path, _answers_500(calls))
    assert len(calls) == 3, "three attempts, no probe resending the same audio"
    assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)
    assert wp._whisper_breaker.state == "closed", "the next episode is not held behind it"


def test_a_500_while_the_server_is_down_still_holds(tmp_path) -> None:
    calls: List[Any] = []
    health = iter([True, True, True, False, False, False, False])
    with patch.object(wp, "check_faster_whisper_health", side_effect=lambda *a, **k: next(health)):
        with pytest.raises(ResilienceFuseOpenError):
            _transcribe(_provider(), tmp_path, _answers_500(calls))


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_a_busy_status_from_a_server_that_is_up_still_holds(tmp_path, status: int) -> None:
    """429/502/503/504 say "not now" (overload, a gateway), not "this input fails"."""
    calls: List[Any] = []
    with patch.object(wp, "check_faster_whisper_health", return_value=True):
        with pytest.raises(ResilienceFuseOpenError):
            _transcribe(_provider(), tmp_path, _answers_500(calls, status))

"""E2E: DGX Whisper + pyannote clients against the mock server with real payloads.

The unit suites mock transport; the integration suite uses a throwaway stub. This
drives the *real* ``TailnetDgx{Whisper,Diarization}Provider`` over real httpx
against the shared e2e HTTP server (the canonical "real payload, mock endpoint"
harness) — the only DGX coverage that exercises the providers through the same
server the rest of the pipeline e2e tests use. Covers the happy round-trip plus
the production failure modes (a hanging socket and an HTTP 5xx) via the server's
``set_error_behavior`` injection, asserting fail-over each time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.e2e]

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import Config  # noqa: E402
from podcast_scraper.providers import resilience  # noqa: E402
from podcast_scraper.providers.ml.diarization.base import (  # noqa: E402
    DiarizationResult,
    DiarizationSegment,
)
from podcast_scraper.providers.tailnet_dgx import (  # noqa: E402
    diarization_provider as dp,
    whisper_provider as wp,
)
from podcast_scraper.providers.tailnet_dgx.diarization_provider import (  # noqa: E402
    TailnetDgxDiarizationProvider,
)
from podcast_scraper.providers.tailnet_dgx.whisper_provider import (  # noqa: E402
    TailnetDgxWhisperTranscriptionProvider,
)

_AUDIO = Path(__file__).parent.parent / "fixtures" / "audio" / "v1" / "p01_e01.mp3"
_TRANSCRIBE_PATH = "/v1/audio/transcriptions"
_DIARIZE_PATH = "/v1/diarize"


@pytest.fixture(autouse=True)
def _reset_dgx_state(e2e_server):
    """Process-wide breakers + the shared server's error registry must be clean
    around every DGX e2e test (the server fixture is session-scoped)."""
    from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

    E2EHTTPRequestHandler.clear_all_error_behaviors()
    dp._diarize_breaker.reset()
    wp._whisper_breaker.reset()
    yield
    E2EHTTPRequestHandler.clear_all_error_behaviors()
    dp._diarize_breaker.reset()
    wp._whisper_breaker.reset()


def _diarize_provider(e2e_server, **overrides):
    host, port = e2e_server.urls.dgx_host_port()
    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "diarization_provider": "tailnet_dgx",
            "diarize": True,
            "dgx_tailnet_host": host,
            "dgx_diarize_port": port,
            "dgx_diarize_request_timeout_sec": 0.5,
            "dgx_diarize_timeout_per_audio_minute_sec": 0,
            "hf_token": "hf_test",
            **overrides,
        }
    )
    p = TailnetDgxDiarizationProvider(cfg)
    p.initialize()
    return p


def _whisper_provider(e2e_server, **overrides):
    # RFC-105 (#1198): whisper is a pure DGX tier now — it raises on failure and the FallbackChain
    # (unit-tested) fails over. These e2e tests assert the tier's own socket-level reaction.
    host, port = e2e_server.urls.dgx_host_port()
    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "transcription_provider": "tailnet_dgx_whisper",
            "transcription_fallback_provider": "openai",
            "dgx_tailnet_host": host,
            "dgx_whisper_port": port,
            "dgx_whisper_model": "large-v3",
            "dgx_request_timeout_sec": 0.5,
            "dgx_timeout_per_audio_minute_sec": 0,
            "openai_api_key": "sk-test",
            **overrides,
        }
    )
    p = TailnetDgxWhisperTranscriptionProvider(cfg)
    p._initialized = True
    return p


class TestDGXWhisperE2E:
    def test_transcribes_against_mock_server(self, e2e_server):
        # Generous request timeout: the happy path must not trip the breaker on transient
        # slowness under parallel CI load (the 0.5s default is only for the fail-over tests).
        text = _whisper_provider(e2e_server, dgx_request_timeout_sec=10.0).transcribe(str(_AUDIO))
        # The mock server answers verbose_json with a canned transcript.
        assert "test transcription" in text.lower()
        assert wp._whisper_breaker.state == "closed"

    def test_http_503_raises_for_the_chain(self, e2e_server):
        import httpx

        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.set_error_behavior(_TRANSCRIBE_PATH, status=503)
        provider = _whisper_provider(e2e_server)
        with patch.object(wp.time, "sleep"):  # skip retry backoff
            with pytest.raises(httpx.HTTPStatusError):
                provider.transcribe(str(_AUDIO))

    def test_hanging_socket_watchdog_raises(self, e2e_server):
        from podcast_scraper.providers.resilience import TimeoutLike
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        # status is irrelevant — the client bails during the injected delay.
        E2EHTTPRequestHandler.set_error_behavior(_TRANSCRIBE_PATH, status=200, delay=10.0)
        provider = _whisper_provider(e2e_server)
        with patch.object(resilience, "WATCHDOG_GRACE_SEC", 0.2):
            with pytest.raises(TimeoutLike):
                provider.transcribe(str(_AUDIO))
        assert wp._whisper_breaker.state == "open"


class TestDGXDiarizeE2E:
    def test_diarizes_against_mock_server(self, e2e_server):
        # Happy-path: give a generous request timeout so transient slowness under
        # parallel CI load can't trip the breaker (the 0.5s default exists only for
        # the fail-over tests, which force failure via 503 / watchdog, not this timeout).
        result = _diarize_provider(e2e_server, dgx_diarize_request_timeout_sec=10.0).diarize(
            str(_AUDIO)
        )
        assert isinstance(result, DiarizationResult)
        assert result.num_speakers == 2
        assert {s.speaker for s in result.segments} == {"SPEAKER_00", "SPEAKER_01"}
        assert dp._diarize_breaker.state == "closed"

    def test_http_503_fails_over_to_local(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.set_error_behavior(_DIARIZE_PATH, status=503)
        provider = _diarize_provider(e2e_server)
        local = MagicMock()
        local.diarize.return_value = DiarizationResult(
            segments=[DiarizationSegment(0.0, 1.0, "SPEAKER_00")],
            num_speakers=1,
            model_name="local",
        )
        with (
            patch.object(provider, "_get_local_fallback", return_value=local),
            patch.object(dp.time, "sleep"),
        ):
            assert provider.diarize(str(_AUDIO)).model_name == "local"

    def test_hanging_socket_watchdog_fails_over(self, e2e_server):
        from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler

        E2EHTTPRequestHandler.set_error_behavior(_DIARIZE_PATH, status=200, delay=10.0)
        provider = _diarize_provider(e2e_server)
        local = MagicMock()
        local.diarize.return_value = DiarizationResult(
            segments=[DiarizationSegment(0.0, 1.0, "SPEAKER_00")],
            num_speakers=1,
            model_name="local-after-hang",
        )
        with (
            patch.object(provider, "_get_local_fallback", return_value=local),
            patch.object(resilience, "WATCHDOG_GRACE_SEC", 0.2),
        ):
            assert provider.diarize(str(_AUDIO)).model_name == "local-after-hang"
        assert dp._diarize_breaker.state == "open"

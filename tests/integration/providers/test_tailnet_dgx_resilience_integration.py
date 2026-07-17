"""Real-socket integration tests for the DGX provider resilience layer (#954).

The unit suites cover the resilience *logic* with mocked transport. These exercise
the real path — actual httpx over a loopback socket against a throwaway stub that
mimics the faster-whisper / pyannote services — so we prove that:

* the happy path round-trips real HTTP (multipart upload -> JSON parse), and
* a *hanging* socket (the production failure: httpx's own timeout never fires
  because the upload trickles) is abandoned by the hard watchdog and fails over,
* an HTTP 5xx fails over after retries,
* the circuit breaker trips on a hard timeout and the next call skips DGX.

Self-contained (its own ``http.server`` stub, not the shared e2e server) so it
can't perturb the e2e suite. Component-level (provider vs a local server) → the
integration tier; external service is the stub, nothing real is hit.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper import Config
from podcast_scraper.providers import resilience
from podcast_scraper.providers.ml.diarization.base import DiarizationResult
from podcast_scraper.providers.tailnet_dgx import (
    diarization_provider as dp,
    whisper_provider as wp,
)
from podcast_scraper.providers.tailnet_dgx.diarization_provider import (
    TailnetDgxDiarizationProvider,
)
from podcast_scraper.providers.tailnet_dgx.whisper_provider import (
    TailnetDgxWhisperTranscriptionProvider,
)

pytestmark = pytest.mark.integration


class _DGXStubHandler(BaseHTTPRequestHandler):
    """Minimal stand-in for the DGX faster-whisper + pyannote services.

    ``mode`` (class attr, set per test) controls only the POST work endpoints:
    ``"ok"`` returns a valid payload, ``"hang"`` sleeps ``hang_sec`` (simulating a
    GPU-stalled request that never returns), ``"503"`` returns Service Unavailable.
    Health/model endpoints always answer 200 so the client gets past the gate.
    """

    mode = "ok"
    hang_sec = 30.0

    def log_message(self, *_args):  # silence the default stderr spam
        pass

    def _json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path == "/health":
            self._json(200, {"status": "ok"})
            return
        if self.path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "large-v3", "object": "model"},
                        {"id": "pyannote/speaker-diarization-community-1", "object": "model"},
                    ],
                },
            )
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802 - http.server API
        # Drain the request body (the real providers upload the audio file here).
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)

        if type(self).mode == "hang":
            time.sleep(type(self).hang_sec)  # never returns within the watchdog window
            return
        if type(self).mode == "503":
            self._json(503, {"error": "service unavailable"})
            return

        if self.path == "/v1/diarize":
            self._json(
                200,
                {
                    "model_name": "pyannote/speaker-diarization-community-1",
                    "num_speakers": 2,
                    "segments": [
                        {"start": 0.0, "end": 4.5, "speaker": "SPEAKER_00"},
                        {"start": 4.5, "end": 9.0, "speaker": "SPEAKER_01"},
                    ],
                },
            )
            return
        if self.path == "/v1/audio/transcriptions":
            self._json(
                200,
                {
                    "text": "hello from the stub",
                    "segments": [{"start": 0.0, "end": 2.0, "text": "hello from the stub"}],
                },
            )
            return
        self._json(404, {"error": "not found"})


@pytest.fixture
def dgx_stub():
    """Start the stub on an ephemeral loopback port; reset breakers + mode each test."""
    _DGXStubHandler.mode = "ok"
    dp._diarize_breaker.reset()
    wp._whisper_breaker.reset()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DGXStubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]  # the bound port
    finally:
        server.shutdown()
        server.server_close()
        _DGXStubHandler.mode = "ok"
        dp._diarize_breaker.reset()
        wp._whisper_breaker.reset()


def _audio(tmp_path) -> str:
    f = tmp_path / "ep.wav"
    f.write_bytes(b"RIFFstub-audio-bytes" * 8)
    return str(f)


def _diarize_cfg(port: int) -> Config:
    return Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "diarization_provider": "tailnet_dgx",
            "diarize": True,
            "dgx_tailnet_host": "127.0.0.1",
            "dgx_diarize_port": port,
            # Tight budget + no per-minute scaling so the watchdog fires fast in-test.
            "dgx_diarize_request_timeout_sec": 0.5,
            "dgx_diarize_timeout_per_audio_minute_sec": 0,
            "hf_token": "hf_test",
        }
    )


def _whisper_cfg(port: int) -> Config:
    return Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "transcription_provider": "tailnet_dgx_whisper",
            "transcription_fallback_provider": "openai",
            "dgx_tailnet_host": "127.0.0.1",
            "dgx_whisper_port": port,
            "dgx_whisper_model": "large-v3",
            "dgx_request_timeout_sec": 0.5,
            "dgx_timeout_per_audio_minute_sec": 0,
            "openai_api_key": "sk-test",
        }
    )


def _diarize_provider(port: int):
    p = TailnetDgxDiarizationProvider(_diarize_cfg(port))
    p.initialize()
    return p


def _whisper_provider(port: int):
    # RFC-105 (#1198): the whisper provider is a pure DGX tier and owns no fallback — it raises on
    # failure and the FallbackChain (exercised in the unit suite) fails over. These socket-level
    # tests assert the tier's own reaction: detect the hang/503 and raise the classified error.
    p = TailnetDgxWhisperTranscriptionProvider(_whisper_cfg(port))
    p._initialized = True
    return p


# --------------------------------------------------------------------------- #
# diarization                                                                  #
# --------------------------------------------------------------------------- #
class TestDiarizeRealSocket:
    def test_happy_path_round_trips(self, dgx_stub, tmp_path):
        result = _diarize_provider(dgx_stub).diarize(_audio(tmp_path))
        assert isinstance(result, DiarizationResult)
        assert result.num_speakers == 2
        assert {s.speaker for s in result.segments} == {"SPEAKER_00", "SPEAKER_01"}
        assert dp._diarize_breaker.state == "closed"

    def test_hanging_socket_watchdog_fails_over(self, dgx_stub, tmp_path):
        _DGXStubHandler.mode = "hang"
        provider = _diarize_provider(dgx_stub)
        local = MagicMock()
        local.diarize.return_value = DiarizationResult(
            segments=[], num_speakers=1, model_name="local"
        )
        started = time.monotonic()
        with (
            patch.object(provider, "_get_local_fallback", return_value=local),
            patch.object(resilience, "WATCHDOG_GRACE_SEC", 0.2),
        ):
            result = provider.diarize(_audio(tmp_path))
        elapsed = time.monotonic() - started
        assert result.model_name == "local"  # failed over
        assert elapsed < 5.0  # bailed at ~0.7s, NOT the 30s stub hang
        assert dp._diarize_breaker.state == "open"  # hard timeout tripped the breaker

    def test_http_503_fails_over(self, dgx_stub, tmp_path):
        _DGXStubHandler.mode = "503"
        provider = _diarize_provider(dgx_stub)
        local = MagicMock()
        local.diarize.return_value = DiarizationResult(
            segments=[], num_speakers=1, model_name="local"
        )
        with (
            patch.object(provider, "_get_local_fallback", return_value=local),
            patch.object(dp.time, "sleep"),  # skip retry backoff
        ):
            result = provider.diarize(_audio(tmp_path))
        assert result.model_name == "local"

    def test_open_breaker_skips_socket_entirely(self, dgx_stub, tmp_path):
        # First call hangs → trips the breaker; second call must not touch the socket.
        _DGXStubHandler.mode = "hang"
        provider = _diarize_provider(dgx_stub)
        local = MagicMock()
        local.diarize.return_value = DiarizationResult(
            segments=[], num_speakers=1, model_name="local"
        )
        with (
            patch.object(provider, "_get_local_fallback", return_value=local),
            patch.object(resilience, "WATCHDOG_GRACE_SEC", 0.2),
        ):
            provider.diarize(_audio(tmp_path))  # trips breaker
            assert dp._diarize_breaker.state == "open"
            # Even though the stub would still hang, this returns immediately (no probe).
            started = time.monotonic()
            provider.diarize(_audio(tmp_path))
            assert time.monotonic() - started < 0.5


# --------------------------------------------------------------------------- #
# whisper                                                                      #
# --------------------------------------------------------------------------- #
class TestWhisperRealSocket:
    def test_happy_path_round_trips(self, dgx_stub, tmp_path):
        text = _whisper_provider(dgx_stub).transcribe(_audio(tmp_path))
        assert text == "hello from the stub"
        assert wp._whisper_breaker.state == "closed"

    def test_hanging_socket_watchdog_raises(self, dgx_stub, tmp_path):
        from podcast_scraper.providers.resilience import TimeoutLike

        _DGXStubHandler.mode = "hang"
        provider = _whisper_provider(dgx_stub)
        started = time.monotonic()
        with patch.object(resilience, "WATCHDOG_GRACE_SEC", 0.2):
            with pytest.raises(TimeoutLike):
                provider.transcribe(_audio(tmp_path))
        elapsed = time.monotonic() - started
        assert elapsed < 5.0  # bailed fast, not the 30s hang
        assert wp._whisper_breaker.state == "open"  # hard timeout tripped the breaker

    def test_http_503_raises(self, dgx_stub, tmp_path):
        import httpx

        _DGXStubHandler.mode = "503"
        provider = _whisper_provider(dgx_stub)
        with patch.object(wp.time, "sleep"):  # skip retry backoff
            with pytest.raises(httpx.HTTPStatusError):
                provider.transcribe(_audio(tmp_path))

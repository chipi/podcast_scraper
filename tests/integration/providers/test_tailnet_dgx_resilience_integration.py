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
from unittest.mock import patch

import pytest

from podcast_scraper import Config
from podcast_scraper.providers import resilience
from podcast_scraper.providers.ml.diarization import moss_provider as mdp
from podcast_scraper.providers.ml.diarization.base import DiarizationResult
from podcast_scraper.providers.ml.diarization.moss_provider import MossDiarizationProvider
from podcast_scraper.providers.moss import moss_provider as mp
from podcast_scraper.providers.moss.moss_provider import MossTranscriptionProvider
from podcast_scraper.providers.resilience import policy as resilience_policy
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
    GPU-stalled request that never returns), ``"503"`` returns Service Unavailable,
    ``"hang_then_ok"`` hangs for the first ``fail_n`` POSTs then answers ``ok`` (ADR-122
    reprocess-mode backoff-retry coverage). Health/model endpoints always answer 200 so
    the client gets past the gate.
    """

    mode = "ok"
    hang_sec = 30.0
    fail_n = 0  # "hang_then_ok" only: how many POSTs hang before answering ok
    _post_count = 0  # class-level counter for "hang_then_ok"; reset per test by the fixture

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
        if type(self).mode == "hang_then_ok":
            type(self)._post_count += 1
            if type(self)._post_count <= type(self).fail_n:
                time.sleep(type(self).hang_sec)  # this call hangs; a later retry won't
                return
            # fail_n calls have already hung; fall through to the normal ok response.
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
        if self.path == "/v1/transcribe":
            self._json(
                200,
                {
                    "text": "hello from moss",
                    "segments": [
                        {"start": 0.0, "end": 2.0, "text": "hello from moss", "speaker": "S01"}
                    ],
                    "num_speakers": 1,
                },
            )
            return
        self._json(404, {"error": "not found"})


@pytest.fixture
def dgx_stub():
    """Start the stub on an ephemeral loopback port; reset breakers + mode each test."""
    _DGXStubHandler.mode = "ok"
    _DGXStubHandler.fail_n = 0
    _DGXStubHandler._post_count = 0
    dp._diarize_breaker.reset()
    wp._whisper_breaker.reset()
    mp._moss_breaker.reset()
    mdp._moss_diarize_breaker.reset()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DGXStubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]  # the bound port
    finally:
        server.shutdown()
        server.server_close()
        _DGXStubHandler.mode = "ok"
        _DGXStubHandler.fail_n = 0
        _DGXStubHandler._post_count = 0
        dp._diarize_breaker.reset()
        wp._whisper_breaker.reset()
        mp._moss_breaker.reset()
        mdp._moss_diarize_breaker.reset()


def _audio(tmp_path) -> str:
    f = tmp_path / "ep.wav"
    f.write_bytes(b"RIFFstub-audio-bytes" * 8)
    return str(f)


def _diarize_cfg(
    port: int,
    *,
    run_context: str | None = None,
    retries_before_trip: int | None = None,
    backoff_schedule_sec: list[float] | None = None,
    on_open_max_wait_sec: float | None = None,
    probe_interval_sec: float | None = None,
    dgx_diarize_request_timeout_sec: float = 0.5,
) -> Config:
    data: dict[str, object] = {
        "rss_url": "https://example.com/feed.xml",
        "diarization_provider": "tailnet_dgx",
        "diarize": True,
        "dgx_tailnet_host": "127.0.0.1",
        "dgx_diarize_port": port,
        # Tight budget + no per-minute scaling so the watchdog fires fast in-test.
        "dgx_diarize_request_timeout_sec": dgx_diarize_request_timeout_sec,
        "dgx_diarize_timeout_per_audio_minute_sec": 0,
        "hf_token": "hf_test",
    }
    # ADR-122 resilience-policy knobs: only set when the test cares (keeps the happy-path
    # / serve-mode configs identical to before this change).
    if run_context is not None:
        data["resilience_run_context"] = run_context
    if retries_before_trip is not None:
        data["resilience_retries_before_trip"] = retries_before_trip
    if backoff_schedule_sec is not None:
        data["resilience_backoff_schedule_sec"] = backoff_schedule_sec
    if on_open_max_wait_sec is not None:
        data["resilience_on_open_max_wait_sec"] = on_open_max_wait_sec
    if probe_interval_sec is not None:
        data["resilience_probe_interval_sec"] = probe_interval_sec
    return Config.model_validate(data)


def _whisper_cfg(
    port: int,
    *,
    run_context: str | None = None,
    retries_before_trip: int | None = None,
    backoff_schedule_sec: list[float] | None = None,
    on_open_max_wait_sec: float | None = None,
    probe_interval_sec: float | None = None,
    dgx_request_timeout_sec: float = 0.5,
) -> Config:
    data: dict[str, object] = {
        "rss_url": "https://example.com/feed.xml",
        "transcription_provider": "tailnet_dgx_whisper",
        "transcription_fallback_provider": "openai",
        "dgx_tailnet_host": "127.0.0.1",
        "dgx_whisper_port": port,
        "dgx_whisper_model": "large-v3",
        "dgx_request_timeout_sec": dgx_request_timeout_sec,
        "dgx_timeout_per_audio_minute_sec": 0,
        "openai_api_key": "sk-test",
    }
    # ADR-122 resilience-policy knobs: only set when the test cares (keeps the happy-path
    # / serve-mode configs identical to before this change).
    if run_context is not None:
        data["resilience_run_context"] = run_context
    if retries_before_trip is not None:
        data["resilience_retries_before_trip"] = retries_before_trip
    if backoff_schedule_sec is not None:
        data["resilience_backoff_schedule_sec"] = backoff_schedule_sec
    if on_open_max_wait_sec is not None:
        data["resilience_on_open_max_wait_sec"] = on_open_max_wait_sec
    if probe_interval_sec is not None:
        data["resilience_probe_interval_sec"] = probe_interval_sec
    return Config.model_validate(data)


def _diarize_provider(port: int):
    p = TailnetDgxDiarizationProvider(_diarize_cfg(port))
    p.initialize()
    return p


def _diarize_provider_from_cfg(cfg: Config):
    p = TailnetDgxDiarizationProvider(cfg)
    p.initialize()
    return p


def _moss_cfg(
    port: int,
    *,
    run_context: str | None = None,
    retries_before_trip: int | None = None,
    backoff_schedule_sec: list[float] | None = None,
    on_open_max_wait_sec: float | None = None,
    probe_interval_sec: float | None = None,
    moss_request_timeout_sec: float = 0.5,
) -> Config:
    data: dict[str, object] = {
        "rss_url": "https://example.com/feed.xml",
        "transcription_provider": "moss",
        "dgx_tailnet_host": "127.0.0.1",
        "moss_port": port,
        "moss_request_timeout_sec": moss_request_timeout_sec,
    }
    # ADR-122 resilience-policy knobs: only set when the test cares (keeps the happy-path
    # / serve-mode configs identical to before this change).
    if run_context is not None:
        data["resilience_run_context"] = run_context
    if retries_before_trip is not None:
        data["resilience_retries_before_trip"] = retries_before_trip
    if backoff_schedule_sec is not None:
        data["resilience_backoff_schedule_sec"] = backoff_schedule_sec
    if on_open_max_wait_sec is not None:
        data["resilience_on_open_max_wait_sec"] = on_open_max_wait_sec
    if probe_interval_sec is not None:
        data["resilience_probe_interval_sec"] = probe_interval_sec
    return Config.model_validate(data)


def _moss_provider_from_cfg(cfg: Config) -> MossTranscriptionProvider:
    p = MossTranscriptionProvider(cfg)
    p._initialized = True
    return p


def _moss_diarize_provider_from_cfg(cfg: Config) -> MossDiarizationProvider:
    p = MossDiarizationProvider(cfg)
    p._initialized = True
    return p


def _whisper_provider(port: int):
    # RFC-106 (#1198): the whisper provider is a pure DGX tier and owns no fallback — it raises on
    # failure and the FallbackChain (exercised in the unit suite) fails over. These socket-level
    # tests assert the tier's own reaction: detect the hang/503 and raise the classified error.
    return _whisper_provider_from_cfg(_whisper_cfg(port))


def _whisper_provider_from_cfg(cfg: Config):
    p = TailnetDgxWhisperTranscriptionProvider(cfg)
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

    def test_hanging_socket_watchdog_raises(self, dgx_stub, tmp_path):
        from podcast_scraper.providers.resilience import TimeoutLike

        _DGXStubHandler.mode = "hang"
        provider = _diarize_provider(dgx_stub)
        started = time.monotonic()
        with patch.object(resilience, "WATCHDOG_GRACE_SEC", 0.2):
            with pytest.raises(TimeoutLike):
                provider.diarize(_audio(tmp_path))
        elapsed = time.monotonic() - started
        assert elapsed < 5.0  # bailed at ~0.7s, NOT the 30s stub hang
        assert dp._diarize_breaker.state == "open"  # hard timeout tripped the breaker

    def test_http_503_raises(self, dgx_stub, tmp_path):
        import httpx

        _DGXStubHandler.mode = "503"
        provider = _diarize_provider(dgx_stub)
        with patch.object(dp.time, "sleep"):  # skip retry backoff
            with pytest.raises(httpx.HTTPStatusError):
                provider.diarize(_audio(tmp_path))

    def test_open_breaker_skips_socket_entirely(self, dgx_stub, tmp_path):
        # First call hangs → trips the breaker; second call must not touch the socket (raises fast).
        _DGXStubHandler.mode = "hang"
        provider = _diarize_provider(dgx_stub)
        with patch.object(resilience, "WATCHDOG_GRACE_SEC", 0.2):
            with pytest.raises(Exception):
                provider.diarize(_audio(tmp_path))  # trips breaker
            assert dp._diarize_breaker.state == "open"
            # Even though the stub would still hang, this raises immediately (no probe).
            started = time.monotonic()
            with pytest.raises(RuntimeError, match="dgx_diarize_circuit_open"):
                provider.diarize(_audio(tmp_path))
            assert time.monotonic() - started < 0.5

    def test_serve_mode_regression_hard_timeout_trips_and_raises(self, dgx_stub, tmp_path):
        """ADR-122 regression guard: even with reprocess-shaped retry knobs populated,
        an explicit ``run_context="serve"`` config keeps today's behaviour — the first
        hard timeout trips the breaker immediately and raises for the FallbackChain.
        Mirrors ``TestWhisperRealSocket``'s serve-mode regression guard, so a future
        default change can't silently break diarize serve mode without failing this test."""
        from podcast_scraper.providers.resilience import TimeoutLike

        _DGXStubHandler.mode = "hang"
        cfg = _diarize_cfg(
            dgx_stub,
            run_context="serve",
            retries_before_trip=5,
            backoff_schedule_sec=[0.01],
            on_open_max_wait_sec=0.05,
            probe_interval_sec=0.01,
        )
        provider = _diarize_provider_from_cfg(cfg)
        started = time.monotonic()
        with patch.object(resilience, "WATCHDOG_GRACE_SEC", 0.2):
            with pytest.raises(TimeoutLike):
                provider.diarize(_audio(tmp_path))
        elapsed = time.monotonic() - started
        assert elapsed < 5.0  # bailed fast, not the 30s hang; never entered pause-and-probe
        assert dp._diarize_breaker.state == "open"  # first hard timeout tripped the breaker


# --------------------------------------------------------------------------- #
# diarization — ADR-122 reprocess-mode resilience policy                      #
# --------------------------------------------------------------------------- #
class TestDiarizeReprocessMode:
    """Reprocess-mode coverage (#1253): backoff-retry the chosen model, trip the fuse
    only after the policy threshold, hold-and-probe (never fall over) on a blown fuse."""

    def test_backoff_retry_then_succeeds_no_trip_no_fallover(self, dgx_stub, tmp_path):
        _DGXStubHandler.mode = "hang_then_ok"
        _DGXStubHandler.fail_n = 1  # first POST hangs; the retry succeeds
        cfg = _diarize_cfg(
            dgx_stub,
            run_context="reprocess",
            retries_before_trip=2,
            backoff_schedule_sec=[0.05],
        )
        provider = _diarize_provider_from_cfg(cfg)
        with patch.object(resilience_policy, "WATCHDOG_GRACE_SEC", 0.2):
            result = provider.diarize(_audio(tmp_path))
        assert isinstance(result, DiarizationResult)
        assert result.num_speakers == 2
        # Backed off and retried the SAME (only) model — no fallover, fuse never tripped.
        assert dp._diarize_breaker.state == "closed"

    def test_fuse_trips_only_after_n_not_on_first(self, dgx_stub, tmp_path):
        _DGXStubHandler.mode = "hang"  # every attempt hangs
        cfg = _diarize_cfg(
            dgx_stub,
            run_context="reprocess",
            retries_before_trip=3,
            backoff_schedule_sec=[0.02, 0.02],
            on_open_max_wait_sec=0.1,
            probe_interval_sec=0.02,
            dgx_diarize_request_timeout_sec=0.15,
        )
        provider = _diarize_provider_from_cfg(cfg)
        with patch.object(resilience_policy, "WATCHDOG_GRACE_SEC", 0.1):
            # Spy on the breaker's own record_failure to prove it is called exactly once,
            # only after all 3 backoff-retries are exhausted — not on the first hard
            # timeout the way serve mode trips it.
            with patch.object(
                dp._diarize_breaker,
                "record_failure",
                wraps=dp._diarize_breaker.record_failure,
            ) as spy:
                with pytest.raises(resilience_policy.ResilienceFuseOpenError):
                    provider.diarize(_audio(tmp_path))
        assert spy.call_count == 1
        spy.assert_called_once_with(hard=True)
        assert dp._diarize_breaker.state == "open"

    def test_fuse_trip_emits_operator_alert(self, dgx_stub, tmp_path):
        """ADR-122: the breaker TRIP itself fires a guarded Sentry alert (distinct from the
        sustained-open escalation) through the real hold flow — proven by spying the hook on the
        shared ASR CircuitBreaker."""
        _DGXStubHandler.mode = "hang"  # every attempt hangs -> trip after the retries
        cfg = _diarize_cfg(
            dgx_stub,
            run_context="reprocess",
            retries_before_trip=2,
            backoff_schedule_sec=[0.02],
            on_open_max_wait_sec=0.05,
            probe_interval_sec=0.02,
            dgx_diarize_request_timeout_sec=0.15,
        )
        provider = _diarize_provider_from_cfg(cfg)
        with patch.object(resilience_policy, "WATCHDOG_GRACE_SEC", 0.1):
            with patch.object(resilience.breakers, "_emit_breaker_trip_alert") as alert:
                with pytest.raises(resilience_policy.ResilienceFuseOpenError):
                    provider.diarize(_audio(tmp_path))
        # The closed->open trip paged the operator exactly once, carrying the endpoint name.
        assert alert.call_count == 1
        assert alert.call_args.args[0] == "dgx-diarize"

    def test_open_fuse_holds_and_probes_then_closes(self, dgx_stub, tmp_path):
        _DGXStubHandler.mode = "ok"  # the endpoint has recovered by the time we probe
        cfg = _diarize_cfg(
            dgx_stub,
            run_context="reprocess",
            retries_before_trip=3,
            backoff_schedule_sec=[0.02],
            on_open_max_wait_sec=2.0,
            probe_interval_sec=0.05,
        )
        provider = _diarize_provider_from_cfg(cfg)
        # Simulate a fuse already blown by an earlier episode in the batch, with its
        # cooldown already elapsed — the breaker has no public "expire now" hook, so we
        # poke the private deadline directly rather than sleeping out the real cooldown.
        dp._diarize_breaker.record_failure(hard=True)
        assert dp._diarize_breaker.state == "open"
        dp._diarize_breaker._open_until = time.monotonic() - 0.01
        with patch.object(resilience_policy, "WATCHDOG_GRACE_SEC", 0.2):
            result = provider.diarize(_audio(tmp_path))
        # Never switched models: the result IS the DGX stub's own response.
        assert result.num_speakers == 2
        # Half-open probe succeeded -> breaker closes.
        assert dp._diarize_breaker.state == "closed"

    def test_open_fuse_alerts_operator_after_max_wait(self, dgx_stub, tmp_path):
        _DGXStubHandler.mode = "hang"  # the endpoint stays down through every probe
        cfg = _diarize_cfg(
            dgx_stub,
            run_context="reprocess",
            retries_before_trip=3,
            backoff_schedule_sec=[0.02],
            on_open_max_wait_sec=0.1,
            probe_interval_sec=0.03,
            dgx_diarize_request_timeout_sec=0.15,
        )
        provider = _diarize_provider_from_cfg(cfg)
        dp._diarize_breaker.record_failure(hard=True)
        dp._diarize_breaker._open_until = time.monotonic() - 0.01
        with patch.object(resilience_policy, "WATCHDOG_GRACE_SEC", 0.1):
            with pytest.raises(resilience_policy.ResilienceFuseOpenError):
                provider.diarize(_audio(tmp_path))
        # Still open — never fell over to another model, just gave up probing and raised.
        assert dp._diarize_breaker.state == "open"


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

    def test_serve_mode_regression_hard_timeout_trips_and_raises(self, dgx_stub, tmp_path):
        """ADR-122 regression guard: even with reprocess-shaped retry knobs populated,
        an explicit ``run_context="serve"`` config keeps today's behaviour — the first
        hard timeout trips the breaker immediately and raises for the FallbackChain.
        Mirrors ``test_hanging_socket_watchdog_raises`` above but proves the gate itself
        (not just the default), so a future change to the default can't silently break
        serve mode without failing this test."""
        from podcast_scraper.providers.resilience import TimeoutLike

        _DGXStubHandler.mode = "hang"
        cfg = _whisper_cfg(
            dgx_stub,
            run_context="serve",
            retries_before_trip=5,
            backoff_schedule_sec=[0.01],
            on_open_max_wait_sec=0.05,
            probe_interval_sec=0.01,
        )
        provider = _whisper_provider_from_cfg(cfg)
        started = time.monotonic()
        with patch.object(resilience, "WATCHDOG_GRACE_SEC", 0.2):
            with pytest.raises(TimeoutLike):
                provider.transcribe(_audio(tmp_path))
        elapsed = time.monotonic() - started
        assert elapsed < 5.0  # bailed fast, not the 30s hang; never entered pause-and-probe
        assert wp._whisper_breaker.state == "open"  # first hard timeout tripped the breaker


# --------------------------------------------------------------------------- #
# whisper — ADR-122 reprocess-mode resilience policy                          #
# --------------------------------------------------------------------------- #
class TestWhisperReprocessMode:
    """Reprocess-mode coverage (#1253): backoff-retry the chosen model, trip the fuse
    only after the policy threshold, hold-and-probe (never fall over) on a blown fuse."""

    def test_backoff_retry_then_succeeds_no_trip_no_fallover(self, dgx_stub, tmp_path):
        _DGXStubHandler.mode = "hang_then_ok"
        _DGXStubHandler.fail_n = 1  # first POST hangs; the retry succeeds
        cfg = _whisper_cfg(
            dgx_stub,
            run_context="reprocess",
            retries_before_trip=2,
            backoff_schedule_sec=[0.05],
        )
        provider = _whisper_provider_from_cfg(cfg)
        with patch.object(resilience_policy, "WATCHDOG_GRACE_SEC", 0.2):
            text = provider.transcribe(_audio(tmp_path))
        assert text == "hello from the stub"
        # Backed off and retried the SAME (only) model — no fallover, fuse never tripped.
        assert wp._whisper_breaker.state == "closed"

    def test_fuse_trips_only_after_n_not_on_first(self, dgx_stub, tmp_path):
        _DGXStubHandler.mode = "hang"  # every attempt hangs
        cfg = _whisper_cfg(
            dgx_stub,
            run_context="reprocess",
            retries_before_trip=3,
            backoff_schedule_sec=[0.02, 0.02],
            on_open_max_wait_sec=0.1,
            probe_interval_sec=0.02,
            dgx_request_timeout_sec=0.15,
        )
        provider = _whisper_provider_from_cfg(cfg)
        with patch.object(resilience_policy, "WATCHDOG_GRACE_SEC", 0.1):
            # Spy on the breaker's own record_failure to prove it is called exactly once,
            # only after all 3 backoff-retries are exhausted — not on the first hard
            # timeout the way serve mode trips it.
            with patch.object(
                wp._whisper_breaker,
                "record_failure",
                wraps=wp._whisper_breaker.record_failure,
            ) as spy:
                with pytest.raises(resilience_policy.ResilienceFuseOpenError):
                    provider.transcribe(_audio(tmp_path))
        assert spy.call_count == 1
        spy.assert_called_once_with(hard=True)
        assert wp._whisper_breaker.state == "open"

    def test_open_fuse_holds_and_probes_then_closes(self, dgx_stub, tmp_path):
        _DGXStubHandler.mode = "ok"  # the endpoint has recovered by the time we probe
        cfg = _whisper_cfg(
            dgx_stub,
            run_context="reprocess",
            retries_before_trip=3,
            backoff_schedule_sec=[0.02],
            on_open_max_wait_sec=2.0,
            probe_interval_sec=0.05,
        )
        provider = _whisper_provider_from_cfg(cfg)
        # Simulate a fuse already blown by an earlier episode in the batch, with its
        # cooldown already elapsed — the breaker has no public "expire now" hook, so we
        # poke the private deadline directly rather than sleeping out the real cooldown.
        wp._whisper_breaker.record_failure(hard=True)
        assert wp._whisper_breaker.state == "open"
        wp._whisper_breaker._open_until = time.monotonic() - 0.01
        with patch.object(resilience_policy, "WATCHDOG_GRACE_SEC", 0.2):
            text = provider.transcribe(_audio(tmp_path))
        # Never switched models: the result IS the DGX stub's own response.
        assert text == "hello from the stub"
        # Half-open probe succeeded -> breaker closes.
        assert wp._whisper_breaker.state == "closed"

    def test_open_fuse_alerts_operator_after_max_wait(self, dgx_stub, tmp_path):
        _DGXStubHandler.mode = "hang"  # the endpoint stays down through every probe
        cfg = _whisper_cfg(
            dgx_stub,
            run_context="reprocess",
            retries_before_trip=3,
            backoff_schedule_sec=[0.02],
            on_open_max_wait_sec=0.1,
            probe_interval_sec=0.03,
            dgx_request_timeout_sec=0.15,
        )
        provider = _whisper_provider_from_cfg(cfg)
        wp._whisper_breaker.record_failure(hard=True)
        wp._whisper_breaker._open_until = time.monotonic() - 0.01
        with patch.object(resilience_policy, "WATCHDOG_GRACE_SEC", 0.1):
            with pytest.raises(resilience_policy.ResilienceFuseOpenError):
                provider.transcribe(_audio(tmp_path))
        # Still open — never fell over to another model, just gave up probing and raised.
        assert wp._whisper_breaker.state == "open"


# --------------------------------------------------------------------------- #
# MOSS — ADR-122: was bare (no retry, no breaker), now gains the same         #
# call + circuit + policy layers as whisper/diarize                          #
# --------------------------------------------------------------------------- #
class TestMossResilience:
    def test_happy_path_round_trips(self, dgx_stub, tmp_path):
        cfg = _moss_cfg(dgx_stub)
        provider = _moss_provider_from_cfg(cfg)
        text = provider.transcribe(_audio(tmp_path))
        assert text == "hello from moss"
        assert mp._moss_breaker.state == "closed"

    def test_serve_mode_hard_timeout_trips_and_raises(self, dgx_stub, tmp_path):
        """ADR-122: MOSS was previously bare (no watchdog, no breaker at all) — this proves
        the newly-added serve-mode call layer actually classifies a hard timeout, trips the
        breaker on the first one, and raises for the wrapping FallbackChain, mirroring
        whisper/diarize's serve branches."""
        from podcast_scraper.providers.resilience import TimeoutLike

        _DGXStubHandler.mode = "hang"
        cfg = _moss_cfg(dgx_stub)
        provider = _moss_provider_from_cfg(cfg)
        started = time.monotonic()
        with patch.object(resilience, "WATCHDOG_GRACE_SEC", 0.2):
            with pytest.raises(TimeoutLike):
                provider.transcribe(_audio(tmp_path))
        elapsed = time.monotonic() - started
        assert elapsed < 5.0  # bailed fast, not the 30s hang
        assert mp._moss_breaker.state == "open"  # hard timeout tripped the breaker

    def test_backoff_retry_then_succeeds_no_trip_no_fallover(self, dgx_stub, tmp_path):
        _DGXStubHandler.mode = "hang_then_ok"
        _DGXStubHandler.fail_n = 1  # first POST hangs; the retry succeeds
        cfg = _moss_cfg(
            dgx_stub,
            run_context="reprocess",
            retries_before_trip=2,
            backoff_schedule_sec=[0.05],
        )
        provider = _moss_provider_from_cfg(cfg)
        with patch.object(resilience_policy, "WATCHDOG_GRACE_SEC", 0.2):
            text = provider.transcribe(_audio(tmp_path))
        assert text == "hello from moss"
        # Backed off and retried the SAME (only) model — no fallover, fuse never tripped.
        assert mp._moss_breaker.state == "closed"

    def test_fuse_trips_only_after_n_not_on_first(self, dgx_stub, tmp_path):
        _DGXStubHandler.mode = "hang"  # every attempt hangs
        cfg = _moss_cfg(
            dgx_stub,
            run_context="reprocess",
            retries_before_trip=3,
            backoff_schedule_sec=[0.02, 0.02],
            on_open_max_wait_sec=0.1,
            probe_interval_sec=0.02,
            moss_request_timeout_sec=0.15,
        )
        provider = _moss_provider_from_cfg(cfg)
        with patch.object(resilience_policy, "WATCHDOG_GRACE_SEC", 0.1):
            # Spy on the breaker's own record_failure to prove it is called exactly once,
            # only after all 3 backoff-retries are exhausted — not on the first hard
            # timeout the way serve mode trips it.
            with patch.object(
                mp._moss_breaker,
                "record_failure",
                wraps=mp._moss_breaker.record_failure,
            ) as spy:
                with pytest.raises(resilience_policy.ResilienceFuseOpenError):
                    provider.transcribe(_audio(tmp_path))
        assert spy.call_count == 1
        spy.assert_called_once_with(hard=True)
        assert mp._moss_breaker.state == "open"


# --------------------------------------------------------------------------- #
# MOSS diarization — ADR-122 item 1: the speaker half of the joint model was   #
# ALSO bare; it gains the same policy (its own _moss_diarize_breaker).         #
# --------------------------------------------------------------------------- #
class TestMossDiarizeResilience:
    def test_backoff_retry_then_succeeds_no_trip_no_fallover(self, dgx_stub, tmp_path):
        _DGXStubHandler.mode = "hang_then_ok"
        _DGXStubHandler.fail_n = 1  # first POST hangs; the retry succeeds
        cfg = _moss_cfg(
            dgx_stub,
            run_context="reprocess",
            retries_before_trip=2,
            backoff_schedule_sec=[0.05],
        )
        provider = _moss_diarize_provider_from_cfg(cfg)
        with patch.object(resilience_policy, "WATCHDOG_GRACE_SEC", 0.2):
            result = provider.diarize(_audio(tmp_path))
        assert isinstance(result, DiarizationResult)
        # Backed off and retried the SAME model — no fallover, fuse never tripped.
        assert mdp._moss_diarize_breaker.state == "closed"

    def test_fuse_trips_only_after_n_not_on_first(self, dgx_stub, tmp_path):
        _DGXStubHandler.mode = "hang"  # every attempt hangs
        cfg = _moss_cfg(
            dgx_stub,
            run_context="reprocess",
            retries_before_trip=3,
            backoff_schedule_sec=[0.02, 0.02],
            on_open_max_wait_sec=0.1,
            probe_interval_sec=0.02,
            moss_request_timeout_sec=0.15,
        )
        provider = _moss_diarize_provider_from_cfg(cfg)
        with patch.object(resilience_policy, "WATCHDOG_GRACE_SEC", 0.1):
            with patch.object(
                mdp._moss_diarize_breaker,
                "record_failure",
                wraps=mdp._moss_diarize_breaker.record_failure,
            ) as spy:
                with pytest.raises(resilience_policy.ResilienceFuseOpenError):
                    provider.diarize(_audio(tmp_path))
        # Tripped exactly once, only after all 3 backoff-retries were exhausted.
        assert spy.call_count == 1
        spy.assert_called_once_with(hard=True)
        assert mdp._moss_diarize_breaker.state == "open"


def test_fuse_open_emits_operator_alert(dgx_stub, tmp_path):
    """ADR-122 item 2: a sustained fuse-open fires a dedicated operator alert (a Sentry
    capture_message via ``_emit_fuse_open_alert``), not just an ERROR log — proven by spying
    on the alert hook. One alert per blown fuse, carrying the endpoint name + wait budget."""
    _DGXStubHandler.mode = "hang"  # never recovers -> the fuse stays open past max-wait
    cfg = _moss_cfg(
        dgx_stub,
        run_context="reprocess",
        retries_before_trip=2,
        backoff_schedule_sec=[0.02],
        on_open_max_wait_sec=0.05,
        probe_interval_sec=0.02,
        moss_request_timeout_sec=0.1,
    )
    provider = _moss_provider_from_cfg(cfg)
    with patch.object(resilience_policy, "WATCHDOG_GRACE_SEC", 0.05):
        with patch.object(resilience_policy, "_emit_fuse_open_alert") as alert:
            with pytest.raises(resilience_policy.ResilienceFuseOpenError):
                provider.transcribe(_audio(tmp_path))
    alert.assert_called_once()

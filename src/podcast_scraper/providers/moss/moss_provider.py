"""MOSS transcription provider — talks to the DGX MOSS service (#1177).

MOSS is a *joint* model: one pass yields transcript, speakers and timestamps. This provider takes
the transcript half; :mod:`podcast_scraper.providers.ml.diarization.moss_provider` takes the
speaker half. They call the same service endpoint, which caches its last few inferences by audio
digest, so the second stage costs nothing — the pipeline keeps its stage independence (the shape
Deepgram already established) without paying to run the model twice.

**Known limit, deliberately not hidden:** MOSS emits *segment*-level timestamps only. There is no
word-level output. That is a precision trade against faster-whisper (p95 ~0.26 s vs ~1.6 s), not a
return of the #1173 drift — which came from silence removal, not from timestamp granularity.

Resilience (ADR-122): this provider gains the same call + circuit + policy layers as the DGX
whisper / diarize providers — a bounded retry loop with a process-wide circuit breaker in serve
mode, and a backoff -> trip-after-N -> hold-and-probe ``ResiliencePolicy`` in reprocess mode. Prior
to ADR-122 this provider was bare (a single POST, no retry, no breaker) — strictly less protected
than whisper/diarize despite sharing the same DGX GPU contention risk.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from ... import config
from ...utils.log_redaction import format_exception_for_log
from .. import resilience
from ..resilience import CircuitBreaker, hardened_http_client, TimeoutLike
from ..resilience.policy import (
    FailureStrategy,
    ResilienceFuseOpenError,
    ResiliencePolicy,
    resolve_failure_strategy,
)

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8004
_DEFAULT_TIMEOUT_SEC = 1800.0
_RETRY_BACKOFF_SEC = 5.0

# Single-flight guard: MOSS shares the DGX GPU with the other self-hosted models and serves
# one inference at a time. Serialize our own calls (ADR-122, mirrors whisper's
# ``_dgx_single_flight`` / diarize's ``_dgx_diarize_single_flight``) so we never self-contend;
# external contention is ridden out by the timeout + watchdog, not by piling on more requests
# (#876).
_moss_single_flight = threading.Lock()

# Process-wide breaker for the MOSS endpoint (:8004). ADR-122: mirrors ``_whisper_breaker`` /
# ``_diarize_breaker`` — one hard timeout trips it immediately, otherwise two failures inside
# the window open it for a 5-minute cooldown before a half-open probe.
_moss_breaker = CircuitBreaker(
    failure_threshold=2, window_sec=300.0, cooldown_sec=300.0, name="moss"
)


def moss_base_url(host: str, port: int) -> str:
    """The MOSS service base URL for a tailnet ``host``:``port``."""
    return f"http://{host}:{port}"


class MossTranscriptionProvider:
    """Transcribe on the DGX MOSS service."""

    name = "moss"

    def __init__(self, cfg: config.Config) -> None:
        self.cfg = cfg
        self._host = (getattr(cfg, "dgx_tailnet_host", "") or "").strip()
        self._port = int(getattr(cfg, "moss_port", None) or _DEFAULT_PORT)
        self._model = (
            getattr(cfg, "moss_model", None) or "OpenMOSS-Team/MOSS-Transcribe-Diarize"
        ).strip()
        self._timeout = float(
            getattr(cfg, "moss_request_timeout_sec", None) or _DEFAULT_TIMEOUT_SEC
        )
        # Retry budget is generic; reuse the shared knob for parity with whisper/diarize.
        self._max_attempts = max(1, int(getattr(cfg, "dgx_max_attempts", 3)))
        # ADR-122: which resilience STRATEGY this provider uses. 'failover' (serve default) fails
        # fast and trips the breaker on the first hard timeout, raising for the wrapping
        # FallbackChain to advance; 'hold' routes through the ResiliencePolicy (backoff-retry ->
        # trip-after-N -> hold-and-probe). Standalone knob defaulted by run context, per-profile
        # overridable.
        self._strategy = resolve_failure_strategy(cfg)
        self._policy = ResiliencePolicy(
            breaker=_moss_breaker,
            retries_before_trip=int(getattr(cfg, "resilience_retries_before_trip", 3)),
            backoff_schedule_sec=tuple(
                getattr(cfg, "resilience_backoff_schedule_sec", (30.0, 60.0, 120.0))
            ),
            on_open_max_wait_sec=float(getattr(cfg, "resilience_on_open_max_wait_sec", 900.0)),
            probe_interval_sec=float(getattr(cfg, "resilience_probe_interval_sec", 30.0)),
            name="moss",
        )
        self._initialized = False

    def initialize(self) -> None:
        """Require the DGX tailnet host and mark the provider ready (no local model load)."""
        if not self._host:
            raise ValueError("dgx_tailnet_host is required for the moss provider")
        self._initialized = True

    def cleanup(self) -> None:
        """Release provider state (idempotent; no local resources held)."""
        self._initialized = False

    def _call_raw(self, audio_path: str) -> Dict[str, Any]:
        """The bare MOSS POST. ADR-122: never call directly outside ``_call``'s
        retry/breaker (serve mode) or ``ResiliencePolicy`` (reprocess mode) wrapping."""
        path = os.fspath(audio_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(audio_path)

        url = f"{moss_base_url(self._host, self._port)}/v1/transcribe"
        with open(path, "rb") as fh:
            files = {"file": (os.path.basename(path), fh, "application/octet-stream")}
            with hardened_http_client(self._timeout) as client:
                response = client.post(url, files=files)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"MOSS returned {type(payload).__name__}, expected an object")
        return payload

    def _call(self, audio_path: str) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()

        if self._strategy is FailureStrategy.HOLD:
            # ADR-122: consistency over availability — backoff-retry the SAME model, trip
            # only after N, hold-and-probe on a blown fuse. Never falls over to another
            # model (unlike the serve branch below, which raises for the FallbackChain).
            return self._call_via_moss_reprocess(audio_path)

        # ---- serve mode (ADR-122: fail fast, trip on the first hard timeout, raise for
        # the FallbackChain to advance — mirrors the DGX whisper/diarize serve branches) ----
        last_err: Optional[Exception] = None
        timed_out = False
        if not _moss_breaker.allow():
            reason = "moss_circuit_open"
        else:
            # Serialize MOSS calls (single GPU, serial server) so we never self-contend.
            with _moss_single_flight:
                for attempt in range(self._max_attempts):
                    try:
                        # Hard wall-clock watchdog: guarantees fail-over even when httpx's
                        # own timeout doesn't fire (a co-tenant GPU stall can make the
                        # multipart upload trickle indefinitely, mirrors #954).
                        result = resilience.run_with_watchdog(
                            lambda: self._call_raw(audio_path),
                            self._timeout + resilience.WATCHDOG_GRACE_SEC,
                            label="moss",
                        )
                        _moss_breaker.record_success()
                        return result
                    except TimeoutLike as exc:
                        # The GPU is busy/contended and the budget elapsed. Retrying would
                        # pile a duplicate request onto the already-overloaded server, so
                        # stop and fall back instead.
                        last_err = exc
                        timed_out = True
                        logger.warning(
                            "MOSS attempt %s timed out after %.0fs (GPU contended?); "
                            "falling back rather than re-queuing: %s",
                            attempt + 1,
                            self._timeout,
                            format_exception_for_log(exc),
                        )
                        break
                    except Exception as exc:
                        # Connection blip / transient server error — safe to retry with
                        # exponential backoff (no duplicate work is in flight).
                        last_err = exc
                        logger.warning(
                            "MOSS attempt %s failed: %s",
                            attempt + 1,
                            format_exception_for_log(exc),
                        )
                    if attempt < self._max_attempts - 1:
                        time.sleep(_RETRY_BACKOFF_SEC * (2**attempt))

            # A hard timeout trips the breaker immediately (one expensive wedge is
            # enough); other failures accrue toward the rolling-window threshold.
            _moss_breaker.record_failure(hard=timed_out)
            reason = format_exception_for_log(last_err) if last_err else "unknown_error"

        logger.warning("MOSS tier exhausted (%s); raising for the fallback chain", reason)
        if last_err is not None:
            raise last_err
        raise RuntimeError(f"MOSS unavailable: {reason}")

    def _call_via_moss_reprocess(self, audio_path: str) -> Dict[str, Any]:
        """ADR-122 reprocess-mode path: backoff-retry the chosen model, trip the fuse only
        after the policy threshold, and hold-and-probe (never fall over) on a blown fuse.

        Mirrors ``TailnetDgxWhisperTranscriptionProvider._transcribe_via_dgx_reprocess``.
        """
        try:
            with _moss_single_flight:
                return self._policy.run(
                    lambda: self._call_raw(audio_path), timeout_sec=self._timeout
                )
        except ResilienceFuseOpenError as exc:
            logger.error(
                "MOSS endpoint did not recover after %.0fs of pause-and-probe "
                "(reprocess mode, ADR-122) — alerting operator, NOT falling over: %s",
                self._policy.on_open_max_wait_sec,
                exc,
            )
            raise

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """Transcribe ``audio_path`` and return the plain text (segments dropped)."""
        return str(self._call(audio_path).get("text") or "").strip()

    def transcribe_with_segments(
        self,
        audio_path: str,
        language: Optional[str] = None,
        pipeline_metrics: Any = None,
        episode_duration_seconds: Optional[int] = None,
        call_metrics: Any = None,
    ) -> Tuple[Dict[str, object], float]:
        """Return ``({text, segments}, elapsed_seconds)``.

        Segments carry ``speaker`` as well as ``start``/``end``/``text`` — harmless to the
        transcription path, and it is what lets the diarization stage reuse this same inference.
        """
        started = time.perf_counter()
        payload = self._call(audio_path)
        elapsed = time.perf_counter() - started

        raw_segments = payload.get("segments")
        segments: List[dict] = (
            [s for s in raw_segments if isinstance(s, dict)]
            if isinstance(raw_segments, list)
            else []
        )
        text = str(payload.get("text") or "").strip()
        if not text and not segments:
            raise ValueError("MOSS returned neither text nor segments")

        logger.info(
            "MOSS transcribed %s in %.1fs (%d segments, %s speakers)",
            os.path.basename(str(audio_path)),
            elapsed,
            len(segments),
            payload.get("num_speakers", "?"),
        )
        return ({"text": text, "segments": segments}, elapsed)

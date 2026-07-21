"""DGX-hosted pyannote diarization — a pure DGX tier (RFC-106 / #1198).

Mirrors ``TailnetDgxWhisperTranscriptionProvider`` (#814): the laptop / prod VPS
uploads an audio file to a pyannote service running on DGX and gets back
diarization segments. On failure it RAISES; the stage factory's
``FallbackChainDiarizationProvider`` owns the ladder and fails over (DGX pyannote
-> local in-process pyannote -> deepgram). #926 originally self-wrapped straight
to local pyannote; RFC-106 moved that into the shared chain and added a cloud
tier for callers that cannot run pyannote in-process (e.g. a lightweight VPS).

Resilience (#954). The client carries the shared DGX defences from
``resilience.py`` — the same ones the #946 Whisper path uses — plus a circuit
breaker borrowed from the battle-tested RSS downloader (``rss/http_policy``):

- Duration-scaled request timeout, sized to diarization's real profile (pyannote
  is far faster than transcription, so the ceiling — and a breaker half-open
  probe — costs single-digit minutes, not the transcription provider's tens).
- Single-flight lock (the DGX GPU is serial; concurrent requests just queue).
- Bounded retries distinguishing a genuine timeout (fail over, don't re-queue)
  from a connection blip (retry with backoff).
- Hard wall-clock watchdog so a wedged request (httpx's own timeout has been seen
  to never fire under a co-tenant GPU stall, #954) always fails over.
- Circuit breaker: once DGX has failed, skip it for a cooldown and raise immediately
  (the chain fails over); a half-open probe re-tests so it self-heals on recovery.

Service contract (DGX-side, deploy.py / infra/dgx/pyannote-server/app.py):

- ``POST /v1/diarize`` — multipart form ``file=<audio>`` plus optional
  ``num_speakers`` / ``min_speakers`` / ``max_speakers`` form fields. Returns
  ``{"model_name": str, "num_speakers": int, "segments": [{"start", "end",
  "speaker"}, ...]}``.
- ``GET /health`` — 200 when pipeline is loaded.
- ``GET /v1/models`` — OpenAI-style envelope so the existing
  ``check_*_health`` helpers can probe it consistently.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, NoReturn, Optional

from ... import config
from ...providers.ml.diarization.base import (
    DiarizationResult,
    DiarizationSegment,
)
from ...utils.log_redaction import format_exception_for_log
from .. import guardrails, resilience
from ..resilience import CircuitBreaker, hardened_http_client, TimeoutLike
from ..resilience.policy import ResilienceFuseOpenError, ResiliencePolicy, RunContext
from .health import check_pyannote_diarize_health, dgx_diarize_base_url
from .telemetry import emit_dgx_fallback_breadcrumb

logger = logging.getLogger(__name__)

_RETRY_BACKOFF_SEC = 5.0

# Single-flight guard: the DGX pyannote server has one GPU and processes
# diarization serially. Serialize our own DGX diarize calls so we never
# self-contend; contention from *other* GPU tenants is ridden out by the
# duration-scaled timeout + watchdog + breaker, not by piling on more requests.
_dgx_diarize_single_flight = threading.Lock()

# Process-wide breaker for the DGX diarize endpoint (:8001). One hard timeout
# trips it immediately; otherwise two failures inside the window open it for a
# 5-minute cooldown before a half-open probe.
_diarize_breaker = CircuitBreaker(
    failure_threshold=2, window_sec=300.0, cooldown_sec=300.0, name="dgx-diarize"
)


class TailnetDgxDiarizationProvider:
    """Diarize on the DGX pyannote service. A pure DGX tier: raises on failure so the wrapping
    FallbackChain (RFC-106) can advance to the next tier."""

    def __init__(self, cfg: config.Config) -> None:
        self.cfg = cfg
        self._host = (cfg.dgx_tailnet_host or "").strip()
        self._port = int(getattr(cfg, "dgx_diarize_port", None) or 8001)
        self._model = (
            getattr(cfg, "dgx_diarize_model", None) or "pyannote/speaker-diarization-community-1"
        ).strip()
        # Diarization-specific timeout knobs: pyannote is far faster than Whisper
        # transcription, so the budget is much tighter (keeps a breaker half-open
        # probe cheap). Profile-tunable, defaults 180s + 6s/audio-min (#954).
        self._base_timeout_sec = float(
            getattr(cfg, "dgx_diarize_request_timeout_sec", None) or 180.0
        )
        self._timeout_per_audio_min = float(
            getattr(cfg, "dgx_diarize_timeout_per_audio_minute_sec", 6.0)
        )
        # Retry budget is generic; reuse the shared knob for parity with Whisper.
        self._max_attempts = max(1, int(getattr(cfg, "dgx_max_attempts", 3)))
        # ADR-119: which resilience behaviour this provider uses. 'serve' (default) keeps
        # today's fail-fast/trip/raise-for-chain logic below untouched; 'reprocess' routes
        # through the ResiliencePolicy (backoff-retry -> trip-after-N -> hold-and-probe).
        self._run_context = RunContext(getattr(cfg, "resilience_run_context", "serve"))
        self._policy = ResiliencePolicy(
            breaker=_diarize_breaker,
            retries_before_trip=int(getattr(cfg, "resilience_retries_before_trip", 3)),
            backoff_schedule_sec=tuple(
                getattr(cfg, "resilience_backoff_schedule_sec", (30.0, 60.0, 120.0))
            ),
            on_open_max_wait_sec=float(getattr(cfg, "resilience_on_open_max_wait_sec", 900.0)),
            probe_interval_sec=float(getattr(cfg, "resilience_probe_interval_sec", 30.0)),
            name="dgx-diarize",
        )
        self._initialized = False

    def initialize(self) -> None:
        """Validate host config and mark ready. RFC-106: this tier owns no fallback."""
        if self._initialized:
            return
        if not self._host:
            raise ValueError("dgx_tailnet_host is required for diarization_provider=tailnet_dgx")
        self._initialized = True

    def _effective_timeout_sec(self, audio_path: str) -> float:
        """Duration-scaled request timeout: base + per-audio-minute budget (#954)."""
        return resilience.effective_timeout_sec(
            self._base_timeout_sec,
            self._timeout_per_audio_min,
            resilience.probe_audio_duration_sec(audio_path),
        )

    def diarize(
        self,
        audio_path: str,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: int = 2,
        max_speakers: int = 20,
    ) -> DiarizationResult:
        """Diarize against DGX. Raise on failure so the wrapping FallbackChain can advance."""
        if not self._initialized:
            self.initialize()

        # Health-check matches the prefix portion of the model id so we don't
        # fail on minor naming drift between client config and server install.
        require_model_substring = self._model.rsplit("/", 1)[-1]
        timeout_sec = self._effective_timeout_sec(audio_path)

        if self._run_context is RunContext.REPROCESS:
            # ADR-119: consistency over availability — backoff-retry the SAME model,
            # trip only after N, hold-and-probe on a blown fuse. Kept as a fully separate
            # method (not folded into the serve loop below) so the serve branch's today
            # behaviour stays byte-for-byte unchanged (RFC-106/#1198 regression guard).
            return self._diarize_via_dgx_reprocess(
                audio_path,
                timeout_sec=timeout_sec,
                require_model_substring=require_model_substring,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

        # ---- serve mode (unchanged from today; RFC-106/#1198) ----
        # Circuit breaker: if DGX diarize is in its cooldown, don't even probe — raise immediately
        # so the FallbackChain fails over without a wasted probe (a wedged batch isn't paced by
        # timeouts).
        if not _diarize_breaker.allow():
            self._raise_exhausted("dgx_diarize_circuit_open")

        last_err: Optional[Exception] = None
        timed_out = False
        # Serialize our own DGX diarize calls (single GPU, serial server) so we
        # never self-contend; external contention is ridden out by the timeout +
        # watchdog inside ``_diarize_dgx_guarded``, not by piling on requests.
        with _dgx_diarize_single_flight:
            for attempt in range(self._max_attempts):
                try:
                    if not check_pyannote_diarize_health(
                        self._host,
                        port=self._port,
                        require_model_substring=require_model_substring,
                    ):
                        last_err = None  # health says unavailable; retry then fall back
                    else:
                        result = self._diarize_dgx_guarded(
                            audio_path,
                            timeout_sec,
                            num_speakers=num_speakers,
                            min_speakers=min_speakers,
                            max_speakers=max_speakers,
                        )
                        _diarize_breaker.record_success()
                        return result
                except TimeoutLike as exc:
                    # The GPU is busy/contended and the (generous, duration-scaled)
                    # budget elapsed. Retrying would pile a duplicate request onto the
                    # already-overloaded server, so stop and fall back instead.
                    last_err = exc
                    timed_out = True
                    logger.warning(
                        "DGX diarize attempt %s timed out after %.0fs (GPU contended?); "
                        "falling back rather than re-queuing: %s",
                        attempt + 1,
                        timeout_sec,
                        format_exception_for_log(exc),
                    )
                    break
                except guardrails.GuardrailViolation as exc:
                    # DGX returned a successful HTTP response but the content
                    # failed the structural sanity check (ADR-099, #999). Retry
                    # would likely return the same shape; fall back to local
                    # pyannote. Counted as a DGX failure (breaker records it).
                    last_err = exc
                    logger.warning(
                        "DGX diarize attempt %s returned guardrail-violating "
                        "response (reason=%s); falling back to local: %s",
                        attempt + 1,
                        exc.reason,
                        exc.response_summary,
                    )
                    break
                except Exception as exc:
                    # Connection blip / transient server error — safe to retry with
                    # exponential backoff (no duplicate work is in flight).
                    last_err = exc
                    logger.warning(
                        "DGX diarize attempt %s failed: %s",
                        attempt + 1,
                        format_exception_for_log(exc),
                    )
                if attempt < self._max_attempts - 1:
                    time.sleep(_RETRY_BACKOFF_SEC * (2**attempt))

        # A hard timeout trips the breaker immediately (one expensive wedge is
        # enough); other failures accrue toward the rolling-window threshold.
        _diarize_breaker.record_failure(hard=timed_out)
        reason = format_exception_for_log(last_err) if last_err else "health_check_failed"
        self._raise_exhausted(reason, last_err)

    def _raise_exhausted(self, reason: str, last_err: Optional[Exception] = None) -> NoReturn:
        """Emit the fallback breadcrumb and RAISE (RFC-106 / #1198).

        This tier no longer runs in-process pyannote itself — the FallbackChain that wraps it owns
        the ladder (DGX pyannote -> local pyannote -> deepgram) and decides whether to advance.
        ``is_infra_failure`` treats the errors raised here (timeouts, guardrail garbage, connection
        blips, circuit-open) as cascade-worthy, so the chain moves on to the next tier.
        """
        emit_dgx_fallback_breadcrumb(
            stage="diarization",
            model=self._model,
            failure_reason=reason,
        )
        logger.warning("DGX diarize tier exhausted (%s); raising for the fallback chain", reason)
        if last_err is not None:
            raise last_err
        raise RuntimeError(f"DGX diarize unavailable: {reason}")

    def _diarize_via_dgx_reprocess(
        self,
        audio_path: str,
        *,
        timeout_sec: float,
        require_model_substring: str,
        num_speakers: Optional[int],
        min_speakers: int,
        max_speakers: int,
    ) -> DiarizationResult:
        """ADR-119 reprocess-mode path: backoff-retry the chosen model, trip the fuse only
        after the policy threshold, and hold-and-probe (never fall over) on a blown fuse.

        Mirrors ``TailnetDgxWhisperTranscriptionProvider._transcribe_via_dgx_reprocess``. The
        single-flight lock spans the whole call (health-check + retries + any pause-and-probe),
        so nothing else in this process piles a second request onto the DGX diarize endpoint
        while we're waiting on it either.
        """

        def _attempt() -> DiarizationResult:
            if not check_pyannote_diarize_health(
                self._host,
                port=self._port,
                require_model_substring=require_model_substring,
            ):
                raise RuntimeError("dgx_diarize_health_check_failed")
            return self._diarize_dgx(
                audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                timeout_sec=timeout_sec,
            )

        try:
            with _dgx_diarize_single_flight:
                return self._policy.run(_attempt, timeout_sec=timeout_sec)
        except ResilienceFuseOpenError as exc:
            emit_dgx_fallback_breadcrumb(
                stage="diarization",
                model=self._model,
                failure_reason=str(exc),
            )
            logger.error(
                "DGX diarize endpoint did not recover after %.0fs of pause-and-probe "
                "(reprocess mode, ADR-119) — alerting operator, NOT falling over: %s",
                self._policy.on_open_max_wait_sec,
                exc,
            )
            raise

    def _diarize_dgx_guarded(
        self,
        audio_path: str,
        timeout_sec: float,
        *,
        num_speakers: Optional[int],
        min_speakers: int,
        max_speakers: int,
    ) -> DiarizationResult:
        """Run the DGX diarize POST under a hard wall-clock deadline (#954)."""
        return resilience.run_with_watchdog(
            lambda: self._diarize_dgx(
                audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                timeout_sec=timeout_sec,
            ),
            timeout_sec + resilience.WATCHDOG_GRACE_SEC,
            label="dgx-diarize",
        )

    def _diarize_dgx(
        self,
        audio_path: str,
        *,
        num_speakers: Optional[int],
        min_speakers: int,
        max_speakers: int,
        timeout_sec: Optional[float] = None,
    ) -> DiarizationResult:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx required for tailnet_dgx diarize") from exc

        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(audio_path)

        base = dgx_diarize_base_url(self._host, self._port)
        url = f"{base}/v1/diarize"
        # Explicit per-phase timeout (defence-in-depth with the watchdog): bound
        # connect tightly, give read/write the generous duration-scaled budget.
        ttl = float(timeout_sec or self._base_timeout_sec)
        timeout = httpx.Timeout(ttl, connect=15.0)
        with path.open("rb") as audio_file:
            files = {"file": (path.name, audio_file, "application/octet-stream")}
            data: dict[str, Any] = {
                "min_speakers": str(min_speakers),
                "max_speakers": str(max_speakers),
            }
            if num_speakers is not None:
                data["num_speakers"] = str(num_speakers)
            with hardened_http_client(timeout, subsystem="dgx_diarize") as client:
                resp = client.post(url, data=data, files=files)
        resp.raise_for_status()
        payload = resp.json()
        raw_segments = payload.get("segments") or []
        segments = [
            DiarizationSegment(
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                speaker=str(s.get("speaker", "SPEAKER_UNKNOWN")),
            )
            for s in raw_segments
            if isinstance(s, dict)
        ]
        # Response-shape guardrail (ADR-099, #999): empty segments for non-trivial
        # audio is structurally invalid (every non-empty audio has at least one
        # speech segment). Replaces the narrower "if not segments: raise ValueError"
        # check that was here before — the guardrail raises GuardrailViolation
        # which the caller treats as a sibling of TimeoutLike (DGX fails, breaker
        # counts, in-process pyannote fallback fires). Preventive — no observed
        # cases of this failure mode in production yet.
        audio_duration_sec = resilience.probe_audio_duration_sec(audio_path)
        guardrails.check_pyannote_response(segments, audio_duration_sec=audio_duration_sec)
        return DiarizationResult(
            segments=segments,
            num_speakers=int(payload.get("num_speakers") or len({s.speaker for s in segments})),
            model_name=str(payload.get("model_name") or self._model),
        )

"""MOSS diarization provider — the speaker half of the joint model (#1177).

MOSS emits transcript + speakers + timestamps in one pass, so this provider and the MOSS
*transcription* provider read the same inference. The service caches its last few results by audio
digest, so whichever stage runs second pays nothing. That preserves the stage independence the
pipeline is built on (the shape Deepgram established — two calls, no response threaded across the
provider-interface boundary) without running a model twice.

The speaker labels MOSS returns (``S01``, ``S02``, …) are **anonymous and relative** — exactly the
semantics of pyannote's ``SPEAKER_00``. They are normalized to ``SPEAKER_NN`` here so every
downstream consumer (roster resolution, the #1167 placeholder guard, host/guest attribution) keeps
working unchanged, whichever diarizer produced them.

Resilience (ADR-122): like the MOSS transcription provider, this gains the call + circuit + policy
layers of the self-hosted-model family — a bounded retry loop with a process-wide circuit breaker
in serve mode, and a backoff -> trip-after-N -> hold-and-probe ``ResiliencePolicy`` in reprocess
mode. It was bare before ADR-122 (a single POST, no retry, no breaker).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from .... import config
from ....utils.log_redaction import format_exception_for_log
from ... import resilience
from ...resilience import CircuitBreaker, hardened_http_client, TimeoutLike
from ...resilience.policy import (
    FailureStrategy,
    ResilienceFuseOpenError,
    ResiliencePolicy,
    resolve_failure_strategy,
)
from .base import DiarizationResult, DiarizationSegment

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8004
_DEFAULT_TIMEOUT_SEC = 1800.0
_RETRY_BACKOFF_SEC = 5.0

# Single-flight guard: MOSS shares the DGX GPU and serves one inference at a time. Serialize our
# own calls (ADR-122, mirrors whisper's ``_dgx_single_flight``) so we never self-contend.
_moss_diarize_single_flight = threading.Lock()

# Process-wide breaker for the MOSS endpoint (:8004), diarization side. Its own instance —
# consistent with whisper/diarize each owning a breaker — mirrors ``_moss_breaker``'s tuning.
_moss_diarize_breaker = CircuitBreaker(
    failure_threshold=2, window_sec=300.0, cooldown_sec=300.0, name="moss-diarize"
)


def _normalize_speaker(label: str) -> str:
    """``S01`` -> ``SPEAKER_01``.

    Downstream code — the roster, the ``SPEAKER_NN`` placeholder predicate (#1167), the host/guest
    rules — keys off pyannote's naming. Normalizing here means MOSS is a drop-in for pyannote and
    nothing downstream has to learn a second dialect.
    """
    text = str(label or "").strip()
    if text.upper().startswith("S") and text[1:].isdigit():
        return f"SPEAKER_{int(text[1:]):02d}"
    return text or "SPEAKER_00"


class MossDiarizationProvider:
    """Diarize on the DGX MOSS service."""

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
        self._max_attempts = max(1, int(getattr(cfg, "dgx_max_attempts", 3)))
        # ADR-122: 'failover' (serve default) fails fast + trips on the first hard timeout, raising
        # for a wrapping FallbackChain; 'hold' routes through the ResiliencePolicy (backoff-retry ->
        # trip-after-N -> hold-and-probe, never a cross-model fallover). Standalone strategy knob
        # defaulted by run context, per-profile overridable.
        self._strategy = resolve_failure_strategy(cfg)
        self._policy = ResiliencePolicy(
            breaker=_moss_diarize_breaker,
            retries_before_trip=int(getattr(cfg, "resilience_retries_before_trip", 3)),
            backoff_schedule_sec=tuple(
                getattr(cfg, "resilience_backoff_schedule_sec", (30.0, 60.0, 120.0))
            ),
            on_open_max_wait_sec=float(getattr(cfg, "resilience_on_open_max_wait_sec", 900.0)),
            probe_interval_sec=float(getattr(cfg, "resilience_probe_interval_sec", 30.0)),
            name="moss-diarize",
        )
        self._initialized = False

    def initialize(self) -> None:
        """Require the DGX tailnet host and mark the provider ready (no local model load)."""
        if not self._host:
            raise ValueError("dgx_tailnet_host is required for the moss diarization provider")
        self._initialized = True

    def cleanup(self) -> None:
        """Release provider state (idempotent; no local resources held)."""
        self._initialized = False

    def _call_raw(self, audio_path: str) -> Dict[str, Any]:
        """The bare MOSS POST. ADR-122: never call directly outside ``_call``'s retry/breaker
        (serve mode) or ``ResiliencePolicy`` (reprocess mode) wrapping."""
        path = os.fspath(audio_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(audio_path)
        url = f"http://{self._host}:{self._port}/v1/transcribe"
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
            return self._diarize_via_moss_reprocess(audio_path)

        # ---- serve mode: fail fast, trip on the first hard timeout, raise for the FallbackChain
        # (mirrors the DGX whisper/diarize + MOSS-transcription serve branches) ----
        last_err: Optional[Exception] = None
        timed_out = False
        if not _moss_diarize_breaker.allow():
            reason = "moss_diarize_circuit_open"
        else:
            with _moss_diarize_single_flight:
                for attempt in range(self._max_attempts):
                    try:
                        result = resilience.run_with_watchdog(
                            lambda: self._call_raw(audio_path),
                            self._timeout + resilience.WATCHDOG_GRACE_SEC,
                            label="moss-diarize",
                        )
                        _moss_diarize_breaker.record_success()
                        return result
                    except TimeoutLike as exc:
                        last_err = exc
                        timed_out = True
                        logger.warning(
                            "MOSS diarize attempt %s timed out after %.0fs (GPU contended?); "
                            "falling back rather than re-queuing: %s",
                            attempt + 1,
                            self._timeout,
                            format_exception_for_log(exc),
                        )
                        break
                    except Exception as exc:
                        last_err = exc
                        logger.warning(
                            "MOSS diarize attempt %s failed: %s",
                            attempt + 1,
                            format_exception_for_log(exc),
                        )
                    if attempt < self._max_attempts - 1:
                        time.sleep(_RETRY_BACKOFF_SEC * (2**attempt))

            _moss_diarize_breaker.record_failure(hard=timed_out)
            reason = format_exception_for_log(last_err) if last_err else "unknown_error"

        logger.warning("MOSS diarize tier exhausted (%s); raising for the fallback chain", reason)
        if last_err is not None:
            raise last_err
        raise RuntimeError(f"MOSS diarization unavailable: {reason}")

    def _diarize_via_moss_reprocess(self, audio_path: str) -> Dict[str, Any]:
        """ADR-122 reprocess-mode path: backoff-retry the chosen model, trip only after the
        policy threshold, hold-and-probe (never fall over) on a blown fuse. Mirrors the MOSS
        transcription provider's ``_call_via_moss_reprocess``."""
        try:
            with _moss_diarize_single_flight:
                return self._policy.run(
                    lambda: self._call_raw(audio_path), timeout_sec=self._timeout
                )
        except ResilienceFuseOpenError as exc:
            logger.error(
                "MOSS endpoint did not recover after %.0fs of pause-and-probe (reprocess mode, "
                "ADR-122) — alerting operator, NOT falling over: %s",
                self._policy.on_open_max_wait_sec,
                exc,
            )
            raise

    def diarize(
        self,
        audio_path: str,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: int = 2,
        max_speakers: int = 20,
    ) -> DiarizationResult:
        """Speaker turns from MOSS.

        ``num_speakers`` / ``min_speakers`` / ``max_speakers`` are accepted for interface
        compatibility but **ignored**: MOSS decides the speaker count autoregressively and exposes
        no knob to constrain it. Callers that rely on bounding the count (pyannote honours these)
        will not get that behaviour here — worth knowing when comparing DER against a diarizer that
        was given the true count.
        """
        started = time.perf_counter()
        payload = self._call(audio_path)
        elapsed = time.perf_counter() - started

        raw_segments = payload.get("segments") if isinstance(payload, dict) else None
        segments: List[DiarizationSegment] = []
        for seg in raw_segments or []:
            if not isinstance(seg, dict):
                continue
            try:
                start, end = float(seg["start"]), float(seg["end"])
            except (KeyError, TypeError, ValueError):
                continue
            segments.append(
                DiarizationSegment(
                    start=start, end=end, speaker=_normalize_speaker(str(seg.get("speaker", "")))
                )
            )

        voices = {s.speaker for s in segments}
        logger.info(
            "MOSS diarized %s in %.1fs (%d turns, %d voices)",
            os.path.basename(os.fspath(audio_path)),
            elapsed,
            len(segments),
            len(voices),
        )
        return DiarizationResult(
            segments=segments,
            num_speakers=len(voices),
            model_name=str((payload or {}).get("model") or self._model),
        )

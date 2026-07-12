"""DGX-hosted Whisper via faster-whisper-server with mandatory cloud fallback.

Architecture (RFC-089 / ADR-096 / #814):

- Whisper service on DGX: faster-whisper-server (#814), OpenAI-compatible,
  listening on ``:8000``. Installed by ``infra/dgx/converge/deploy.py`` via
  pyinfra. Speaks ``POST /v1/audio/transcriptions`` with multipart audio.
- Fallback: ``transcription_fallback_provider`` (default ``openai``) when
  DGX is unhealthy or the request fails. Mandatory per ADR-096 — no
  hard-required-DGX paths.

Pre-#814 history: this provider targeted ``POST /api/transcribe`` on Ollama's
port ``:11434``. Ollama doesn't actually serve Whisper — that endpoint never
existed in production. The fallback covered for it. Post-#814 the service
exists and the endpoint + port reflect that.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, cast, List, Optional

from ... import config
from ...transcription.base import TranscriptionProvider
from ...transcription.factory import create_transcription_provider
from ...utils.log_redaction import format_exception_for_log
from .. import guardrails, resilience
from ..resilience import CircuitBreaker, hardened_http_client, TimeoutLike
from .health import check_faster_whisper_health, dgx_whisper_base_url
from .telemetry import emit_dgx_fallback_breadcrumb

logger = logging.getLogger(__name__)

_RETRY_BACKOFF_SEC = 5.0

# Single-flight guard: the DGX faster-whisper server has one GPU and processes
# transcriptions serially. Sending concurrent requests just queues them server-side
# and makes every client wait (and risk a false timeout). Serialize DGX calls within
# this process so we never self-contend; a busy GPU from *other* workloads is ridden
# out by the duration-scaled timeout + watchdog, not by piling on more requests (#876).
_dgx_single_flight = threading.Lock()

# Process-wide breaker for the DGX Whisper endpoint (:8002). One hard timeout trips
# it immediately; otherwise two failures inside the window open it for a 5-minute
# cooldown before a half-open probe, so a wedged batch isn't paced by per-episode
# timeouts but still self-heals when DGX recovers (#954).
_whisper_breaker = CircuitBreaker(
    failure_threshold=2, window_sec=300.0, cooldown_sec=300.0, name="dgx-whisper"
)


def _refine_segment_times(
    segments: list[dict[str, object]], flat_words: object
) -> list[dict[str, object]]:
    """Reset segment start/end from word-level times (#1173).

    faster-whisper-server may return the words flat alongside the segments (the OpenAI shape) or
    nested inside each segment, depending on version — accept either, and return the segments
    untouched when the server returned no word timestamps at all.
    """
    from podcast_scraper.transcription.word_timestamps import (
        apply_nested_word_timestamps,
        apply_word_timestamps,
        word_dicts,
    )

    words = word_dicts(flat_words)
    if words:
        return cast(List[dict[str, object]], apply_word_timestamps(segments, words))
    if any(isinstance(s.get("words"), list) for s in segments):
        return cast(List[dict[str, object]], apply_nested_word_timestamps(segments))
    return segments


class TailnetDgxWhisperTranscriptionProvider:
    """Transcribe on DGX faster-whisper-server; fall back to cloud on failure."""

    def __init__(self, cfg: config.Config) -> None:
        """Store config and DGX connection parameters."""
        self.cfg = cfg
        self._host = (cfg.dgx_tailnet_host or "").strip()
        # #814: separate from dgx_ollama_port (11434) because faster-whisper-server
        # is a different service on a different port.
        self._port = int(getattr(cfg, "dgx_whisper_port", None) or 8000)
        self._model = (cfg.dgx_whisper_model or "Systran/faster-whisper-large-v3").strip()
        self._timeout_sec = float(cfg.dgx_request_timeout_sec or 600.0)
        self._timeout_per_audio_min = float(getattr(cfg, "dgx_timeout_per_audio_minute_sec", 20.0))
        self._max_attempts = max(1, int(getattr(cfg, "dgx_max_attempts", 3)))
        self._fallback_name = (cfg.transcription_fallback_provider or "openai").strip()
        self._fallback: Optional[TranscriptionProvider] = None
        self._initialized = False

    def initialize(self) -> None:
        """Load fallback transcription provider (cloud) per ADR-096."""
        if self._initialized:
            return
        if not self._host:
            raise ValueError("dgx_tailnet_host is required for tailnet_dgx_whisper")
        fb_data = self.cfg.model_dump()
        fb_data["transcription_provider"] = self._fallback_name
        fb_cfg = config.Config.model_validate(fb_data)
        self._fallback = create_transcription_provider(fb_cfg)
        self._fallback.initialize()
        self._initialized = True

    def _ensure_init(self) -> None:
        if not self._initialized:
            self.initialize()

    def _effective_timeout_sec(self, episode_duration_seconds: int | None) -> float:
        """Duration-scaled request timeout: base + per-audio-minute budget.

        A flat timeout false-fails long episodes whenever the shared GPU is briefly
        contended. Scaling the budget by audio length lets the transcription wait the
        contention out instead of bailing to the cloud fallback (#876).
        """
        base = self._timeout_sec
        if (
            episode_duration_seconds
            and episode_duration_seconds > 0
            and self._timeout_per_audio_min
        ):
            base += (float(episode_duration_seconds) / 60.0) * self._timeout_per_audio_min
        return base

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """Return transcript text, using DGX or fallback provider."""
        self._ensure_init()
        text, _segments, _dur, _actual = self._transcribe_with_fallback(audio_path, language)
        return text

    def transcribe_with_segments(
        self,
        audio_path: str,
        language: str | None = None,
        # The workflow passes these for metrics + chunked-call accounting on
        # cloud providers. Speaches doesn't use them directly (we have our own
        # breadcrumb emission via emit_dgx_fallback_breadcrumb), but accepting
        # them keeps the signature compatible with the rest of the provider
        # protocol so the workflow can pass them uniformly. When fallback
        # fires, we forward them to the cloud provider so its metrics +
        # cost tracking still work end-to-end.
        pipeline_metrics: Any | None = None,
        episode_duration_seconds: int | None = None,
        call_metrics: Any | None = None,
        # #1046 — per-call model override. When set, this overrides
        # ``cfg.dgx_whisper_model`` for THIS call only. Used by the
        # sniff-pass orchestrator (when it lands) to transcribe with a
        # cheap model first, then re-call with the deep model only when
        # the gate fires. Passing None (the default) preserves single-
        # model behaviour. The faster-whisper-server container must have
        # the override model loaded in its cache (preload via env or
        # warm via a no-op call before the first real transcription).
        model_override: str | None = None,
    ) -> tuple[dict[str, object], float]:
        """Return transcript dict with segments and elapsed seconds."""
        self._ensure_init()
        text, segments, duration, actual_model = self._transcribe_with_fallback(
            audio_path,
            language,
            pipeline_metrics=pipeline_metrics,
            episode_duration_seconds=episode_duration_seconds,
            call_metrics=call_metrics,
            model_override=model_override,
        )
        # #1046 provenance: record what was REQUESTED (the override or default,
        # for the gate orchestrator's bookkeeping) AND what actually ran
        # (post-fallback). Pre-fix these were conflated, so a fallback call
        # falsely reported the DGX model in ``model_used`` (#1046 deep-review).
        return (
            {
                "text": text,
                "segments": segments,
                "language": language or "en",
                "model_requested": (model_override or self._model),
                "model_used": actual_model,
            },
            duration,
        )

    def _transcribe_with_fallback(
        self,
        audio_path: str,
        language: str | None,
        # Forwarded to the fallback provider when DGX is unhealthy and we
        # route to cloud Whisper. DGX-side call doesn't use them.
        pipeline_metrics: Any | None = None,
        episode_duration_seconds: int | None = None,
        call_metrics: Any | None = None,
        # #1046 — propagated to _transcribe_dgx. When set, overrides the
        # DGX-side model for this call only. None (default) keeps
        # ``self._model`` (the prod default).
        model_override: str | None = None,
    ) -> tuple[str, list[dict[str, object]], float, str]:
        assert self._fallback is not None
        last_err: Optional[Exception] = None
        timed_out = False
        # Effective model for THIS call — the override wins when present, else
        # the provider's configured default. The health-check substring is
        # derived from the effective model so we probe for the right loaded
        # variant when the sniff-pass workflow uses a different repo id.
        effective_model = (model_override or self._model).strip() or self._model
        # Health-check substring matches the model's slug portion (after the
        # ``Systran/`` namespace) so we don't fail on small repo-id variations.
        health_substring = effective_model.rsplit("/", 1)[-1]
        timeout_sec = self._effective_timeout_sec(episode_duration_seconds)
        # Circuit breaker: if DGX Whisper is in its cooldown, skip it entirely and
        # go straight to the cloud fallback — a wedged batch isn't paced by timeouts.
        if not _whisper_breaker.allow():
            reason = "dgx_whisper_circuit_open"
        else:
            # Serialize DGX calls (single GPU, serial server) so we never self-contend.
            with _dgx_single_flight:
                for attempt in range(self._max_attempts):
                    try:
                        if not check_faster_whisper_health(
                            self._host,
                            port=self._port,
                            require_model_substring=health_substring,
                        ):
                            last_err = None  # health says unavailable; retry then fall back
                        else:
                            # Hard wall-clock watchdog: guarantees fail-over even when
                            # httpx's own timeout doesn't fire (a co-tenant GPU stall can
                            # make the multipart upload trickle indefinitely, #954).
                            result_dgx = resilience.run_with_watchdog(
                                lambda: self._transcribe_dgx(
                                    audio_path,
                                    language,
                                    timeout_sec,
                                    model_override=model_override,
                                ),
                                timeout_sec + resilience.WATCHDOG_GRACE_SEC,
                                label="dgx-whisper",
                            )
                            _whisper_breaker.record_success()
                            # DGX path won: the model that actually ran IS
                            # the effective_model (override-or-default).
                            text_dgx, segments_dgx, duration_dgx = result_dgx
                            return (text_dgx, segments_dgx, duration_dgx, effective_model)
                    except TimeoutLike as exc:
                        # The GPU is busy/contended and the (generous, duration-scaled)
                        # budget elapsed. Retrying would pile a duplicate request onto the
                        # already-overloaded server, so stop and fall back instead.
                        last_err = exc
                        timed_out = True
                        logger.warning(
                            "DGX Whisper attempt %s timed out after %.0fs (GPU contended?); "
                            "falling back rather than re-queuing: %s",
                            attempt + 1,
                            timeout_sec,
                            format_exception_for_log(exc),
                        )
                        break
                    except guardrails.GuardrailViolation as exc:
                        # DGX returned a successful HTTP response but the content
                        # failed the structural sanity check (ADR-099, #999) —
                        # e.g. WER=1.0 garbage transcript under GPU contention.
                        # Retry would likely return the same garbage; fall back
                        # to cloud. Counted as a DGX failure (breaker records it).
                        last_err = exc
                        logger.warning(
                            "DGX Whisper attempt %s returned guardrail-violating "
                            "response (reason=%s); falling back to cloud: %s",
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
                            "DGX Whisper attempt %s failed: %s",
                            attempt + 1,
                            format_exception_for_log(exc),
                        )
                    if attempt < self._max_attempts - 1:
                        time.sleep(_RETRY_BACKOFF_SEC * (2**attempt))

            # A hard timeout trips the breaker immediately (one expensive wedge is
            # enough); other failures accrue toward the rolling-window threshold.
            _whisper_breaker.record_failure(hard=timed_out)
            reason = format_exception_for_log(last_err) if last_err else "health_check_failed"

        emit_dgx_fallback_breadcrumb(
            stage="transcription",
            model=self._model,
            failure_reason=reason,
        )
        logger.warning(
            "Falling back from DGX Whisper to %s (%s)",
            self._fallback_name,
            reason,
        )
        # Forward metrics kwargs so the cloud fallback's cost + provider metrics
        # tracking still works when DGX is denied / unhealthy.
        result = self._fallback.transcribe_with_segments(
            audio_path,
            language,
            pipeline_metrics=pipeline_metrics,
            episode_duration_seconds=episode_duration_seconds,
            call_metrics=call_metrics,
        )
        raw_segments = result[0].get("segments") or []
        segments = cast(List[dict[str, object]], raw_segments)
        # Fallback path won: attribute the actual model to the cloud provider.
        # If the cloud provider populated ``model_used`` in its own dict we
        # surface it; otherwise we mark the model as the fallback provider's
        # default. Either way the answer is NOT the DGX model that wasn't used.
        cloud_model = result[0].get("model_used")
        actual_model = (
            f"{self._fallback_name}:{cloud_model}"
            if cloud_model
            else f"{self._fallback_name}:default"
        )
        return (
            str(result[0].get("text", "")),
            segments,
            float(result[1]),
            actual_model,
        )

    def cleanup(self) -> None:
        """Release fallback provider resources."""
        if self._fallback is not None:
            self._fallback.cleanup()

    def _transcribe_dgx(
        self,
        audio_path: str,
        language: str | None,
        timeout_sec: Optional[float] = None,
        # #1046 — per-call model override (e.g. sniff-pass uses ``small.en``).
        # None preserves the configured default.
        model_override: str | None = None,
    ) -> tuple[str, list[dict[str, object]], float]:
        """Call faster-whisper-server's OpenAI-compatible transcribe endpoint.

        The server speaks ``POST /v1/audio/transcriptions`` with multipart form
        data. Setting ``response_format=verbose_json`` gets us segments in
        addition to the flat text (we need segments for downstream stages —
        speaker assignment, screenplay, etc.).
        """
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(audio_path)

        base = dgx_whisper_base_url(self._host, self._port)
        url = f"{base}/v1/audio/transcriptions"
        started = time.perf_counter()
        with path.open("rb") as audio_file:
            files = {"file": (path.name, audio_file, "application/octet-stream")}
            # #1046 — use the per-call override when set, otherwise default.
            effective_model = (model_override or self._model).strip() or self._model
            data: dict[str, Any] = {
                "model": effective_model,
                "response_format": "verbose_json",
                # Segment-level times drift on long audio; word-level ones don't (#1173). If the
                # server ignores this, the refinement below simply no-ops.
                "timestamp_granularities[]": ["word", "segment"],
            }
            if language:
                data["language"] = language
            with hardened_http_client(timeout_sec or self._timeout_sec) as client:
                resp = client.post(url, data=data, files=files)
        resp.raise_for_status()
        payload = resp.json()
        text = str(payload.get("text") or "").strip()
        # Response-shape guardrail (ADR-099, #999) — catches both the empty-
        # response case AND the WER=1.0 garbage-content case observed in #996.
        # Replaces the narrower "if not text: raise ValueError" pattern that
        # was here before; the guardrail raises GuardrailViolation, which the
        # caller treats as a sibling of TimeoutLike (DGX fails, breaker counts,
        # cloud fallback fires).
        audio_duration_sec = resilience.probe_audio_duration_sec(audio_path)
        guardrails.check_whisper_response(text, audio_duration_sec=audio_duration_sec)
        duration = float(time.perf_counter() - started)
        segments: list[dict[str, object]] = []
        if isinstance(payload.get("segments"), list):
            segments = [s for s in payload["segments"] if isinstance(s, dict)]
            segments = _refine_segment_times(segments, payload.get("words"))
        return text, segments, duration

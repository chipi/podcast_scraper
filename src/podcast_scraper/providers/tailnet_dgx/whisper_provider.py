"""DGX-hosted Whisper via faster-whisper-server with mandatory cloud fallback.

Architecture (RFC-089 / ADR-096 / #814):

- Whisper service on DGX: faster-whisper-server (#814), OpenAI-compatible,
  listening on ``:8000``. Installed by ``infra/dgx/converge/deploy.py`` via
  pyinfra. Speaks ``POST /v1/audio/transcriptions`` with multipart audio.
- Fallback: owned by the stage factory's ``FallbackChainTranscriptionProvider``
  (RFC-106 / #1198), not by this provider. This tier tries DGX and RAISES on
  failure; the chain advances to the next tier (DGX-whisper -> cloud). ADR-096's
  "no hard-required-DGX path" still holds — it is just enforced one layer up now.

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
from ...transcription import punctuation
from ...utils.log_redaction import format_exception_for_log
from .. import guardrails, resilience
from ..resilience import CircuitBreaker, hardened_http_client, TimeoutLike
from ..resilience.policy import (
    FailureStrategy,
    ResilienceFuseOpenError,
    ResiliencePolicy,
    resolve_failure_strategy,
)
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
#: A clip is a gap (seconds) or a punctuation window (10 minutes: 127-155 s on the DGX, V.6b
#: pt-PT 2026-10-08); a request still waiting after this is not coming back usefully.
CLIP_TIMEOUT_SEC = 600.0

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
    """Transcribe on DGX faster-whisper-server. A pure DGX tier: raises on failure so the wrapping
    FallbackChain (RFC-106) can advance to the next tier."""

    def __init__(self, cfg: config.Config) -> None:
        """Store config and DGX connection parameters."""
        self.cfg = cfg
        self._host = (cfg.dgx_tailnet_host or "").strip()
        # #814: separate from dgx_ollama_port (11434) because faster-whisper-server
        # is a different service on a different port.
        self._port = int(getattr(cfg, "dgx_whisper_port", None) or 8000)
        self._model = (cfg.dgx_whisper_model or "Systran/faster-whisper-large-v3").strip()
        #: The language the SERVER reported on the most recent transcribe call (S0.6, #2177).
        #: Initialized here so a call that fails before reaching the transport leaves an honest
        #: None rather than raising AttributeError on the result-dict read.
        self._last_detected_language: str | None = None
        self._timeout_sec = float(cfg.dgx_request_timeout_sec or 600.0)
        self._timeout_per_audio_min = float(getattr(cfg, "dgx_timeout_per_audio_minute_sec", 20.0))
        self._max_attempts = max(1, int(getattr(cfg, "dgx_max_attempts", 3)))
        # ADR-122: which resilience STRATEGY this provider uses. 'failover' (serve default) keeps
        # today's fail-fast/trip/raise-for-chain logic below untouched; 'hold' routes through the
        # ResiliencePolicy (backoff-retry -> trip-after-N -> hold-and-probe). The strategy is a
        # standalone knob defaulted by run context (reprocess -> hold), overridable per profile.
        self._strategy = resolve_failure_strategy(cfg)
        self._policy = ResiliencePolicy(
            breaker=_whisper_breaker,
            retries_before_trip=int(getattr(cfg, "resilience_retries_before_trip", 3)),
            backoff_schedule_sec=tuple(
                getattr(cfg, "resilience_backoff_schedule_sec", (30.0, 60.0, 120.0))
            ),
            on_open_max_wait_sec=float(getattr(cfg, "resilience_on_open_max_wait_sec", 900.0)),
            probe_interval_sec=float(getattr(cfg, "resilience_probe_interval_sec", 30.0)),
            name="dgx-whisper",
        )
        self._initialized = False
        # #2284: when the punctuation prompt is sent (see Config.dgx_whisper_punctuation_prompt)
        # and what the last call on THIS thread did about it -- read by transcribe_with_segments.
        self._punctuation_mode = str(getattr(cfg, "dgx_whisper_punctuation_prompt", "on_retry"))
        self._local = threading.local()

    def initialize(self) -> None:
        """Mark the DGX Whisper tier ready.

        RFC-106 (#1198): this provider is now a **pure DGX tier**. It no longer builds or owns a
        cloud fallback — the stage factory wraps it in a ``FallbackChainTranscriptionProvider`` and
        the chain owns the failover ladder. On a DGX failure this tier RAISES (classified); the
        chain decides whether to advance. Retiring the self-wrap is what stops the double-fallback
        that promoting MOSS above whisper would otherwise create.
        """
        if self._initialized:
            return
        if not self._host:
            raise ValueError("dgx_tailnet_host is required for tailnet_dgx_whisper")
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
        text, _segments, _dur, _actual = self._transcribe_via_dgx(audio_path, language)
        return text

    def transcribe_clip(
        self, audio_path: str, language: str | None = None, prompt: str | None = None
    ) -> dict[str, Any]:
        """#2187 A2: transcribe a few seconds cut from an episode — ONE request, nothing else.

        A gap clip is not an episode, and "few or no words" is a real answer for one (music, a
        breath, a short remark). Through ``transcribe_with_segments`` that answer failed the
        episode-scale length floor, was retried on the same model, tripped the breaker and then
        held the process-wide single-flight lock through up to 900 s of pause-and-probe (V.6b
        French feed, 2026-10-08: 3 words for a 5.5 s clip -> "circuit breaker OPEN"). So: no
        response guardrail, no retry policy, no breaker, no punctuation retry. The lock is still
        taken per request so a clip never piles onto an episode in flight. A transport error
        propagates; the caller records the clip as failed and keeps the transcript.

        ``prompt`` is Whisper's initial prompt (a punctuation-window repair sends one).
        """
        self._ensure_init()
        with _dgx_single_flight:
            text, segments, _duration = self._transcribe_dgx(
                audio_path,
                language,
                CLIP_TIMEOUT_SEC,
                prompt=prompt,
                check_response=False,
                record_language=False,
            )
        return {"text": text, "segments": segments}

    def transcribe_with_segments(
        self,
        audio_path: str,
        language: str | None = None,
        # The workflow passes these for metrics + chunked-call accounting on
        # cloud providers. Speaches doesn't use them directly (we have our own
        # breadcrumb emission via emit_dgx_fallback_breadcrumb), but accepting
        # them keeps the signature compatible with the rest of the provider
        # protocol so the workflow can pass them uniformly. RFC-106: the wrapping
        # FallbackChain forwards them to whichever tier ultimately serves the call.
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
        text, segments, duration, actual_model = self._transcribe_via_dgx(
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
                # THE PROVENANCE FIX (S0.6, #2177). This was `language or "en"`, which is a
                # LIE whenever the caller passed None: the request omits `language`, the server
                # auto-detects, and the artifact then recorded "en" for a Spanish episode.
                # Precedence: what we ASKED for, else what the server DETECTED, else None --
                # never a fabricated default. `None` is an honest "nobody said", which the
                # metadata layer can normalize or report; "en" is an assertion we cannot make.
                "language": language or self._last_detected_language,
                # The two halves of that, kept apart so the artifacts can show both (#2187):
                # what the pipeline ASKED for, and what the server SAID the audio is. A mismatch
                # is the hazard-3 signal; merged into one field it is invisible.
                "language_requested": language,
                "language_reported": self._last_detected_language,
                "model_requested": (model_override or self._model),
                "model_used": actual_model,
                # #2284: whether the transcript came back unpunctuated and what was done about it.
                "punctuation": getattr(self._local, "punctuation", None),
            },
            duration,
        )

    def _transcribe_via_dgx(
        self,
        audio_path: str,
        language: str | None,
        # Accepted for protocol symmetry; the DGX-side call doesn't use them (the
        # FallbackChain forwards them to whichever tier ultimately serves the request).
        pipeline_metrics: Any | None = None,
        episode_duration_seconds: int | None = None,
        call_metrics: Any | None = None,
        # #1046 — propagated to _transcribe_dgx. When set, overrides the
        # DGX-side model for this call only. None (default) keeps
        # ``self._model`` (the prod default).
        model_override: str | None = None,
    ) -> tuple[str, list[dict[str, object]], float, str]:
        last_err: Optional[Exception] = None
        timed_out = False
        self._local.punctuation = None
        first_prompt = self._first_request_prompt(language)
        # Effective model for THIS call — the override wins when present, else
        # the provider's configured default. The health-check substring is
        # derived from the effective model so we probe for the right loaded
        # variant when the sniff-pass workflow uses a different repo id.
        effective_model = (model_override or self._model).strip() or self._model
        # Health-check substring matches the model's slug portion (after the
        # ``Systran/`` namespace) so we don't fail on small repo-id variations.
        health_substring = effective_model.rsplit("/", 1)[-1]
        timeout_sec = self._effective_timeout_sec(episode_duration_seconds)

        if self._strategy is FailureStrategy.HOLD:
            # ADR-122: consistency over availability — backoff-retry the SAME model,
            # trip only after N, hold-and-probe on a blown fuse. Kept as a fully separate
            # method (not folded into the serve loop below) so the serve branch's today
            # behaviour stays byte-for-byte unchanged (RFC-106/#1198 regression guard).
            return self._transcribe_via_dgx_reprocess(
                audio_path,
                language,
                timeout_sec=timeout_sec,
                health_substring=health_substring,
                effective_model=effective_model,
                model_override=model_override,
                first_prompt=first_prompt,
            )

        # ---- serve mode (unchanged from today; RFC-106/#1198) ----
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
                                    prompt=first_prompt,
                                ),
                                timeout_sec + resilience.WATCHDOG_GRACE_SEC,
                                label="dgx-whisper",
                            )
                            _whisper_breaker.record_success()
                            # DGX path won: the model that actually ran IS
                            # the effective_model (override-or-default).
                            text_dgx, segments_dgx, duration_dgx = self._ensure_punctuated(
                                result_dgx,
                                audio_path,
                                language,
                                timeout_sec,
                                model_override=model_override,
                                prompted=first_prompt is not None,
                            )
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
        # RFC-106 (#1198): this tier is exhausted. RAISE rather than self-serving a cloud fallback —
        # the FallbackChain that wraps this provider owns the ladder and decides whether to advance
        # to the next tier. ``is_infra_failure`` treats the errors raised here (timeouts, guardrail
        # garbage, connection blips) as cascade-worthy, so the chain moves on; a content failure
        # (payload limit) raised from _transcribe_dgx propagates unwrapped and stops the chain.
        logger.warning("DGX Whisper tier exhausted (%s); raising for the fallback chain", reason)
        if last_err is not None:
            raise last_err
        raise RuntimeError(f"DGX Whisper unavailable: {reason}")

    def _transcribe_via_dgx_reprocess(
        self,
        audio_path: str,
        language: str | None,
        *,
        timeout_sec: float,
        health_substring: str,
        effective_model: str,
        model_override: str | None,
        first_prompt: str | None = None,
    ) -> tuple[str, list[dict[str, object]], float, str]:
        """ADR-122 reprocess-mode path: backoff-retry the chosen model, trip the fuse only
        after the policy threshold, and hold-and-probe (never fall over) on a blown fuse.

        The single-flight lock spans the whole call (health-check + retries + any
        pause-and-probe), mirroring the serve branch's one-acquisition-per-call pattern —
        while we're waiting on the DGX endpoint, nothing else in this process piles a
        second request onto it either.
        """

        def _attempt() -> tuple[str, list[dict[str, object]], float]:
            if not check_faster_whisper_health(
                self._host,
                port=self._port,
                require_model_substring=health_substring,
            ):
                raise RuntimeError("dgx_whisper_health_check_failed")
            return self._transcribe_dgx(
                audio_path,
                language,
                timeout_sec,
                model_override=model_override,
                prompt=first_prompt,
            )

        try:
            with _dgx_single_flight:
                first = self._policy.run(_attempt, timeout_sec=timeout_sec)
                text, segments, duration = self._ensure_punctuated(
                    first,
                    audio_path,
                    language,
                    timeout_sec,
                    model_override=model_override,
                    prompted=first_prompt is not None,
                )
        except ResilienceFuseOpenError as exc:
            emit_dgx_fallback_breadcrumb(
                stage="transcription",
                model=self._model,
                failure_reason=str(exc),
            )
            logger.error(
                "DGX Whisper endpoint did not recover after %.0fs of pause-and-probe "
                "(reprocess mode, ADR-122) — alerting operator, NOT falling over: %s",
                self._policy.on_open_max_wait_sec,
                exc,
            )
            raise
        return (text, segments, duration, effective_model)

    def cleanup(self) -> None:
        """No owned resources to release (the chain owns the fallback tiers)."""
        return None

    def _first_request_prompt(self, language: str | None) -> str | None:
        """The prompt for the first request: only in 'always' mode, and only for English."""
        if self._punctuation_mode == "always" and punctuation.prompt_suits_language(language):
            return punctuation.PUNCTUATION_PROMPT
        return None

    def _ensure_punctuated(
        self,
        first: tuple[str, list[dict[str, object]], float],
        audio_path: str,
        language: str | None,
        timeout_sec: float,
        *,
        model_override: str | None,
        prompted: bool,
    ) -> tuple[str, list[dict[str, object]], float]:
        """Return ``first``, or a prompted re-transcription when ``first`` is unpunctuated (#2284).

        At most ONE extra request, never a loop: the same audio at temperature 0 gives the same
        text, so only a CHANGED request (the prompt) can change the outcome. The retry runs under
        the caller's single-flight lock and its own watchdog; whatever happens to it -- an
        error, a timeout, a guardrail, a still-unpunctuated or prompt-echoing result -- the first
        transcript is kept, and the circuit breaker is not touched (the endpoint answered; the
        content is the problem). An episode is never failed or dropped for punctuation: an
        unpunctuated result is recorded (``punctuation`` on the result, an ``asr_unpunctuated``
        manifest flag) and logged, so it is visible and countable.
        """
        text = first[0]
        outcome: dict[str, object] = {
            "mode": self._punctuation_mode,
            "prompted_first": prompted,
            "retried": False,
            "unpunctuated": False,
            "sentence_ends_per_1000_words": round(
                punctuation.sentence_ends_per_1000_words(text), 1
            ),
        }
        self._local.punctuation = outcome
        if not punctuation.is_unpunctuated(text):
            return first
        outcome["unpunctuated"] = True
        if prompted or self._punctuation_mode != "on_retry":
            reason = "already prompted" if prompted else f"mode={self._punctuation_mode}"
            logger.warning("DGX Whisper transcript is unpunctuated (%s); kept as is", reason)
            return first
        if not punctuation.prompt_suits_language(language):
            outcome["skipped"] = f"language={language}"
            logger.warning(
                "DGX Whisper transcript is unpunctuated; no retry: the prompt is English and the "
                "episode language is %s",
                language,
            )
            return first

        outcome["retried"] = True
        try:
            second = resilience.run_with_watchdog(
                lambda: self._transcribe_dgx(
                    audio_path,
                    language,
                    timeout_sec,
                    model_override=model_override,
                    prompt=punctuation.PUNCTUATION_PROMPT,
                ),
                timeout_sec + resilience.WATCHDOG_GRACE_SEC,
                label="dgx-whisper-punctuation",
            )
        except Exception as exc:  # noqa: BLE001 - a failed retry must never cost the transcript
            outcome["retry_error"] = format_exception_for_log(exc)
            logger.warning(
                "DGX Whisper punctuation retry failed (%s); keeping the unpunctuated transcript",
                format_exception_for_log(exc),
            )
            return first

        second_text = second[0]
        second_ends = punctuation.sentence_ends_per_1000_words(second_text)
        outcome["retry_sentence_ends_per_1000_words"] = round(second_ends, 1)
        if punctuation.echoes_prompt(second_text):
            outcome["retry_rejected"] = "echoed the prompt"
        elif punctuation.is_unpunctuated(second_text):
            outcome["retry_rejected"] = "still unpunctuated"
        else:
            outcome["unpunctuated"] = False
            logger.info(
                "DGX Whisper transcript was unpunctuated; the prompted retry restored it "
                "(%.0f sentence ends per 1,000 words)",
                second_ends,
            )
            return second
        logger.warning(
            "DGX Whisper transcript is unpunctuated and the prompted retry did not fix it (%s); "
            "keeping the first transcript",
            outcome["retry_rejected"],
        )
        return first

    def _transcribe_dgx(
        self,
        audio_path: str,
        language: str | None,
        timeout_sec: Optional[float] = None,
        # #1046 — per-call model override (e.g. sniff-pass uses ``small.en``).
        # None preserves the configured default.
        model_override: str | None = None,
        # #2284 — Whisper's initial prompt; sets a punctuated, capitalised output style.
        prompt: str | None = None,
        # #2187 A2 — False for a gap clip, where few or no words is a real answer.
        check_response: bool = True,
        # False for a clip: `_last_detected_language` belongs to the EPISODE call, which reads it
        # after releasing the lock — a clip writing it could stamp another episode's record when
        # transcription_parallelism > 1.
        record_language: bool = True,
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
            if prompt:
                data["prompt"] = prompt
            with hardened_http_client(
                timeout_sec or self._timeout_sec, subsystem="dgx_whisper"
            ) as client:
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
        if check_response:
            audio_duration_sec = resilience.probe_audio_duration_sec(audio_path)
            guardrails.check_whisper_response(text, audio_duration_sec=audio_duration_sec)
        duration = float(time.perf_counter() - started)
        segments: list[dict[str, object]] = []
        if isinstance(payload.get("segments"), list):
            segments = [s for s in payload["segments"] if isinstance(s, dict)]
            segments = _refine_segment_times(segments, payload.get("words"))
        # S0.6 (#2177): keep the language the SERVER reports. When `language` was None the
        # request above omitted it, so the server auto-detected and told us the answer -- and
        # this provider used to throw that away and claim "en" in its result dict. Recorded on
        # the instance rather than threaded through four return signatures as a fifth tuple
        # element; each provider instance serves one call at a time, and the reader is the very
        # next statement after the call.
        detected = payload.get("language")
        if record_language:
            self._last_detected_language = (
                str(detected).strip() or None if isinstance(detected, str) else None
            )
        return text, segments, duration

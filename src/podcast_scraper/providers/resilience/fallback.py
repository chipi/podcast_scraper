"""Provider-agnostic tiered failover (RFC-106 / #1198).

A DGX-backed stage degrades through an **ordered chain** of providers, not a single fallback. The
chain is registry-governed data (``<stage>_fallback_providers`` in the materialized profile) and is
constructed once in each stage's factory, so every provider gets identical behaviour instead of
each one re-implementing its own self-wrapping fallback.

WHAT CASCADES, AND WHAT DOES NOT
--------------------------------
The chain advances to the next tier only on an **infrastructure** failure — connection refused,
timeout, health-check down, an HTTP 5xx, or a service that returned garbage. A **content** failure
does not cascade: an oversized-audio payload-limit rejection or a deterministic 4xx would be
rejected identically by every tier, so retrying the next one only wastes money and hides the bug.

:func:`is_infra_failure` encodes exactly that split. It **defaults to True** (cascade) because that
is the behaviour the hand-wired ``tailnet_dgx_whisper`` fallback already had — a connection blip, a
transient 5xx, a WER=1.0 garbage transcript (a ``GuardrailViolation``) all fell through to the cloud
tier. The deny-list is small and argued: only failures that are deterministic in the *input* stay
on this tier and raise.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Tuple

from ...providers.ml.diarization.base import DiarizationProvider, DiarizationResult
from ...transcription.base import TranscriptionProvider
from ...utils.audio_payload_limits import is_provider_audio_payload_limit_error
from ...utils.log_redaction import format_exception_for_log
from ..resilience import probe_audio_duration_sec, TimeoutLike

logger = logging.getLogger(__name__)


def is_infra_failure(exc: BaseException) -> bool:
    """Should the failover chain advance past the tier that raised ``exc``?

    True  -> infrastructure failure (this tier is unhealthy / produced garbage); another tier may
             succeed, so cascade.
    False -> content failure (deterministic in the input); every tier would reject it identically,
             so stay here and raise.

    The default is **cascade** (True): it preserves the pre-RFC-106 ``tailnet_dgx_whisper``
    behaviour, which fell back on connection blips, transient 5xx, timeouts, and garbage-response
    guardrail violations alike. Only the explicitly content-deterministic cases below return False.
    """
    # Oversized-audio / payload-limit rejections are a property of the FILE, not the service: the
    # next tier's limit rejects the same bytes. Do not cascade — chunk or shrink instead.
    if is_provider_audio_payload_limit_error(exc):
        return False

    # A deterministic client error is the request's fault, not the tier's — do not cascade. The
    # exceptions are the transient/infra 4xx: 408 (Request Timeout) and 429 (Too Many Requests) are
    # a slow/pressured tier, not a bad request, so they stay cascade-worthy like a timeout.
    _TRANSIENT_4XX = frozenset({408, 429})
    try:  # pragma: no cover - import guard mirrors resilience.exceptions
        import httpx

        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if 400 <= status < 500 and status not in _TRANSIENT_4XX:
                return False
    except ImportError:  # pragma: no cover
        pass

    # Everything else — timeouts, connection errors, 5xx, health-down sentinels, guardrail-flagged
    # garbage, and unclassified exceptions — is treated as an infra failure and cascades. Timeouts
    # are called out only for clarity; they already fall through the default.
    _ = TimeoutLike  # documents intent; timeouts hit the default-True path
    return True


class FallbackChainTranscriptionProvider:
    """Try an ordered list of transcription providers, advancing on infra failure (RFC-106).

    Tier 0 is the primary; the rest are the failover ladder. Each tier is initialized lazily, right
    before it is first used, so a healthy primary never pays for the cloud tiers' client setup (and
    a missing cloud API key is not an error until the chain actually needs that tier).
    """

    name = "fallback_chain"

    def __init__(self, tiers: List[Tuple[str, Callable[[], TranscriptionProvider]]]) -> None:
        """``tiers`` is ``[(provider_name, builder), ...]``, primary first (>= 1 tier).

        Each ``builder`` is a zero-arg callable that CONSTRUCTS the tier's provider. Tiers are built
        lazily — a fallback tier's provider is not constructed until the chain actually reaches it,
        so a healthy primary never pays for (and a missing cloud API key never crashes) a tier that
        is never used. This is what preserves the #926 lazy-fallback guarantee.
        """
        if not tiers:
            raise ValueError("FallbackChainTranscriptionProvider requires at least one tier")
        self._names = [n for n, _ in tiers]
        self._builders = [b for _, b in tiers]
        self._providers: List[Optional[TranscriptionProvider]] = [None] * len(tiers)
        self._inited = [False] * len(tiers)

    def initialize(self) -> None:
        """Eagerly construct + initialize the primary so its config errors surface at startup;
        fallback tiers stay lazy (constructed + initialized on first use)."""
        self._ensure_tier(0)

    def _ensure_tier(self, i: int) -> TranscriptionProvider:
        provider = self._providers[i]
        if provider is None:
            provider = self._builders[i]()  # lazy construct: never-reached tiers never built
            self._providers[i] = provider
        if not self._inited[i]:
            provider.initialize()
            self._inited[i] = True
        return provider

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """Return transcript text from the first tier that succeeds."""
        result, _elapsed = self.transcribe_with_segments(audio_path, language)
        return str(result.get("text", ""))

    def transcribe_with_segments(
        self,
        audio_path: str,
        language: str | None = None,
        pipeline_metrics: Any | None = None,
        episode_duration_seconds: int | None = None,
        call_metrics: Any | None = None,
    ) -> Tuple[dict[str, object], float]:
        """Return ``(result_dict, elapsed)`` from the first tier that succeeds.

        Advances to the next tier on an infra failure; re-raises immediately on a content failure or
        once the last tier is exhausted. The winning tier's provider name is recorded on the result
        (``model_used`` is prefixed with it when the tier did not already attribute a model), so
        post-fallback provenance is preserved (#1046).
        """
        last = len(self._names) - 1
        last_exc: Optional[BaseException] = None
        for i, pname in enumerate(self._names):
            try:
                provider = self._ensure_tier(i)
                result, elapsed = provider.transcribe_with_segments(
                    audio_path,
                    language,
                    pipeline_metrics=pipeline_metrics,
                    episode_duration_seconds=episode_duration_seconds,
                    call_metrics=call_metrics,
                )
            except Exception as exc:  # noqa: BLE001 - classified below; re-raised if not infra
                last_exc = exc
                if i == last or not is_infra_failure(exc):
                    raise
                logger.warning(
                    "transcription tier %r failed (infra); advancing to %r: %s",
                    pname,
                    self._names[i + 1],
                    format_exception_for_log(exc),
                )
                continue
            if i > 0 and not result.get("model_used"):
                # A fallback tier won and did not attribute a model — record which tier, so the run
                # does not silently credit the primary's model for a call it never made (#1046).
                result = {**result, "model_used": f"{pname}:default"}
            return result, elapsed
        # Unreachable: the last tier either returns or raises above. Kept for the type checker.
        assert last_exc is not None
        raise last_exc

    def cleanup(self) -> None:
        """Release every built tier's resources; a tier that raises on cleanup does not prevent the
        others from being released."""
        for i, provider in enumerate(self._providers):
            if provider is not None and self._inited[i]:
                try:
                    provider.cleanup()
                except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                    logger.warning(
                        "transcription tier %r cleanup failed: %s",
                        self._names[i],
                        format_exception_for_log(exc),
                    )


def _segment_coverage(
    result: dict[str, object], audio_path: str, episode_duration_seconds: Optional[int]
) -> Optional[float]:
    """ADR-123: fraction of the audio the transcript's segments cover.

    ``Σ(segment_end − segment_start) / audio_duration``. Returns None when it cannot be measured
    (no segments, or unknown duration) — the gate treats "unknowable" as "do not failover", so a
    provider that emits no timestamps is never spuriously re-routed. Uses the passed episode
    duration when available, else probes the audio.
    """
    segs = result.get("segments")
    if not isinstance(segs, list) or not segs:
        return None
    covered = 0.0
    for s in segs:
        if not isinstance(s, dict):
            continue
        try:
            covered += max(0.0, float(s["end"]) - float(s["start"]))
        except (KeyError, TypeError, ValueError):
            continue
    duration = float(episode_duration_seconds) if episode_duration_seconds else 0.0
    if duration <= 0:
        probed = probe_audio_duration_sec(audio_path)
        duration = float(probed) if probed else 0.0
    if duration <= 0:
        return None
    return covered / duration


class CoverageGatedTranscriptionProvider:
    """Quality-gate transcription failover (ADR-123 / #1258).

    Wraps a primary provider. After a SUCCESSFUL transcription it measures
    :func:`_segment_coverage`; if coverage is below ``coverage_min`` the primary silently dropped
    speech (turbo's long-form VAD/segmentation failure returns a clean but incomplete transcript),
    so the episode is re-transcribed on the ``failover`` model and that result is used instead.

    Distinct from :class:`FallbackChainTranscriptionProvider`, which advances on an EXCEPTION — this
    fires on a successful-but-incomplete OUTPUT, and is orthogonal to ADR-122 ``hold`` (which
    governs infra-failure behaviour). Both tiers are built lazily, so a corpus that never trips it
    never constructs the failover model. The winning model is recorded on the result
    (``model_used`` + a ``coverage_failover`` breadcrumb) so per-episode provenance survives.
    """

    name = "coverage_gated"

    def __init__(
        self,
        primary: Tuple[str, Callable[[], TranscriptionProvider]],
        failover: Tuple[str, Callable[[], TranscriptionProvider]],
        coverage_min: float,
    ) -> None:
        self._primary_name, self._primary_builder = primary
        self._failover_name, self._failover_builder = failover
        self._coverage_min = coverage_min
        self._primary: Optional[TranscriptionProvider] = None
        self._failover: Optional[TranscriptionProvider] = None
        self._primary_inited = False
        self._failover_inited = False

    def initialize(self) -> None:
        """Eagerly construct + initialize the primary; the failover model stays lazy."""
        self._ensure(primary=True)

    def _ensure(self, *, primary: bool) -> TranscriptionProvider:
        if primary:
            if self._primary is None:
                self._primary = self._primary_builder()
            if not self._primary_inited:
                self._primary.initialize()
                self._primary_inited = True
            return self._primary
        if self._failover is None:
            self._failover = self._failover_builder()
        if not self._failover_inited:
            self._failover.initialize()
            self._failover_inited = True
        return self._failover

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        result, _elapsed = self.transcribe_with_segments(audio_path, language)
        return str(result.get("text", ""))

    def transcribe_with_segments(
        self,
        audio_path: str,
        language: str | None = None,
        pipeline_metrics: Any | None = None,
        episode_duration_seconds: int | None = None,
        call_metrics: Any | None = None,
    ) -> Tuple[dict[str, object], float]:
        """Transcribe on the primary; if coverage < ``coverage_min``, re-transcribe on the failover
        model and return that result instead (with provenance)."""
        primary = self._ensure(primary=True)
        result, elapsed = primary.transcribe_with_segments(
            audio_path,
            language,
            pipeline_metrics=pipeline_metrics,
            episode_duration_seconds=episode_duration_seconds,
            call_metrics=call_metrics,
        )
        coverage = _segment_coverage(result, audio_path, episode_duration_seconds)
        if coverage is None or coverage >= self._coverage_min:
            return result, elapsed
        logger.warning(
            "transcription coverage %.1f%% < %.1f%% on %s — primary %r silently dropped speech; "
            "quality failover to %r (ADR-123)",
            coverage * 100,
            self._coverage_min * 100,
            audio_path,
            self._primary_name,
            self._failover_name,
        )
        failover = self._ensure(primary=False)
        fo_result, fo_elapsed = failover.transcribe_with_segments(
            audio_path,
            language,
            pipeline_metrics=pipeline_metrics,
            episode_duration_seconds=episode_duration_seconds,
            call_metrics=call_metrics,
        )
        fo_coverage = _segment_coverage(fo_result, audio_path, episode_duration_seconds)
        fo_result = {
            **fo_result,
            "model_used": fo_result.get("model_used") or f"{self._failover_name}:coverage_failover",
            "coverage_failover": {
                "primary": self._primary_name,
                "primary_coverage": round(coverage, 3),
                "failover_coverage": round(fo_coverage, 3) if fo_coverage is not None else None,
                "coverage_min": self._coverage_min,
            },
        }
        # elapsed reflects the full cost (both passes) — the wasted primary pass is real time spent.
        return fo_result, elapsed + fo_elapsed

    def cleanup(self) -> None:
        """Release both tiers' resources; one raising does not block the other."""
        for provider, inited in (
            (self._primary, self._primary_inited),
            (self._failover, self._failover_inited),
        ):
            if provider is not None and inited:
                try:
                    provider.cleanup()
                except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                    logger.warning(
                        "coverage-gated tier cleanup failed: %s", format_exception_for_log(exc)
                    )


class FallbackChainDiarizationProvider:
    """Try an ordered list of diarization providers, advancing on infra failure (RFC-106 / #1198).

    The diarization analogue of :class:`FallbackChainTranscriptionProvider`. Tier 0 is the primary
    (e.g. DGX pyannote); the rest are the ladder (e.g. local in-process pyannote, then a cloud
    diarizer). Fallback tiers initialize lazily so a healthy primary never pays the in-process
    pyannote model-load cost the DGX-primary path was designed to avoid (#926).
    """

    name = "fallback_chain"

    def __init__(self, tiers: List[Tuple[str, Callable[[], DiarizationProvider]]]) -> None:
        """``tiers`` is ``[(provider_name, builder), ...]``, primary first (>= 1 tier).

        Each ``builder`` CONSTRUCTS the tier's provider, called lazily on first use — a fallback
        tier that is never reached is never built, so a healthy DGX primary never pays the
        in-process pyannote model-load cost (#926) and a missing cloud/HF credential for a
        never-used tier never crashes provider creation.
        """
        if not tiers:
            raise ValueError("FallbackChainDiarizationProvider requires at least one tier")
        self._names = [n for n, _ in tiers]
        self._builders = [b for _, b in tiers]
        self._providers: List[Optional[DiarizationProvider]] = [None] * len(tiers)
        self._inited = [False] * len(tiers)

    def initialize(self) -> None:
        """Eagerly construct + initialize the primary; fallback tiers stay lazy."""
        self._ensure_tier(0)

    def _ensure_tier(self, i: int) -> DiarizationProvider:
        provider = self._providers[i]
        if provider is None:
            provider = self._builders[i]()  # lazy construct: never-reached tiers never built
            self._providers[i] = provider
        if not self._inited[i]:
            init = getattr(provider, "initialize", None)
            if callable(init):
                init()
            self._inited[i] = True
        return provider

    def diarize(
        self,
        audio_path: str,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: int = 2,
        max_speakers: int = 20,
    ) -> DiarizationResult:
        """Return the first tier's result, advancing on infra failure and re-raising on a content
        failure or once the chain is exhausted."""
        last = len(self._names) - 1
        last_exc: Optional[BaseException] = None
        for i, pname in enumerate(self._names):
            try:
                provider = self._ensure_tier(i)
                return provider.diarize(
                    audio_path,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
            except Exception as exc:  # noqa: BLE001 - classified below; re-raised if not infra
                last_exc = exc
                if i == last or not is_infra_failure(exc):
                    raise
                logger.warning(
                    "diarization tier %r failed (infra); advancing to %r: %s",
                    pname,
                    self._names[i + 1],
                    format_exception_for_log(exc),
                )
        assert last_exc is not None  # unreachable; the last tier returns or raises above
        raise last_exc

    def cleanup(self) -> None:
        """Release every built tier's resources (tiers may omit cleanup); one tier raising on
        cleanup does not prevent the others from being released."""
        for i, provider in enumerate(self._providers):
            if provider is None or not self._inited[i]:
                continue
            cleanup = getattr(provider, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                    logger.warning(
                        "diarization tier %r cleanup failed: %s",
                        self._names[i],
                        format_exception_for_log(exc),
                    )


__all__ = [
    "FallbackChainDiarizationProvider",
    "FallbackChainTranscriptionProvider",
    "is_infra_failure",
]

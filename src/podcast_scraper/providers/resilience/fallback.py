"""Provider-agnostic tiered failover (RFC-105 / #1198).

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
from typing import Any, List, Optional, Tuple

from ...providers.ml.diarization.base import DiarizationProvider, DiarizationResult
from ...transcription.base import TranscriptionProvider
from ...utils.audio_payload_limits import is_provider_audio_payload_limit_error
from ...utils.log_redaction import format_exception_for_log
from ..resilience import TimeoutLike

logger = logging.getLogger(__name__)


def is_infra_failure(exc: BaseException) -> bool:
    """Should the failover chain advance past the tier that raised ``exc``?

    True  -> infrastructure failure (this tier is unhealthy / produced garbage); another tier may
             succeed, so cascade.
    False -> content failure (deterministic in the input); every tier would reject it identically,
             so stay here and raise.

    The default is **cascade** (True): it preserves the pre-RFC-105 ``tailnet_dgx_whisper``
    behaviour, which fell back on connection blips, transient 5xx, timeouts, and garbage-response
    guardrail violations alike. Only the explicitly content-deterministic cases below return False.
    """
    # Oversized-audio / payload-limit rejections are a property of the FILE, not the service: the
    # next tier's limit rejects the same bytes. Do not cascade — chunk or shrink instead.
    if is_provider_audio_payload_limit_error(exc):
        return False

    # A deterministic client error (4xx other than 429) is the request's fault, not the tier's.
    # 429 (rate limit) is transient infra pressure, so it stays cascade-worthy.
    try:  # pragma: no cover - import guard mirrors resilience.exceptions
        import httpx

        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if 400 <= status < 500 and status != 429:
                return False
    except ImportError:  # pragma: no cover
        pass

    # Everything else — timeouts, connection errors, 5xx, health-down sentinels, guardrail-flagged
    # garbage, and unclassified exceptions — is treated as an infra failure and cascades. Timeouts
    # are called out only for clarity; they already fall through the default.
    _ = TimeoutLike  # documents intent; timeouts hit the default-True path
    return True


class FallbackChainTranscriptionProvider:
    """Try an ordered list of transcription providers, advancing on infra failure (RFC-105).

    Tier 0 is the primary; the rest are the failover ladder. Each tier is initialized lazily, right
    before it is first used, so a healthy primary never pays for the cloud tiers' client setup (and
    a missing cloud API key is not an error until the chain actually needs that tier).
    """

    name = "fallback_chain"

    def __init__(self, tiers: List[Tuple[str, TranscriptionProvider]]) -> None:
        """``tiers`` is ``[(provider_name, provider_instance), ...]``, primary first (>= 1 tier)."""
        if not tiers:
            raise ValueError("FallbackChainTranscriptionProvider requires at least one tier")
        self._tiers = tiers
        self._inited = [False] * len(tiers)

    def initialize(self) -> None:
        """Eagerly initialize the primary so its config errors surface at startup; fallbacks stay
        lazy (initialized on first use)."""
        self._ensure_tier(0)

    def _ensure_tier(self, i: int) -> None:
        if not self._inited[i]:
            self._tiers[i][1].initialize()
            self._inited[i] = True

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
        last = len(self._tiers) - 1
        last_exc: Optional[BaseException] = None
        for i, (pname, provider) in enumerate(self._tiers):
            try:
                self._ensure_tier(i)
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
                    self._tiers[i + 1][0],
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
        """Release every initialized tier's resources."""
        for i, (_name, provider) in enumerate(self._tiers):
            if self._inited[i]:
                provider.cleanup()


class FallbackChainDiarizationProvider:
    """Try an ordered list of diarization providers, advancing on infra failure (RFC-105 / #1198).

    The diarization analogue of :class:`FallbackChainTranscriptionProvider`. Tier 0 is the primary
    (e.g. DGX pyannote); the rest are the ladder (e.g. local in-process pyannote, then a cloud
    diarizer). Fallback tiers initialize lazily so a healthy primary never pays the in-process
    pyannote model-load cost the DGX-primary path was designed to avoid (#926).
    """

    name = "fallback_chain"

    def __init__(self, tiers: List[Tuple[str, DiarizationProvider]]) -> None:
        """``tiers`` is ``[(provider_name, provider_instance), ...]``, primary first (>= 1 tier)."""
        if not tiers:
            raise ValueError("FallbackChainDiarizationProvider requires at least one tier")
        self._tiers = tiers
        self._inited = [False] * len(tiers)

    def initialize(self) -> None:
        """Eagerly initialize the primary; fallback tiers stay lazy (initialized on first use)."""
        self._ensure_tier(0)

    def _ensure_tier(self, i: int) -> None:
        if not self._inited[i]:
            provider = self._tiers[i][1]
            init = getattr(provider, "initialize", None)
            if callable(init):
                init()
            self._inited[i] = True

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
        last = len(self._tiers) - 1
        last_exc: Optional[BaseException] = None
        for i, (pname, provider) in enumerate(self._tiers):
            try:
                self._ensure_tier(i)
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
                    self._tiers[i + 1][0],
                    format_exception_for_log(exc),
                )
        assert last_exc is not None  # unreachable; the last tier returns or raises above
        raise last_exc

    def cleanup(self) -> None:
        """Release every initialized tier's resources (tiers may omit cleanup)."""
        for i, (_name, provider) in enumerate(self._tiers):
            if self._inited[i]:
                cleanup = getattr(provider, "cleanup", None)
                if callable(cleanup):
                    cleanup()


__all__ = [
    "FallbackChainDiarizationProvider",
    "FallbackChainTranscriptionProvider",
    "is_infra_failure",
]

"""A diarization provider that falls back down a chain (moved from
``providers.resilience.fallback``, E04 S2: that module is shared with transcription, which
stays public).
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

from podcast_scraper.providers.ml.diarization.base import DiarizationProvider, DiarizationResult
from podcast_scraper.providers.resilience.fallback import is_infra_failure
from podcast_scraper.utils.log_redaction import format_exception_for_log

logger = logging.getLogger(__name__)


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

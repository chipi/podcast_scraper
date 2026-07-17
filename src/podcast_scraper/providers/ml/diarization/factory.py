"""Factory for diarization providers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

from .... import config
from .base import DiarizationProvider
from .pyannote_provider import PyAnnoteDiarizationProvider

logger = logging.getLogger(__name__)


def resolve_hf_token(cfg: config.Config) -> Optional[str]:
    """Resolve HuggingFace token from config, env, or the HF CLI token files."""
    if cfg.hf_token:
        return cfg.hf_token.strip()
    for env_var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        env_token = os.getenv(env_var)
        if env_token:
            return env_token.strip()
    # Legacy path first, then the modern ``huggingface-cli login`` location
    # (~/.cache/huggingface/token) — the old code only checked the legacy path
    # and missed tokens stored by a current HF CLI.
    for token_path in (
        Path.home() / ".huggingface" / "token",
        Path.home() / ".cache" / "huggingface" / "token",
    ):
        if token_path.is_file():
            token = token_path.read_text(encoding="utf-8").strip()
            if token:
                return token
    return None


def create_local_pyannote_provider(cfg: config.Config) -> PyAnnoteDiarizationProvider:
    """Create the in-process pyannote.audio diarization provider from config.

    Exposed as its own function (not just inlined in ``create_diarization_provider``)
    so the DGX client can build it as a lazy fallback (#926) without going
    through the dispatch logic that picked the DGX backend in the first place.
    """
    hf_token = resolve_hf_token(cfg)
    if not hf_token:
        raise ValueError(
            "HuggingFace token required for diarize=true. "
            "Set HF_TOKEN, hf_token in config, or ~/.huggingface/token."
        )
    return PyAnnoteDiarizationProvider(
        hf_token=hf_token,
        device=cfg.diarization_device,
        model_name=cfg.diarization_model,
        clustering_threshold=cfg.diarization_clustering_threshold,
        min_cluster_size=cfg.diarization_min_cluster_size,
        min_segment_ms=cfg.diarization_min_segment_ms,
    )


def _diarization_fallback_tiers(cfg: config.Config) -> list[str]:
    """The ordered diarization failover ladder for ``cfg`` (RFC-106 / #1198), each a
    ``diarization_provider`` backend string."""
    return [
        str(p).strip()
        for p in (getattr(cfg, "diarization_fallback_providers", None) or [])
        if str(p).strip()
    ]


def _build_diarization_tier(cfg: config.Config, backend: str) -> DiarizationProvider:
    """Build one chain tier for ``backend``, with fallback-wrapping suppressed so tiers do not
    recursively re-wrap."""
    data = cfg.model_dump()
    data["diarization_provider"] = backend
    sub = config.Config.model_validate(data)
    return create_diarization_provider(sub, _wrap_fallback=False)


def _diarization_tier_builder(
    cfg: config.Config, backend: str
) -> Callable[[], DiarizationProvider]:
    """A zero-arg closure that constructs the ``backend`` tier on demand (lazy chain construction).
    Binds ``backend`` per call, so a loop over tiers gets distinct builders."""
    return lambda: _build_diarization_tier(cfg, backend)


def create_diarization_provider(
    cfg: config.Config, *, _wrap_fallback: bool = True
) -> DiarizationProvider:
    """Dispatch to the configured diarization backend.

    ``cfg.diarization_provider``:

    - ``local`` (default) — in-process pyannote.audio on the pipeline host.
    - ``tailnet_dgx`` — POST audio to the DGX-hosted pyannote service over
      the tailnet (#926).
    - ``gemini`` — Gemini 2.5 audio understanding (#962). Cloud-only path
      with no local pyannote install required.

    RFC-106 (#1198): when the profile declares ``diarization_fallback_providers``, the primary and
    each tier are wrapped in a ``FallbackChainDiarizationProvider`` that owns the failover ladder
    (DGX pyannote -> local pyannote -> deepgram). The per-provider self-wrap in
    ``TailnetDgxDiarizationProvider`` is retired in favour of that chain.
    """
    backend = getattr(cfg, "diarization_provider", "local")

    if _wrap_fallback:
        tiers = _diarization_fallback_tiers(cfg)
        if tiers:
            from ...resilience.fallback import FallbackChainDiarizationProvider

            # Pass BUILDERS, not instances: the chain constructs each tier lazily on first use, so a
            # never-reached fallback tier never crashes a healthy-DGX run when its credential (HF
            # token / DEEPGRAM_API_KEY) is absent — preserving #926's lazy fallback.
            chain: list[tuple[str, Callable[[], DiarizationProvider]]] = [
                (backend, _diarization_tier_builder(cfg, backend))
            ]
            for tier_backend in tiers:
                chain.append((tier_backend, _diarization_tier_builder(cfg, tier_backend)))
            wrapped = FallbackChainDiarizationProvider(chain)
            wrapped.initialize()
            return wrapped

    if backend == "tailnet_dgx":
        # Import lazily so the local path doesn't need the tailnet_dgx package
        # to even exist on systems that won't use DGX.
        from ...tailnet_dgx.diarization_provider import (
            TailnetDgxDiarizationProvider,
        )

        provider = TailnetDgxDiarizationProvider(cfg)
        provider.initialize()
        return provider
    if backend == "moss":
        # The speaker half of the joint MOSS model (#1177). Pairs with
        # ``transcription_provider: moss`` — the DGX service caches its inference, so whichever
        # stage runs second reads the same pass rather than re-running the model.
        from .moss_provider import MossDiarizationProvider

        moss = MossDiarizationProvider(cfg)
        moss.initialize()
        return moss
    if backend == "gemini":
        from .gemini_provider import GeminiDiarizationProvider

        api_key = getattr(cfg, "gemini_api_key", None) or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY required for diarization_provider=gemini. "
                "Set gemini_api_key in config or the GEMINI_API_KEY env var."
            )
        return GeminiDiarizationProvider(
            api_key=api_key,
            model_name=getattr(cfg, "gemini_diarization_model", "gemini-2.5-flash"),
            temperature=getattr(cfg, "gemini_temperature", 0.0),
        )
    if backend == "deepgram":
        from .deepgram_provider import DeepgramDiarizationProvider

        api_key = getattr(cfg, "deepgram_api_key", None) or os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPGRAM_API_KEY required for diarization_provider=deepgram. "
                "Set deepgram_api_key in config or the DEEPGRAM_API_KEY env var."
            )
        dg_provider: DiarizationProvider = DeepgramDiarizationProvider(
            api_key=api_key,
            model=getattr(cfg, "deepgram_diarization_model", "nova-3-general"),
            api_base=getattr(cfg, "deepgram_api_base", None),
        )
        if hasattr(dg_provider, "initialize"):
            dg_provider.initialize()
        return dg_provider
    return create_local_pyannote_provider(cfg)

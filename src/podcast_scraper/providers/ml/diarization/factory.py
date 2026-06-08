"""Factory for diarization providers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

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
    )


def create_diarization_provider(cfg: config.Config) -> DiarizationProvider:
    """Dispatch to the configured diarization backend.

    ``cfg.diarization_provider``:

    - ``local`` (default) — in-process pyannote.audio on the pipeline host.
    - ``tailnet_dgx`` — POST audio to the DGX-hosted pyannote service over
      the tailnet (#926). Falls back to local pyannote on DGX failure.
    """
    backend = getattr(cfg, "diarization_provider", "local")
    if backend == "tailnet_dgx":
        # Import lazily so the local path doesn't need the tailnet_dgx package
        # to even exist on systems that won't use DGX.
        from ...tailnet_dgx.diarization_provider import (
            TailnetDgxDiarizationProvider,
        )

        provider = TailnetDgxDiarizationProvider(cfg)
        provider.initialize()
        return provider
    return create_local_pyannote_provider(cfg)

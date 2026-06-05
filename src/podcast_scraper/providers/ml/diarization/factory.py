"""Factory for diarization providers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from .... import config
from .pyannote_provider import PyAnnoteDiarizationProvider

logger = logging.getLogger(__name__)


def resolve_hf_token(cfg: config.Config) -> Optional[str]:
    """Resolve HuggingFace token from config, env, or default token file."""
    if cfg.hf_token:
        return cfg.hf_token.strip()
    env_token = os.getenv("HF_TOKEN")
    if env_token:
        return env_token.strip()
    token_path = Path.home() / ".huggingface" / "token"
    if token_path.is_file():
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            return token
    return None


def create_diarization_provider(cfg: config.Config) -> PyAnnoteDiarizationProvider:
    """Create pyannote diarization provider from config."""
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

"""Factory for creating transcription providers.

This module provides a factory function to create transcription
providers based on configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from podcast_scraper import config
    from podcast_scraper.transcription.base import TranscriptionProvider


def create_transcription_provider(cfg: config.Config) -> TranscriptionProvider:
    """Create a transcription provider based on configuration.

    Args:
        cfg: Configuration object

    Returns:
        TranscriptionProvider instance

    Raises:
        ValueError: If provider type is not supported

    Note:
        Stage 2: Returns WhisperTranscriptionProvider for "whisper" provider type.
    """
    provider_type = cfg.transcription_provider

    if provider_type == "whisper":
        from .whisper_provider import WhisperTranscriptionProvider

        return WhisperTranscriptionProvider(cfg)
    elif provider_type == "openai":
        from .openai_provider import OpenAITranscriptionProvider

        return OpenAITranscriptionProvider(cfg)
    else:
        raise ValueError(
            f"Unsupported transcription provider: {provider_type}. "
            "Supported providers: 'whisper', 'openai'"
        )

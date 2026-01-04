"""Factory for creating transcription providers.

This module provides a factory function to create transcription
providers based on configuration.
"""

from __future__ import annotations

from typing import cast, TYPE_CHECKING

if TYPE_CHECKING:
    from podcast_scraper import config
    from podcast_scraper.transcription.base import TranscriptionProvider
else:
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
        Returns MLProvider for "whisper" provider type (unified ML provider).
        Returns OpenAIProvider for "openai" provider type (unified OpenAI provider).
        Reuses preloaded MLProvider instance if available (from early preloading).
    """
    provider_type = cfg.transcription_provider

    if provider_type == "whisper":
        # Check for preloaded MLProvider instance (from early preloading)
        try:
            from ..workflow import _preloaded_ml_provider

            if _preloaded_ml_provider is not None:
                return cast(TranscriptionProvider, _preloaded_ml_provider)
        except ImportError:
            # workflow module not available (e.g., in tests), create new instance
            pass

        # Create new instance if no preloaded instance available
        from ..ml.ml_provider import MLProvider

        return MLProvider(cfg)
    elif provider_type == "openai":
        from ..openai.openai_provider import OpenAIProvider

        return OpenAIProvider(cfg)
    else:
        raise ValueError(
            f"Unsupported transcription provider: {provider_type}. "
            "Supported providers: 'whisper', 'openai'"
        )

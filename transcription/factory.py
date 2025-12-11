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
        Stage 0: Currently returns NotImplementedError.
        Implementations will be added in later stages.
    """
    # Stage 0: Factory is empty - implementations will be added in later stages
    raise NotImplementedError(
        "Transcription provider factory not yet implemented. "
        "This will be implemented in Stage 3."
    )

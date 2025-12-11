"""Factory for creating speaker detection providers.

This module provides a factory function to create speaker detection
providers based on configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from podcast_scraper import config
    from podcast_scraper.speaker_detectors.base import SpeakerDetector


def create_speaker_detector(cfg: config.Config) -> SpeakerDetector:
    """Create a speaker detection provider based on configuration.

    Args:
        cfg: Configuration object

    Returns:
        SpeakerDetector instance

    Raises:
        ValueError: If provider type is not supported

    Note:
        Stage 0: Currently returns NotImplementedError.
        Implementations will be added in later stages.
    """
    # Stage 0: Factory is empty - implementations will be added in later stages
    raise NotImplementedError(
        "Speaker detector factory not yet implemented. " "This will be implemented in Stage 2."
    )

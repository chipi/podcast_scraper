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
        Stage 3: Returns NERSpeakerDetector for "ner" provider type.
    """
    # Support both new and deprecated field names for backward compatibility
    provider_type = getattr(cfg, "speaker_detector_provider", None) or getattr(
        cfg, "speaker_detector_type", "ner"
    )

    if provider_type == "ner":
        from .ner_detector import NERSpeakerDetector

        return NERSpeakerDetector(cfg)
    elif provider_type == "openai":
        from .openai_detector import OpenAISpeakerDetector

        return OpenAISpeakerDetector(cfg)
    else:
        raise ValueError(
            f"Unsupported speaker detector type: {provider_type}. "
            "Supported types: 'ner', 'openai'"
        )

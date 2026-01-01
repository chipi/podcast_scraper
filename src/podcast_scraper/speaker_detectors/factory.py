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
        Returns MLProvider for "spacy" provider type (unified ML provider).
        Returns OpenAIProvider for "openai" provider type (unified OpenAI provider).
        Deprecated: "ner" is accepted as alias for "spacy" for backward compatibility.
    """
    # Support both new and deprecated field names for backward compatibility
    provider_type = getattr(cfg, "speaker_detector_provider", None) or getattr(
        cfg, "speaker_detector_type", "spacy"
    )

    # Handle deprecated "ner" alias
    if provider_type == "ner":
        provider_type = "spacy"

    if provider_type == "spacy":
        from ..ml.ml_provider import MLProvider

        return MLProvider(cfg)
    elif provider_type == "openai":
        from ..openai.openai_provider import OpenAIProvider

        return OpenAIProvider(cfg)
    else:
        raise ValueError(
            f"Unsupported speaker detector type: {provider_type}. "
            "Supported types: 'spacy', 'openai' (deprecated: 'ner')"
        )

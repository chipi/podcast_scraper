"""Factory for creating summarization providers.

This module provides a factory function to create summarization
providers based on configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from podcast_scraper import config
    from podcast_scraper.summarization.base import SummarizationProvider


def create_summarization_provider(cfg: config.Config) -> SummarizationProvider:
    """Create a summarization provider based on configuration.

    Args:
        cfg: Configuration object

    Returns:
        SummarizationProvider instance

    Raises:
        ValueError: If provider type is not supported

    Note:
        Stage 0: Currently returns NotImplementedError.
        Implementations will be added in later stages.
    """
    # Stage 0: Factory is empty - implementations will be added in later stages
    raise NotImplementedError(
        "Summarization provider factory not yet implemented. "
        "This will be implemented in Stage 4."
    )

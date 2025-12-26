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
        Stage 4: Returns TransformersSummarizationProvider for "local" provider type.
    """
    provider_type = cfg.summary_provider

    if provider_type == "local":
        from .local_provider import TransformersSummarizationProvider

        return TransformersSummarizationProvider(cfg)
    elif provider_type == "openai":
        from .openai_provider import OpenAISummarizationProvider

        return OpenAISummarizationProvider(cfg)
    else:
        raise ValueError(
            f"Unsupported summarization provider: {provider_type}. "
            "Supported providers: 'local', 'openai'"
        )

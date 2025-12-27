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
        NotImplementedError: If provider type is accepted but not yet implemented

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
    elif provider_type == "anthropic":
        raise NotImplementedError(
            "Anthropic summarization provider is not yet implemented. "
            "Currently supported providers: 'local', 'openai'. "
            "Please use 'local' or 'openai' for now."
        )
    else:
        raise ValueError(
            f"Unsupported summarization provider: {provider_type}. "
            "Supported providers: 'local', 'openai'. "
            "Note: 'anthropic' is accepted in configuration but not yet implemented."
        )

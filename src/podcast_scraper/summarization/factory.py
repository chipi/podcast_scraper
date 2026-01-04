"""Factory for creating summarization providers.

This module provides a factory function to create summarization
providers based on configuration.
"""

from __future__ import annotations

from typing import cast, TYPE_CHECKING

if TYPE_CHECKING:
    from podcast_scraper import config
    from podcast_scraper.summarization.base import SummarizationProvider
else:
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
        Returns MLProvider for "transformers" provider type (unified ML provider).
        Returns OpenAIProvider for "openai" provider type (unified OpenAI provider).
        Deprecated: "local" is accepted as alias for "transformers" for backward compatibility.
        Reuses preloaded MLProvider instance if available (from early preloading).
    """
    provider_type = cfg.summary_provider

    # Handle deprecated "local" alias
    if provider_type == "local":
        provider_type = "transformers"

    if provider_type == "transformers":
        # Check for preloaded MLProvider instance (from early preloading)
        try:
            from ..workflow import _preloaded_ml_provider

            if _preloaded_ml_provider is not None:
                return cast(SummarizationProvider, _preloaded_ml_provider)
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
            f"Unsupported summarization provider: {provider_type}. "
            "Supported providers: 'transformers', 'openai' (deprecated: 'local')."
        )

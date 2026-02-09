"""LLM-based transcript cleaning (semantic filtering).

This module provides LLM-powered semantic cleaning that can identify and remove
sponsor segments, advertisements, intros, outros, and meta-commentary using
natural language understanding rather than pattern matching.
"""

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)


class LLMBasedCleaner:
    """LLM-powered semantic transcript cleaner.

    Uses instruction-tuned LLMs to semantically filter unwanted content from
    podcast transcripts. This is smarter than pattern-based cleaning but slower
    and requires API calls (costs tokens).

    This cleaner should typically be used in combination with pattern-based
    cleaning (see HybridCleaner) for optimal cost/performance balance.
    """

    def __init__(self):
        """Initialize LLM-based cleaner."""
        pass

    def clean(self, text: str, provider: Any) -> str:
        """Clean transcript using LLM for semantic filtering.

        Args:
            text: Transcript text to clean (should already be pattern-cleaned)
            provider: Provider instance that supports semantic cleaning
                (must have clean_transcript() method)

        Returns:
            Cleaned transcript text

        Raises:
            AttributeError: If provider doesn't support semantic cleaning
            RuntimeError: If cleaning fails
        """
        if not hasattr(provider, "clean_transcript"):
            raise AttributeError(
                f"Provider {type(provider).__name__} does not support semantic cleaning. "
                "Provider must implement clean_transcript() method."
            )

        try:
            cleaned = provider.clean_transcript(text)
            if not cleaned or not isinstance(cleaned, str):
                logger.warning("LLM cleaning returned empty or invalid result, using original text")
                return text
            return cast(str, cleaned)
        except Exception as e:
            logger.error(f"LLM cleaning failed: {e}", exc_info=True)
            # Fallback to original text if cleaning fails
            return text

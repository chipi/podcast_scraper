"""LLM-based transcript cleaning (semantic filtering).

This module provides LLM-powered semantic cleaning that can identify and remove
sponsor segments, advertisements, intros, outros, and meta-commentary using
natural language understanding rather than pattern matching.
"""

import logging
from typing import Any, cast, Optional

from ..utils.log_redaction import format_exception_for_log

logger = logging.getLogger(__name__)

# If pattern-cleaned input is at least this long, reject LLM output that shrinks the
# transcript by too much (models sometimes return a short unrelated paragraph instead
# of the full cleaned transcript, which breaks downstream summarization).
_MIN_INPUT_CHARS_FOR_LENGTH_GUARD = 2000
_MIN_OUTPUT_TO_INPUT_RATIO = 0.20


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

    def clean(
        self,
        text: str,
        provider: Any,
        pipeline_metrics: Optional[Any] = None,
    ) -> str:
        """Clean transcript using LLM for semantic filtering.

        Args:
            text: Transcript text to clean (should already be pattern-cleaned)
            provider: Provider instance that supports semantic cleaning
                (must have clean_transcript() method)
            pipeline_metrics: Optional pipeline ``Metrics`` for LLM token accounting

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
            cleaned = provider.clean_transcript(text, pipeline_metrics=pipeline_metrics)
            if not cleaned or not isinstance(cleaned, str):
                logger.warning("LLM cleaning returned empty or invalid result, using original text")
                return text
            stripped = cleaned.strip()
            input_len = len(text)
            if input_len >= _MIN_INPUT_CHARS_FOR_LENGTH_GUARD and input_len > 0:
                ratio = len(stripped) / input_len
                if ratio < _MIN_OUTPUT_TO_INPUT_RATIO:
                    out_len = len(stripped)
                    logger.debug(
                        "LLM cleaning length guard: input_len=%d output_len=%d ratio=%.4f "
                        "threshold=%.2f (reject LLM output, keep pattern-cleaned text)",
                        input_len,
                        out_len,
                        ratio,
                        _MIN_OUTPUT_TO_INPUT_RATIO,
                    )
                    logger.warning(
                        "LLM cleaning shortened transcript excessively "
                        "(output/input length ratio=%.3f); using pattern-cleaned text",
                        ratio,
                    )
                    return text
            return cast(str, cleaned)
        except Exception as e:
            logger.error("LLM cleaning failed: %s", format_exception_for_log(e), exc_info=True)
            # Fallback to original text if cleaning fails
            return text

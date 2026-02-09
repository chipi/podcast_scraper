"""Hybrid transcript cleaning (pattern-based + conditional LLM-based).

This module provides a cost-efficient hybrid cleaning approach that combines
fast pattern-based cleaning with smart LLM-based cleaning, only using LLM
when pattern-based cleaning likely failed.
"""

import logging
from typing import Any, cast, Optional

from .llm_based import LLMBasedCleaner
from .pattern_based import PatternBasedCleaner

logger = logging.getLogger(__name__)


class HybridCleaner:
    """Hybrid cleaner combining pattern-based and conditional LLM-based cleaning.

    This cleaner uses a two-stage approach:
    1. Fast pattern-based removal of obvious sponsors/outros (always runs)
    2. Conditional LLM semantic filtering (only when pattern-based likely failed)

    This provides the best balance of speed, cost, and quality:
    - Most transcripts (~70-90%) only need pattern-based cleaning (free, fast)
    - Only problematic transcripts (~10-30%) need LLM cleaning (paid, slower)
    - Total cost impact: ~10-20% increase (not 2x) due to conditional logic
    """

    def __init__(self):
        """Initialize hybrid cleaner."""
        self.pattern_cleaner = PatternBasedCleaner()
        self.llm_cleaner = LLMBasedCleaner()

    def clean(self, text: str, provider: Optional[Any] = None) -> str:
        """Clean transcript using hybrid approach.

        Args:
            text: Raw transcript text
            provider: Optional provider instance for LLM cleaning.
                If None, only pattern-based cleaning is used.

        Returns:
            Cleaned transcript text
        """
        # Stage 1: Fast pattern-based removal of obvious sponsors
        cleaned = self.pattern_cleaner.clean(text)

        # Stage 2: Conditional LLM semantic filtering
        if provider is not None and self._needs_llm_cleaning(text, cleaned, provider):
            logger.debug("Pattern-based cleaning likely insufficient, using LLM cleaning")
            try:
                cleaned = self.llm_cleaner.clean(cleaned, provider)
            except Exception as e:
                logger.warning(f"LLM cleaning failed, using pattern-cleaned text: {e}")
                # Continue with pattern-cleaned text if LLM fails

        return cast(str, cleaned)

    def _needs_llm_cleaning(self, original: str, cleaned: str, provider: Any) -> bool:
        """Heuristic to detect if pattern-based cleaning was insufficient.

        Args:
            original: Original transcript text
            cleaned: Pattern-cleaned transcript text
            provider: Provider instance

        Returns:
            True if LLM cleaning should be used, False otherwise
        """
        # Check if provider supports semantic cleaning
        if not hasattr(provider, "clean_transcript"):
            return False

        # Heuristic 1: Text length reduction too small (< 5%)
        # If pattern-based didn't remove much, it might have missed sponsors
        if original:
            reduction_ratio = (len(original) - len(cleaned)) / len(original)
            if reduction_ratio < 0.05:
                logger.debug(
                    f"Low reduction ratio ({reduction_ratio:.2%}), "
                    "pattern-based may have missed content"
                )
                return True

        # Heuristic 2: Sponsor keywords still present after pattern cleaning
        sponsor_keywords = [
            "sponsored by",
            "brought to you by",
            "advertisement",
            "promo code",
            "discount code",
            "use code",
            "visit our sponsor",
            "thanks to our sponsor",
            "this episode is sponsored",
        ]
        cleaned_lower = cleaned.lower()
        keyword_matches = sum(1 for keyword in sponsor_keywords if keyword in cleaned_lower)
        if keyword_matches >= 2:  # Multiple sponsor keywords detected
            logger.debug(
                f"Found {keyword_matches} sponsor keywords, "
                "pattern-based may have missed sponsor content"
            )
            return True

        # Heuristic 3: High density of promotional phrases
        # Count promotional phrases per 1000 characters
        promotional_phrases = [
            "check out",
            "visit",
            "go to",
            "sign up",
            "subscribe",
            "follow us",
            "rate and review",
            "leave a review",
        ]
        phrase_count = sum(1 for phrase in promotional_phrases if phrase in cleaned_lower)
        if len(cleaned) > 0:
            phrase_density = (phrase_count / len(cleaned)) * 1000
            if phrase_density > 5:  # More than 5 promotional phrases per 1000 chars
                logger.debug(
                    f"High promotional phrase density ({phrase_density:.1f}/1000 chars), "
                    "may need LLM cleaning"
                )
                return True

        # Pattern-based cleaning appears sufficient
        return False

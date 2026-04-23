"""Pattern-based transcript cleaning (default implementation).

This module provides the default pattern-based cleaning implementation
that is used by all providers unless they provide a custom cleaner.
"""

import logging

logger = logging.getLogger(__name__)


class PatternBasedCleaner:
    """Default pattern-based transcript cleaner.

    This is the standard cleaning implementation used by all providers
    unless they provide a custom cleaner. Uses pattern matching to remove
    timestamps, normalize speakers, remove sponsor blocks, etc.
    """

    def __init__(self):
        """Initialize pattern-based cleaner."""
        pass

    def clean(self, text: str) -> str:
        """Clean transcript using pattern-based rules.

        Wraps :func:`preprocessing.clean_for_summarization` for the baseline
        cleaning (timestamps, speaker normalization, phrase-trigger sponsor
        removal), then applies :func:`gi.ad_regions.excise_ad_regions` to
        remove pre-roll and post-roll ad blocks detected by pattern density
        (#663). The density pass catches modern sponsor reads (e.g.,
        ``Visit ramp.com``, ``Learn more at rogo.ai``) that the four
        built-in trigger phrases in ``remove_sponsor_blocks`` miss.

        Args:
            text: Raw transcript text

        Returns:
            Cleaned transcript text
        """
        # Import here to avoid circular dependency
        from .. import preprocessing
        from ..gi.ad_regions import excise_ad_regions

        cleaned: str = preprocessing.clean_for_summarization(text)  # type: ignore[attr-defined]
        cleaned, _, meta = excise_ad_regions(cleaned)
        if meta.chars_removed:
            logger.debug(
                "PatternBasedCleaner: excised %d ad chars " "(preroll=%s, postroll=%s, hits=%d/%d)",
                meta.chars_removed,
                meta.preroll_cut_end,
                meta.postroll_cut_start,
                meta.preroll_pattern_hits,
                meta.postroll_pattern_hits,
            )
        return cleaned

    def remove_sponsors(self, text: str) -> str:
        """Remove sponsor blocks from transcript.

        Args:
            text: Transcript text

        Returns:
            Text with sponsor blocks removed
        """
        from .. import preprocessing

        result: str = preprocessing.remove_sponsor_blocks(text)  # type: ignore[attr-defined]
        return result

    def remove_outros(self, text: str) -> str:
        """Remove outro blocks from transcript.

        Args:
            text: Transcript text

        Returns:
            Text with outro blocks removed
        """
        from .. import preprocessing

        result: str = preprocessing.remove_outro_blocks(text)  # type: ignore[attr-defined]
        return result

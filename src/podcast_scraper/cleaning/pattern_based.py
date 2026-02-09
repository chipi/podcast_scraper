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

        This is a wrapper around the existing clean_for_summarization()
        function from preprocessing.core to maintain backward compatibility.

        Args:
            text: Raw transcript text

        Returns:
            Cleaned transcript text
        """
        # Import here to avoid circular dependency
        from .. import preprocessing

        # Use existing clean_for_summarization() which includes all standard cleaning steps
        cleaned: str = preprocessing.clean_for_summarization(text)  # type: ignore[attr-defined]
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

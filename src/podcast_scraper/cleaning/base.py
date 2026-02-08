"""Base protocol for transcript cleaning processors.

This module defines the protocol interface for transcript cleaning,
enabling provider-specific cleaning implementations.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class TranscriptCleaningProcessor(Protocol):
    """Protocol for transcript cleaning processors.

    Providers can implement custom cleaning logic by implementing this protocol.
    The default implementation uses pattern-based cleaning.
    """

    def clean(self, text: str) -> str:
        """Clean transcript text for summarization.

        Args:
            text: Raw transcript text

        Returns:
            Cleaned transcript text
        """
        ...

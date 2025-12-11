"""SummarizationProvider protocol definition.

This module defines the protocol that all summarization providers must implement.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol


class SummarizationProvider(Protocol):
    """Protocol for summarization providers.

    All summarization providers must implement this protocol to ensure
    consistent interface across different implementations (transformers, OpenAI, etc.).
    """

    def summarize(
        self,
        text: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Summarize text.

        Args:
            text: Transcript text to summarize
            episode_title: Optional episode title
            episode_description: Optional episode description
            params: Optional parameters (max_length, min_length, etc.)

        Returns:
            Dictionary with summary results:
            {
                "summary": str,
                "summary_short": Optional[str],
                "metadata": {...}
            }

        Raises:
            ValueError: If summarization fails
        """
        ...

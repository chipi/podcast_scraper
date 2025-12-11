"""SpeakerDetector protocol definition.

This module defines the protocol that all speaker detection providers must implement.
"""

from __future__ import annotations

from typing import Protocol, Set, Tuple

from podcast_scraper import models


class SpeakerDetector(Protocol):
    """Protocol for speaker detection providers.

    All speaker detection providers must implement this protocol to ensure
    consistent interface across different implementations (NER, OpenAI, etc.).
    """

    def detect_speakers(
        self,
        episode_title: str,
        episode_description: str | None,
        known_hosts: Set[str],
    ) -> Tuple[list[str], Set[str], bool]:
        """Detect speaker names from episode metadata.

        Args:
            episode_title: Episode title
            episode_description: Optional episode description
            known_hosts: Set of known host names (for context)

        Returns:
            Tuple of:
            - List of detected speaker names
            - Set of detected host names
            - Success flag (True if detection succeeded)
        """
        ...

    def analyze_patterns(
        self,
        episodes: list[models.Episode],
        known_hosts: Set[str],
    ) -> dict[str, object] | None:
        """Analyze patterns across multiple episodes (optional).

        Args:
            episodes: List of episodes to analyze
            known_hosts: Set of known host names

        Returns:
            Optional dictionary with pattern analysis results, or None
            if pattern analysis is not supported.
        """
        ...

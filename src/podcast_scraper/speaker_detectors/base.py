"""SpeakerDetector protocol definition.

This module defines the protocol that all speaker detection providers must implement.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from podcast_scraper.models import Episode
else:
    from podcast_scraper import models

    Episode = models.Episode  # type: ignore[assignment]


@runtime_checkable
class SpeakerDetector(Protocol):
    """Protocol for speaker detection providers.

    All speaker detection providers must implement this protocol to ensure
    consistent interface across different implementations (NER, OpenAI, etc.).
    """

    def initialize(self) -> None:
        """Initialize provider (load models, setup API clients, etc.).

        This method should be called before other methods are used.
        It may be called multiple times safely (idempotent).
        """
        ...

    def detect_hosts(
        self,
        feed_title: str | None,
        feed_description: str | None,
        feed_authors: list[str] | None = None,
    ) -> Set[str]:
        """Detect host names from feed-level metadata.

        Args:
            feed_title: Feed title
            feed_description: Optional feed description
            feed_authors: Optional list of author names from RSS feed

        Returns:
            Set of detected host names
        """
        ...

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
        episodes: list[Episode],  # type: ignore[valid-type]
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

    def cleanup(self) -> None:
        """Cleanup provider resources (unload models, close connections, etc.).

        This method should be called when the provider is no longer needed.
        It may be called multiple times safely (idempotent).
        """
        ...

    def clear_cache(self) -> None:
        """Clear any cached models or resources.

        This method clears module-level caches (e.g., spaCy model cache).
        Useful for freeing memory or forcing model reloads.
        It may be called multiple times safely (idempotent).

        Note:
            For providers that don't use cached models (e.g., API-based providers),
            this method may be a no-op.
        """
        ...

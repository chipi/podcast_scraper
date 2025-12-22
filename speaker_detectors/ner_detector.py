"""NER-based speaker detection provider implementation.

This module provides a SpeakerDetector implementation using spaCy NER
for automatic speaker name detection from episode metadata.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

# Import speaker_detection functions (keeping existing implementation)
from .. import config, models, speaker_detection

logger = logging.getLogger(__name__)


class NERSpeakerDetector:
    """NER-based speaker detection provider.

    This provider uses spaCy NER for automatic speaker name detection from
    episode metadata. It implements the SpeakerDetector protocol.
    """

    def __init__(self, cfg: config.Config):
        """Initialize NER speaker detection provider.

        Args:
            cfg: Configuration object with auto_speakers and ner_model settings
        """
        self.cfg = cfg
        self._nlp: Optional[Any] = None
        self._heuristics: Optional[Dict[str, Any]] = None

    def initialize(self) -> None:
        """Initialize spaCy NER model.

        This method loads the spaCy model using the configuration.
        It should be called before detect_speakers() is used.
        """
        if self._nlp is not None:
            return

        if not self.cfg.auto_speakers:
            logger.debug("Auto-speakers disabled, NER model not loaded")
            return

        logger.debug("Initializing NER speaker detection provider (model: %s)", self.cfg.ner_model)
        self._nlp = speaker_detection.get_ner_model(self.cfg)
        if self._nlp is None:
            logger.warning(
                "Failed to load spaCy NER model. Speaker detection may be limited. " "Model: %s",
                self.cfg.ner_model or "default",
            )
        else:
            logger.debug("NER speaker detection provider initialized successfully")

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
            - Set of detected host names (subset of known_hosts)
            - Success flag (True if detection succeeded)
        """
        # Ensure model is loaded
        if self._nlp is None:
            self.initialize()

        # Use detect_speaker_names with adapted parameters
        # Note: detect_speaker_names expects cfg, cached_hosts, and heuristics
        # We adapt known_hosts to cached_hosts and use internal heuristics
        speaker_names, detected_hosts_set, detection_succeeded = (
            speaker_detection.detect_speaker_names(
                episode_title=episode_title,
                episode_description=episode_description,
                cfg=self.cfg,
                known_hosts=None,  # Use cached_hosts instead
                cached_hosts=known_hosts,  # Map known_hosts to cached_hosts
                heuristics=self._heuristics,
            )
        )

        return speaker_names, detected_hosts_set, detection_succeeded

    def detect_hosts(
        self,
        feed_title: Optional[str],
        feed_description: Optional[str],
        feed_authors: Optional[List[str]] = None,
    ) -> Set[str]:
        """Detect host names from feed-level metadata.

        This is a convenience method that wraps detect_hosts_from_feed().

        Args:
            feed_title: Feed title
            feed_description: Feed description (optional)
            feed_authors: List of author names from RSS feed (optional, preferred source)

        Returns:
            Set of detected host names
        """
        # Ensure model is loaded
        if self._nlp is None:
            self.initialize()

        return speaker_detection.detect_hosts_from_feed(
            feed_title=feed_title,
            feed_description=feed_description,
            feed_authors=feed_authors,
            nlp=self._nlp,
        )

    def analyze_patterns(
        self,
        episodes: list[models.Episode],
        known_hosts: Set[str],
    ) -> dict[str, object] | None:
        """Analyze patterns across multiple episodes.

        Args:
            episodes: List of episodes to analyze
            known_hosts: Set of known host names

        Returns:
            Dictionary with pattern analysis results, or None if analysis fails
        """
        # Ensure model is loaded
        if self._nlp is None:
            self.initialize()

        if not self._nlp:
            return None

        # Analyze patterns and cache heuristics for use in detect_speakers
        self._heuristics = speaker_detection.analyze_episode_patterns(
            episodes=episodes,
            nlp=self._nlp,
            cached_hosts=known_hosts,
            sample_size=speaker_detection.DEFAULT_SAMPLE_SIZE,
        )

        return self._heuristics

    @property
    def nlp(self) -> Optional[Any]:
        """Get the loaded spaCy NER model (for backward compatibility).

        Returns:
            Loaded spaCy nlp object or None if not initialized
        """
        return self._nlp

    @property
    def heuristics(self) -> Optional[Dict[str, Any]]:
        """Get cached heuristics from pattern analysis.

        Returns:
            Heuristics dictionary or None if not analyzed
        """
        return self._heuristics

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized.

        Returns:
            True if provider is initialized, False otherwise
        """
        return self._nlp is not None

"""Host detection from feed metadata and transcript intro."""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Set

from .entities import extract_person_entities as _extract_person_entities_direct

logger = logging.getLogger(__name__)


def _extract_person_entities(text: str, nlp: Any) -> list[tuple[str, float]]:
    """Resolve extract_person_entities via public wrapper when loaded (patchable in tests)."""
    try:
        from podcast_scraper.providers.ml import speaker_detection

        return speaker_detection.extract_person_entities(text, nlp)
    except ImportError:
        return _extract_person_entities_direct(text, nlp)


def _log(logger_method: str, message: str, *args: object) -> None:
    """Emit log via wrapper module logger when available (patchable in tests)."""
    try:
        from podcast_scraper.providers.ml import speaker_detection

        getattr(speaker_detection.logger, logger_method)(message, *args)
    except ImportError:
        getattr(logger, logger_method)(message, *args)


def detect_hosts_from_transcript_intro(
    transcript_text: str,
    nlp: Optional[Any] = None,
    intro_duration_seconds: int = 120,
    words_per_second: float = 2.5,
) -> Set[str]:
    """Detect host names from transcript intro patterns (first 60-120 seconds)."""
    if not transcript_text or not nlp:
        return set()

    intro_word_count = int(intro_duration_seconds * words_per_second)
    words = transcript_text.split()[:intro_word_count]
    intro_text = " ".join(words)

    intro_patterns = [
        r"I'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"This is\s+[^.]+\s+I'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"Welcome to\s+[^.]+\s+I'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    ]

    detected_names = set()
    for pattern in intro_patterns:
        matches = re.finditer(pattern, intro_text, re.IGNORECASE)
        for match in matches:
            name = match.group(1).strip()
            if name and len(name) > 2 and name.lower() not in ["the", "this", "that"]:
                detected_names.add(name)

    if nlp:
        intro_persons = _extract_person_entities(intro_text, nlp)
        for name, _ in intro_persons:
            detected_names.add(name)

    return detected_names


def detect_hosts_from_feed(
    feed_title: Optional[str],
    feed_description: Optional[str],
    feed_authors: Optional[List[str]] = None,
    nlp: Optional[Any] = None,
) -> Set[str]:
    """Detect host names from feed-level metadata."""
    hosts: Set[str] = set()

    if feed_authors:
        for author in feed_authors:
            if author and author.strip():
                author_clean = author.strip()
                if "<" in author_clean and ">" in author_clean:
                    author_clean = author_clean.split("<")[0].strip()
                if author_clean:
                    is_likely_org = (
                        len(author_clean) <= 10
                        and author_clean.isupper()
                        and " " not in author_clean
                    )
                    if is_likely_org:
                        logger.debug(
                            "RSS author '%s' appears to be an organization name, "
                            "treating as publisher metadata rather than host",
                            author_clean,
                        )
                    else:
                        hosts.add(author_clean)
        if hosts:
            logger.debug(
                "Detected hosts from RSS author tags (author/itunes:author/itunes:owner): %s",
                list(hosts),
            )
            return hosts
        if feed_authors:
            _log(
                "info",
                "All RSS author(s) treated as organisation(s); host detection will use "
                "NER from feed title/description, episode-level authors, or config known_hosts",
            )

    if nlp:
        if feed_title:
            title_persons = _extract_person_entities(feed_title, nlp)
            hosts.update(name for name, _ in title_persons)
        if feed_description:
            desc_persons = _extract_person_entities(feed_description, nlp)
            hosts.update(name for name, _ in desc_persons)
        if hosts:
            logger.debug("Detected hosts via NER from feed metadata: %s", list(hosts))

    return hosts

"""Named Entity Recognition (NER) for automatic speaker name detection from episode metadata."""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any, List, Optional, Set, Tuple

import spacy

from . import config

logger = logging.getLogger(__name__)

# Default speaker names when detection fails
DEFAULT_SPEAKER_NAMES = ["Host", "Guest"]


def _load_spacy_model(model_name: str) -> Optional[Any]:
    """Load spaCy model, automatically downloading if missing.

    Similar to Whisper's automatic model download, this function will attempt
    to download the model if it's not found locally.

    Args:
        model_name: Name of the spaCy model to load (e.g., 'en_core_web_sm')

    Returns:
        Loaded spaCy nlp object or None if download/load fails
    """
    try:
        nlp = spacy.load(model_name)
        logger.debug("Loaded spaCy model: %s", model_name)
        return nlp
    except OSError:
        logger.info("spaCy model '%s' not found locally, attempting to download...", model_name)
        try:
            # Use subprocess to call 'python -m spacy download' (most reliable method)
            # This ensures we use the same Python interpreter and environment
            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("Successfully downloaded spaCy model: %s", model_name)
            # Now try loading again
            nlp = spacy.load(model_name)
            logger.debug("Loaded spaCy model after download: %s", model_name)
            return nlp
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Failed to download spaCy model '%s': %s. Output: %s",
                model_name,
                exc,
                exc.stderr or exc.stdout or "",
            )
            logger.warning(
                "You can manually install with: python -m spacy download %s", model_name
            )
            return None
        except OSError as exc:
            logger.error(
                "Failed to load spaCy model '%s' after download attempt: %s", model_name, exc
            )
            return None


def _get_ner_model(cfg: config.Config) -> Optional[Any]:
    """Get the appropriate spaCy NER model based on configuration."""
    if not cfg.auto_speakers:
        return None

    model_name = cfg.ner_model
    if not model_name:
        # Derive default model from language
        if cfg.language == "en":
            model_name = config.DEFAULT_NER_MODEL
        else:
            # For other languages, try to construct model name (e.g., "fr" -> "fr_core_news_sm")
            # This is a simple heuristic; users can override with --ner-model
            logger.debug("No default NER model for language '%s', skipping detection", cfg.language)
            return None

    return _load_spacy_model(model_name)


def _extract_person_entities(text: str, nlp: Any) -> List[str]:
    """Extract PERSON entities from text using spaCy NER."""
    if not text or not nlp:
        return []

    try:
        doc = nlp(text)
        persons = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                if name and name not in persons:
                    persons.append(name)
        return persons
    except Exception as exc:
        logger.debug("Error extracting PERSON entities: %s", exc)
        return []


def _detect_hosts_from_feed(
    feed_title: Optional[str], feed_description: Optional[str], nlp: Any
) -> Set[str]:
    """Detect host names from feed-level metadata."""
    hosts: Set[str] = set()
    if feed_title:
        hosts.update(_extract_person_entities(feed_title, nlp))
    if feed_description:
        hosts.update(_extract_person_entities(feed_description, nlp))
    return hosts


def detect_speaker_names(
    episode_title: str,
    episode_description: Optional[str],
    feed_title: Optional[str] = None,
    feed_description: Optional[str] = None,
    cfg: Optional[config.Config] = None,
    known_hosts: Optional[Set[str]] = None,
    cached_hosts: Optional[Set[str]] = None,
) -> Tuple[List[str], Set[str]]:
    """
    Detect speaker names from episode metadata using NER.

    Args:
        episode_title: Episode title
        episode_description: Episode description (optional)
        feed_title: Feed title for host detection (optional)
        feed_description: Feed description for host detection (optional)
        cfg: Configuration object
        known_hosts: Manually specified host names (optional)
        cached_hosts: Previously detected hosts to reuse (optional)

    Returns:
        Tuple of (speaker_names_list, detected_hosts_set)
    """
    if not cfg or not cfg.auto_speakers:
        return DEFAULT_SPEAKER_NAMES.copy(), set()

    nlp = _get_ner_model(cfg)
    if not nlp:
        logger.debug("spaCy model not available, using default speaker names")
        return DEFAULT_SPEAKER_NAMES.copy(), set()

    # Detect hosts from feed metadata or use cached/known hosts
    hosts: Set[str] = set()
    if known_hosts:
        hosts.update(known_hosts)
    elif cached_hosts:
        hosts.update(cached_hosts)
    else:
        hosts.update(_detect_hosts_from_feed(feed_title, feed_description, nlp))

    # Extract all PERSON entities from episode
    episode_persons: List[str] = []
    episode_persons.extend(_extract_person_entities(episode_title, nlp))
    if episode_description:
        episode_persons.extend(_extract_person_entities(episode_description, nlp))

    # Remove hosts from episode persons to get guests
    guests = [p for p in episode_persons if p not in hosts]

    # Cap total names at configured limit
    max_names = config.DEFAULT_MAX_DETECTED_NAMES
    all_names = list(hosts) + guests
    if len(all_names) > max_names:
        # Prioritize hosts, then guests
        capped_names = list(hosts)[:max_names] + guests[: max_names - len(hosts)]
        logger.debug(
            "Capped detected names from %d to %d (max=%d)",
            len(all_names),
            len(capped_names),
            max_names,
        )
        all_names = capped_names

    # If no guests detected, use host-only labels
    if not guests and hosts:
        speaker_names = [
            f"Host {i+1}" if i > 0 else "Host" for i in range(min(len(hosts), max_names))
        ]
    elif not all_names:
        # Fallback to defaults
        speaker_names = DEFAULT_SPEAKER_NAMES.copy()
    else:
        # Combine hosts and guests, prioritizing hosts
        speaker_names = list(hosts)[:max_names] + guests[: max_names - len(hosts)]
        if len(speaker_names) < 2:
            # Ensure at least 2 speakers for screenplay formatting
            speaker_names.extend(DEFAULT_SPEAKER_NAMES[len(speaker_names) :])

    logger.info(
        "Detected speaker names: %s (hosts=%s, guests=%s)",
        speaker_names,
        list(hosts),
        guests,
    )

    return speaker_names[:max_names], hosts


__all__ = ["detect_speaker_names", "DEFAULT_SPEAKER_NAMES"]

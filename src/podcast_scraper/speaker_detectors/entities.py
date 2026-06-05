"""PERSON entity extraction from text via spaCy NER."""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Set, Tuple

from .constants import MIN_SEGMENT_LENGTH, PATTERN_BASED_CONFIDENCE_SCORE
from .normalization import (
    _extract_confidence_score,
    _sanitize_person_name,
    _validate_person_entity,
)

logger = logging.getLogger(__name__)


def _extract_entities_from_doc(
    doc: Any, seen_raw_names: Set[str], seen_sanitized_names: Set[str]
) -> List[Tuple[str, float]]:
    """Extract PERSON entities from a spaCy document."""
    persons = []
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue

        raw_name = ent.text.strip()

        if raw_name in seen_raw_names:
            continue

        if not _validate_person_entity(raw_name):
            continue

        sanitized_name = _sanitize_person_name(raw_name)
        if not sanitized_name:
            continue

        sanitized_lower = sanitized_name.lower()
        if sanitized_lower in seen_sanitized_names:
            continue

        seen_raw_names.add(raw_name)
        seen_sanitized_names.add(sanitized_lower)

        confidence = _extract_confidence_score(ent)
        persons.append((sanitized_name, confidence))

    return persons


def _split_text_on_separators(text: str) -> Tuple[List[str], Optional[str]]:
    """Split text on common separators used in episode titles."""
    separators = ["|", "—", "–", " - "]
    segments = [text]
    last_segment = None

    for sep in separators:
        if sep in text:
            segments = [s.strip() for s in text.split(sep)]
            last_segment = segments[-1] if segments else None
            break

    return segments, last_segment


def _extract_entities_from_segments(
    segments: List[str],
    nlp: Any,
    seen_raw_names: Set[str],
    seen_sanitized_names: Set[str],
) -> List[Tuple[str, float]]:
    """Extract PERSON entities from text segments, prioritizing last segment."""
    persons = []
    for segment in reversed(segments):
        if not segment or len(segment) < MIN_SEGMENT_LENGTH:
            continue

        segment_doc = nlp(segment)
        segment_persons = _extract_entities_from_doc(
            segment_doc, seen_raw_names, seen_sanitized_names
        )
        persons.extend(segment_persons)

        if persons:
            break

    return persons


def _pattern_based_fallback(
    last_segment: str, seen_sanitized_names: Set[str]
) -> Optional[Tuple[str, float]]:
    """Pattern-based fallback for name extraction when NER fails."""
    if not last_segment:
        return None

    name_pattern = r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}$"
    if not re.match(name_pattern, last_segment):
        return None

    common_phrases = {
        "guest",
        "host",
        "episode",
        "title",
        "interview",
        "conversation",
    }
    last_segment_lower = last_segment.lower()
    if any(phrase in last_segment_lower for phrase in common_phrases):
        return None

    sanitized_name = _sanitize_person_name(last_segment)
    if not sanitized_name:
        return None

    sanitized_lower = sanitized_name.lower()
    if sanitized_lower in seen_sanitized_names:
        return None

    logger.debug(
        "Pattern-based fallback: extracted '%s' from last segment '%s'",
        sanitized_name,
        last_segment,
    )
    return (sanitized_name, PATTERN_BASED_CONFIDENCE_SCORE)


def extract_person_entities(text: str, nlp: Any) -> List[Tuple[str, float]]:
    """Extract PERSON entities from text using spaCy NER with confidence scores."""
    if not text or not nlp:
        return []

    try:
        seen_raw_names: Set[str] = set()
        seen_sanitized_names: Set[str] = set()

        doc = nlp(text)
        persons = _extract_entities_from_doc(doc, seen_raw_names, seen_sanitized_names)

        if not persons:
            segments, last_segment = _split_text_on_separators(text)
            persons = _extract_entities_from_segments(
                segments, nlp, seen_raw_names, seen_sanitized_names
            )

            if not persons and last_segment:
                pattern_result = _pattern_based_fallback(last_segment, seen_sanitized_names)
                if pattern_result:
                    sanitized_name, confidence = pattern_result
                    seen_sanitized_names.add(sanitized_name.lower())
                    persons.append((sanitized_name, confidence))

        return persons
    except Exception as exc:
        logger.debug("Error extracting PERSON entities: %s", exc)
        return []

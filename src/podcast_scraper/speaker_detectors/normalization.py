"""Name sanitization, validation, and default-speaker filtering."""

from __future__ import annotations

import re
from typing import Any, List, Optional

from .. import config_constants
from .constants import (
    DEFAULT_CONFIDENCE_SCORE,
    DEFAULT_SPEAKER_NAMES,
    MIN_NAME_LENGTH,
    MIN_RAW_NAME_LENGTH,
)


def is_default_speaker_name(name: str) -> bool:
    """Check if a speaker name is a default placeholder."""
    return name in DEFAULT_SPEAKER_NAMES or name == config_constants.LEGACY_PLACEHOLDER_GUEST


def filter_default_speaker_names(names: List[str]) -> List[str]:
    """Filter out default speaker names from a list."""
    return [name for name in names if not is_default_speaker_name(name)]


def _sanitize_person_name(name: str) -> Optional[str]:
    """Sanitize a person name by removing non-letter characters and normalizing."""
    if not name:
        return None

    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"[,.;:!?]+$", "", name)
    name = re.sub(r"^[,.;:!?]+", "", name)
    name = name.strip()
    name = re.sub(r"[^\w\s\-\']+", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    if not name or len(name) < MIN_NAME_LENGTH:
        return None

    if not re.search(r"[a-zA-Z]", name):
        return None

    return name


def _validate_person_entity(raw_name: str) -> bool:
    """Validate that a raw entity name is likely a person."""
    if not raw_name or len(raw_name) < MIN_RAW_NAME_LENGTH:
        return False
    if re.match(r"^\d+$", raw_name) or re.search(r"[<>]", raw_name):
        return False
    return True


def _extract_confidence_score(ent: Any) -> float:
    """Extract confidence score from spaCy entity."""
    if hasattr(ent, "score") and ent.score is not None:
        return float(ent.score)
    if hasattr(ent, "_") and hasattr(ent._, "score") and ent._.score is not None:
        return float(ent._.score)
    return DEFAULT_CONFIDENCE_SCORE

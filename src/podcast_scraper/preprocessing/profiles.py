"""Registered preprocessing profiles for transcript cleaning.

This module implements versioned preprocessing profiles (ADR-029) that
encapsulate cleaning logic into reproducible, versioned configurations.
Each profile specifies exactly which cleaning steps are active, enabling
researchers to isolate variables when comparing models.

See ADR-029 for design rationale.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

# Import preprocessing functions from core.py
from . import core

preprocessing = core

# Registry of available preprocessing profiles
_PROFILE_REGISTRY: Dict[str, Callable[[str], str]] = {}


def register_profile(profile_id: str, cleaning_function: Callable[[str], str]) -> None:
    """Register a preprocessing profile.

    Args:
        profile_id: Unique profile identifier (e.g., "cleaning_v3")
        cleaning_function: Function that takes raw text and returns cleaned text

    Example:
        >>> def my_cleaner(text: str) -> str:
        ...     return text.strip()
        >>> register_profile("my_profile_v1", my_cleaner)
    """
    _PROFILE_REGISTRY[profile_id] = cleaning_function


def get_profile(profile_id: str) -> Optional[Callable[[str], str]]:
    """Get a preprocessing profile by ID.

    Args:
        profile_id: Profile identifier

    Returns:
        Cleaning function or None if profile not found

    Raises:
        ValueError: If profile_id is not registered
    """
    if profile_id not in _PROFILE_REGISTRY:
        raise ValueError(
            f"Preprocessing profile '{profile_id}' not found. "
            f"Available profiles: {list(_PROFILE_REGISTRY.keys())}"
        )
    return _PROFILE_REGISTRY[profile_id]


def list_profiles() -> list[str]:
    """List all registered preprocessing profiles.

    Returns:
        List of profile IDs
    """
    return sorted(_PROFILE_REGISTRY.keys())


def apply_profile(text: str, profile_id: str) -> str:
    """Apply a preprocessing profile to text.

    Args:
        text: Raw transcript text
        profile_id: Profile identifier

    Returns:
        Cleaned text

    Raises:
        ValueError: If profile_id is not registered
    """
    profile = get_profile(profile_id)
    if profile is None:
        raise ValueError(f"Preprocessing profile '{profile_id}' not found")
    return profile(text)


# Define standard preprocessing profiles


def _cleaning_v1(text: str) -> str:
    """Cleaning profile v1: Basic cleaning only.

    - Removes timestamps
    - Normalizes speakers
    - Collapses blank lines
    """
    return str(
        preprocessing.clean_transcript(  # type: ignore[attr-defined]
            text,
            remove_timestamps=True,
            normalize_speakers=True,
            collapse_blank_lines=True,
            remove_fillers=False,
        )
    )


def _cleaning_v2(text: str) -> str:
    """Cleaning profile v2: Basic + sponsor/outro removal.

    - All v1 steps
    - Removes sponsor blocks
    - Removes outro blocks
    """
    cleaned = _cleaning_v1(text)
    cleaned = str(preprocessing.remove_sponsor_blocks(cleaned))  # type: ignore[attr-defined]
    cleaned = str(preprocessing.remove_outro_blocks(cleaned))  # type: ignore[attr-defined]
    return cleaned


def _cleaning_v3(text: str) -> str:
    """Cleaning profile v3: Full cleaning pipeline (current default).

    - All v2 steps
    - Strips credits FIRST (high-confidence summary targets)
    - Strips garbage lines (website boilerplate)
    - Removes summarization artifacts

    This is the current production cleaning pipeline used by clean_for_summarization().
    """
    # Strip credits FIRST - they're high-confidence summary targets if left in
    cleaned = preprocessing.strip_credits(text)

    # Strip garbage lines - website boilerplate
    cleaned = preprocessing.strip_garbage_lines(cleaned)

    # Basic cleaning
    cleaned = preprocessing.clean_transcript(
        cleaned,
        remove_timestamps=True,
        normalize_speakers=True,
        collapse_blank_lines=True,
        remove_fillers=False,
    )

    # Remove sponsor and outro blocks
    cleaned = str(preprocessing.remove_sponsor_blocks(cleaned))  # type: ignore[attr-defined]
    cleaned = str(preprocessing.remove_outro_blocks(cleaned))  # type: ignore[attr-defined]

    # Remove summarization artifacts
    # type: ignore[attr-defined]
    cleaned = str(preprocessing.remove_summarization_artifacts(cleaned))

    return cleaned.strip()


def _cleaning_none(text: str) -> str:
    """No-op profile: Returns text unchanged.

    Useful for experiments comparing cleaned vs. raw transcripts.
    """
    return text


# Register standard profiles
register_profile("cleaning_v1", _cleaning_v1)
register_profile("cleaning_v2", _cleaning_v2)
register_profile("cleaning_v3", _cleaning_v3)
register_profile("cleaning_none", _cleaning_none)

# Default profile (matches current production behavior)
DEFAULT_PROFILE = "cleaning_v3"

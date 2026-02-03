"""Registered preprocessing profiles for transcript cleaning.

This module implements versioned preprocessing profiles (ADR-029) that
encapsulate cleaning logic into reproducible, versioned configurations.
Each profile specifies exactly which cleaning steps are active, enabling
researchers to isolate variables when comparing models.

See ADR-029 for design rationale.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

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


def apply_profile_with_stats(text: str, profile_id: str) -> Tuple[str, Dict[str, int]]:
    """Apply a preprocessing profile to text and return statistics about lines removed.

    Args:
        text: Raw transcript text
        profile_id: Profile identifier

    Returns:
        Tuple of (cleaned_text, stats_dict) where stats_dict contains:
        - "initial_lines": Number of lines in original text
        - "final_lines": Number of lines in cleaned text
        - "lines_removed": Total lines removed
        - "step_*": Lines removed by each step (profile-specific)

    Raises:
        ValueError: If profile_id is not registered
    """
    initial_lines = len(text.splitlines())
    stats: Dict[str, int] = {"initial_lines": initial_lines}

    # Apply profile-specific tracking
    if profile_id == "cleaning_v4":
        # Track each step for cleaning_v4
        lines_before = len(text.splitlines())
        cleaned = preprocessing.strip_episode_header(text)
        lines_after = len(cleaned.splitlines())
        stats["step_header_stripped"] = lines_before - lines_after

        lines_before = len(cleaned.splitlines())
        cleaned = preprocessing.strip_credits(cleaned)
        lines_after = len(cleaned.splitlines())
        stats["step_credits_stripped"] = lines_before - lines_after

        lines_before = len(cleaned.splitlines())
        cleaned = preprocessing.strip_garbage_lines(cleaned)
        lines_after = len(cleaned.splitlines())
        stats["step_garbage_stripped"] = lines_before - lines_after

        lines_before = len(cleaned.splitlines())
        lines = cleaned.splitlines()
        cleaned = "\n".join(line for line in lines if not preprocessing.is_junk_line(line))
        lines_after = len(cleaned.splitlines())
        stats["step_junk_filtered"] = lines_before - lines_after

        # Anonymize speakers (doesn't remove lines, just modifies them)
        cleaned = preprocessing.anonymize_speakers(cleaned)

        # Standard cleaning (may collapse blank lines)
        lines_before = len(cleaned.splitlines())
        cleaned = preprocessing.clean_transcript(
            cleaned,
            remove_timestamps=True,
            normalize_speakers=True,
            collapse_blank_lines=True,
            remove_fillers=False,
        )
        lines_after = len(cleaned.splitlines())
        stats["step_standard_cleaning"] = lines_before - lines_after

        lines_before = len(cleaned.splitlines())
        cleaned = str(preprocessing.remove_sponsor_blocks(cleaned))  # type: ignore[attr-defined]
        lines_after = len(cleaned.splitlines())
        stats["step_sponsor_removed"] = lines_before - lines_after

        lines_before = len(cleaned.splitlines())
        cleaned = str(preprocessing.remove_outro_blocks(cleaned))  # type: ignore[attr-defined]
        lines_after = len(cleaned.splitlines())
        stats["step_outro_removed"] = lines_before - lines_after

        lines_before = len(cleaned.splitlines())
        cleaned = preprocessing.remove_summarization_artifacts(cleaned)
        cleaned = str(cleaned)  # type: ignore[attr-defined]
        lines_after = len(cleaned.splitlines())
        stats["step_artifacts_removed"] = lines_before - lines_after

        # Artifact scrub (doesn't remove lines, just cleans tokens)
        cleaned = preprocessing.artifact_scrub_v1(cleaned)

        cleaned = cleaned.strip()
    elif profile_id == "cleaning_v3":
        # Track steps for cleaning_v3
        lines_before = len(text.splitlines())
        cleaned = preprocessing.strip_credits(text)
        lines_after = len(cleaned.splitlines())
        stats["step_credits_stripped"] = lines_before - lines_after

        lines_before = len(cleaned.splitlines())
        cleaned = preprocessing.strip_garbage_lines(cleaned)
        lines_after = len(cleaned.splitlines())
        stats["step_garbage_stripped"] = lines_before - lines_after

        lines_before = len(cleaned.splitlines())
        cleaned = preprocessing.clean_transcript(
            cleaned,
            remove_timestamps=True,
            normalize_speakers=True,
            collapse_blank_lines=True,
            remove_fillers=False,
        )
        lines_after = len(cleaned.splitlines())
        stats["step_standard_cleaning"] = lines_before - lines_after

        lines_before = len(cleaned.splitlines())
        cleaned = str(preprocessing.remove_sponsor_blocks(cleaned))  # type: ignore[attr-defined]
        lines_after = len(cleaned.splitlines())
        stats["step_sponsor_removed"] = lines_before - lines_after

        lines_before = len(cleaned.splitlines())
        cleaned = str(preprocessing.remove_outro_blocks(cleaned))  # type: ignore[attr-defined]
        lines_after = len(cleaned.splitlines())
        stats["step_outro_removed"] = lines_before - lines_after

        lines_before = len(cleaned.splitlines())
        # type: ignore[attr-defined]
        cleaned = str(preprocessing.remove_summarization_artifacts(cleaned))
        lines_after = len(cleaned.splitlines())
        stats["step_artifacts_removed"] = lines_before - lines_after

        cleaned = cleaned.strip()
    elif profile_id == "cleaning_v2":
        # Track steps for cleaning_v2
        cleaned = _cleaning_v1(text)
        lines_before = len(cleaned.splitlines())
        cleaned = str(preprocessing.remove_sponsor_blocks(cleaned))  # type: ignore[attr-defined]
        lines_after = len(cleaned.splitlines())
        stats["step_sponsor_removed"] = lines_before - lines_after

        lines_before = len(cleaned.splitlines())
        cleaned = str(preprocessing.remove_outro_blocks(cleaned))  # type: ignore[attr-defined]
        lines_after = len(cleaned.splitlines())
        stats["step_outro_removed"] = lines_before - lines_after
    elif profile_id == "cleaning_v1":
        # cleaning_v1 doesn't remove lines, just modifies them
        cleaned = _cleaning_v1(text)
    elif profile_id == "cleaning_none":
        cleaned = text
    else:
        # Fallback: use regular apply_profile
        cleaned = apply_profile(text, profile_id)

    final_lines = len(cleaned.splitlines())
    stats["final_lines"] = final_lines
    stats["lines_removed"] = initial_lines - final_lines

    return cleaned, stats


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


def _cleaning_v4(text: str) -> str:
    """Cleaning profile v4: Enhanced cleaning with speaker anonymization and artifact scrubbing.

    This profile adds:
    - Speaker anonymization (Maya: → A:)
    - Episode header stripping (title, Host:, Guest:)
    - Artifact scrub: Removes transcript artifacts and render tokens
      (bullet corruption: =-/ -=, divider noise: ///, render tokens: TextColor,
      prefix artifacts: Subur-, Exting, etc.)
    - All cleaning_v3 steps

    Expected impact:
    - Speaker name leak rate: 80% → <10%
    - Header artifacts: Removed
    - Junk lines: Removed
    - Transcript artifacts: Removed (prevents models from copying garbage tokens)
    """
    # 1. Strip episode header (title, Host:, Guest:)
    cleaned = preprocessing.strip_episode_header(text)

    # 2. Strip credits (before chunking)
    cleaned = preprocessing.strip_credits(cleaned)

    # 3. Strip garbage/junk lines (using both strip_garbage_lines and is_junk_line)
    cleaned = preprocessing.strip_garbage_lines(cleaned)
    # Also filter lines using is_junk_line for additional cleanup
    lines = cleaned.splitlines()
    cleaned = "\n".join(line for line in lines if not preprocessing.is_junk_line(line))

    # 4. Anonymize speakers (Maya: → A:)
    cleaned = preprocessing.anonymize_speakers(cleaned)

    # 5. Standard cleaning (timestamps, normalize generic speakers)
    cleaned = preprocessing.clean_transcript(
        cleaned,
        remove_timestamps=True,
        normalize_speakers=True,
        collapse_blank_lines=True,
        remove_fillers=False,
    )

    # 6. Remove sponsor/outro blocks
    cleaned = str(preprocessing.remove_sponsor_blocks(cleaned))  # type: ignore[attr-defined]
    cleaned = str(preprocessing.remove_outro_blocks(cleaned))  # type: ignore[attr-defined]

    # 7. Remove BART/LED artifacts
    cleaned = preprocessing.remove_summarization_artifacts(cleaned)
    cleaned = str(cleaned)  # type: ignore[attr-defined]

    # 8. Artifact scrub: Remove transcript artifacts and render tokens
    # This removes bullet corruption, divider noise, render tokens, and prefix artifacts
    # that leak into summaries (e.g., "=-", "///", "TextColor", "Subur-", "Exting")
    cleaned = preprocessing.artifact_scrub_v1(cleaned)

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
register_profile("cleaning_v4", _cleaning_v4)
register_profile("cleaning_none", _cleaning_none)

# Default profile (matches current production behavior)
# Note: Production baseline (baseline_ml_prod_authority_v1) uses cleaning_v4
# Default changed to cleaning_v4 to match production baseline
DEFAULT_PROFILE = "cleaning_v4"

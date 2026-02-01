"""Metadata validation utilities for episode metadata files.

This module provides validation functions to ensure metadata files conform
to the expected schema and structure.
"""

from __future__ import annotations

import logging
from typing import Any, cast, Dict

logger = logging.getLogger(__name__)


def validate_episode_metadata(metadata: Dict[str, Any], episode_id: str) -> None:
    """Validate episode metadata structure with assertions.

    Raises AssertionError if validation fails.

    Args:
        metadata: Metadata dictionary to validate
        episode_id: Episode identifier for error messages

    Raises:
        AssertionError: If validation fails
    """
    # Assert: episode_id or source_episode_id must be present
    has_episode_id = "episode_id" in metadata or "source_episode_id" in metadata
    assert (
        has_episode_id
    ), f"Metadata for {episode_id}: Missing 'episode_id' or 'source_episode_id' field"

    # Assert: metadata_version must be present
    assert (
        "metadata_version" in metadata
    ), f"Metadata for {episode_id}: Missing 'metadata_version' field"

    # Assert: speakers must be a list (if present)
    if "speakers" in metadata:
        speakers = metadata["speakers"]
        assert isinstance(
            speakers, list
        ), f"Metadata for {episode_id}: 'speakers' must be a list, got {type(speakers).__name__}"

    logger.debug(f"Metadata validation passed for {episode_id}")


def validate_and_load_metadata(metadata_path: str, episode_id: str) -> Dict[str, Any] | None:
    """Load and validate metadata from a file.

    Args:
        metadata_path: Path to metadata JSON file
        episode_id: Episode identifier for validation

    Returns:
        Validated metadata dictionary, or None if file doesn't exist or validation fails

    Raises:
        AssertionError: If validation fails (when file exists and is valid JSON)
    """
    import json
    from pathlib import Path

    meta_file = Path(metadata_path)
    if not meta_file.exists():
        return None

    try:
        metadata = json.loads(meta_file.read_text(encoding="utf-8"))
        validate_episode_metadata(metadata, episode_id)
        return cast(Dict[str, Any], metadata)
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in {metadata_path}: {e}")
        return None
    except AssertionError:
        # Re-raise assertion errors (validation failures)
        raise
    except Exception as e:
        logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
        return None

"""Path validation and sanitization utilities for security (Issue #379).

This module provides functions to validate and sanitize file paths to prevent
path traversal attacks and ensure cache paths are within designated directories.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def validate_cache_path(path: str | Path, base_dir: Optional[str | Path] = None) -> Path:
    """Validate and sanitize cache path to prevent path traversal.

    Args:
        path: Cache path to validate
        base_dir: Base directory that cache must be within (optional)

    Returns:
        Resolved, validated Path object

    Raises:
        ValueError: If path is invalid or outside base_dir
    """
    path_obj = Path(path).resolve()

    # Check for path traversal attempts
    if ".." in str(path):
        raise ValueError(
            f"Invalid cache path: '{path}' contains '..' (path traversal attempt). "
            "Cache paths must be absolute or relative to base directory."
        )

    # If base_dir is provided, ensure path is within it
    if base_dir:
        base_path = Path(base_dir).resolve()
        try:
            # Check if path is within base_dir
            path_obj.relative_to(base_path)
        except ValueError:
            raise ValueError(
                f"Cache path '{path}' is outside base directory '{base_dir}'. "
                "This prevents path traversal attacks."
            )

    return path_obj


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name to prevent path injection.

    Args:
        model_name: Model identifier (e.g., "facebook/bart-base")

    Returns:
        Sanitized model name

    Raises:
        ValueError: If model name contains invalid characters
    """
    # Model names should only contain alphanumeric, dash, underscore, and slash
    import re

    if not re.match(r"^[a-zA-Z0-9_\-/]+$", model_name):
        raise ValueError(
            f"Invalid model name: '{model_name}'. "
            "Model names must only contain alphanumeric characters, "
            "dashes, underscores, and slashes."
        )

    # Prevent path traversal in model names
    if ".." in model_name or model_name.startswith("/"):
        raise ValueError(
            f"Invalid model name: '{model_name}'. "
            "Model names cannot contain '..' or start with '/'."
        )

    return model_name


def validate_path_is_safe(
    path: str,
    trusted_roots: list[Path],
    allow_absolute: bool = False,
) -> bool:
    """Validate that a path is within trusted root directories.

    Args:
        path: Path to validate
        trusted_roots: List of trusted root directories
        allow_absolute: If True, allow absolute paths outside trusted roots
                        (with warning). If False, only allow paths within
                        trusted roots.

    Returns:
        True if path is safe, False otherwise
    """
    try:
        path_obj = Path(path).resolve()
    except (OSError, RuntimeError):
        return False

    # Check if path is within any trusted root
    for root in trusted_roots:
        try:
            path_obj.relative_to(root)
            return True
        except ValueError:
            # Path is not relative to this root, try next
            continue

    # If allow_absolute is True, allow absolute paths (but they're not "safe")
    if allow_absolute and path_obj.is_absolute():
        return True

    return False


def safe_resolve_directory(path: Path) -> Optional[Path]:
    """Resolve a directory path for local corpus use; reject traversal and null bytes.

    Used as a CodeQL anchor before joining or globbing under a user-provided root.
    """
    try:
        candidate = path.expanduser()
    except (OSError, RuntimeError):
        return None
    if "\x00" in str(candidate):
        return None
    if any(part == ".." for part in candidate.parts):
        return None
    try:
        return candidate.resolve()
    except (OSError, RuntimeError):
        return None


def is_resolved_path_under_root(candidate: Path, root: Path) -> bool:
    """Return True if ``candidate.resolve()`` lies under ``root.resolve()`` (or equals it)."""
    try:
        resolved_candidate = candidate.resolve()
        resolved_root = root.resolve()
    except (OSError, RuntimeError):
        return False
    try:
        resolved_candidate.relative_to(resolved_root)
        return True
    except ValueError:
        return False

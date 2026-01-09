"""ML model cache management utilities.

This module provides functions to inspect and manage ML model caches for
Whisper, Transformers (Hugging Face), and spaCy models.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cache_utils import (
    get_spacy_cache_dir,
    get_transformers_cache_dir,
    get_whisper_cache_dir,
)

logger = logging.getLogger(__name__)


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    size: float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def calculate_directory_size(directory: Path) -> int:
    """Calculate total size of a directory in bytes.

    Args:
        directory: Directory path

    Returns:
        Total size in bytes
    """
    if not directory.exists():
        return 0

    total_size = 0
    try:
        for item in directory.rglob("*"):
            if item.is_file():
                try:
                    total_size += item.stat().st_size
                except (OSError, PermissionError):
                    # Skip files we can't access
                    pass
    except (OSError, PermissionError):
        # Partial results are acceptable
        logger.debug(f"Could not fully calculate directory size: {directory}")

    return total_size


def get_whisper_cache_info() -> Tuple[Path, int, List[Dict[str, Any]]]:
    """Get Whisper cache information.

    Returns:
        Tuple of (cache_dir, total_size_bytes, list of model info dicts)
    """
    cache_dir = get_whisper_cache_dir()
    total_size = 0
    models = []

    if cache_dir.exists():
        # Find all .pt files (Whisper model files)
        for model_file in cache_dir.rglob("*.pt"):
            try:
                size = model_file.stat().st_size
                total_size += size
                models.append(
                    {
                        "name": model_file.name,
                        "size": size,
                        "path": model_file,
                    }
                )
            except (OSError, PermissionError):
                pass

    return cache_dir, total_size, sorted(models, key=lambda x: x["name"])


def get_transformers_cache_info() -> Tuple[Path, int, List[Dict[str, Any]]]:
    """Get Transformers (Hugging Face) cache information.

    Returns:
        Tuple of (cache_dir, total_size_bytes, list of model info dicts)
    """
    cache_dir = get_transformers_cache_dir()
    total_size = 0
    models = []

    if cache_dir.exists():
        # Find all models--* directories
        for model_dir in cache_dir.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith("models--"):
                try:
                    size = calculate_directory_size(model_dir)
                    total_size += size
                    # Convert "models--facebook--bart-base" to "facebook/bart-base"
                    model_name = model_dir.name.replace("models--", "").replace("--", "/")
                    models.append(
                        {
                            "name": model_name,
                            "size": size,
                            "path": model_dir,
                        }
                    )
                except (OSError, PermissionError):
                    pass

    return cache_dir, total_size, sorted(models, key=lambda x: x["name"])


def get_spacy_cache_info() -> Tuple[Optional[Path], int, List[Dict[str, Any]]]:
    """Get spaCy cache information.

    Note: spaCy models are typically installed as Python packages, so this
    may return limited information.

    Returns:
        Tuple of (cache_dir or None, total_size_bytes, list of model info dicts)
    """
    cache_dir = get_spacy_cache_dir()
    total_size = 0
    models = []

    if cache_dir and cache_dir.exists():
        # spaCy models are typically in subdirectories
        for item in cache_dir.iterdir():
            if item.is_dir():
                try:
                    size = calculate_directory_size(item)
                    if size > 0:
                        total_size += size
                        models.append(
                            {
                                "name": item.name,
                                "size": size,
                                "path": item,
                            }
                        )
                except (OSError, PermissionError):
                    pass

    return cache_dir, total_size, sorted(models, key=lambda x: x["name"])


def get_all_cache_info() -> Dict[str, Any]:
    """Get information about all ML model caches.

    Returns:
        Dictionary with cache information for each type
    """
    whisper_dir, whisper_size, whisper_models = get_whisper_cache_info()
    transformers_dir, transformers_size, transformers_models = get_transformers_cache_info()
    spacy_dir, spacy_size, spacy_models = get_spacy_cache_info()

    total_size = whisper_size + transformers_size + spacy_size

    return {
        "whisper": {
            "dir": whisper_dir,
            "size": whisper_size,
            "models": whisper_models,
            "count": len(whisper_models),
        },
        "transformers": {
            "dir": transformers_dir,
            "size": transformers_size,
            "models": transformers_models,
            "count": len(transformers_models),
        },
        "spacy": {
            "dir": spacy_dir,
            "size": spacy_size,
            "models": spacy_models,
            "count": len(spacy_models),
        },
        "total_size": total_size,
    }


def clean_whisper_cache(confirm: bool = True) -> Tuple[int, int]:
    """Clean Whisper model cache.

    Args:
        confirm: If True, require confirmation before deleting

    Returns:
        Tuple of (deleted_count, freed_bytes)
    """
    cache_dir, total_size, models = get_whisper_cache_info()

    if not cache_dir.exists() or not models:
        return 0, 0

    if confirm:
        print(
            f"Warning: This will delete {len(models)} Whisper model(s) ({format_size(total_size)})"
        )
        print(f"Cache directory: {cache_dir}")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() not in ("yes", "y"):
            print("Cancelled.")
            return 0, 0

    deleted_count = 0
    freed_bytes = 0

    for model in models:
        try:
            model_path = model["path"]
            if model_path.is_file():
                size = model_path.stat().st_size
                model_path.unlink()
                deleted_count += 1
                freed_bytes += size
            elif model_path.is_dir():
                size = calculate_directory_size(model_path)
                shutil.rmtree(model_path)
                deleted_count += 1
                freed_bytes += size
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to delete {model['name']}: {e}")

    return deleted_count, freed_bytes


def clean_transformers_cache(confirm: bool = True) -> Tuple[int, int]:
    """Clean Transformers (Hugging Face) model cache.

    Args:
        confirm: If True, require confirmation before deleting

    Returns:
        Tuple of (deleted_count, freed_bytes)
    """
    cache_dir, total_size, models = get_transformers_cache_info()

    if not cache_dir.exists() or not models:
        return 0, 0

    if confirm:
        print(
            f"Warning: This will delete {len(models)} Transformers model(s) "
            f"({format_size(total_size)})"
        )
        print(f"Cache directory: {cache_dir}")
        print("Note: This cache may be shared with other applications using Hugging Face models.")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() not in ("yes", "y"):
            print("Cancelled.")
            return 0, 0

    deleted_count = 0
    freed_bytes = 0

    for model in models:
        try:
            model_path = model["path"]
            size = calculate_directory_size(model_path)
            shutil.rmtree(model_path)
            deleted_count += 1
            freed_bytes += size
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to delete {model['name']}: {e}")

    return deleted_count, freed_bytes


def clean_spacy_cache(confirm: bool = True) -> Tuple[int, int]:
    """Clean spaCy model cache.

    Args:
        confirm: If True, require confirmation before deleting

    Returns:
        Tuple of (deleted_count, freed_bytes)
    """
    cache_dir, total_size, models = get_spacy_cache_info()

    if not cache_dir or not cache_dir.exists() or not models:
        return 0, 0

    if confirm:
        print(f"Warning: This will delete {len(models)} spaCy model(s) ({format_size(total_size)})")
        print(f"Cache directory: {cache_dir}")
        print("Note: spaCy models are typically installed as Python packages.")
        print("This will only remove cached data, not the installed packages.")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() not in ("yes", "y"):
            print("Cancelled.")
            return 0, 0

    deleted_count = 0
    freed_bytes = 0

    for model in models:
        try:
            model_path = model["path"]
            size = calculate_directory_size(model_path)
            shutil.rmtree(model_path)
            deleted_count += 1
            freed_bytes += size
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to delete {model['name']}: {e}")

    return deleted_count, freed_bytes


def clean_all_caches(confirm: bool = True) -> Dict[str, Tuple[int, int]]:
    """Clean all ML model caches.

    Args:
        confirm: If True, require confirmation before deleting

    Returns:
        Dictionary mapping cache type to (deleted_count, freed_bytes) tuple
    """
    cache_info = get_all_cache_info()
    total_size = cache_info["total_size"]

    if total_size == 0:
        return {"whisper": (0, 0), "transformers": (0, 0), "spacy": (0, 0)}

    if confirm:
        print("Warning: This will delete all cached ML models:")
        whisper_count = cache_info["whisper"]["count"]
        whisper_size = format_size(cache_info["whisper"]["size"])
        print(f"  Whisper: {whisper_count} model(s) ({whisper_size})")
        transformers_count = cache_info["transformers"]["count"]
        transformers_size = format_size(cache_info["transformers"]["size"])
        print(f"  Transformers: {transformers_count} model(s) ({transformers_size})")
        spacy_count = cache_info["spacy"]["count"]
        spacy_size = format_size(cache_info["spacy"]["size"])
        print(f"  spaCy: {spacy_count} model(s) ({spacy_size})")
        print(f"Total: {format_size(total_size)}")
        print("\nNote: Transformers cache may be shared with other applications.")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() not in ("yes", "y"):
            print("Cancelled.")
            return {"whisper": (0, 0), "transformers": (0, 0), "spacy": (0, 0)}

    results = {}
    results["whisper"] = clean_whisper_cache(confirm=False)
    results["transformers"] = clean_transformers_cache(confirm=False)
    results["spacy"] = clean_spacy_cache(confirm=False)

    return results

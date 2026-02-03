"""Cache management for ML models.

This package provides:
- Cache directory utilities (Whisper, transformers)
- Cache inspection and management
"""

from .directories import (
    get_project_root,
    get_spacy_cache_dir,
    get_transformers_cache_dir,
    get_whisper_cache_dir,
)
from .manager import (
    calculate_directory_size,
    clean_all_caches,
    clean_spacy_cache,
    clean_transformers_cache,
    clean_whisper_cache,
    format_size,
    get_all_cache_info,
    get_spacy_cache_info,
    get_transformers_cache_info,
    get_whisper_cache_info,
)

__all__ = [
    # Directories
    "get_project_root",
    "get_spacy_cache_dir",
    "get_transformers_cache_dir",
    "get_whisper_cache_dir",
    # Manager
    "calculate_directory_size",
    "clean_all_caches",
    "clean_spacy_cache",
    "clean_transformers_cache",
    "clean_whisper_cache",
    "format_size",
    "get_all_cache_info",
    "get_spacy_cache_info",
    "get_transformers_cache_info",
    "get_whisper_cache_info",
]

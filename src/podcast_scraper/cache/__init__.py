"""Cache modules for podcast_scraper."""

from . import directories, manager, transcript_cache

# Re-export cache directory functions for backward compatibility
from .directories import (
    get_project_root,
    get_spacy_cache_dir,
    get_transformers_cache_dir,
    get_whisper_cache_dir,
)

__all__ = [
    "directories",
    "manager",
    "transcript_cache",
    "get_project_root",
    "get_whisper_cache_dir",
    "get_transformers_cache_dir",
    "get_spacy_cache_dir",
]

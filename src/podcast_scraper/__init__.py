# This project is intended for personal, non-commercial use only.
# See README and docs/legal.md for details.

"""Podcast Scraper - Download podcast transcripts from RSS feeds.

This package provides a simple API for downloading podcast transcripts:
- From published transcript URLs (Podcasting 2.0 namespace)
- Via Whisper transcription fallback for episodes without transcripts
- With multi-threaded downloads and resumable runs

Programmatic API Example:
    >>> import podcast_scraper
    >>>
    >>> config = podcast_scraper.Config(
    ...     rss_url="https://example.com/feed.xml",
    ...     output_dir="./transcripts",
    ...     max_episodes=10,
    ... )
    >>> count, summary = podcast_scraper.run_pipeline(config)
    >>> print(f"Downloaded {count} transcripts")

Service API Example (for daemon/service use):
    >>> from podcast_scraper import service
    >>> result = service.run_from_config_file("config.yaml")
    >>> if result.success:
    ...     print(f"Processed {result.episodes_processed} episodes")
    ... else:
    ...     print(f"Error: {result.error}")

CLI Usage:
    $ python -m podcast_scraper.cli https://example.com/feed.xml
    $ python -m podcast_scraper.cli --config config.yaml

Service Mode (for supervisor/systemd):
    $ python -m podcast_scraper.service --config config.yaml
"""

from __future__ import annotations

from .config import Config, load_config_file
from .workflow import run_pipeline

__all__ = [
    "Config",
    "load_config_file",
    "run_pipeline",
    "__version__",
    "__api_version__",
    "cache_manager",
]
# Note: 'cli' and 'service' are available via __getattr__ for lazy loading
# Use: from podcast_scraper import cli, service
# Or: import podcast_scraper.cli as cli
__version__ = "2.4.0"

# API version follows semantic versioning and is tied to module version
# - Major version (X.y.z): Breaking API changes
# - Minor version (x.Y.z): New features, backward compatible
# - Patch version (x.y.Z): Bug fixes, backward compatible
__api_version__ = __version__

# Cache for lazy-loaded modules to prevent circular imports
_import_cache: dict[str, object] = {}


def __getattr__(name: str):
    if name in _import_cache:
        return _import_cache[name]

    if name == "cli":
        import importlib

        _cli = importlib.import_module(f"{__name__}.cli")
        _import_cache[name] = _cli
        return _cli
    if name == "service":
        import importlib

        _service = importlib.import_module(f"{__name__}.service")
        _import_cache[name] = _service
        return _service
    if name == "cache_manager":
        import importlib

        _cache_manager = importlib.import_module(f"{__name__}.cache_manager")
        _import_cache[name] = _cache_manager
        return _cache_manager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

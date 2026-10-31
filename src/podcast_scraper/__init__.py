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
from .workflow.orchestration import run_pipeline

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
# Build/package version (PEP 440). A pre-release (``2.7.0.dev0``) sorts BEFORE the
# final ``2.7.0`` — so builds on the way to 2.7.0 are not mislabelled as the release.
# Feeds the corpus ``code_version`` stamp, the CLI ``--version``, the Sentry release,
# and the FastAPI ``version``. Per-build identity is the immutable ``sha-<7>`` image
# tag; bump this marker at milestones (dev0 → b1 → rc1 → 2.7.0).
__version__ = "2.7.0.dev0"

# API CONTRACT version — a clean semantic ``X.Y.Z``, DECOUPLED from the build version
# so a pre-release build doesn't move the contract. It is the release BASE of
# ``__version__`` (test-enforced: ``Version(__version__).base_version == __api_version__``).
# - Major (X.y.z): breaking API changes · Minor (x.Y.z): new features · Patch: fixes.
__api_version__ = "2.7.0"

# Cache for lazy-loaded modules to prevent circular imports
_import_cache: dict[str, object] = {}


def __getattr__(name: str):
    """Lazy-load submodules to avoid circular imports and heavy imports at package load."""
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

        _cache_manager = importlib.import_module(f"{__name__}.cache.manager")
        _import_cache[name] = _cache_manager
        return _cache_manager
    if name == "summarizer":
        import importlib

        _summarizer = importlib.import_module(f"{__name__}.providers.ml.summarizer")
        _import_cache[name] = _summarizer
        return _summarizer
    if name == "downloader":
        import importlib

        _downloader = importlib.import_module(f"{__name__}.rss.downloader")
        _import_cache[name] = _downloader
        return _downloader
    if name == "rss_parser":
        import importlib

        _rss_parser = importlib.import_module(f"{__name__}.rss.parser")
        _import_cache[name] = _rss_parser
        return _rss_parser
    if name == "speaker_detection":
        import importlib

        _speaker_detection = importlib.import_module(f"{__name__}.providers.ml.speaker_detection")
        _import_cache[name] = _speaker_detection
        return _speaker_detection
    if name == "metrics":
        import importlib

        _metrics = importlib.import_module(f"{__name__}.workflow.metrics")
        _import_cache[name] = _metrics
        return _metrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# This project is intended for personal, non-commercial use only.
# See README and docs/legal.md for details.

"""Podcast Scraper - Download podcast transcripts from RSS feeds.

This package provides a simple API for downloading podcast transcripts:
- From published transcript URLs (Podcasting 2.0 namespace)
- Via Whisper transcription fallback for episodes without transcripts
- With multi-threaded downloads and resumable runs

Example:
    >>> import podcast_scraper
    >>>
    >>> config = podcast_scraper.Config(
    ...     rss_url="https://example.com/feed.xml",
    ...     output_dir="./transcripts",
    ...     max_episodes=10,
    ... )
    >>> count, summary = podcast_scraper.run_pipeline(config)
    >>> print(f"Downloaded {count} transcripts")

CLI Usage:
    $ python -m podcast_scraper.cli https://example.com/feed.xml
    $ python -m podcast_scraper.cli --config config.yaml
"""

from __future__ import annotations

from .config import Config, load_config_file
from .workflow import run_pipeline

__all__ = ["Config", "load_config_file", "run_pipeline", "cli"]
__version__ = "2.2.0"


def __getattr__(name: str):
    if name == "cli":
        from . import cli as _cli  # type: ignore

        return _cli
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Optional on-disk cache for RSS feed XML (reduces repeated HTTP fetches).

Enabled when :envvar:`PODCAST_SCRAPER_RSS_CACHE_DIR` is set to a writable directory.
When unset, no caching occurs (default for normal CLI use).

The acceptance test runner sets this per session so multiple sequential configs that
share the same ``rss_url`` reuse one downloaded feed XML.

Does not cache episode media downloads; see ``reuse_media`` in config for that.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from podcast_scraper.utils.log_redaction import format_exception_for_log

logger = logging.getLogger(__name__)

ENV_RSS_CACHE_DIR = "PODCAST_SCRAPER_RSS_CACHE_DIR"


def cache_dir_from_env() -> Optional[Path]:
    """Return resolved cache directory if ``PODCAST_SCRAPER_RSS_CACHE_DIR`` is set and non-empty."""
    raw = os.environ.get(ENV_RSS_CACHE_DIR)
    if not raw or not str(raw).strip():
        return None
    return Path(raw).expanduser().resolve()


def cache_path_for_url(cache_dir: Path, rss_url: str) -> Path:
    """Stable path for a normalized RSS URL (SHA-256 prefix)."""
    from .downloader import normalize_url

    key = hashlib.sha256(normalize_url(rss_url).encode("utf-8")).hexdigest()[:24]
    return cache_dir / f"rss_{key}.xml"


def read_cached_rss(rss_url: str) -> Optional[bytes]:
    """Return cached RSS XML bytes if present, else None."""
    base = cache_dir_from_env()
    if base is None:
        return None
    path = cache_path_for_url(base, rss_url)
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError as exc:
        logger.warning(
            "Could not read RSS cache %s: %s",
            path,
            format_exception_for_log(exc),
        )
        return None
    if not data:
        return None
    logger.info(
        "Using RSS feed from cache (%s): %s",
        ENV_RSS_CACHE_DIR,
        path.name,
    )
    return data


def write_cached_rss(rss_url: str, rss_bytes: bytes) -> None:
    """Write RSS XML to cache directory if env is set."""
    base = cache_dir_from_env()
    if base is None:
        return
    try:
        base.mkdir(parents=True, exist_ok=True)
        path = cache_path_for_url(base, rss_url)
        path.write_bytes(rss_bytes)
        logger.debug("Wrote RSS feed cache: %s", path)
    except OSError as exc:
        logger.warning(
            "Could not write RSS cache: %s",
            format_exception_for_log(exc),
        )

"""RSS feed fetching, parsing, and episode metadata extraction.

This module provides:
- RSS feed fetching via HTTP
- RSS/XML parsing and episode extraction
- Episode metadata extraction
"""

from .downloader import (
    BYTES_PER_MB,
    DOWNLOAD_CHUNK_SIZE,
    fetch_url,
    http_download_to_file,
    http_get,
    http_head,
    normalize_url,
    OPENAI_MAX_FILE_SIZE_BYTES,
    should_log_download_summary,
)
from .parser import (
    choose_transcript_url,
    create_episode_from_item,
    extract_episode_description,
    extract_episode_metadata,
    extract_episode_published_date,
    extract_feed_metadata,
    fetch_and_parse_rss,
    find_enclosure_media,
    find_transcript_urls,
    parse_rss_items,
)

__all__ = [
    # Downloader functions
    "BYTES_PER_MB",
    "DOWNLOAD_CHUNK_SIZE",
    "OPENAI_MAX_FILE_SIZE_BYTES",
    "fetch_url",
    "http_download_to_file",
    "http_get",
    "http_head",
    "normalize_url",
    "should_log_download_summary",
    # Parser functions
    "choose_transcript_url",
    "create_episode_from_item",
    "extract_episode_description",
    "extract_episode_metadata",
    "extract_episode_published_date",
    "extract_feed_metadata",
    "fetch_and_parse_rss",
    "find_enclosure_media",
    "find_transcript_urls",
    "parse_rss_items",
]

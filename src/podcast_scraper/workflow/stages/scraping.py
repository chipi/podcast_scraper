"""Scraping stage for RSS feed fetching and parsing.

This module handles RSS feed fetching, parsing, and episode preparation.
"""

from __future__ import annotations

import logging
from typing import List

from ... import config, models
from ...rss import (
    create_episode_from_item,
    extract_feed_metadata,
)
from ..types import FeedMetadata

logger = logging.getLogger(__name__)


def fetch_and_parse_feed(cfg: config.Config) -> tuple[models.RssFeed, bytes]:
    """Fetch and parse RSS feed.

    Fetches RSS feed once and returns both the parsed feed and raw XML bytes
    to avoid duplicate network requests.

    Args:
        cfg: Configuration object

    Returns:
        Tuple of (Parsed RssFeed object, RSS XML bytes)
    """
    from ...rss import downloader, parse_rss_items

    if cfg.rss_url is None:
        raise ValueError("RSS URL is required")

    # Fetch RSS feed once
    resp = downloader.fetch_url(cfg.rss_url, cfg.user_agent, cfg.timeout, stream=False)
    if resp is None:
        raise ValueError("Failed to fetch RSS feed.")

    try:
        rss_bytes = resp.content
        feed_base_url = resp.url or cfg.rss_url
    finally:
        resp.close()

    # Parse RSS feed
    try:
        feed_title, feed_authors, items = parse_rss_items(rss_bytes)
    except Exception as exc:
        raise ValueError(f"Failed to parse RSS XML: {exc}") from exc

    feed = models.RssFeed(
        title=feed_title, authors=feed_authors, items=items, base_url=feed_base_url
    )
    logger.debug("Fetched RSS feed title=%s (%s items)", feed.title, len(feed.items))

    return feed, rss_bytes


def extract_feed_metadata_for_generation(
    cfg: config.Config, feed: models.RssFeed, rss_bytes: bytes
) -> FeedMetadata:
    """Extract feed metadata for metadata generation.

    Args:
        cfg: Configuration object
        feed: Parsed RssFeed object
        rss_bytes: Raw RSS XML bytes (reused from initial fetch to avoid duplicate request)

    Returns:
        FeedMetadata tuple
    """
    if not cfg.generate_metadata or not rss_bytes:
        return FeedMetadata(None, None, None)

    try:
        feed_description, feed_image_url, feed_last_updated = extract_feed_metadata(
            rss_bytes, feed.base_url
        )
        return FeedMetadata(feed_description, feed_image_url, feed_last_updated)
    except Exception as exc:
        logger.debug("Failed to extract feed metadata: %s", exc)
        return FeedMetadata(None, None, None)


def prepare_episodes_from_feed(feed: models.RssFeed, cfg: config.Config) -> List[models.Episode]:
    """Create Episode objects from RSS items.

    Args:
        feed: Parsed RssFeed object
        cfg: Configuration object

    Returns:
        List of Episode objects
    """
    items = feed.items
    total_items = len(items)
    if cfg.max_episodes is not None:
        items = items[: cfg.max_episodes]

    logger.info(f"Episodes to process: {len(items)} of {total_items}")

    episodes = [
        create_episode_from_item(item, idx, feed.base_url)
        for idx, item in enumerate(items, start=1)
    ]
    logger.debug("Materialized %s episode objects", len(episodes))
    return episodes

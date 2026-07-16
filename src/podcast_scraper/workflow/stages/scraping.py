"""Scraping stage for RSS feed fetching and parsing.

This module handles RSS feed fetching, parsing, and episode preparation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Set, TYPE_CHECKING

from ... import config, models

if TYPE_CHECKING:
    from ...models import Episode, RssFeed
else:
    Episode = models.Episode  # type: ignore[assignment]
    RssFeed = models.RssFeed  # type: ignore[assignment]
from ...rss import (
    create_episode_from_item,
    extract_feed_metadata,
    extract_item_guid,
    published_date_for_episode_filter,
)
from ..types import FeedMetadata

logger = logging.getLogger(__name__)


def fetch_and_parse_feed(cfg: config.Config) -> tuple[RssFeed, bytes]:  # type: ignore[valid-type]
    """Fetch and parse RSS feed.

    Fetches RSS feed once and returns both the parsed feed and raw XML bytes
    to avoid duplicate network requests.

    Args:
        cfg: Configuration object

    Returns:
        Tuple of (Parsed RssFeed object, RSS XML bytes)
    """
    from ...rss import downloader, feed_cache, parse_rss_items

    if cfg.rss_url is None:
        raise ValueError("RSS URL is required")

    # Optional disk cache (e.g. acceptance session sets PODCAST_SCRAPER_RSS_CACHE_DIR)
    cached_rss = feed_cache.read_cached_rss(cfg.rss_url)
    if cached_rss is not None:
        rss_bytes = cached_rss
        feed_base_url = cfg.rss_url
    else:
        resp = downloader.fetch_rss_feed_url(cfg.rss_url, cfg.user_agent, cfg.timeout, stream=False)
        if resp is None:
            raise ValueError("Failed to fetch RSS feed.")

        try:
            rss_bytes = resp.content
            # httpx.Response.url is an httpx.URL object; downstream ``urljoin``
            # expects a plain str.
            feed_base_url = str(resp.url) if resp.url else cfg.rss_url
        finally:
            resp.close()

    # Parse RSS feed
    try:
        feed_title, feed_authors, items = parse_rss_items(rss_bytes)
    except Exception as exc:
        raise ValueError(f"Failed to parse RSS XML: {exc}") from exc

    if cached_rss is None:
        feed_cache.write_cached_rss(cfg.rss_url, rss_bytes)

    feed = RssFeed(title=feed_title, authors=feed_authors, items=items, base_url=feed_base_url)
    logger.debug("Fetched RSS feed title=%s (%s items)", feed.title, len(feed.items))

    return feed, rss_bytes


def extract_feed_metadata_for_generation(
    cfg: config.Config, feed: RssFeed, rss_bytes: bytes  # type: ignore[valid-type]
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


def collect_existing_guids(output_dir: str) -> Set[str]:
    """Collect the GUIDs of episodes already persisted under ``output_dir``.

    Scans ``output_dir/run_*/metadata/*.metadata.json`` (the per-feed corpus
    layout — ``cfg.output_dir`` is already the feed leaf during selection) and
    reads each ``episode.guid``. Used by the #876 existing-only migration mode to
    restrict the episode set to data already on disk. Dedupes the cross-run
    duplicate dirs into a set; corrupt or guid-less files are skipped, not fatal.
    """
    root = Path(output_dir)
    # Per-feed leaf layout, plus a defensive fallback for a flat single-run dir.
    patterns = ("run_*/metadata/*.metadata.json", "metadata/*.metadata.json")
    guids: Set[str] = set()
    scanned = 0
    for pattern in patterns:
        for meta_path in root.glob(pattern):
            scanned += 1
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.debug("Skipping unreadable metadata %s: %s", meta_path, exc)
                continue
            guid = (data.get("episode") or {}).get("guid")
            if isinstance(guid, str) and guid.strip():
                guids.add(guid.strip())
    logger.info(
        "Existing-only (#876): scanned %s on-disk metadata file(s) under %s; "
        "%s distinct GUID(s)",
        scanned,
        output_dir,
        len(guids),
    )
    return guids


def _filter_items_to_existing_guids(
    items: List[Any], cfg: config.Config  # type: ignore[valid-type]
) -> List[Any]:
    """#876 migration mode: keep only feed items whose GUID is already on disk.

    Hard invariant: the result is a strict subset of the on-disk GUID set, so no
    new episode can ever enter the pipeline. ``max_episodes`` / offset / date caps
    are intentionally not applied in this mode — every matched existing episode is
    processed. Aborts loudly on an empty on-disk set to guard against a mis-pointed
    ``--output-dir`` silently degrading into a full-feed ingest.
    """
    if cfg.output_dir is None:
        raise ValueError("reprocess_existing_only requires output_dir to locate the on-disk corpus")
    existing = collect_existing_guids(cfg.output_dir)
    if not existing:
        raise ValueError(
            "reprocess_existing_only is set but no on-disk episode GUIDs were found "
            f"under {cfg.output_dir}/run_*/metadata/. Wrong --output-dir, or the corpus "
            "is not present. Aborting to avoid ingesting the entire live feed."
        )
    kept: List[Any] = []
    seen_guids: Set[str] = set()
    dropped_new = 0
    dropped_no_guid = 0
    for it in items:
        guid = extract_item_guid(it)
        if not guid:
            dropped_no_guid += 1
            continue
        if guid in existing:
            kept.append(it)
            seen_guids.add(guid)
        else:
            dropped_new += 1
    rolled_off = len(existing - seen_guids)
    logger.info(
        "Existing-only re-diarization (#876): kept %s, dropped %s new feed item(s), "
        "dropped %s item(s) with no GUID, %s on-disk GUID(s) not in the live feed "
        "(rolled off — cannot re-diarize without stored audio)",
        len(kept),
        dropped_new,
        dropped_no_guid,
        rolled_off,
    )
    return kept


def prepare_episodes_from_feed(
    feed: RssFeed, cfg: config.Config  # type: ignore[valid-type]
) -> List[Episode]:  # type: ignore[valid-type]
    """Create Episode objects from RSS items.

    Args:
        feed: Parsed RssFeed object
        cfg: Configuration object

    Returns:
        List of Episode objects
    """
    items = list(feed.items)
    total_items = len(items)

    if cfg.episode_order == "oldest":
        items = list(reversed(items))

    if cfg.reprocess_existing_only:
        # #876 migration mode: episode set is defined entirely by on-disk GUIDs.
        # Bypasses date/offset/max caps so every matched existing episode processes.
        items = _filter_items_to_existing_guids(items, cfg)
    else:
        if cfg.episode_since is not None or cfg.episode_until is not None:
            kept: List[Any] = []
            missing_pub = 0
            for it in items:
                pub_d = published_date_for_episode_filter(it)
                if pub_d is None:
                    missing_pub += 1
                    kept.append(it)
                    continue
                if cfg.episode_since is not None and pub_d < cfg.episode_since:
                    continue
                if cfg.episode_until is not None and pub_d > cfg.episode_until:
                    continue
                kept.append(it)
            if missing_pub:
                logger.warning(
                    "Episode date filter: %s item(s) had no parseable pubDate; "
                    "keeping them in the selection (GitHub #521)",
                    missing_pub,
                )
            items = kept

        if cfg.episode_offset:
            items = items[cfg.episode_offset :]

        if cfg.max_episodes is not None:
            items = items[: cfg.max_episodes]

    logger.info(
        "Episodes to process: %s of %s (after order/date filter/offset/limit)",
        len(items),
        total_items,
    )

    episodes = [
        create_episode_from_item(item, idx, feed.base_url)
        for idx, item in enumerate(items, start=1)
    ]
    logger.debug("Materialized %s episode objects", len(episodes))
    return episodes

"""Generic feed-metadata backfill (BS.1) — refresh the `feed` block for feeds already on disk.

Re-fetches each feed's RSS and rewrites the *derivable* fields of the ``feed`` block in every
episode metadata file — description, image_url, last_updated, category — WITHOUT touching
transcription, GI, KG, or any episode-level data. This is the cheap, targeted alternative to a full
reprocess when only feed-level metadata (a newly-parsed field like category) needs to reach feeds
already ingested.

Generic by design: add a new feed-level field to :func:`_derive_feed_updates` and every existing
feed picks it up on the next backfill. Only non-None derived values are written (a merge, never a
wipe), so a transient/partial fetch cannot blank a good stored value, and re-running an unchanged
feed is a no-op.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from ..rss.parser import extract_feed_category, extract_feed_metadata
from ..server.atomic_write import atomic_write_text

logger = logging.getLogger(__name__)

#: One glob covers the per-feed run layout (``feeds/<id>/run_*/metadata/*.metadata.json``) and a
#: flat ``metadata/*.metadata.json``.
_METADATA_GLOB = "**/metadata/*.metadata.json"

#: ``url -> raw RSS bytes`` (None on failure). Injected so tests need no network.
FetchFn = Callable[[str], Optional[bytes]]


@dataclass
class BackfillResult:
    """What a backfill run touched."""

    feeds_seen: int = 0
    feeds_refreshed: int = 0
    files_updated: int = 0
    skipped: list[str] = field(default_factory=list)  # feed_ids with no url or a failed fetch


def _default_fetch(url: str) -> Optional[bytes]:
    from ..rss.downloader import fetch_rss_feed_url

    resp = fetch_rss_feed_url(url, "podcast-scraper/backfill", 30)
    return resp.content if resp is not None else None


def _derive_feed_updates(rss_bytes: bytes, base_url: str) -> dict[str, Any]:
    """The feed-level fields a fresh RSS parse can supply. Extend here to backfill new fields."""
    description, image_url, last_updated = extract_feed_metadata(rss_bytes, base_url)
    category = extract_feed_category(rss_bytes)
    updates: dict[str, Any] = {}
    if description is not None:
        updates["description"] = description
    if image_url is not None:
        updates["image_url"] = image_url
    if last_updated is not None:
        updates["last_updated"] = last_updated.isoformat()
    if category is not None:
        updates["category"] = category
    return updates


def _value_current(key: str, stored: Any, new: Any) -> bool:
    """Whether a stored feed field already equals the derived one — with datetime normalization for
    ``last_updated`` (advisor M2): the pipeline serializes Zulu and our isoformat gives ``+00:00``,
    so a naive `==` would rewrite (and format-flip) every pipeline-written file forever."""
    if key == "last_updated" and isinstance(stored, str) and isinstance(new, str):
        try:
            return datetime.fromisoformat(stored.replace("Z", "+00:00")) == datetime.fromisoformat(
                new.replace("Z", "+00:00")
            )
        except ValueError:
            return stored == new
    return bool(stored == new)


def _load(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError) as exc:
        logger.debug("Skipping unreadable metadata %s: %s", path, exc)
        return None


def refresh_feed_metadata(
    corpus_dir: Path,
    *,
    feed_id: Optional[str] = None,
    fetch: FetchFn = _default_fetch,
) -> BackfillResult:
    """Re-derive + patch the feed block for every (or one) feed under ``corpus_dir``.

    Groups metadata files by ``feed.feed_id``, fetches each feed's RSS once (from ``feed.url``),
    derives the feed-level updates, and merges them into each file's ``feed`` block. Only the
    derivable fields change; feed_id/title/local artwork and all episode data are preserved.
    """
    result = BackfillResult()
    # Group files by feed, capturing each feed's RSS url from the first file that carries one.
    by_feed: dict[str, list[Path]] = {}
    urls: dict[str, str] = {}
    for path in sorted(corpus_dir.glob(_METADATA_GLOB)):
        doc = _load(path)
        feed = doc.get("feed") if doc else None
        if not isinstance(feed, dict):
            continue
        fid = str(feed.get("feed_id") or "")
        if not fid or (feed_id is not None and fid != feed_id):
            continue
        by_feed.setdefault(fid, []).append(path)
        url = feed.get("url")
        if isinstance(url, str) and url.strip() and fid not in urls:
            urls[fid] = url.strip()

    for fid, paths in by_feed.items():
        result.feeds_seen += 1
        url = urls.get(fid)
        if not url:
            logger.warning("Backfill: feed %s has no RSS url in its metadata; skipping", fid)
            result.skipped.append(fid)
            continue
        rss_bytes = fetch(url)
        if not rss_bytes:
            logger.warning("Backfill: could not fetch RSS for feed %s (%s); skipping", fid, url)
            result.skipped.append(fid)
            continue
        updates = _derive_feed_updates(rss_bytes, url)
        if not updates:
            continue
        refreshed_any = False
        for path in paths:
            doc = _load(path)
            if doc is None or not isinstance(doc.get("feed"), dict):
                continue
            feed_block = doc["feed"]
            if all(_value_current(k, feed_block.get(k), v) for k, v in updates.items()):
                continue  # already current — idempotent no-op
            feed_block.update(updates)
            # Atomic (advisor M1): the backfill rewrites every file of every feed, so a mid-write
            # crash must not truncate one — write a temp then os.replace.
            atomic_write_text(path, json.dumps(doc, ensure_ascii=False, indent=2))
            result.files_updated += 1
            refreshed_any = True
        if refreshed_any:
            result.feeds_refreshed += 1

    logger.info(
        "Backfill complete: %d feed(s) seen, %d refreshed, %d file(s) updated, %d skipped",
        result.feeds_seen,
        result.feeds_refreshed,
        result.files_updated,
        len(result.skipped),
    )
    return result

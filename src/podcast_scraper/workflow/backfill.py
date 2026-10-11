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

Licence records (#2325) are filled where MISSING only, never replaced: a stored record is what
the feed declared when that episode was read, and that history must survive. The channel's goes
into ``feed.licence``; an episode's own (its item-level ``<podcast:license>``,
``<creativeCommons:license>``, show-note notices) into ``episode.licence``, read from the item
whose guid matches -- an episode no longer in the feed keeps no record ("not checked"), rather
than one guessed from the channel.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from podcast_scraper.utils.corpus_walk import corpus_rglob

from ..rss.licence import channel_licence_from_bytes, episode_licence
from ..rss.parser import (
    extract_feed_category,
    extract_feed_metadata,
    extract_item_guid,
    parse_rss_items,
)
from ..server.atomic_write import atomic_write_text

logger = logging.getLogger(__name__)

#: Covers the per-feed run layout (``feeds/<id>/run_*/metadata/*.metadata.<ext>``) and a flat
#: ``metadata/*.metadata.<ext>`` — for both metadata_format kinds (advisor L7).
_METADATA_GLOBS = (
    "metadata/*.metadata.json",
    "metadata/*.metadata.yaml",
    "metadata/*.metadata.yml",
)

#: ``url -> (raw RSS bytes, effective URL after redirects)``, or None on failure. The effective URL
#: is the base for resolving relative image hrefs — matching what ingest sees (advisor L5). Injected
#: so tests need no network.
FetchFn = Callable[[str], Optional["tuple[bytes, str]"]]


@dataclass
class BackfillResult:
    """What a backfill run touched."""

    feeds_seen: int = 0
    feeds_refreshed: int = 0
    files_updated: int = 0
    skipped: list[str] = field(default_factory=list)  # feed_ids with no url or a failed fetch


def _default_fetch(url: str) -> Optional["tuple[bytes, str]"]:
    from ..rss.downloader import fetch_rss_feed_url

    resp = fetch_rss_feed_url(url, "podcast-scraper/backfill", 30)
    # str(resp.url) is the FINAL url after redirects — the correct base for relative hrefs (L5).
    return (resp.content, str(resp.url)) if resp is not None else None


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


def _is_yaml(path: Path) -> bool:
    return path.suffix in (".yaml", ".yml")


def _load(path: Path) -> Optional[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
        if _is_yaml(path):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError) as exc:
        logger.debug("Skipping unreadable metadata %s: %s", path, exc)
        return None


def _dump(path: Path, doc: dict[str, Any]) -> str:
    """Serialize in the file's own format (advisor L7) so a YAML corpus stays YAML."""
    if _is_yaml(path):
        import yaml

        return yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)
    return json.dumps(doc, ensure_ascii=False, indent=2)


def _licence_sources(
    rss_bytes: bytes, base_url: str, source_url: str
) -> tuple[Any, dict[str, Any]]:
    """The channel's licence record and the feed's items by guid (#2325)."""
    channel = channel_licence_from_bytes(rss_bytes, base_url, source_url)
    items: dict[str, Any] = {}
    try:
        _title, _authors, elements = parse_rss_items(rss_bytes)
    except Exception:  # noqa: BLE001 - the feed fields above still apply
        return channel, items
    for item in elements:
        guid = extract_item_guid(item)
        if guid:
            items.setdefault(guid.strip(), item)
    return channel, items


def _fill_licences(
    doc: dict[str, Any], channel: Any, items_by_guid: dict[str, Any], base_url: str
) -> bool:
    """Fill a missing ``feed.licence`` / ``episode.licence``; never replace a stored one."""
    if channel is None:
        return False
    changed = False
    feed_block = doc["feed"]
    if not isinstance(feed_block.get("licence"), dict):
        feed_block["licence"] = channel.model_dump()
        changed = True
    episode = doc.get("episode")
    if isinstance(episode, dict) and not isinstance(episode.get("licence"), dict):
        guid = str(episode.get("guid") or "").strip()
        item = items_by_guid.get(guid) if guid else None
        if item is not None:
            episode["licence"] = episode_licence(
                item,
                base_url,
                channel=channel,
                source_feed_url=channel.source_feed_url,
                retrieved_at=channel.retrieved_at,
            ).model_dump()
            changed = True
    return changed


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
    all_paths = sorted(p for glob in _METADATA_GLOBS for p in corpus_rglob(corpus_dir, glob))
    for path in all_paths:
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
        fetched = fetch(url)
        if not fetched:
            logger.warning("Backfill: could not fetch RSS for feed %s (%s); skipping", fid, url)
            result.skipped.append(fid)
            continue
        rss_bytes, base_url = fetched
        updates = _derive_feed_updates(rss_bytes, base_url)
        channel, items_by_guid = _licence_sources(rss_bytes, base_url, url)
        if not updates and channel is None:
            continue
        refreshed_any = False
        for path in paths:
            doc = _load(path)
            if doc is None or not isinstance(doc.get("feed"), dict):
                continue
            feed_block = doc["feed"]
            licence_changed = _fill_licences(doc, channel, items_by_guid, base_url)
            if not licence_changed and all(
                _value_current(k, feed_block.get(k), v) for k, v in updates.items()
            ):
                continue  # already current — idempotent no-op
            feed_block.update(updates)
            # Atomic (advisor M1): the backfill rewrites every file of every feed, so a mid-write
            # crash must not truncate one — write a temp then os.replace. Serialized in the file's
            # own format (advisor L7).
            atomic_write_text(path, _dump(path, doc))
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

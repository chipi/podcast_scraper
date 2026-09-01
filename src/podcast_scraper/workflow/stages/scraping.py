"""Scraping stage for RSS feed fetching and parsing.

This module handles RSS feed fetching, parsing, and episode preparation.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, TYPE_CHECKING

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

# Module scope is safe here — verified under five import orders (scraping-first, package-first,
# corpus_scope-first, cli entrypoint, indexer-first). An earlier version of this import sat
# inline under a comment claiming a cycle through ``workflow.metadata_generation``; that claim
# was wrong and is recorded here so nobody re-inlines it on the strength of it.
from ...search.corpus_scope import dedupe_metadata_paths_newest_run_per_episode
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

    # Populate the channel-level description. parse_rss_items only returns title/authors/items, so
    # without this feed.description is None on the pipeline path and every description-driven step —
    # host statement/NER (detect_hosts_from_feed) above all — runs blind. The RssFeed.description
    # field is documented and read downstream; it was simply never wired here (smell-audit F6).
    from ...rss.parser import _channel_description

    feed = RssFeed(
        title=feed_title,
        authors=feed_authors,
        items=items,
        base_url=feed_base_url,
        description=_channel_description(rss_bytes),
    )
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


def _drop_already_ingested(items: List[Any], cfg: config.Config) -> List[Any]:
    """Remove feed items whose guid is already on disk (``episode_selection=unprocessed``).

    THE DRIFT THIS FIXES. ``episode_offset`` counts POSITIONS in the feed as it stands right
    now, and positions move as a feed publishes. "Give me the next 10 I do not have" expressed
    as "skip the newest 10" is only equivalent while the feed is frozen. Measured on the
    2026-08-31 batch: feeds finished at 8, 8, 8, 8, 7, 7, 7 of a requested 10, because each had
    published 2-3 episodes since the previous run, so ``offset=10`` landed 2-3 items shallower
    than intended and re-selected episodes already ingested.

    Nothing was corrupted — ``skip_existing`` dropped the overlap correctly, which is the safety
    net working. But the drop happens AFTER ``max_episodes`` has already been spent on those
    items, so the limit is consumed by work that will not happen. The shortfall grows with the
    gap between runs.

    Filtering by GUID before the limit makes "10" mean ten episodes of actual work, and is
    immune to feed movement — the same principle ``corpus_metadata_index`` applies elsewhere:
    resolve by stable identity, never by position.

    Deliberately NOT the default, and deliberately not a change to ``episode_offset``: that flag
    is documented positional behaviour (#521, RSS_GUIDE, CONFIGURATION, and an E2E suite), and
    silently redefining it would break callers who want a genuine positional window.
    """
    out_dir = getattr(cfg, "output_dir", None)
    if not out_dir:
        # No output dir means nothing is on disk to compare against. Keep every item rather
        # than filtering against an empty set by accident.
        logger.warning(
            "episode_selection=unprocessed: no output_dir on the config; cannot tell what is "
            "already ingested, so NOTHING is filtered and --max-episodes still counts "
            "positions."
        )
        return items
    known = collect_existing_guids(str(out_dir))
    if not known:
        logger.info(
            "episode_selection=unprocessed: no episodes on disk under %s; nothing to filter "
            "(every feed item is a candidate).",
            cfg.output_dir,
        )
        return items

    kept: List[Any] = []
    dropped = 0
    unidentified = 0
    for it in items:
        guid = extract_item_guid(it)
        if not guid:
            # An item we cannot identify must stay a candidate: dropping it would silently
            # shrink the reachable feed, and skip_existing still guards the duplicate case.
            unidentified += 1
            kept.append(it)
            continue
        if str(guid).strip() in known:
            dropped += 1
            continue
        kept.append(it)

    logger.info(
        "episode_selection=unprocessed: %d feed item(s) already ingested and dropped BEFORE the "
        "limit; %d candidate(s) remain%s. This is what makes --max-episodes mean 'N episodes of "
        "work' rather than 'N positions in a feed that moves'.",
        dropped,
        len(kept),
        f" ({unidentified} item(s) had no guid and were kept)" if unidentified else "",
    )
    return kept


def _on_disk_guid_index(output_dir: str) -> Dict[str, Tuple[int, Dict[str, Any]]]:
    """``{guid: (on_disk_idx, episode_metadata)}`` for the corpus under ``output_dir``.

    The ``idx`` is read from the ``NNNN - Title.metadata.json`` filename prefix — the same number
    the transcript files carry — so a reprocess assigns each episode the idx its on-disk transcript
    is stored under. Assigning idx by feed-enumerate position (as the normal ingest path does) only
    aligned by luck when the newest feed items happened to be files 0001..N; aged-out episodes need
    their true on-disk idx or ``relabel_only``/``rediarize_only`` (which glob ``{idx} - *.txt``)
    cannot find them.
    """
    root = Path(output_dir)
    out: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    # The feed-nested patterns are NOT optional extras — ``feeds/<slug>/run_*/metadata/`` is the
    # layout production actually writes (every one of prod's 397 run dirs lives under it). With
    # only the flat patterns, ``reprocess_existing_only`` raised "no on-disk episode GUIDs were
    # found" against a corpus that was sitting right there, which reads as a missing corpus
    # rather than an unsupported layout. Same four patterns ``corpus_metadata_index`` scans.
    #
    # The candidates are DEDUPED to the newest run per episode before indexing, and sorted.
    # Without that, adding the feed-nested patterns would make the same guid appear in several
    # run dirs and let ``if guid in out: continue`` keep whichever ``Path.glob`` yielded first
    # — filesystem order, not even lexicographic. The ``idx`` this returns is load-bearing:
    # ``_reprocess_existing_episodes`` uses it so the relabel_only / rediarize_only transcript
    # glob matches, so a non-deterministic idx silently targets a SUPERSEDED run's transcript.
    # Same central membership rule ``corpus_metadata_index`` uses, for the same reason.
    candidates: List[Path] = []
    for pattern in (
        "run_*/metadata/*.metadata.json",
        "metadata/*.metadata.json",
        "feeds/*/run_*/metadata/*.metadata.json",
        "feeds/*/metadata/*.metadata.json",
    ):
        candidates.extend(sorted(root.glob(pattern)))

    # Deliberately NOT wrapped in ``except ImportError`` (the import is at module scope above).
    # It carried one until a test of that branch showed the fallback was worse than the failure
    # it handled: with no dedupe, ``if guid in out: continue`` keeps the FIRST candidate and the
    # globs are sorted ascending, so a reprocessed episode resolved to its OLDEST run every
    # time. That idx feeds the ``{idx} - *.txt`` transcript glob, so relabel_only /
    # rediarize_only would have re-derived a superseded transcript and written it over the
    # current one — silent corruption behind a WARNING and a zero exit.
    candidates = dedupe_metadata_paths_newest_run_per_episode(root, candidates)

    for meta_path in candidates:
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        episode = data.get("episode", {}) if isinstance(data, dict) else {}
        guid = episode.get("guid")
        if not guid or guid in out:
            continue
        # Leading digits of the filename are the episode idx: "0003 - Title.metadata.json" -> 3.
        digits = ""
        for ch in meta_path.name:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            continue
        out[str(guid)] = (int(digits), episode)
    return out


def _synthesize_feed_item(guid: str, episode_meta: Dict[str, Any]) -> ET.Element:
    """An RSS ``<item>`` reconstructed from on-disk metadata for an aged-out episode.

    Carries guid, title, pubDate to identify + place the episode, AND the ``description`` (+
    ``link``) from the stored episode block. The description is load-bearing: speaker detection
    reads the episode title+description to extract guest names (``_detect_speakers_for_episode`` ->
    ``extract_episode_description``), so a synthesized item WITHOUT it silently guts metadata-driven
    guest naming for every reconstructed episode on a reprocess — while live-served episodes in the
    same run keep it. It has NO enclosure, so ``media_url`` resolves to ``None`` — fine for
    ``relabel_only`` (reads the on-disk transcript), while ``rediarize_only`` resolves audio from
    the cache by guid. (Storing the enclosure URL at ingest is the forward-looking re-download fix.)
    """
    item = ET.Element("item")
    ET.SubElement(item, "guid").text = str(guid)
    ET.SubElement(item, "title").text = str(episode_meta.get("title") or "")
    published = episode_meta.get("published_date")
    if published:
        ET.SubElement(item, "pubDate").text = str(published)
    description = episode_meta.get("description")
    if description:
        ET.SubElement(item, "description").text = str(description)
    link = episode_meta.get("link")
    if link:
        ET.SubElement(item, "link").text = str(link)
    # DURATION. Carried so a reconstructed episode is PRICEABLE. The pre-flight cost gate values
    # a selection from each episode's audio duration; without this an aged-out episode reads as
    # unknown-duration and the gate can only guess at it. The parser reads
    # <itunes:duration> (rss/parser.py:491) and accepts a bare seconds string, which is exactly
    # what the stored metadata holds.
    duration_seconds = episode_meta.get("duration_seconds")
    if duration_seconds is not None:
        try:
            ET.SubElement(item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}duration").text = str(
                int(float(duration_seconds))
            )
        except (TypeError, ValueError):
            pass
    return item


def _reprocess_existing_episodes(
    feed: RssFeed,  # type: ignore[valid-type]
    feed_items: List[Any],
    cfg: config.Config,  # type: ignore[valid-type]
    total_items: int,
) -> List[Episode]:  # type: ignore[valid-type]
    """Build the reprocess episode set from the on-disk corpus, drift-immune (#876 follow-up).

    Every on-disk episode is reached: the live feed supplies items it still serves (preferred — a
    real enclosure for audio), and the rest are reconstructed from on-disk metadata. Each episode is
    given its on-disk idx so the ``relabel_only`` / ``rediarize_only`` transcript glob matches.
    """
    if cfg.output_dir is None:
        raise ValueError("reprocess_existing_only requires output_dir to locate the on-disk corpus")
    guid_index = _on_disk_guid_index(cfg.output_dir)
    if not guid_index:
        raise ValueError(
            "reprocess_existing_only is set but no on-disk episode GUIDs were found under "
            f"{cfg.output_dir}/run_*/metadata/. Wrong --output-dir, or the corpus is not present."
        )
    # The DENOMINATOR, captured before the work-list narrows guid_index below. "32 of 678" is
    # the line whose absence let a 32-episode repair select all 678 unnoticed for six hours.
    on_disk_total = len(guid_index)
    feed_by_guid: Dict[str, Any] = {}
    for it in feed_items:
        g = extract_item_guid(it)
        if g in guid_index and g not in feed_by_guid:
            feed_by_guid[g] = it

    # A WORK-LIST RESTRICTS THE RUN. It does not merely nominate episodes within it.
    #
    # 2026-08-19 incident. `--reprocess-episode-ids <32 episodes>` re-transcribed ~181 episodes
    # across healthy feeds, downloaded 15 GB of fresh media, drained the operator's Deepgram
    # balance to zero, and never reached the 32 it was asked to repair. Cost: ~$50 and six hours,
    # for nothing.
    #
    # Every part behaved as written. `reprocess_episode_ids` implies `reprocess_existing_only`,
    # which correctly stops NEW episodes being fetched — and then makes the episode set the WHOLE
    # on-disk corpus. The work-list's only other job was to force its members past
    # `skip_existing`... which defaults to False, is unset in cloud_balanced, and is not passed by
    # reprocess-prod.yml. Nothing was being skipped, so "force past the skip" selected everything.
    # The one guard that mentions this (`_warn_reprocess_existing_only_without_source`) warns that
    # matched episodes "would simply be skipped under skip_existing" — describing a world where
    # that flag is on.
    #
    # Naming 32 episodes can only ever mean "these 32". Restricting here makes that true
    # regardless of skip_existing, the profile, or which flags the caller remembered.
    wanted_ids = set(getattr(cfg, "reprocess_episode_ids", None) or ())
    if wanted_ids:
        # Register the ask ONCE per process, and this feed's hits, so the end of the batch can
        # report against the denominator. No single feed can do that: a 32-episode list drawn
        # from two feeds matches nothing in the other twelve, which is normal.
        from ..worklist_report import get_worklist_report

        _report = get_worklist_report()
        _report.request(wanted_ids)
        _report.mark_feed_searched()

        kept = {
            guid: entry
            for guid, entry in guid_index.items()
            if guid in wanted_ids or str(entry[1].get("episode_id") or "") in wanted_ids
        }
        _report.mark_matched(
            [g for g in kept] + [str(e[1].get("episode_id") or "") for e in kept.values()]
        )
        # NO MATCHES IN *THIS FEED* IS NORMAL AND MUST NOT FAIL THE RUN.
        #
        # Prod is multi-feed: cli.py loops feed_targets and builds a per-feed config with its own
        # output_dir (feeds/<slug>), while every feed's config carries the WHOLE work-list. A
        # 32-episode list drawn from one feed therefore matches nothing in the other 13 — the
        # normal case, not an error. Raising here would be classified "hard"
        # (corpus_operations.py:176-179 — a ValueError not naming an RSS fetch/parse failure) and
        # would exit the batch red with ~11 incidents logged against healthy feeds, even when all
        # 32 targets repaired perfectly. The first version of this guard did exactly that; it was
        # tested only against a single-feed corpus, which is not the topology prod runs.
        #
        # Returning an empty set keeps the safety property that matters — a work-list run NEVER
        # falls back to the whole corpus — while letting the other feeds proceed.
        if not kept:
            logger.info(
                "reprocess work-list: none of the %d listed episode(s) are in this feed's corpus "
                "(%s) — nothing to do here; other feeds are unaffected",
                len(wanted_ids),
                cfg.output_dir,
            )
            return []
        logger.info(
            "reprocess work-list: restricting this run to %d of %d on-disk episodes in %s",
            len(kept),
            len(guid_index),
            cfg.output_dir,
        )
        guid_index = kept

    episodes: List[Episode] = []  # type: ignore[valid-type]
    reconstructed = 0
    for guid, (idx, episode_meta) in sorted(guid_index.items(), key=lambda kv: kv[1][0]):
        item = feed_by_guid.get(guid)
        if item is None:
            item = _synthesize_feed_item(guid, episode_meta)
            reconstructed += 1
        episodes.append(create_episode_from_item(item, idx, feed.base_url))

    logger.info(
        "reprocess existing-only: %d on-disk episodes reached (%d served live by the %d-item feed, "
        "%d reconstructed from on-disk metadata after ageing out of the feed window)",
        len(episodes),
        len(episodes) - reconstructed,
        total_items,
        reconstructed,
    )
    # PRICE IT BEFORE SPENDING IT. This is the last point at which the run has cost nothing;
    # everything after it downloads media and calls a paid transcriber. Raising here is safe —
    # selection is on the main thread, before any worker exists.
    from ..selection_gate import enforce_selection_budget

    enforce_selection_budget(
        episodes, cfg, available=on_disk_total, scope=str(cfg.output_dir or "")
    )
    return episodes


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
        # #876 migration mode: the episode set is the WHOLE on-disk corpus, driven by on-disk
        # METADATA rather than the live-feed intersection. Feeds drift — episodes scroll out of the
        # feed's fetch window — so intersecting the live feed silently shrinks the reachable corpus
        # over time (only ~40 of 91 were reachable once). Reconstructing the aged-out episodes from
        # their on-disk metadata keeps a reprocess reaching every episode we have.
        return _reprocess_existing_episodes(feed, items, cfg, total_items)
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

        if str(getattr(cfg, "episode_selection", "position") or "position") == "unprocessed":
            items = _drop_already_ingested(items, cfg)

        if cfg.episode_offset:
            items = items[cfg.episode_offset :]

        if cfg.max_episodes is not None:
            items = items[: cfg.max_episodes]

    logger.info(
        "Episodes to process: %s of %s (after order/date filter/%soffset/limit)",
        len(items),
        total_items,
        "unprocessed-filter/" if getattr(cfg, "episode_selection", None) == "unprocessed" else "",
    )

    episodes = [
        create_episode_from_item(item, idx, feed.base_url)
        for idx, item in enumerate(items, start=1)
    ]
    logger.debug("Materialized %s episode objects", len(episodes))
    from ..selection_gate import enforce_selection_budget

    enforce_selection_budget(episodes, cfg, available=total_items, scope=str(cfg.output_dir or ""))
    return episodes

"""Apply operator overrides (#2283) at the seams where each field is decided.

Called from ``run_pipeline`` right after the feed is fetched and its episodes are prepared, and
BEFORE the language gate, so an override that gives a feed its language is what lets it in.

Where each field lands, so every downstream reader sees the overridden value:

* feed ``language``  -> ``cfg.language_override`` (outranks the feed's tag in every resolver);
* feed ``title`` / ``description`` / ``authors`` -> the ``RssFeed``; ``description`` /
  ``image_url`` also -> the extracted ``FeedMetadata`` the artifacts are written from;
* feed ``hosts``     -> REPLACES the detected host set (applied after host detection);
* episode ``title`` / ``description`` / ``published_date`` / ``image_url`` -> the episode's RSS
  ``<item>`` element itself. Every reader re-parses that element (metadata, summarisation,
  selection, episode ids), so patching it is the one seam they all share. The file-safe title is
  NOT changed, so an episode's files keep their names and nothing is re-downloaded;
* episode ``hosts`` / ``guests`` / ``speaker_renames`` -> carried on the ``Episode`` as
  ``override_fields`` and read where the per-episode hosts, guests and published speaker names
  are decided.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET  # nosec B405 - builds elements, parses nothing
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from podcast_scraper import overrides as ov

logger = logging.getLogger(__name__)

_ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"


def corpus_root_for(cfg: Any) -> Optional[Path]:
    """Where ``overrides.json`` lives for this run: the corpus root, beside ``feeds.spec.yaml``."""
    from .corpus_operations import corpus_parent_for_manifest_stamp_from_cfg

    parent = corpus_parent_for_manifest_stamp_from_cfg(cfg)
    if parent:
        return Path(parent)
    out = getattr(cfg, "output_dir", None)
    return Path(str(out)) if out else None


def load_for_run(cfg: Any) -> Tuple[Optional[ov.FeedFields], Dict[str, ov.EpisodeFields]]:
    """This feed's overrides: ``(feed fields or None, {guid: episode fields})``.

    A broken ``overrides.json`` RAISES (see ``overrides.load_overrides``): ignoring it would
    publish exactly what an operator overrode.
    """
    root = corpus_root_for(cfg)
    url = getattr(cfg, "rss_url", None) or ""
    if root is None or not url:
        return None, {}
    doc = ov.load_overrides(root)
    entry = doc.feeds.get(ov.feed_key(url))
    if entry is None:
        return None, {}
    return entry.fields, dict(entry.episodes)


def apply_feed_fields(cfg: Any, feed: Any, feed_metadata: Any, fields: ov.FeedFields) -> Any:
    """Patch the feed and its extracted metadata; returns the (possibly updated) config."""
    applied = []
    if fields.language:
        cfg = cfg.model_copy(update={"language_override": fields.language})
        applied.append("language")
    if fields.title:
        feed.title = fields.title
        applied.append("title")
    if fields.description:
        feed.description = fields.description
        applied.append("description")
    if fields.authors is not None:
        feed.authors = list(fields.authors)
        applied.append("authors")
    if feed_metadata is not None:
        if fields.description and hasattr(feed_metadata, "description"):
            feed_metadata.description = fields.description
        if fields.image_url and hasattr(feed_metadata, "image_url"):
            feed_metadata.image_url = fields.image_url
            applied.append("image_url")
    if applied:
        logger.info("    operator overrides applied to the feed: %s", ", ".join(applied))
    return cfg


def _set_child_text(item: Any, tag: str, text: str) -> None:
    child = item.find(tag)
    if child is None:
        child = ET.SubElement(item, tag)
    child.text = text


def _rfc2822(day: Any) -> str:
    from datetime import datetime, timezone
    from email.utils import format_datetime

    return format_datetime(datetime(day.year, day.month, day.day, tzinfo=timezone.utc))


def episode_guid(episode: Any) -> Optional[str]:
    """The episode's ``<guid>`` from its RSS item — the key episode overrides are stored under."""
    item = getattr(episode, "item", None)
    if item is None:
        return None
    guid = item.find("guid")
    return guid.text.strip() if guid is not None and guid.text else None


def apply_episode_fields(episode: Any, fields: ov.EpisodeFields) -> None:
    """Patch one episode's ``<item>`` and carry the derived-field overrides on the episode."""
    item = getattr(episode, "item", None)
    if item is not None:
        if fields.title:
            _set_child_text(item, "title", fields.title)
            episode.title = fields.title  # displayed title; `title_safe` (file names) unchanged
        if fields.description:
            _set_child_text(item, "description", fields.description)
        if fields.published_date is not None:
            _set_child_text(item, "pubDate", _rfc2822(fields.published_date))
        if fields.image_url:
            image = item.find(f"{{{_ITUNES_NS}}}image")
            if image is None:
                image = ET.SubElement(item, f"{{{_ITUNES_NS}}}image")
            image.set("href", fields.image_url)
    # Derived fields are read later, where they are decided; the episode carries them there.
    episode.override_fields = fields


def apply_to_run(
    cfg: Any, feed: Any, feed_metadata: Any, episodes: list
) -> Tuple[Any, list, Optional[ov.FeedFields]]:
    """Apply this feed's overrides; returns ``(cfg, episodes, feed fields or None)``.

    Episodes are matched by GUID; an override for a GUID the feed does not contain is logged,
    not an error (the episode may have aged out of the feed window).
    """
    feed_fields, by_guid = load_for_run(cfg)
    if feed_fields is not None:
        cfg = apply_feed_fields(cfg, feed, feed_metadata, feed_fields)
    if by_guid:
        seen = set()
        for episode in episodes:
            guid = episode_guid(episode)
            if guid and guid in by_guid:
                apply_episode_fields(episode, by_guid[guid])
                seen.add(guid)
        missing = sorted(set(by_guid) - seen)
        if missing:
            logger.info("    episode overrides with no matching episode in this run: %s", missing)
        if seen:
            logger.info("    operator overrides applied to %d episode(s)", len(seen))
    return cfg, episodes, feed_fields


def _fields_of(episode: Any) -> Optional[ov.EpisodeFields]:
    """The episode's override fields — ONLY a real ``EpisodeFields``.

    Not "whatever ``override_fields`` holds": a test's MagicMock episode answers every attribute
    with another Mock, which is truthy and would read as an override of every field — measured,
    it emptied the detected guests of eleven unit tests.
    """
    fields = getattr(episode, "override_fields", None)
    return fields if isinstance(fields, ov.EpisodeFields) else None


def override_hosts(episode: Any) -> Optional[list]:
    """The episode's ``hosts`` override, or ``None`` when it has none (``[]`` means: no hosts)."""
    fields = _fields_of(episode)
    return list(fields.hosts) if fields is not None and fields.hosts is not None else None


def override_guests(episode: Any) -> Optional[list]:
    """The episode's ``guests`` override, or ``None`` when it has none (``[]`` means: no guests)."""
    fields = _fields_of(episode)
    return list(fields.guests) if fields is not None and fields.guests is not None else None


def speaker_renames(episode: Any) -> Dict[str, str]:
    """The episode's ``speaker_renames`` (published name -> name to publish); empty when none."""
    fields = _fields_of(episode)
    return dict(fields.speaker_renames) if fields is not None and fields.speaker_renames else {}


def episode_language(episode: Any) -> Optional[str]:
    """The episode's ``language`` override (an ISO code), or ``None`` when it has none."""
    fields = _fields_of(episode)
    return fields.language if fields is not None and fields.language else None


def gate_languages(cfg: Any, episodes: list, feed_refusal: Any) -> Tuple[Any, list, Optional[str]]:
    """The language gate with episode overrides: ``(cfg, episodes to process, refusal or None)``.

    ONE LANGUAGE PER RUN: the config, and every stage reading it, carries a single language for
    the feed. So:

    * feed refused (no language, or not enabled) but some episodes carry a language override —
      if they all name the SAME enabled language, the run proceeds for exactly those episodes in
      that language; mixed languages are refused, with the reason;
    * feed accepted — an episode whose override names a DIFFERENT language is skipped, with the
      reason, rather than being processed in the feed's language.
    """
    from podcast_scraper.languages import is_language_enabled, resolve_config_language

    refusal = feed_refusal(cfg)
    with_lang = [e for e in episodes if episode_language(e)]
    if refusal is not None:
        langs = sorted({lang for e in with_lang if (lang := episode_language(e))})
        if not langs:
            return cfg, [], refusal
        if len(langs) > 1:
            return (
                cfg,
                [],
                (
                    f"{refusal} Episode overrides name more than one language "
                    f"({', '.join(langs)}); one run processes one language."
                ),
            )
        lang = langs[0]
        if not is_language_enabled(lang):
            return cfg, [], f"{refusal} The episode override language {lang!r} is not enabled."
        logger.info(
            "    feed refused on language, but %d episode(s) carry a %r override: processing "
            "only those",
            len(with_lang),
            lang,
        )
        return cfg.model_copy(update={"language_override": lang}), with_lang, None

    _raw, feed_lang, _src = resolve_config_language(cfg)
    keep = []
    for e in episodes:
        lang = episode_language(e)
        if lang and lang != feed_lang:
            logger.warning(
                "SKIPPING episode %r: its override language %r differs from the feed's %r, and "
                "one run processes one language",
                getattr(e, "title", "?"),
                lang,
                feed_lang,
            )
            continue
        keep.append(e)
    return cfg, keep, None

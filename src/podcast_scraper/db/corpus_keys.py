"""The natural keys of the corpus database's ``sources`` and ``content_items`` rows (CRFC-005 §3.1).

The one place they are derived: the pipeline's row writer and the file-to-rows importer both call
it, so a row written either way carries the key today's file-era code already gives the item. Every
function takes the item's metadata document (a ``*.metadata.json`` read as a dict) and returns a
string (``raw_feed_id`` returns ``None`` for a news item); none reads a file or a
database. A key the metadata does not carry raises
:class:`KeyNotDerivable`: a caller decides what to do with such an item, this module never invents a
key for it.

Where each rule comes from (a rule changes only with its source):

- ``item_natural_id``: a podcast's ``episode.episode_id``
  (``server.corpus_catalog._feed_and_episode_ids``, the id the GI and KG roots write; ``gi_as_v4``:
  "the root ``episode_id`` is also the ``item_id``"); a news item's top-level ``item_id``
  (``news_items.served_item``).
- ``source_natural_id``: a podcast's ``kg.pipeline._podcast_node_id`` of its feed id (the id the
  KG's Podcast node and GI 4.0's ``ContentItem.source_id`` carry); a news item's top-level
  ``source_id`` (``news_items.F_SOURCE_ID``).
- ``raw_feed_id``: a podcast's ``feed.feed_id`` (``CatalogEpisodeRow.feed_id``). A news item has
  none: ``None`` (``sources.raw_feed_id`` is nullable and unique, so many NULLs are fine).
- ``external_item_id``: a podcast's ``episode.guid`` (``generate_episode_id`` makes it the episode
  id when present), else its episode id; a news item's ``item_id``, which is stable, unique and
  independent of order (its ``news_article.transport_ids`` stay aliases in the metadata).
- ``item_slug``: :func:`item_slug`, moved here from ``server.slugs.episode_slug``;
  :func:`item_slug_for` applies it to a metadata document. A news item has no slug today (the
  consumer API gives it none), so its rule here is new: the slug of ``(source_id, item_id)``.

``content_items.natural_id`` is unique, so two feeds writing the same guid are refused by the
database and listed by the importer, never merged. A podcast with no ``episode_id`` has no
``item_natural_id`` and raises :class:`KeyNotDerivable` (none in the 2026-11-24 prod snapshot).
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Optional

from podcast_scraper.identity.slugify import slugify
from podcast_scraper.kg.pipeline import _podcast_node_id

NEWS_ARTICLE = "news_article"

_HASH_LEN = 10  # hex chars of the stable discriminator suffix


class KeyNotDerivable(ValueError):
    """The metadata does not carry what the key is derived from."""


def _str(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _block(doc: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    block = doc.get(name)
    return block if isinstance(block, Mapping) else {}


def is_news(doc: Mapping[str, Any]) -> bool:
    """Whether ``doc`` is a news item's metadata; any other document is a podcast episode's."""
    return doc.get("content_type") == NEWS_ARTICLE


def item_natural_id(doc: Mapping[str, Any]) -> str:
    """The item's written id, the one GI and KG name it by."""
    key = (
        _str(doc.get("item_id")) if is_news(doc) else _str(_block(doc, "episode").get("episode_id"))
    )
    if not key:
        raise KeyNotDerivable("the metadata names no item id")
    return key


def source_natural_id(doc: Mapping[str, Any]) -> str:
    """The item's source's written id: ``podcast:{slug}`` for a podcast (KG's id), the news item's
    own ``source:{hash}``."""
    if is_news(doc):
        key = _str(doc.get("source_id"))
    else:
        key = _podcast_node_id(_str(_block(doc, "feed").get("feed_id")) or "")
    if not key:
        raise KeyNotDerivable("the metadata names no source id")
    return key


def raw_feed_id(doc: Mapping[str, Any]) -> Optional[str]:
    """The feed id the catalog serves (``feed.feed_id``, ``sha256:{hex}`` of the feed URL); ``None``
    for a news item, which has no feed."""
    if is_news(doc):
        return None
    key = _str(_block(doc, "feed").get("feed_id"))
    if not key:
        raise KeyNotDerivable("the metadata names no feed id")
    return key


def external_item_id(doc: Mapping[str, Any]) -> str:
    """The item's id inside its source, stable across re-scrapes."""
    if is_news(doc):
        key = _str(doc.get("item_id"))
    else:
        episode = _block(doc, "episode")
        key = _str(episode.get("guid")) or _str(episode.get("episode_id"))
    if not key:
        raise KeyNotDerivable("the metadata names no id of the item inside its source")
    return key


def item_slug(feed_id: str, episode_id: Optional[str], metadata_relpath: str) -> str:
    """Deterministic, URL-safe slug ``{feed-slug}-{hash}`` for one episode.

    The hash is taken over the stable identity ``(feed_id, episode_id)``; when an episode has no
    ``episode_id`` it falls back to the metadata relpath so every episode still gets a
    deterministic slug. Title and run suffix never affect it, so it survives re-scrapes.
    """
    if episode_id:
        key = f"{feed_id}\x00{episode_id}"
    else:
        key = f"{feed_id}\x00path:{metadata_relpath}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:_HASH_LEN]
    return f"{_feed_slug(feed_id)}-{digest}"


def item_slug_for(doc: Mapping[str, Any], *, metadata_relpath: str = "") -> str:
    """The item's slug. A podcast: :func:`item_slug` of its catalog feed id and episode id (the
    relpath only matters when it has no episode id). A news item: of its source id and item id."""
    if is_news(doc):
        return item_slug(source_natural_id(doc), item_natural_id(doc), "")
    feed = _str(_block(doc, "feed").get("feed_id")) or ""
    return item_slug(feed, _str(_block(doc, "episode").get("episode_id")), metadata_relpath)


def _feed_slug(feed_id: str) -> str:
    """Readable, URL-safe prefix from a feed id; ``"feed"`` when empty/unslug-able."""
    fid = (feed_id or "").strip()
    if not fid:
        return "feed"
    try:
        return slugify(fid)
    except ValueError:
        return "feed"

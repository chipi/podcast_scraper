"""The column values of the corpus database's ``sources`` and ``content_items`` rows.

CRFC-005 §3.1.

The one place an item's metadata document becomes those rows: the pipeline's row writer and the
file-to-rows importer both call it, so a row written either way says what the file-era catalog says
about the item today. Like :mod:`podcast_scraper.db.corpus_keys`, every function takes the item's
metadata document (a ``*.metadata.json`` read as a dict), reads no file and no database, and raises
:class:`~podcast_scraper.db.corpus_keys.KeyNotDerivable` when a key is missing.

Which copy of an item is written is the caller's: the newest run per episode
(``search.corpus_scope.discover_metadata_files``) and, for news, the copy that decides the item
(``news_items.current_copies``). Superseded copies are not rows.

What each column comes from (a column changes only with its source):

- ``sources``: ``title``, ``category`` and ``licence`` from the catalog's feed readers
  (``server.corpus_catalog._feed_display_title``, ``_feed_category``, ``_feed_licence``);
  ``language`` from ``languages.feed_language``; each from the newest item that states it
  (:func:`source_row_for`: the catalog's ``aggregate_feeds`` ordered by :func:`newest_key`). A
  news source has no feed block, so these are ``None`` except ``title``: the publisher's name,
  else the URL's host.
- Times: UTC datetimes; one stored with no offset is UTC (:func:`_instant`).
- ``content_items``: ``title`` as the catalog shows it (``_episode_title``, ``"(untitled)"`` when
  missing; a news item's ``title`` as ``news_items.served_item`` gives it); ``duration_s`` from
  ``episode.duration_seconds``; ``language`` from ``languages.episode_language`` (the feed's when
  the episode has none) and a news item's own ``language``; ``editorial_type`` only for news.
- ``served``: :func:`served_for`.
- ``unusable``: ``languages.episode_is_unusable`` (D-44); a news item's top-level ``unusable``.
  ``withheld``: a news item whose ``serveable`` is ``none``.
- ``type_attrs``: the fields surfaces only display (DECISIONS rule 16).

Left to others: ``publisher_id`` and ``artwork_asset_id`` are references the writer resolves to row
ids; ``has_transcript``, ``has_gi``, ``has_kg`` and ``has_moments`` come from the item's other
files, so their stages' writers set them; ``filed_at``, ``period_start``, ``period_end`` and
``observed_at`` have no source in a podcast's or a news item's metadata and stay null
(DECISIONS rule 18).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Tuple

from podcast_scraper.content_types import item_content_type
from podcast_scraper.db.corpus_keys import (
    external_item_id,
    is_news,
    item_natural_id,
    item_slug_for,
    raw_feed_id,
    source_natural_id,
)
from podcast_scraper.languages import (
    episode_is_unusable,
    episode_language,
    feed_language,
    UNUSABLE_FIELD,
)
from podcast_scraper.news_items import (
    _publisher,
    _site,
    EDITORIAL_TYPES,
    F_BYLINE,
    F_EDITORIAL_TYPE,
    F_ITEM_CLASS,
    F_LANGUAGE,
    F_PAYWALLED,
    F_PUBLISHED_AT,
    F_TEXT_STATUS,
    F_TITLE,
    F_URL,
    is_shown,
    NEWS_ARTICLE,
    serveable,
    SERVEABLE_NONE,
)
from podcast_scraper.publishers import Registry
from podcast_scraper.server.corpus_catalog import (
    _episode_title,
    _feed_category,
    _feed_display_title,
    _feed_licence,
    _optional_non_negative_int,
)

#: ``content_items.served`` (CRFC-005 §3.1). The schema also allows 2, "served and in the newest
#: run"; it is written nowhere, because both of today's catalog shapes are the same rows (a census
#: of snapshot-prod-20261124: 3,522 and 3,522, none in only one).
NOT_SERVED = 0
SERVED = 1

_EPISODE_TEXT_ATTRS = ("description", "link", "image_url")
_CONTENT_TEXT_ATTRS = ("media_url", "media_type")
_NEWS_TEXT_ATTRS = (F_TEXT_STATUS, F_ITEM_CLASS)
#: The news writer's run time, top level on the item (``news.document``).
_F_INGESTED_AT = "ingested_at"


def _str(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _block(doc: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    block = doc.get(name)
    return block if isinstance(block, Mapping) else {}


def _instant(raw: Any) -> Optional[datetime]:
    """An ISO 8601 time as a UTC datetime, or ``None`` when it is missing or unreadable.

    A time stored with no offset is UTC. A podcast's ``published_date`` lost its marker where
    ``email.utils.parsedate_to_datetime`` returns a naive datetime for the zone ``-0000``, which
    RFC 5322 defines as UTC (953 of the 958 such episodes on snapshot-prod-20261124 equal their
    live pubDate). ``processing_timestamp`` is ``datetime.now()`` in a container with no TZ set,
    whose clock is UTC (database, 2026-11-26, against each item's ``licence.retrieved_at``).
    """
    text = _str(raw)
    if text is None:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _source_title(doc: Mapping[str, Any], registry: Optional[Registry]) -> str:
    """The source's display name. A podcast: its feed's title. A news item names none, so it is
    the label readers see today: the publisher's name (``news_items._publisher``'s order), else
    the URL's host (``news_items._site``)."""
    if is_news(doc):
        if registry is None:
            raise ValueError("a news source's title needs the publisher registry")
        publisher = _publisher(doc, registry)
        title = (publisher or {}).get("name") or _site(doc)
    else:
        title = _feed_display_title(dict(doc))
    if not title:
        raise ValueError("the metadata names no source title")
    return title


def _unusable_episode(doc: Mapping[str, Any]) -> bool:
    # episode_is_unusable answers None for anything that is not a dict, so a copy is passed.
    return episode_is_unusable(dict(doc)) is not None


def served_for(doc: Mapping[str, Any]) -> int:
    """Whether a surface may show the item: a podcast episode the pipeline completed (D-44), a news
    item :func:`news_items.is_shown` shows."""
    if is_news(doc):
        return SERVED if is_shown(doc) else NOT_SERVED
    return NOT_SERVED if _unusable_episode(doc) else SERVED


def source_row(doc: Mapping[str, Any], *, registry: Optional[Registry] = None) -> dict[str, Any]:
    """The ``sources`` columns as one item's metadata states them, keyed by column name. Items of
    one show can state them differently; :func:`source_row_for` decides between them. A news
    item's source needs ``registry`` (``publishers.load_registry()``) for its title."""
    podcast = None if is_news(doc) else dict(doc)
    return {
        "natural_id": source_natural_id(doc),
        "raw_feed_id": raw_feed_id(doc),
        "content_type": item_content_type(dict(doc)),
        "title": _source_title(doc, registry),
        "language": feed_language(podcast) if podcast else None,
        "category": _feed_category(podcast) if podcast else None,
        "licence": _feed_licence(podcast) if podcast else None,
    }


#: An item that states no readable publish time sorts below every dated one.
NewestKey = Tuple[int, float, str]


def newest_key(published_at: Optional[datetime], natural_id: str) -> NewestKey:
    """The order in which items decide their source's columns: the greater key is newer. Dated
    items above undated ones, then by the UTC instant, then by the greater ``natural_id``, so the
    order never depends on the order items arrive in. The pipeline's row writer
    (``corpus_writer``) compares the same key against the rows it holds."""
    if published_at is None:
        return (0, 0.0, natural_id)
    return (1, published_at.timestamp(), natural_id)


def _newest(doc: Mapping[str, Any]) -> NewestKey:
    raw = doc.get(F_PUBLISHED_AT) if is_news(doc) else _block(doc, "episode").get("published_date")
    return newest_key(_instant(raw), item_natural_id(doc))


#: The ``sources`` columns an item states for its source: what ``content_items.source_attrs``
#: holds (corpus_0008), so a source row can be merged again from its items' rows.
SOURCE_ATTRS = ("raw_feed_id", "title", "language", "category", "licence")


def source_attrs(doc: Mapping[str, Any], *, registry: Optional[Registry] = None) -> dict[str, Any]:
    """The item's ``content_items.source_attrs``: the :data:`SOURCE_ATTRS` of its
    :func:`source_row`."""
    row = source_row(doc, registry=registry)
    return {column: row[column] for column in SOURCE_ATTRS}


def merge_source_attrs(newest_first: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Each of :data:`SOURCE_ATTRS` from the first of ``newest_first`` that states it (not
    ``None``): the catalog's rule (``corpus_catalog.aggregate_feeds``). The one merge
    :func:`source_row_for` and the pipeline's writer (``corpus_writer``) both use."""
    merged: dict[str, Any] = {column: None for column in SOURCE_ATTRS}
    for attrs in newest_first:
        for column in SOURCE_ATTRS:
            if merged[column] is None and attrs.get(column) is not None:
                merged[column] = attrs[column]
    return merged


def source_row_for(
    docs: Iterable[Mapping[str, Any]], *, registry: Optional[Registry] = None
) -> dict[str, Any]:
    """The ``sources`` columns of the show whose served items are ``docs``:
    :func:`merge_source_attrs` over them in :func:`newest_key` order (newest first)."""
    ordered = sorted(docs, key=_newest, reverse=True)
    rows = [source_row(doc, registry=registry) for doc in ordered]
    if not rows:
        raise ValueError("a source needs at least one item")
    keys = {(row["natural_id"], row["raw_feed_id"]) for row in rows}
    if len(keys) > 1:
        raise ValueError(f"items of different sources: {sorted(map(str, keys))}")
    first = rows[0]
    return {
        "natural_id": first["natural_id"],
        "content_type": first["content_type"],
        **merge_source_attrs(rows),
    }


def _podcast_type_attrs(doc: Mapping[str, Any]) -> dict[str, Any]:
    episode = _block(doc, "episode")
    content = _block(doc, "content")
    attrs: dict[str, Any] = {}
    number = _optional_non_negative_int(episode.get("episode_number"))
    if number is not None:
        attrs["episode_number"] = number
    for block, names in ((episode, _EPISODE_TEXT_ATTRS), (content, _CONTENT_TEXT_ATTRS)):
        for name in names:
            value = _str(block.get(name))
            if value is not None:
                attrs[name] = value
    if isinstance(episode.get("licence"), Mapping):
        attrs["licence"] = dict(episode["licence"])
    return attrs


def _news_type_attrs(doc: Mapping[str, Any]) -> dict[str, Any]:
    own = _block(doc, NEWS_ARTICLE)
    attrs: dict[str, Any] = {"serveable": serveable(doc)}
    url = _str(doc.get(F_URL))
    if url is not None:
        attrs["url"] = url
    paywalled = doc.get(F_PAYWALLED)
    if isinstance(paywalled, bool):
        attrs["paywalled"] = paywalled
    byline = own.get(F_BYLINE)
    if isinstance(byline, list):
        names = [n for n in byline if isinstance(n, str) and n.strip()]
        if names:
            attrs["byline"] = names
    for name in _NEWS_TEXT_ATTRS:
        value = _str(own.get(name))
        if value is not None:
            attrs[name] = value
    if own.get("roundup") is True:
        attrs["roundup"] = True
    return attrs


def item_row(doc: Mapping[str, Any], *, metadata_relpath: str = "") -> dict[str, Any]:
    """The ``content_items`` columns the item's metadata decides, keyed by column name.
    ``metadata_relpath`` matters only for the slug of an episode with no ``episode_id``."""
    if is_news(doc):
        own = _block(doc, NEWS_ARTICLE)
        type_attrs = _news_type_attrs(doc)
        editorial = own.get(F_EDITORIAL_TYPE)
        title = _str(doc.get(F_TITLE)) or ""
        published = _instant(doc.get(F_PUBLISHED_AT))
        ingested = _instant(doc.get(_F_INGESTED_AT))
        duration = None
        language = _str(doc.get(F_LANGUAGE))
        unusable = doc.get(UNUSABLE_FIELD) is True
        withheld = serveable(doc) == SERVEABLE_NONE
    else:
        podcast = dict(doc)
        episode = _block(doc, "episode")
        type_attrs = _podcast_type_attrs(doc)
        editorial = None
        title = _episode_title(podcast)
        published = _instant(episode.get("published_date"))
        ingested = _instant(_block(doc, "processing").get("processing_timestamp"))
        duration = _optional_non_negative_int(episode.get("duration_seconds"))
        language = episode_language(podcast)
        unusable = _unusable_episode(doc)
        withheld = False
    return {
        "natural_id": item_natural_id(doc),
        "slug": item_slug_for(doc, metadata_relpath=metadata_relpath),
        "external_item_id": external_item_id(doc),
        "content_type": item_content_type(dict(doc)),
        "editorial_type": editorial if editorial in EDITORIAL_TYPES else None,
        "title": title,
        "published_at": published,
        "filed_at": None,
        "period_start": None,
        "period_end": None,
        "observed_at": None,
        "ingested_at": ingested,
        "duration_s": duration,
        "language": language,
        "served": served_for(doc),
        "withheld": withheld,
        "unusable": unusable,
        "type_attrs": type_attrs,
    }

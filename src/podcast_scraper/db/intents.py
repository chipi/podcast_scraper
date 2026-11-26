"""Named query intents over the corpus rows: the only way a route reads them (DECISIONS rule 9).

Each intent is one fixed statement with array parameters; no client text reaches SQL, and the
number of statements an intent runs is fixed by its arguments, asserted by
``tests/unit/podcast_scraper/db/test_db_intents.py``. This module holds the item intents the
Player's routes need first (CRFC-005 §2; e06-analysis reads-player-and-studio AP-01, AP-02, AP-03,
AP-04, AP-23):

- items by key: slug, natural id, ``(raw feed id, external item id)``; every item whatever its
  ``served``, which each row carries, since a listener's saved item may no longer be served;
- the cards of a set of items (``item_cards``, one row per item);
- served items newest first, of some sources or of one content type, with a keyset cursor
  ``(published_at, item_id)``: dated items first, then undated ones, item id descending within a
  time.

Results are plain dicts keyed by column name. No heavy column is read unless named: ``type_attrs``
is not in :data:`ITEM_COLUMNS`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Sequence, Tuple

from sqlalchemy import and_, or_, select, tuple_
from sqlalchemy.engine import Connection

from podcast_scraper.db.schema import content_items, item_cards, sources

#: The most keys one by-key call takes (AP-01: rails, favourites and world sets reach hundreds).
MAX_KEYS = 500
#: The longest page a list intent returns.
MAX_PAGE = 200

ITEM_COLUMNS = (
    "item_id",
    "natural_id",
    "slug",
    "source_id",
    "external_item_id",
    "content_type",
    "editorial_type",
    "title",
    "published_at",
    "duration_s",
    "language",
    "served",
    "has_transcript",
    "has_gi",
    "has_kg",
    "has_moments",
    "withheld",
    "unusable",
)

CARD_COLUMNS = (
    "item_id",
    "title",
    "source_title",
    "published_at",
    "duration_s",
    "artwork_asset_id",
    "lede",
    "summary_bullets",
    "top_topics",
    "has_transcript",
    "has_gi",
    "has_kg",
    "has_moments",
    "withheld",
    "unusable",
)

#: A list cursor: the last row's ``(published_at, item_id)``.
Cursor = Tuple[Optional[datetime], int]

_ITEM = [content_items.c[name] for name in ITEM_COLUMNS]
_CARD = [item_cards.c[name] for name in CARD_COLUMNS]


def _keys(values: Sequence[Any], what: str) -> list[Any]:
    keys = list(dict.fromkeys(values))
    if len(keys) > MAX_KEYS:
        raise ValueError(f"{len(keys)} {what} is more than {MAX_KEYS}")
    return keys


def _rows(conn: Connection, statement: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(statement).mappings()]


def items_by_slugs(conn: Connection, slugs: Sequence[str]) -> list[dict[str, Any]]:
    """The items with these slugs, any ``served``; unknown slugs are left out (AP-01)."""
    keys = _keys(slugs, "slugs")
    if not keys:
        return []
    return _rows(conn, select(*_ITEM).where(content_items.c.slug.in_(keys)))


def items_by_natural_ids(conn: Connection, natural_ids: Sequence[str]) -> list[dict[str, Any]]:
    """The items with these natural ids (``episode:...``, ``item:...``), any ``served`` (AP-02,
    a search hit's id to its row)."""
    keys = _keys(natural_ids, "natural ids")
    if not keys:
        return []
    return _rows(conn, select(*_ITEM).where(content_items.c.natural_id.in_(keys)))


def items_by_feed_episode(
    conn: Connection, pairs: Sequence[Tuple[str, str]]
) -> list[dict[str, Any]]:
    """The items of these ``(raw feed id, external item id)`` pairs, any ``served`` (AP-02); each
    row also carries ``raw_feed_id``."""
    keys = _keys([tuple(p) for p in pairs], "pairs")
    if not keys:
        return []
    statement = (
        select(*_ITEM, sources.c.raw_feed_id)
        .join(sources, sources.c.source_id == content_items.c.source_id)
        .where(tuple_(sources.c.raw_feed_id, content_items.c.external_item_id).in_(keys))
    )
    return _rows(conn, statement)


def cards(conn: Connection, item_ids: Sequence[int]) -> list[dict[str, Any]]:
    """The ``item_cards`` rows of these items; an item with no card is left out (AP-23)."""
    keys = _keys(item_ids, "item ids")
    if not keys:
        return []
    return _rows(conn, select(*_CARD).where(item_cards.c.item_id.in_(keys)))


def _after(cursor: Optional[Cursor]) -> Any:
    if cursor is None:
        return None
    published_at, item_id = cursor
    published = content_items.c.published_at
    if published_at is None:
        return and_(published.is_(None), content_items.c.item_id < item_id)
    return or_(
        published < published_at,
        and_(published == published_at, content_items.c.item_id < item_id),
        published.is_(None),
    )


def newest_items(
    conn: Connection,
    *,
    source_ids: Optional[Sequence[int]] = None,
    content_type: Optional[str] = None,
    after: Optional[Cursor] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Served items newest first (dated before undated, then item id descending), of these
    sources or of this content type when given, after ``after`` (AP-03, AP-04)."""
    if not 1 <= limit <= MAX_PAGE:
        raise ValueError(f"limit {limit} is not between 1 and {MAX_PAGE}")
    conditions = [content_items.c.served > 0]
    if source_ids is not None:
        conditions.append(content_items.c.source_id.in_(_keys(source_ids, "source ids")))
    if content_type is not None:
        conditions.append(content_items.c.content_type == content_type)
    cursor = _after(after)
    if cursor is not None:
        conditions.append(cursor)
    statement = (
        select(*_ITEM)
        .where(and_(*conditions))
        .order_by(content_items.c.published_at.desc().nulls_last(), content_items.c.item_id.desc())
        .limit(limit)
    )
    return _rows(conn, statement)


def cursor_of(row: dict[str, Any]) -> Cursor:
    """The cursor that continues a list after ``row``."""
    return (row["published_at"], row["item_id"])

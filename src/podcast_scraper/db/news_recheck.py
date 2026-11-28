"""The news re-check's rows (CRFC-006 Part B, G7; tracker n-w9, corpus_0015).

``news_recheck_sources`` holds a source's window in days and ``news_recheck`` one item's state
(``news.recheck.RecheckState``) with the validators its page last sent. A source with no window row
has a window of 0: :func:`enrol` and :func:`due` never return its items, so the job does nothing
until the operator sets a window (W9 ships dark; W12 switches each source on).

The schedule is the window's share of ``recheck.DEFAULT_DAYS`` (a 7-day window checks on days 1, 2,
4 and 7; a 3-day one on 1 and 2), read from the source's window on every call, so a window set back
to 0 stops its items at once. The job (W9's next slice) fetches and compares; these functions only
read and write the state, in the caller's transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

from sqlalchemy import func, select, true, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from podcast_scraper.db.corpus_keys import NEWS_ARTICLE
from podcast_scraper.db.schema import (
    content_items,
    documents,
    news_recheck,
    news_recheck_sources,
    sources,
)
from podcast_scraper.news import recheck
from podcast_scraper.news.recheck_fetch import Validators

_STATE = news_recheck.c
_WINDOW = news_recheck_sources.c
_ITEM = content_items.c


def schedule(window_days: int) -> tuple[int, ...]:
    """The re-check days inside a window of ``window_days``; empty for 0."""
    return tuple(day for day in recheck.DEFAULT_DAYS if day <= window_days)


def set_window(conn: Connection, source_id: int, window_days: int) -> None:
    """Set a source's re-check window (0 switches it off)."""
    statement = insert(news_recheck_sources).values(source_id=source_id, window_days=window_days)
    conn.execute(
        statement.on_conflict_do_update(
            index_elements=[_WINDOW.source_id],
            set_={"window_days": window_days, "updated_at": func.now()},
        )
    )


def enrol(conn: Connection, now: datetime) -> int:
    """Give a state row to every served news item of a source with a window, that has none yet
    and whose first fetch (its earliest ``article`` document's ``fetched_at``) is inside the
    window; returns how many. An item with no fetch time is left out: its window cannot be
    placed."""
    first = (
        select(documents.c.item_id, func.min(documents.c.fetched_at).label("first_fetched_at"))
        .where(documents.c.kind == "article", documents.c.fetched_at.is_not(None))
        .group_by(documents.c.item_id)
        .subquery()
    )
    rows = conn.execute(
        select(_ITEM.item_id, first.c.first_fetched_at, _WINDOW.window_days)
        .join(first, first.c.item_id == _ITEM.item_id)
        .join(news_recheck_sources, _WINDOW.source_id == _ITEM.source_id)
        .outerjoin(news_recheck, _STATE.item_id == _ITEM.item_id)
        .where(
            _ITEM.content_type == NEWS_ARTICLE,
            _ITEM.served > 0,
            _WINDOW.window_days > 0,
            _STATE.item_id.is_(None),
            first.c.first_fetched_at + func.make_interval(0, 0, 0, _WINDOW.window_days) >= now,
        )
    ).all()
    values = []
    for item_id, first_fetched_at, window_days in rows:
        state = recheck.start(first_fetched_at, schedule(window_days))
        values.append(
            dict(
                item_id=item_id,
                first_fetched_at=first_fetched_at,
                next_due_at=state.next_due,
                done=state.done,
            )
        )
    if values:
        conn.execute(insert(news_recheck).values(values).on_conflict_do_nothing())
    return len(values)


@dataclass(frozen=True)
class Due:
    """An item due for a check: its state on its source's current schedule, and the validators
    to send."""

    item_id: int
    source_id: int
    natural_id: str
    source_natural_id: str
    state: recheck.RecheckState
    validators: Validators


def due(
    conn: Connection,
    now: datetime,
    limit: int = 500,
    *,
    source_natural_ids: Optional[Sequence[str]] = None,
) -> list[Due]:
    """The served items of a source with a window whose next check is due at ``now``, oldest due
    first, at most ``limit``; only those of ``source_natural_ids`` when it is given."""
    only = (
        sources.c.natural_id.in_(list(source_natural_ids))
        if source_natural_ids is not None
        else true()
    )
    rows = conn.execute(
        select(
            news_recheck,
            _ITEM.source_id,
            _ITEM.natural_id,
            sources.c.natural_id.label("source_natural_id"),
            _WINDOW.window_days,
        )
        .join(content_items, _ITEM.item_id == _STATE.item_id)
        .join(sources, sources.c.source_id == _ITEM.source_id)
        .join(news_recheck_sources, _WINDOW.source_id == _ITEM.source_id)
        .where(
            _STATE.done.is_(False),
            _STATE.next_due_at <= now,
            _ITEM.served > 0,
            _WINDOW.window_days > 0,
            only,
        )
        .order_by(_STATE.next_due_at, _STATE.item_id)
        .limit(limit)
    ).mappings()
    return [
        Due(
            item_id=row["item_id"],
            source_id=row["source_id"],
            natural_id=row["natural_id"],
            source_natural_id=row["source_natural_id"],
            state=recheck.RecheckState(
                first_fetched_at=row["first_fetched_at"],
                checks_done=row["checks_done"],
                next_due=row["next_due_at"],
                last_status=row["last_status"],
                gone_in_a_row=row["gone_in_a_row"],
                done=row["done"],
                days=schedule(row["window_days"]),
            ),
            validators=Validators(etag=row["etag"], last_modified=row["last_modified"]),
        )
        for row in rows
    ]


def record(
    conn: Connection,
    item_id: int,
    state: recheck.RecheckState,
    validators: Validators,
    checked_at: datetime,
) -> None:
    """Store an item's state after a check (``recheck.advance``'s) and the validators to send
    next."""
    conn.execute(
        update(news_recheck)
        .where(_STATE.item_id == item_id)
        .values(
            checks_done=state.checks_done,
            next_due_at=None if state.done else state.next_due,
            last_status=state.last_status,
            gone_in_a_row=state.gone_in_a_row,
            done=state.done,
            etag=validators.etag,
            last_modified=validators.last_modified,
            last_checked_at=checked_at,
            updated_at=func.now(),
        )
    )


#: ``documents.change_kind`` of a withdrawal (CRFC-006 B.1).
WITHDRAW = "withdraw"


def withdraw(conn: Connection, item_id: int) -> bool:
    """Withdraw an item whose page answered gone on ``recheck.GONE_CHECKS`` checks in a row
    (CRFC-006 B.1, B.3), in the caller's transaction; returns whether it was served. Each document
    series of the item gets a ``withdraw`` row with no body that supersedes its newest version,
    so nothing of it is served; the item is ``served`` 0; its ``item_entities``,
    ``insight_mentions`` and card are deleted and the card's flags cleared; ``corpus_revision`` is
    bumped last (DECISIONS rule 7). Insights, evidence and segments are kept, never served (B.3).
    A second call writes nothing."""
    from podcast_scraper.db.corpus_cards import FLAGS
    from podcast_scraper.db.schema import insight_mentions, item_cards, item_entities
    from podcast_scraper.upgrade.rows import bump_corpus_revision

    served = conn.execute(
        select(_ITEM.served).where(_ITEM.item_id == item_id).with_for_update()
    ).scalar_one_or_none()
    if not served:
        return False
    d, newer = documents.c, documents.alias("newer")
    newest = conn.execute(
        select(d.document_id, d.kind, d.variant, d.version).where(
            d.item_id == item_id,
            ~select(newer.c.document_id).where(newer.c.supersedes_id == d.document_id).exists(),
        )
    ).all()
    for row in newest:
        conn.execute(
            insert(documents).values(
                item_id=item_id,
                kind=row.kind,
                variant=row.variant,
                version=row.version + 1,
                supersedes_id=row.document_id,
                change_kind=WITHDRAW,
            )
        )
    conn.execute(
        update(content_items)
        .where(_ITEM.item_id == item_id)
        .values(served=0, updated_at=func.now(), **{flag: False for flag in FLAGS})
    )
    conn.execute(item_entities.delete().where(item_entities.c.item_id == item_id))
    conn.execute(insight_mentions.delete().where(insight_mentions.c.item_id == item_id))
    conn.execute(item_cards.delete().where(item_cards.c.item_id == item_id))
    bump_corpus_revision(conn)
    return True


__all__ = ["Due", "due", "enrol", "record", "schedule", "set_window", "withdraw", "WITHDRAW"]

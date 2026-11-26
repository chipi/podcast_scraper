"""An insight's attribution, ``STATED_BY``, kept in ``insight_attribution`` (CRFC-004 §1).

The rule is ``GI_4.0.md``'s: take the insight's supporting quotes, count each quote's speaker when
it is a person, ignore a quote with none, and attribute the insight to the person holding a strict
majority of the counted quotes. A tie, or no counted quote, gives no row.

A speaker is counted by its ``canonical_id``, so two ids the published identity build joined are
one person (the rule's point 4). The derivation reads only stored rows, so it runs in the caller's
transaction: per item from every evidence writer for the items it touched, and for the whole corpus
when an identity build is published (:func:`derive_attribution` with ``item_ids`` ``None``).
"""

from __future__ import annotations

from typing import Optional, Sequence

from sqlalchemy import and_, delete, func, insert, literal_column, select
from sqlalchemy.engine import Connection

from podcast_scraper.db.schema import (
    entities,
    evidence,
    insight_attribution,
    insight_support,
    insights,
)

PERSON = "person"


def derive_attribution(conn: Connection, item_ids: Optional[Sequence[int]] = None) -> int:
    """Rewrite the attribution of ``item_ids``' insights, or of every insight when ``None``.

    Returns the number of rows written.
    """
    if item_ids is not None and not item_ids:
        return 0
    wipe = delete(insight_attribution)
    if item_ids is not None:
        wipe = wipe.where(insight_attribution.c.item_id.in_(list(item_ids)))
    conn.execute(wipe)

    voices = (
        select(
            insights.c.insight_id,
            insights.c.item_id,
            entities.c.canonical_id.label("person_id"),
            func.count().label("quotes"),
        )
        .select_from(
            insight_support.join(insights, insights.c.insight_id == insight_support.c.insight_id)
            .join(evidence, evidence.c.evidence_id == insight_support.c.evidence_id)
            .join(
                entities,
                and_(entities.c.entity_id == evidence.c.speaker_id, entities.c.type == PERSON),
            )
        )
        .group_by(insights.c.insight_id, insights.c.item_id, entities.c.canonical_id)
    )
    if item_ids is not None:
        voices = voices.where(insights.c.item_id.in_(list(item_ids)))
    counted = voices.subquery("voices")
    totals = select(
        counted.c.insight_id,
        counted.c.item_id,
        counted.c.person_id,
        counted.c.quotes,
        func.sum(counted.c.quotes).over(partition_by=counted.c.insight_id).label("counted"),
    ).subquery("totals")
    majority = select(totals.c.insight_id, totals.c.person_id, totals.c.item_id).where(
        totals.c.quotes * literal_column("2") > totals.c.counted
    )
    written = conn.execute(
        insert(insight_attribution)
        .from_select(["insight_id", "person_id", "item_id"], majority)
        .returning(insight_attribution.c.insight_id)
    )
    return len(written.all())

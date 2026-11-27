"""The projections job (CRFC-005 §5, tracker r-projections): what is computed once per batch.

One build after a batch of writes (the nightly, a migration, the cut-over), in one transaction
under the leases ``corpus`` shared and ``build:projections`` exclusive (w-leases), with
``work_mem`` raised for that transaction only (§5: run 3 measured the builds at 64 MB). The
build is published in the same transaction, so a read during a rebuild sees the previous build.
Its id is the graph's ``graph_build_id``: preset caches, ETags and a client's stale mark key on
it (§4), not on ``corpus_revision``, which moves hundreds of times a night.

Every projection counts what the graph serves (``db.graph``): shown items (served, not
withheld), and insights that are visible, current and on a shown item. Entities count under
their canonical id, every member's rows included (§3.3).

- :data:`~podcast_scraper.db.schema.entity_item_counts`: per canonical entity, its distinct
  items and insights.
- :data:`~podcast_scraper.db.schema.entity_weeks`: the same per UTC ISO week (its Monday), for
  trending and timelines; an item with no publish time has no week.
- :data:`~podcast_scraper.db.schema.cluster_reach`: per group of the published cluster build,
  the distinct items its members reach (AP-34). A theme or storyline reaches its member topics'
  items, an insight cluster its insights' items, a super-theme (no member rows of its own) its
  child storylines' members' items.

The published build and the one before it are kept, so undo can publish the previous one; older
builds are deleted with their rows.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    and_,
    BigInteger,
    cast,
    Date,
    delete,
    distinct,
    exists,
    func,
    insert,
    literal,
    null,
    select,
    union_all,
    update,
)
from sqlalchemy.engine import Connection, Engine

from podcast_scraper.db import leases
from podcast_scraper.db.schema import (
    cluster_builds,
    cluster_members,
    cluster_reach,
    clusters,
    content_items,
    corpus_revision,
    documents,
    entities,
    entity_item_counts,
    entity_weeks,
    identity_builds,
    insight_mentions,
    insights,
    item_entities,
    projection_builds,
)

#: The lease resource of this job (w-leases: a corpus-wide build takes ``corpus`` shared, then
#: its ``build:<kind>`` exclusive).
RESOURCE = leases.build_resource("projections")
#: §5: run 3 measured the builds with 64 MB, not the plane's 16 MB.
WORK_MEM = "64MB"
#: Builds kept: the published one and the one before it.
KEEP = 2


@dataclass(frozen=True)
class Build:
    """A published projections build and what it wrote."""

    build_id: int
    stats: dict[str, int]


def graph_build_id(conn: Connection) -> Optional[int]:
    """The published projections build: the key the graph's caches and ETags carry."""
    found: Optional[int] = conn.execute(
        select(projection_builds.c.build_id).where(projection_builds.c.state == "published")
    ).scalar_one_or_none()
    return found


def _shown(item_id: Any) -> Any:
    c = content_items.alias()
    return exists().where(c.c.item_id == item_id, c.c.served > 0, ~c.c.withheld)


def _insight_ok(i: Any) -> Any:
    d = documents.alias()
    return and_(
        i.c.visible,
        ~exists().where(d.c.item_id == i.c.item_id, d.c.supersedes_id == i.c.document_id),
        _shown(i.c.item_id),
    )


def _written(conn: Connection, statement: Any) -> int:
    """Run an ``INSERT ... SELECT`` and count its rows (the driver reports no rowcount for it)."""
    return len(conn.execute(statement.returning(literal(1))).all())


def _week(at: Any) -> Any:
    return cast(func.date_trunc("week", func.timezone("UTC", at)), Date)


def _reached() -> Any:
    """``(canonical entity, item, insight, week)``: one row per shown item an entity is on and
    per visible current insight that mentions it, under the entity's canonical id."""
    e = entities.alias("e")
    ie = item_entities.alias("ie")
    m = insight_mentions.alias("m")
    i = insights.alias("i")
    c = content_items.alias("c")
    on_items = (
        select(
            e.c.canonical_id.label("entity_id"),
            ie.c.item_id.label("item_id"),
            cast(null(), BigInteger).label("insight_id"),
            _week(ie.c.published_at).label("week"),
        )
        .select_from(ie.join(e, e.c.entity_id == ie.c.entity_id))
        .where(_shown(ie.c.item_id))
    )
    in_insights = (
        select(
            e.c.canonical_id.label("entity_id"),
            cast(null(), BigInteger).label("item_id"),
            m.c.insight_id.label("insight_id"),
            _week(c.c.published_at).label("week"),
        )
        .select_from(
            m.join(e, e.c.entity_id == m.c.entity_id)
            .join(i, i.c.insight_id == m.c.insight_id)
            .join(c, c.c.item_id == i.c.item_id)
        )
        .where(_insight_ok(i))
    )
    return union_all(on_items, in_insights).subquery("r")


def _write_counts(conn: Connection, build_id: int) -> int:
    r = _reached()
    statement = insert(entity_item_counts).from_select(
        ["build_id", "entity_id", "items", "insights"],
        select(
            literal(build_id, BigInteger),
            r.c.entity_id,
            func.count(distinct(r.c.item_id)),
            func.count(distinct(r.c.insight_id)),
        ).group_by(r.c.entity_id),
    )
    return _written(conn, statement)


def _write_weeks(conn: Connection, build_id: int) -> int:
    r = _reached()
    k = entities.alias("k")
    statement = insert(entity_weeks).from_select(
        ["build_id", "kind", "entity_id", "week", "items", "insights"],
        select(
            literal(build_id, BigInteger),
            k.c.type,
            r.c.entity_id,
            r.c.week,
            func.count(distinct(r.c.item_id)),
            func.count(distinct(r.c.insight_id)),
        )
        .select_from(r.join(k, k.c.entity_id == r.c.entity_id))
        .where(r.c.week.is_not(None))
        .group_by(k.c.type, r.c.entity_id, r.c.week),
    )
    return _written(conn, statement)


def _cluster_items(cluster_build_id: int) -> Any:
    """``(cluster, item)``: each shown item a group of this cluster build reaches."""
    g = clusters.alias("g")
    cm = cluster_members.alias("cm")
    e = entities.alias("e")
    em = entities.alias("em")
    ie = item_entities.alias("ie")
    i = insights.alias("i")
    child = clusters.alias("child")

    topics = (
        select(g.c.cluster_id.label("cluster_id"), ie.c.item_id.label("item_id"))
        .select_from(
            g.join(cm, and_(cm.c.build_id == g.c.build_id, cm.c.cluster_id == g.c.cluster_id))
            .join(e, e.c.entity_id == cm.c.entity_id)
            .join(em, em.c.canonical_id == e.c.canonical_id)
            .join(ie, ie.c.entity_id == em.c.entity_id)
        )
        .where(g.c.build_id == cluster_build_id, _shown(ie.c.item_id))
    )
    of_insights = (
        select(g.c.cluster_id.label("cluster_id"), i.c.item_id.label("item_id"))
        .select_from(
            g.join(cm, and_(cm.c.build_id == g.c.build_id, cm.c.cluster_id == g.c.cluster_id)).join(
                i, i.c.insight_id == cm.c.insight_id
            )
        )
        .where(g.c.build_id == cluster_build_id, _insight_ok(i))
    )
    super_themes = (
        select(g.c.cluster_id.label("cluster_id"), ie.c.item_id.label("item_id"))
        .select_from(
            g.join(
                child,
                and_(child.c.build_id == g.c.build_id, child.c.parent_id == g.c.cluster_id),
            )
            .join(
                cm, and_(cm.c.build_id == child.c.build_id, cm.c.cluster_id == child.c.cluster_id)
            )
            .join(e, e.c.entity_id == cm.c.entity_id)
            .join(em, em.c.canonical_id == e.c.canonical_id)
            .join(ie, ie.c.entity_id == em.c.entity_id)
        )
        .where(g.c.build_id == cluster_build_id, g.c.kind == "super_theme", _shown(ie.c.item_id))
    )
    return union_all(topics, of_insights, super_themes).subquery("ci")


def _write_reach(conn: Connection, build_id: int, cluster_build_id: Optional[int]) -> int:
    if cluster_build_id is None:
        return 0
    ci = _cluster_items(cluster_build_id)
    reached = (
        select(ci.c.cluster_id, func.count(distinct(ci.c.item_id)).label("reach"))
        .group_by(ci.c.cluster_id)
        .subquery("reached")
    )
    # Every group of the build gets a row, 0 when its members reach no shown item, so a reader
    # tells a group that reaches nothing from one this build did not compute.
    statement = insert(cluster_reach).from_select(
        ["build_id", "cluster_build_id", "cluster_id", "reach"],
        select(
            literal(build_id, BigInteger),
            literal(cluster_build_id, BigInteger),
            clusters.c.cluster_id,
            func.coalesce(reached.c.reach, 0),
        )
        .select_from(clusters.outerjoin(reached, reached.c.cluster_id == clusters.c.cluster_id))
        .where(clusters.c.build_id == cluster_build_id),
    )
    return _written(conn, statement)


def _published(conn: Connection, table: Any) -> Optional[int]:
    found: Optional[int] = conn.execute(
        select(table.c.build_id).where(table.c.state == "published")
    ).scalar_one_or_none()
    return found


def _publish(conn: Connection, build_id: int, now: datetime) -> list[int]:
    """Publish ``build_id``, retire the previous published build, delete every build but these
    two; the build ids deleted."""
    conn.execute(
        update(projection_builds)
        .where(projection_builds.c.state == "published")
        .values(state="retired")
    )
    conn.execute(
        update(projection_builds)
        .where(projection_builds.c.build_id == build_id)
        .values(state="published", published_at=now)
    )
    kept = (
        select(projection_builds.c.build_id)
        .order_by(projection_builds.c.build_id.desc())
        .limit(KEEP)
        .subquery("kept")
    )
    gone = conn.execute(
        delete(projection_builds)
        .where(projection_builds.c.build_id.not_in(select(kept.c.build_id)))
        .returning(projection_builds.c.build_id)
    )
    return sorted(int(row[0]) for row in gone)


def build_in(conn: Connection, *, now: Optional[datetime] = None) -> Build:
    """Build and publish one projections build in the caller's transaction."""
    at = now or datetime.now(timezone.utc)
    conn.execute(select(func.set_config("work_mem", WORK_MEM, True)))
    clusters_of = _published(conn, cluster_builds)
    inputs = {
        "identity_build_id": _published(conn, identity_builds),
        "cluster_build_id": clusters_of,
        "corpus_revision": conn.execute(select(corpus_revision.c.revision)).scalar(),
    }
    build_id = int(
        conn.execute(
            insert(projection_builds)
            .values(inputs=inputs, state="built", built_at=at)
            .returning(projection_builds.c.build_id)
        ).scalar_one()
    )
    stats = {
        "entity_item_counts": _write_counts(conn, build_id),
        "entity_weeks": _write_weeks(conn, build_id),
        "cluster_reach": _write_reach(conn, build_id, clusters_of),
    }
    conn.execute(
        update(projection_builds)
        .where(projection_builds.c.build_id == build_id)
        .values(stats=stats)
    )
    stats["builds_deleted"] = len(_publish(conn, build_id, at))
    return Build(build_id=build_id, stats=stats)


def build(engine: Engine, *, now: Optional[datetime] = None) -> Build:
    """Take the leases, build and publish one projections build, release the leases. Raises
    :class:`~podcast_scraper.db.leases.LeaseRefused` when another build or a corpus-exclusive
    job holds them: a refused build is not deferred, the next batch builds again."""
    holder = f"projections:{uuid.uuid4().hex[:12]}"
    taken = leases.acquire(
        engine,
        [
            leases.LeaseRequest(leases.CORPUS, leases.SHARED),
            leases.LeaseRequest(RESOURCE, leases.EXCLUSIVE),
        ],
        holder,
    )
    try:
        with engine.begin() as conn:
            leases.heartbeat(conn, holder, taken)
            return build_in(conn, now=now)
    finally:
        leases.release(engine, holder)


__all__ = ["Build", "build", "build_in", "graph_build_id", "KEEP", "RESOURCE", "WORK_MEM"]

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
- :data:`~podcast_scraper.db.schema.person_rankings` (``GET /api/corpus/persons/top``, the MCP
  ``top_people``; Common's ``_rank_all_persons`` on files): per canonical person, the distinct
  insights a quote the person spoke supports (Insight SUPPORTED_BY Quote SPOKEN_BY Person), the
  items they are on, and the rank (insights descending, then the canonical natural id). A
  speaker whose own id is an unresolved diarization voice
  (``graph_id_utils.is_unresolved_speaker_placeholder``, the file code's test, on the same id)
  never counts; a person with no such insight has no row. No top topics until
  ``insight_mentions.confidence`` is filled (corpus_0016), and no ``total_persons``. The file
  code counts SPOKEN_BY only, while ``evidence.speaker_id`` also holds an ATTRIBUTED_TO target
  for a quote with no SPOKEN_BY; snapshot-prod-20261127 has no ATTRIBUTED_TO edge in any of its
  4,125 episodes' GI, so the two agree there. Should one appear, ``evidence`` needs the edge it
  came from.
- :data:`~podcast_scraper.db.schema.topic_relations`, kind ``shares_insight``
  (``GET /relational/related-topics``, the MCP ``related_topics``; ``relational_queries.
  related_topics`` on files): per ordered pair of canonical topics, the distinct current insights
  on shown items ABOUT both, in both directions; each topic keeps its :data:`RELATED_PER_TOPIC`
  strongest, by weight and then the other topic's natural id, as the route ranks them. Each
  direction is cut by its own topic's list, so ``(a, b)`` can be stored without ``(b, a)``: read
  one topic's related topics by ``topic_a``, never the table as symmetric. Not shown-item
  co-occurrence: ``topic_cooccurrence_corpus``'s ``entity_pairs`` give that.
- :data:`~podcast_scraper.db.schema.topic_speakers` (one topic's perspectives, unscoped;
  ``cil_queries.topics_perspectives`` on files): per canonical topic and canonical speaker, the
  distinct current insights on shown items ABOUT the topic that the speaker is attributed, and
  their items, ranked by insights and then the speaker's natural id. An insight's speaker is its
  first supporting quote's (the lowest ``insight_support.ord``), as the file code takes the first
  SUPPORTED_BY edge; a first quote with no speaker attributes it to no one. A canonical speaker
  that is an unresolved voice (the file code's predicate, on its id and label) has no row. What
  it cannot serve stays per request: ``scope=mine``, the insight-type filter, and a grouping's
  union of topics.

The published build and the one before it are kept, so undo can publish the previous one; older
builds are deleted with their rows.
"""

from __future__ import annotations

import collections
import logging
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
from sqlalchemy.dialects.postgresql import distinct_on
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
    evidence,
    identity_builds,
    insight_mentions,
    insight_support,
    insights,
    item_entities,
    person_rankings,
    projection_builds,
    topic_relations,
    topic_speakers,
)

logger = logging.getLogger(__name__)

#: How each :func:`after_batch` ended, since the process started.
OUTCOMES: collections.Counter[str] = collections.Counter()

#: The lease resource of this job (w-leases: a corpus-wide build takes ``corpus`` shared, then
#: its ``build:<kind>`` exclusive).
RESOURCE = leases.build_resource("projections")
#: §5: run 3 measured the builds with 64 MB, not the plane's 16 MB.
WORK_MEM = "64MB"
#: Builds kept: the published one and the one before it.
KEEP = 2
#: The related topics kept per topic: the related-topics route's ``k`` maximum.
RELATED_PER_TOPIC = 200
#: ``topic_relations.kind`` for two topics ABOUT the same insight.
SHARES_INSIGHT = "shares_insight"


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
    """Run an ``INSERT ... SELECT`` and count its rows in the same statement (the driver reports
    no rowcount for it): ``WITH w AS (INSERT ... RETURNING 1) SELECT count(*) FROM w``."""
    w = statement.returning(literal(1)).cte("w")
    return int(conn.execute(select(func.count()).select_from(w)).scalar_one())


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


def _placeholder_speakers(conn: Connection) -> list[int]:
    """The person entities that speak a quote and are unresolved diarization voices. The test is
    the file code's own predicate in Python, not a copy of its patterns in SQL, so the two cannot
    drift; it reads one row per distinct speaker."""
    from podcast_scraper.graph_id_utils import is_unresolved_speaker_placeholder

    e = entities.alias("ps")
    rows = conn.execute(
        select(e.c.entity_id, e.c.natural_id)
        .where(
            e.c.type == "person",
            exists().where(evidence.c.speaker_id == e.c.entity_id),
        )
        .order_by(e.c.entity_id)
    )
    return [int(r.entity_id) for r in rows if is_unresolved_speaker_placeholder(r.natural_id)]


def _write_person_rankings(conn: Connection, build_id: int) -> int:
    sp = entities.alias("sp")
    k = entities.alias("pk")
    s = insight_support.alias("s")
    ev = evidence.alias("ev")
    i = insights.alias("i")
    placeholders = _placeholder_speakers(conn)
    insight_count = func.count(distinct(i.c.insight_id))
    ranked = (
        select(
            k.c.entity_id.label("entity_id"),
            func.count(distinct(i.c.item_id)).label("items"),
            insight_count.label("insights"),
            func.row_number()
            .over(order_by=(insight_count.desc(), k.c.natural_id.asc()))
            .label("rank"),
        )
        .select_from(
            s.join(ev, ev.c.evidence_id == s.c.evidence_id)
            .join(sp, sp.c.entity_id == ev.c.speaker_id)
            .join(k, k.c.entity_id == sp.c.canonical_id)
            .join(i, i.c.insight_id == s.c.insight_id)
        )
        .where(
            sp.c.type == "person",
            sp.c.entity_id.not_in(placeholders) if placeholders else literal(True),
            _insight_ok(i),
        )
        .group_by(k.c.entity_id, k.c.natural_id)
        .subquery("ranked")
    )
    statement = insert(person_rankings).from_select(
        ["build_id", "entity_id", "items", "insights", "rank"],
        select(
            literal(build_id, BigInteger),
            ranked.c.entity_id,
            ranked.c["items"],
            ranked.c.insights,
            ranked.c.rank,
        ),
    )
    return _written(conn, statement)


def _write_topic_relations(conn: Connection, build_id: int) -> int:
    e = entities.alias("te")
    m = insight_mentions.alias("tm")
    i = insights.alias("ti")
    about = (
        select(e.c.canonical_id.label("topic_id"), m.c.insight_id.label("insight_id"))
        .distinct()
        .select_from(
            m.join(e, e.c.entity_id == m.c.entity_id).join(i, i.c.insight_id == m.c.insight_id)
        )
        .where(m.c.kind == "ABOUT", e.c.type == "topic", _insight_ok(i))
        .cte("about")
    )
    x, y = about.alias("x"), about.alias("y")
    other = entities.alias("tb")
    weight = func.count(distinct(x.c.insight_id))
    ranked = (
        select(
            x.c.topic_id.label("topic_a"),
            y.c.topic_id.label("topic_b"),
            weight.label("weight"),
            func.row_number()
            .over(partition_by=x.c.topic_id, order_by=(weight.desc(), other.c.natural_id.asc()))
            .label("n"),
        )
        .select_from(
            x.join(y, and_(y.c.insight_id == x.c.insight_id, y.c.topic_id != x.c.topic_id)).join(
                other, other.c.entity_id == y.c.topic_id
            )
        )
        .group_by(x.c.topic_id, y.c.topic_id, other.c.natural_id)
        .subquery("ranked")
    )
    statement = insert(topic_relations).from_select(
        ["build_id", "topic_a", "topic_b", "kind", "weight"],
        select(
            literal(build_id, BigInteger),
            ranked.c.topic_a,
            ranked.c.topic_b,
            literal(SHARES_INSIGHT),
            ranked.c.weight,
        ).where(ranked.c.n <= RELATED_PER_TOPIC),
    )
    return _written(conn, statement)


def _placeholder_canonical_speakers(conn: Connection) -> list[int]:
    """The canonical persons of quote speakers that are unresolved voices, tested as the
    perspectives code tests them: on the canonical id and its name."""
    from podcast_scraper.graph_id_utils import is_unresolved_speaker_placeholder

    sp, k = entities.alias("cs"), entities.alias("ck")
    rows = conn.execute(
        select(k.c.entity_id, k.c.natural_id, k.c.label)
        .distinct()
        .select_from(
            evidence.join(sp, sp.c.entity_id == evidence.c.speaker_id).join(
                k, k.c.entity_id == sp.c.canonical_id
            )
        )
        .where(sp.c.type == "person")
        .order_by(k.c.entity_id)
    )
    return [
        int(r.entity_id) for r in rows if is_unresolved_speaker_placeholder(r.natural_id, r.label)
    ]


def _write_topic_speakers(conn: Connection, build_id: int) -> int:
    s = insight_support.alias("fs")
    ev = evidence.alias("fe")
    first = (
        select(s.c.insight_id.label("insight_id"), ev.c.speaker_id.label("speaker_id"))
        .ext(distinct_on(s.c.insight_id))
        .select_from(s.join(ev, ev.c.evidence_id == s.c.evidence_id))
        .order_by(s.c.insight_id, s.c.ord)
        .subquery("first_quote")
    )
    e = entities.alias("ae")
    m = insight_mentions.alias("am")
    i = insights.alias("ai")
    about = (
        select(
            e.c.canonical_id.label("topic_id"),
            m.c.insight_id.label("insight_id"),
            i.c.item_id.label("item_id"),
        )
        .distinct()
        .select_from(
            m.join(e, e.c.entity_id == m.c.entity_id).join(i, i.c.insight_id == m.c.insight_id)
        )
        .where(m.c.kind == "ABOUT", e.c.type == "topic", _insight_ok(i))
        .subquery("about")
    )
    sp, k = entities.alias("ss"), entities.alias("sk")
    placeholders = _placeholder_canonical_speakers(conn)
    insight_count = func.count(distinct(about.c.insight_id))
    ranked = (
        select(
            about.c.topic_id.label("topic_id"),
            k.c.entity_id.label("person_id"),
            func.count(distinct(about.c.item_id)).label("items"),
            insight_count.label("insights"),
            func.row_number()
            .over(partition_by=about.c.topic_id, order_by=(insight_count.desc(), k.c.natural_id))
            .label("rank"),
        )
        .select_from(
            about.join(first, first.c.insight_id == about.c.insight_id)
            .join(sp, sp.c.entity_id == first.c.speaker_id)
            .join(k, k.c.entity_id == sp.c.canonical_id)
        )
        .where(
            sp.c.type == "person",
            k.c.entity_id.not_in(placeholders) if placeholders else literal(True),
        )
        .group_by(about.c.topic_id, k.c.entity_id, k.c.natural_id)
        .subquery("ranked")
    )
    statement = insert(topic_speakers).from_select(
        ["build_id", "topic_id", "person_id", "items", "insights", "rank"],
        select(
            literal(build_id, BigInteger),
            ranked.c.topic_id,
            ranked.c.person_id,
            ranked.c["items"],
            ranked.c.insights,
            ranked.c.rank,
        ),
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
        "person_rankings": _write_person_rankings(conn, build_id),
        "topic_relations": _write_topic_relations(conn, build_id),
        "topic_speakers": _write_topic_speakers(conn, build_id),
    }
    stats["builds_deleted"] = len(_publish(conn, build_id, at))
    conn.execute(
        update(projection_builds)
        .where(projection_builds.c.build_id == build_id)
        .values(stats=stats)
    )
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


def after_batch() -> str:
    """The job's call after a batch (``stage_hooks.corpus_indexed``, after every hook): build and
    publish on the pipeline's engine; returns and counts the outcome. ``not_configured`` without a
    pipeline database URL; ``lease_refused`` when another build holds the lease (the next batch
    builds again); ``failed`` is logged. Never raises unless ``PODCAST_CORPUS_WRITE_STRICT`` is
    set, as the row writer (``corpus_writer``)."""
    from podcast_scraper.db import corpus_writer
    from podcast_scraper.db.settings import pipeline_is_configured

    if not pipeline_is_configured():
        outcome = "not_configured"
    else:
        from podcast_scraper.db.engines import pipeline_rw_engine

        try:
            built = build(pipeline_rw_engine())
            logger.info("projections: build %s published %s", built.build_id, built.stats)
            outcome = "published"
        except leases.LeaseRefused:
            outcome = "lease_refused"
        except Exception as exc:  # noqa: BLE001 - a failed build keeps the published one
            logger.warning("projections: build failed: %s", exc)
            OUTCOMES["failed"] += 1
            if corpus_writer._strict():
                raise corpus_writer.SyncFailed(f"projections build failed: {exc}") from exc
            return "failed"
    OUTCOMES[outcome] += 1
    return outcome


__all__ = [
    "after_batch",
    "Build",
    "build",
    "build_in",
    "graph_build_id",
    "KEEP",
    "OUTCOMES",
    "RELATED_PER_TOPIC",
    "RESOURCE",
    "SHARES_INSIGHT",
    "WORK_MEM",
]

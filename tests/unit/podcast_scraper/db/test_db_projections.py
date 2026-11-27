"""The projections job (CRFC-005 §5, r-projections): entity_item_counts, entity_weeks and
cluster_reach count what the graph serves, under canonical ids, and the build is published as the
graph's ``graph_build_id`` with the previous build kept."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pytest
from sqlalchemy import create_engine, insert, select, text
from sqlalchemy.engine import Connection

from podcast_scraper.db import leases, projections
from podcast_scraper.db.schema import (
    cluster_builds,
    cluster_ids,
    cluster_members,
    cluster_reach,
    clusters,
    entities,
    entity_item_counts,
    entity_weeks,
    person_rankings,
    projection_builds,
    topic_relations,
    topic_speakers,
)
from podcast_scraper.db.settings import normalize_url
from tests.unit.podcast_scraper.db.test_corpus_writer import pipeline_db  # noqa: F401
from tests.unit.podcast_scraper.db.test_db_corpus import url  # noqa: F401
from tests.unit.podcast_scraper.db.test_db_graph import _one, conn, Corpus  # noqa: F401

SUNDAY_LATE = datetime(2026, 11, 15, 23, 30, tzinfo=timezone.utc)


def _by_natural(c: Corpus, table: Any, build_id: int) -> dict[str, tuple[int, ...]]:
    names = {v: k for k, v in c.e.items()}
    rows = c.conn.execute(select(table).where(table.c.build_id == build_id)).mappings().all()
    return {names[r["entity_id"]]: (r["items"], r["insights"]) for r in rows}


def _group(c: Corpus, build: int, kind: str, n: str, parent: int | None = None) -> int:
    cluster = _one(c.conn, cluster_ids, kind=kind, natural_id=n)
    c.conn.execute(
        insert(clusters).values(
            build_id=build, cluster_id=cluster, kind=kind, label=n, parent_id=parent
        )
    )
    return cluster


def _member(c: Corpus, build: int, cluster: int, **member: int) -> None:
    c.conn.execute(insert(cluster_members).values(build_id=build, cluster_id=cluster, **member))


def test_counts_are_distinct_shown_items_and_current_insights_under_the_canonical_id(
    conn: Connection,  # noqa: F811
) -> None:
    c = Corpus(conn)
    built = projections.build_in(conn)
    counts = _by_natural(c, entity_item_counts, built.build_id)
    # topic:ai-2 is merged into topic:ai: items 1 and 2 (3 is not served), insights n1 and n3
    # (n4 is on a superseded document, n5 is hidden).
    assert counts == {"topic:ai": (2, 2), "person:ann": (1, 1)}
    assert built.stats["entity_item_counts"] == 2


def test_weeks_are_utc_iso_weeks_whatever_the_session_zone(conn: Connection) -> None:  # noqa: F811
    c = Corpus(conn)
    late, _ = c.item("late", SUNDAY_LATE)
    c.mention(late, "topic:ai")
    conn.execute(text("SET LOCAL TimeZone = 'Europe/Belgrade'"))
    built = projections.build_in(conn)
    names = {v: k for k, v in c.e.items()}
    rows = conn.execute(
        select(entity_weeks).where(entity_weeks.c.build_id == built.build_id)
    ).mappings()
    weeks = {
        (names[r["entity_id"]], r["kind"], r["week"]): (r["items"], r["insights"]) for r in rows
    }
    # In Belgrade the late item is Monday 2026-11-16 00:30; in UTC it is still the week before.
    assert weeks == {
        ("topic:ai", "topic", date(2026, 11, 16)): (2, 2),
        ("topic:ai", "topic", date(2026, 11, 9)): (1, 0),
        ("person:ann", "person", date(2026, 11, 16)): (1, 1),
    }


def test_reach_per_group_kind_and_zero_for_a_group_that_reaches_nothing(
    conn: Connection,  # noqa: F811
) -> None:
    c = Corpus(conn)
    build = _one(conn, cluster_builds, rule_version="test/1", state="published")
    storyline = _group(c, build, "storyline", "thc:s")
    _member(c, build, storyline, entity_id=c.e["topic:ai-2"])
    sup = _group(c, build, "super_theme", "sth:x")
    conn.execute(
        clusters.update()
        .where(clusters.c.build_id == build, clusters.c.cluster_id == storyline)
        .values(parent_id=sup)
    )
    insight_cluster = _group(c, build, "insight", "ic:i")
    _member(c, build, insight_cluster, insight_id=c.n3)
    _member(c, build, insight_cluster, insight_id=c.n4)
    quiet = _group(c, build, "theme", "tc:quiet")
    _member(c, build, quiet, entity_id=c.e["topic:other"])
    built = projections.build_in(conn)
    reach = dict(
        conn.execute(
            select(cluster_reach.c.cluster_id, cluster_reach.c.reach).where(
                cluster_reach.c.build_id == built.build_id,
                cluster_reach.c.cluster_build_id == build,
            )
        ).all()
    )
    # The storyline's member topic:ai-2 reaches every item of its canonical topic:ai (1 and 2),
    # the super-theme reaches them through its storyline, the insight cluster reaches item 2
    # through n3 only (n4 is on a superseded document), and the quiet theme reaches nothing.
    assert reach == {storyline: 2, sup: 2, insight_cluster: 1, quiet: 0}


def test_a_build_is_published_as_graph_build_id_and_keeps_the_one_before(
    conn: Connection,  # noqa: F811
) -> None:
    Corpus(conn)
    assert projections.graph_build_id(conn) is None
    first = projections.build_in(conn).build_id
    second = projections.build_in(conn).build_id
    assert projections.graph_build_id(conn) == second
    third = projections.build_in(conn)
    states = dict(
        conn.execute(select(projection_builds.c.build_id, projection_builds.c.state)).all()
    )
    assert states == {second: "retired", third.build_id: "published"}
    assert third.stats["builds_deleted"] == 1
    gone = conn.execute(
        select(entity_item_counts).where(entity_item_counts.c.build_id == first)
    ).first()
    assert gone is None
    inputs = conn.execute(
        select(projection_builds.c.inputs).where(projection_builds.c.build_id == third.build_id)
    ).scalar_one()
    assert set(inputs) == {"identity_build_id", "cluster_build_id", "corpus_revision"}
    stored = conn.execute(
        select(projection_builds.c.stats).where(projection_builds.c.build_id == third.build_id)
    ).scalar_one()
    assert stored == third.stats


def test_after_batch_builds_on_the_pipeline_engine_and_is_refused_while_another_build_runs(
    pipeline_db: None, url: str, conn: Connection  # noqa: F811
) -> None:
    engine = create_engine(normalize_url(url))
    leases.acquire(engine, [leases.LeaseRequest(projections.RESOURCE, leases.EXCLUSIVE)], "o:t")
    assert projections.after_batch() == "lease_refused"
    leases.release(engine, "o:t")
    engine.dispose()
    assert projections.after_batch() == "published"
    assert projections.graph_build_id(conn) is not None
    assert conn.execute(text("SELECT count(*) FROM corpus.job_leases")).scalar_one() == 0


def test_after_batch_counts_its_outcome_and_raises_only_when_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from podcast_scraper.db import corpus_writer, settings

    projections.OUTCOMES.clear()
    monkeypatch.setattr(settings, "pipeline_is_configured", lambda: False)
    assert projections.after_batch() == "not_configured"
    monkeypatch.setattr(settings, "pipeline_is_configured", lambda: True)
    monkeypatch.setattr("podcast_scraper.db.engines.pipeline_rw_engine", lambda: object())

    def refused(_engine: Any) -> None:
        raise leases.LeaseRefused("held")

    def broken(_engine: Any) -> None:
        raise RuntimeError("database down")

    monkeypatch.setattr(projections, "build", refused)
    assert projections.after_batch() == "lease_refused"
    monkeypatch.setattr(projections, "build", broken)
    monkeypatch.delenv(corpus_writer.ENV_STRICT, raising=False)
    assert projections.after_batch() == "failed"
    monkeypatch.setenv(corpus_writer.ENV_STRICT, "1")
    with pytest.raises(corpus_writer.SyncFailed):
        projections.after_batch()
    assert projections.OUTCOMES == {"not_configured": 1, "lease_refused": 1, "failed": 2}


def test_person_rankings_count_the_insights_a_persons_quotes_support_under_the_canonical(
    conn: Connection,  # noqa: F811
) -> None:
    # Common's _rank_all_persons on rows (database, 2026-11-27): Insight SUPPORTED_BY Quote
    # SPOKEN_BY Person, distinct current insights on shown items; rank by insights, then the
    # canonical natural id; an unresolved voice never ranks.
    c = Corpus(conn)
    bob_2 = _one(
        conn,
        entities,
        natural_id="person:bob-2",
        type="person",
        label="Bob",
        normalized_label="bob",
        canonical_id=c.e["person:bob"],
    )
    c.e["person:bob-2"] = bob_2
    c.entity("person:speaker-03")
    c.entity("person:carl")
    d2 = c.items["2"]
    # The fixture's q1: bob's quote supports n1 (item 1). bob-2 is bob: n3 on item 2.
    c.quote(d2, "person:bob-2", c.n3, "b2")
    # ann: n1 and n3, so 2 and 2 like bob; the tie goes to the natural id, person:ann first.
    c.quote(c.items["1"], "person:ann", c.n1, "a1")
    c.quote(d2, "person:ann", c.n3, "a3")
    # A diarization voice ranks for nothing.
    c.quote(d2, "person:speaker-03", c.n3, "s3")
    # carl supports only insights the graph does not serve: on a superseded document, hidden.
    c.quote(c.stale, "person:carl", c.n4, "c4")
    c.quote(d2, "person:carl", c.n5, "c5")
    built = projections.build_in(conn)
    rows = conn.execute(
        select(entities.c.natural_id, person_rankings.c["items"], person_rankings.c.insights)
        .join(entities, entities.c.entity_id == person_rankings.c.entity_id)
        .where(person_rankings.c.build_id == built.build_id)
        .order_by(person_rankings.c.rank)
    ).all()
    assert [tuple(r) for r in rows] == [("person:ann", 2, 2), ("person:bob", 2, 2)]
    assert built.stats["person_rankings"] == 2
    ranks = conn.execute(
        select(person_rankings.c.rank)
        .where(person_rankings.c.build_id == built.build_id)
        .order_by(person_rankings.c.rank)
    ).scalars()
    assert list(ranks) == [1, 2]


def test_the_placeholder_test_is_the_file_codes_on_the_speakers_own_id(
    conn: Connection,  # noqa: F811
) -> None:
    from podcast_scraper.graph_id_utils import is_unresolved_speaker_placeholder

    c = Corpus(conn)
    for natural_id in (
        "person:speaker-03",
        "person:speaker-ep1-host",
        "person:speaker-john-knight",
    ):
        c.entity(natural_id)
        c.quote(c.items["2"], natural_id, c.n3, natural_id)
    got = set(projections._placeholder_speakers(conn))
    wanted = {
        c.e[n]
        for n in ("person:speaker-03", "person:speaker-ep1-host", "person:speaker-john-knight")
        if is_unresolved_speaker_placeholder(n)
    }
    assert got == wanted and c.e["person:speaker-03"] in got
    assert c.e["person:speaker-john-knight"] not in got


def _relations(conn: Connection, build_id: int) -> list[tuple[str, str, float]]:  # noqa: F811
    a, b = entities.alias("a"), entities.alias("b")
    return [
        tuple(r)
        for r in conn.execute(
            select(a.c.natural_id, b.c.natural_id, topic_relations.c.weight)
            .select_from(
                topic_relations.join(a, a.c.entity_id == topic_relations.c.topic_a).join(
                    b, b.c.entity_id == topic_relations.c.topic_b
                )
            )
            .where(
                topic_relations.c.build_id == build_id,
                topic_relations.c.kind == projections.SHARES_INSIGHT,
            )
            .order_by(a.c.natural_id, topic_relations.c.weight.desc(), b.c.natural_id)
        )
    ]


def test_topic_relations_count_the_current_shown_insights_two_canonical_topics_share(
    conn: Connection,  # noqa: F811
) -> None:
    # relational_queries.related_topics on rows (database, 2026-11-27): topic ABOUT insight ABOUT
    # other topic, the distinct insights the two share, both directions, canonical ids.
    c = Corpus(conn)
    c.entity("topic:ml")
    # n1 (item 1) is about topic:ai; n3 (item 2) about topic:ai-2, which is topic:ai.
    c.tag(c.n1, "topic:other", "ABOUT")
    c.tag(c.n1, "topic:ml", "ABOUT")
    c.tag(c.n3, "topic:ml", "ABOUT")
    # Not counted: a stale document's insight, a hidden one, a person mention.
    c.tag(c.n4, "topic:other", "ABOUT")
    c.tag(c.n5, "topic:other", "ABOUT")
    c.tag(c.n3, "person:ann", "ABOUT")
    built = projections.build_in(conn)
    assert _relations(conn, built.build_id) == [
        ("topic:ai", "topic:ml", 2.0),
        ("topic:ai", "topic:other", 1.0),
        ("topic:ml", "topic:ai", 2.0),
        ("topic:ml", "topic:other", 1.0),
        ("topic:other", "topic:ai", 1.0),
        ("topic:other", "topic:ml", 1.0),
    ]
    assert built.stats["topic_relations"] == 6


def test_each_topic_keeps_its_strongest_related_topics_ties_by_natural_id(
    conn: Connection, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    monkeypatch.setattr(projections, "RELATED_PER_TOPIC", 2)
    c = Corpus(conn)
    for name in ("topic:b", "topic:c", "topic:d"):
        c.entity(name)
        c.tag(c.n1, name, "ABOUT")
    c.tag(c.n3, "topic:d", "ABOUT")
    built = projections.build_in(conn)
    kept = [r for r in _relations(conn, built.build_id) if r[0] == "topic:ai"]
    # topic:d shares 2, then topic:b before topic:c on the id; topic:c is cut.
    assert kept == [("topic:ai", "topic:d", 2.0), ("topic:ai", "topic:b", 1.0)]


def _speakers(
    conn: Connection, build_id: int  # noqa: F811
) -> list[tuple[str, str, int, int, int]]:
    t, p = entities.alias("t"), entities.alias("p")
    return [
        tuple(r)
        for r in conn.execute(
            select(
                t.c.natural_id,
                p.c.natural_id,
                topic_speakers.c["items"],
                topic_speakers.c.insights,
                topic_speakers.c.rank,
            )
            .select_from(
                topic_speakers.join(t, t.c.entity_id == topic_speakers.c.topic_id).join(
                    p, p.c.entity_id == topic_speakers.c.person_id
                )
            )
            .where(topic_speakers.c.build_id == build_id)
            .order_by(t.c.natural_id, topic_speakers.c.rank)
        )
    ]


def test_topic_speakers_attribute_each_insight_to_its_first_quotes_speaker(
    conn: Connection,  # noqa: F811
) -> None:
    # cil_queries.topics_perspectives on rows, one topic, unscoped (database, 2026-11-27).
    c = Corpus(conn)
    c.e["person:bob-2"] = _one(
        conn,
        entities,
        natural_id="person:bob-2",
        type="person",
        label="Bob",
        normalized_label="bob",
        canonical_id=c.e["person:bob"],
    )
    c.entity("person:speaker-03")
    d2 = c.items["2"]
    # n1 (topic:ai, item 1): the fixture's q1 by bob is its first quote (ord 0); ann's later
    # quote does not take it.
    c.quote(c.items["1"], "person:ann", c.n1, "a1")
    # n3 (topic:ai-2, which is topic:ai; item 2): first quote by bob-2, so bob again.
    c.quote(d2, "person:bob-2", c.n3, "b3")
    # An insight about topic:other first spoken by ann, and one by a diarization voice.
    other = c.insight("o1", d2, 3, about=["topic:other"])
    c.quote(d2, "person:ann", other, "a-o1")
    voice = c.insight("o2", d2, 4, about=["topic:other"])
    c.quote(d2, "person:speaker-03", voice, "s-o2")
    # Not counted: a hidden insight's quote.
    c.quote(d2, "person:ann", c.n5, "a5")
    built = projections.build_in(conn)
    assert _speakers(conn, built.build_id) == [
        ("topic:ai", "person:bob", 2, 2, 1),
        ("topic:other", "person:ann", 1, 1, 1),
    ]
    assert built.stats["topic_speakers"] == 2


def test_topic_speakers_rank_by_insights_then_the_speakers_natural_id(
    conn: Connection,  # noqa: F811
) -> None:
    c = Corpus(conn)
    c.entity("person:abe")
    d2 = c.items["2"]
    # topic:other: abe and ann one insight each; the tie goes to person:abe.
    for n, person in (("x1", "person:ann"), ("x2", "person:abe")):
        insight = c.insight(n, d2, 5 + len(n), about=["topic:other"])
        c.quote(d2, person, insight, n)
    built = projections.build_in(conn)
    other = [r for r in _speakers(conn, built.build_id) if r[0] == "topic:other"]
    assert other == [
        ("topic:other", "person:abe", 1, 1, 1),
        ("topic:other", "person:ann", 1, 1, 2),
    ]

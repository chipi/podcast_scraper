"""The graph engine's walk core (db.graph, ADR-183): presets over fixture rows, their statement
counts, merged entities, what is never served, and the transaction they run in."""

from __future__ import annotations

import contextlib
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

import pytest
from sqlalchemy import create_engine, event, insert, select, Table, text
from sqlalchemy.engine import Connection

from podcast_scraper.db import graph, projections
from podcast_scraper.db.schema import (
    cluster_builds,
    cluster_ids,
    cluster_links,
    cluster_members,
    clusters,
    content_items,
    documents,
    entities,
    evidence,
    insight_attribution,
    insight_mentions,
    insight_support,
    insights,
    item_entities,
    person_roles,
    sources,
)
from podcast_scraper.db.settings import normalize_url
from tests._database import worker_database_url

pytestmark = pytest.mark.integration

T0 = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn() -> Iterator[Connection]:
    url = worker_database_url()
    assert url, "set PODCAST_TEST_DATABASE_URL (make test-db-up)"
    engine = create_engine(normalize_url(url))
    with engine.connect() as connection:
        yield connection
        connection.rollback()
    engine.dispose()


@contextlib.contextmanager
def _statements(connection: Connection) -> Iterator[list[str]]:
    seen: list[str] = []

    def count(_c: Any, _cursor: Any, statement: str, *_rest: Any) -> None:
        if not statement.lstrip().upper().startswith("SET "):
            seen.append(statement)

    event.listen(connection, "before_cursor_execute", count)
    try:
        yield seen
    finally:
        event.remove(connection, "before_cursor_execute", count)


def _one(conn: Connection, table: Table, **values: object) -> int:
    key = next(iter(table.primary_key.columns))
    found: int = conn.execute(insert(table).values(values).returning(key)).scalar_one()
    return found


class Corpus:
    """One podcast published by an organisation, three items (the third not served), a topic
    written under two ids and merged, two people, insights on served and superseded documents."""

    def __init__(self, conn: Connection) -> None:
        self.conn = conn
        self.e: dict[str, int] = {}
        for natural_id in ("topic:ai", "person:ann", "person:bob", "org:acme", "topic:other"):
            self.entity(natural_id)
        self.e["topic:ai-2"] = _one(
            conn,
            entities,
            natural_id="topic:ai-2",
            type="topic",
            label="AI again",
            normalized_label="ai again",
            canonical_id=self.e["topic:ai"],
        )
        self.source = _one(
            conn,
            sources,
            natural_id="podcast:s",
            content_type="podcast",
            title="The Show",
            publisher_id=self.e["org:acme"],
        )
        self.items = {
            "1": self.item("1", T0 - timedelta(days=1)),
            "2": self.item("2", T0 - timedelta(days=2)),
            "3": self.item("3", T0 - timedelta(days=3), served=0),
            "old": self.item("old", T0 - timedelta(days=60)),
        }
        i1, stale = self.items["1"]
        d1 = _one(
            conn,
            documents,
            item_id=i1,
            kind="transcript",
            variant="cleaned",
            version=2,
            supersedes_id=stale,
        )
        self.items["1"] = (i1, d1)
        self.stale = (i1, stale)
        i2, _ = self.items["2"]
        i3, _ = self.items["3"]
        self.mention(i1, "topic:ai")
        self.mention(i2, "topic:ai-2")
        self.mention(i1, "person:ann")
        self.mention(i3, "topic:ai")
        for item, person, role in (
            (i1, "person:bob", "host"),
            (i2, "person:bob", "guest"),
            (i1, "person:ann", "mentioned"),
        ):
            conn.execute(
                insert(person_roles).values(item_id=item, person_id=self.e[person], role=role)
            )
        self.n1 = self.insight("n1", self.items["1"], 0, about=["topic:ai"])
        self.n2 = self.insight("n2", self.items["1"], 1, people=["person:ann"])
        self.n3 = self.insight("n3", self.items["2"], 0, about=["topic:ai-2"])
        self.n4 = self.insight("n4", (i1, stale), 2, about=["topic:ai"])
        self.n5 = self.insight("n5", self.items["2"], 1, about=["topic:ai"], visible=False)
        self.q1 = _one(
            conn,
            evidence,
            natural_id="quote:1",
            item_id=i1,
            document_id=d1,
            speaker_id=self.e["person:bob"],
            start_seq=0,
            start_offset=0,
            end_seq=0,
            end_offset=1,
            text="q",
        )
        conn.execute(insert(insight_support).values(insight_id=self.n1, evidence_id=self.q1, ord=0))
        conn.execute(
            insert(insight_attribution).values(
                insight_id=self.n1, person_id=self.e["person:bob"], item_id=i1
            )
        )

    def entity(self, natural_id: str) -> int:
        kind = natural_id.split(":", 1)[0]
        self.e[natural_id] = _one(
            self.conn,
            entities,
            natural_id=natural_id,
            type=kind,
            label=natural_id.title(),
            normalized_label=natural_id,
        )
        return self.e[natural_id]

    def item(self, n: str, published: datetime, served: int = 1) -> tuple[int, int]:
        item = _one(
            self.conn,
            content_items,
            natural_id=f"episode:{n}",
            slug=f"slug-{n}",
            source_id=self.source,
            external_item_id=f"ext-{n}",
            content_type="podcast",
            title=f"Episode {n}",
            published_at=published,
            served=served,
        )
        document = _one(self.conn, documents, item_id=item, kind="transcript", variant="cleaned")
        self._published = getattr(self, "_published", {})
        self._published[item] = published
        return item, document

    def mention(self, item: int, natural_id: str) -> None:
        self.conn.execute(
            insert(item_entities).values(
                item_id=item,
                entity_id=self.e[natural_id],
                role="mentions",
                published_at=self._published[item],
                content_type="podcast",
            )
        )

    def insight(
        self,
        natural_id: str,
        doc: tuple[int, int],
        rank: int,
        *,
        about: Optional[list[str]] = None,
        people: Optional[list[str]] = None,
        visible: bool = True,
    ) -> int:
        item, document = doc
        insight = _one(
            self.conn,
            insights,
            natural_id=f"insight:{natural_id}",
            item_id=item,
            document_id=document,
            insight_type="claim",
            rank=rank,
            routing_tag="surface",
            text_origin="authored",
            text=f"Insight {natural_id}",
            visible=visible,
        )
        for kind, names in (("ABOUT", about or []), ("MENTIONS_PERSON", people or [])):
            for name in names:
                self.conn.execute(
                    insert(insight_mentions).values(
                        insight_id=insight,
                        entity_id=self.e[name],
                        kind=kind,
                        item_id=item,
                        rank=rank,
                    )
                )
        return insight

    def tag(self, insight: int, natural_id: str, kind: str) -> None:
        """One more ``insight_mentions`` row for ``insight``, of any kind."""
        item, rank = self.conn.execute(
            text("SELECT item_id, rank FROM corpus.insights WHERE insight_id = :i"), {"i": insight}
        ).one()
        self.conn.execute(
            insert(insight_mentions).values(
                insight_id=insight, entity_id=self.e[natural_id], kind=kind, item_id=item, rank=rank
            )
        )

    def stated(self, insight: int, person: str) -> None:
        item = self.conn.execute(
            text("SELECT item_id FROM corpus.insights WHERE insight_id = :i"), {"i": insight}
        ).scalar_one()
        self.conn.execute(
            insert(insight_attribution).values(
                insight_id=insight, person_id=self.e[person], item_id=item
            )
        )

    def quote(self, doc: tuple[int, int], speaker: str, support: int, n: str) -> int:
        item, document = doc
        quote = _one(
            self.conn,
            evidence,
            natural_id=f"quote:{n}",
            item_id=item,
            document_id=document,
            speaker_id=self.e[speaker],
            start_seq=0,
            start_offset=0,
            end_seq=0,
            end_offset=1,
            text="q",
        )
        self.conn.execute(
            insert(insight_support).values(insight_id=support, evidence_id=quote, ord=1)
        )
        return quote


def _ids(slice_: dict[str, Any]) -> set[str]:
    return {n["id"] for n in slice_["nodes"]}


def _edges(slice_: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {(e["type"], e["from"], e["to"]) for e in slice_["edges"]}


def test_slice_items_returns_the_items_graph_in_seven_statements(conn: Connection) -> None:
    Corpus(conn)
    with _statements(conn) as seen:
        got = graph.slice_items(conn, ["episode:1", "episode:3", "episode:x"])
    # 1 (revision and seeds) + 2 steps + nodes, items, induced edges, groups: ADR-183's 1 + 2 + 4.
    assert len(seen) == 7
    assert got["format"] == "graph-slice/1"
    assert got["corpus_revision"] == 0
    assert got["unknown_seeds"] == ["episode:3", "episode:x"]
    assert _ids(got) == {
        "episode:1",
        "podcast:s",
        "topic:ai",
        "person:ann",
        "person:bob",
        "insight:n1",
        "insight:n2",
    }
    assert ("HAS_ITEM", "podcast:s", "episode:1") in _edges(got)
    assert ("MENTIONS", "topic:ai", "episode:1") in _edges(got)
    assert ("PARTICIPATES_IN", "person:bob", "episode:1") in _edges(got)
    assert ("HAS_INSIGHT", "episode:1", "insight:n1") in _edges(got)
    assert ("ABOUT", "insight:n1", "topic:ai") in _edges(got)
    assert ("MENTIONS_PERSON", "insight:n2", "person:ann") in _edges(got)
    assert ("STATED_BY", "insight:n1", "person:bob") in _edges(got)
    # ann is only "mentioned" in person_roles: a mention, not a participant.
    assert ("PARTICIPATES_IN", "person:ann", "episode:1") not in _edges(got)
    assert [i["item_id"] for i in got["items"]] == ["episode:1"]
    assert got["items"][0]["published_at"] == "2026-11-19T12:00:00Z"
    owned = {n["id"]: n.get("item_id") for n in got["nodes"]}
    assert owned["insight:n1"] == "episode:1" and owned["topic:ai"] is None
    assert got["truncated"]["reasons"] == []


def test_a_superseded_or_hidden_insight_and_an_unserved_item_are_never_served(
    conn: Connection,
) -> None:
    Corpus(conn)
    got = graph.slice_items(conn, ["episode:1", "episode:2"])
    assert "insight:n4" not in _ids(got)
    assert "insight:n5" not in _ids(got)
    near = graph.slice_neighbourhood(conn, "topic:ai", "node.items")
    assert "episode:3" not in _ids(near)


def test_a_merged_entity_is_one_node_under_its_canonical_id(conn: Connection) -> None:
    Corpus(conn)
    got = graph.slice_items(conn, ["episode:1", "episode:2"])
    assert "topic:ai-2" not in _ids(got)
    assert ("MENTIONS", "topic:ai", "episode:2") in _edges(got)
    assert ("ABOUT", "insight:n3", "topic:ai") in _edges(got)
    by_member = graph.slice_neighbourhood(conn, "topic:ai-2", "node.items")
    assert by_member["nodes"][0]["id"] == "topic:ai"


def test_node_items_walks_every_branch_and_joins_seed_and_item_by_path_count(
    conn: Connection,
) -> None:
    Corpus(conn)
    with _statements(conn) as seen:
        got = graph.slice_neighbourhood(conn, "topic:ai", "node.items")
    # 1 (revision and seed) + 2 levels (the longest branch) + nodes, items, induced edges, groups.
    assert len(seen) == 7
    assert _ids(got) == {"topic:ai", "episode:1", "episode:2"}
    weights = {(e["from"], e["to"]): e["weight"] for e in got["edges"] if e["type"] == "node.items"}
    # episode:1 by MENTIONS and by n1's ABOUT; episode:2 by ai-2's MENTIONS and n3's ABOUT.
    assert weights == {("topic:ai", "episode:1"): 2, ("topic:ai", "episode:2"): 2}
    assert ("MENTIONS", "topic:ai", "episode:1") in _edges(got)
    assert got["edges"][0]["weight"] == 2


def test_node_items_for_a_person_reaches_items_through_quotes_and_roles(conn: Connection) -> None:
    Corpus(conn)
    with _statements(conn) as seen:
        got = graph.slice_neighbourhood(conn, "person:bob", "node.items")
    assert len(seen) == 1 + 3 + 4
    assert _ids(got) == {"person:bob", "episode:1", "episode:2"}


def test_person_topics_reads_entity_types_for_the_kept_ends(conn: Connection) -> None:
    Corpus(conn)
    with _statements(conn) as seen:
        got = graph.slice_neighbourhood(conn, "person:bob", "person.topics")
    # 1 + 2 levels + the reached entities' types + 4.
    assert len(seen) == 1 + 2 + 1 + 4
    assert _ids(got) == {"person:bob", "topic:ai"}
    assert [i["item_id"] for i in got["items"]] == ["episode:1"]


def test_an_unknown_seed_is_not_an_error_and_an_unknown_path_is(conn: Connection) -> None:
    Corpus(conn)
    got = graph.slice_neighbourhood(conn, "topic:none", "node.items")
    assert got["unknown_seeds"] == ["topic:none"] and got["nodes"] == []
    with pytest.raises(graph.UnknownPath):
        graph.slice_neighbourhood(conn, "topic:ai", "node.nowhere")
    with pytest.raises(graph.UnknownPath):
        graph.slice_neighbourhood(conn, "topic:ai", "person.topics")


def test_slice_recent_widens_its_window_until_enough_items_match(conn: Connection) -> None:
    Corpus(conn)
    with _statements(conn) as seen:
        got = graph.slice_recent(conn, window="7d", min_items=3, now=T0)
    assert len(seen) == 7
    assert got["applied_window"] == "90d"
    assert [i["item_id"] for i in got["items"]] == ["episode:1", "episode:2", "episode:old"]
    narrow = graph.slice_recent(conn, window="7d", min_items=2, now=T0)
    assert narrow["applied_window"] == "7d"
    assert [i["item_id"] for i in narrow["items"]] == ["episode:1", "episode:2"]


def test_the_budget_admits_items_first_then_round_robin_and_says_it_cut(conn: Connection) -> None:
    Corpus(conn)
    got = graph.slice_items(conn, ["episode:1", "episode:2"], budget=4)
    ids = _ids(got)
    assert {"episode:1", "episode:2"} <= ids and len(ids) <= 4
    assert got["truncated"]["reasons"] == ["budget"]
    assert got["truncated"]["nodes_left"] > 0


def test_a_cut_path_walk_lists_only_the_items_of_the_nodes_it_returns(conn: Connection) -> None:
    Corpus(conn)
    got = graph.slice_neighbourhood(conn, "topic:ai", "node.items", budget=2)
    returned = {n["id"] for n in got["nodes"] if n["type"] == "ContentItem"}
    assert len(returned) == 1
    assert {i["item_id"] for i in got["items"]} == returned
    assert got["truncated"]["reasons"] == ["budget"] and got["truncated"]["nodes_left"] == 1


def test_the_edge_budget_cuts_the_lightest_edges_and_says_so(conn: Connection) -> None:
    Corpus(conn)
    got = graph.slice_neighbourhood(conn, "topic:ai", "node.items", edge_budget=2)
    assert len(got["edges"]) == 2
    assert {e["type"] for e in got["edges"]} == {"node.items"}
    assert "edge_budget" in got["truncated"]["reasons"]


def test_a_hub_past_the_fanout_cap_is_named(conn: Connection) -> None:
    c = Corpus(conn)
    for n in range(graph.FANOUT + 1):
        item, _ = c.item(f"hub{n}", T0 - timedelta(hours=n))
        c.mention(item, "topic:other")
    got = graph.slice_neighbourhood(conn, "topic:other", "node.items", budget=3000)
    assert "fanout" in got["truncated"]["reasons"]
    assert got["truncated"]["fanout_capped"] == ["topic:other"]
    assert len([n for n in got["nodes"] if n["type"] == "ContentItem"]) == graph.FANOUT


def test_bounds_are_refused(conn: Connection) -> None:
    with pytest.raises(ValueError):
        graph.slice_items(conn, [f"episode:{n}" for n in range(graph.MAX_ITEM_SEEDS + 1)])
    with pytest.raises(ValueError):
        graph.slice_items(conn, ["episode:1"], budget=graph.MAX_BUDGET + 1)
    with pytest.raises(ValueError):
        graph.slice_recent(conn, window="1d")


def test_read_slice_is_read_only_repeatable_read_and_times_out() -> None:
    url = worker_database_url()
    assert url, "set PODCAST_TEST_DATABASE_URL (make test-db-up)"
    engine = create_engine(normalize_url(url))
    try:
        level = graph.read_slice(
            engine,
            lambda c: (
                c.execute(text("SHOW transaction_isolation")).scalar_one(),
                c.execute(text("SHOW transaction_read_only")).scalar_one(),
                c.execute(text("SHOW statement_timeout")).scalar_one(),
            ),
            timeout_ms=1234,
        )
        assert level == ("repeatable read", "on", "1234ms")
        with pytest.raises(graph.GraphTimeout):
            graph.read_slice(engine, lambda c: c.execute(text("SELECT pg_sleep(1)")), timeout_ms=50)
        with engine.connect() as after:
            assert after.execute(text("SHOW statement_timeout")).scalar_one() != "50ms"
    finally:
        engine.dispose()


def _weights(slice_: dict[str, Any], path: str) -> dict[str, int]:
    return {e["to"]: e["weight"] for e in slice_["edges"] if e["type"] == path}


def test_an_unserved_or_withheld_items_rows_reach_no_walk(conn: Connection) -> None:
    c = Corpus(conn)
    hidden = c.insight("n6", c.items["3"], 0, about=["topic:ai"])
    c.stated(hidden, "person:ann")
    withheld, _ = c.item("w", T0 - timedelta(hours=1))
    conn.execute(
        content_items.update().where(content_items.c.item_id == withheld).values(withheld=True)
    )
    c.mention(withheld, "topic:ai")
    topics = graph.slice_neighbourhood(conn, "person:ann", "person.topics")
    assert _ids(topics) == {"person:ann"} and topics["items"] == []
    people = graph.slice_neighbourhood(conn, "topic:ai", "topic.people")
    assert "person:ann" not in _ids(people)
    assert "episode:3" not in {i["item_id"] for i in people["items"]}
    items = graph.slice_neighbourhood(conn, "topic:ai", "node.items")
    assert _ids(items) == {"topic:ai", "episode:1", "episode:2"}
    assert graph.slice_items(conn, ["episode:w"])["unknown_seeds"] == ["episode:w"]


def test_a_cut_keeps_the_node_with_more_paths_then_the_newer_row(conn: Connection) -> None:
    Corpus(conn)
    got = graph.slice_neighbourhood(conn, "topic:ai", "node.items", budget=2)
    # episode:1 and episode:2 both have 2 paths; episode:2's row is the newer (higher key).
    assert {n["id"] for n in got["nodes"] if n["type"] == "ContentItem"} == {"episode:2"}


def test_item_slices_say_when_the_budget_cut_items(conn: Connection) -> None:
    Corpus(conn)
    got = graph.slice_items(conn, ["episode:1", "episode:2"], budget=1)
    assert _ids(got) == {"episode:1"}
    assert got["truncated"]["reasons"] == ["budget"] and got["truncated"]["items_left"] == 1
    recent = graph.slice_recent(conn, window="all", min_items=1, budget=2, now=T0)
    assert [i["item_id"] for i in recent["items"]] == ["episode:1", "episode:2"]
    assert "budget" in recent["truncated"]["reasons"]


def test_a_quote_on_a_superseded_document_is_not_walked(conn: Connection) -> None:
    c = Corpus(conn)
    c.quote(c.stale, "person:ann", c.n2, "stale")
    got = graph.slice_neighbourhood(conn, "person:ann", "node.items")
    # ann reaches episode:1 by MENTIONS and by n2's MENTIONS_PERSON, not by the stale quote.
    assert _weights(got, "node.items") == {"episode:1": 2}


def test_a_person_reaches_items_by_role_and_by_quote_and_a_second_role_is_one_edge(
    conn: Connection,
) -> None:
    c = Corpus(conn)
    i1, _ = c.items["1"]
    conn.execute(insert(person_roles).values(item_id=i1, person_id=c.e["person:bob"], role="guest"))
    got = graph.slice_neighbourhood(conn, "person:bob", "node.items")
    # episode:1: PARTICIPATES_IN once (host and guest are one edge), and q1 -> n1 -> episode:1.
    assert _weights(got, "node.items") == {"episode:1": 2, "episode:2": 1}
    roles = [e for e in got["edges"] if e["type"] == "PARTICIPATES_IN" and e["to"] == "episode:1"]
    assert len(roles) == 1


def test_the_insight_paths_between_people_topics_and_organisations(conn: Connection) -> None:
    c = Corpus(conn)
    c.tag(c.n1, "org:acme", "MENTIONS_ORG")
    c.stated(c.n3, "person:ann")
    c.entity("object:mic")
    c.tag(c.n3, "object:mic", "MENTIONS_OBJECT")
    assert _ids(graph.slice_neighbourhood(conn, "topic:ai", "topic.people")) == {
        "topic:ai",
        "person:bob",
        "person:ann",
    }
    assert _ids(graph.slice_neighbourhood(conn, "topic:ai", "topic.entities")) == {
        "topic:ai",
        "org:acme",
    }
    assert _ids(graph.slice_neighbourhood(conn, "org:acme", "org.topics")) == {
        "org:acme",
        "topic:ai",
    }
    assert _ids(graph.slice_neighbourhood(conn, "person:bob", "person.co_speakers")) == {
        "person:bob",
        "person:ann",
    }
    objects = graph.slice_neighbourhood(conn, "object:mic", "node.items")
    assert _ids(objects) == {"object:mic", "episode:2"}


def test_exactly_fanout_edges_is_not_capped(conn: Connection) -> None:
    c = Corpus(conn)
    for n in range(graph.FANOUT):
        item, _ = c.item(f"edge{n}", T0 - timedelta(hours=n))
        c.mention(item, "topic:other")
    got = graph.slice_neighbourhood(conn, "topic:other", "node.items", budget=3000)
    assert got["truncated"]["reasons"] == [] and got["truncated"]["fanout_capped"] == []
    assert len([n for n in got["nodes"] if n["type"] == "ContentItem"]) == graph.FANOUT


def test_with_counts_reads_item_counts_from_the_published_build_in_one_statement(
    conn: Connection,
) -> None:
    Corpus(conn)
    plain = graph.slice_items(conn, ["episode:1"])
    assert plain["graph_build_id"] is None
    assert all("item_count" not in n for n in plain["nodes"])
    unbuilt = graph.slice_items(conn, ["episode:1"], with_=["counts"])
    assert all("item_count" not in n for n in unbuilt["nodes"])
    built = projections.build_in(conn)
    with _statements(conn) as seen:
        got = graph.slice_items(conn, ["episode:1"], with_=["counts"])
    # 7 as without counts, plus the one counts statement.
    assert len(seen) == 8
    assert got["graph_build_id"] == built.build_id
    assert got["params"]["with"] == ["counts"]
    counts = {n["id"]: n.get("item_count") for n in got["nodes"]}
    # topic:ai is on items 1 and 2 under its canonical id; bob is on items only by role, which
    # entity_item_counts does not count; nodes that are not entities carry no item_count.
    assert counts["topic:ai"] == 2 and counts["person:ann"] == 1 and counts["person:bob"] == 0
    assert counts["episode:1"] is None and counts["insight:n1"] is None
    with pytest.raises(ValueError):
        graph.slice_items(conn, ["episode:1"], with_=["groups"])


def test_nodes_carry_their_groups_and_the_slice_a_catalogue_with_their_parents(
    conn: Connection,
) -> None:
    c = Corpus(conn)

    def group(build: int, kind: str, natural_id: str, parent: Optional[int] = None) -> int:
        cluster = _one(conn, cluster_ids, kind=kind, natural_id=natural_id)
        conn.execute(
            insert(clusters).values(
                build_id=build,
                cluster_id=cluster,
                kind=kind,
                label=natural_id.upper(),
                parent_id=parent,
            )
        )
        return cluster

    def member(build: int, cluster: int, **key: int) -> None:
        conn.execute(insert(cluster_members).values(build_id=build, cluster_id=cluster, **key))

    old = _one(conn, cluster_builds, rule_version="test/1", state="retired")
    member(old, group(old, "theme", "tc:old"), entity_id=c.e["topic:ai"])
    build = _one(conn, cluster_builds, rule_version="test/1", state="published")
    sup = group(build, "super_theme", "sth:x")
    # A member of topic:ai's merged row puts its canonical node in the storyline.
    member(build, group(build, "storyline", "thc:s", parent=sup), entity_id=c.e["topic:ai-2"])
    member(build, group(build, "insight", "ic:i"), insight_id=c.n1)
    member(build, group(build, "theme", "tc:quiet"), entity_id=c.e["topic:other"])
    with _statements(conn) as seen:
        got = graph.slice_items(conn, ["episode:1"])
    assert len(seen) == 7
    groups = {n["id"]: n["groups"] for n in got["nodes"]}
    assert groups["topic:ai"] == ["thc:s"]
    assert groups["insight:n1"] == ["ic:i"]
    assert groups["person:ann"] == [] and groups["episode:1"] == []
    # The parent comes with its storyline; a group no node is in, and the retired build, do not.
    assert got["groups"] == [
        {"id": "ic:i", "kind": "insight", "label": "IC:I", "parent_id": None},
        {"id": "sth:x", "kind": "super_theme", "label": "STH:X", "parent_id": None},
        {"id": "thc:s", "kind": "storyline", "label": "THC:S", "parent_id": "sth:x"},
    ]
    # IN_GROUP: the topic to its storyline, the storyline to its super-theme; an insight's
    # cluster is in its node's groups, not an edge (decision 10: a topic or group to a group).
    in_group = [e for e in got["edges"] if e["type"] == "IN_GROUP"]
    assert sorted((e["from"], e["to"]) for e in in_group) == [
        ("thc:s", "sth:x"),
        ("topic:ai", "thc:s"),
    ]
    assert all(e["class"] == "structural" and "item_id" not in e for e in in_group)


def test_a_slice_with_no_entity_or_insight_node_sends_no_groups_statement(
    conn: Connection,
) -> None:
    # Decision 9's counts are an upper bound fixed by the preset: a read whose input is empty
    # sends nothing. An item with no KG or GI rows reaches only its source, so the tail is
    # nodes, items and induced edges, without groups: 3, not 4.
    c = Corpus(conn)
    c.item("bare", T0)
    with _statements(conn) as seen:
        got = graph.slice_items(conn, ["episode:bare"])
    assert _ids(got) == {"episode:bare", "podcast:s"}
    assert got["groups"] == []
    # 1 (revision and seeds) + 1 item step (the insight step has no frontier) + 3.
    assert len(seen) == 1 + 1 + 3


def test_with_no_published_cluster_build_nodes_are_in_no_group(conn: Connection) -> None:
    c = Corpus(conn)
    old = _one(conn, cluster_builds, rule_version="test/1", state="retired")
    cluster = _one(conn, cluster_ids, kind="theme", natural_id="tc:old")
    conn.execute(insert(clusters).values(build_id=old, cluster_id=cluster, kind="theme", label="x"))
    conn.execute(
        insert(cluster_members).values(build_id=old, cluster_id=cluster, entity_id=c.e["topic:ai"])
    )
    got = graph.slice_items(conn, ["episode:1"])
    assert all(n["groups"] == [] for n in got["nodes"]) and got["groups"] == []


def _groups_build(c: Corpus) -> dict[str, int]:
    conn = c.conn
    build = _one(conn, cluster_builds, rule_version="test/1", state="published")
    made: dict[str, int] = {}

    def group(kind: str, natural_id: str, parent: Optional[str] = None) -> int:
        cluster = _one(conn, cluster_ids, kind=kind, natural_id=natural_id)
        conn.execute(
            insert(clusters).values(
                build_id=build,
                cluster_id=cluster,
                kind=kind,
                label=natural_id,
                parent_id=made[parent] if parent else None,
            )
        )
        made[natural_id] = cluster
        return cluster

    def member(natural_id: str, **key: int) -> None:
        conn.execute(
            insert(cluster_members).values(build_id=build, cluster_id=made[natural_id], **key)
        )

    group("super_theme", "sth:x")
    group("storyline", "thc:s", parent="sth:x")
    member("thc:s", entity_id=c.e["topic:ai-2"])
    group("insight", "ic:i")
    member("ic:i", insight_id=c.n3)
    group("theme", "tc:quiet")
    member("tc:quiet", entity_id=c.e["topic:other"])
    return made


def test_slice_group_is_a_groups_topics_and_the_newest_items_that_carry_them(
    conn: Connection,
) -> None:
    c = Corpus(conn)
    _groups_build(c)
    with _statements(conn) as seen:
        got = graph.slice_group(conn, "thc:s")
    # 1 (revision, the group, its topics and items) + 2 item steps + the tail of 4.
    assert len(seen) == 7
    assert got["preset"] == "slice.group" and got["params"]["group"] == "thc:s"
    # topic:ai-2's canonical topic:ai is on items 1 and 2; item 3 is not served.
    assert [i["item_id"] for i in got["items"]] == ["episode:1", "episode:2"]
    assert "topic:ai" in _ids(got) and "topic:ai-2" not in _ids(got)
    assert {g["id"] for g in got["groups"]} >= {"thc:s", "sth:x"}
    assert got["unknown_seeds"] == []


def test_a_super_theme_reaches_its_child_groups_topics_and_an_insight_cluster_its_items(
    conn: Connection,
) -> None:
    c = Corpus(conn)
    _groups_build(c)
    sup = graph.slice_group(conn, "sth:x")
    assert [i["item_id"] for i in sup["items"]] == ["episode:1", "episode:2"]
    assert "topic:ai" in _ids(sup)
    ins = graph.slice_group(conn, "ic:i")
    assert [i["item_id"] for i in ins["items"]] == ["episode:2"]
    quiet = graph.slice_group(conn, "tc:quiet")
    assert quiet["items"] == [] and quiet["unknown_seeds"] == []


def test_an_unknown_group_or_one_only_in_a_retired_build_is_an_unknown_seed(
    conn: Connection,
) -> None:
    c = Corpus(conn)
    made = _groups_build(c)
    assert graph.slice_group(conn, "thc:none")["unknown_seeds"] == ["thc:none"]
    conn.execute(cluster_builds.update().values(state="retired"))
    gone = graph.slice_group(conn, "thc:s")
    assert gone["unknown_seeds"] == ["thc:s"] and gone["nodes"] == []
    assert made["thc:s"]


def test_slice_group_cuts_its_items_at_the_budget(conn: Connection) -> None:
    c = Corpus(conn)
    _groups_build(c)
    got = graph.slice_group(conn, "thc:s", budget=2)
    assert "budget" in got["truncated"]["reasons"]
    assert len(got["nodes"]) <= 2


def test_a_group_with_more_topics_than_the_budget_still_shows_its_items(
    conn: Connection,
) -> None:
    # Database's review of #2628: topics admitted first left a wide storyline no items at all.
    c = Corpus(conn)
    made = _groups_build(c)
    build = conn.execute(select(cluster_builds.c.build_id)).scalar_one()
    wide = _one(conn, cluster_ids, kind="storyline", natural_id="thc:wide")
    conn.execute(
        insert(clusters).values(build_id=build, cluster_id=wide, kind="storyline", label="w")
    )
    for name in ("topic:ai-2", "topic:other", "person:ann"):
        conn.execute(
            insert(cluster_members).values(build_id=build, cluster_id=wide, entity_id=c.e[name])
        )
    got = graph.slice_group(conn, "thc:wide", budget=2)
    assert [i["item_id"] for i in got["items"]] == ["episode:1"]
    assert len(got["nodes"]) <= 2 and "budget" in got["truncated"]["reasons"]
    # With room for every topic and both items, nothing is cut for the rule.
    roomy = graph.slice_group(conn, "thc:wide", budget=50)
    assert [i["item_id"] for i in roomy["items"]] == ["episode:1", "episode:2"]
    assert {"topic:ai", "topic:other", "person:ann"} <= _ids(roomy)
    assert made["thc:s"]


def test_expand_groups_adds_each_groups_newest_other_items_after_the_primary_ones(
    conn: Connection,
) -> None:
    """The sibling pull (CRFC-004 §2): thc:s has topic:ai-2, whose canonical topic:ai is in the
    7d slice; its newest other item is the old one, outside the window."""
    c = Corpus(conn)
    _groups_build(c)
    c.mention(c.items["old"][0], "topic:ai-2")
    c.insight("nold", c.items["old"], 0, about=["topic:ai"])
    plain = graph.slice_recent(conn, window="7d", min_items=1, now=T0)
    assert [i["item_id"] for i in plain["items"]] == ["episode:1", "episode:2"]
    with _statements(conn) as seen:
        got = graph.slice_recent(conn, window="7d", min_items=1, now=T0, expand_groups=1)
    # decision 9: 1 + 2 + 4, and with expand_groups +1 to choose the items and +2 for their edges.
    assert len(seen) == 10
    assert [i["item_id"] for i in got["items"]] == ["episode:1", "episode:2", "episode:old"]
    assert got["params"]["expand_groups"] == 1
    assert got["applied_window"] == "7d"


def test_expand_groups_takes_each_groups_own_newest_item(conn: Connection) -> None:
    """Two groups in the slice, one item each: each group's newest is chosen from its own members,
    not from every group's."""
    c = Corpus(conn)
    _groups_build(c)
    c.mention(c.items["1"][0], "topic:other")
    c.mention(c.items["old"][0], "topic:ai-2")
    older, _ = c.item("older", T0 - timedelta(days=90))
    c.mention(older, "topic:other")
    got = graph.slice_recent(conn, window="7d", min_items=1, now=T0, expand_groups=1)
    assert [i["item_id"] for i in got["items"]] == [
        "episode:1",
        "episode:2",
        "episode:old",
        "episode:older",
    ]


def test_expand_groups_with_no_sibling_sends_no_item_steps_and_a_full_budget_none(
    conn: Connection,
) -> None:
    c = Corpus(conn)
    _groups_build(c)
    with _statements(conn) as seen:
        got = graph.slice_recent(conn, window="7d", min_items=1, now=T0, expand_groups=2)
    # The choosing statement finds no other item (item 3 is not served): no item steps.
    assert len(seen) == 8
    assert [i["item_id"] for i in got["items"]] == ["episode:1", "episode:2"]
    c.mention(c.items["old"][0], "topic:ai-2")
    full = graph.slice_recent(conn, window="7d", min_items=1, now=T0, budget=2, expand_groups=1)
    assert "episode:old" not in {i["item_id"] for i in full["items"]}
    with pytest.raises(ValueError, match="expand_groups"):
        graph.slice_recent(conn, now=T0, expand_groups=-1)


def test_groups_summary_is_every_group_of_a_kind_with_children_and_link_weights(
    conn: Connection,
) -> None:
    c = Corpus(conn)
    made = _groups_build(c)
    build = conn.execute(
        select(cluster_builds.c.build_id).where(cluster_builds.c.state == "published")
    ).scalar_one()
    other = _one(conn, cluster_ids, kind="super_theme", natural_id="sth:y")
    conn.execute(
        insert(clusters).values(build_id=build, cluster_id=other, kind="super_theme", label="Y")
    )
    a, b = sorted((made["sth:x"], other))
    conn.execute(
        insert(cluster_links).values(build_id=build, kind="super_theme", a=a, b=b, weight=4)
    )
    # A retired build's groups and links are not read.
    old = _one(conn, cluster_builds, rule_version="test/1", state="retired")
    gone = _one(conn, cluster_ids, kind="super_theme", natural_id="sth:old")
    conn.execute(
        insert(clusters).values(build_id=old, cluster_id=gone, kind="super_theme", label="old")
    )
    with _statements(conn) as seen:
        got = graph.groups_summary(conn, "super_theme")
    assert len(seen) == 1
    assert got["preset"] == "groups.summary" and got["params"] == {"kind": "super_theme"}
    assert got["corpus_revision"] == 0
    assert got["groups"] == [
        {
            "id": "sth:x",
            "kind": "super_theme",
            "label": "sth:x",
            "parent_id": None,
            "child_count": 1,
        },
        {"id": "sth:y", "kind": "super_theme", "label": "Y", "parent_id": None, "child_count": 0},
    ]
    assert got["links"] == [{"a": "sth:x", "b": "sth:y", "weight": 4}]
    storylines = graph.groups_summary(conn, "storyline")
    assert storylines["groups"] == [
        {
            "id": "thc:s",
            "kind": "storyline",
            "label": "thc:s",
            "parent_id": "sth:x",
            "child_count": 0,
        }
    ]
    assert storylines["links"] == []


def test_groups_summary_refuses_an_unknown_kind_and_is_empty_without_a_published_build(
    conn: Connection,
) -> None:
    Corpus(conn)
    with pytest.raises(graph.UnknownKind):
        graph.groups_summary(conn, "galaxy")
    empty = graph.groups_summary(conn, "theme")
    assert empty["groups"] == [] and empty["links"] == [] and empty["corpus_revision"] == 0


def test_the_preset_caches_probe_reads_the_revision_and_published_build() -> None:
    """server.graph_presets keys its cache on what this statement reads (ADR-183 decision 11)."""
    from podcast_scraper.server import graph_presets

    url = worker_database_url()
    assert url, "set PODCAST_TEST_DATABASE_URL (make test-db-up)"
    engine = create_engine(normalize_url(url))
    try:
        revision, build = graph_presets._probe(engine)
        with engine.connect() as conn:
            expected = conn.execute(
                text("SELECT revision FROM corpus.corpus_revision WHERE id = 1")
            ).scalar_one()
            published = conn.execute(
                text("SELECT build_id FROM corpus.projection_builds WHERE state = 'published'")
            ).scalar()
        assert (revision, build) == (expected, published)
    finally:
        engine.dispose()

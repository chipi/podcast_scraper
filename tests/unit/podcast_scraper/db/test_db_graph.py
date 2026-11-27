"""The graph engine's walk core (db.graph, ADR-183): presets over fixture rows, their statement
counts, merged entities, what is never served, and the transaction they run in."""

from __future__ import annotations

import contextlib
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

import pytest
from sqlalchemy import create_engine, event, insert, Table, text
from sqlalchemy.engine import Connection

from podcast_scraper.db import graph
from podcast_scraper.db.schema import (
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


def _ids(slice_: dict[str, Any]) -> set[str]:
    return {n["id"] for n in slice_["nodes"]}


def _edges(slice_: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {(e["type"], e["from"], e["to"]) for e in slice_["edges"]}


def test_slice_items_returns_the_items_graph_in_six_statements(conn: Connection) -> None:
    Corpus(conn)
    with _statements(conn) as seen:
        got = graph.slice_items(conn, ["episode:1", "episode:3", "episode:x"])
    # 1 (revision and seeds) + 2 steps + nodes, items, induced edges. No groups statement until
    # the clusters tables exist (s-clusters), so ADR-183's 1 + 2 + 4 is 1 + 2 + 3 here.
    assert len(seen) == 6
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
    # 1 (revision and seed) + 2 levels (the longest branch) + nodes, items, induced edges.
    assert len(seen) == 6
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
    assert len(seen) == 1 + 3 + 3
    assert _ids(got) == {"person:bob", "episode:1", "episode:2"}


def test_person_topics_reads_entity_types_for_the_kept_ends(conn: Connection) -> None:
    Corpus(conn)
    with _statements(conn) as seen:
        got = graph.slice_neighbourhood(conn, "person:bob", "person.topics")
    # 1 + 2 levels + the reached entities' types + 3.
    assert len(seen) == 1 + 2 + 1 + 3
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
    assert len(seen) == 6
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

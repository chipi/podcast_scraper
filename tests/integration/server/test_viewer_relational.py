"""Viewer API: GET /api/relational/* (RFC-094 §2, #882).

Requires ``fastapi`` (``pip install -e '.[dev]'``). The corpus graph is mocked with a
hand-built typed graph so the tests assert routing + edge-type semantics, not artifact
loading (``tests/unit/search`` covers the graph and query layer directly).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.search.corpus_graph import Node
from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


class _FakeGraph:
    def __init__(
        self,
        nodes: Dict[str, Tuple[str, Dict[str, object]]],
        edges: List[Tuple[str, str, str]],
    ) -> None:
        self._nodes = {nid: Node(id=nid, type=t, payload=p) for nid, (t, p) in nodes.items()}
        self._typed: Dict[str, List[Tuple[str, str]]] = {}
        for a, b, et in edges:
            self._typed.setdefault(a, []).append((b, et))
            self._typed.setdefault(b, []).append((a, et))

    def get_node(self, node_id: Optional[str]) -> Optional[Node]:
        return self._nodes.get(node_id) if node_id else None

    def typed_neighbors(self, node_id: str, edge_type: str) -> List[str]:
        return sorted({n for n, e in self._typed.get(node_id, ()) if e == edge_type})


def _fixture_graph() -> _FakeGraph:
    nodes: Dict[str, Tuple[str, Dict[str, object]]] = {
        "person:alice": ("person", {"name": "Alice"}),
        "person:bob": ("person", {"name": "Bob"}),
        "org:acme": ("org", {"name": "Acme"}),
        "topic:ai": ("topic", {"label": "AI"}),
        "insight:1": ("insight", {"text": "Alice on AI"}),
        "insight:2": ("insight", {"text": "Bob on AI"}),
        "episode:e1": ("episode", {}),
        "episode:e2": ("episode", {}),
        "podcast:show1": ("podcast", {"name": "Show One"}),
        "podcast:show2": ("podcast", {"name": "Show Two"}),
    }
    edges: List[Tuple[str, str, str]] = [
        ("person:alice", "insight:1", "STATES"),
        ("person:bob", "insight:2", "STATES"),
        ("insight:1", "topic:ai", "ABOUT"),
        ("insight:2", "topic:ai", "ABOUT"),
        ("insight:1", "org:acme", "MENTIONS"),
        ("episode:e1", "insight:1", "HAS_INSIGHT"),
        ("episode:e2", "insight:2", "HAS_INSIGHT"),
        ("podcast:show1", "episode:e1", "HAS_EPISODE"),
        ("podcast:show2", "episode:e2", "HAS_EPISODE"),
    ]
    return _FakeGraph(nodes, edges)


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    graph = _fixture_graph()
    monkeypatch.setattr(
        "podcast_scraper.server.routes.relational.get_corpus_graph",
        lambda *a, **k: graph,
    )
    return TestClient(create_app(tmp_path, static_dir=False))


def test_positions_returns_only_stated(client: TestClient) -> None:
    body = client.get("/api/relational/positions", params={"person": "person:alice"}).json()
    assert body["subject"] == "person:alice"
    assert [r["id"] for r in body["results"]] == ["insight:1"]


def test_positions_hybrid_reranks_by_relevance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # person:p stated insight:a + insight:b (structural order = sorted = a, b).
    graph = _FakeGraph(
        {
            "person:p": ("person", {"name": "Jane Doe"}),
            "insight:a": ("insight", {"text": "A"}),
            "insight:b": ("insight", {"text": "B"}),
        },
        [("person:p", "insight:a", "STATES"), ("person:p", "insight:b", "STATES")],
    )
    monkeypatch.setattr(
        "podcast_scraper.server.routes.relational.get_corpus_graph",
        lambda *a, **k: graph,
    )
    # Hybrid scores insight:b above insight:a → re-rank flips structural order.
    from podcast_scraper.search.corpus_search import CorpusSearchOutcome

    def fake_search(*_a: object, **_k: object) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(
            results=[
                {"doc_id": "x", "score": 0.9, "metadata": {"source_id": "insight:b"}},
                {"doc_id": "y", "score": 0.4, "metadata": {"source_id": "insight:a"}},
            ]
        )

    monkeypatch.setattr(
        "podcast_scraper.search.relational_capability.run_corpus_search", fake_search
    )
    client = TestClient(create_app(tmp_path, static_dir=False))
    body = client.get(
        "/api/relational/positions", params={"person": "person:p", "path": str(tmp_path)}
    ).json()
    assert [r["id"] for r in body["results"]] == ["insight:b", "insight:a"]


def test_positions_keeps_structural_order_when_search_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _FakeGraph(
        {
            "person:p": ("person", {"name": "Jane"}),
            "insight:a": ("insight", {}),
            "insight:b": ("insight", {}),
        },
        [("person:p", "insight:a", "STATES"), ("person:p", "insight:b", "STATES")],
    )
    monkeypatch.setattr(
        "podcast_scraper.server.routes.relational.get_corpus_graph",
        lambda *a, **k: graph,
    )

    def boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("no index")

    monkeypatch.setattr("podcast_scraper.search.relational_capability.run_corpus_search", boom)
    client = TestClient(create_app(tmp_path, static_dir=False))
    body = client.get(
        "/api/relational/positions", params={"person": "person:p", "path": str(tmp_path)}
    ).json()
    # Best-effort: search failure degrades to structural (sorted) order, not an error.
    assert [r["id"] for r in body["results"]] == ["insight:a", "insight:b"]


def test_entities_in_returns_mentioned_not_speaker(client: TestClient) -> None:
    body = client.get("/api/relational/entities-in", params={"insight": "insight:1"}).json()
    ids = [r["id"] for r in body["results"]]
    assert ids == ["org:acme"]
    assert "person:alice" not in ids


def test_insights_about_uses_mentions(client: TestClient) -> None:
    body = client.get("/api/relational/insights-about", params={"entity": "org:acme"}).json()
    assert [r["id"] for r in body["results"]] == ["insight:1"]


def test_episodes_walks_has_episode(client: TestClient) -> None:
    body = client.get("/api/relational/episodes", params={"podcast": "podcast:show1"}).json()
    assert [r["id"] for r in body["results"]] == ["episode:e1"]


def test_related_insights_via_shared_topic(client: TestClient) -> None:
    body = client.get("/api/relational/related-insights", params={"insight": "insight:1"}).json()
    assert [r["id"] for r in body["results"]] == ["insight:2"]


def test_topic_entities_returns_mentioned_entities(client: TestClient) -> None:
    body = client.get("/api/relational/topic-entities", params={"topic": "topic:ai"}).json()
    assert body["subject"] == "topic:ai"
    # topic:ai's insight:1 mentions org:acme.
    assert [r["id"] for r in body["results"]] == ["org:acme"]


def test_episode_insights_excludes_own(client: TestClient) -> None:
    body = client.get("/api/relational/episode-insights", params={"episode": "e1"}).json()
    assert body["subject"] == "e1"
    assert [r["id"] for r in body["results"]] == ["insight:2"]


def test_who_said_groups_by_person(client: TestClient) -> None:
    body = client.get("/api/relational/who-said", params={"topic": "topic:ai"}).json()
    assert set(body["groups"]) == {"person:alice", "person:bob"}
    assert [r["id"] for r in body["groups"]["person:alice"]] == ["insight:1"]


def test_cross_show_groups_by_show(client: TestClient) -> None:
    body = client.get("/api/relational/cross-show", params={"topic": "topic:ai"}).json()
    assert set(body["groups"]) == {"podcast:show1", "podcast:show2"}


def test_no_corpus_path_returns_error() -> None:
    app = create_app(None, static_dir=False)
    body = (
        TestClient(app).get("/api/relational/positions", params={"person": "person:alice"}).json()
    )
    assert body["error"] == "no_corpus_path"
    assert body["results"] == []


def test_missing_required_param_is_422(client: TestClient) -> None:
    assert client.get("/api/relational/positions").status_code == 422


# --- connectivity routes (#1055): topics / co-speakers / related-topics ---


def test_topics_returns_person_topics(client: TestClient) -> None:
    body = client.get("/api/relational/topics", params={"person": "person:alice"}).json()
    assert body["subject"] == "person:alice"
    assert [r["id"] for r in body["results"]] == ["topic:ai"]


def test_co_speakers_returns_people_on_shared_topics(client: TestClient) -> None:
    body = client.get("/api/relational/co-speakers", params={"person": "person:alice"}).json()
    assert [r["id"] for r in body["results"]] == ["person:bob"]


def test_related_topics_empty_for_lone_topic(client: TestClient) -> None:
    # the fixture has a single topic → no co-occurring topics.
    body = client.get("/api/relational/related-topics", params={"topic": "topic:ai"}).json()
    assert body["subject"] == "topic:ai"
    assert body["results"] == []


def test_related_topics_via_shared_insight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    graph = _FakeGraph(
        {
            "topic:a": ("topic", {"label": "A"}),
            "topic:b": ("topic", {"label": "B"}),
            "insight:1": ("insight", {}),
        },
        [("insight:1", "topic:a", "ABOUT"), ("insight:1", "topic:b", "ABOUT")],
    )
    monkeypatch.setattr(
        "podcast_scraper.server.routes.relational.get_corpus_graph",
        lambda *a, **k: graph,
    )
    client = TestClient(create_app(tmp_path, static_dir=False))
    body = client.get("/api/relational/related-topics", params={"topic": "topic:a"}).json()
    assert [r["id"] for r in body["results"]] == ["topic:b"]


def test_co_speakers_no_corpus_path_errors(tmp_path: Path) -> None:
    # create_app with no output_dir → graph unresolved → uniform error envelope.
    client = TestClient(create_app(None, static_dir=False))
    body = client.get("/api/relational/co-speakers", params={"person": "person:alice"}).json()
    assert body["error"] == "no_corpus_path"


# ---------------------------------------------------------------------------
# RFC-097 v3.0 typed MENTIONS family at the HTTP relational layer.
# The query layer treats MENTIONS / MENTIONS_PERSON / MENTIONS_ORG as one
# semantic family (see ``search.relational_queries._MENTIONS_FAMILY``).
# The relational HTTP endpoints inherit that — typed variants must
# surface through ``/api/relational/insights-about``, ``/entities-in``,
# and the topic-entities query.
# ---------------------------------------------------------------------------


def _typed_mentions_graph() -> _FakeGraph:
    nodes: Dict[str, Tuple[str, Dict[str, object]]] = {
        "person:ada": ("person", {"name": "Ada"}),
        "person:bob": ("person", {"name": "Bob"}),
        "org:acme": ("org", {"name": "Acme"}),
        "topic:ai": ("topic", {"label": "AI"}),
        "insight:1": ("insight", {"text": "Ada on AI at Acme"}),
        "insight:2": ("insight", {"text": "Bob on AI"}),
        "insight:3": ("insight", {"text": "legacy mention of Ada"}),
        "episode:e1": ("episode", {}),
    }
    edges: List[Tuple[str, str, str]] = [
        # Typed MENTIONS_PERSON: insight:1 mentions Ada
        ("insight:1", "person:ada", "MENTIONS_PERSON"),
        # Typed MENTIONS_ORG: insight:1 mentions Acme
        ("insight:1", "org:acme", "MENTIONS_ORG"),
        # Typed MENTIONS_PERSON: insight:2 mentions Bob
        ("insight:2", "person:bob", "MENTIONS_PERSON"),
        # Legacy generic MENTIONS (mid-migration corpus): insight:3 mentions Ada
        ("insight:3", "person:ada", "MENTIONS"),
        # Topic links
        ("insight:1", "topic:ai", "ABOUT"),
        ("insight:2", "topic:ai", "ABOUT"),
        ("insight:3", "topic:ai", "ABOUT"),
        ("episode:e1", "insight:1", "HAS_INSIGHT"),
    ]
    return _FakeGraph(nodes, edges)


@pytest.fixture()
def typed_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    graph = _typed_mentions_graph()
    monkeypatch.setattr(
        "podcast_scraper.server.routes.relational.get_corpus_graph",
        lambda *a, **k: graph,
    )
    return TestClient(create_app(tmp_path, static_dir=False))


def test_insights_about_surfaces_via_typed_mentions_person(typed_client: TestClient) -> None:
    """The MENTIONS family includes the typed PERSON variant — insights linked
    via ``MENTIONS_PERSON`` must surface alongside any legacy ``MENTIONS``.
    """
    body = typed_client.get(
        "/api/relational/insights-about", params={"entity": "person:ada"}
    ).json()
    ids = sorted(r["id"] for r in body["results"])
    # Both the typed-edge insight AND the legacy-edge insight surface.
    assert ids == ["insight:1", "insight:3"]


def test_insights_about_surfaces_via_typed_mentions_org(typed_client: TestClient) -> None:
    body = typed_client.get("/api/relational/insights-about", params={"entity": "org:acme"}).json()
    ids = [r["id"] for r in body["results"]]
    assert ids == ["insight:1"]


def test_entities_in_returns_typed_neighbours(typed_client: TestClient) -> None:
    """``/api/relational/entities-in`` walks MENTIONS_PERSON + MENTIONS_ORG
    and returns both the Person and the Organization neighbour of insight:1.
    """
    body = typed_client.get("/api/relational/entities-in", params={"insight": "insight:1"}).json()
    ids = sorted(r["id"] for r in body["results"])
    assert ids == ["org:acme", "person:ada"]


def test_topic_entities_ranks_across_typed_family(typed_client: TestClient) -> None:
    """``/api/relational/topic-entities`` walks ABOUT → MENTIONS-family.
    With insight:1 mentioning Ada + Acme, insight:2 mentioning Bob, insight:3
    mentioning Ada (legacy), Ada has 2 mentions and Acme + Bob each have 1.
    """
    body = typed_client.get("/api/relational/topic-entities", params={"topic": "topic:ai"}).json()
    ids = [r["id"] for r in body["results"]]
    # Most-mentioned first (Ada twice across the typed+legacy family). The
    # ranking is what proves the family count actually CROSSED the
    # typed+legacy boundary — without family awareness Ada would tie with
    # Acme + Bob (one mention each).
    assert len(ids) == 3, f"expected 3 entity rows, got {ids}"
    assert ids[0] == "person:ada", (
        f"family-count traversal regression — Ada (2 mentions across "
        f"typed+legacy) must rank first; got {ids}"
    )
    assert set(ids[1:]) == {"org:acme", "person:bob"}

"""The graph API's catalogue (ADR-183 decision 11) and the health fields that go with it: the
catalogue is the engine's own constants, and health's corpus revision is read off the event loop."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from podcast_scraper.db import graph
from podcast_scraper.server.app import create_app
from podcast_scraper.server.routes import health

pytestmark = [pytest.mark.unit]


def test_the_catalogue_lists_every_edge_path_preset_and_bound_of_the_engine() -> None:
    got = graph.catalogue()
    assert got["format"] == graph.FORMAT
    assert {e["type"] for e in got["edge_types"]} == {*graph.EDGES, graph.IN_GROUP}
    about = next(e for e in got["edge_types"] if e["type"] == "ABOUT")
    assert about == {
        "type": "ABOUT",
        "class": "descriptive",
        "from": ["Insight"],
        "to": ["Object", "Organization", "Person", "Topic"],
        "item_scoped": True,
    }
    assert set(got["named_paths"]) == set(graph.PATHS)
    assert got["named_paths"]["topic.people"]["seeds"]["Topic"] == [
        [[["ABOUT", "in"]], [["STATED_BY", "out"]]]
    ]
    assert set(got["presets"]) == set(graph.PRESETS)
    assert got["bounds"]["budget"] == {"default": 1_500, "max": 3_000}
    assert got["bounds"]["exclude_items"] == 2_000 and got["bounds"]["node_seeds"] == 50


def test_the_catalogue_route_serves_it_and_health_says_the_graph_api_is_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "appdata"))
    monkeypatch.setattr(health, "_corpus_revision", lambda: 7)
    client = TestClient(create_app(output_dir=tmp_path))
    resp = client.get("/api/graph/catalogue")
    assert resp.status_code == 200 and resp.json() == graph.catalogue()
    body = client.get("/api/health").json()
    assert body["graph_api"] is True and body["corpus_revision"] == 7


def test_the_player_serves_no_graph_api_and_health_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "appdata"))
    monkeypatch.setenv("PODCAST_SERVE_APP_ONLY", "1")
    monkeypatch.setattr(health, "_corpus_revision", lambda: 7)
    client = TestClient(create_app(output_dir=tmp_path))
    assert client.get("/api/graph/catalogue").status_code == 404
    body = client.get("/api/health").json()
    assert body["graph_api"] is False and body["corpus_revision"] is None


def test_the_revision_hold_answers_at_once_and_reads_in_the_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    reads: list[int] = []

    def read() -> int:
        release.wait(5)
        reads.append(1)
        return 41 + len(reads)

    monkeypatch.setattr(health, "_read_corpus_revision", read)
    hold = health._RevisionHold()
    # The first call does not wait for the database; a second call while it reads starts no other.
    assert hold() is None and hold() is None
    release.set()
    for thread in threading.enumerate():
        if thread.name == "health-revision":
            thread.join(5)
    assert hold() == 42 and reads == [1]

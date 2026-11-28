"""The graph presets' routes (ADR-183 decisions 5, 9, 11): GET with ETag and 304, POST with the
ETag and no conditional read, 422 naming the field, 503 past the deadline. The engine's reads are
faked; the engine's own refusals run for real (they come before any statement)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest
from starlette.testclient import TestClient

from podcast_scraper import perf_cache
from podcast_scraper.db import engines, graph
from podcast_scraper.server import graph_presets
from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.unit]


class _Reached(Exception):
    """The engine reached its first statement: its parameter checks have all passed."""


class _NoConnection:
    def __getattr__(self, name: str) -> Any:
        raise _Reached(name)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    perf_cache.clear(graph_presets.NAMESPACE)
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "appdata"))
    monkeypatch.setattr(engines, "corpus_ro_engine", lambda: object())
    monkeypatch.setattr(graph_presets, "_probe", lambda engine: (3, 9))
    calls: list[dict[str, Any]] = []

    def read(engine: Any, fn: Any) -> dict[str, Any]:
        calls.append({})
        # Let the engine refuse bad parameters: its checks run before any statement, and the
        # first use of the connection ends the read, so nothing else is swallowed.
        try:
            fn(_NoConnection())
        except _Reached:
            pass
        return {"corpus_revision": 3, "graph_build_id": 9, "nodes": [], "edges": []}

    monkeypatch.setattr(graph, "read_slice", read)
    yield TestClient(create_app(output_dir=tmp_path))
    perf_cache.clear(graph_presets.NAMESPACE)


def test_get_serves_a_preset_with_its_etag_and_304_when_it_matches(client: TestClient) -> None:
    got = client.get("/api/graph/presets/slice.recent?window=30d&content_types=podcast")
    assert got.status_code == 200 and got.json()["corpus_revision"] == 3
    etag = got.headers["etag"]
    assert etag.startswith('"3:')
    again = client.get(
        "/api/graph/presets/slice.recent?window=30d&content_types=podcast",
        headers={"If-None-Match": etag},
    )
    assert again.status_code == 304 and again.headers["etag"] == etag


def test_post_carries_the_etag_and_never_answers_304(client: TestClient) -> None:
    body = {"items": ["episode:1", "episode:2"], "exclude_items": ["episode:3"]}
    got = client.post("/api/graph/presets/slice.items", json=body)
    assert got.status_code == 200 and got.headers["etag"].startswith('"3:')
    again = client.post(
        "/api/graph/presets/slice.items", json=body, headers={"If-None-Match": got.headers["etag"]}
    )
    assert again.status_code == 200


@pytest.mark.parametrize(
    ("url", "loc"),
    [
        ("/api/graph/presets/slice.everything", ["path", "name"]),
        ("/api/graph/presets/slice.recent?bogus=1", ["query", "bogus"]),
        ("/api/graph/presets/slice.recent?budget=many", ["query", "budget"]),
        ("/api/graph/presets/slice.recent?budget=9000", ["query", "budget"]),
        ("/api/graph/presets/slice.recent?window=1y", ["query", "window"]),
        ("/api/graph/presets/slice.recent?window=7d&window=30d", ["query", "window"]),
        ("/api/graph/presets/groups.summary?kind=galaxy", ["query", "kind"]),
        ("/api/graph/presets/groups.summary?kind=theme&budget=5", ["query", "budget"]),
        ("/api/graph/presets/groups.summary", ["query", "kind"]),
        ("/api/graph/presets/slice.items", ["query", "items"]),
        ("/api/graph/presets/slice.neighbourhood?node=topic:ai", ["query", "path"]),
        ("/api/graph/presets/slice.group", ["query", "group"]),
    ],
)
def test_a_bad_parameter_is_422_naming_it(client: TestClient, url: str, loc: list[str]) -> None:
    got = client.get(url)
    assert got.status_code == 422
    assert got.json()["detail"][0]["loc"] == loc


def test_a_read_past_the_deadline_is_503_retry_narrower(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def late(engine: Any, fn: Any) -> Any:
        raise graph.GraphTimeout("the graph read ran past 5000 ms")

    monkeypatch.setattr(graph, "read_slice", late)
    got = client.get("/api/graph/presets/slice.recent")
    assert got.status_code == 503 and got.json()["retry"] == "narrower"
    assert "retry-after" not in got.headers

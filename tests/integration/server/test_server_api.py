"""Integration tests for the FastAPI server (RFC-062).

These tests exercise the wired application with real filesystem artifacts
(no mocking of route internals). Skipped when ``fastapi`` is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """Minimal corpus with GI + KG artifacts in a metadata subdirectory."""
    meta = tmp_path / "metadata"
    meta.mkdir()

    gi_payload = {
        "grounded_insights": {
            "version": "1.0",
            "episode_id": "ep1",
            "insights": [
                {
                    "id": "n1",
                    "text": "Climate change accelerates",
                    "topic": "climate",
                    "confidence": 0.9,
                    "grounded": True,
                    "supporting_quotes": ["q1"],
                }
            ],
            "quotes": [
                {
                    "id": "q1",
                    "text": "The planet is warming faster than expected.",
                    "speaker": "Dr. Smith",
                    "start_time": 10.0,
                    "end_time": 15.0,
                }
            ],
            "edges": [{"source": "n1", "target": "q1", "type": "supported_by"}],
        }
    }
    (meta / "ep1.gi.json").write_text(json.dumps(gi_payload), encoding="utf-8")

    kg_payload = {
        "knowledge_graph": {
            "version": "1.0",
            "episode_id": "ep1",
            "nodes": [{"id": "t1", "label": "climate", "type": "topic"}],
            "edges": [],
        }
    }
    (meta / "ep1.kg.json").write_text(json.dumps(kg_payload), encoding="utf-8")

    return tmp_path


@pytest.fixture()
def client(corpus: Path) -> TestClient:
    """TestClient wired to a real corpus directory (no mocks)."""
    app = create_app(corpus, static_dir=False)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body.get("artifacts_api") is True
        assert body.get("search_api") is True
        assert body.get("explore_api") is True
        assert body.get("index_routes_api") is True
        assert body.get("corpus_metrics_api") is True
        assert body.get("corpus_library_api") is True
        assert body.get("corpus_digest_api") is True
        assert body.get("corpus_binary_api") is True


# ---------------------------------------------------------------------------
# Artifacts — real filesystem scan (no monkeypatch)
# ---------------------------------------------------------------------------


class TestArtifacts:
    def test_list_discovers_real_files(self, client: TestClient, corpus: Path) -> None:
        resp = client.get("/api/artifacts", params={"path": str(corpus)})
        assert resp.status_code == 200
        body = resp.json()
        names = sorted(a["name"] for a in body["artifacts"])
        assert names == ["ep1.gi.json", "ep1.kg.json"]

    def test_load_gi_artifact(self, client: TestClient, corpus: Path) -> None:
        resp = client.get(
            "/api/artifacts/metadata/ep1.gi.json",
            params={"path": str(corpus)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "grounded_insights" in data
        assert data["grounded_insights"]["episode_id"] == "ep1"

    def test_load_kg_artifact(self, client: TestClient, corpus: Path) -> None:
        resp = client.get(
            "/api/artifacts/metadata/ep1.kg.json",
            params={"path": str(corpus)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "knowledge_graph" in data

    def test_traversal_blocked(self, client: TestClient, corpus: Path) -> None:
        resp = client.get(
            "/api/artifacts/metadata%2F..%2F..%2Fetc%2Fpasswd",
            params={"path": str(corpus)},
        )
        assert resp.status_code in (400, 404)

    def test_missing_artifact_404(self, client: TestClient, corpus: Path) -> None:
        resp = client.get(
            "/api/artifacts/metadata/nope.gi.json",
            params={"path": str(corpus)},
        )
        assert resp.status_code == 404

    def test_bad_corpus_path_400(self, client: TestClient, corpus: Path) -> None:
        resp = client.get(
            "/api/artifacts",
            params={"path": str(corpus / "nonexistent")},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Index stats — no FAISS index present
# ---------------------------------------------------------------------------


class TestIndexStats:
    def test_no_index_reports_unavailable(self, client: TestClient, corpus: Path) -> None:
        resp = client.get("/api/index/stats", params={"path": str(corpus)})
        assert resp.status_code == 200
        body = resp.json()
        assert body["available"] is False
        assert body["reason"] in ("no_index", "faiss_unavailable")

    def test_uses_app_state_fallback(self, client: TestClient) -> None:
        resp = client.get("/api/index/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["available"] is False


# ---------------------------------------------------------------------------
# Search — no index, so expect graceful error
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_without_index_returns_error(self, client: TestClient, corpus: Path) -> None:
        resp = client.get(
            "/api/search",
            params={"q": "climate", "path": str(corpus)},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "climate"
        assert body["error"] is not None or body["results"] == []

    def test_search_no_corpus_path(self) -> None:
        app = create_app(None, static_dir=False)
        c = TestClient(app)
        resp = c.get("/api/search", params={"q": "hello"})
        assert resp.status_code == 200
        assert resp.json()["error"] == "no_corpus_path"


# ---------------------------------------------------------------------------
# Explore — real artifacts on disk
# ---------------------------------------------------------------------------


class TestExplore:
    def test_explore_no_corpus_path(self) -> None:
        app = create_app(None, static_dir=False)
        c = TestClient(app)
        resp = c.get("/api/explore")
        assert resp.status_code == 200
        assert resp.json()["error"] == "no_corpus_path"

    def test_explore_filter_with_real_artifacts(self, client: TestClient, corpus: Path) -> None:
        resp = client.get(
            "/api/explore",
            params={"path": str(corpus), "topic": "climate"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["kind"] == "explore"

    def test_explore_nl_no_artifacts_dir(self, client: TestClient) -> None:
        app = create_app(Path("/tmp/nonexistent_xyz_test"), static_dir=False)
        c = TestClient(app)
        resp = c.get(
            "/api/explore",
            params={"q": "What about climate?", "path": "/tmp/nonexistent_xyz_test"},
        )
        assert resp.status_code in (200, 400)


# ---------------------------------------------------------------------------
# App factory edge cases
# ---------------------------------------------------------------------------


class TestAppFactory:
    def test_static_dir_false_no_mount(self, corpus: Path) -> None:
        app = create_app(corpus, static_dir=False)
        route_names = [r.name for r in app.routes]
        assert "viewer" not in route_names

    def test_output_dir_stored_on_state(self, corpus: Path) -> None:
        app = create_app(corpus, static_dir=False)
        assert app.state.output_dir == corpus.resolve()

    def test_none_output_dir_accepted(self) -> None:
        app = create_app(None, static_dir=False)
        assert app.state.output_dir is None

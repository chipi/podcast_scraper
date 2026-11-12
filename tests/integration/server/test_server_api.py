"""Integration tests for the FastAPI server.

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

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


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

    bridge_payload = {
        "schema_version": "3.0",
        "episode_id": "ep1",
        "emitted_at": "2026-04-13T00:00:00Z",
        "identities": [],
    }
    (meta / "ep1.bridge.json").write_text(json.dumps(bridge_payload), encoding="utf-8")

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

    def test_enriched_search_available_flips_true_when_topic_similarity_present(
        self, client: TestClient, corpus: Path
    ) -> None:
        """The probe is `enrichments/topic_similarity.json` (the exact
        file the shipped `query_topic_relatedness` enricher reads). Its
        presence should flip the capability to True so the S5 Enriched
        chip auto-adopts and the hero renders."""
        enrich_dir = corpus / "enrichments"
        enrich_dir.mkdir(parents=True, exist_ok=True)
        (enrich_dir / "topic_similarity.json").write_text('{"topics": {}}', encoding="utf-8")
        resp = client.get("/api/health", params={"path": str(corpus)})
        assert resp.status_code == 200
        assert resp.json().get("enriched_search_available") is True

    def test_enriched_search_available_stays_false_without_the_file(
        self, client: TestClient, corpus: Path
    ) -> None:
        """Absent file → probe returns False (indistinguishable to the
        client from 'no enrichment configured' — matches the previous
        default behaviour)."""
        # Make sure the file does not exist for this test.
        candidate = corpus / "enrichments" / "topic_similarity.json"
        if candidate.exists():
            candidate.unlink()
        resp = client.get("/api/health", params={"path": str(corpus)})
        assert resp.status_code == 200
        assert resp.json().get("enriched_search_available") is False


# ---------------------------------------------------------------------------
# Artifacts — real filesystem scan (no monkeypatch)
# ---------------------------------------------------------------------------


class TestArtifacts:

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
# Index stats — no search index present
# ---------------------------------------------------------------------------


class TestIndexStats:
    def test_no_index_reports_unavailable(self, client: TestClient, corpus: Path) -> None:
        resp = client.get("/api/index/stats", params={"path": str(corpus)})
        assert resp.status_code == 200
        body = resp.json()
        assert body["available"] is False
        assert body["reason"] == "no_index"

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


# ---------------------------------------------------------------------------
# App factory edge cases
# ---------------------------------------------------------------------------


class TestAppFactory:
    def test_static_dir_false_no_mount(self, corpus: Path) -> None:
        app = create_app(corpus, static_dir=False)
        # Newer starlette wraps `include_router` results in `_IncludedRouter`,
        # which doesn't expose `.name`. Mounts (the case we care about — the
        # viewer static mount) still have `.name`. `getattr` keeps the test
        # forward-compatible without changing what it asserts.
        route_names = [getattr(r, "name", None) for r in app.routes]
        assert "viewer" not in route_names

    def test_output_dir_stored_on_state(self, corpus: Path) -> None:
        app = create_app(corpus, static_dir=False)
        assert app.state.output_dir == corpus.resolve()

    def test_none_output_dir_accepted(self) -> None:
        app = create_app(None, static_dir=False)
        assert app.state.output_dir is None

"""Pytest E2E: FastAPI viewer app (``create_app`` + ``TestClient``).

Exercises ``podcast_scraper.server`` in the E2E coverage job (full-package denominator).
Complements integration tests under ``tests/integration/server/``; kept here so CI E2E
coverage includes the HTTP layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from urllib.parse import quote

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.search.corpus_search import CorpusSearchOutcome
from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.e2e, pytest.mark.critical_path]


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """Minimal corpus with GI + KG + bridge under ``metadata/``."""
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
        "schema_version": "1.0",
        "episode_id": "ep1",
        "emitted_at": "2026-04-13T00:00:00Z",
        "identities": [],
    }
    (meta / "ep1.bridge.json").write_text(json.dumps(bridge_payload), encoding="utf-8")

    return tmp_path


@pytest.fixture()
def client(corpus: Path) -> TestClient:
    return TestClient(create_app(corpus, static_dir=False))


def _cil_bundle(directory: Path, stem: str, *, episode_id: str, person: str, topic: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    bridge = {
        "schema_version": "1.0",
        "episode_id": episode_id,
        "identities": [
            {
                "id": person,
                "type": "person",
                "sources": {"gi": True, "kg": True},
                "display_name": "P",
                "aliases": [],
            },
            {
                "id": topic,
                "type": "topic",
                "sources": {"gi": True, "kg": True},
                "display_name": "T",
                "aliases": [],
            },
        ],
    }
    gi = {
        "episode_id": episode_id,
        "nodes": [
            {
                "id": "insight-1",
                "type": "Insight",
                "properties": {"text": "Hello", "insight_type": "claim", "position_hint": 0.1},
            },
            {"id": "quote-1", "type": "Quote", "properties": {"text": "said"}},
        ],
        "edges": [
            {"type": "SPOKEN_BY", "from": "quote-1", "to": person},
            {"type": "SUPPORTED_BY", "from": "insight-1", "to": "quote-1"},
            {"type": "ABOUT", "from": "insight-1", "to": topic},
        ],
    }
    kg = {
        "nodes": [
            {
                "id": "ep1",
                "type": "Episode",
                "properties": {"publish_date": "2024-05-01"},
            }
        ],
        "edges": [],
    }
    (directory / f"{stem}.bridge.json").write_text(json.dumps(bridge), encoding="utf-8")
    (directory / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")
    (directory / f"{stem}.kg.json").write_text(json.dumps(kg), encoding="utf-8")


@pytest.fixture()
def cil_corpus(tmp_path: Path) -> Path:
    meta = tmp_path / "metadata"
    _cil_bundle(
        meta,
        "ep1",
        episode_id="episode:one",
        person="person:pat",
        topic="topic:science",
    )
    return tmp_path


@pytest.fixture()
def cil_client(cil_corpus: Path) -> TestClient:
    return TestClient(create_app(cil_corpus, static_dir=False))


class TestServerViewerHealthArtifacts:
    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body.get("cil_queries_api") is True

    def test_artifacts_lists_files(self, client: TestClient, corpus: Path) -> None:
        resp = client.get("/api/artifacts", params={"path": str(corpus)})
        assert resp.status_code == 200
        names = sorted(a["name"] for a in resp.json()["artifacts"])
        assert names == ["ep1.bridge.json", "ep1.gi.json", "ep1.kg.json"]

    def test_index_stats_no_faiss(self, client: TestClient, corpus: Path) -> None:
        resp = client.get("/api/index/stats", params={"path": str(corpus)})
        assert resp.status_code == 200
        body = resp.json()
        assert body["available"] is False


class TestServerViewerCil:
    def test_person_positions(self, cil_client: TestClient, cil_corpus: Path) -> None:
        pid = quote("person:pat", safe="")
        resp = cil_client.get(
            f"/api/persons/{pid}/positions",
            params={"topic": "topic:science", "path": str(cil_corpus)},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["person_id"] == "person:pat"
        assert len(body["episodes"]) == 1


class TestServerViewerCorpusLibrary:
    def test_corpus_feeds_minimal_metadata(self, tmp_path: Path) -> None:
        """``GET /api/corpus/feeds`` with one flat metadata row (RFC-067)."""
        meta = tmp_path / "metadata"
        meta.mkdir()
        episode_doc = {
            "feed": {"feed_id": "e2efeed", "title": "E2E Show"},
            "episode": {
                "episode_id": "ep-e2e",
                "title": "Hello",
                "published_date": "2024-03-10T00:00:00",
            },
            "summary": {
                "title": "Sum",
                "bullets": ["a"],
                "short_summary": "Short.",
            },
        }
        (meta / "row.metadata.json").write_text(
            json.dumps(episode_doc),
            encoding="utf-8",
        )
        (meta / "row.gi.json").write_text("{}", encoding="utf-8")

        app = create_app(tmp_path, static_dir=False)
        http_client = TestClient(app)
        resp = http_client.get("/api/corpus/feeds", params={"path": str(tmp_path)})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["feeds"]) == 1
        assert body["feeds"][0]["feed_id"] == "e2efeed"

        er = http_client.get(
            "/api/corpus/episodes",
            params={"path": str(tmp_path), "limit": 5},
        )
        assert er.status_code == 200
        items = er.json()["items"]
        assert len(items) == 1
        assert items[0]["episode_title"] == "Hello"


class TestServerViewerExplorePatched:
    def test_explore_filter_mocked(
        self,
        corpus: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from podcast_scraper.gi.explore import build_explore_output

        def fake_uc5(root: Path, **kwargs: Any) -> Any:
            assert root == corpus.resolve()
            assert kwargs.get("topic") == "climate"
            return build_explore_output(
                [],
                2,
                topic="climate",
                speaker_filter=None,
                topics=[],
            )

        monkeypatch.setattr(
            "podcast_scraper.server.routes.explore.run_uc5_insight_explorer",
            fake_uc5,
        )
        http_client = TestClient(create_app(corpus, static_dir=False))
        resp = http_client.get(
            "/api/explore",
            params={"path": str(corpus), "topic": "climate"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["kind"] == "explore"
        assert body["error"] is None


class TestServerViewerSearchPatched:
    def test_search_mocked_corpus_search(
        self,
        corpus: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: Dict[str, Any] = {}

        def fake_run(
            output_dir: Path,
            query: str,
            **kwargs: Any,
        ) -> CorpusSearchOutcome:
            captured["output_dir"] = output_dir
            captured["query"] = query
            return CorpusSearchOutcome(
                results=[
                    {
                        "doc_id": "insight:ep1:n1",
                        "score": 0.88,
                        "metadata": {"doc_type": "insight", "episode_id": "ep1"},
                        "text": "hit",
                    }
                ],
                lift_stats={"transcript_hits_returned": 0, "lift_applied": 0},
            )

        monkeypatch.setattr(
            "podcast_scraper.server.routes.search.run_corpus_search",
            fake_run,
        )
        app = create_app(corpus, static_dir=False)
        http_client = TestClient(app)
        resp = http_client.get(
            "/api/search",
            params={"q": "climate", "path": str(corpus), "top_k": "3"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["error"] is None
        assert len(body["results"]) == 1
        assert captured["query"] == "climate"


class TestServerViewerCorpusTextFile:
    """``GET /api/corpus/text-file`` — inline transcripts and JSON (viewer parity)."""

    def test_serves_txt_under_corpus_root(self, tmp_path: Path) -> None:
        notes = tmp_path / "notes"
        notes.mkdir()
        (notes / "hello.txt").write_text("transcript body", encoding="utf-8")
        http_client = TestClient(create_app(tmp_path, static_dir=False))
        resp = http_client.get(
            "/api/corpus/text-file",
            params={"path": str(tmp_path), "relpath": "notes/hello.txt"},
        )
        assert resp.status_code == 200
        assert resp.text == "transcript body"
        ctype = (resp.headers.get("content-type") or "").lower()
        assert "text/plain" in ctype

    def test_cleaned_txt_fallback_when_raw_missing(self, tmp_path: Path) -> None:
        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        (transcripts / "ep.cleaned.txt").write_text("cleaned-only", encoding="utf-8")
        http_client = TestClient(create_app(tmp_path, static_dir=False))
        resp = http_client.get(
            "/api/corpus/text-file",
            params={"path": str(tmp_path), "relpath": "transcripts/ep.txt"},
        )
        assert resp.status_code == 200
        assert resp.text == "cleaned-only"

    def test_json_uses_json_media_type(self, tmp_path: Path) -> None:
        (tmp_path / "sidecar.json").write_text('{"a": 1}', encoding="utf-8")
        http_client = TestClient(create_app(tmp_path, static_dir=False))
        resp = http_client.get(
            "/api/corpus/text-file",
            params={"path": str(tmp_path), "relpath": "sidecar.json"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"a": 1}
        ctype = (resp.headers.get("content-type") or "").lower()
        assert "application/json" in ctype

    def test_disallowed_suffix_returns_400(self, tmp_path: Path) -> None:
        http_client = TestClient(create_app(tmp_path, static_dir=False))
        resp = http_client.get(
            "/api/corpus/text-file",
            params={"path": str(tmp_path), "relpath": "evil.bin"},
        )
        assert resp.status_code == 400

    def test_missing_file_returns_404(self, tmp_path: Path) -> None:
        http_client = TestClient(create_app(tmp_path, static_dir=False))
        resp = http_client.get(
            "/api/corpus/text-file",
            params={"path": str(tmp_path), "relpath": "missing.txt"},
        )
        assert resp.status_code == 404

    def test_empty_relpath_returns_400(self, tmp_path: Path) -> None:
        http_client = TestClient(create_app(tmp_path, static_dir=False))
        resp = http_client.get(
            "/api/corpus/text-file",
            params={"path": str(tmp_path), "relpath": ""},
        )
        assert resp.status_code == 400

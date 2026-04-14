"""Viewer API: GET /api/search (RFC-062).

Requires ``fastapi`` (``pip install -e '.[server]'``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast, Dict, List

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.search.corpus_search import CorpusSearchOutcome
from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


def test_search_no_corpus_path() -> None:
    app = create_app(None, static_dir=False)
    client = TestClient(app)
    response = client.get("/api/search", params={"q": "hello"})
    assert response.status_code == 200
    body = response.json()
    assert body["error"] == "no_corpus_path"
    assert body["results"] == []


def test_search_returns_results_when_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(
        output_dir: Path,
        query: str,
        **kwargs: Any,
    ) -> CorpusSearchOutcome:
        assert output_dir == tmp_path.resolve()
        assert query == "climate"
        return CorpusSearchOutcome(
            results=[
                {
                    "doc_id": "insight:ep1:n1",
                    "score": 0.91,
                    "metadata": {"doc_type": "insight", "episode_id": "ep1"},
                    "text": "hello",
                }
            ],
            lift_stats={
                "transcript_hits_returned": 0,
                "lift_applied": 0,
            },
        )

    monkeypatch.setattr(
        "podcast_scraper.server.routes.search.run_corpus_search",
        fake_run,
    )

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/search",
        params={"q": "climate", "path": str(tmp_path), "top_k": "5"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["error"] is None
    assert len(body["results"]) == 1
    assert body["results"][0]["doc_id"] == "insight:ep1:n1"
    assert body["results"][0]["score"] == pytest.approx(0.91)
    assert body.get("lift_stats") == {
        "transcript_hits_returned": 0,
        "lift_applied": 0,
    }


def test_search_type_query_params(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_run(
        output_dir: Path,
        query: str,
        *,
        doc_types: List[str] | None = None,
        **kwargs: Any,
    ) -> CorpusSearchOutcome:
        captured["doc_types"] = doc_types
        return CorpusSearchOutcome(results=[])

    monkeypatch.setattr(
        "podcast_scraper.server.routes.search.run_corpus_search",
        fake_run,
    )

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/search",
        params=[
            ("q", "q"),
            ("path", str(tmp_path)),
            ("type", "insight,quote"),
            ("type", "summary"),
        ],
    )
    assert response.status_code == 200
    assert captured.get("doc_types") == ["insight", "quote", "summary"]


def test_search_dedupe_kg_surfaces_query_param(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: Dict[str, Any] = {}

    def fake_run(
        output_dir: Path,
        query: str,
        *,
        dedupe_kg_surfaces: bool = True,
        **kwargs: Any,
    ) -> CorpusSearchOutcome:
        captured["dedupe_kg_surfaces"] = dedupe_kg_surfaces
        return CorpusSearchOutcome(results=[])

    monkeypatch.setattr(
        "podcast_scraper.server.routes.search.run_corpus_search",
        fake_run,
    )

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r1 = client.get(
        "/api/search",
        params={"q": "q", "path": str(tmp_path)},
    )
    assert r1.status_code == 200
    assert captured.get("dedupe_kg_surfaces") is True

    r2 = client.get(
        "/api/search",
        params={"q": "q", "path": str(tmp_path), "dedupe_kg_surfaces": "false"},
    )
    assert r2.status_code == 200
    assert captured.get("dedupe_kg_surfaces") is False


def test_search_maps_outcome_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_a: Any, **_k: Any) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(error="no_index", detail="/tmp/x")

    monkeypatch.setattr(
        "podcast_scraper.server.routes.search.run_corpus_search",
        fake_run,
    )

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/search",
        params={"q": "hello", "path": str(tmp_path)},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["error"] == "no_index"
    assert body["detail"] == "/tmp/x"
    assert body["results"] == []
    assert body.get("lift_stats") is None


def test_search_lift_stats_reflects_transcript_and_lift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(
        output_dir: Path,
        query: str,
        **kwargs: Any,
    ) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(
            results=[
                {
                    "doc_id": "t1",
                    "score": 0.5,
                    "metadata": {"doc_type": "transcript", "episode_id": "e1"},
                    "text": "chunk",
                    "lifted": {"insight": {"id": "i1"}},
                },
                {
                    "doc_id": "t2",
                    "score": 0.4,
                    "metadata": {"doc_type": "transcript", "episode_id": "e1"},
                    "text": "chunk2",
                },
            ],
            lift_stats={"transcript_hits_returned": 2, "lift_applied": 1},
        )

    monkeypatch.setattr(
        "podcast_scraper.server.routes.search.run_corpus_search",
        fake_run,
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/search",
        params={"q": "q", "path": str(tmp_path)},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["lift_stats"] == {
        "transcript_hits_returned": 2,
        "lift_applied": 1,
    }


def test_search_lift_stats_invalid_dropped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(
        *_a: Any,
        **_k: Any,
    ) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(
            results=[],
            lift_stats=cast(
                Dict[str, int],
                {"transcript_hits_returned": "bad", "lift_applied": 1},
            ),
        )

    monkeypatch.setattr(
        "podcast_scraper.server.routes.search.run_corpus_search",
        fake_run,
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/search",
        params={"q": "q", "path": str(tmp_path)},
    )
    assert response.status_code == 200
    assert response.json().get("lift_stats") is None

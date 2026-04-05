"""M5 viewer API: GET /api/explore (RFC-062). Skipped when ``fastapi`` is missing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.gi.explore import build_explore_output
from podcast_scraper.server.app import create_app


def test_explore_no_corpus_path() -> None:
    app = create_app(None, static_dir=False)
    client = TestClient(app)
    response = client.get("/api/explore", params={"topic": "x"})
    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "explore"
    assert body["error"] == "no_corpus_path"


def test_explore_filter_mode_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_uc5(root: Path, **kwargs: Any) -> Any:
        assert root == tmp_path.resolve()
        assert kwargs.get("topic") == "climate"
        return build_explore_output([], 4, topic="climate", speaker_filter=None, topics=[])

    monkeypatch.setattr(
        "podcast_scraper.server.routes.explore.run_uc5_insight_explorer",
        fake_uc5,
    )

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/explore",
        params={"path": str(tmp_path), "topic": "climate"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "explore"
    assert body["error"] is None
    assert body["data"] is not None
    assert body["data"]["episodes_searched"] == 4
    assert body["data"]["insights"] == []


def test_explore_natural_language_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_qa(root: Path, question: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "question": question,
            "answer": {"insights": [], "summary": {"insight_count": 0}},
            "explanation": "matched",
        }

    monkeypatch.setattr(
        "podcast_scraper.server.routes.explore.scan_artifact_paths",
        lambda _p: [Path("dummy.gi.json")],
    )
    monkeypatch.setattr(
        "podcast_scraper.server.routes.explore.run_uc4_semantic_qa",
        fake_qa,
    )

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/explore",
        params={"path": str(tmp_path), "q": "What insights about tea?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "natural_language"
    assert body["error"] is None
    assert body["question"] == "What insights about tea?"
    assert body["explanation"] == "matched"


def test_explore_nl_no_pattern(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "podcast_scraper.server.routes.explore.scan_artifact_paths",
        lambda _p: [Path("dummy.gi.json")],
    )
    monkeypatch.setattr(
        "podcast_scraper.server.routes.explore.run_uc4_semantic_qa",
        lambda *a, **k: None,
    )

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/explore",
        params={"path": str(tmp_path), "question": "random gibberish xyz"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "natural_language"
    assert body["error"] == "no_pattern_match"

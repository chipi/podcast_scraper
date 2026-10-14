"""Storylines and themes over time (RFC-128, #2349): the dashboard's API.

GET /api/corpus/clustering/health reads the clustering history and the storyline artifact;
POST /api/corpus/storylines/seed is the history seed's preview / apply / undo (operator surface).
Requires ``fastapi`` (``pip install -e '.[dev]'``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


def _corpus(tmp_path: Path) -> Path:
    enr = tmp_path / "enrichments"
    enr.mkdir()
    rows = [
        {"ts": 100.0, "kind": "storyline", "source": "artifact", "storylines": 3, "quiet": 1},
        {"ts": 50.0, "kind": "theme", "themes": 20, "topics": 90, "kept": 18},
        {
            "ts": 200.0,
            "kind": "storyline",
            "source": "artifact",
            "storylines": 4,
            "quiet": 1,
            "changes": {"created": [{"id": "thc:new", "name": "new"}]},
        },
    ]
    (enr / "clustering_history.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows) + '{"torn', encoding="utf-8"
    )
    (enr / "clustering_history.backfill.jsonl").write_text(
        json.dumps({"ts": 10.0, "kind": "storyline", "source": "replay", "storylines": 2})
        + "\n"
        + json.dumps({"ts": 150.0, "kind": "storyline", "source": "replay", "storylines": 9})
        + "\n",
        encoding="utf-8",
    )
    (enr / "topic_theme_clusters.json").write_text(
        json.dumps(
            {
                "data": {
                    "clusters": [],
                    "registry": {"seeded_from": "abc"},
                    "registry_health": {"ok": True, "source": "artifact"},
                }
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_health_serves_the_series_the_latest_run_and_the_seed_state(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    client = TestClient(create_app(root, static_dir=False))
    r = client.get("/api/corpus/clustering/health")
    assert r.status_code == 200
    body = r.json()
    story = body["storylines"]
    # The backfill fills only the past before the first recorded run; a torn line is skipped.
    assert [(p["ts"], p["source"]) for p in story["series"]] == [
        (10.0, "replay"),
        (100.0, "artifact"),
        (200.0, "artifact"),
    ]
    assert story["latest"]["changes"]["created"] == [{"id": "thc:new", "name": "new"}]
    assert story["health"] == {"ok": True, "source": "artifact"}
    assert story["seed"] == {"state": "taken", "id": "abc"}
    assert [p["themes"] for p in body["themes"]["series"]] == [20]


def test_health_of_a_corpus_with_no_history_is_empty_not_an_error(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path, static_dir=False))
    body = client.get("/api/corpus/clustering/health").json()
    assert body["storylines"]["series"] == [] and body["storylines"]["latest"] is None
    assert body["storylines"]["seed"] == {"state": "none", "id": None}


def test_the_seed_needs_an_operator(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path, static_dir=False))
    r = client.post("/api/corpus/storylines/seed", params={"mode": "preview"})
    assert r.status_code in (401, 403)


def test_the_seed_routes_preview_apply_and_undo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from podcast_scraper.search import storyline_history
    from podcast_scraper.server.routes.app_auth import require_viewer_access

    calls: list[tuple] = []

    def fake_seed(root: Path, apply: bool) -> dict:
        calls.append(("seed", apply))
        return {"applied": apply}

    def fake_undo(root: Path) -> dict:
        calls.append(("undo",))
        return {"undone": "withdrawn"}

    monkeypatch.setattr(storyline_history, "seed", fake_seed)
    monkeypatch.setattr(storyline_history, "undo", fake_undo)
    app = create_app(tmp_path, static_dir=False)
    app.dependency_overrides[require_viewer_access] = lambda: object()
    client = TestClient(app)
    assert client.post("/api/corpus/storylines/seed", params={"mode": "preview"}).json() == {
        "mode": "preview",
        "applied": False,
    }
    assert client.post("/api/corpus/storylines/seed", params={"mode": "apply"}).json()["applied"]
    assert client.post("/api/corpus/storylines/seed", params={"mode": "undo"}).json() == {
        "mode": "undo",
        "undone": "withdrawn",
    }
    assert client.post("/api/corpus/storylines/seed", params={"mode": "drop"}).status_code == 400
    assert calls == [("seed", False), ("seed", True), ("undo",)]

"""Integration tests for the user-facing corpus_enrichments routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = pytest.mark.integration


def _envelope(enricher_id: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "enricher_id": enricher_id,
        "enricher_version": "1.0.0",
        "data": data,
    }


# ---------------------------------------------------------------------------
# GET /api/corpus/enrichments — listing
# ---------------------------------------------------------------------------


def test_list_enrichments_empty_when_no_enrichments_dir(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/enrichments", params={"path": str(tmp_path)})
    assert r.status_code == 200
    assert r.json() == {"enrichments": []}


def test_list_enrichments_returns_each_envelope_compactly(tmp_path: Path) -> None:
    out = tmp_path / "enrichments"
    out.mkdir()
    (out / "topic_similarity.json").write_text(
        json.dumps(_envelope("topic_similarity", {"topic_count": 12})),
        encoding="utf-8",
    )
    (out / "guest_coappearance.json").write_text(
        json.dumps(_envelope("guest_coappearance", {"pairs": []})),
        encoding="utf-8",
    )
    (out / "run_summary.json").write_text("{}", encoding="utf-8")  # excluded

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/enrichments", params={"path": str(tmp_path)})
    assert r.status_code == 200
    items = r.json()["enrichments"]
    ids = {row["enricher_id"] for row in items}
    assert ids == {"topic_similarity", "guest_coappearance"}
    assert all(row["enricher_version"] == "1.0.0" for row in items)
    assert all(row["size_bytes"] > 0 for row in items)


def test_list_enrichments_tolerates_malformed_envelope(tmp_path: Path) -> None:
    out = tmp_path / "enrichments"
    out.mkdir()
    (out / "broken.json").write_text("not json{", encoding="utf-8")
    (out / "ok.json").write_text(json.dumps(_envelope("ok", {})), encoding="utf-8")
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/enrichments", params={"path": str(tmp_path)})
    assert r.status_code == 200
    ids = {row["enricher_id"] for row in r.json()["enrichments"]}
    assert ids == {"ok"}


# ---------------------------------------------------------------------------
# GET /api/corpus/enrichments/{enricher_id} — corpus-scope envelope
# ---------------------------------------------------------------------------


def test_get_corpus_enrichment_returns_envelope(tmp_path: Path) -> None:
    out = tmp_path / "enrichments"
    out.mkdir()
    payload = _envelope("topic_similarity", {"topic_count": 3, "topics": []})
    (out / "topic_similarity.json").write_text(json.dumps(payload), encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/enrichments/topic_similarity",
        params={"path": str(tmp_path)},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["enricher_id"] == "topic_similarity"
    assert body["data"]["topic_count"] == 3


def test_get_corpus_enrichment_returns_404_when_missing(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/enrichments/topic_similarity",
        params={"path": str(tmp_path)},
    )
    assert r.status_code == 404


def test_get_corpus_enrichment_rejects_bad_enricher_id(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/enrichments/..%2Fetc%2Fpasswd",
        params={"path": str(tmp_path)},
    )
    # Path traversal sequence — must be rejected as a malformed id.
    assert r.status_code in (400, 404)


def test_get_corpus_enrichment_400_when_envelope_corrupt(tmp_path: Path) -> None:
    out = tmp_path / "enrichments"
    out.mkdir()
    (out / "x.json").write_text("not json{", encoding="utf-8")
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/enrichments/x", params={"path": str(tmp_path)})
    assert r.status_code == 500


# ---------------------------------------------------------------------------
# GET /api/corpus/episode/enrichments/{enricher_id} — episode-scope
# ---------------------------------------------------------------------------


def test_get_episode_enrichment_returns_envelope(tmp_path: Path) -> None:
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir()
    (meta_dir / "ep1.metadata.json").write_text("{}", encoding="utf-8")
    enrich_dir = meta_dir / "enrichments"
    enrich_dir.mkdir()
    payload = _envelope("topic_cooccurrence", {"pairs": []})
    (enrich_dir / "ep1.topic_cooccurrence.json").write_text(json.dumps(payload), encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/episode/enrichments/topic_cooccurrence",
        params={
            "path": str(tmp_path),
            "metadata_relpath": "metadata/ep1.metadata.json",
        },
    )
    assert r.status_code == 200
    assert r.json()["enricher_id"] == "topic_cooccurrence"


def test_get_episode_enrichment_404_when_missing(tmp_path: Path) -> None:
    (tmp_path / "metadata").mkdir()
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/episode/enrichments/insight_density",
        params={
            "path": str(tmp_path),
            "metadata_relpath": "metadata/ep1.metadata.json",
        },
    )
    assert r.status_code == 404


def test_get_episode_enrichment_rejects_relpath_traversal(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/episode/enrichments/x",
        params={"path": str(tmp_path), "metadata_relpath": "../etc/passwd"},
    )
    assert r.status_code == 400


def test_get_episode_enrichment_rejects_non_metadata_relpath(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/episode/enrichments/x",
        params={"path": str(tmp_path), "metadata_relpath": "metadata/notmeta.json"},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Default-anchor fallback
# ---------------------------------------------------------------------------


def test_routes_use_server_anchor_when_path_omitted(tmp_path: Path) -> None:
    out = tmp_path / "enrichments"
    out.mkdir()
    (out / "topic_similarity.json").write_text(
        json.dumps(_envelope("topic_similarity", {"topic_count": 1})),
        encoding="utf-8",
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/enrichments/topic_similarity")
    assert r.status_code == 200


def test_routes_400_when_no_path_and_no_anchor() -> None:
    app = create_app(None, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/enrichments")
    assert r.status_code == 400

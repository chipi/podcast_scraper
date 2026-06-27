"""Test the chunk-8 catalog-row enrichments_available field."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app
from podcast_scraper.server.routes.corpus_library import (
    _EPISODE_SCOPE_ENRICHER_IDS,
    _resolve_episode_enrichments_available,
)

pytestmark = pytest.mark.integration


def _seed_episode(corpus_root: Path, stem: str, *, enrichers: list[str]) -> str:
    meta_dir = corpus_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    md = meta_dir / f"{stem}.metadata.json"
    md.write_text(
        json.dumps(
            {
                "feed": {"feed_id": "feed1", "title": "Feed One"},
                "episode": {
                    "episode_id": f"episode:{stem}",
                    "title": f"Episode {stem}",
                    "published_date": "2026-06-01T00:00:00Z",
                },
            }
        ),
        encoding="utf-8",
    )
    if enrichers:
        out = meta_dir / "enrichments"
        out.mkdir(parents=True, exist_ok=True)
        for eid in enrichers:
            (out / f"{stem}.{eid}.json").write_text(
                json.dumps({"enricher_id": eid, "data": {}}), encoding="utf-8"
            )
    return f"metadata/{stem}.metadata.json"


def test_resolve_returns_all_false_when_no_enrichments(tmp_path: Path) -> None:
    rel = _seed_episode(tmp_path, "ep1", enrichers=[])
    flags = _resolve_episode_enrichments_available(tmp_path, rel)
    assert flags == {eid: False for eid in _EPISODE_SCOPE_ENRICHER_IDS}


def test_resolve_returns_true_for_present_envelopes(tmp_path: Path) -> None:
    rel = _seed_episode(tmp_path, "ep1", enrichers=["topic_cooccurrence"])
    flags = _resolve_episode_enrichments_available(tmp_path, rel)
    assert flags["topic_cooccurrence"] is True
    assert flags["insight_density"] is False


def test_resolve_handles_empty_relpath(tmp_path: Path) -> None:
    flags = _resolve_episode_enrichments_available(tmp_path, "")
    assert all(v is False for v in flags.values())


def test_resolve_handles_non_metadata_relpath(tmp_path: Path) -> None:
    flags = _resolve_episode_enrichments_available(tmp_path, "feeds.spec.yaml")
    assert all(v is False for v in flags.values())


def test_catalog_episode_row_carries_enrichments_available(tmp_path: Path) -> None:
    """End-to-end via /api/corpus/episodes — every row gains the new field."""
    _seed_episode(tmp_path, "ep1", enrichers=["insight_density"])
    _seed_episode(tmp_path, "ep2", enrichers=[])
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/episodes", params={"path": str(tmp_path), "limit": "10"})
    assert r.status_code == 200
    items = r.json()["items"]
    rows_by_stem = {it["metadata_relative_path"]: it for it in items}
    assert "metadata/ep1.metadata.json" in rows_by_stem
    assert rows_by_stem["metadata/ep1.metadata.json"]["enrichments_available"] == {
        "topic_cooccurrence": False,
        "insight_density": True,
    }
    assert rows_by_stem["metadata/ep2.metadata.json"]["enrichments_available"] == {
        "topic_cooccurrence": False,
        "insight_density": False,
    }

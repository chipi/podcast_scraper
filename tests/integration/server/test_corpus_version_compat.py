"""Integration: current server code against prior-release corpus fixture (#796)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper import __version__
from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

FIXTURE_CORPUS = Path(__file__).resolve().parents[2] / "fixtures" / "viewer-validation-corpus"


@pytest.fixture
def prior_release_corpus(tmp_path: Path) -> Path:
    """Copy validation corpus without ``produced_by`` (N-1 on-disk shape)."""
    import shutil

    dest = tmp_path / "corpus"
    shutil.copytree(FIXTURE_CORPUS, dest)
    manifest = {
        "schema_version": "1.1.0",
        "tool_version": "2.5.0",
        "corpus_parent": str(dest),
        "updated_at": "2026-01-01T00:00:00Z",
        "feeds": [],
        "cost_rollup": {"total_cost_usd": 0.0, "run_count": 0},
    }
    (dest / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dest


def test_health_warns_on_prior_release_corpus(prior_release_corpus: Path) -> None:
    app = create_app(prior_release_corpus, static_dir=False)
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["code_version"] == __version__
    assert body["corpus_code_version"] == "2.5.0"
    assert body["corpus_version_warning"] is not None


def test_prior_release_corpus_surfaces_do_not_500(prior_release_corpus: Path) -> None:
    """Latest code × N-1 fixture — routes return structured responses, not 5xx."""
    app = create_app(prior_release_corpus, static_dir=False)
    client = TestClient(app)
    path = str(prior_release_corpus.resolve())
    for route in (
        f"/api/corpus/feeds?path={path}",
        f"/api/corpus/digest?path={path}",
        f"/api/artifacts?path={path}",
        f"/api/search?q=test&path={path}&limit=1",
    ):
        response = client.get(route)
        assert response.status_code < 500, route

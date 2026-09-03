"""Integration tests for GET /api/corpus/theme-clusters.

Theme clusters (co-occurrence lift) are served from ``enrichments/`` — the
sibling of the semantic ``/api/corpus/topic-clusters`` endpoint.
Requires ``fastapi`` (``pip install -e '.[dev]'``).
"""

# These assert payload SHAPE and envelope-unwrapping, not navigation policy, so they
# pass min_members=0 and use minimal 2-member fixtures. The route filters small themes
# out of the navigation surface by default (see DEFAULT_MIN_THEME_MEMBERS); that
# behaviour is covered in test_corpus_theme_clusters_min_members.py. Opting out here
# keeps these tests from re-breaking every time the threshold is retuned.

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


def test_theme_clusters_uses_default_output_dir(tmp_path: Path) -> None:
    enr = tmp_path / "enrichments"
    enr.mkdir()
    payload = {
        "schema_version": "1",
        "method": "cooccurrence_lift",
        "merge_threshold": 2.0,
        "clusters": [],
        "topic_count": 0,
        "cluster_count": 0,
    }
    (enr / "topic_theme_clusters.json").write_text(json.dumps(payload), encoding="utf-8")
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/theme-clusters")
    assert r.status_code == 200
    body = r.json()
    assert body.get("method") == "cooccurrence_lift"
    assert body.get("merge_threshold") == 2.0


def test_theme_clusters_404_when_missing(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/theme-clusters", params={"path": str(tmp_path)})
    assert r.status_code == 404
    body = r.json()
    assert body.get("available") is False


def test_theme_clusters_200_returns_theme_payload(tmp_path: Path) -> None:
    enr = tmp_path / "enrichments"
    enr.mkdir()
    payload = {
        "schema_version": "1",
        "method": "cooccurrence_lift",
        "clusters": [
            {
                "cluster_type": "theme",
                "graph_compound_parent_id": "thc:shadow-fleet",
                "canonical_label": "shadow fleet",
                "member_count": 2,
                "members": [
                    {"topic_id": "topic:shadow-fleet", "label": "shadow fleet"},
                    {"topic_id": "topic:oil-prices", "label": "oil prices"},
                ],
            }
        ],
        "topic_count": 2,
        "cluster_count": 1,
        "singletons": 0,
    }
    (enr / "topic_theme_clusters.json").write_text(json.dumps(payload), encoding="utf-8")
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/theme-clusters", params={"path": str(tmp_path), "min_members": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["clusters"][0]["cluster_type"] == "theme"
    assert body["clusters"][0]["graph_compound_parent_id"] == "thc:shadow-fleet"


def test_theme_clusters_unwraps_enrichment_envelope(tmp_path: Path) -> None:
    """The enricher writes the payload WRAPPED in the framework envelope
    ({derived, enricher_id, ..., data: {...}}); the route must serve the inner
    payload so clients read ``clusters`` at the top level (like /topic-clusters)."""
    enr = tmp_path / "enrichments"
    enr.mkdir()
    inner = {
        "schema_version": "1",
        "method": "cooccurrence_lift",
        "clusters": [
            {
                "cluster_type": "theme",
                "graph_compound_parent_id": "thc:oil",
                "canonical_label": "oil",
                "member_count": 2,
                "members": [{"topic_id": "topic:oil"}, {"topic_id": "topic:lng"}],
            }
        ],
        "cluster_count": 1,
    }
    envelope = {
        "derived": True,
        "enricher_id": "topic_theme_clusters",
        "enricher_version": "1.0.0",
        "status": "ok",
        "data": inner,
    }
    (enr / "topic_theme_clusters.json").write_text(json.dumps(envelope), encoding="utf-8")
    app = create_app(tmp_path, static_dir=False)
    r = TestClient(app).get(
        "/api/corpus/theme-clusters", params={"path": str(tmp_path), "min_members": 0}
    )
    assert r.status_code == 200
    body = r.json()
    # Unwrapped: clusters at the top level, not nested under "data".
    assert "data" not in body
    assert body["clusters"][0]["graph_compound_parent_id"] == "thc:oil"


def test_theme_clusters_cached_by_file_mtime(tmp_path: Path) -> None:
    """The whole-artifact parse is cached by the file's OWN mtime, not corpus mtime.

    Proven without a spy: rewrite the file's CONTENT but hold its mtime → the cached (stale) parse
    is served (a cache hit); bump the mtime → the new content is served (invalidation). Using the
    file's own mtime matters because the enricher rewrites this file without an ingest.
    """
    import os

    from podcast_scraper import perf_cache

    perf_cache.clear()
    enr = tmp_path / "enrichments"
    enr.mkdir()
    artifact = enr / "topic_theme_clusters.json"

    def write(method: str, mtime: float) -> None:
        artifact.write_text(json.dumps({"method": method, "clusters": []}), encoding="utf-8")
        os.utime(artifact, (mtime, mtime))

    client = TestClient(create_app(tmp_path, static_dir=False))

    write("first", 1_000_000.0)
    assert (
        client.get("/api/corpus/theme-clusters", params={"path": str(tmp_path)}).json()["method"]
        == "first"
    )

    # New content, SAME mtime → the cache must still serve the first parse.
    write("second", 1_000_000.0)
    assert (
        client.get("/api/corpus/theme-clusters", params={"path": str(tmp_path)}).json()["method"]
        == "first"
    ), "a content change without an mtime bump was NOT served from cache"

    # Mtime advances → the cache invalidates and the new content is served.
    os.utime(artifact, (2_000_000.0, 2_000_000.0))
    assert (
        client.get("/api/corpus/theme-clusters", params={"path": str(tmp_path)}).json()["method"]
        == "second"
    ), "the cache did not invalidate on an mtime bump"

    perf_cache.clear()

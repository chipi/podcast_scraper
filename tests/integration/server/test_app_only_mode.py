"""``PODCAST_SERVE_APP_ONLY`` — the low-privilege public serve mode (#1163 / ADR-116).

The public consumer player proxies to a backend that must expose ONLY the authed
``/api/app/*`` plane (+ health) — never the operator/read ``/api/*`` routes, and never
the always-on operator-grade ones (``index_rebuild``, ``corpus_media``, ``ops``). This
locks that contract so a future always-on ``/api/*`` route cannot silently leak onto a
public deployment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


def _api_paths(output_dir: Path) -> set[str]:
    app = create_app(output_dir=output_dir)
    return {getattr(r, "path", "") for r in app.routes if getattr(r, "path", "").startswith("/api")}


def test_full_mode_mounts_operator_read_api(tmp_path: pytest.TempPathFactory, monkeypatch) -> None:
    monkeypatch.delenv("PODCAST_SERVE_APP_ONLY", raising=False)
    paths = _api_paths(Path(str(tmp_path)))
    # Sanity: the default (non-app-only) build DOES mount the operator/read plane.
    assert any(p.startswith("/api/search") for p in paths)
    assert any("/api/corpus/media" in p for p in paths)
    assert any(p.startswith("/api/app") for p in paths)


def test_app_only_mounts_only_consumer_plane(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PODCAST_SERVE_APP_ONLY", "1")
    paths = _api_paths(Path(str(tmp_path)))

    # The consumer plane + health are present.
    assert any(p.startswith("/api/app") for p in paths), "consumer /api/app/* must mount"
    assert any(p.startswith("/api/health") for p in paths), "health must mount"

    # NONE of the operator/read /api/* routes — including the always-on operator-grade
    # ones — may be present on a public app-only deployment.
    forbidden = ("/api/search", "/api/relational", "/api/explore", "/api/corpus/media")
    for sub in forbidden:
        assert not any(sub in p for p in paths), f"{sub} must NOT mount in app-only mode"
    assert not any("rebuild" in p for p in paths), "index_rebuild must NOT mount in app-only mode"
    assert not any(p.startswith("/api/jobs") for p in paths), "jobs must NOT mount in app-only mode"
    assert not any(p.startswith("/api/ops") for p in paths), "ops must NOT mount in app-only mode"


def test_app_only_serves_app_but_404s_operator_routes(tmp_path, monkeypatch) -> None:
    """Behavioral: an app-only server actually returns 404 for /api/* operator routes
    (the route simply isn't mounted), while health + the consumer plane respond."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("PODCAST_SERVE_APP_ONLY", "1")
    client = TestClient(create_app(output_dir=Path(str(tmp_path))))

    # Operator/read + operator-grade routes must be absent (404), not merely unauth'd.
    for path in (
        "/api/jobs",
        "/api/search",
        "/api/relational",
        "/api/corpus/media",
        "/api/ops/summary",
    ):
        assert client.get(path).status_code == 404, f"{path} must 404 in app-only mode"

    # Health responds; the consumer plane is mounted (not 404 — may be 401/503 unauthed).
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/app/auth/status").status_code != 404


def test_full_mode_serves_operator_read_routes(tmp_path, monkeypatch) -> None:
    """Contrast: the default (non-app-only) server DOES mount /api/search (not 404)."""
    from fastapi.testclient import TestClient

    monkeypatch.delenv("PODCAST_SERVE_APP_ONLY", raising=False)
    client = TestClient(create_app(output_dir=Path(str(tmp_path))))
    assert client.get("/api/search").status_code != 404

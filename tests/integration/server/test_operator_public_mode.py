"""``PODCAST_SERVE_OPERATOR_PUBLIC`` — the public operator surface (RFC-108).

`operator.closelistening.app` exposes a **curated read-only subset** of the operator
routes, each mounted with a **router-level ≥creator gate**. `index_rebuild`, `ops`, and
the privileged flag-gated plane (jobs/operator-config/feeds) stay tailnet-only. This locks
the exposed-vs-private split AND that the gate is enforced (unauthed → 401, never open).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from podcast_scraper.server import app as app_mod
from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


def _api_paths(output_dir: Path) -> set[str]:
    app = create_app(output_dir=output_dir)
    return {getattr(r, "path", "") for r in app.routes if getattr(r, "path", "").startswith("/api")}


def test_operator_public_mounts_curated_read_subset(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("PODCAST_SERVE_APP_ONLY", raising=False)
    monkeypatch.setenv("PODCAST_SERVE_OPERATOR_PUBLIC", "1")
    paths = _api_paths(Path(str(tmp_path)))

    # Curated read routes + consumer plane + health ARE present.
    assert any(p.startswith("/api/search") for p in paths), "search must mount"
    assert any("/api/corpus/media" in p for p in paths), "corpus media must mount"
    assert any(p.startswith("/api/app") for p in paths), "consumer plane must mount"
    assert any(p.startswith("/api/health") for p in paths), "health must mount"

    # The privileged / compute routes must NOT mount on the public operator surface.
    assert not any("rebuild" in p for p in paths), "index_rebuild must NOT mount"
    assert not any(p.startswith("/api/ops") for p in paths), "ops must NOT mount"
    assert not any(p.startswith("/api/jobs") for p in paths), "jobs must NOT mount"


def test_operator_public_gates_read_routes_unauthed_401(tmp_path, monkeypatch) -> None:
    """The gate fires: curated operator routes are mounted (not 404) but require a
    signed-in ≥creator — an unauthenticated request gets **401**, never an open 200."""
    from fastapi.testclient import TestClient

    monkeypatch.delenv("PODCAST_SERVE_APP_ONLY", raising=False)
    monkeypatch.setenv("PODCAST_SERVE_OPERATOR_PUBLIC", "1")
    client = TestClient(create_app(output_dir=Path(str(tmp_path))))

    for path in ("/api/search", "/api/artifacts"):
        code = client.get(path).status_code
        assert (
            code == 401
        ), f"{path} must be ≥creator-gated (401 unauthed) in operator-public, got {code}"

    # Privileged routes stay absent (404, not merely gated); health is open.
    assert client.get("/api/ops/summary").status_code == 404
    assert client.get("/api/health").status_code == 200


def test_operator_public_subset_excludes_the_mutating_operator_routes() -> None:
    public = set(app_mod._OPERATOR_PUBLIC_READ_ROUTES)
    full = set(app_mod._OPERATOR_READ_ROUTES)
    assert public < full, "public subset must be strictly smaller than the full read set"
    assert full - public == {
        app_mod.index_rebuild,
        app_mod.ops,
        app_mod.resilience_routes,
    }, "operator-public must exclude the mutating/control routes (index_rebuild, ops, resilience)"


def test_operator_public_curated_subset_has_no_mutating_operator_writes() -> None:
    """Guard the read-only invariant: no curated public module may expose a genuinely
    *mutating* operator route. POST-for-query (search/compare, corpus resolve,
    topics/timeline) is allowed; a reset/write is not."""
    allowed_post_for_query = {
        "/api/search/compare",
        "/api/corpus/resolve-episode-artifacts",
        "/api/corpus/node-episodes",
        "/api/topics/timeline",
    }
    offenders = []
    for mod in app_mod._OPERATOR_PUBLIC_READ_ROUTES:
        for r in mod.router.routes:
            writes: set[str] = (getattr(r, "methods", set()) or set()) - {"GET", "HEAD", "OPTIONS"}
            path = "/api" + mod.router.prefix + getattr(r, "path", "")
            if writes and path not in allowed_post_for_query:
                offenders.append((sorted(writes), path))
    assert not offenders, f"curated public subset must not expose mutating routes: {offenders}"


def test_full_tailnet_mode_leaves_operator_routes_ungated(tmp_path, monkeypatch) -> None:
    """Contrast: the default (tailnet) operator serve mounts /api/search WITHOUT the
    ≥creator gate — tailnet privacy is the gate, so it is not 401/404."""
    from fastapi.testclient import TestClient

    monkeypatch.delenv("PODCAST_SERVE_APP_ONLY", raising=False)
    monkeypatch.delenv("PODCAST_SERVE_OPERATOR_PUBLIC", raising=False)
    client = TestClient(create_app(output_dir=Path(str(tmp_path))))
    code = client.get("/api/search").status_code
    assert code not in (401, 404), f"tailnet operator /api/search must be ungated, got {code}"

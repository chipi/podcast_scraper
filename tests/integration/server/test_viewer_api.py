"""Viewer API: health + artifacts.

Requires ``fastapi`` (``pip install -e '.[dev]'``).
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app
from podcast_scraper.server.schemas import HealthResponse

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


def test_health_ok(tmp_path: Path) -> None:
    """Health JSON includes version contract fields when output_dir is set."""
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    body = HealthResponse.model_validate(response.json())
    assert body.status == "ok"
    assert body.code_version
    assert body.min_supported_corpus_code_version
    assert body.corpus_version_warning is not None  # empty corpus — no produced_by


def test_get_artifact_rejects_traversal(tmp_path: Path) -> None:
    """Use percent-encoding so ``..`` is not stripped during URL normalization."""
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    malicious = "metadata/../../../outside.gi.json"
    encoded = quote(malicious, safe="")
    response = client.get(
        f"/api/artifacts/{encoded}",
        params={"path": str(tmp_path)},
    )
    assert response.status_code == 400


def test_get_artifact_404(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/artifacts/metadata/missing.gi.json",
        params={"path": str(tmp_path)},
    )
    assert response.status_code == 404


def test_get_artifact_rejects_non_gi_kg_suffix(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "readme.txt").write_text("x", encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    response = client.get(
        "/api/artifacts/metadata/readme.txt",
        params={"path": str(tmp_path)},
    )
    assert response.status_code == 400

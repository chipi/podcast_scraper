"""CORS allowance for the Capacitor native-shell origins (#1310).

The native app's WebView serves from a fixed local origin (capacitor://localhost, https://localhost)
and calls this API cross-origin, so those origins must be on the CORS allowlist regardless of the
web origins pinned via PODCAST_SERVE_CORS_ORIGINS.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path, static_dir=False))


@pytest.mark.parametrize(
    "origin",
    ["capacitor://localhost", "https://localhost", "http://localhost"],
)
def test_native_origins_are_cors_allowed(tmp_path: Path, origin: str) -> None:
    client = _client(tmp_path)
    resp = client.get("/api/health", headers={"Origin": origin})
    assert resp.headers.get("access-control-allow-origin") == origin
    assert resp.headers.get("access-control-allow-credentials") == "true"


def test_native_origins_allowed_even_when_web_origin_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Prod pins the public web hostname — the native origins must still be allowed alongside it.
    monkeypatch.setenv("PODCAST_SERVE_CORS_ORIGINS", "https://player.example")
    client = _client(tmp_path)
    resp = client.get("/api/health", headers={"Origin": "capacitor://localhost"})
    assert resp.headers.get("access-control-allow-origin") == "capacitor://localhost"


def test_unknown_origin_is_not_cors_allowed(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert resp.headers.get("access-control-allow-origin") != "https://evil.example"

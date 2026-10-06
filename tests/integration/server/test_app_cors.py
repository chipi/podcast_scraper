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


# --- production trusts only its real domains (operator, 2026-10-05) ---------------------------

_DEV = "http://localhost:5173"


@pytest.mark.parametrize("env", ["prod", "preprod", "PROD"])
def test_a_deployed_api_without_a_pin_trusts_no_dev_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: str
) -> None:
    monkeypatch.delenv("PODCAST_SERVE_CORS_ORIGINS", raising=False)
    monkeypatch.setenv("PODCAST_ENV", env)
    app = create_app(tmp_path, static_dir=False)
    assert app.state.trusted_origins == []
    resp = TestClient(app).get("/api/health", headers={"Origin": _DEV})
    assert resp.headers.get("access-control-allow-origin") != _DEV


@pytest.mark.parametrize("env", [None, "dev"])
def test_a_dev_box_trusts_the_local_dev_servers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: str | None
) -> None:
    monkeypatch.delenv("PODCAST_SERVE_CORS_ORIGINS", raising=False)
    if env is None:
        monkeypatch.delenv("PODCAST_ENV", raising=False)
    else:
        monkeypatch.setenv("PODCAST_ENV", env)
    resp = _client(tmp_path).get("/api/health", headers={"Origin": _DEV})
    assert resp.headers.get("access-control-allow-origin") == _DEV


def test_the_pin_replaces_the_dev_servers_even_on_a_dev_box(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PODCAST_SERVE_CORS_ORIGINS", "https://player.example")
    monkeypatch.delenv("PODCAST_ENV", raising=False)
    app = create_app(tmp_path, static_dir=False)
    assert app.state.trusted_origins == ["https://player.example"]
    resp = TestClient(app).get("/api/health", headers={"Origin": _DEV})
    assert resp.headers.get("access-control-allow-origin") != _DEV


def test_native_origins_are_never_trusted_for_cookie_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CORS-allowed so the app can call the API (asserted above); never trusted with the cookie —
    # native auth is a Bearer token, and http(s)://localhost is any local server.
    monkeypatch.setenv("PODCAST_SERVE_CORS_ORIGINS", "https://player.example")
    app = create_app(tmp_path, static_dir=False)
    for native in ("capacitor://localhost", "https://localhost", "http://localhost"):
        assert native not in app.state.trusted_origins


def test_the_public_player_pins_its_real_domain() -> None:
    compose = Path(__file__).resolve().parents[3] / "compose/docker-compose.player-public.yml"
    text = compose.read_text(encoding="utf-8")
    assert (
        "PODCAST_SERVE_CORS_ORIGINS: ${PODCAST_SERVE_CORS_ORIGINS:-https://closelistening.app}"
        in text
    )

"""Integration tests for the operator API guard + audit (#1071; admin gating #1128).

Access rule: an operator request (read *or* write) is allowed with a valid admin **session** OR a
valid operator **key**; otherwise 403. The gate is enforced only when platform auth is configured
or a key is set — a bare deployment (neither) keeps the prior network-only behavior.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_user_store import create_user

pytestmark = [pytest.mark.integration]


def _app(tmp_path: Path, *, key: str = "", auth: bool = False):
    app = create_app(tmp_path, static_dir=False, enable_feeds_api=True)
    app.state.operator_api_key = key
    app.state.audit_path = tmp_path / "audit.jsonl"
    # Platform auth configured (session resolvable) vs a bare deployment (neither auth nor key).
    app.state.session_secret = "test-secret" if auth else ""
    app.state.app_data_dir = (tmp_path / "appdata") if auth else None
    return app


def _as(app, role: str) -> TestClient:
    """A client whose ``lp_session`` cookie resolves to a freshly-created user of ``role``."""
    user = create_user(
        app.state.app_data_dir,
        provider="mock",
        subject=role,
        email=f"{role}@x.io",
        name=role,
        role=role,
    )
    client = TestClient(app)
    cookie = app_sessions.sign(
        {"user_id": user.user_id, "iat": int(time.time())}, app.state.session_secret
    )
    client.cookies.set(app_sessions.SESSION_COOKIE, cookie)
    return client


# --- enforced when auth is configured -----------------------------------------------------------


def test_no_credentials_denied_for_reads_and_writes(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path, auth=True))
    assert client.get("/api/feeds").status_code == 403  # reads gated too
    assert client.put("/api/feeds", json={"feeds": []}).status_code == 403


def test_admin_session_is_allowed(tmp_path: Path) -> None:
    app = _app(tmp_path, auth=True)
    admin = _as(app, "admin")
    assert admin.get("/api/feeds").status_code != 403  # guard passed (route handles it)
    assert admin.put("/api/feeds", json={"feeds": []}).status_code != 403


def test_non_admin_session_is_denied(tmp_path: Path) -> None:
    app = _app(tmp_path, auth=True)
    for role in ("creator", "listener"):
        client = _as(app, role)
        assert client.get("/api/feeds").status_code == 403
        assert client.put("/api/feeds", json={"feeds": []}).status_code == 403


def test_operator_key_is_allowed_without_a_session(tmp_path: Path) -> None:
    app = _app(tmp_path, key="secret", auth=True)
    client = TestClient(app)
    # no key, no session → denied
    assert client.put("/api/feeds", json={"feeds": []}).status_code == 403
    # with the right key → guard passes (headless automation path)
    ok = client.put("/api/feeds", json={"feeds": []}, headers={"X-Operator-Key": "secret"})
    assert ok.status_code != 403
    assert client.get("/api/feeds", headers={"X-Operator-Key": "secret"}).status_code != 403


def test_writes_are_audited(tmp_path: Path) -> None:
    app = _app(tmp_path, key="secret", auth=True)
    client = TestClient(app)
    client.put("/api/feeds", json={"feeds": []})  # denied
    client.put("/api/feeds", json={"feeds": []}, headers={"X-Operator-Key": "secret"})  # allowed
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2


# --- bare deployment: neither auth nor key → prior network-only behavior -------------------------


def test_bare_deployment_is_not_gated(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path, key="", auth=False))
    assert client.get("/api/feeds").status_code != 403
    assert client.put("/api/feeds", json={"feeds": []}).status_code != 403


def test_key_only_deployment_enforces_even_without_platform_auth(tmp_path: Path) -> None:
    # A key alone is enough to switch enforcement on (no OAuth configured).
    client = TestClient(_app(tmp_path, key="secret", auth=False))
    assert client.put("/api/feeds", json={"feeds": []}).status_code == 403
    ok = client.put("/api/feeds", json={"feeds": []}, headers={"X-Operator-Key": "secret"})
    assert ok.status_code != 403

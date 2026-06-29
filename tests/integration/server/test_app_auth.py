"""Integration tests for /api/app/auth/* + get_current_user (#1063).

Uses a stub OAuth provider — no real Google call in CI (per the no-real-services rule).
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_oauth import MockOAuthProvider, OAuthError, OAuthIdentity
from podcast_scraper.server.app_user_store import set_disabled

pytestmark = [pytest.mark.integration]


class _StubProvider:
    name = "stub"

    def authorization_url(
        self, *, state: str, redirect_uri: str, login_hint: str | None = None
    ) -> str:
        return f"https://stub.example/authorize?state={state}"

    def exchange_code(self, *, code: str, redirect_uri: str) -> OAuthIdentity:
        if code == "bad":
            raise OAuthError("bad code")
        return OAuthIdentity(
            provider="stub", subject="sub-123", email="jane@example.com", name="Jane"
        )


def _app(
    tmp_path: Path,
    *,
    with_provider: bool = True,
    secret: str = "test-secret",
    access_policy: AccessPolicy | None = None,
):
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = secret
    app.state.app_data_dir = tmp_path / "appdata"
    app.state.oauth_provider = _StubProvider() if with_provider else None
    app.state.access_policy = access_policy or AccessPolicy("open", frozenset(), frozenset())
    return app


def _login_state(client: TestClient) -> str:
    resp = client.get("/api/app/auth/login", follow_redirects=False)
    assert resp.status_code == 307, resp.text
    loc = resp.headers["location"]
    assert loc.startswith("https://stub.example/authorize")
    return str(parse_qs(urlparse(loc).query)["state"][0])


def test_full_login_callback_me_logout(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path))
    state = _login_state(client)
    assert client.get("/api/app/me").status_code == 401  # state set, not yet authed

    cb = client.get(
        "/api/app/auth/callback", params={"code": "good", "state": state}, follow_redirects=False
    )
    assert cb.status_code == 307
    assert cb.headers["location"] == "/"

    me = client.get("/api/app/me")
    assert me.status_code == 200
    body = me.json()
    assert body["email"] == "jane@example.com"
    assert body["name"] == "Jane"
    assert body["user_id"].startswith("u_")

    assert client.post("/api/app/auth/logout").status_code == 204
    assert client.get("/api/app/me").status_code == 401


def test_callback_rejects_bad_state(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path))
    _login_state(client)
    resp = client.get(
        "/api/app/auth/callback", params={"code": "good", "state": "wrong"}, follow_redirects=False
    )
    assert resp.status_code == 400


def test_callback_oauth_error_is_502(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path))
    state = _login_state(client)
    resp = client.get(
        "/api/app/auth/callback", params={"code": "bad", "state": state}, follow_redirects=False
    )
    assert resp.status_code == 502


def test_login_503_when_unconfigured(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path, with_provider=False))
    assert client.get("/api/app/auth/login", follow_redirects=False).status_code == 503


def test_me_401_without_session(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path))
    assert client.get("/api/app/me").status_code == 401


def test_callback_rejects_disallowed_email(tmp_path: Path) -> None:
    policy = AccessPolicy("allowlist", frozenset({"allowed@example.com"}), frozenset())
    client = TestClient(_app(tmp_path, access_policy=policy))
    state = _login_state(client)
    resp = client.get(
        "/api/app/auth/callback", params={"code": "good", "state": state}, follow_redirects=False
    )
    assert resp.status_code == 403  # jane@example.com is not on the allowlist
    assert client.get("/api/app/me").status_code == 401  # and no account was created


def test_mock_provider_full_flow_dev_identity(tmp_path: Path) -> None:
    """The real MockOAuthProvider self-completes the code flow with a dev identity.

    Mirrors what local dev + Playwright e2e drive: login redirects straight back to
    the callback with a mock code, no network, and ``/me`` returns the dev account.
    """
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    app.state.oauth_provider = MockOAuthProvider()
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    client = TestClient(app)

    resp = client.get("/api/app/auth/login", follow_redirects=False)
    assert resp.status_code == 307, resp.text
    loc = resp.headers["location"]
    # Mock redirects back to our own callback with a code + the CSRF state.
    assert "/api/app/auth/callback" in loc
    q = parse_qs(urlparse(loc).query)
    assert q["code"] == [MockOAuthProvider.MOCK_CODE]

    cb = client.get(
        "/api/app/auth/callback",
        params={"code": q["code"][0], "state": q["state"][0]},
        follow_redirects=False,
    )
    assert cb.status_code == 307
    me = client.get("/api/app/me")
    assert me.status_code == 200
    assert me.json()["email"] == "dev@localhost"


def test_disabled_user_is_locked_out(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path))
    state = _login_state(client)
    client.get(
        "/api/app/auth/callback", params={"code": "good", "state": state}, follow_redirects=False
    )
    uid = client.get("/api/app/me").json()["user_id"]
    assert set_disabled(tmp_path / "appdata", uid, True) is True
    assert client.get("/api/app/me").status_code == 401


def test_auth_status_enabled_anonymous(tmp_path: Path) -> None:
    # Auth configured (secret + provider + data dir) but no session → enabled, user None.
    client = TestClient(_app(tmp_path))
    resp = client.get("/api/app/auth/status")
    assert resp.status_code == 200
    assert resp.json() == {"enabled": True, "user": None}


def test_auth_status_enabled_with_signed_in_user(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path))
    state = _login_state(client)
    client.get("/api/app/auth/callback", params={"code": "good", "state": state})
    body = client.get("/api/app/auth/status").json()
    assert body["enabled"] is True
    assert body["user"]["email"] == "jane@example.com"
    assert body["user"]["role"] == "listener"


def test_auth_status_disabled_when_unconfigured(tmp_path: Path) -> None:
    # No provider + no secret → auth is NOT enabled → the viewer renders open (never 401s here).
    client = TestClient(_app(tmp_path, with_provider=False, secret=""))
    assert client.get("/api/app/auth/status").json() == {"enabled": False, "user": None}

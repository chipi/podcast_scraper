"""Integration tests for /api/app/auth/* + get_current_user (#1063).

Uses a stub OAuth provider — no real Google call in CI (per the no-real-services rule).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions
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


def test_login_return_to_round_trips_through_callback(tmp_path: Path) -> None:
    """A safe same-origin ``return_to`` on /auth/login is honoured by the callback redirect.

    This is what makes the MCP /authorize bounce (RFC-112) land back on the consent screen
    after Google sign-in instead of dumping the user on the player home.
    """
    client = TestClient(_app(tmp_path))
    dest = "/api/app/mcp/oauth/authorize?client_id=x&scope=mcp"
    resp = client.get("/api/app/auth/login", params={"return_to": dest}, follow_redirects=False)
    state = str(parse_qs(urlparse(resp.headers["location"]).query)["state"][0])
    cb = client.get(
        "/api/app/auth/callback", params={"code": "good", "state": state}, follow_redirects=False
    )
    assert cb.status_code == 307
    assert cb.headers["location"] == dest


def test_login_return_to_open_redirect_guard(tmp_path: Path) -> None:
    """An off-site ``return_to`` is dropped — the callback falls back to the home path."""
    client = TestClient(_app(tmp_path))
    for evil in ("https://evil.example/x", "//evil.example/x"):
        resp = client.get("/api/app/auth/login", params={"return_to": evil}, follow_redirects=False)
        state = str(parse_qs(urlparse(resp.headers["location"]).query)["state"][0])
        cb = client.get(
            "/api/app/auth/callback",
            params={"code": "good", "state": state},
            follow_redirects=False,
        )
        assert cb.headers["location"] == "/", f"open-redirect not guarded for {evil!r}"
        client.cookies.clear()


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


def test_native_login_callback_returns_deep_link_token(tmp_path: Path) -> None:
    """Native (#1310): ?platform=native → callback redirects to the app's custom-scheme deep link
    carrying the signed token (no cookie), and that token authenticates /me as a Bearer header."""
    client = TestClient(_app(tmp_path))
    resp = client.get("/api/app/auth/login", params={"platform": "native"}, follow_redirects=False)
    assert resp.status_code == 307
    state = str(parse_qs(urlparse(resp.headers["location"]).query)["state"][0])

    cb = client.get(
        "/api/app/auth/callback", params={"code": "good", "state": state}, follow_redirects=False
    )
    assert cb.status_code == 307
    loc = cb.headers["location"]
    assert loc.startswith("closelistening://auth#token="), loc
    # No session cookie on the native path — the token rides the deep link instead.
    assert app_sessions.SESSION_COOKIE not in cb.headers.get("set-cookie", "")

    token = loc.split("#token=", 1)[1]
    # The native path set no session cookie, so this client has none — /me now succeeds purely on
    # the Bearer header (the exact native scenario), and still 401s without it.
    assert client.get("/api/app/me").status_code == 401
    me = client.get("/api/app/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "jane@example.com"


def test_bearer_token_rejected_when_invalid(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path))
    assert (
        client.get("/api/app/me", headers={"Authorization": "Bearer not.a.valid.token"}).status_code
        == 401
    )


def test_web_callback_still_sets_cookie_not_deep_link(tmp_path: Path) -> None:
    """Regression guard: the web path (no platform hint) is unchanged — cookie + redirect to /."""
    client = TestClient(_app(tmp_path))
    state = _login_state(client)
    cb = client.get(
        "/api/app/auth/callback", params={"code": "good", "state": state}, follow_redirects=False
    )
    assert cb.headers["location"] == "/"
    assert app_sessions.SESSION_COOKIE in cb.headers.get("set-cookie", "")


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


def test_auth_dev_users_disabled_for_non_mock_provider(tmp_path: Path) -> None:
    # Stub (non-mock) provider → no dev picker; the UI shows the normal sign-in button.
    client = TestClient(_app(tmp_path))
    assert client.get("/api/app/auth/dev-users").json() == {"enabled": False, "users": []}


def test_auth_dev_users_lists_seed_roster_for_mock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = tmp_path / "seed.json"
    seed.write_text(
        json.dumps(
            [
                {"hint": "ada-admin", "name": "Ada Admin", "role": "admin"},
                {"hint": "pat-player", "name": "Pat Player", "role": "listener"},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("APP_SEED_USERS_FILE", str(seed))
    app = _app(tmp_path)
    app.state.oauth_provider = MockOAuthProvider()  # the mock provider activates the picker
    body = TestClient(app).get("/api/app/auth/dev-users").json()
    assert body["enabled"] is True
    by_hint = {u["hint"]: u for u in body["users"]}
    assert set(by_hint) == {"ada-admin", "pat-player"}
    assert by_hint["ada-admin"]["role"] == "admin" and by_hint["ada-admin"]["name"] == "Ada Admin"
    assert by_hint["pat-player"]["role"] == "listener"


# --- #1977: the connector cold-start case ------------------------------------------------------
#
# Adding an MCP server as a claude.ai connector starts the flow in one browser context, hands
# sign-in to another (Safari / the Google app), and returns to a third. The `lp_oauth_state` cookie
# set in the first context is not present on return, so the callback 400'd and the connector failed
# with an opaque error. Measured in prod 2026-09-05: authorize 302 -> login 307 -> callback 400 in
# 24 seconds, one login, cookie correctly configured (`HttpOnly; Max-Age=600; Path=/; SameSite=lax;
# Secure`). The same device completed the identical flow minutes later when Safari itself started
# it — so the discriminator is the context handoff, not the browser.
#
# The state we send the provider is now our own signed, unexpired payload, so the callback can
# validate the flow from what is echoed back even with no cookie.


def test_callback_succeeds_when_the_state_cookie_never_came_back(tmp_path: Path) -> None:
    """THE regression: cookie dropped by a cross-context handoff, signed state intact."""
    client = TestClient(_app(tmp_path))
    state = _login_state(client)

    client.cookies.clear()  # the returning context has no cookie jar from the first one

    cb = client.get(
        "/api/app/auth/callback", params={"code": "good", "state": state}, follow_redirects=False
    )
    assert cb.status_code == 307, f"cold-start callback rejected: {cb.text}"
    assert client.get("/api/app/me").status_code == 200


def test_forged_state_is_still_rejected_without_a_cookie(tmp_path: Path) -> None:
    """The fallback must not become an open door: an unsigned state is not a flow."""
    client = TestClient(_app(tmp_path))
    _login_state(client)
    client.cookies.clear()
    for bogus in ("wrong", "a.b", "", "x" * 200):
        resp = client.get(
            "/api/app/auth/callback",
            params={"code": "good", "state": bogus},
            follow_redirects=False,
        )
        assert resp.status_code == 400, f"forged state {bogus!r} was accepted"


def test_state_signed_with_another_secret_is_rejected(tmp_path: Path) -> None:
    """Tamper-evidence is the whole basis for trusting the echoed state."""
    client = TestClient(_app(tmp_path))
    _login_state(client)
    client.cookies.clear()
    forged = app_sessions.sign(
        {
            "state": "attacker",
            "iat": int(time.time()),
            "grant": "",
            "platform": "",
            "return_to": "",
        },
        "not-the-server-secret",
    )
    resp = client.get(
        "/api/app/auth/callback", params={"code": "good", "state": forged}, follow_redirects=False
    )
    assert resp.status_code == 400


def test_return_to_still_round_trips_without_a_cookie(tmp_path: Path) -> None:
    """The MCP /authorize bounce must survive the cold-start path, or connectors land on home."""
    client = TestClient(_app(tmp_path))
    resp = client.get(
        "/api/app/auth/login",
        params={"return_to": "/api/app/mcp/oauth/authorize?client_id=x"},
        follow_redirects=False,
    )
    state = str(parse_qs(urlparse(resp.headers["location"]).query)["state"][0])
    client.cookies.clear()
    cb = client.get(
        "/api/app/auth/callback", params={"code": "good", "state": state}, follow_redirects=False
    )
    assert cb.status_code == 307
    assert cb.headers["location"] == "/api/app/mcp/oauth/authorize?client_id=x"


def test_overlong_return_to_is_dropped_not_carried(tmp_path: Path) -> None:
    """`return_to` rides inside the provider `state` now; providers cap its length."""
    client = TestClient(_app(tmp_path))
    resp = client.get(
        "/api/app/auth/login", params={"return_to": "/" + "a" * 900}, follow_redirects=False
    )
    state = str(parse_qs(urlparse(resp.headers["location"]).query)["state"][0])
    assert len(state) < 700, "signed state grew unbounded with return_to"
    cb = client.get(
        "/api/app/auth/callback", params={"code": "good", "state": state}, follow_redirects=False
    )
    assert cb.status_code == 307
    assert cb.headers["location"] == "/"  # dropped, not honoured

"""Sign in with Apple through the auth routes (#2275) — a stub Apple provider, no real call."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_oauth import _apple_name, OAuthIdentity

pytestmark = [pytest.mark.integration, pytest.mark.app]


class _Google:
    name = "google"

    def authorization_url(
        self, *, state: str, redirect_uri: str, login_hint: str | None = None
    ) -> str:
        return f"https://google.example/authorize?state={state}"

    def exchange_code(self, *, code: str, redirect_uri: str) -> OAuthIdentity:
        return OAuthIdentity(
            provider="google", subject="g-1", email="jane@example.com", name="Jane"
        )


class _Apple:
    name = "apple"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def authorization_url(
        self, *, state: str, redirect_uri: str, login_hint: str | None = None
    ) -> str:
        return f"https://appleid.example/authorize?state={state}&response_mode=form_post"

    def exchange_code(
        self, *, code: str, redirect_uri: str, user_json: str | None = None
    ) -> OAuthIdentity:
        self.calls.append({"code": code, "user_json": user_json})
        return OAuthIdentity(
            provider="apple",
            subject="a-1",
            email="x@privaterelay.appleid.com",
            name=_apple_name(user_json) or "x@privaterelay.appleid.com",
        )


def _app(tmp_path: Path) -> tuple[TestClient, _Apple]:
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    apple = _Apple()
    app.state.oauth_providers = {"google": _Google(), "apple": apple}
    app.state.oauth_provider = app.state.oauth_providers["google"]
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    return TestClient(app), apple


def _state(client: TestClient, provider: str | None) -> str:
    params = {"provider": provider} if provider else {}
    resp = client.get("/api/app/auth/login", params=params, follow_redirects=False)
    assert resp.status_code == 307, resp.text
    return str(parse_qs(urlparse(resp.headers["location"]).query)["state"][0])


def test_login_with_provider_apple_goes_to_apple_and_the_state_names_it(tmp_path: Path) -> None:
    client, _ = _app(tmp_path)
    resp = client.get("/api/app/auth/login?provider=apple", follow_redirects=False)
    assert resp.headers["location"].startswith("https://appleid.example/authorize")
    state = parse_qs(urlparse(resp.headers["location"]).query)["state"][0]
    verified = app_sessions.verify(state, "test-secret", max_age=600)
    assert verified is not None and verified["provider"] == "apple"


def test_login_without_provider_still_goes_to_the_primary(tmp_path: Path) -> None:
    client, _ = _app(tmp_path)
    resp = client.get("/api/app/auth/login", follow_redirects=False)
    assert resp.headers["location"].startswith("https://google.example/authorize")


def test_an_unconfigured_provider_is_404_not_a_silent_fallback(tmp_path: Path) -> None:
    client, _ = _app(tmp_path)
    assert (
        client.get("/api/app/auth/login?provider=facebook", follow_redirects=False).status_code
        == 404
    )


def test_apple_form_post_without_the_state_cookie_signs_in_with_a_303(tmp_path: Path) -> None:
    # Apple's callback is a cross-site POST: the SameSite=lax state cookie does not come with it.
    client, apple = _app(tmp_path)
    state = _state(client, "apple")
    client.cookies.clear()
    user = json.dumps({"name": {"firstName": "Ada", "lastName": "Lovelace"}})
    resp = client.post(
        "/api/app/auth/callback",
        content=f"code=C1&state={state}&user={user}".encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        follow_redirects=False,
    )
    assert resp.status_code == 303, resp.text  # 307 would re-POST to the home page
    assert resp.headers["location"] == "/"
    assert apple.calls == [{"code": "C1", "user_json": user}]
    me = client.get("/api/app/me")
    assert me.status_code == 200 and me.json()["name"] == "Ada Lovelace"


def test_native_apple_sign_in_returns_the_deep_link(tmp_path: Path) -> None:
    client, _ = _app(tmp_path)
    resp = client.get("/api/app/auth/login?provider=apple&platform=native", follow_redirects=False)
    state = parse_qs(urlparse(resp.headers["location"]).query)["state"][0]
    resp = client.post(
        "/api/app/auth/callback",
        content=f"code=C2&state={state}".encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert (
        resp.headers["location"].startswith("closelistening")
        and "#token=" in resp.headers["location"]
    )


def test_a_cancelled_apple_sign_in_is_a_400(tmp_path: Path) -> None:
    client, _ = _app(tmp_path)
    resp = client.post(
        "/api/app/auth/callback",
        content=b"error=user_cancelled_authorize&state=x",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_the_provider_comes_from_the_verified_state_not_from_the_request(tmp_path: Path) -> None:
    # A flow started with Google must complete with Google even when it arrives as a form POST.
    client, apple = _app(tmp_path)
    state = _state(client, None)
    resp = client.post(
        "/api/app/auth/callback",
        content=f"code=C3&state={state}&provider=apple".encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert apple.calls == []
    assert client.get("/api/app/me").json()["email"] == "jane@example.com"


def test_health_lists_the_configured_providers(tmp_path: Path) -> None:
    client, _ = _app(tmp_path)
    assert client.get("/api/health").json()["auth_providers"] == ["google", "apple"]


# --- Observability: every way through the OAuth flow leaves one event (#2275) ---------------------

import logging  # noqa: E402

from podcast_scraper.server.app_access import AccessPolicy as _Policy  # noqa: E402
from podcast_scraper.server.app_oauth import OAuthError  # noqa: E402


def _events(caplog: pytest.LogCaptureFixture, kind: str) -> list[dict]:
    out = []
    for rec in caplog.records:
        try:
            payload = json.loads(rec.getMessage())
        except ValueError:
            continue
        if payload.get("event_type") == kind:
            out.append(payload)
    return out


def _form_post(client: TestClient, body: str):
    return client.post(
        "/api/app/auth/callback",
        content=body.encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        follow_redirects=False,
    )


def test_a_successful_apple_sign_in_logs_start_and_ok_once_new_then_returning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    client, _ = _app(tmp_path)
    for _ in range(2):
        state = _state(client, "apple")
        assert _form_post(client, f"code=C&state={state}").status_code == 303
    started = _events(caplog, "oauth_login_started")
    done = _events(caplog, "oauth_callback")
    assert [(e["provider"], e["platform"]) for e in started] == [("apple", "web"), ("apple", "web")]
    assert [(e["outcome"], e["provider"], e["new_account"]) for e in done] == [
        ("ok", "apple", True),
        ("ok", "apple", False),
    ]
    assert all(len(e["email_fp"]) == 16 for e in done)


def test_events_never_carry_the_address_a_code_or_a_token(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    client, _ = _app(tmp_path)
    state = _state(client, "apple")
    _form_post(client, f"code=SECRET-CODE&state={state}")
    blob = json.dumps(_events(caplog, "oauth_login_started") + _events(caplog, "oauth_callback"))
    for leaked in ("privaterelay.appleid.com", "SECRET-CODE", state):
        assert leaked not in blob


def test_an_exchange_failure_is_logged_with_its_reason_not_a_silent_502(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    client, apple = _app(tmp_path)

    def boom(**_: object) -> OAuthIdentity:
        raise OAuthError("Apple id_token rejected: Signature verification failed")

    apple.exchange_code = boom  # type: ignore[method-assign]
    state = _state(client, "apple")
    assert _form_post(client, f"code=C&state={state}").status_code == 502
    (event,) = _events(caplog, "oauth_callback")
    assert (event["outcome"], event["provider"]) == ("exchange_failed", "apple")
    assert "Signature verification failed" in event["reason"]


def test_not_allowed_cancelled_and_bad_state_each_say_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    client, _ = _app(tmp_path)
    client.app.state.access_policy = _Policy("allowlist", frozenset(), frozenset())  # type: ignore[attr-defined]
    state = _state(client, "apple")
    assert _form_post(client, f"code=C&state={state}").status_code == 403
    state = _state(client, "apple")
    assert _form_post(client, f"error=user_cancelled_authorize&state={state}").status_code == 400
    assert _form_post(client, "code=C&state=forged").status_code == 400
    outcomes = [(e["outcome"], e.get("provider")) for e in _events(caplog, "oauth_callback")]
    assert outcomes == [("not_allowed", "apple"), ("cancelled", "apple"), ("state_invalid", None)]

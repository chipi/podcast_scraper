"""Integration tests for the store-review code sign-in (``app_review_signin``).

One configured address may sign in with a code instead of an emailed link, so a Play or App Store
reviewer can get in. What must hold: it does not exist unless configured, it opens that one address
and nothing else, wrong codes are paused, and past the code it is the email link's own path (the
allowlist, the account, the session).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_review_signin import review_signin_from_env, ReviewSignin
from podcast_scraper.server.routes import app_auth

pytestmark = [pytest.mark.integration]

_SECRET = "test-secret"
_REVIEW = "review@example.test"
_CODE = "correct-horse-battery-staple-42"
_URL = "/api/app/auth/email/code"


def _client(
    tmp_path: Path, *, configured: bool = True, allowed: frozenset[str] | None = None
) -> tuple[TestClient, Path]:
    app = create_app(tmp_path, static_dir=False)
    data_dir = tmp_path / "appdata"
    app.state.session_secret = _SECRET
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy(
        "allowlist", allowed if allowed is not None else frozenset({_REVIEW}), frozenset()
    )
    app.state.review_signin = (
        ReviewSignin(_REVIEW, hashlib.sha256(_CODE.encode()).hexdigest()) if configured else None
    )
    return TestClient(app, follow_redirects=False), data_dir


@pytest.fixture(autouse=True)
def _no_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most tests make one wrong attempt and then a right one; the pause has its own test."""
    monkeypatch.setattr(app_auth, "_REVIEW_FAILURE_PAUSE_S", 0)


def test_not_configured_the_route_does_not_exist(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, configured=False)
    assert client.post(_URL, json={"email": _REVIEW, "code": _CODE}).status_code == 404


def test_the_right_code_signs_in_the_web_with_a_cookie_and_lands_new_on_the_profile(
    tmp_path: Path,
) -> None:
    client, _ = _client(tmp_path)
    r = client.post(_URL, json={"email": " Review@Example.test ", "code": _CODE})
    assert r.status_code == 200
    assert r.json() == {"dest": "/profile?welcome=1"}
    assert r.headers["cache-control"] == "no-store"
    assert app_sessions.SESSION_COOKIE in r.cookies
    me = client.get("/api/app/me")
    assert me.status_code == 200 and me.json()["email"] == _REVIEW

    again = client.post(_URL, json={"email": _REVIEW, "code": _CODE})
    assert again.json() == {"dest": "/"}


def test_native_gets_the_token_in_the_body(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    r = client.post(_URL, json={"email": _REVIEW, "code": _CODE, "platform": "native"})
    assert r.status_code == 200
    body = r.json()
    assert body["new"] is True
    assert app_sessions.SESSION_COOKIE not in r.cookies
    me = client.get("/api/app/me", headers={"Authorization": f"Bearer {body['token']}"})
    assert me.status_code == 200 and me.json()["email"] == _REVIEW


def test_a_wrong_code_and_a_wrong_address_get_the_same_answer(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, allowed=frozenset({_REVIEW, "someone@example.test"}))
    wrong_code = client.post(_URL, json={"email": _REVIEW, "code": "nope"})
    wrong_email = client.post(_URL, json={"email": "someone@example.test", "code": _CODE})
    assert wrong_code.status_code == wrong_email.status_code == 400
    assert wrong_code.json() == wrong_email.json()
    assert app_sessions.SESSION_COOKIE not in wrong_email.cookies


def test_after_a_wrong_code_the_next_attempt_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_auth, "_REVIEW_FAILURE_PAUSE_S", 60)
    client, _ = _client(tmp_path)
    assert client.post(_URL, json={"email": _REVIEW, "code": "nope"}).status_code == 400
    # Even the right code: the pause is what makes guessing hopeless.
    assert client.post(_URL, json={"email": _REVIEW, "code": _CODE}).status_code == 429


def test_the_allowlist_still_decides(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, allowed=frozenset())
    r = client.post(_URL, json={"email": _REVIEW, "code": _CODE})
    assert r.status_code == 403
    assert app_sessions.SESSION_COOKIE not in r.cookies


def test_it_opens_the_same_account_as_the_emailed_link(tmp_path: Path) -> None:
    """One account per address, whichever door: the code must not create a second one."""
    client, _ = _client(tmp_path)
    via_code = client.post(_URL, json={"email": _REVIEW, "code": _CODE, "platform": "native"})
    code_payload = app_sessions.verify(via_code.json()["token"], _SECRET)
    assert code_payload is not None

    from podcast_scraper.server import app_magic_link

    token, _ = app_magic_link.issue(_REVIEW, _SECRET)
    via_link = client.get(f"/api/app/auth/email/verify?token={token}&platform=native")
    link_session = via_link.headers["location"].split("token=", 1)[1].split("&", 1)[0]
    link_payload = app_sessions.verify(link_session, _SECRET)
    assert link_payload is not None
    assert link_payload["user_id"] == code_payload["user_id"]


@pytest.mark.parametrize(
    ("email", "digest"),
    [
        ("", "a" * 64),
        ("review@example.test", ""),
        ("review@example.test", "a" * 63),
        ("review@example.test", "z" * 64),
        ("not-an-address", "a" * 64),
    ],
)
def test_half_configured_is_off(monkeypatch: pytest.MonkeyPatch, email: str, digest: str) -> None:
    monkeypatch.setenv("APP_REVIEW_SIGNIN_EMAIL", email)
    monkeypatch.setenv("APP_REVIEW_SIGNIN_CODE_SHA256", digest)
    assert review_signin_from_env() is None


def test_configured_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_REVIEW_SIGNIN_EMAIL", " Review@Example.test")
    monkeypatch.setenv("APP_REVIEW_SIGNIN_CODE_SHA256", hashlib.sha256(_CODE.encode()).hexdigest())
    review = review_signin_from_env()
    assert review is not None and review.email == _REVIEW
    assert review.matches(_REVIEW, f" {_CODE} ")
    assert not review.matches(_REVIEW, _CODE.upper())

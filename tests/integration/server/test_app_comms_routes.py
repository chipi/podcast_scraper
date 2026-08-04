"""Route tests for the delivery-consent surface ``/api/app/comms`` (#1414)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = [pytest.mark.integration]


def _authed(
    tmp_path: Path, *, provider: str = "stub", email: str = "j@x.com", vapid: str = ""
) -> TestClient:
    app = create_app(tmp_path, static_dir=False)
    data_dir = tmp_path / "appdata"
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    app.state.vapid_public_key = vapid
    user = get_or_create_user(data_dir, provider=provider, subject="s1", email=email, name="J")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def test_vapid_key_503_when_unconfigured(tmp_path: Path) -> None:
    assert _authed(tmp_path).get("/api/app/push/vapid-key").status_code == 503


def test_vapid_key_returned_when_configured(tmp_path: Path) -> None:
    resp = _authed(tmp_path, vapid="BPublicKeyValue").get("/api/app/push/vapid-key")
    assert resp.status_code == 200 and resp.json()["key"] == "BPublicKeyValue"


def test_push_subscribe_stores_and_enables(tmp_path: Path) -> None:
    client = _authed(tmp_path)
    sub = {"endpoint": "https://push.invalid/x", "keys": {"p256dh": "p", "auth": "a"}}
    resp = client.post("/api/app/push/subscribe", json=sub)
    assert resp.status_code == 200 and resp.json()["count"] == 1
    assert client.get("/api/app/comms").json()["push"]["enabled"] is True


def test_push_unsubscribe_last_disables(tmp_path: Path) -> None:
    client = _authed(tmp_path)
    client.post(
        "/api/app/push/subscribe",
        json={"endpoint": "https://push.invalid/x", "keys": {"auth": "a"}},
    )
    resp = client.request(
        "DELETE", "/api/app/push/subscribe", json={"endpoint": "https://push.invalid/x"}
    )
    assert resp.status_code == 200 and resp.json()["count"] == 0
    assert client.get("/api/app/comms").json()["push"]["enabled"] is False


def test_get_comms_defaults_when_unset(tmp_path: Path) -> None:
    resp = _authed(tmp_path).get("/api/app/comms")
    assert resp.status_code == 200
    body = resp.json()
    assert body["digest"]["enabled"] is False
    assert body["digest"]["cadence"] == "weekly"
    assert body["push"]["enabled"] is False
    assert body["unsubscribe_ref"] is None
    # stub provider is not google → email delivery not permitted
    assert body["email_verified"] is False


def test_google_user_is_email_verified(tmp_path: Path) -> None:
    body = _authed(tmp_path, provider="google", email="u@gmail.com").get("/api/app/comms").json()
    assert body["email_verified"] is True


def test_put_enables_digest_and_mints_ref(tmp_path: Path) -> None:
    client = _authed(tmp_path)
    resp = client.put("/api/app/comms", json={"digest": {"enabled": True, "cadence": "daily"}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["digest"]["enabled"] is True
    assert body["digest"]["cadence"] == "daily"
    assert isinstance(body["unsubscribe_ref"], str) and body["unsubscribe_ref"]

    # persisted across requests
    again = client.get("/api/app/comms").json()
    assert again["digest"]["enabled"] is True
    assert again["unsubscribe_ref"] == body["unsubscribe_ref"]


def test_put_rejects_out_of_range_hour(tmp_path: Path) -> None:
    resp = _authed(tmp_path).put("/api/app/comms", json={"digest": {"hour": 99}})
    assert resp.status_code == 422


def test_public_unsubscribe_disables_digest(tmp_path: Path) -> None:
    client = _authed(tmp_path)
    ref = client.put("/api/app/comms", json={"digest": {"enabled": True}}).json()["unsubscribe_ref"]

    # No auth cookie — the ref is the capability.
    public = TestClient(client.app)
    resp = public.post("/api/app/comms/unsubscribe", params={"ref": ref})
    assert resp.status_code == 200
    assert resp.json() == {"unsubscribed": True}

    assert client.get("/api/app/comms").json()["digest"]["enabled"] is False


def test_public_unsubscribe_unknown_ref(tmp_path: Path) -> None:
    public = TestClient(_authed(tmp_path).app)
    resp = public.post("/api/app/comms/unsubscribe", params={"ref": "nope"})
    assert resp.status_code == 200
    assert resp.json() == {"unsubscribed": False}


def test_comms_requires_auth(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    app.state.app_data_dir = tmp_path / "appdata"
    app.state.session_secret = "test-secret"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    resp = TestClient(app).get("/api/app/comms")
    assert resp.status_code in (401, 403)

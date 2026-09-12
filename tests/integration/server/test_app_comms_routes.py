"""Route tests for the delivery-consent surface ``/api/app/comms`` — matrix (#1414 → wave-I)."""

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


def test_push_subscribe_registers_endpoint(tmp_path: Path) -> None:
    # Registration and consent are separate gates: subscribing stores the endpoint but does NOT
    # flip any per-type push toggle — the client PUTs the toggle it wants.
    client = _authed(tmp_path)
    sub = {"endpoint": "https://push.invalid/x", "keys": {"p256dh": "p", "auth": "a"}}
    resp = client.post("/api/app/push/subscribe", json=sub)
    assert resp.status_code == 200 and resp.json()["count"] == 1
    types = client.get("/api/app/comms").json()["types"]
    assert all(not types[t]["push"] for t in types)  # not auto-enabled


def test_push_unsubscribe_last_disables_push_everywhere(tmp_path: Path) -> None:
    client = _authed(tmp_path)
    # Enable push for a type, then register + drop the last subscription.
    client.put("/api/app/comms", json={"types": {"new_episodes": {"push": True}}})
    client.post(
        "/api/app/push/subscribe",
        json={"endpoint": "https://push.invalid/x", "keys": {"auth": "a"}},
    )
    resp = client.request(
        "DELETE", "/api/app/push/subscribe", json={"endpoint": "https://push.invalid/x"}
    )
    assert resp.status_code == 200 and resp.json()["count"] == 0
    types = client.get("/api/app/comms").json()["types"]
    assert all(not types[t]["push"] for t in types)  # unreachable → all push off


def test_get_comms_defaults_when_unset(tmp_path: Path) -> None:
    resp = _authed(tmp_path).get("/api/app/comms")
    assert resp.status_code == 200
    body = resp.json()
    for ntype in ("digest", "new_episodes", "product"):
        assert body["types"][ntype] == {"email": False, "push": False, "in_app": True}
    assert body["digest_schedule"]["cadence"] == "weekly"
    assert body["unsubscribe_ref"] is None
    # stub provider is not google → email delivery not permitted
    assert body["email_verified"] is False


def test_google_user_is_email_verified(tmp_path: Path) -> None:
    body = _authed(tmp_path, provider="google", email="u@gmail.com").get("/api/app/comms").json()
    assert body["email_verified"] is True


def test_put_enables_digest_email_and_mints_ref(tmp_path: Path) -> None:
    client = _authed(tmp_path)
    resp = client.put(
        "/api/app/comms",
        json={"types": {"digest": {"email": True}}, "digest_schedule": {"cadence": "daily"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["types"]["digest"]["email"] is True
    assert body["digest_schedule"]["cadence"] == "daily"
    assert isinstance(body["unsubscribe_ref"], str) and body["unsubscribe_ref"]

    # persisted across requests
    again = client.get("/api/app/comms").json()
    assert again["types"]["digest"]["email"] is True
    assert again["unsubscribe_ref"] == body["unsubscribe_ref"]


def test_put_rejects_out_of_range_hour(tmp_path: Path) -> None:
    resp = _authed(tmp_path).put("/api/app/comms", json={"digest_schedule": {"hour": 99}})
    assert resp.status_code == 422


def test_put_persists_timezone_and_get_returns_it(tmp_path: Path) -> None:
    # #2041: the client PUTs the auto-detected IANA tz; GET round-trips it. Default is "".
    client = _authed(tmp_path)
    assert client.get("/api/app/comms").json()["timezone"] == ""
    client.put("/api/app/comms", json={"timezone": "America/New_York"})
    assert client.get("/api/app/comms").json()["timezone"] == "America/New_York"
    # A timezone-only PUT leaves the matrix untouched (partial merge).
    types = client.get("/api/app/comms").json()["types"]
    assert types["digest"]["in_app"] is True


def test_public_unsubscribe_disables_digest_email(tmp_path: Path) -> None:
    client = _authed(tmp_path)
    ref = client.put("/api/app/comms", json={"types": {"digest": {"email": True}}}).json()[
        "unsubscribe_ref"
    ]

    # No auth cookie — the ref is the capability.
    public = TestClient(client.app)
    resp = public.post("/api/app/comms/unsubscribe", params={"ref": ref})
    assert resp.status_code == 200
    assert resp.json() == {"unsubscribed": True}

    assert client.get("/api/app/comms").json()["types"]["digest"]["email"] is False


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


def test_push_unsubscribe_partial_keeps_others(tmp_path: Path) -> None:
    client = _authed(tmp_path)
    client.put("/api/app/comms", json={"types": {"new_episodes": {"push": True}}})
    client.post(
        "/api/app/push/subscribe",
        json={"endpoint": "https://push.invalid/a", "keys": {"auth": "a"}},
    )
    client.post(
        "/api/app/push/subscribe",
        json={"endpoint": "https://push.invalid/b", "keys": {"auth": "b"}},
    )
    resp = client.request(
        "DELETE", "/api/app/push/subscribe", json={"endpoint": "https://push.invalid/a"}
    )
    # A subscription remains → push consent is NOT torn down.
    assert resp.status_code == 200 and resp.json()["count"] == 1
    assert client.get("/api/app/comms").json()["types"]["new_episodes"]["push"] is True


def test_unsubscribe_get_page_does_not_mutate(tmp_path: Path) -> None:
    # Email-link GET must render a confirm form and NOT unsubscribe (prefetch-safe).
    client = _authed(tmp_path)
    ref = client.put("/api/app/comms", json={"types": {"digest": {"email": True}}}).json()[
        "unsubscribe_ref"
    ]
    page = TestClient(client.app).get("/api/app/comms/unsubscribe", params={"ref": ref})
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "<form method=post" in page.text and ref in page.text
    # still subscribed — the GET did not mutate
    assert client.get("/api/app/comms").json()["types"]["digest"]["email"] is True
    # the POST (form submit / RFC-8058 one-click) does mutate
    assert TestClient(client.app).post(
        "/api/app/comms/unsubscribe", params={"ref": ref}
    ).json() == {"unsubscribed": True}
    assert client.get("/api/app/comms").json()["types"]["digest"]["email"] is False

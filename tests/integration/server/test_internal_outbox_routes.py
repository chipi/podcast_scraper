"""Route tests for the internal outbox seam ``/internal/outbox/*`` (#1415)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_comms_store, app_outbox_store
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy

_UID = "u_0123456789abcdef01234567"
_TOKEN = "tok-test-01"


def _app(tmp_path: Path, *, token: str = _TOKEN):
    app = create_app(tmp_path, static_dir=False)
    data_dir = tmp_path / "appdata"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    app.state.internal_outbox_token = token
    return app, data_dir


def _envelope(eid: str = "dgst_x_" + _UID, channel: str = "email") -> dict:
    return {
        "schema_version": "1",
        "id": eid,
        "user_id": _UID,
        "channel": channel,
        "template": "your-week-digest.v1",
        "recipient": {"email": "u@x.com", "email_verified": True},
        "consent_snapshot": {"digest_enabled": True, "cadence": "weekly", "unsubscribe_ref": "r"},
        "payload": {"sections": []},
        "created_at": "2026-08-04T00:00:00Z",
    }


def test_requires_token(tmp_path: Path) -> None:
    app, _ = _app(tmp_path)
    client = TestClient(app)
    assert client.get("/internal/outbox/pending", params={"channel": "email"}).status_code == 401
    ok = client.get(
        "/internal/outbox/pending",
        params={"channel": "email"},
        headers={"X-Internal-Token": _TOKEN},
    )
    assert ok.status_code == 200


def test_disabled_when_unconfigured(tmp_path: Path) -> None:
    app, _ = _app(tmp_path, token="")
    resp = TestClient(app).get(
        "/internal/outbox/pending",
        params={"channel": "email"},
        headers={"X-Internal-Token": "anything"},
    )
    assert resp.status_code == 503


def test_pending_returns_consented_envelopes(tmp_path: Path) -> None:
    app, data_dir = _app(tmp_path)
    app_comms_store.set_comms(data_dir, _UID, digest={"enabled": True})
    app_outbox_store.enqueue(data_dir, _envelope())
    resp = TestClient(app).get(
        "/internal/outbox/pending",
        params={"channel": "email"},
        headers={"X-Internal-Token": _TOKEN},
    )
    assert resp.status_code == 200
    assert [e["id"] for e in resp.json()["envelopes"]] == [_envelope()["id"]]


def test_status_idempotent_and_suppresses(tmp_path: Path) -> None:
    app, data_dir = _app(tmp_path)
    app_comms_store.set_comms(data_dir, _UID, digest={"enabled": True})
    app_outbox_store.enqueue(data_dir, _envelope())
    client = TestClient(app)
    h = {"X-Internal-Token": _TOKEN}

    r1 = client.post(
        f"/internal/outbox/{_envelope()['id']}/status", json={"status": "bounced"}, headers=h
    )
    assert r1.status_code == 200 and r1.json()["status"] == "bounced"
    # idempotent: a second (conflicting) terminal report is a no-op returning the stored status
    r2 = client.post(
        f"/internal/outbox/{_envelope()['id']}/status", json={"status": "delivered"}, headers=h
    )
    assert r2.json()["status"] == "bounced"
    # bounce suppressed the digest
    assert app_comms_store.get_comms(data_dir, _UID)["digest"]["enabled"] is False


def test_status_unknown_id(tmp_path: Path) -> None:
    app, _ = _app(tmp_path)
    resp = TestClient(app).post(
        "/internal/outbox/nope/status",
        json={"status": "delivered"},
        headers={"X-Internal-Token": _TOKEN},
    )
    assert resp.status_code == 200 and resp.json()["status"] == "unknown"


def test_status_rejects_bad_enum(tmp_path: Path) -> None:
    app, _ = _app(tmp_path)
    resp = TestClient(app).post(
        "/internal/outbox/x/status",
        json={"status": "pending"},
        headers={"X-Internal-Token": _TOKEN},
    )
    assert resp.status_code == 422

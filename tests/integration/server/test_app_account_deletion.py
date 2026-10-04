"""Account deletion (#2273) — self-service route, the full purge, Apple revocation, retention."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.server import (
    app_account_deletion,
    app_mcp_tokens,
    app_oauth_server,
    app_outbox_store,
)
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_oauth import MockOAuthProvider, OAuthError
from podcast_scraper.server.app_user_store import get_user

pytestmark = [pytest.mark.integration, pytest.mark.app]


def _app(tmp_path: Path):
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    app.state.oauth_provider = MockOAuthProvider()
    app.state.admin_emails = frozenset({"boss@e2e.local"})
    return app


def _login(app, who: str) -> TestClient:
    client = TestClient(app)
    client.get("/api/app/auth/login", params={"as": who}, follow_redirects=True)
    assert client.get("/api/app/me").status_code == 200
    return client


def _uid(client: TestClient) -> str:
    return str(client.get("/api/app/me").json()["user_id"])


def _envelope(eid: str, user_id: str, email: str, **extra: object) -> dict:
    return {
        "id": eid,
        "user_id": user_id,
        "channel": "email",
        "type": extra.pop("type", "digest"),
        "recipient": {"email": email},
        **extra,
    }


def _delete(client: TestClient, confirm: str = "DELETE"):
    return client.request("DELETE", "/api/app/me", json={"confirm": confirm})


def test_without_the_typed_word_nothing_is_deleted(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client = _login(app, "ada")
    for word in ("delete", "", "DELETE "):
        assert _delete(client, word).status_code == 400
    assert get_user(app.state.app_data_dir, _uid(client)) is not None


def test_signed_out_cannot_delete(tmp_path: Path) -> None:
    assert _delete(TestClient(_app(tmp_path))).status_code == 401


def test_delete_removes_the_account_and_everything_outside_it_that_names_it(tmp_path: Path) -> None:
    app = _app(tmp_path)
    data = app.state.app_data_dir
    ada = _login(app, "ada")
    bob = _login(app, "bob")
    ada_id, bob_id = _uid(ada), _uid(bob)
    # Outbox: ada's digest, a sign-in email to her address (empty user_id), and bob's digest.
    app_outbox_store.enqueue(data, _envelope("dgst_1_" + ada_id, ada_id, "ada@e2e.local"))
    app_outbox_store.enqueue(data, _envelope("auth_x", "", "ADA@e2e.local", type="auth_link"))
    app_outbox_store.enqueue(data, _envelope("dgst_1_" + bob_id, bob_id, "bob@e2e.local"))
    # MCP: a personal token and an OAuth grant + consent for each.
    for uid in (ada_id, bob_id):
        app_mcp_tokens.create_token(data, uid, "laptop")
        app_oauth_server.remember_consent(data, user_id=uid, client_id="c1", scope="mcp:read")
        app_oauth_server.create_authorization_code(
            data, user_id=uid, client_id="c1", redirect_uri="https://x/cb", code_challenge="cc"
        )

    resp = _delete(ada)
    assert resp.status_code == 204
    assert "session" in resp.headers.get("set-cookie", "").lower()

    assert get_user(data, ada_id) is None
    assert not (data / "users" / ada_id).exists()
    remaining = [
        json.loads(p.read_text())["envelope"]["id"] for p in (data / "outbox").glob("*.json")
    ]
    assert remaining == ["dgst_1_" + bob_id], "only bob's envelope may survive"
    index = json.loads((data / "mcp_token_index.json").read_text())
    assert set(index.values()) == {bob_id}
    grants = json.loads((data / "oauth_grants.json").read_text())
    assert {g["user_id"] for g in grants.values()} == {bob_id}
    consents = json.loads((data / "oauth_consents.json").read_text())
    assert all(k.startswith(bob_id) for k in consents)
    # bob is untouched
    assert get_user(data, bob_id) is not None
    assert bob.get("/api/app/me").status_code == 200


def test_the_deleted_session_no_longer_works(tmp_path: Path) -> None:
    client = _login(_app(tmp_path), "ada")
    assert _delete(client).status_code == 204
    assert client.get("/api/app/me").status_code == 401


def test_a_late_bounce_or_a_leaked_token_does_not_resurrect_the_directory(tmp_path: Path) -> None:
    app = _app(tmp_path)
    data = app.state.app_data_dir
    client = _login(app, "ada")
    uid = _uid(client)
    token, _ = app_mcp_tokens.create_token(data, uid, "laptop")
    # Re-point a still-present index entry at the user, as a stale index would after a crash.
    assert _delete(client).status_code == 204
    index_path = data / "mcp_token_index.json"
    index = json.loads(index_path.read_text())
    index[app_mcp_tokens._hash(token)] = uid
    index_path.write_text(json.dumps(index))
    assert app_mcp_tokens.verify_token(data, token) is None
    # A bounce for an envelope enqueued (and claimed by the worker) before the deletion.
    app_outbox_store.enqueue(data, _envelope("late_" + uid, uid, "ada@e2e.local"))
    app_outbox_store.record_status(data, "late_" + uid, "bounced", "mailbox gone")
    assert not (data / "users" / uid).exists()


def test_events_and_audit_say_it_happened_without_the_address(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    app = _app(tmp_path)
    app.state.audit_path = tmp_path / "audit.jsonl"
    client = _login(app, "ada")
    uid = _uid(client)
    assert _delete(client).status_code == 204
    events = [
        json.loads(r.getMessage())
        for r in caplog.records
        if r.getMessage().startswith("{") and '"account_deleted"' in r.getMessage()
    ]
    assert len(events) == 1
    assert events[0]["initiated_by"] == "self" and events[0]["user_dir"] is True
    assert "ada@e2e.local" not in json.dumps(events)
    audit = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert {"event": "user.self_delete", "user": uid, "provider": "mock"}.items() <= audit[
        -1
    ].items()


class _Apple:
    name = "apple"

    def __init__(self, fail: bool = False) -> None:
        self.revoked: list[str] = []
        self.fail = fail

    def revoke(self, token: str) -> None:
        if self.fail:
            raise OAuthError("Apple token revocation failed: 503")
        self.revoked.append(token)


def _apple_user(app) -> tuple[TestClient, str]:
    """A signed-in user whose stored identity is Apple's, with a refresh token on file."""
    client = _login(app, "appleuser")
    uid = _uid(client)
    profile = app.state.app_data_dir / "users" / uid / "profile.json"
    doc = json.loads(profile.read_text())
    doc["provider"] = "apple"
    profile.write_text(json.dumps(doc))
    app_account_deletion.store_apple_refresh_token(app.state.app_data_dir, uid, "r.apple-token")
    return client, uid


def test_an_apple_account_has_its_apple_tokens_revoked(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    app = _app(tmp_path)
    apple = _Apple()
    app.state.oauth_providers = {"mock": app.state.oauth_provider, "apple": apple}
    client, _ = _apple_user(app)
    assert _delete(client).status_code == 204
    assert apple.revoked == ["r.apple-token"]
    (event,) = [
        json.loads(r.getMessage()) for r in caplog.records if '"account_deleted"' in r.getMessage()
    ]
    assert event["apple_revoked"] is True
    assert "r.apple-token" not in json.dumps(event)


def test_a_failed_apple_revoke_still_deletes_and_says_why(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    app = _app(tmp_path)
    app.state.oauth_providers = {"mock": app.state.oauth_provider, "apple": _Apple(fail=True)}
    client, uid = _apple_user(app)
    assert _delete(client).status_code == 204
    assert get_user(app.state.app_data_dir, uid) is None
    (event,) = [
        json.loads(r.getMessage()) for r in caplog.records if '"account_deleted"' in r.getMessage()
    ]
    assert event["apple_revoked"] is False and "503" in event["apple_reason"]


def test_the_admin_route_runs_the_same_full_purge(tmp_path: Path) -> None:
    app = _app(tmp_path)
    data = app.state.app_data_dir
    target = _login(app, "ada")
    uid = _uid(target)
    app_outbox_store.enqueue(data, _envelope("dgst_9_" + uid, uid, "ada@e2e.local"))
    admin = _login(app, "boss")
    assert admin.request("DELETE", f"/api/app/admin/users/{uid}").status_code == 204
    assert list((data / "outbox").glob("*.json")) == []


def test_finished_envelopes_age_out_and_sign_in_ones_sooner(tmp_path: Path) -> None:
    data = tmp_path / "appdata"
    now = int(time.time())
    for eid, etype, age in (
        ("old_digest", "digest", 31 * 86400),
        ("new_digest", "digest", 2 * 86400),
        ("old_signin", "auth_link", 2 * 86400),
    ):
        app_outbox_store.enqueue(data, _envelope(eid, "u1", "a@x.org", type=etype))
        app_outbox_store.record_status(data, eid, "delivered")
        path = app_outbox_store._envelope_path(data, eid)
        record = json.loads(path.read_text())
        record["updated_at"] = now - age
        path.write_text(json.dumps(record))
    app_outbox_store.enqueue(data, _envelope("pending_old", "u1", "a@x.org"))
    app_outbox_store.list_pending(data, channel="email", now=now)
    left = sorted(
        json.loads(p.read_text())["envelope"]["id"] for p in (data / "outbox").glob("*.json")
    )
    assert left == ["new_digest", "pending_old"], "pending is never pruned; finished ones age out"

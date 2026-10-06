"""The four admin-surface gaps found 2026-10-05, each pinned so it cannot reopen.

1. ``POST /api/corpus/topic-clusters/rebuild`` was outside every operator base — a CREATOR could
   start it. Now operator-gated like ``/api/index/rebuild``.
2. Operator writes were audited as ``actor: "operator"`` with no identity. Now ``via`` + ``by``.
3. ``PUT /api/app/ranking-config`` wrote no audit record. Now who + before + after.
5. Cookie-authenticated writes rested on ``SameSite=Lax`` alone. Now a foreign ``Origin`` is
   refused (``app_csrf``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_user_store import create_user

pytestmark = [pytest.mark.integration]


def _app(tmp_path: Path, *, key: str = "", auth: bool = True):
    app = create_app(tmp_path, static_dir=False, enable_feeds_api=True)
    app.state.operator_api_key = key
    app.state.audit_path = tmp_path / "audit.jsonl"
    app.state.session_secret = "test-secret" if auth else ""
    app.state.app_data_dir = (tmp_path / "appdata") if auth else None
    return app


def _as(app, role: str) -> tuple[TestClient, str]:
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
    return client, user.user_id


def _audit(tmp_path: Path) -> List[Dict[str, Any]]:
    path = tmp_path / "audit.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --- 1. topic-clusters rebuild is operator surface ------------------------------------------------

_REBUILD = "/api/corpus/topic-clusters/rebuild"


@pytest.mark.parametrize("role", ["creator", "listener"])
def test_a_non_admin_cannot_start_a_topic_cluster_rebuild(tmp_path: Path, role: str) -> None:
    app = _app(tmp_path)
    client, _ = _as(app, role)
    assert client.post(_REBUILD, params={"path": str(tmp_path)}).status_code == 403


def test_an_admin_passes_the_guard_for_a_topic_cluster_rebuild(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client, _ = _as(app, "admin")
    # The guard lets it through; what the route then does (202 / 503 without LanceDB) is its own.
    assert client.post(_REBUILD, params={"path": str(tmp_path)}).status_code != 403


def test_the_operator_key_passes_the_guard_for_a_topic_cluster_rebuild(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path, key="k"))
    r = client.post(_REBUILD, params={"path": str(tmp_path)}, headers={"X-Operator-Key": "k"})
    assert r.status_code != 403


# --- 2. operator writes record WHO -----------------------------------------------------------------


def test_an_operator_write_by_an_admin_session_records_the_user(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client, user_id = _as(app, "admin")
    client.put("/api/feeds", json={"feeds": []})
    allowed = [r for r in _audit(tmp_path) if r.get("outcome") == "allowed"]
    assert allowed and allowed[-1]["via"] == "admin_session"
    assert allowed[-1]["by"] == user_id


def test_an_operator_write_by_key_records_the_key(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path, key="k"))
    client.put("/api/feeds", json={"feeds": []}, headers={"X-Operator-Key": "k"})
    allowed = [r for r in _audit(tmp_path) if r.get("outcome") == "allowed"]
    assert allowed and allowed[-1]["via"] == "operator_key"
    assert "by" not in allowed[-1]


def test_a_refused_operator_write_records_that_no_credential_matched(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client, _ = _as(app, "creator")
    client.put("/api/feeds", json={"feeds": []})
    denied = [r for r in _audit(tmp_path) if r.get("outcome") == "denied"]
    assert denied and denied[-1]["via"] == "none"


# --- 3. ranking-config writes are audited ----------------------------------------------------------


def test_a_ranking_config_change_records_who_and_before_and_after(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client, user_id = _as(app, "admin")
    before = client.get("/api/app/ranking-config").json()
    changed = json.loads(json.dumps(before))

    def bump(node: Any) -> bool:  # change the first number, so before != after is observable
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    node[k] = v + 1
                    return True
                if bump(v):
                    return True
        if isinstance(node, list):
            return any(bump(item) for item in node)
        return False

    assert bump(changed), "the ranking config has no numeric field to change"
    r = client.put("/api/app/ranking-config", json=changed)
    assert r.status_code == 200
    assert r.json() != before, "the change must be visible, or before/after cannot be told apart"
    rows = [x for x in _audit(tmp_path) if x.get("action") == "ranking_config_set"]
    assert len(rows) == 1
    assert rows[0]["by"] == user_id
    assert rows[0]["before"] == before and rows[0]["after"] == r.json()


# --- 5. cross-site cookie writes are refused -------------------------------------------------------


def test_a_cookie_write_from_a_foreign_origin_is_refused(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client, _ = _as(app, "admin")
    r = client.put("/api/app/ranking-config", json={}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert r.json()["detail"] == "Cross-site request refused."
    assert any(x.get("action") == "cross_site_write_refused" for x in _audit(tmp_path))


def test_a_foreign_referer_is_refused_when_there_is_no_origin(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client, _ = _as(app, "admin")
    r = client.put(
        "/api/app/ranking-config", json={}, headers={"Referer": "https://evil.example/page"}
    )
    assert r.status_code == 403


def test_an_opaque_null_origin_is_refused(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client, _ = _as(app, "admin")
    r = client.put("/api/app/ranking-config", json={}, headers={"Origin": "null"})
    assert r.status_code == 403


def test_a_same_host_origin_passes(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client, _ = _as(app, "admin")
    # TestClient addresses http://testserver; the page that posts is that same host.
    r = client.put("/api/app/ranking-config", json={}, headers={"Origin": "http://testserver"})
    assert r.status_code == 200


def test_a_trusted_cors_origin_passes(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client, _ = _as(app, "admin")
    r = client.put("/api/app/ranking-config", json={}, headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 200


def test_the_proxied_public_host_passes(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client, _ = _as(app, "admin")
    r = client.put(
        "/api/app/ranking-config",
        json={},
        headers={"Origin": "https://player.example", "X-Forwarded-Host": "player.example"},
    )
    assert r.status_code == 200


@pytest.mark.parametrize(
    "extra",
    [
        {"Authorization": "Basic dXNlcjpwYXNz"},  # browser-cached /preview credentials
        {"Authorization": "Bearer whatever"},
        {"X-Operator-Key": "k"},
    ],
)
def test_another_credential_does_not_exempt_a_cookie_write(tmp_path: Path, extra: dict) -> None:
    """The cookie authenticates first, so with the cookie present no other header may switch the
    check off — a browser attaches cached Basic credentials on its own (review finding)."""
    app = _app(tmp_path, key="k")
    client, _ = _as(app, "admin")
    r = client.put(
        "/api/app/ranking-config",
        json={},
        headers={"Origin": "https://evil.example", **extra},
    )
    assert r.status_code == 403
    assert r.json()["detail"] == "Cross-site request refused."


def test_a_bearer_or_key_client_without_the_cookie_is_not_checked(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path, key="k"))
    r = client.put(
        "/api/feeds",
        json={"feeds": []},
        headers={"Origin": "https://evil.example", "X-Operator-Key": "k"},
    )
    assert r.status_code != 403


@pytest.mark.parametrize(
    "native_origin", ["capacitor://localhost", "https://localhost", "http://localhost"]
)
def test_a_native_bearer_write_without_the_cookie_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_origin: str
) -> None:
    """The native shell (iOS capacitor://localhost, Android https/http://localhost) authenticates
    with the session token as a Bearer header and never holds the cookie: the sign-in callbacks
    hand the token back through the deep link and set no session cookie. Its writes must pass the
    cross-site check under prod's config, where the native origins are CORS-allowed but NOT among
    the origins trusted with the cookie."""
    monkeypatch.setenv("PODCAST_ENV", "prod")
    monkeypatch.setenv("PODCAST_SERVE_CORS_ORIGINS", "https://closelistening.app")
    app = _app(tmp_path)
    assert native_origin not in app.state.trusted_origins
    user = create_user(
        app.state.app_data_dir,
        provider="apple",
        subject="native",
        email="native@x.io",
        name="native",
        role="listener",
    )
    token = app_sessions.sign(
        {"user_id": user.user_id, "iat": int(time.time())}, app.state.session_secret
    )
    client = TestClient(app)
    r = client.patch(
        "/api/app/preferences",
        json={"preferences": {"autoplay": True}},
        headers={"Origin": native_origin, "Authorization": f"Bearer {token}"},
    )
    assert app_sessions.SESSION_COOKIE not in client.cookies
    assert r.status_code == 200, r.text
    assert r.json()["preferences"]["autoplay"] is True
    assert not any(x.get("action") == "cross_site_write_refused" for x in _audit(tmp_path))


def test_the_same_host_on_another_port_passes(tmp_path: Path) -> None:
    """nginx's $host drops the port: behind it on :8081, Origin has the port and Host does not."""
    app = _app(tmp_path)
    client, _ = _as(app, "admin")
    r = client.put(
        "/api/app/ranking-config",
        json={},
        headers={"Origin": "http://localhost:8081", "Host": "localhost"},
    )
    assert r.status_code == 200


def test_a_request_with_no_cookie_is_not_checked(tmp_path: Path) -> None:
    # Nothing to forge without the cookie: the route's own auth decides (401 here).
    client = TestClient(_app(tmp_path))
    r = client.put("/api/app/ranking-config", json={}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 401


def test_a_safe_method_is_never_refused(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client, _ = _as(app, "admin")
    r = client.get("/api/app/ranking-config", headers={"Origin": "https://evil.example"})
    assert r.status_code == 200


def test_the_apple_form_post_callback_is_exempt(tmp_path: Path) -> None:
    # Apple posts the callback from appleid.apple.com by design (response_mode=form_post). Asserted
    # by BEHAVIOUR: whatever the callback answers, it must not be the cross-site refusal.
    app = _app(tmp_path)
    client, _ = _as(app, "listener")
    r = client.post(
        "/api/app/auth/callback",
        data={"code": "x", "state": "y"},
        headers={"Origin": "https://appleid.apple.com"},
    )
    assert r.status_code != 404, "the exempt path no longer names the real callback route"
    assert "Cross-site request refused" not in r.text

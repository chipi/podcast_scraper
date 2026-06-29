"""Admin user-management + role isolation under parallel load (#1128).

Mirrors ``test_app_multiuser_concurrency`` for the role layer: viewer logins resolve **distinct**
identities at the right role, admin endpoints are **403 for non-admins**, a role/active change to one
user never bleeds into another, and concurrent admin mutations neither lose updates nor deadlock. The
self-lockout guards (an admin can't demote / deactivate / delete themselves) are covered too.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_oauth import MockOAuthProvider

pytestmark = [pytest.mark.integration]


def _mock_app(tmp_path: Path, *, admin_emails: frozenset[str] = frozenset()):
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    app.state.oauth_provider = MockOAuthProvider()
    app.state.admin_emails = admin_emails
    return app


def _login(app, who: str, *, grant: str | None = None) -> TestClient:
    client = TestClient(app)
    params = {"as": who}
    if grant is not None:
        params["grant"] = grant
    client.get("/api/app/auth/login", params=params, follow_redirects=True)
    me = client.get("/api/app/me")
    assert me.status_code == 200, f"login as {who} failed: {me.text}"
    return client


def _uid(client: TestClient) -> str:
    return str(client.get("/api/app/me").json()["user_id"])


# --- role resolution at login -------------------------------------------------------------------


def test_plain_login_is_listener_grant_is_creator_allowlist_is_admin(tmp_path: Path) -> None:
    app = _mock_app(tmp_path, admin_emails=frozenset({"boss@e2e.local"}))
    assert _login(app, "plain").get("/api/app/me").json()["role"] == "listener"
    assert _login(app, "maker", grant="creator").get("/api/app/me").json()["role"] == "creator"
    assert _login(app, "boss").get("/api/app/me").json()["role"] == "admin"
    # 'admin' is never grantable via the request hint.
    assert _login(app, "sneaky", grant="admin").get("/api/app/me").json()["role"] == "listener"


def test_login_never_downgrades_an_existing_higher_role(tmp_path: Path) -> None:
    app = _mock_app(tmp_path, admin_emails=frozenset({"boss@e2e.local"}))
    assert _login(app, "boss").get("/api/app/me").json()["role"] == "admin"
    # A later plain (no-grant) login of the same identity keeps admin — not knocked down to listener.
    app.state.admin_emails = frozenset()  # even if dropped from the allowlist, no downgrade
    assert _login(app, "boss").get("/api/app/me").json()["role"] == "admin"


# --- admin endpoint authz -----------------------------------------------------------------------


def test_admin_endpoints_are_403_for_non_admins(tmp_path: Path) -> None:
    app = _mock_app(tmp_path)
    listener = _login(app, "lis")
    creator = _login(app, "cre", grant="creator")
    for client in (listener, creator):
        assert client.get("/api/app/admin/users").status_code == 403
        assert client.post("/api/app/admin/users", json={"email": "x@y.z"}).status_code == 403
    # an anonymous client is 401 (not authenticated), not 403
    assert TestClient(app).get("/api/app/admin/users").status_code == 401


def test_admin_lists_all_users_spanning_roles(tmp_path: Path) -> None:
    app = _mock_app(tmp_path, admin_emails=frozenset({"boss@e2e.local"}))
    admin = _login(app, "boss")
    _login(app, "lis")  # listener
    _login(app, "cre", grant="creator")  # creator
    rows = admin.get("/api/app/admin/users").json()
    by_email = {r["email"]: r["role"] for r in rows}
    assert by_email["boss@e2e.local"] == "admin"
    assert by_email["lis@e2e.local"] == "listener"
    assert by_email["cre@e2e.local"] == "creator"


# --- admin CRUD ---------------------------------------------------------------------------------


def test_admin_create_patch_delete_user(tmp_path: Path) -> None:
    app = _mock_app(tmp_path, admin_emails=frozenset({"boss@e2e.local"}))
    admin = _login(app, "boss")

    created = admin.post("/api/app/admin/users", json={"email": "new@x.io", "role": "creator"})
    assert created.status_code == 201
    uid = created.json()["user_id"]
    assert created.json()["role"] == "creator" and created.json()["name"] == "new@x.io"

    # duplicate email → 409; unknown role → 422
    assert admin.post("/api/app/admin/users", json={"email": "new@x.io"}).status_code == 409
    assert (
        admin.post("/api/app/admin/users", json={"email": "z@x.io", "role": "wizard"}).status_code
        == 422
    )

    # patch role + deactivate
    patched = admin.patch(f"/api/app/admin/users/{uid}", json={"role": "admin", "disabled": True})
    assert patched.status_code == 200
    assert patched.json()["role"] == "admin" and patched.json()["disabled"] is True
    assert admin.patch(f"/api/app/admin/users/{uid}", json={"role": "wizard"}).status_code == 422
    assert admin.patch("/api/app/admin/users/u_missing", json={"role": "admin"}).status_code == 404

    # delete
    assert admin.delete(f"/api/app/admin/users/{uid}").status_code == 204
    assert admin.delete(f"/api/app/admin/users/{uid}").status_code == 404


def test_self_lockout_guards(tmp_path: Path) -> None:
    app = _mock_app(tmp_path, admin_emails=frozenset({"boss@e2e.local"}))
    admin = _login(app, "boss")
    me = _uid(admin)
    assert admin.patch(f"/api/app/admin/users/{me}", json={"role": "creator"}).status_code == 400
    assert admin.patch(f"/api/app/admin/users/{me}", json={"disabled": True}).status_code == 400
    assert admin.delete(f"/api/app/admin/users/{me}").status_code == 400
    # still admin + active after the rejected self-mutations
    assert admin.get("/api/app/me").json()["role"] == "admin"


# --- isolation + concurrency --------------------------------------------------------------------


def test_role_change_to_one_user_does_not_bleed_into_another(tmp_path: Path) -> None:
    app = _mock_app(tmp_path, admin_emails=frozenset({"boss@e2e.local"}))
    admin = _login(app, "boss")
    a = _login(app, "alice", grant="creator")
    b = _login(app, "bob", grant="creator")
    admin.patch(f"/api/app/admin/users/{_uid(a)}", json={"role": "admin"})
    # bob is untouched; alice now reads back admin, bob still creator
    assert a.get("/api/app/me").json()["role"] == "admin"
    assert b.get("/api/app/me").json()["role"] == "creator"


def test_concurrent_admin_mutations_no_deadlock_no_lost_updates(tmp_path: Path) -> None:
    app = _mock_app(tmp_path, admin_emails=frozenset({"boss@e2e.local"}))
    admin = _login(app, "boss")
    n = 16
    uids = [_uid(_login(app, f"u{i}", grant="creator")) for i in range(n)]

    def flip(i: int) -> int:
        uid = uids[i]
        # each distinct user: promote to admin + deactivate, concurrently
        r = admin.patch(f"/api/app/admin/users/{uid}", json={"role": "admin", "disabled": True})
        return int(r.status_code)

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as pool:
        codes = list(pool.map(flip, range(n)))
    assert time.monotonic() - start < 30  # no deadlock
    assert all(c == 200 for c in codes)

    rows = {r["user_id"]: r for r in admin.get("/api/app/admin/users").json()}
    for uid in uids:
        assert rows[uid]["role"] == "admin" and rows[uid]["disabled"] is True


def test_same_user_concurrent_role_flips_converge(tmp_path: Path) -> None:
    # Many concurrent PATCHes to the SAME user (the per-profile lock orders read-modify-write so the
    # final state is consistent and the profile is never corrupted).
    app = _mock_app(tmp_path, admin_emails=frozenset({"boss@e2e.local"}))
    admin = _login(app, "boss")
    target = _uid(_login(app, "target", grant="creator"))
    m = 25

    def flip(i: int) -> int:
        role = "admin" if i % 2 else "creator"
        return int(admin.patch(f"/api/app/admin/users/{target}", json={"role": role}).status_code)

    with ThreadPoolExecutor(max_workers=m) as pool:
        codes = list(pool.map(flip, range(m)))
    assert all(c == 200 for c in codes)
    # final role is one of the two written values, profile intact + readable
    final = admin.get("/api/app/admin/users").json()
    role = {r["user_id"]: r["role"] for r in final}[target]
    assert role in ("admin", "creator")

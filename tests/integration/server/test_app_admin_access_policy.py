"""Admin-managed sign-in access policy (#2190).

Admitting a beta tester used to mean editing a GitHub repo variable and running a production
deploy, because the allowlist was read from env once at startup. These endpoints move it to a file
the admin surface writes and every sign-in reads.

The two failure modes worth a test are the ones that hurt: an admin replacing the policy and
locking themselves out, and a non-admin being able to grant themselves access.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_oauth import MockOAuthProvider

pytestmark = [pytest.mark.integration]

ADMIN = "boss@e2e.local"
POLICY = "/api/app/admin/access-policy"


def _app(tmp_path: Path, *, env_policy: AccessPolicy | None = None):
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    # `open` so the mock logins below can get in; the policy under test is the PERSISTED one.
    app.state.access_policy = env_policy or AccessPolicy("open", frozenset(), frozenset())
    app.state.oauth_provider = MockOAuthProvider()
    app.state.admin_emails = frozenset({ADMIN})
    return app


def _login(app, who: str) -> TestClient:
    client = TestClient(app)
    client.get("/api/app/auth/login", params={"as": who}, follow_redirects=True)
    assert client.get("/api/app/me").status_code == 200
    return client


def _admin(app) -> TestClient:
    client = _login(app, "boss")
    assert client.get("/api/app/me").json()["role"] == "admin"
    return client


# --- authorisation ------------------------------------------------------------------------------


def test_non_admin_cannot_read_or_change_the_policy(tmp_path: Path) -> None:
    """Otherwise any signed-in listener could add their friends."""
    app = _app(tmp_path)
    listener = _login(app, "plain")
    assert listener.get(POLICY).status_code == 403
    assert listener.put(POLICY, json={"mode": "open"}).status_code == 403


def test_anonymous_cannot_change_the_policy(tmp_path: Path) -> None:
    app = _app(tmp_path)
    anon = TestClient(app)
    assert anon.put(POLICY, json={"mode": "open"}).status_code in (401, 403)


# --- reading ------------------------------------------------------------------------------------


def test_get_reports_the_env_policy_until_one_is_persisted(tmp_path: Path) -> None:
    env = AccessPolicy("allowlist", frozenset({ADMIN, "seed@e2e.local"}), frozenset())
    app = _app(tmp_path, env_policy=env)
    body = _admin(app).get(POLICY).json()
    assert body["persisted"] is False
    assert body["allowed_emails"] == [ADMIN, "seed@e2e.local"]


def test_get_reports_the_persisted_policy_once_written(tmp_path: Path) -> None:
    app = _app(tmp_path)
    admin = _admin(app)
    admin.put(POLICY, json={"mode": "allowlist", "allowed_emails": [ADMIN, "tester@e2e.local"]})
    body = admin.get(POLICY).json()
    assert body["persisted"] is True
    assert body["allowed_emails"] == [ADMIN, "tester@e2e.local"]


# --- the self-lockout guard ---------------------------------------------------------------------


def test_admin_cannot_write_a_policy_that_excludes_themselves(tmp_path: Path) -> None:
    """The obvious mistake: paste in the testers and forget your own address."""
    app = _app(tmp_path)
    admin = _admin(app)
    resp = admin.put(POLICY, json={"mode": "allowlist", "allowed_emails": ["tester@e2e.local"]})
    assert resp.status_code == 400
    assert ADMIN in resp.json()["detail"]
    # and nothing was written
    assert admin.get(POLICY).json()["persisted"] is False


def test_an_empty_allowlist_is_refused_rather_than_denying_everyone(tmp_path: Path) -> None:
    app = _app(tmp_path)
    admin = _admin(app)
    assert admin.put(POLICY, json={"mode": "allowlist"}).status_code == 400


def test_a_covering_domain_satisfies_the_guard(tmp_path: Path) -> None:
    """The guard asks 'are you allowed', not 'are you listed' — a domain rule counts."""
    app = _app(tmp_path)
    admin = _admin(app)
    resp = admin.put(POLICY, json={"mode": "allowlist", "allowed_domains": ["e2e.local"]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["persisted"] is True


def test_open_mode_satisfies_the_guard(tmp_path: Path) -> None:
    app = _app(tmp_path)
    assert _admin(app).put(POLICY, json={"mode": "open"}).status_code == 200


# --- it actually gates sign-in, without a restart -----------------------------------------------


def test_a_written_policy_admits_a_new_user_with_no_restart(tmp_path: Path) -> None:
    """The point of the whole feature: grant access to a person the env policy never mentioned."""
    env = AccessPolicy("allowlist", frozenset({ADMIN}), frozenset())
    app = _app(tmp_path, env_policy=env)

    # Before: the env policy does not know this address.
    assert (
        TestClient(app)
        .get("/api/app/auth/login", params={"as": "newbie"}, follow_redirects=True)
        .status_code
        == 403
    )

    _admin(app).put(  # same process, no restart, no redeploy
        POLICY, json={"mode": "allowlist", "allowed_emails": [ADMIN, "newbie@e2e.local"]}
    )

    after = TestClient(app)
    after.get("/api/app/auth/login", params={"as": "newbie"}, follow_redirects=True)
    assert after.get("/api/app/me").status_code == 200


def test_a_written_policy_revokes_an_address_the_env_allowed(tmp_path: Path) -> None:
    """Replacement, not merge — otherwise access could be granted but never taken back."""
    env = AccessPolicy("allowlist", frozenset({ADMIN, "goner@e2e.local"}), frozenset())
    app = _app(tmp_path, env_policy=env)
    _admin(app).put(POLICY, json={"mode": "allowlist", "allowed_emails": [ADMIN]})

    assert (
        TestClient(app)
        .get("/api/app/auth/login", params={"as": "goner"}, follow_redirects=True)
        .status_code
        == 403
    )


# --- RFC-108: the operator-public surface must not be openable at runtime ------------------------


def test_put_open_is_refused_on_the_operator_public_surface(tmp_path: Path) -> None:
    """`app.py` refuses to BOOT operator-public under open signup, because any signed-in account
    self-grants `creator` over the operator-read corpus. An endpoint that could set the same mode
    at runtime would walk straight past that guard."""
    app = _app(tmp_path)
    app.state.operator_public = True
    admin = _admin(app)

    resp = admin.put(POLICY, json={"mode": "open"})
    assert resp.status_code == 400, resp.text
    assert "operator-read corpus" in resp.json()["detail"]
    assert admin.get(POLICY).json()["persisted"] is False  # nothing written


def test_put_allowlist_is_still_fine_on_the_operator_public_surface(tmp_path: Path) -> None:
    """The refusal is about `open` only — managing the allowlist is the point of the endpoint."""
    app = _app(tmp_path)
    app.state.operator_public = True
    resp = _admin(app).put(POLICY, json={"mode": "allowlist", "allowed_emails": [ADMIN]})
    assert resp.status_code == 200, resp.text


def test_a_persisted_open_policy_blocks_an_operator_public_BOOT(tmp_path: Path) -> None:
    """The other half: the file must not survive a restart unnoticed. The boot guard used to read
    only env, so a persisted `open` would have booted clean and served the corpus wide."""
    from podcast_scraper.server.app import _guard_operator_public_open_signup
    from podcast_scraper.server.app_access_store import save_policy

    data_dir = tmp_path / "appdata"
    save_policy(data_dir, AccessPolicy("open", frozenset(), frozenset()))

    with pytest.raises(RuntimeError, match="operator-read corpus"):
        _guard_operator_public_open_signup(True, data_dir)

    # ... and a persisted ALLOWLIST policy boots fine even when env says open.
    save_policy(data_dir, AccessPolicy("allowlist", frozenset({ADMIN}), frozenset()))
    _guard_operator_public_open_signup(True, data_dir)


# --- input validation ---------------------------------------------------------------------------


def test_a_misspelled_mode_is_rejected_rather_than_silently_coerced(tmp_path: Path) -> None:
    """It used to return 200 with mode `allowlist` — fail-closed, but the caller was told nothing
    and would reasonably believe they had opened signup."""
    app = _app(tmp_path)
    assert _admin(app).put(POLICY, json={"mode": "opeen"}).status_code == 422


# --- the audit record has to be able to answer "who did we cut off" -----------------------------


def test_the_audit_record_names_the_addresses(tmp_path: Path) -> None:
    import json as _json

    app = _app(tmp_path)
    audit = tmp_path / "audit.jsonl"
    app.state.audit_path = audit
    admin = _admin(app)

    admin.put(POLICY, json={"mode": "allowlist", "allowed_emails": [ADMIN, "keep@e2e.local"]})
    admin.put(POLICY, json={"mode": "allowlist", "allowed_emails": [ADMIN, "new@e2e.local"]})

    records = [
        _json.loads(line)
        for line in audit.read_text(encoding="utf-8").splitlines()
        if "access_policy" in line
    ]
    assert records, "no access-policy audit records were written"
    last = records[-1]
    assert last["removed_emails"] == ["keep@e2e.local"]
    assert last["added_emails"] == ["new@e2e.local"]
    assert last["was_mode"] == "allowlist"

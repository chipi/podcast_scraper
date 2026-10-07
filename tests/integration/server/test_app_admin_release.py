"""The released app version, changed at runtime (``/api/app/admin/release``).

What a native-only release relies on: an admin sets the version and the very next health read —
on ``/api/health`` and on the public ``/api/app/auth/status`` the app actually reaches — reports
it, with no restart; clearing it falls back to the deploy's ``APP_PLAYER_VERSION``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_oauth import MockOAuthProvider

pytestmark = [pytest.mark.integration]

ADMIN = "boss@e2e.local"
RELEASE = "/api/app/admin/release"


def _app(tmp_path: Path, *, deploy_default: str | None = "1.0.1"):
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    app.state.oauth_provider = MockOAuthProvider()
    app.state.admin_emails = frozenset({ADMIN})
    app.state.audit_path = tmp_path / "audit.jsonl"
    app.state.player_version = deploy_default
    return app


def _login(app, who: str) -> TestClient:
    client = TestClient(app)
    client.get("/api/app/auth/login", params={"as": who}, follow_redirects=True)
    assert client.get("/api/app/me").status_code == 200
    return client


def _served(client: TestClient) -> tuple[str | None, str | None]:
    return (
        client.get("/api/app/auth/status").json()["player_version"],
        client.get("/api/health").json()["player_version"],
    )


def test_an_admin_sets_the_version_and_it_is_served_at_once(tmp_path: Path) -> None:
    app = _app(tmp_path)
    admin = _login(app, "boss")
    assert _served(admin) == ("1.0.1", "1.0.1")  # the deploy default

    resp = admin.put(RELEASE, json={"player_version": "1.0.2"})
    assert resp.status_code == 200
    assert resp.json() == {
        "player_version": "1.0.2",
        "override": "1.0.2",
        "deploy_default": "1.0.1",
    }
    # Same app instance, no restart: both routes report the override on the next read.
    assert _served(TestClient(app)) == ("1.0.2", "1.0.2")

    # Clearing it hands back to the deploy default.
    assert admin.put(RELEASE, json={"player_version": None}).json()["player_version"] == "1.0.1"
    assert _served(TestClient(app)) == ("1.0.1", "1.0.1")


def test_no_deploy_default_and_no_override_means_no_prompt(tmp_path: Path) -> None:
    app = _app(tmp_path, deploy_default=None)
    assert _served(TestClient(app)) == (None, None)


def test_a_malformed_version_is_refused_and_nothing_changes(tmp_path: Path) -> None:
    app = _app(tmp_path)
    admin = _login(app, "boss")
    for bad in ("v1.0.2", "1.0.2-beta", "latest", "1..2"):
        assert admin.put(RELEASE, json={"player_version": bad}).status_code == 422, bad
    assert admin.get(RELEASE).json()["override"] is None


def test_only_an_admin_can_read_or_change_it(tmp_path: Path) -> None:
    app = _app(tmp_path)
    listener = _login(app, "plain")
    assert listener.get(RELEASE).status_code == 403
    assert listener.put(RELEASE, json={"player_version": "9.9.9"}).status_code == 403
    assert TestClient(app).put(RELEASE, json={"player_version": "9.9.9"}).status_code == 401
    assert _served(TestClient(app))[0] == "1.0.1"


def test_a_change_is_audited(tmp_path: Path) -> None:
    app = _app(tmp_path)
    _login(app, "boss").put(RELEASE, json={"player_version": "1.0.2"})
    records = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert any(
        r.get("action") == "player_release_set" and r.get("after") == "1.0.2" for r in records
    )

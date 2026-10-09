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
        "app": "player",
        "version": "1.0.2",
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


def test_the_operator_viewer_sets_the_version_the_player_serves(tmp_path: Path) -> None:
    """#2296: the Admin field is in the OPERATOR viewer, phones read the PLAYER api.

    The two apis have separate data dirs, so a write landed in the operator's copy and phones never
    saw it. Both now point ``app_release_dir`` at one shared dir.
    """
    shared = tmp_path / "release"
    operator = _app(tmp_path / "op")
    operator.state.app_data_dir = tmp_path / "operator-appdata"
    operator.state.app_release_dir = shared
    player = _app(tmp_path / "pl")
    player.state.app_data_dir = tmp_path / "player-appdata"
    player.state.app_release_dir = shared

    assert (
        _login(operator, "boss").put(RELEASE, json={"player_version": "1.0.2"}).status_code == 200
    )
    assert _served(TestClient(player)) == ("1.0.2", "1.0.2")
    # Neither stack's user data holds it — only the shared dir does.
    assert not (tmp_path / "operator-appdata" / "player_release.json").exists()
    assert not (tmp_path / "player-appdata" / "player_release.json").exists()


def test_without_a_release_dir_the_data_dir_holds_it(tmp_path: Path) -> None:
    app = _app(tmp_path)
    assert app.state.app_release_dir is None
    _login(app, "boss").put(RELEASE, json={"player_version": "1.0.2"})
    assert (tmp_path / "appdata" / "player_release.json").is_file()


def test_each_app_has_its_own_version(tmp_path: Path) -> None:
    """ADR-162: the kernel serves more than one client app, so a release is per app."""
    app = _app(tmp_path)
    admin = _login(app, "boss")
    resp = admin.put(RELEASE, json={"app": "news", "version": "0.3.0"})
    assert resp.status_code == 200
    assert resp.json()["app"] == "news" and resp.json()["version"] == "0.3.0"
    # The player is untouched, and health lists both apps.
    assert _served(TestClient(app)) == ("1.0.1", "1.0.1")
    assert TestClient(app).get("/api/health").json()["app_versions"] == {
        "player": "1.0.1",
        "news": "0.3.0",
    }
    assert admin.get(RELEASE, params={"app": "news"}).json()["override"] == "0.3.0"
    # Clearing news leaves the player's entry alone.
    admin.put(RELEASE, json={"app": "news", "version": None})
    assert TestClient(app).get("/api/health").json()["app_versions"] == {"player": "1.0.1"}


def test_the_player_override_reads_and_writes_the_old_file_too(tmp_path: Path) -> None:
    """An instance upgraded from before per-app versions keeps its override, and a rollback to
    code that knows only ``player_release.json`` still sees what was set after the upgrade."""
    app = _app(tmp_path)
    data = tmp_path / "appdata"
    data.mkdir(parents=True)
    (data / "player_release.json").write_text(json.dumps({"player_version": "1.0.5"}))
    assert _served(TestClient(app)) == ("1.0.5", "1.0.5")

    admin = _login(app, "boss")
    admin.put(RELEASE, json={"version": "1.0.6"})
    assert json.loads((data / "player_release.json").read_text()) == {"player_version": "1.0.6"}
    admin.put(RELEASE, json={"version": None})
    assert not (data / "player_release.json").exists()
    assert _served(TestClient(app)) == ("1.0.1", "1.0.1")


def test_a_body_naming_no_version_is_refused_not_a_silent_clear(tmp_path: Path) -> None:
    app = _app(tmp_path)
    admin = _login(app, "boss")
    admin.put(RELEASE, json={"version": "1.0.2"})
    assert admin.put(RELEASE, json={}).status_code == 422
    assert admin.put(RELEASE, json={"app": "player"}).status_code == 422
    assert admin.get(RELEASE).json()["override"] == "1.0.2"


def test_a_malformed_app_id_is_refused(tmp_path: Path) -> None:
    admin = _login(_app(tmp_path), "boss")
    for bad in ("News", "../x", "", "a" * 40):
        assert admin.put(RELEASE, json={"app": bad, "version": "1.0.0"}).status_code == 422, bad
        assert admin.get(RELEASE, params={"app": bad}).status_code == 422, bad


def test_app_release_dir_comes_from_the_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APP_RELEASE_DIR", str(tmp_path / "shared"))
    assert create_app(tmp_path, static_dir=False).state.app_release_dir == tmp_path / "shared"
    monkeypatch.delenv("APP_RELEASE_DIR")
    assert create_app(tmp_path, static_dir=False).state.app_release_dir is None

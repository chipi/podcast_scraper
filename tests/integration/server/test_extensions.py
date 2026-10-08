"""Extensions add routes and take part in account deletion (ADR-158 decision 4), with a fake app.

The platform names nothing private: a package publishes an ``Extension`` and the server mounts its
routers under the same serve postures as the core, and account deletion runs its hooks. These tests
use a fake extension, so they hold whatever private packages are installed — and they check the
platform still works with none at all.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from podcast_scraper import extensions
from podcast_scraper.extensions import Extension, RouterMount, use_extensions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_account_deletion import delete_account
from podcast_scraper.server.app_oauth import MockOAuthProvider
from podcast_scraper.server.app_user_store import get_user

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


def _router(path: str) -> APIRouter:
    router = APIRouter()

    @router.get(path)
    def _ok() -> dict[str, str]:
        return {"ok": path}

    return router


def _fake(deleted: list[tuple[str, bool]] | None = None) -> Extension:
    def _on_delete(data_dir: Path, user) -> dict[str, int]:
        if deleted is not None:
            # Records the directory still exists: hooks run before the user dir goes.
            deleted.append((user.user_id, (data_dir / "users" / user.user_id).is_dir()))
        return {"fake_records": 2}

    return Extension(
        name="fake",
        routers=lambda: (
            RouterMount(_router("/fake/app"), "app"),
            RouterMount(_router("/fake/op"), "operator"),
            RouterMount(_router("/fake/op-public"), "operator", operator_public=True),
            RouterMount(_router("/fake/internal"), "internal"),
            RouterMount(_router("/.well-known/fake"), "root"),
        ),
        account_deleted=(_on_delete,),
    )


def _paths(app) -> set[str]:
    # fastapi >=0.137 nests include_router calls; walk the effective contexts too (see
    # test_operator_public_mode._api_paths).
    paths: set[str] = set()
    for route in app.routes:
        if getattr(route, "path", ""):
            paths.add(route.path)
        contexts = getattr(route, "effective_route_contexts", None)
        if callable(contexts):
            for ctx in contexts():
                child_path = getattr(getattr(ctx, "route", ctx), "path", None)
                if child_path:
                    paths.add(child_path)
    return paths


FAKE_PATHS = {
    "/api/app/fake/app",
    "/api/fake/op",
    "/api/fake/op-public",
    "/internal/fake/internal",
    "/.well-known/fake",
}


@pytest.fixture
def posture(monkeypatch):
    def _set(name: str) -> None:
        monkeypatch.delenv("PODCAST_SERVE_APP_ONLY", raising=False)
        monkeypatch.delenv("PODCAST_SERVE_OPERATOR_PUBLIC", raising=False)
        if name == "app_only":
            monkeypatch.setenv("PODCAST_SERVE_APP_ONLY", "1")
        elif name == "operator_public":
            monkeypatch.setenv("PODCAST_SERVE_OPERATOR_PUBLIC", "1")

    return _set


def test_the_operator_serve_mounts_every_plane(tmp_path: Path, posture) -> None:
    posture("full")
    with use_extensions([_fake()]):
        assert FAKE_PATHS <= _paths(create_app(tmp_path, static_dir=False))


def test_the_player_never_gets_operator_routes(tmp_path: Path, posture) -> None:
    posture("app_only")
    with use_extensions([_fake()]):
        paths = _paths(create_app(tmp_path, static_dir=False))
    assert {"/api/app/fake/app", "/internal/fake/internal", "/.well-known/fake"} <= paths
    assert not {"/api/fake/op", "/api/fake/op-public"} & paths


def test_the_public_operator_surface_gets_only_marked_routes_and_gates_them(
    tmp_path: Path, posture
) -> None:
    posture("operator_public")
    with use_extensions([_fake()]):
        app = create_app(tmp_path, static_dir=False)
    paths = _paths(app)
    assert "/api/fake/op-public" in paths
    assert "/api/fake/op" not in paths
    # Router-level >=creator gate, as for the core curated routes: no session, no data.
    assert TestClient(app).get("/api/fake/op-public").status_code in (401, 403)


def test_mounted_routes_answer(tmp_path: Path, posture) -> None:
    posture("full")
    with use_extensions([_fake()]):
        client = TestClient(create_app(tmp_path, static_dir=False))
    assert client.get("/api/app/fake/app").json() == {"ok": "/fake/app"}
    assert client.get("/.well-known/fake").json() == {"ok": "/.well-known/fake"}


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


def test_account_deletion_runs_every_hook_before_the_directory_goes(tmp_path: Path) -> None:
    deleted: list[tuple[str, bool]] = []
    with use_extensions([_fake(deleted)]):
        app = _app(tmp_path)
        client = _login(app, "ada")
        uid = client.get("/api/app/me").json()["user_id"]
        resp = client.request("DELETE", "/api/app/me", json={"confirm": "DELETE"})
    assert resp.status_code == 204
    assert deleted == [(uid, True)]
    assert not (tmp_path / "appdata" / "users" / uid).exists()


def test_the_deletion_report_carries_the_hook_counts(tmp_path: Path) -> None:
    """The counts reach the self-service response and the admin route's log event."""
    with use_extensions([_fake()]):
        app = _app(tmp_path)
        uid = _login(app, "ada").get("/api/app/me").json()["user_id"]
        user = get_user(tmp_path / "appdata", uid)
        assert user is not None
        report = delete_account(tmp_path / "appdata", user)
    assert report.extensions == {"fake_records": 2}
    assert report.user_dir is True


def test_the_platform_runs_with_no_extensions(tmp_path: Path, posture) -> None:
    """Absent package, absent feature: no MCP routes, and deletion still removes the account."""
    posture("full")
    with use_extensions([]):
        app = _app(tmp_path)
        assert not any("mcp" in p for p in _paths(app))
        client = _login(app, "ada")
        uid = client.get("/api/app/me").json()["user_id"]
        assert (
            client.request("DELETE", "/api/app/me", json={"confirm": "DELETE"}).status_code == 204
        )
    assert not (tmp_path / "appdata" / "users" / uid).exists()


def test_an_in_tree_module_the_split_removed_is_skipped(monkeypatch) -> None:
    monkeypatch.setattr(extensions, "_IN_TREE", ("podcast_scraper.server.no_such_extension",))
    monkeypatch.setattr(extensions.metadata, "entry_points", lambda group: [])
    assert extensions._discover() == []


def test_a_broken_import_inside_an_extension_is_not_hidden(tmp_path: Path, monkeypatch) -> None:
    """Only the extension's OWN absence means "not installed"; a missing dependency is a bug."""
    (tmp_path / "broken_ext_for_test.py").write_text("import no_such_dependency_for_test\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(extensions, "_IN_TREE", ("broken_ext_for_test",))
    monkeypatch.setattr(extensions.metadata, "entry_points", lambda group: [])
    with pytest.raises(ModuleNotFoundError, match="no_such_dependency_for_test"):
        extensions._discover()


def test_an_extension_found_twice_loads_once_and_the_in_tree_copy_wins(monkeypatch) -> None:
    """Before the cutover a dev checkout has both the in-tree module and the private package's
    entry point; mounting both would register every route twice."""
    in_tree = Extension(name="fake", routers=lambda: (RouterMount(_router("/in-tree"), "app"),))
    installed = Extension(name="fake", routers=lambda: (RouterMount(_router("/installed"), "app"),))

    class _EP:
        name = "fake"
        value = "somewhere:EXTENSION"

        def load(self):
            return installed

    module = types.ModuleType("podcast_scraper._fake_ext")
    module.EXTENSION = in_tree  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "podcast_scraper._fake_ext", module)
    monkeypatch.setattr(extensions, "_IN_TREE", ("podcast_scraper._fake_ext",))
    monkeypatch.setattr(extensions.metadata, "entry_points", lambda group: [_EP()])
    assert extensions._discover() == [in_tree]


def test_discovery_runs_once_per_process(monkeypatch) -> None:
    """Read paths ask for extensions per request; scanning the installed distributions is slow."""
    calls: list[str] = []
    monkeypatch.setattr(extensions, "_discovered", None)
    monkeypatch.setattr(extensions, "_IN_TREE", ())

    def _entry_points(group: str) -> list:
        calls.append(group)
        return []

    monkeypatch.setattr(extensions.metadata, "entry_points", _entry_points)
    assert extensions.load_extensions() == []
    assert extensions.load_extensions() == []
    assert calls == [extensions.ENTRY_POINT_GROUP]
    with use_extensions([Extension(name="fake")]):
        assert [e.name for e in extensions.load_extensions()] == ["fake"]
    assert calls == [extensions.ENTRY_POINT_GROUP]


def test_account_created_runs_once_at_sign_up_not_at_later_sign_ins(tmp_path: Path) -> None:
    created: list[tuple[str, str]] = []
    ext = Extension(
        name="fake",
        account_created=(lambda data_dir, user, provider: created.append((user.email, provider)),),
    )
    with use_extensions([ext]):
        app = _app(tmp_path)
        _login(app, "ada")
        _login(app, "ada")
    assert created == [("ada@e2e.local", "mock")]


def test_the_player_extension_records_the_sign_up_in_the_users_event_log(tmp_path: Path) -> None:
    """The in-tree player extension keeps today's behaviour: one account_created event per user."""
    player = extensions._from_module("podcast_scraper.server.app_player_extension")
    assert player is not None
    with use_extensions([player]):
        app = _app(tmp_path)
        uid = _login(app, "ada").get("/api/app/me").json()["user_id"]
    events = list((tmp_path / "appdata" / "users" / uid).glob("*.jsonl"))
    assert any("account_created" in p.read_text() for p in events), [p.name for p in events]


def test_loading_every_in_tree_extension_imports_no_web_stack() -> None:
    """The pipeline image has no FastAPI, and enrichment loads extensions too."""
    import subprocess

    code = (
        "import sys\n"
        "from podcast_scraper import extensions\n"
        "exts = extensions.load_extensions()\n"
        "assert {e.name for e in exts} >= {'mcp', 'player'}, [e.name for e in exts]\n"
        "web = sorted(m for m in sys.modules if m.split('.')[0] in ('fastapi', 'starlette'))\n"
        "assert not web, web\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr

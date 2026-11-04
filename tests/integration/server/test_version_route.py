"""GET /api/version: public fields for anyone, admin fields for an admin session."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.db import migrate
from podcast_scraper.extensions import Extension, use_extensions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_oauth import MockOAuthProvider

pytestmark = [pytest.mark.integration]

ADMIN = "boss@e2e.local"
ADMIN_ONLY = {"modules"}
ADMIN_ONLY_DB = {"plane", "current_revision", "head_revision", "server_version"}


@pytest.fixture(autouse=True)
def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "VERSION.json"
    f.write_text(
        json.dumps(
            {
                "component": "api",
                "version": "9.9.9",
                "git_sha": "a" * 40,
                "built_at": "2026-11-05T01:02:03Z",
            }
        )
    )
    monkeypatch.setenv("PODCAST_VERSION_FILE", str(f))
    monkeypatch.setenv("CL_RELEASE", "b5a243d")
    monkeypatch.setenv("PODCAST_RELEASE", "sha-f493b65")


def _app(tmp_path: Path):
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    app.state.oauth_provider = MockOAuthProvider()
    app.state.admin_emails = frozenset({ADMIN})
    app.state.audit_path = tmp_path / "audit.jsonl"
    return app


def _login(app, who: str) -> TestClient:
    client = TestClient(app)
    client.get("/api/app/auth/login", params={"as": who}, follow_redirects=True)
    assert client.get("/api/app/me").status_code == 200
    return client


def _module() -> Extension:
    return Extension(
        name="private-thing",
        version_info=lambda: {"version": "1.2.3", "git_sha": "b" * 40, "public_ref": "c" * 40},
    )


def test_a_signed_out_caller_gets_the_public_fields_only(tmp_path: Path) -> None:
    with use_extensions([_module()]):
        app = _app(tmp_path)
        body = TestClient(app).get("/api/version").json()
    assert body["release"] == "b5a243d"
    assert body["image_release"] == "sha-f493b65"
    assert body["api"] == {
        "version": "9.9.9",
        "git_sha": "a" * 40,
        "built_at": "2026-11-05T01:02:03Z",
    }
    assert body["database"]["configured"] is True
    assert body["database"]["schema_at_head"] is True
    assert body["corpus"] == {"code_version": None, "produced_by": None}
    assert not (ADMIN_ONLY & body.keys())
    assert not (ADMIN_ONLY_DB & body["database"].keys())


def test_an_admin_session_also_gets_modules_and_the_database_detail(tmp_path: Path) -> None:
    with use_extensions([_module()]):
        app = _app(tmp_path)
        body = _login(app, "boss").get("/api/version").json()
    assert body["modules"] == [
        {"name": "private-thing", "version": "1.2.3", "git_sha": "b" * 40, "public_ref": "c" * 40}
    ]
    db = body["database"]
    assert db["plane"] == "player"
    assert db["current_revision"] == db["head_revision"] == migrate.head_revision()
    assert db["server_version"]


def test_a_signed_in_non_admin_gets_the_public_fields(tmp_path: Path) -> None:
    with use_extensions([_module()]):
        app = _app(tmp_path)
        body = _login(app, "listener").get("/api/version").json()
    assert "modules" not in body
    assert not (ADMIN_ONLY_DB & body["database"].keys())


def test_the_player_edge_path_serves_the_same_body(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client = TestClient(app)
    assert client.get("/api/app/version").json() == client.get("/api/version").json()


def test_no_database_url_reports_configured_false_and_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PODCAST_DATABASE_URL", raising=False)
    from podcast_scraper.db import reset_engines

    reset_engines()
    app = _app(tmp_path)
    resp = TestClient(app).get("/api/version")
    assert resp.status_code == 200
    assert resp.json()["database"] == {"configured": False}


def test_a_module_whose_hook_raises_is_listed_with_an_error(tmp_path: Path) -> None:
    def boom() -> dict:
        raise RuntimeError("nope")

    with use_extensions([Extension(name="bad", version_info=boom), Extension(name="plain")]):
        app = _app(tmp_path)
        body = _login(app, "boss").get("/api/version").json()
    assert body["modules"] == [{"name": "bad", "error": "RuntimeError"}, {"name": "plain"}]


def test_the_snapshot_is_cached_for_a_few_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def counting() -> dict:
        calls.append(1)
        return {"version": "1"}

    with use_extensions([Extension(name="m", version_info=counting)]):
        app = _app(tmp_path)
        client = TestClient(app)
        client.get("/api/version")
        client.get("/api/version")
        assert len(calls) == 1
        from podcast_scraper.server.routes import version as route

        monkeypatch.setattr(route, "CACHE_SECONDS", 0.0)
        client.get("/api/version")
        assert len(calls) == 2


def test_the_corpus_stamp_is_the_one_health_reports(tmp_path: Path) -> None:
    (tmp_path / "corpus_manifest.json").write_text(
        json.dumps(
            {
                "produced_by": {
                    "code_version": "2.6.0",
                    "git_sha": "abc1234",
                    "produced_at": "2026-11-01T00:00:00Z",
                }
            }
        )
    )
    app = _app(tmp_path)
    client = TestClient(app)
    corpus = client.get("/api/version").json()["corpus"]
    health = client.get("/api/health").json()
    assert corpus["produced_by"] == health["corpus_produced_by"]
    assert corpus["code_version"] == health["corpus_code_version"]
    assert corpus["produced_by"]["code_version"] == "2.6.0"


def test_the_info_metrics_carry_the_build_and_the_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("prometheus_client")
    from prometheus_client import generate_latest

    from podcast_scraper.server.routes import version as route

    monkeypatch.setenv("PODCAST_METRICS_ENABLED", "1")
    with use_extensions([_module()]):
        app = _app(tmp_path)
        assert route.install_version_metrics(app) is True
        text = generate_latest().decode()
    assert (
        f'podcast_build_info{{component="api",git_sha="{"a" * 40}",release="b5a243d",version="9.9.9"}} 1.0'
        in text
    )
    assert 'component="private-thing"' in text
    assert (
        'podcast_db_schema_info{at_head="true",plane="player",revision="'
        + migrate.head_revision()
        + '"} 1.0'
        in text
    )

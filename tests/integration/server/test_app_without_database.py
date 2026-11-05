"""A server with no database URL keeps starting (ADR-169).

The control-plane api runs this same server with an app data dir, an operator key and no session
secret, and never touches the users store; it has no ``PODCAST_DATABASE_URL``. Starting must not
require one, and a store call must name the missing configuration instead of failing obscurely.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.db import engines, reset_engines
from podcast_scraper.db.engines import DatabaseNotReady
from podcast_scraper.db.settings import DatabaseNotConfigured
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_user_store import get_or_create_user, get_user, user_id_for
from tests._database import worker_database_url

pytestmark = [pytest.mark.integration]


def _control_plane_app(tmp_path: Path):
    app = create_app(tmp_path, static_dir=False, enable_feeds_api=True)
    app.state.operator_api_key = "secret"
    app.state.session_secret = ""
    app.state.app_data_dir = tmp_path / ".app"
    return app


def test_the_app_starts_and_an_operator_route_works_with_the_key_and_no_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reset_engines()
    monkeypatch.delenv("PODCAST_DATABASE_URL", raising=False)
    with TestClient(_control_plane_app(tmp_path)) as client:  # runs the lifespan
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/feeds").status_code == 403
        assert client.get("/api/feeds", headers={"X-Operator-Key": "secret"}).status_code != 403
    assert not engines.engines_open(), "no URL, so nothing was opened"


def test_a_store_call_with_no_url_names_the_missing_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reset_engines()
    monkeypatch.delenv("PODCAST_DATABASE_URL", raising=False)
    with pytest.raises(DatabaseNotConfigured, match="PODCAST_DATABASE_URL"):
        get_user(tmp_path, user_id_for("google", "s"))
    with pytest.raises(DatabaseNotConfigured, match="make test-db-up"):
        get_or_create_user(tmp_path, provider="google", subject="s", email="a@x.com", name="A")


def test_with_a_url_the_start_checks_the_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = worker_database_url()
    assert url, "set PODCAST_TEST_DATABASE_URL (make test-db-up)"
    reset_engines()
    monkeypatch.setenv("PODCAST_DATABASE_URL", url)
    with TestClient(_control_plane_app(tmp_path)) as client:
        assert client.get("/api/health").status_code == 200
    assert engines.engines_open()

    reset_engines()
    monkeypatch.setenv("PODCAST_DB_PLANE", "operator")
    from sqlalchemy import create_engine, text

    from podcast_scraper.db.settings import normalize_url

    admin = create_engine(normalize_url(url), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS operator CASCADE"))
    try:
        with pytest.raises(DatabaseNotReady, match="db upgrade --plane operator"):
            with TestClient(_control_plane_app(tmp_path)):
                pass
    finally:
        from podcast_scraper.db import migrate

        migrate.upgrade(url, "operator")
        admin.dispose()

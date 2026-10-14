"""#2345: the player reads which stored references are gone."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from podcast_scraper.server import app_sessions, app_user_state  # noqa: E402
from podcast_scraper.server.app import create_app  # noqa: E402
from podcast_scraper.server.app_access import AccessPolicy  # noqa: E402
from podcast_scraper.server.app_user_store import get_or_create_user  # noqa: E402
from podcast_scraper.server.reference_integrity import repair_user, Resolver  # noqa: E402

pytestmark = pytest.mark.integration


def _client(root: Path, data_dir: Path) -> tuple[TestClient, str]:
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    client = TestClient(app)
    user = get_or_create_user(data_dir, provider="stub", subject="s", email="u@x.com", name="U")
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client, user.user_id


def test_the_gone_references_are_listed_after_a_pass(tmp_path: Path) -> None:
    client, uid = _client(tmp_path, tmp_path / "app")
    assert client.get("/api/app/discover/references/gone").json() == {"items": []}

    app_user_state.add_interest(tmp_path / "app", uid, "thc:dissolved")
    repair_user(tmp_path / "app", uid, Resolver(storylines={"thc:other"}))
    body = client.get("/api/app/discover/references/gone").json()
    assert body == {"items": [{"ref": "thc:dissolved", "kind": "storyline"}]}
    # Flagged, never removed: the follow is still there for the listener to remove.
    assert app_user_state.get_interests(tmp_path / "app", uid) == ["thc:dissolved"]


def test_signed_out_is_refused(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path, static_dir=False))
    assert client.get("/api/app/discover/references/gone").status_code in (401, 403, 503)

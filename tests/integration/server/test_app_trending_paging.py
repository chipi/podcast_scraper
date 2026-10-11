"""Trends are paged on the server (operator 2026-10-11): "Show more" is the next page of ONE list,
ordered by the chosen sort on the server — never a reveal of rows the browser already holds."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from podcast_scraper.server import app_sessions  # noqa: E402
from podcast_scraper.server.app import create_app  # noqa: E402
from podcast_scraper.server.app_access import AccessPolicy  # noqa: E402
from podcast_scraper.server.app_user_store import get_or_create_user  # noqa: E402

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "app-validation-corpus" / "v3"


def _client(data_dir: Path) -> TestClient:
    app = create_app(ROOT, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    client = TestClient(app)
    user = get_or_create_user(data_dir, provider="stub", subject="s", email="u@x.com", name="U")
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


@pytest.mark.parametrize("sort", ["score", "rising", "trending"])
def test_pages_are_slices_of_one_list_in_the_servers_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sort: str
) -> None:
    monkeypatch.setenv("APP_TRENDING_NOW", "2026-07-20")
    monkeypatch.setenv("APP_MOMENTUM_MIN_TOTAL", "1")
    client = _client(tmp_path)

    def page(offset: int, limit: int) -> dict:
        body: dict = client.get(
            "/api/app/trending",
            params={"kind": "topic", "sort": sort, "offset": offset, "limit": limit},
        ).json()
        return body

    whole = page(0, 50)
    ids = [i["entity_id"] for i in whole["items"]]
    assert whole["total"] == len(ids) > 6
    stitched: list[str] = []
    for offset in range(0, whole["total"], 3):
        got = page(offset, 3)
        assert got["total"] == whole["total"]
        stitched += [i["entity_id"] for i in got["items"]]
    assert stitched == ids
    if sort == "rising":
        v = [i["velocity"] for i in whole["items"]]
        assert v == sorted(v, reverse=True)
    if sort == "trending":
        n = [i["volume"] for i in whole["items"]]
        assert n == sorted(n, reverse=True)
    assert page(whole["total"], 3)["items"] == []


def test_an_unknown_sort_is_refused(tmp_path: Path) -> None:
    r = _client(tmp_path).get("/api/app/trending", params={"kind": "topic", "sort": "loud"})
    assert r.status_code == 422

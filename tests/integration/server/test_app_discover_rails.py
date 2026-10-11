"""Discover's rails, paged on the server (ADR-163; operator 2026-10-11) — over the COMMITTED fixture.

One request gives every rail's first page in page order, one card per episode across the page;
"Show more" pages a rail from the server, leaving out what the rails above it showed. Cold start is
decided server-side and flagged.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from podcast_scraper.server import app_sessions, app_user_state  # noqa: E402
from podcast_scraper.server.app import create_app  # noqa: E402
from podcast_scraper.server.app_access import AccessPolicy  # noqa: E402
from podcast_scraper.server.app_discover_rails import FIRST, MINE_RAILS  # noqa: E402
from podcast_scraper.server.app_user_store import get_or_create_user  # noqa: E402

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "app-validation-corpus" / "v3"
HEARD = ("p07-a5eb82076a", "p07-2aceab172c", "p02-8fdd7b7f79", "p07-769d04686d")


@pytest.fixture(autouse=True)
def _pinned_now(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fixture's momentum is anchored in mid-2026, as the e2e server pins it.
    monkeypatch.setenv("APP_TRENDING_NOW", "2026-07-20")
    monkeypatch.setenv("APP_MOMENTUM_MIN_TOTAL", "1")


def _client(data_dir: Path, *, history: bool) -> TestClient:
    app = create_app(ROOT, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    client = TestClient(app)
    user = get_or_create_user(data_dir, provider="stub", subject="s", email="u@x.com", name="U")
    if history:
        for i, slug in enumerate(HEARD):
            app_user_state.add_highlight(
                data_dir,
                user.user_id,
                {
                    "id": f"h{i}",
                    "episode_slug": slug,
                    "kind": "moment",
                    "quote_text": "k",
                    "created_at": i + 1,
                },
            )
        app_user_state.add_interest(data_dir, user.user_id, "thc:managing-risk")
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def _slugs(rail: dict) -> list[str]:
    return [i["episode"]["slug"] for i in rail["items"]]


def test_everyone_gets_the_quote_cards_only(tmp_path: Path) -> None:
    body = _client(tmp_path, history=False).get("/api/app/discover/rails").json()
    assert body["scope"] == "corpus"
    [rail] = body["rails"]
    assert rail["rail"] == "quotes" and len(rail["items"]) == FIRST and rail["total"] > FIRST
    assert rail["fallback"] is False


def test_mine_is_every_rail_in_page_order_one_card_per_episode(tmp_path: Path) -> None:
    body = (
        _client(tmp_path, history=True)
        .get("/api/app/discover/rails", params={"scope": "mine"})
        .json()
    )
    assert body["scope"] == "mine"
    order = [r["rail"] for r in body["rails"]]
    assert order == [r for r in MINE_RAILS if r in order] and len(order) >= 4
    seen: list[str] = []
    for rail in body["rails"]:
        assert 1 <= len(rail["items"]) <= FIRST and len(rail["items"]) <= rail["total"]
        seen += _slugs(rail)
    assert len(seen) == len(set(seen)), "an episode appears in two rails"


def test_show_more_pages_on_from_the_server_without_repeating_the_page(tmp_path: Path) -> None:
    client = _client(tmp_path, history=True)
    rails = client.get("/api/app/discover/rails", params={"scope": "mine"}).json()["rails"]
    i = next(k for k, r in enumerate(rails) if r["total"] > FIRST and k > 0)
    above = [s for r in rails[:i] for s in _slugs(r)]
    first = _slugs(rails[i])
    more = client.get(
        f"/api/app/discover/rails/{rails[i]['rail']}",
        params={"scope": "mine", "offset": FIRST, "limit": 4, "exclude": above},
    ).json()
    assert more["total"] == rails[i]["total"]
    nxt = _slugs(more)
    assert nxt and not set(nxt) & set(above) and not set(nxt) & set(first)
    # The same rail asked from the top is first + more, in order: one list, sliced.
    whole = client.get(
        f"/api/app/discover/rails/{rails[i]['rail']}",
        params={"scope": "mine", "offset": 0, "limit": FIRST + 4, "exclude": above},
    ).json()
    assert _slugs(whole) == first + nxt


def test_cold_start_is_everyones_cards_flagged(tmp_path: Path) -> None:
    body = (
        _client(tmp_path, history=False)
        .get("/api/app/discover/rails", params={"scope": "mine"})
        .json()
    )
    [rail] = body["rails"]  # nothing of theirs: no personal rails
    assert rail["rail"] == "quotes" and rail["fallback"] is True and rail["items"]


def test_a_mine_rail_does_not_exist_under_everyone(tmp_path: Path) -> None:
    client = _client(tmp_path, history=True)
    assert client.get("/api/app/discover/rails/captured").status_code == 404
    app = create_app(ROOT, static_dir=False)
    app.state.app_data_dir = tmp_path
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    assert TestClient(app).get("/api/app/discover/rails").status_code == 401

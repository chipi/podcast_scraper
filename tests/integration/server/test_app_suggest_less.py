""" "Suggest less" (ADR-163 §6; operator 2026-10-11): one tap hides the subject a card's reason
names — that topic, person or storyline — from every Discover rail, and can be undone.

Over the COMMITTED fixture corpus (read only), through the real routes.
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
from podcast_scraper.server.app_user_store import get_or_create_user  # noqa: E402

# The rails, reels and Mine read the search sidecar (`interest_episode_index`), which is gitignored:
# this builds it once per session, rebuilds it when the corpus changes, and skips without a model.
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("app_validation_search_index")]

ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "app-validation-corpus" / "v3"


def _client(data_dir: Path) -> tuple[TestClient, str]:
    app = create_app(ROOT, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    client = TestClient(app)
    user = get_or_create_user(data_dir, provider="stub", subject="s", email="u@x.com", name="U")
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client, user.user_id


def _subjects(body: dict) -> list[str]:
    return [i["why"]["subject"]["id"] for i in body["items"] if i["why"].get("subject")]


def test_hiding_a_storyline_empties_the_thread_rail_and_undo_brings_it_back(tmp_path: Path) -> None:
    client, uid = _client(tmp_path)
    app_user_state.add_interest(tmp_path, uid, "thc:managing-risk")
    before = client.get(
        "/api/app/discover/rails/thread", params={"scope": "mine", "limit": 12}
    ).json()
    assert before["items"] and set(_subjects(before)) == {"thc:managing-risk"}

    r = client.post(
        "/api/app/discover/suggest-less/thc:managing-risk",
        json={"label": "Managing risk across domains"},
    )
    assert r.status_code == 200
    assert [(i["token"], i["label"]) for i in r.json()["items"]] == [
        ("thc:managing-risk", "Managing risk across domains")
    ]
    assert (
        client.get("/api/app/discover/rails/thread", params={"scope": "mine", "limit": 12}).json()[
            "items"
        ]
        == []
    )

    listed = client.get("/api/app/discover/suggest-less").json()["items"]
    assert [i["token"] for i in listed] == ["thc:managing-risk"]
    assert client.delete("/api/app/discover/suggest-less/thc:managing-risk").json()["items"] == []
    assert client.get(
        "/api/app/discover/rails/thread", params={"scope": "mine", "limit": 12}
    ).json()["items"]


def test_a_hidden_topic_leaves_every_card_that_names_it_and_only_those(tmp_path: Path) -> None:
    client, uid = _client(tmp_path)
    for i, slug in enumerate(("p07-a5eb82076a", "p07-2aceab172c", "p02-8fdd7b7f79")):
        app_user_state.add_highlight(
            tmp_path,
            uid,
            {
                "id": f"h{i}",
                "episode_slug": slug,
                "kind": "moment",
                "quote_text": "k",
                "created_at": i + 1,
            },
        )
    before = _subjects(
        client.get(
            "/api/app/discover/rails/another_view", params={"scope": "mine", "limit": 12}
        ).json()
    )
    assert "topic:risk-management" in before
    client.post(
        "/api/app/discover/suggest-less/topic:risk-management", json={"label": "risk management"}
    )
    after = _subjects(
        client.get(
            "/api/app/discover/rails/another_view", params={"scope": "mine", "limit": 12}
        ).json()
    )
    assert "topic:risk-management" not in after
    assert after, "the other topics' cards stay"


def test_repeat_is_idempotent_and_junk_is_refused(tmp_path: Path) -> None:
    client, _uid = _client(tmp_path)
    for _ in range(2):
        r = client.post("/api/app/discover/suggest-less/person:maya", json={"label": "Maya"})
    assert [i["token"] for i in r.json()["items"]] == ["person:maya"]
    assert (
        client.post("/api/app/discover/suggest-less/feed:p01", json={"label": "x"}).status_code
        == 422
    )
    assert (
        client.post("/api/app/discover/suggest-less/topic:x", json={"label": ""}).status_code == 422
    )


def test_signed_out_is_refused(tmp_path: Path) -> None:
    app = create_app(ROOT, static_dir=False)
    app.state.app_data_dir = tmp_path
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    assert TestClient(app).get("/api/app/discover/suggest-less").status_code == 401

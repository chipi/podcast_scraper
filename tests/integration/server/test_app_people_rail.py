"""People you keep hearing (ADR-163; Discover arc Step 3) — the rail's eval.

Frozen listener state over the COMMITTED fixture corpus (``tests/fixtures/app-validation-corpus/v3``,
read only): highlights in four episodes of The Long View (p07), where Dr. Elena Fischer and Skanda
Amarnath each guest twice and Alex Morgan hosts all four. Both guests also appear on Cross-Show
(p09); the host appears nowhere else. The rail must offer the guests' p09 episodes, taking turns, each
with a quote that guest says there — and nothing for the host, whose other episodes are on the show
the listener already hears. Its answer is pinned. Deterministic, disk-only, no LLM.
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
from podcast_scraper.server.app_people_rail import people_kept_hearing, people_rail  # noqa: E402
from podcast_scraper.server.app_slugs import resolve_slug  # noqa: E402
from podcast_scraper.server.app_user_store import get_or_create_user  # noqa: E402

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "app-validation-corpus" / "v3"
HEARD = ("p07-a5eb82076a", "p07-769d04686d", "p07-2aceab172c", "p07-3f1584920f")


def _seed(data_dir: Path, user_id: str) -> None:
    for i, slug in enumerate(HEARD):
        app_user_state.add_highlight(
            data_dir,
            user_id,
            {
                "id": f"h-{slug}",
                "episode_slug": slug,
                "kind": "moment",
                "quote_text": "kept",
                "created_at": 1000 + i,
            },
        )


def test_kept_hearing_counts_speakers_in_two_or_more_episodes(tmp_path: Path) -> None:
    _seed(tmp_path, "u1")
    got = [
        (k.person_id, k.episodes, set(k.shows)) for k in people_kept_hearing(ROOT, tmp_path, "u1")
    ]
    assert got == [
        ("person:alex-morgan", 4, {"p07"}),
        ("person:dr-elena-fischer", 2, {"p07"}),
        ("person:skanda-amarnath", 2, {"p07"}),
    ]


def test_recommends_them_on_other_shows_with_their_own_words(tmp_path: Path) -> None:
    _seed(tmp_path, "u1")
    cards = people_rail(ROOT, tmp_path, "u1", limit=10)
    assert cards
    for card in cards:
        w = card.why
        row = resolve_slug(ROOT, card.slug)
        assert row is not None and row.feed_id != "p07"  # not the show they heard them on
        assert w.kind == "person" and w.subject is not None and w.subject.kind == "person"
        assert w.subject.id != "person:alex-morgan"  # no other show to find the host on
        assert w.quote is not None and w.quote.episode_slug == card.slug
        assert w.quote.speaker == w.subject.label  # a quote that person says
        assert w.events == 2  # heard in two of the listener's episodes


def test_the_answer_is_pinned(tmp_path: Path) -> None:
    """The eval's frozen answer (2026-10-11). A change here is a ranking change: review it."""
    _seed(tmp_path, "u1")
    assert [(c.slug, c.why.subject.label) for c in people_rail(ROOT, tmp_path, "u1")] == [  # type: ignore[union-attr]
        ("p09-a4bbb5dde3", "Dr. Elena Fischer"),
        ("p09-6ac0bf4914", "Skanda Amarnath"),
        ("p09-db2d39ef2e", "Dr. Elena Fischer"),
        ("p09-39f3a01006", "Skanda Amarnath"),
    ]


def test_once_is_not_keeping_hearing(tmp_path: Path) -> None:
    app_user_state.add_highlight(
        tmp_path,
        "u1",
        {"id": "h", "episode_slug": HEARD[0], "kind": "moment", "quote_text": "k", "created_at": 1},
    )
    assert people_rail(ROOT, tmp_path, "u1") == []


def test_following_the_other_show_removes_it(tmp_path: Path) -> None:
    _seed(tmp_path, "u1")
    app_user_state.add_subscription(tmp_path, "u1", {"feed_id": "p09", "title": "Cross-Show"})
    assert people_rail(ROOT, tmp_path, "u1") == []


def test_the_route_returns_cards_with_their_reason(tmp_path: Path) -> None:
    app = create_app(ROOT, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    client = TestClient(app)
    user = get_or_create_user(tmp_path, provider="stub", subject="s", email="u@x.com", name="U")
    _seed(tmp_path, user.user_id)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    body = client.get("/api/app/discover/people").json()
    assert body["scope"] == "mine" and body["items"]
    item = body["items"][0]
    assert item["why"]["kind"] == "person" and item["why"]["events"] == 2
    assert item["episode"]["slug"] == item["why"]["quote"]["episode_slug"]
    assert TestClient(app).get("/api/app/discover/people").status_code == 401

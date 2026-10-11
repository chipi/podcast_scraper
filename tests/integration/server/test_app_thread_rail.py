"""Continue the thread (ADR-163; Discover arc Step 2.3) — the rail's eval.

Frozen listener state over the COMMITTED fixture corpus (``tests/fixtures/app-validation-corpus/v3``,
read only): the listener follows its one storyline, ``thc:managing-risk``. The rail must offer the
storyline's episodes they have not heard, captured or saved, newest first, and — once they have
listened in the storyline — only what was published after that listen. Its top results are pinned,
so a ranking change shows up in review. Deterministic, disk-only, no LLM.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from podcast_scraper.server import app_sessions, app_user_state  # noqa: E402
from podcast_scraper.server.app import create_app  # noqa: E402
from podcast_scraper.server.app_access import AccessPolicy  # noqa: E402
from podcast_scraper.server.app_slugs import resolve_slug  # noqa: E402
from podcast_scraper.server.app_thread_rail import thread_rail  # noqa: E402
from podcast_scraper.server.app_user_store import get_or_create_user  # noqa: E402

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "app-validation-corpus" / "v3"
STORYLINE = "thc:managing-risk"


def _epoch(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())


def _published(slug: str) -> str:
    row = resolve_slug(ROOT, slug)
    assert row is not None and row.publish_date
    return row.publish_date[:10]


def test_offers_the_storylines_episodes_newest_first_with_a_grounded_quote(tmp_path: Path) -> None:
    app_user_state.add_interest(tmp_path, "u1", STORYLINE)
    cards = thread_rail(ROOT, tmp_path, "u1", limit=20)
    assert len(cards) >= 5
    for card in cards:
        w = card.why
        assert w.kind == "thread"
        assert w.subject is not None and (w.subject.kind, w.subject.id) == ("storyline", STORYLINE)
        assert w.subject.label == "Managing risk across domains"
        assert w.quote is not None and w.quote.episode_slug == card.slug and w.quote.text
    days = [_published(c.slug) for c in cards]
    assert days == sorted(days, reverse=True)
    assert len({c.slug for c in cards}) == len(cards)  # one card per episode


def test_top_results_are_pinned(tmp_path: Path) -> None:
    """The eval's frozen answer (2026-10-11). A change here is a ranking change: review it."""
    app_user_state.add_interest(tmp_path, "u1", STORYLINE)
    assert [c.slug for c in thread_rail(ROOT, tmp_path, "u1", limit=5)] == [
        "p09-a4bbb5dde3",
        "p07-2aceab172c",
        "p07-a5eb82076a",
        "p03-5b8ca67393",
        "p05-25cb576153",
    ]


def test_after_a_listen_in_the_storyline_only_newer_episodes(tmp_path: Path) -> None:
    app_user_state.add_interest(tmp_path, "u1", STORYLINE)
    # Heard the 2025-07-09 dive-planning episode (about risk management) on 2025-07-10.
    app_user_state.append_listen_event(
        tmp_path, "u1", "p03-5b8ca67393", "p03", _epoch("2025-07-10")
    )
    got = thread_rail(ROOT, tmp_path, "u1", limit=20)
    assert got, "episodes published after 2025-07-10 continue the thread"
    assert all(_published(c.slug) > "2025-07-10" for c in got)
    assert "p03-5b8ca67393" not in [c.slug for c in got]


def test_never_an_episode_already_saved(tmp_path: Path) -> None:
    app_user_state.add_interest(tmp_path, "u1", STORYLINE)
    first = thread_rail(ROOT, tmp_path, "u1", limit=6)[0].slug
    app_user_state.add_favorite(tmp_path, "u1", {"kind": "episode", "ref": first, "added_at": 1})
    assert first not in [c.slug for c in thread_rail(ROOT, tmp_path, "u1", limit=6)]


def test_no_storyline_followed_means_no_rail(tmp_path: Path) -> None:
    app_user_state.add_interest(tmp_path, "u1", "topic:risk-management")  # a topic, not a storyline
    assert thread_rail(ROOT, tmp_path, "u1", limit=6) == []
    assert thread_rail(ROOT, tmp_path, "nobody", limit=6) == []


def _client(data_dir: Path) -> TestClient:
    app = create_app(ROOT, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    client = TestClient(app)
    user = get_or_create_user(data_dir, provider="stub", subject="s", email="u@x.com", name="U")
    app_user_state.add_interest(data_dir, user.user_id, STORYLINE)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def test_the_route_returns_cards_with_their_reason(tmp_path: Path) -> None:
    body = _client(tmp_path).get("/api/app/discover/thread").json()
    assert body["scope"] == "mine" and body["items"]
    item = body["items"][0]
    assert item["why"]["kind"] == "thread"
    assert item["why"]["subject"]["id"] == STORYLINE
    assert item["episode"]["slug"] == item["why"]["quote"]["episode_slug"]


def test_signed_out_is_refused(tmp_path: Path) -> None:
    app = create_app(ROOT, static_dir=False)
    app.state.app_data_dir = tmp_path
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    assert TestClient(app).get("/api/app/discover/thread").status_code == 401

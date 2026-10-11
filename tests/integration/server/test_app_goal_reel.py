"""N minutes on a topic (ADR-163; Discover arc Step 3, goal-shaped listening) — the reel's eval.

Over the COMMITTED fixture corpus (read only): clips ABOUT the topic across episodes, breadth first,
each saying what its insight says, within the budget. The answer is pinned.
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
from podcast_scraper.server.app_goal_reel import goal_reel  # noqa: E402
from podcast_scraper.server.app_reasons import MIN_SUPPORT, quote_support  # noqa: E402
from podcast_scraper.server.app_user_store import get_or_create_user  # noqa: E402

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "app-validation-corpus" / "v3"


def test_the_reel_is_pinned_breadth_first_and_says_what_its_insights_say() -> None:
    """The eval's frozen answer (2026-10-11). A change here is a ranking change: review it."""
    reel = goal_reel(ROOT, "topic:systems-thinking", minutes=15)
    assert [(c.episode_slug, c.moment.speaker) for c in reel] == [
        ("p07-769d04686d", "Alex Morgan"),
        ("p02-8fdd7b7f79", "Priya Nair"),
        ("p07-a5eb82076a", "Dr. Elena Fischer"),
        ("p09-a4bbb5dde3", "Dr. Elena Fischer"),
        ("p05-25cb576153", "Scott Bessent"),
        ("p07-2aceab172c", "Skanda Amarnath"),
    ]
    assert len({c.episode_slug for c in reel}) == len(reel)  # every episode once before any twice
    for c in reel:
        assert quote_support(c.moment.quote_text, c.moment.text) >= MIN_SUPPORT
        assert c.moment.end_ms > c.moment.start_ms


def test_the_budget_is_never_exceeded() -> None:
    reel = goal_reel(ROOT, "topic:systems-thinking", minutes=1)
    assert reel and sum(c.moment.end_ms - c.moment.start_ms for c in reel) <= 60_000


def test_a_topic_nobody_speaks_to_has_no_reel() -> None:
    assert goal_reel(ROOT, "topic:no-such-topic", minutes=15) == []


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


def test_the_route_reports_what_the_reel_actually_runs(tmp_path: Path) -> None:
    client, _uid = _client(tmp_path)
    body = client.get(
        "/api/app/discover/reel", params={"topic": "topic:systems-thinking", "minutes": 15}
    ).json()
    assert body["topic_label"] == "systems thinking"
    assert body["minutes"] == 15
    assert body["total_seconds"] == 72.0  # the small fixture: far less than the budget
    first = body["clips"][0]
    assert first["episode"]["slug"] == "p07-769d04686d" and first["end_ms"] > first["start_ms"]
    assert client.get("/api/app/discover/reel", params={"topic": "person:x"}).status_code == 422


def test_reel_topics_offers_only_topics_with_a_real_reel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_TRENDING_NOW", "2026-07-20")
    monkeypatch.setenv("APP_MOMENTUM_MIN_TOTAL", "1")
    client, uid = _client(tmp_path)
    everyone = client.get("/api/app/discover/reel-topics").json()
    assert everyone["scope"] == "corpus"
    # Rising topics with a reel; the rest (no clip says what its insight says) are not offered.
    assert [(i["topic_id"], i["clips"]) for i in everyone["items"]] == [
        ("topic:systems-thinking", 6),
        ("topic:risk-management", 4),
    ]
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
    mine = client.get("/api/app/discover/reel-topics", params={"scope": "mine"}).json()
    assert mine["scope"] == "mine"
    assert [i["topic_id"] for i in mine["items"]][:2] == [
        "topic:risk-management",
        "topic:systems-thinking",
    ]


def test_reel_topics_leave_out_what_the_listener_asked_to_see_less_of(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_TRENDING_NOW", "2026-07-20")
    monkeypatch.setenv("APP_MOMENTUM_MIN_TOTAL", "1")
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
    app_user_state.add_suggest_less(tmp_path, uid, "topic:risk-management", "risk management", 1)
    mine = client.get("/api/app/discover/reel-topics", params={"scope": "mine"}).json()
    assert "topic:risk-management" not in [i["topic_id"] for i in mine["items"]]
    assert mine["items"], "the other topics are still offered"

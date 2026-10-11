"""N minutes on a topic, theme or storyline (ADR-163; goal-shaped listening) — the reel's eval.

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

# The rails, reels and Mine read the search sidecar (`interest_episode_index`), which is gitignored:
# this builds it once per session, rebuilds it when the corpus changes, and skips without a model.
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("app_validation_search_index")]

ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "app-validation-corpus" / "v3"


def test_the_reel_is_pinned_breadth_first_and_says_what_its_insights_say() -> None:
    """The eval's frozen answer (2026-10-11). A change here is a ranking change: review it."""
    reel = goal_reel(ROOT, "topic:systems-thinking", minutes=15)
    assert [(c.episode_slug, c.moment.speaker) for c in reel] == [
        ("p02-8fdd7b7f79", "Priya Nair"),
        ("p07-3f1584920f", "Skanda Amarnath"),
        ("p07-769d04686d", "Dr. Elena Fischer"),
        ("p07-a5eb82076a", "Dr. Elena Fischer"),
        ("p05-693ecaa09e", "Scott Bessent"),
        ("p09-db2d39ef2e", "Dr. Elena Fischer"),
        ("p07-2aceab172c", "Skanda Amarnath"),
        ("p09-a4bbb5dde3", "Dr. Elena Fischer"),
        ("p05-25cb576153", "Scott Bessent"),
        ("p02-8fdd7b7f79", "Priya Nair"),
        ("p07-a5eb82076a", "Dr. Elena Fischer"),
    ]
    # Breadth first: every episode's first clip comes before any episode's second.
    episodes = [c.episode_slug for c in reel]
    distinct = len(set(episodes))
    assert len(set(episodes[:distinct])) == distinct
    for c in reel:
        assert quote_support(c.moment.quote_text, c.moment.text) >= MIN_SUPPORT
        assert c.moment.end_ms > c.moment.start_ms


def test_the_budget_is_never_exceeded() -> None:
    reel = goal_reel(ROOT, "topic:systems-thinking", minutes=1)
    assert reel and sum(c.moment.end_ms - c.moment.start_ms for c in reel) <= 60_000


def test_a_topic_nobody_speaks_to_has_no_reel() -> None:
    assert goal_reel(ROOT, "topic:no-such-topic", minutes=15) == []
    assert goal_reel(ROOT, "tc:no-such-theme", minutes=15) == []
    assert goal_reel(ROOT, "thc:no-such-storyline", minutes=15) == []


def test_a_storyline_reel_is_clips_about_any_of_its_member_topics() -> None:
    """Operator 2026-10-11: reels for themes and storylines too. A storyline is its member topics;
    every clip is ABOUT one of them, more episodes than any one member reaches, breadth first."""
    import json

    from podcast_scraper.server.app_goal_reel import reel_subject
    from podcast_scraper.server.app_slugs import resolve_slug

    subject = reel_subject(ROOT, "thc:managing-risk")
    assert (subject.kind, subject.label) == ("storyline", "Managing risk across domains")
    assert subject.topics == {
        "topic:endurance-sport",
        "topic:risk-management",
        "topic:safety-practices",
        "topic:systems-thinking",
    }
    reel = goal_reel(ROOT, "thc:managing-risk", minutes=15)
    assert len(reel) == 28 and len({c.episode_slug for c in reel}) == 16
    one_member = {c.episode_slug for c in goal_reel(ROOT, "topic:risk-management", minutes=15)}
    assert len({c.episode_slug for c in reel}) > len(one_member)
    for c in reel:
        row = resolve_slug(ROOT, c.episode_slug)
        assert row is not None
        gi = json.loads((ROOT / row.gi_relative_path).read_text(encoding="utf-8"))
        about = {
            e["to"]
            for e in gi["edges"]
            if e.get("type") == "ABOUT" and e.get("from") == c.moment.insight_id
        }
        assert about & subject.topics, (c.episode_slug, c.moment.insight_id)
    theme = reel_subject(ROOT, "tc:reliability")
    assert (theme.kind, theme.label) == ("theme", "reliability")


def test_the_route_names_a_storyline_reel_by_its_storyline(tmp_path: Path) -> None:
    client, _uid = _client(tmp_path)
    body = client.get(
        "/api/app/discover/reel", params={"subject": "thc:managing-risk", "minutes": 15}
    ).json()
    assert (body["subject_kind"], body["subject_label"]) == (
        "storyline",
        "Managing risk across domains",
    )
    assert len(body["clips"]) == 28


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
        "/api/app/discover/reel", params={"subject": "topic:systems-thinking", "minutes": 15}
    ).json()
    assert (body["subject_label"], body["subject_kind"]) == ("systems thinking", "topic")
    assert body["minutes"] == 15
    assert body["total_seconds"] == 132.0  # the small fixture: far less than the budget
    first = body["clips"][0]
    assert first["episode"]["slug"] == "p02-8fdd7b7f79" and first["end_ms"] > first["start_ms"]
    assert client.get("/api/app/discover/reel", params={"subject": "person:x"}).status_code == 422


def test_reel_topics_offers_only_topics_with_a_real_reel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_TRENDING_NOW", "2026-07-20")
    monkeypatch.setenv("APP_MOMENTUM_MIN_TOTAL", "1")
    client, uid = _client(tmp_path)
    everyone = client.get("/api/app/discover/reel-topics").json()
    assert everyone["scope"] == "corpus"
    # Rising topics, themes and storylines with a reel, taking turns; the rest (no clip says what
    # its insight says) are not offered. topic:reliability's reel is tc:reliability's (its other
    # member says nothing quotable), so it is the same offer and appears once.
    assert [(i["subject_id"], i["subject_kind"], i["clips"]) for i in everyone["items"]] == [
        ("topic:systems-thinking", "topic", 11),
        ("tc:reliability", "theme", 15),
        ("thc:managing-risk", "storyline", 28),
        ("topic:risk-management", "topic", 20),
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
    # Their own top topics lead the topic turns (most engaged first).
    assert [i["subject_id"] for i in mine["items"] if i["subject_kind"] == "topic"][:2] == [
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
    assert "topic:risk-management" not in [i["subject_id"] for i in mine["items"]]
    assert mine["items"], "the other topics are still offered"

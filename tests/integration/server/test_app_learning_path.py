"""Learn X this week (operator 2026-10-11): a seven-day path through a theme, storyline or topic.

Over the COMMITTED fixture corpus (read only): the subject's strongest episodes, oldest first, each
with its best clip ON the subject; frozen at the start; one step opens a day, never early.
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
from podcast_scraper.server.app_learning_path import (  # noqa: E402
    build_path,
    MIN_STEPS,
    PATH_DAYS,
)
from podcast_scraper.server.app_reasons import MIN_SUPPORT, quote_support  # noqa: E402
from podcast_scraper.server.app_slugs import resolve_slug  # noqa: E402
from podcast_scraper.server.app_user_store import get_or_create_user  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("app_validation_search_index")]

ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "app-validation-corpus" / "v3"
_DAY = 86_400


def test_the_path_is_pinned_oldest_first_each_step_saying_what_its_insight_says() -> None:
    """The eval's frozen answer (2026-10-11). A change here is a ranking change: review it."""
    path = build_path(ROOT, "thc:managing-risk")
    assert [(s.episode_slug, s.speaker) for s in path] == [
        ("p01-74565a3a19", "Sophie Laurent"),
        ("p02-4487cfb778", "Jonas Weisenthal"),
        ("p01-3cad86824c", "Noah Brier"),
        ("p01-f2b497cfa3", "Sophie Laurent"),
        ("p02-8fdd7b7f79", "Priya Nair"),
        ("p05-25cb576153", "Daniel Cho"),
        ("p03-5b8ca67393", "Marco Silva"),
    ]
    assert len(path) == PATH_DAYS and len({s.episode_slug for s in path}) == PATH_DAYS
    dates = []
    for s in path:
        row = resolve_slug(ROOT, s.episode_slug)
        assert row is not None
        dates.append(str(row.publish_date))
        assert quote_support(s.quote_text, s.text) >= MIN_SUPPORT
        assert s.end_ms > s.start_ms
    assert dates == sorted(dates)  # the week follows how the conversation developed


def test_a_subject_with_too_few_good_episodes_has_no_path() -> None:
    assert build_path(ROOT, "tc:dialogue") == []  # 2 episodes say something quotable on it
    assert MIN_STEPS == 3
    assert build_path(ROOT, "thc:no-such-storyline") == []


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


def test_without_a_path_discover_offers_subjects_that_have_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_TRENDING_NOW", "2026-07-20")
    monkeypatch.setenv("APP_MOMENTUM_MIN_TOTAL", "1")
    client, _uid = _client(tmp_path)
    body = client.get("/api/app/discover/learn").json()
    assert body["path"] is None
    assert [(c["subject_id"], c["subject_kind"], c["steps"]) for c in body["candidates"]] == [
        ("tc:reliability", "theme", 7),
        ("thc:managing-risk", "storyline", 7),
        ("tc:safety-practices", "theme", 7),
    ]


def test_one_step_opens_a_day_never_early_and_the_path_is_frozen(tmp_path: Path) -> None:
    client, uid = _client(tmp_path)
    started = client.post("/api/app/discover/learn", json={"subject_id": "thc:managing-risk"})
    assert started.status_code == 200, started.text
    path = started.json()
    assert (path["subject_kind"], path["subject_label"], path["day"]) == (
        "storyline",
        "Managing risk across domains",
        1,
    )
    assert [s["unlocked"] for s in path["steps"]] == [True] + [False] * 6
    # Today's step can be done; tomorrow's cannot.
    assert client.post("/api/app/discover/learn/steps/1/done").status_code == 409
    done = client.post("/api/app/discover/learn/steps/0/done").json()
    assert [s["done"] for s in done["steps"]][:2] == [True, False]

    # Two days on: day 3, three steps open, the first still done; the steps did not move.
    stored = app_user_state.get_learning_path(tmp_path, uid)
    assert stored is not None
    app_user_state.set_learning_path(
        tmp_path, uid, {**stored, "started_at": stored["started_at"] - 2 * _DAY}
    )
    later = client.get("/api/app/discover/learn").json()["path"]
    assert later["day"] == 3
    assert [s["unlocked"] for s in later["steps"]] == [True] * 3 + [False] * 4
    assert [s["episode"]["slug"] for s in later["steps"]] == [
        s["episode"]["slug"] for s in path["steps"]
    ]
    assert client.post("/api/app/discover/learn/steps/2/done").status_code == 200

    assert client.delete("/api/app/discover/learn").status_code == 204
    assert client.get("/api/app/discover/learn").json()["path"] is None
    assert client.post("/api/app/discover/learn/steps/0/done").status_code == 404


def test_a_subject_without_a_path_cannot_be_started(tmp_path: Path) -> None:
    client, _uid = _client(tmp_path)
    r = client.post("/api/app/discover/learn", json={"subject_id": "tc:dialogue"})
    assert r.status_code == 422
    assert client.post("/api/app/discover/learn", json={"subject_id": "x"}).status_code == 422


def test_a_page_can_ask_whether_its_subject_has_a_path(tmp_path: Path) -> None:
    """The topic card and theme / storyline pages offer "Learn this in a week" only with a path."""
    client, _uid = _client(tmp_path)
    body = client.get("/api/app/discover/learn/subjects/thc:managing-risk").json()
    assert (body["subject_kind"], body["subject_label"], body["steps"]) == (
        "storyline",
        "Managing risk across domains",
        7,
    )
    assert client.get("/api/app/discover/learn/subjects/tc:dialogue").status_code == 404
    assert client.get("/api/app/discover/learn/subjects/person:x").status_code == 422

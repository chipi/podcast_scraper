"""Learn X this week (operator 2026-10-11): a seven-day path through a theme, storyline or topic.

Over the COMMITTED fixture corpus (read only): the subject's strongest episodes, oldest first, each
with its best clip ON the subject; frozen at the start; one step opens a day (or early, once the
newest is done); up to three paths at once; Home's "In progress" lists them with the last
unfinished short listen.
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
        ("p01-3cad86824c", "Noah Brier"),
        ("p01-f2b497cfa3", "Sophie Laurent"),
        ("p02-8fdd7b7f79", "Priya Nair"),
        ("p05-25cb576153", "Daniel Cho"),
        ("p03-5b8ca67393", "Marco Silva"),
        ("p07-a5eb82076a", "Dr. Elena Fischer"),
        ("p07-2aceab172c", "Skanda Amarnath"),
    ]
    # 2026-10-12: clips stop before an ad break and where their speaker does, and 62 connective
    # templates stopped being insights; the strongest seven moved (a clip's length is part of its
    # score). p02-4487's clip had run into "We'll be right back after a quick word from our
    # sponsors."
    assert len(path) == PATH_DAYS and len({s.episode_slug for s in path}) == PATH_DAYS
    dates = []
    for s in path:
        row = resolve_slug(ROOT, s.episode_slug)
        assert row is not None
        dates.append(str(row.publish_date))
        assert quote_support(s.quote_text, s.text) >= MIN_SUPPORT
        assert s.end_ms > s.start_ms
    assert dates == sorted(dates)  # the week follows how the conversation developed


def test_no_speaker_or_show_takes_more_than_two_days() -> None:
    """Operator 2026-10-11 ("what about path variety?"): tc:reliability's week was four days of one
    host. Now at most MAX_PER_SPEAKER per speaker and MAX_PER_SHOW per show, in every path."""
    from collections import Counter

    from podcast_scraper.server.app_learning_path import MAX_PER_SHOW, MAX_PER_SPEAKER

    for subject in ("tc:reliability", "thc:managing-risk", "tc:risk-management"):
        path = build_path(ROOT, subject)
        assert len(path) == PATH_DAYS, subject
        speakers = Counter(s.speaker for s in path)
        shows = Counter(s.episode_slug.split("-")[0] for s in path)
        assert max(speakers.values()) <= MAX_PER_SPEAKER, (subject, speakers)
        assert max(shows.values()) <= MAX_PER_SHOW, (subject, shows)


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
    assert body["paths"] == [] and body["max_paths"] == 3
    assert [(c["subject_id"], c["subject_kind"], c["steps"]) for c in body["candidates"]] == [
        ("tc:reliability", "theme", 7),
        ("thc:managing-risk", "storyline", 7),
        ("tc:risk-management", "theme", 7),
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
    assert (
        client.post("/api/app/discover/learn/paths/thc:managing-risk/steps/1/done").status_code
        == 409
    )
    done = client.post("/api/app/discover/learn/paths/thc:managing-risk/steps/0/done").json()
    assert [s["done"] for s in done["steps"]][:2] == [True, False]

    # Two days on: day 3, three steps open, the first still done; the steps did not move.
    stored = app_user_state.get_learning_paths(tmp_path, uid)
    app_user_state._write(
        tmp_path,
        uid,
        "learning_paths",
        [{**stored[0], "started_at": stored[0]["started_at"] - 2 * _DAY}],
    )
    later = client.get("/api/app/discover/learn/paths/thc:managing-risk").json()
    assert later["day"] == 3
    assert [s["unlocked"] for s in later["steps"]] == [True] * 3 + [False] * 4
    assert [s["episode"]["slug"] for s in later["steps"]] == [
        s["episode"]["slug"] for s in path["steps"]
    ]
    assert (
        client.post("/api/app/discover/learn/paths/thc:managing-risk/steps/2/done").status_code
        == 200
    )

    assert client.delete("/api/app/discover/learn/paths/thc:managing-risk").status_code == 204
    assert client.get("/api/app/discover/learn").json()["paths"] == []
    assert (
        client.post("/api/app/discover/learn/paths/thc:managing-risk/steps/0/done").status_code
        == 404
    )
    assert client.delete("/api/app/discover/learn/paths/thc:managing-risk").status_code == 404


def test_once_the_open_day_is_done_the_next_can_be_opened_now(tmp_path: Path) -> None:
    """Operator 2026-10-11: a day a day is the rhythm, not a wall. "Next day now" only after the
    newest open day is done; a keen listener can finish in one sitting."""
    client, _uid = _client(tmp_path)
    path = client.post("/api/app/discover/learn", json={"subject_id": "thc:managing-risk"}).json()
    assert path["can_open_next"] is False
    assert (
        client.post("/api/app/discover/learn/paths/thc:managing-risk/next").status_code == 409
    )  # day 1 not done
    assert (
        client.post("/api/app/discover/learn/paths/thc:managing-risk/steps/0/done").json()[
            "can_open_next"
        ]
        is True
    )
    ahead = client.post("/api/app/discover/learn/paths/thc:managing-risk/next").json()
    assert ahead["day"] == 2
    assert [s["unlocked"] for s in ahead["steps"]] == [True, True] + [False] * 5
    assert ahead["can_open_next"] is False
    assert (
        client.post("/api/app/discover/learn/paths/thc:managing-risk/next").status_code == 409
    )  # day 2 not done
    for i in range(1, 7):
        assert (
            client.post(
                f"/api/app/discover/learn/paths/thc:managing-risk/steps/{i}/done"
            ).status_code
            == 200
        )
        if i < 6:
            assert (
                client.post("/api/app/discover/learn/paths/thc:managing-risk/next").status_code
                == 200
            )
    final = client.get("/api/app/discover/learn/paths/thc:managing-risk").json()
    assert final["day"] == 7 and all(s["unlocked"] and s["done"] for s in final["steps"])
    assert final["can_open_next"] is False
    assert (
        client.post("/api/app/discover/learn/paths/thc:managing-risk/next").status_code == 409
    )  # nothing left
    client.delete("/api/app/discover/learn/paths/thc:managing-risk")
    assert client.post("/api/app/discover/learn/paths/thc:managing-risk/next").status_code == 404


def test_up_to_three_paths_at_once_each_its_own(tmp_path: Path) -> None:
    """Operator 2026-10-11: several paths at once, capped at three ("a habit, not a backlog").
    A fourth is refused until one is stopped, or replaces one in the same request; each path keeps
    its own days; Discover stops offering what is being learned, and offers nothing at the cap."""
    client, _uid = _client(tmp_path)
    for sid in ("thc:managing-risk", "thc:second-order-effects", "tc:reliability"):
        assert client.post("/api/app/discover/learn", json={"subject_id": sid}).status_code == 200
    again = client.post("/api/app/discover/learn", json={"subject_id": "thc:managing-risk"})
    assert again.status_code == 200 and again.json()["day"] == 1  # already learning it: unchanged
    body = client.get("/api/app/discover/learn").json()
    assert [p["subject_id"] for p in body["paths"]] == [
        "thc:managing-risk",
        "thc:second-order-effects",
        "tc:reliability",
    ]
    assert body["candidates"] == []  # at the cap
    fourth = client.post("/api/app/discover/learn", json={"subject_id": "tc:risk-management"})
    assert (fourth.status_code, fourth.json()["detail"]) == (409, "at_cap")

    client.post(
        "/api/app/discover/learn/paths/thc:managing-risk/steps/0/done"
    )  # one path's progress is its own
    other = client.get("/api/app/discover/learn/paths/tc:reliability").json()
    assert not any(s["done"] for s in other["steps"])

    swapped = client.post(
        "/api/app/discover/learn",
        json={"subject_id": "tc:risk-management", "replace": "tc:reliability"},
    )
    assert swapped.status_code == 200, swapped.text
    left = [p["subject_id"] for p in client.get("/api/app/discover/learn").json()["paths"]]
    assert left == ["thc:managing-risk", "thc:second-order-effects", "tc:risk-management"]
    client.delete("/api/app/discover/learn/paths/thc:managing-risk")
    offered = client.get("/api/app/discover/learn").json()
    assert len(offered["paths"]) == 2
    learning = {p["subject_id"] for p in offered["paths"]}
    assert not learning & {c["subject_id"] for c in offered["candidates"]}


def test_a_day_whose_episode_left_still_lets_the_path_go_on(tmp_path: Path) -> None:
    """Review S1 (2026-10-11): a step whose episode left the corpus is not shown and can never be
    done; the days after keep their own day, and once the open day is gone "Next day now" opens
    the next. An index past the path is 404, not "its day has not come"."""
    client, uid = _client(tmp_path)
    client.post("/api/app/discover/learn", json={"subject_id": "thc:managing-risk"})
    stored = app_user_state.get_learning_paths(tmp_path, uid)[0]
    steps = [dict(s) for s in stored["steps"]]
    steps[0]["episode_slug"] = "gone-from-the-corpus"
    app_user_state._write(tmp_path, uid, "learning_paths", [{**stored, "steps": steps}])
    path = client.get("/api/app/discover/learn/paths/thc:managing-risk").json()
    assert [s["day"] for s in path["steps"]] == [2, 3, 4, 5, 6, 7]
    assert path["can_open_next"] is True  # day 1 can never be done
    assert path["next_opens_at"] == stored["started_at"] + _DAY
    ahead = client.post("/api/app/discover/learn/paths/thc:managing-risk/next").json()
    assert ahead["day"] == 2
    done = client.post("/api/app/discover/learn/paths/thc:managing-risk/steps/1/done").json()
    assert [s["done"] for s in done["steps"]][:1] == [True]  # day 2, the one tapped
    bad = client.post("/api/app/discover/learn/paths/thc:managing-risk/steps/99/done")
    assert bad.status_code == 404


def test_a_corrupt_paths_file_is_never_overwritten(tmp_path: Path) -> None:
    """Review S6: starting a path must not turn an unreadable learning_paths.json into one path,
    losing the others."""
    client, uid = _client(tmp_path)
    client.post("/api/app/discover/learn", json={"subject_id": "thc:managing-risk"})
    f = tmp_path / "users" / uid / "learning_paths.json"
    f.write_text("{not json", encoding="utf-8")
    with pytest.raises(Exception):
        app_user_state.add_learning_path(tmp_path, uid, {"subject_id": "tc:x", "steps": []}, cap=3)
    assert f.read_text(encoding="utf-8") == "{not json"


def test_in_progress_lists_the_paths_and_the_last_unfinished_short_listen(tmp_path: Path) -> None:
    """Home's door (operator 2026-10-11): every path, and the last short listen left unfinished,
    resumable at its clip; finishing it (DELETE) clears it, and a newer one replaces it."""
    client, _uid = _client(tmp_path)
    assert client.get("/api/app/discover/in-progress").json() == {"paths": [], "reel": None}
    client.post("/api/app/discover/learn", json={"subject_id": "thc:managing-risk"})
    put = client.put(
        "/api/app/discover/reel/progress",
        json={"subject_id": "topic:systems-thinking", "index": 4, "clips": 9},
    )
    assert put.status_code == 200, put.text
    body = client.get("/api/app/discover/in-progress").json()
    assert [p["subject_id"] for p in body["paths"]] == ["thc:managing-risk"]
    reel = body["reel"]
    assert (reel["subject_id"], reel["subject_kind"], reel["subject_label"]) == (
        "topic:systems-thinking",
        "topic",
        "systems thinking",
    )
    assert (reel["index"], reel["clips"]) == (4, 9)
    # The clip it resumes at, for the card: clip 5 of the systems-thinking reel (re-pinned
    # 2026-10-13 with that reel, test_app_goal_reel).
    assert (reel["episode"]["slug"], reel["speaker"]) == ("p07-769d04686d", "Dr. Elena Fischer")
    assert reel["quote_text"]
    client.put(
        "/api/app/discover/reel/progress",
        json={"subject_id": "thc:second-order-effects", "index": 99, "clips": 12},
    )
    newer = client.get("/api/app/discover/in-progress").json()["reel"]
    assert (newer["subject_id"], newer["subject_label"], newer["index"]) == (
        "thc:second-order-effects",
        "Second-order effects",
        11,  # clamped to the last clip
    )
    bad = client.put(
        "/api/app/discover/reel/progress", json={"subject_id": "person:x", "index": 0, "clips": 1}
    )
    assert bad.status_code == 422
    # Finishing ANOTHER short listen (another device) leaves this one; finishing this one clears it.
    other = client.delete(
        "/api/app/discover/reel/progress", params={"subject": "topic:reliability"}
    )
    assert other.status_code == 204
    assert client.get("/api/app/discover/in-progress").json()["reel"] is not None
    gone = client.delete(
        "/api/app/discover/reel/progress", params={"subject": "thc:second-order-effects"}
    )
    assert gone.status_code == 204
    assert client.get("/api/app/discover/in-progress").json()["reel"] is None
    assert client.delete("/api/app/discover/reel/progress").status_code == 422  # whose?


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


def test_the_offer_counts_the_days_the_path_has() -> None:
    """Learn offers count days without reading transcripts (``path_plan``); a started path has
    exactly those days, on those episodes."""
    from podcast_scraper.server.app_learning_path import build_path, path_plan

    plan = path_plan(ROOT, "thc:second-order-effects")
    steps = build_path(ROOT, "thc:second-order-effects")
    assert len(plan) == len(steps) == 7
    assert [e.slug for e in plan] == [s.episode_slug for s in steps]

"""A board's Brief, its short listen, and boards in search (RFC-119 amendment 2026-10-12;
#2340, #2341), over the committed fixture corpus.

The rules pinned here, each from the operator's decisions of 2026-10-12:
- the Brief reads only what is saved on the board: a saved show adds no episodes to it;
- what it says the episodes share, at least two of them do share (checked against their KGs);
- the short listen plays the saved moments first, exactly as saved, then each saved episode's clip;
- search finds a board by its name first, then by something saved on it, and only the user's own.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions, app_user_state
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_slugs import _slug_index
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = [pytest.mark.integration]

CORPUS = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "app-validation-corpus" / "v3"


def _client(tmp_path: Path, subject: str = "s1") -> tuple[TestClient, Path, str]:
    data_dir = tmp_path / "appdata"
    app = create_app(CORPUS, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(
        data_dir, provider="stub", subject=subject, email=f"{subject}@x.com", name=subject
    )
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client, data_dir, user.user_id


def _one_show_episodes(n: int) -> list[tuple[str, object]]:
    """``n`` episodes of the first show that has as many: one show's episodes share subjects."""
    by_feed: dict[str, list[tuple[str, object]]] = {}
    for slug, row in sorted(_slug_index(CORPUS).items()):
        by_feed.setdefault(str(row.feed_id), []).append((slug, row))
    return next(eps[:n] for _feed, eps in sorted(by_feed.items()) if len(eps) >= n)


def _kg_subjects(row: object) -> set[str]:
    kg = json.loads((CORPUS / row.kg_relative_path).read_text("utf-8"))  # type: ignore[attr-defined]
    return {str(n["id"]) for n in kg["nodes"] if n.get("type") in ("Topic", "Person")}


def _kg_hosts(row: object) -> set[str]:
    kg = json.loads((CORPUS / row.kg_relative_path).read_text("utf-8"))  # type: ignore[attr-defined]
    return {
        str(n["id"])
        for n in kg["nodes"]
        if n.get("type") == "Person" and (n.get("properties") or {}).get("role") == "host"
    }


def _board(client: TestClient, name: str, items: list[dict]) -> str:
    cid = client.post("/api/app/collections", json={"name": name}).json()["id"]
    for item in items:
        assert client.post(f"/api/app/collections/{cid}/items", json=item).status_code == 200
    return str(cid)


def test_the_brief_reads_only_what_is_saved_and_says_what_its_episodes_share(
    tmp_path: Path,
) -> None:
    client, data_dir, uid = _client(tmp_path)
    (a, row_a), (b, row_b), (c, row_c) = _one_show_episodes(3)
    app_user_state.add_highlight(
        data_dir,
        uid,
        {"id": "h1", "episode_slug": c, "kind": "span", "created_at": 1, "quote_text": "x"},
    )
    feed = str(row_a.feed_id)  # type: ignore[attr-defined]
    cid = _board(
        client,
        "Deep dive",
        [
            {"kind": "episode", "ref": a},
            {"kind": "episode", "ref": b},
            {"kind": "highlight", "ref": "h1"},
            {"kind": "show", "ref": feed},
            {"kind": "search", "ref": "risk"},
            {"kind": "link", "ref": "https://example.org/a"},
        ],
    )
    brief = client.get(f"/api/app/collections/{cid}/brief").json()

    assert brief["name"] == "Deep dive"
    # Three episodes: two saved, one through its saved moment. The saved SHOW adds none of its
    # other episodes (operator 2026-10-12: only what is saved on the board).
    assert brief["holds"] == {
        "episodes": 3,
        "moments": 1,
        "shows": 1,
        "subjects": 0,
        "searches": 1,
        "links": 1,
    }
    assert [p["episode"]["slug"] for p in brief["points"]] == [a, b, c][: len(brief["points"])]
    assert brief["points"], "the fixture's episodes have insights"
    subjects = [_kg_subjects(r) for r in (row_a, row_b, row_c)]
    assert brief["about"], "three episodes of one show share subjects"
    for s in brief["about"]:
        shared = sum(1 for kg in subjects if s["id"] in kg)
        assert s["episodes"] >= 2 and shared == s["episodes"], s
    # The show's host is in all three and says nothing about the board.
    hosts = set().union(*(_kg_hosts(r) for r in (row_a, row_b, row_c)))
    assert hosts, "the fixture's KGs carry the host's role"
    assert not {s["id"] for s in brief["about"]} & hosts
    for t in brief["threads"]:
        assert t["episodes"] >= 2 and t["kind"] in ("storyline", "theme")


def test_a_board_of_shows_and_searches_has_nothing_to_brief(tmp_path: Path) -> None:
    client, _d, _u = _client(tmp_path)
    (_slug, row), *_ = _one_show_episodes(1)
    cid = _board(
        client,
        "Shows only",
        [{"kind": "show", "ref": str(row.feed_id)}, {"kind": "search", "ref": "q"}],  # type: ignore[attr-defined]
    )
    brief = client.get(f"/api/app/collections/{cid}/brief").json()
    assert brief["holds"]["episodes"] == 0
    assert (brief["about"], brief["points"], brief["threads"]) == ([], [], [])


def test_another_users_board_is_not_found(tmp_path: Path) -> None:
    owner, _d, _u = _client(tmp_path, "owner")
    cid = _board(owner, "Mine", [])
    other, _d2, _u2 = _client(tmp_path, "other")
    assert other.get(f"/api/app/collections/{cid}/brief").status_code == 404
    assert other.get(f"/api/app/collections/{cid}/reel").status_code == 404
    assert owner.get("/api/app/collections/col_nope/brief").status_code == 404


def test_the_short_listen_plays_saved_moments_first_then_each_saved_episode(
    tmp_path: Path,
) -> None:
    client, data_dir, uid = _client(tmp_path)
    (a, _ra), (b, _rb), (c, _rc) = _one_show_episodes(3)
    app_user_state.add_highlight(
        data_dir,
        uid,
        {
            "id": "h1",
            "episode_slug": c,
            "kind": "span",
            "created_at": 1,
            "start_ms": 61_000,
            "end_ms": 79_000,
            "quote_text": "the words I saved",
            "speaker": "Host",
        },
    )
    # A saved moment with no time cannot be played; it is left out, not played from 0.
    app_user_state.add_highlight(
        data_dir, uid, {"id": "h2", "episode_slug": a, "kind": "insight", "created_at": 2}
    )
    cid = _board(
        client,
        "Listen",
        [
            {"kind": "episode", "ref": a},
            {"kind": "highlight", "ref": "h1"},
            {"kind": "highlight", "ref": "h2"},
            {"kind": "episode", "ref": b},
        ],
    )
    reel = client.get(f"/api/app/collections/{cid}/reel", params={"minutes": 15}).json()

    assert (reel["subject_id"], reel["subject_kind"], reel["subject_label"]) == (
        f"board:{cid}",
        "board",
        "Listen",
    )
    first, *rest = reel["clips"]
    assert (first["episode"]["slug"], first["start_ms"], first["end_ms"]) == (c, 61_000, 79_000)
    assert first["quote_text"] == "the words I saved"
    assert [x["episode"]["slug"] for x in rest] == [a, b][: len(rest)]
    assert rest, "the fixture's episodes have clips"
    assert reel["total_seconds"] == round(
        sum(x["end_ms"] - x["start_ms"] for x in reel["clips"]) / 1000.0, 1
    )

    short = client.get(f"/api/app/collections/{cid}/reel", params={"minutes": 1}).json()
    assert short["total_seconds"] <= 60


def test_an_unfinished_board_listen_resumes_from_home_and_goes_with_its_board(
    tmp_path: Path,
) -> None:
    client, _d, _u = _client(tmp_path)
    (a, _ra), (b, _rb) = _one_show_episodes(2)
    cid = _board(client, "Commute", [{"kind": "episode", "ref": a}, {"kind": "episode", "ref": b}])
    put = client.put(
        "/api/app/discover/reel/progress",
        json={"subject_id": f"board:{cid}", "index": 1, "clips": 2, "minutes": 15},
    )
    assert put.status_code == 200, put.text
    reel = client.get("/api/app/discover/in-progress").json()["reel"]
    assert (reel["subject_kind"], reel["subject_label"], reel["index"]) == ("board", "Commute", 1)
    assert reel["episode"]["slug"] == b

    client.delete(f"/api/app/collections/{cid}")
    assert client.get("/api/app/discover/in-progress").json()["reel"] is None
    gone = client.put(
        "/api/app/discover/reel/progress",
        json={"subject_id": f"board:{cid}", "index": 0, "clips": 2, "minutes": 15},
    )
    assert gone.status_code == 404


def test_search_finds_a_board_by_its_name_first_then_by_what_is_saved_on_it(
    tmp_path: Path,
) -> None:
    client, data_dir, uid = _client(tmp_path)
    (a, row_a), *_ = _one_show_episodes(1)
    title = str(row_a.episode_title)  # type: ignore[attr-defined]
    word = max((w for w in title.split() if w.isalpha()), key=len)
    app_user_state.add_highlight(
        data_dir,
        uid,
        {
            "id": "h1",
            "episode_slug": a,
            "kind": "span",
            "created_at": 1,
            "quote_text": "Pressure gauges never lie",
        },
    )
    by_item = _board(client, "Weekend", [{"kind": "episode", "ref": a}])
    by_name = _board(client, f"All about {word}", [])
    by_moment = _board(client, "Quotes", [{"kind": "highlight", "ref": "h1"}])
    _board(client, "Nothing here", [{"kind": "link", "ref": "https://example.org"}])
    other, _d, _u = _client(tmp_path, "other")
    _board(other, f"Their {word}", [])

    hits = client.get("/api/app/collections/search", params={"q": word.upper()}).json()["items"]
    assert [h["collection"]["id"] for h in hits] == [by_name, by_item]
    assert hits[0]["matched"] == "name"
    assert (hits[1]["matched"], hits[1]["item_kind"], hits[1]["item_label"]) == (
        "item",
        "episode",
        title,
    )
    # Every word, in any order.
    moment = client.get("/api/app/collections/search", params={"q": "lie gauges"}).json()["items"]
    assert [(h["collection"]["id"], h["item_kind"]) for h in moment] == [(by_moment, "highlight")]
    none = client.get("/api/app/collections/search", params={"q": "gauges zebra"}).json()["items"]
    assert none == []

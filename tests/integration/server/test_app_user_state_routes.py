"""Integration tests for per-user state routes — playback, queue, library (#1065).

Auth is established by forging a signed session cookie (the secret is known in-test),
avoiding the full OAuth dance.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = [pytest.mark.integration]


def _authed_client(tmp_path: Path) -> TestClient:
    app = create_app(tmp_path, static_dir=False)
    data_dir = tmp_path / "appdata"
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="stub", subject="s1", email="j@x.com", name="J")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def test_requires_auth(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    client = TestClient(app)
    assert client.get("/api/app/queue").status_code == 401
    assert client.get("/api/app/playback/ep").status_code == 401
    assert client.get("/api/app/library").status_code == 401
    assert client.get("/api/app/me/stats").status_code == 401
    assert client.post("/api/app/listen/ep").status_code == 401


def test_listen_then_my_stats(tmp_path: Path) -> None:
    client = _authed_client(tmp_path)
    # Empty to start.
    empty = client.get("/api/app/me/stats")
    assert empty.status_code == 200, empty.text
    assert empty.json()["episodes"] == 0 and empty.json()["day_streak"] == 0
    # Record an open (no corpus in-test → feed_id resolves to None, but the event still logs).
    assert client.post("/api/app/listen/ep1").status_code == 204
    stats = client.get("/api/app/me/stats").json()
    assert stats["episodes"] == 1
    assert stats["day_streak"] == 1
    assert stats["active_days"] == 1
    assert stats["daily"][-1]["count"] == 1  # today's bucket


def test_listen_logs_without_feed_when_slug_resolve_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Analytics must never break opening an episode: if slug→feed resolution throws, the listen
    # event is still recorded (with feed_id=None) and the request succeeds.
    from podcast_scraper.server.routes import app_user_state as routes

    def boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("corpus index exploded")

    monkeypatch.setattr(routes, "resolve_slug", boom)
    client = _authed_client(tmp_path)
    assert client.post("/api/app/listen/ep1").status_code == 204
    assert client.get("/api/app/me/stats").json()["episodes"] == 1


def test_playback_save_and_resume(tmp_path: Path) -> None:
    client = _authed_client(tmp_path)
    assert client.get("/api/app/playback/ep").json()["position_seconds"] == 0.0
    put = client.put("/api/app/playback/ep", json={"position_seconds": 42.5})
    assert put.status_code == 200, put.text
    assert put.json()["position_seconds"] == 42.5
    assert client.get("/api/app/playback/ep").json()["position_seconds"] == 42.5


def test_playback_list_pages_and_filters_when_asked(tmp_path: Path) -> None:
    client = _authed_client(tmp_path)
    for slug, pos in [("a", 0.5), ("b", 120), ("c", 300)]:
        assert client.put(f"/api/app/playback/{slug}", json={"position_seconds": pos}).status_code
    full = client.get("/api/app/playback").json()
    assert set(full) == {"items"} and len(full["items"]) == 3
    started = client.get("/api/app/playback", params={"limit": 10, "in_progress": True}).json()
    assert {p["slug"] for p in started["items"]} == {"b", "c"} and started["total"] == 2
    some = client.get(
        "/api/app/playback", params=[("limit", 10), ("slugs", "a"), ("slugs", "c")]
    ).json()
    assert {p["slug"] for p in some["items"]} == {"a", "c"}
    one = client.get("/api/app/playback", params={"limit": 1}).json()
    assert len(one["items"]) == 1 and one["total"] == 3


def test_recap_route(tmp_path: Path) -> None:
    """The windowed read (#1914). Auth-gated, and honest about what it does not have."""
    client = _authed_client(tmp_path)

    empty = client.get("/api/app/me/recap")
    assert empty.status_code == 200, empty.text
    assert empty.json()["window"] == "week"
    assert empty.json()["listening_seconds"] == 0.0
    assert len(empty.json()["by_day"]) == 7

    # Two saves 20s apart is 20s of listening — the position DELTA, not the position.
    client.put("/api/app/playback/ep", json={"position_seconds": 100.0})
    client.put("/api/app/playback/ep", json={"position_seconds": 120.0})

    week = client.get("/api/app/me/recap?window=week").json()
    assert week["listening_seconds"] == 20.0
    assert week["days_recorded"] == 1
    # The anti-fabrication pair: a year window covers one recorded day, and says so.
    year = client.get("/api/app/me/recap?window=year").json()
    assert year["days_in_window"] == 365 and year["days_recorded"] == 1

    # An unknown window is rejected by the enum rather than silently defaulting.
    assert client.get("/api/app/me/recap?window=decade").status_code == 422


def test_recap_requires_a_session(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    app.state.app_data_dir = tmp_path / "appdata"
    app.state.session_secret = "test-secret"
    assert TestClient(app).get("/api/app/me/recap").status_code == 401


def test_queue_item_routes_are_replay_safe(tmp_path: Path) -> None:
    """The item-level half of the queue API (#1925).

    PUT /queue replaces the whole list, so an offline write replayed later overwrites whatever
    another device did meanwhile — which is why the client refuses to write a cached queue at all.
    These routes carry the same intents idempotently, so the outbox can replay them.
    """
    client = _authed_client(tmp_path)

    # Add appends, and a repeat is a no-op rather than a duplicate — the replay case.
    assert client.post("/api/app/queue/items", json={"slug": "a"}).json()["items"] == ["a"]
    assert client.post("/api/app/queue/items", json={"slug": "b"}).json()["items"] == ["a", "b"]
    assert client.post("/api/app/queue/items", json={"slug": "a"}).json()["items"] == ["a", "b"]

    # "Play next" anchors after another slug, and re-anchoring moves it rather than duplicating.
    assert client.post("/api/app/queue/items", json={"slug": "c", "after": "a"}).json()[
        "items"
    ] == ["a", "c", "b"]
    assert client.post("/api/app/queue/items", json={"slug": "c", "after": "b"}).json()[
        "items"
    ] == ["a", "b", "c"]

    # An anchor that is gone means the front: the user asked for it NEXT.
    assert client.post("/api/app/queue/items", json={"slug": "d", "after": "zzz"}).json()[
        "items"
    ] == ["d", "a", "b", "c"]

    # Remove is idempotent too, so a replayed removal cannot fail.
    assert client.delete("/api/app/queue/items/d").json()["items"] == ["a", "b", "c"]
    assert client.delete("/api/app/queue/items/d").json()["items"] == ["a", "b", "c"]

    assert client.get("/api/app/queue").json()["items"] == ["a", "b", "c"]


def test_queue_item_routes_require_a_session(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    app.state.app_data_dir = tmp_path / "appdata"
    app.state.session_secret = "test-secret"
    client = TestClient(app)
    assert client.post("/api/app/queue/items", json={"slug": "a"}).status_code == 401
    assert client.delete("/api/app/queue/items/a").status_code == 401


def test_queue_roundtrip(tmp_path: Path) -> None:
    client = _authed_client(tmp_path)
    assert client.get("/api/app/queue").json()["items"] == []
    assert client.put("/api/app/queue", json={"items": ["a", "b"]}).status_code == 200
    assert client.get("/api/app/queue").json()["items"] == ["a", "b"]


def test_interests_roundtrip_and_dedup(tmp_path: Path) -> None:
    client = _authed_client(tmp_path)
    assert client.get("/api/app/interests").json()["items"] == []
    put = client.put("/api/app/interests", json={"items": ["tc:ai", "tc:health", "tc:ai", ""]})
    assert put.status_code == 200, put.text
    # Dedup + blank-drop, order preserved.
    assert put.json()["items"] == ["tc:ai", "tc:health"]
    assert client.get("/api/app/interests").json()["items"] == ["tc:ai", "tc:health"]


def test_follow_unfollow_interest_token(tmp_path: Path) -> None:
    client = _authed_client(tmp_path)
    # Follow a topic and a person from an entity card (URL-encoded `:`); idempotent.
    client.post("/api/app/interests/topic%3Aai")
    body = client.post("/api/app/interests/person%3Ajane").json()
    assert body["items"] == ["topic:ai", "person:jane"]
    assert client.post("/api/app/interests/topic%3Aai").json()["items"] == [
        "topic:ai",
        "person:jane",
    ]
    # Unfollow.
    assert client.delete("/api/app/interests/topic%3Aai").json()["items"] == ["person:jane"]


def test_follow_and_save_write_timestamped_engagement(tmp_path: Path) -> None:
    # RFC-103 Phase 2: a follow logs a timestamped event and a save stamps added_at, so the
    # engagement aggregator picks both up (the routes are the only writers of these signals).
    from podcast_scraper.server.app_engagement_series import engagement_series

    client = _authed_client(tmp_path)
    data_dir = tmp_path / "appdata"
    client.post("/api/app/interests/thc%3Amanaging-risk")  # follow a storyline
    client.put("/api/app/favorites", json={"kind": "topic", "ref": "topic:ai", "label": "AI"})

    ents = {(e["kind"], e["entity_id"]) for e in engagement_series(data_dir)["entities"]}
    assert ("storyline", "thc:managing-risk") in ents  # follow logged with a timestamp
    assert ("topic", "topic:ai") in ents  # favorite stamped added_at


def _write_kg_episode(root: Path, *, stem: str, episode_id: str) -> None:
    """Minimal episode so favorite-episode hydration resolves a real slug."""
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": "f", "title": "Show", "url": "https://p.example/f.xml"},
        "episode": {"episode_id": episode_id, "title": "Hello", "published_date": "2024-01-01"},
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / f"{stem}.txt").write_text("hi", encoding="utf-8")


def test_favorites_roundtrip_hydrated(tmp_path: Path) -> None:
    from podcast_scraper.server.app_slugs import slug_for_row
    from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

    _write_kg_episode(tmp_path, stem="0001-hello", episode_id="ep1")
    slug = slug_for_row(build_catalog_rows_cumulative(tmp_path)[0])
    client = _authed_client(tmp_path)

    assert client.get("/api/app/favorites").json() == {"episodes": [], "entities": []}
    # save an episode via the route (hydrated fresh from the catalog)
    body = client.put(
        "/api/app/favorites", json={"kind": "episode", "ref": slug, "label": "Hello"}
    ).json()
    assert [e["slug"] for e in body["episodes"]] == [slug]
    # remove it (url-encoded ref)
    after = client.delete(f"/api/app/favorites/episode/{slug}").json()
    assert after["episodes"] == []


def test_favorites_entity_roundtrip(tmp_path: Path) -> None:
    # F2.2: shows/topics/people/storylines are favoritable and come back in the `entities` group.
    client = _authed_client(tmp_path)
    assert client.get("/api/app/favorites").json()["entities"] == []
    body = client.put(
        "/api/app/favorites",
        json={"kind": "topic", "ref": "topic:ai", "label": "AI"},
    ).json()
    assert body["entities"] == [
        {"kind": "topic", "ref": "topic:ai", "label": "AI", "sublabel": None, "color": None}
    ]
    client.put("/api/app/favorites", json={"kind": "show", "ref": "p05", "label": "The Drift"})
    kinds = {e["kind"] for e in client.get("/api/app/favorites").json()["entities"]}
    assert kinds == {"topic", "show"}
    after = client.delete("/api/app/favorites/topic/topic:ai").json()
    assert [e["kind"] for e in after["entities"]] == ["show"]


def _two_saved_episodes(tmp_path: Path) -> tuple[TestClient, str, str]:
    from podcast_scraper.server.app_slugs import slug_for_row
    from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

    _write_kg_episode(tmp_path, stem="0001-hello", episode_id="ep1")
    doc_path = tmp_path / "metadata" / "0002-zebra.metadata.json"
    _write_kg_episode(tmp_path, stem="0002-zebra", episode_id="ep2")
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    doc["episode"]["title"] = "Apple Orchard"
    doc_path.write_text(json.dumps(doc), encoding="utf-8")
    slugs = {r.episode_title: slug_for_row(r) for r in build_catalog_rows_cumulative(tmp_path)}
    client = _authed_client(tmp_path)
    hello, apple = slugs["Hello"], slugs["Apple Orchard"]
    client.put("/api/app/favorites", json={"kind": "episode", "ref": hello, "label": "Hello"})
    client.put("/api/app/favorites", json={"kind": "episode", "ref": apple, "label": "Apple"})
    client.put("/api/app/favorites", json={"kind": "topic", "ref": "topic:ai", "label": "AI"})
    client.put("/api/app/favorites", json={"kind": "show", "ref": "p05", "label": "The Drift"})
    client.patch(f"/api/app/favorites/episode/{hello}", json={"color": "red"})
    return client, hello, apple


def test_favorites_unpaged_response_is_unchanged_for_old_clients(tmp_path: Path) -> None:
    # 1.0.2 sends no paging params: every favourite, and NO paging fields (not even as null).
    client, hello, apple = _two_saved_episodes(tmp_path)
    body = client.get("/api/app/favorites").json()
    assert set(body) == {"episodes", "entities"}
    assert [e["slug"] for e in body["episodes"]] == [apple, hello]  # newest first
    assert {e["kind"] for e in body["entities"]} == {"topic", "show"}


def test_favorites_paged_by_kind_with_counts(tmp_path: Path) -> None:
    client, hello, apple = _two_saved_episodes(tmp_path)
    page = client.get("/api/app/favorites", params={"kind": "episode", "limit": 1}).json()
    assert [e["slug"] for e in page["episodes"]] == [apple]
    assert page["entities"] == []
    assert page["total"] == 2
    assert page["counts"] == {
        "episode": 2,
        "show": 1,
        "topic": 1,
        "person": 0,
        "theme": 0,
        "storyline": 0,
    }
    nxt = client.get(
        "/api/app/favorites", params={"kind": "episode", "limit": 1, "offset": 1}
    ).json()
    assert [e["slug"] for e in nxt["episodes"]] == [hello]


def test_favorites_paged_search_colour_and_sort(tmp_path: Path) -> None:
    client, hello, apple = _two_saved_episodes(tmp_path)
    by_title = client.get(
        "/api/app/favorites", params={"kind": "episode", "sort": "title", "limit": 10}
    ).json()
    assert [e["title"] for e in by_title["episodes"]] == ["Apple Orchard", "Hello"]
    searched = client.get("/api/app/favorites", params={"q": "ORCH", "limit": 10}).json()
    assert [e["slug"] for e in searched["episodes"]] == [apple]
    assert searched["entities"] == []
    assert searched["counts"]["episode"] == 1 and searched["counts"]["topic"] == 0
    red = client.get("/api/app/favorites", params={"color": "red", "limit": 10}).json()
    assert [e["slug"] for e in red["episodes"]] == [hello]
    assert red["total"] == 1


def test_favorite_refs_lists_identity_only(tmp_path: Path) -> None:
    client, hello, apple = _two_saved_episodes(tmp_path)
    items = client.get("/api/app/favorites/refs").json()["items"]
    assert items == [
        {"kind": "show", "ref": "p05", "color": None},
        {"kind": "topic", "ref": "topic:ai", "color": None},
        {"kind": "episode", "ref": apple, "color": None},
        {"kind": "episode", "ref": hello, "color": "red"},
    ]


def test_favorites_write_rejects_insight_kind(tmp_path: Path) -> None:
    """RFC-121 / #1593: an insight is saved via the highlights path, never as a favorite.

    The route rejects a favorite(insight) write with a 422 so the banned second write path — the
    "same text, two destinations" #1593 closed — cannot be reopened by a stray caller.
    """
    client = _authed_client(tmp_path)
    resp = client.put("/api/app/favorites", json={"kind": "insight", "ref": "ep1#i1", "label": "x"})
    assert resp.status_code == 422
    # refused, not silently accepted
    assert client.get("/api/app/favorites").json() == {"episodes": [], "entities": []}


def test_favorites_requires_auth(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    client = TestClient(app)
    assert client.get("/api/app/favorites").status_code == 401
    assert client.put("/api/app/favorites", json={"kind": "episode", "ref": "x"}).status_code == 401


def test_completed_roundtrip(tmp_path: Path) -> None:
    client = _authed_client(tmp_path)
    assert client.get("/api/app/completed").json() == {"slugs": []}
    assert client.put("/api/app/completed/ep-1").json()["slugs"] == ["ep-1"]
    # idempotent — a set, not a log
    assert client.put("/api/app/completed/ep-1").json()["slugs"] == ["ep-1"]
    assert client.put("/api/app/completed/ep-2").json()["slugs"] == ["ep-1", "ep-2"]
    assert client.delete("/api/app/completed/ep-1").json()["slugs"] == ["ep-2"]
    # clearing what isn't set is a no-op
    assert client.delete("/api/app/completed/ep-1").json()["slugs"] == ["ep-2"]


def test_completed_requires_auth(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    client = TestClient(app)
    assert client.get("/api/app/completed").status_code == 401
    assert client.put("/api/app/completed/x").status_code == 401


def test_interests_requires_auth(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    client = TestClient(app)
    assert client.get("/api/app/interests").status_code == 401
    assert client.put("/api/app/interests", json={"items": ["tc:x"]}).status_code == 401


def test_disabled_user_blocked_on_state_route(tmp_path: Path) -> None:
    from podcast_scraper.server.app_user_store import set_disabled, user_id_for

    client = _authed_client(tmp_path)
    assert client.get("/api/app/queue").status_code == 200  # works while enabled
    set_disabled(tmp_path / "appdata", user_id_for("stub", "s1"), True)
    assert client.get("/api/app/queue").status_code == 401  # disabled → locked out


def test_library_subscribe_list_unsubscribe(tmp_path: Path) -> None:
    client = _authed_client(tmp_path)
    assert client.get("/api/app/library").json()["items"] == []
    added = client.post("/api/app/library", json={"feed_id": "f1", "title": "Show One"})
    assert [i["feed_id"] for i in added.json()["items"]] == ["f1"]
    client.post("/api/app/library", json={"feed_id": "f2"})
    listed = client.get("/api/app/library").json()["items"]
    assert {i["feed_id"] for i in listed} == {"f1", "f2"}
    removed = client.delete("/api/app/library/f1")
    assert [i["feed_id"] for i in removed.json()["items"]] == ["f2"]


# --- "played" means played, however you got there (operator 2026-09-23) --------------------------


def test_finishing_an_episode_makes_it_completed_without_touching_the_menu(tmp_path: Path) -> None:
    """The reported bug, at its source.

    The listener heard the episode to the end; the player reported `finished`. Nothing in the app
    called it played, because `/completed` only ever returned the hand-marked list -- so "Jump back
    in" kept offering it and the catalogue's Played filter matched nothing.
    """
    client = _authed_client(tmp_path)
    assert client.get("/api/app/completed").json() == {"slugs": []}

    client.put("/api/app/playback/ep-heard", json={"position_seconds": 1800.0, "finished": True})

    assert client.get("/api/app/completed").json()["slugs"] == ["ep-heard"]


def test_a_position_save_that_is_not_a_finish_does_not_count_as_played(tmp_path: Path) -> None:
    """The other half: stopping halfway is exactly what "Jump back in" is FOR."""
    client = _authed_client(tmp_path)
    client.put("/api/app/playback/ep-part", json={"position_seconds": 90.0, "finished": False})
    assert client.get("/api/app/completed").json()["slugs"] == []


def test_the_two_sources_merge_without_duplicating(tmp_path: Path) -> None:
    client = _authed_client(tmp_path)
    client.put("/api/app/completed/ep-both")
    client.put("/api/app/playback/ep-both", json={"position_seconds": 10.0, "finished": True})
    client.put("/api/app/playback/ep-heard", json={"position_seconds": 10.0, "finished": True})

    slugs = client.get("/api/app/completed").json()["slugs"]
    assert sorted(slugs) == ["ep-both", "ep-heard"]
    assert len(slugs) == len(set(slugs)), f"same episode listed twice: {slugs}"


def test_mark_unplayed_actually_un_plays_a_FINISHED_episode(tmp_path: Path) -> None:
    """The toggle has to come back.

    With `/completed` merging the finish record, deleting only the manual mark would leave the
    episode played forever: tap "Mark unplayed", get "Mark played" back on the next read. So the
    delete retracts the finish too -- including when there was never a manual mark to remove.
    """
    client = _authed_client(tmp_path)
    client.put("/api/app/playback/ep-heard", json={"position_seconds": 1800.0, "finished": True})
    assert client.get("/api/app/completed").json()["slugs"] == ["ep-heard"]

    assert client.delete("/api/app/completed/ep-heard").json()["slugs"] == []
    assert client.get("/api/app/completed").json()["slugs"] == [], "it came back"

    # The position record is retracted with it, not just the list entry -- otherwise the next
    # client that reads /playback re-derives "finished" and re-plays the whole cycle.
    rec = client.get("/api/app/playback/ep-heard").json()
    assert rec["finished"] is False
    assert rec["position_seconds"] == 1800.0, "unplaying must not lose the resume point"

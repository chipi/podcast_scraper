"""Integration tests for personalized discovery (#1098).

GET /api/app/clusters (interests picker) and GET /api/app/discover (flag-gated ranking):
- flag OFF (default) → recency, identical to the catalog;
- flag ON + signed-in user with interests → significance × interest-affinity re-ranking.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_ranking_telemetry, app_sessions, app_user_state
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = [pytest.mark.integration]


def _write_episode(
    root: Path,
    *,
    stem: str,
    episode_id: str,
    topics: list[tuple[str, str]],
    published: str,
    with_gi: bool = False,
    persons: list[tuple[str, str]] | None = None,
) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": "myfeed", "title": "My Show", "url": "https://pod.example/feed.xml"},
        "episode": {
            "episode_id": episode_id,
            "title": f"Episode {episode_id}",
            "published_date": published,
            "duration_seconds": 1000,
        },
        "summary": {"title": "Sum", "bullets": ["a"]},
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / f"{stem}.txt").write_text("hello", encoding="utf-8")
    nodes = [{"id": tid, "type": "Topic", "properties": {"label": label}} for tid, label in topics]
    nodes += [
        {"id": pid, "type": "Person", "properties": {"name": name}} for pid, name in (persons or [])
    ]
    (root / "metadata" / f"{stem}.kg.json").write_text(
        json.dumps({"episode_id": episode_id, "nodes": nodes}), encoding="utf-8"
    )
    if with_gi:
        gi = {"episode_id": episode_id, "nodes": [], "edges": []}
        (root / "metadata" / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")


def _corpus(root: Path) -> None:
    # epOld is older but about AI; epNew is newer but about Health.
    _write_episode(
        root,
        stem="0001-old",
        episode_id="old",
        topics=[("topic:ai", "AI")],
        published="2024-01-01T00:00:00",
        with_gi=True,
        persons=[("person:jane", "Jane")],
    )
    _write_episode(
        root,
        stem="0002-new",
        episode_id="new",
        topics=[("topic:health", "Health")],
        published="2024-06-01T00:00:00",
    )
    (root / "search").mkdir(parents=True, exist_ok=True)
    payload = {
        "clusters": [
            {
                "graph_compound_parent_id": "tc:ai",
                "canonical_label": "AI",
                "member_count": 3,
                "members": [{"topic_id": "topic:ai", "label": "AI"}],
            },
            {
                "graph_compound_parent_id": "tc:health",
                "canonical_label": "Health",
                "member_count": 1,
                "members": [{"topic_id": "topic:health", "label": "Health"}],
            },
        ]
    }
    (root / "search" / "topic_clusters.json").write_text(json.dumps(payload), encoding="utf-8")


def _client(root: Path, *, personalized: bool, derived: bool = False) -> TestClient:
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = root / "appdata"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    app.state.personalized_ranking = personalized
    app.state.derived_interests = derived
    return TestClient(app)


def _sign_in_heard(client: TestClient, root: Path, heard_episode_ids: list[str]) -> None:
    """Sign in a user with NO explicit interests, but who has *heard* the given episodes."""
    from podcast_scraper.server.app_slugs import slug_for_row
    from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

    _sign_in(client, root, [])  # same user (subject 's1'), no explicit interests
    data_dir = root / "appdata"
    user = get_or_create_user(data_dir, provider="stub", subject="s1", email="j@x.com", name="J")
    slugs = {r.episode_id: slug_for_row(r) for r in build_catalog_rows_cumulative(root)}
    for eid in heard_episode_ids:
        app_user_state.set_playback(data_dir, user.user_id, slugs[eid], 400.0, 1)  # 40% → heard


def _sign_in(client: TestClient, root: Path, interests: list[str]) -> None:
    data_dir = root / "appdata"
    user = get_or_create_user(data_dir, provider="stub", subject="s1", email="j@x.com", name="J")
    app_user_state.set_interests(data_dir, user.user_id, interests)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)


def test_clusters_endpoint_returns_top_by_prevalence(tmp_path: Path) -> None:
    _corpus(tmp_path)
    body = (
        _client(tmp_path, personalized=False).get("/api/app/clusters", params={"limit": 5}).json()
    )
    ids = [c["id"] for c in body["items"]]
    assert ids == ["tc:ai", "tc:health"]  # ranked by member_count desc
    assert body["items"][0] == {"id": "tc:ai", "label": "AI", "size": 3}


def test_discover_recency_when_flag_off(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=False)
    _sign_in(client, tmp_path, ["tc:ai"])  # interests present but flag off → still recency
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    assert titles == ["Episode new", "Episode old"]  # newest-first


def test_discover_personalizes_when_flag_on_and_interests_set(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in(client, tmp_path, ["tc:ai"])
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    # epOld (about AI, the user's interest) now leads despite being older.
    assert titles == ["Episode old", "Episode new"]


def test_discover_personalizes_by_followed_topic(tmp_path: Path) -> None:
    # Following the topic itself (topic: token) — not its cluster — also re-ranks.
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in(client, tmp_path, ["topic:ai"])
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    assert titles == ["Episode old", "Episode new"]


def test_discover_personalizes_by_followed_person(tmp_path: Path) -> None:
    # Following a person (person: token) boosts episodes that feature them.
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in(client, tmp_path, ["person:jane"])  # Jane appears only in (older) epOld
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    assert titles == ["Episode old", "Episode new"]


def test_discover_derives_interests_from_heard_episodes(tmp_path: Path) -> None:
    # #1139: NO explicit interests, but the user has *heard* the (older) AI episode.
    # Its entities (topic:ai / person:jane) become derived interests → epOld is lifted
    # above the newer Health episode, personalizing from behaviour alone.
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True, derived=True)
    _sign_in_heard(client, tmp_path, ["old"])
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    assert titles == ["Episode old", "Episode new"]


def test_discover_derived_off_by_default_stays_recency(tmp_path: Path) -> None:
    # Personalization on, but the derived-interests flag is OFF and there are no explicit
    # interests → recency, unchanged. Guards the new signal behind its own toggle.
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)  # derived defaults off
    _sign_in_heard(client, tmp_path, ["old"])
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    assert titles == ["Episode new", "Episode old"]


def test_discover_recency_when_flag_on_but_anonymous(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)  # no sign-in → no interests → recency
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    assert titles == ["Episode new", "Episode old"]


def _user_id(tmp_path: Path) -> str:
    data_dir = tmp_path / "appdata"
    return get_or_create_user(
        data_dir, provider="stub", subject="s1", email="j@x.com", name="J"
    ).user_id


def test_discover_records_impressions_for_signed_in_user(tmp_path: Path) -> None:
    # #11 telemetry: a signed-in discover call logs the shown slugs (rank order) + the variant.
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in(client, tmp_path, ["topic:ai"])
    resp = client.get("/api/app/discover", params={"limit": 5})
    assert resp.status_code == 200
    shown = [it["slug"] for it in resp.json()["items"]]
    events = app_ranking_telemetry.read_events(tmp_path / "appdata", _user_id(tmp_path))
    imps = [e for e in events if e["kind"] == "impression"]
    assert imps, "expected an impression event"
    assert imps[-1]["shown"] == shown
    assert imps[-1]["variant"] == "personalized"


def test_discover_click_records_event(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in(client, tmp_path, ["topic:ai"])
    resp = client.post("/api/app/discover/click", json={"slug": "some-slug", "position": 2})
    assert resp.status_code == 204
    clicks = [
        e
        for e in app_ranking_telemetry.read_events(tmp_path / "appdata", _user_id(tmp_path))
        if e["kind"] == "click"
    ]
    assert clicks and clicks[-1]["slug"] == "some-slug" and clicks[-1]["position"] == 2


def test_discover_click_signed_out_is_noop_204(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)  # not signed in
    resp = client.post("/api/app/discover/click", json={"slug": "x", "position": 0})
    assert resp.status_code == 204


def _sign_in_admin(client: TestClient, root: Path) -> None:
    from podcast_scraper.server.app_user_store import set_role

    data_dir = root / "appdata"
    user = get_or_create_user(data_dir, provider="stub", subject="s1", email="j@x.com", name="J")
    set_role(data_dir, user.user_id, "admin")
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)


def test_ranking_config_get_requires_admin(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in(client, tmp_path, [])  # signed in but not admin
    assert client.get("/api/app/ranking-config").status_code == 403


def test_ranking_config_get_returns_signals_for_admin(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in_admin(client, tmp_path)
    resp = client.get("/api/app/ranking-config")
    assert resp.status_code == 200
    names = [s["name"] for s in resp.json()["signals"]]
    assert "significance" in names and "trend_velocity" in names


def test_ranking_config_put_persists_and_reads_back(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in_admin(client, tmp_path)
    put = client.put(
        "/api/app/ranking-config",
        json={"signals": [{"name": "trend_velocity", "enabled": True, "weight": 5.0}]},
    )
    assert put.status_code == 200
    got = client.get("/api/app/ranking-config").json()
    trend = next(s for s in got["signals"] if s["name"] == "trend_velocity")
    assert trend["enabled"] is True and trend["weight"] == 5.0
    # untouched signals survive the merge
    assert any(s["name"] == "significance" for s in got["signals"])

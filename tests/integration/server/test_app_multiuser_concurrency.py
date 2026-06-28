"""Multi-user robustness — parallel logins, per-user isolation, no lost writes, no deadlock.

Proves the operator's concern: when several users sign in at once, OAuth resolves **distinct**
identities, each user's per-user files stay **private** (nobody overwrites anybody), everybody reads
back exactly their own content, and concurrent writes neither lose updates nor deadlock.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.search.corpus_search import CorpusSearchOutcome
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_oauth import MockOAuthProvider

pytestmark = [pytest.mark.integration]


def _write_episode(
    root: Path,
    *,
    stem: str,
    eid: str,
    topics: list[tuple[str, str]],
    pub: str,
    with_gi: bool = False,
) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": "f", "title": "Show", "url": "https://f.ex/f.xml"},
        "episode": {
            "episode_id": eid,
            "title": f"Episode {eid}",
            "published_date": pub,
            "duration_seconds": 1000,
        },
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    nodes = [{"id": tid, "type": "Topic", "properties": {"label": la}} for tid, la in topics]
    (root / "metadata" / f"{stem}.kg.json").write_text(
        json.dumps({"episode_id": eid, "nodes": nodes}), encoding="utf-8"
    )
    if with_gi:
        (root / "metadata" / f"{stem}.gi.json").write_text(
            json.dumps({"episode_id": eid, "nodes": [], "edges": []}), encoding="utf-8"
        )


def _personalized_corpus(root: Path) -> None:
    # ep1 (AI, +GI depth) is OLDER; ep2 (cooking) is NEWER — so recency puts ep2 first, but a user
    # who follows the AI cluster sees ep1 boosted. Two distinct clusters → two distinct preferences.
    _write_episode(
        root,
        stem="0001",
        eid="ep1",
        topics=[("topic:ai", "AI"), ("topic:ml", "ML")],
        pub="2024-01-01T00:00:00",
        with_gi=True,
    )
    _write_episode(
        root,
        stem="0002",
        eid="ep2",
        topics=[("topic:cooking", "Cooking")],
        pub="2024-06-01T00:00:00",
    )
    (root / "search").mkdir(parents=True, exist_ok=True)
    (root / "search" / "topic_clusters.json").write_text(
        json.dumps(
            {
                "clusters": [
                    {
                        "graph_compound_parent_id": "tc:ai",
                        "canonical_label": "AI",
                        "member_count": 2,
                        "members": [
                            {"topic_id": "topic:ai", "label": "AI"},
                            {"topic_id": "topic:ml", "label": "ML"},
                        ],
                    },
                    {
                        "graph_compound_parent_id": "tc:cooking",
                        "canonical_label": "Cooking",
                        "member_count": 1,
                        "members": [{"topic_id": "topic:cooking", "label": "Cooking"}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def _slug(root: Path, eid: str) -> str:
    from podcast_scraper.server.app_slugs import slug_for_row
    from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

    for row in build_catalog_rows_cumulative(root):
        if row.episode_id == eid:
            return slug_for_row(row)
    raise AssertionError(eid)


def _mock_app(tmp_path: Path):
    """An app wired with the mock OAuth provider + open policy + a per-user data dir."""
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    app.state.oauth_provider = MockOAuthProvider()
    return app


def _login(app, who: str) -> TestClient:
    """Drive the real mock-OAuth flow as identity ``who`` (distinct user per hint); return a client.

    The flow redirects login → callback (sets the session cookie) → ``/`` (the SPA index, absent in
    this static-less test app, so the final hop 404s — the session is already established by then).
    """
    client = TestClient(app)
    client.get("/api/app/auth/login", params={"as": who}, follow_redirects=True)
    me = client.get("/api/app/me")
    assert me.status_code == 200, f"login as {who} failed: {me.text}"
    return client


def test_oauth_returns_a_distinct_user_per_hint(tmp_path: Path) -> None:
    app = _mock_app(tmp_path)
    seen: dict[str, str] = {}
    for who in ("alice", "bob", "carol"):
        me = _login(app, who).get("/api/app/me").json()
        assert me["email"] == f"{who}@e2e.local"
        seen[who] = me["user_id"]
    assert len(set(seen.values())) == 3  # three distinct user ids — independent lifecycles


def test_concurrent_users_are_isolated_with_no_crosstalk(tmp_path: Path) -> None:
    app = _mock_app(tmp_path)
    n = 16

    def work(i: int) -> tuple[int, list, list, list, dict]:
        client = _login(app, f"u{i}")
        # Each user writes only their OWN content across several profile files.
        client.post(
            "/api/app/highlights",
            json={"episode_slug": f"ep{i}", "kind": "moment", "start_ms": i},
        )
        client.put(f"/api/app/playback/ep{i}", json={"position_seconds": float(i)})
        client.put("/api/app/queue", json={"items": [f"q{i}"]})
        client.put("/api/app/interests", json={"items": [f"topic:t{i}"]})
        highlights = client.get("/api/app/highlights").json()["items"]
        queue = client.get("/api/app/queue").json()["items"]
        interests = client.get("/api/app/interests").json()["items"]
        playback = client.get(f"/api/app/playback/ep{i}").json()
        return i, highlights, queue, interests, playback

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(work, range(n)))
    elapsed = time.monotonic() - start

    # Completes quickly → no deadlock (a lock bug would hang to the pytest timeout).
    assert elapsed < 30
    for i, highlights, queue, interests, playback in results:
        # Each user sees EXACTLY their own writes — no other user's data leaked in.
        assert [h["episode_slug"] for h in highlights] == [f"ep{i}"]
        assert queue == [f"q{i}"]
        assert interests == [f"topic:t{i}"]
        assert playback["position_seconds"] == float(i)


def test_same_user_concurrent_writes_do_not_lose_updates(tmp_path: Path) -> None:
    # Rapid concurrent captures by ONE user (e.g. double-taps) must all persist — the per-user
    # highlight file is a read-modify-write, so this guards against lost updates + tmp-file clobber.
    app = _mock_app(tmp_path)
    client = _login(app, "solo")
    m = 25

    def add(i: int) -> int:
        r = client.post(
            "/api/app/highlights",
            json={"episode_slug": "ep", "kind": "moment", "start_ms": i},
        )
        return int(r.status_code)

    with ThreadPoolExecutor(max_workers=m) as pool:
        codes = list(pool.map(add, range(m)))
    assert all(c == 201 for c in codes)
    items = client.get("/api/app/highlights").json()["items"]
    assert len(items) == m  # every concurrent add survived — no lost updates
    assert {h["start_ms"] for h in items} == set(range(m))


def test_per_user_personalization_search_and_library_are_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _mock_app(tmp_path)
    _personalized_corpus(tmp_path)
    app.state.personalized_ranking = True
    ep1, ep2 = _slug(tmp_path, "ep1"), _slug(tmp_path, "ep2")
    alice, bob = _login(app, "alice"), _login(app, "bob")

    # (1) Different interests → different suggestions: each user's discover feed leads with the
    # episode matching THEIR followed cluster.
    alice.put("/api/app/interests", json={"items": ["tc:ai"]})
    bob.put("/api/app/interests", json={"items": ["tc:cooking"]})
    a_top = alice.get("/api/app/discover").json()["items"][0]["title"]
    b_top = bob.get("/api/app/discover").json()["items"][0]["title"]
    assert a_top == "Episode ep1" and b_top == "Episode ep2" and a_top != b_top

    # (2) Each plays a different episode → unique "now playing"; neither sees the other's position.
    alice.put(f"/api/app/playback/{ep1}", json={"position_seconds": 120.0})
    bob.put(f"/api/app/playback/{ep2}", json={"position_seconds": 300.0})
    assert alice.get("/api/app/playback").json()["items"][0]["slug"] == ep1
    assert bob.get("/api/app/playback").json()["items"][0]["slug"] == ep2
    assert alice.get(f"/api/app/playback/{ep2}").json()["position_seconds"] == 0.0  # never played

    # (3) Different library / saved things → each sees only their own favourites.
    alice.put("/api/app/favorites", json={"kind": "episode", "ref": ep1, "label": "A"})
    bob.put("/api/app/favorites", json={"kind": "episode", "ref": ep2, "label": "B"})
    assert [e["slug"] for e in alice.get("/api/app/favorites").json()["episodes"]] == [ep1]
    assert [e["slug"] for e in bob.get("/api/app/favorites").json()["episodes"]] == [ep2]

    # (4) Different searches (scope=mine) → each gets results only from their own corpus. The inner
    # search returns BOTH episodes; the heard∪captured filter narrows to each user's set.
    def both(output_dir: Path, query: str, **kw: Any) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(
            results=[
                {
                    "doc_id": "i1",
                    "score": 0.9,
                    "text": "a",
                    "metadata": {"doc_type": "insight", "feed_id": "f", "episode_id": "ep1"},
                },
                {
                    "doc_id": "i2",
                    "score": 0.8,
                    "text": "b",
                    "metadata": {"doc_type": "insight", "feed_id": "f", "episode_id": "ep2"},
                },
            ],
            lift_stats={"transcript_hits_returned": 0, "lift_applied": 0},
        )

    monkeypatch.setattr("podcast_scraper.search.capability.run_corpus_search", both)
    a_hits = alice.get("/api/app/search", params={"q": "x", "scope": "mine"}).json()["results"]
    b_hits = bob.get("/api/app/search", params={"q": "x", "scope": "mine"}).json()["results"]
    assert [r["doc_id"] for r in a_hits] == ["i1"]  # alice recalls only her captured episode
    assert [r["doc_id"] for r in b_hits] == ["i2"]  # bob recalls only his

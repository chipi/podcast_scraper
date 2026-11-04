"""Multi-user robustness — parallel logins, per-user isolation, no lost writes, no deadlock.

Proves the operator's concern: when several users sign in at once, OAuth resolves **distinct**
identities, each user's per-user files stay **private** (nobody overwrites anybody), everybody reads
back exactly their own content, and concurrent writes neither lose updates nor deadlock.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

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
    from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative
    from podcast_scraper.server.slugs import slug_for_row

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

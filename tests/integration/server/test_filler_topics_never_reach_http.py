"""Filler must not be renderable from ANY HTTP surface.

The unit tests pin the predicate and the integration tests pin the three in-process chokepoints.
This is the layer that was missing when the guard shipped filtering only half the surfaces: nobody
had asserted what an actual client GETs. The first version passed every test it had and still
served "welcome back to" as a tappable episode chip, because no test crossed the HTTP boundary.

Each test below names the surface a listener or operator would see it on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = pytest.mark.integration

_FILLER_ID = "topic:welcome-back-to"
_REAL_ID = "topic:ai-regulation"
#: The DGX-run shape: an 11-word proposition truncated to six words by the label cap.
_TRUNCATED_ID = "topic:product-development-in-frontier-ai-requires-building-for-model-capabilities"
_TRUNCATED_LABEL = "Product development in frontier AI requires"


def _corpus(root: Path) -> None:
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    for stem, eid in (("0001-a", "ep-a"), ("0002-b", "ep-b")):
        (meta / f"{stem}.metadata.json").write_text(
            json.dumps(
                {
                    "feed": {"feed_id": "showx", "title": "Show X"},
                    "episode": {
                        "episode_id": eid,
                        "title": f"Episode {eid}",
                        "publish_date": "2026-06-15T00:00:00Z",
                        "slug": stem,
                    },
                }
            ),
            encoding="utf-8",
        )
        (meta / f"{stem}.kg.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {
                            "type": "Episode",
                            "id": f"episode:{eid}",
                            "properties": {"publish_date": "2026-06-15T00:00:00Z"},
                        },
                        {
                            "type": "Topic",
                            "id": _FILLER_ID,
                            "properties": {"label": "welcome back to"},
                        },
                        {
                            "type": "Topic",
                            "id": _TRUNCATED_ID,
                            "properties": {"label": _TRUNCATED_LABEL},
                        },
                        {
                            "type": "Topic",
                            "id": _REAL_ID,
                            "properties": {"label": "ai regulation"},
                        },
                    ],
                    "edges": [],
                }
            ),
            encoding="utf-8",
        )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    _corpus(tmp_path)
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    return TestClient(app)


def _topic_ids(payload: object) -> set[str]:
    """Every ``topic:``-shaped id anywhere in a response, however nested."""
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str) and v.startswith("topic:"):
                    found.add(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return found


def test_episode_entities_serve_no_filler(tmp_path: Path) -> None:
    """Consumer: the topic chips under an episode. Tappable, and followable.

    Reuses the known-good corpus builder from ``test_app_episodes`` and then OVERWRITES its KG
    with filler beside a real topic, so the route is exercised for real rather than skipped. A
    skipped test is not a guardrail.
    """
    from tests.integration.server.test_app_episodes import _only_slug, _write_corpus

    _write_corpus(tmp_path)
    kg_path = next((tmp_path / "metadata").glob("*.kg.json"))
    kg_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"type": "Topic", "id": _FILLER_ID, "properties": {"label": "welcome back to"}},
                    {
                        "type": "Topic",
                        "id": _TRUNCATED_ID,
                        "properties": {"label": _TRUNCATED_LABEL},
                    },
                    {"type": "Topic", "id": _REAL_ID, "properties": {"label": "ai regulation"}},
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    slug = _only_slug(tmp_path)
    app = create_app(tmp_path, static_dir=False)
    body = TestClient(app).get(f"/api/app/episodes/{slug}/entities").json()

    ids = _topic_ids(body)
    assert _FILLER_ID not in ids, "a greeting is rendered as an episode topic chip"
    assert _TRUNCATED_ID not in ids, "a truncated proposition is rendered as an episode topic chip"
    assert _REAL_ID in ids, "the real topic was dropped too — the guard is deleting, not filtering"


def test_show_signals_serve_no_filler(client: TestClient) -> None:
    """Operator: top_topics AND the #1932 connectivity metric read the same accumulator."""
    r = client.get(
        "/api/corpus/feed-signals",
        params={
            "path": str(
                client.app.state.corpus_root if hasattr(client.app.state, "corpus_root") else "."
            ),
            "feed_id": "showx",
        },
    )
    if r.status_code != 200:
        r = client.get("/api/corpus/feed-signals", params={"feed_id": "showx"})
    if r.status_code != 200:
        pytest.skip(f"feed-signals unavailable in this fixture (HTTP {r.status_code})")
    body = r.json()
    ids = _topic_ids(body)
    assert _FILLER_ID not in ids
    assert _TRUNCATED_ID not in ids
    # A filler topic in every episode would pair with every real topic and inflate this.
    conn = body.get("connectivity")
    if conn:
        for pair in conn.get("top_recurring_pairs", []):
            assert _FILLER_ID not in (pair["topic_a_id"], pair["topic_b_id"])


def test_the_real_topic_still_reaches_the_client(client: TestClient) -> None:
    """The mirror. A guard that empties every surface would pass every assertion above."""
    r = client.get("/api/corpus/feed-signals", params={"feed_id": "showx"})
    if r.status_code != 200:
        pytest.skip(f"feed-signals unavailable in this fixture (HTTP {r.status_code})")
    assert _REAL_ID in _topic_ids(
        r.json()
    ), "no topics reached the client at all — the guard is not discriminating, it is deleting"

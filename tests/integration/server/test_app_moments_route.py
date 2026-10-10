"""GET /api/app/episodes/{slug}/moments (operator 2026-10-10) over a small on-disk corpus.

The route wires ``app_moments`` to the corpus: the GI artifact for which moments, the raw
transcript segments for what plays, the catalog duration for how many.
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
from podcast_scraper.server.app_moments import MomentsConfig
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = [pytest.mark.integration]


def _episode(root: Path, stem: str, episode_id: str, *, insights: int, segments: bool) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": "f", "title": "Show", "url": "https://pod.example/feed.xml"},
        "episode": {
            "episode_id": episode_id,
            "title": f"Episode {episode_id}",
            "published_date": "2024-03-10T00:00:00",
            "duration_seconds": 1800,
        },
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / f"{stem}.txt").write_text("words", encoding="utf-8")
    if segments:
        segs = [
            {"start": t, "end": t + 5.0, "text": f"Sentence at {int(t)}."}
            for t in range(0, 1800, 5)
        ]
        (root / "transcripts" / f"{stem}.segments.json").write_text(
            json.dumps(segs), encoding="utf-8"
        )
    if not insights:
        return
    nodes: list[dict] = [{"id": "person:ann", "type": "Person", "properties": {"name": "Ann"}}]
    edges: list[dict] = []
    for n in range(insights):
        iid, qid = f"insight:{n}", f"quote:{n}"
        start = 60 + n * 290  # spread across the half hour, ~5 min apart
        nodes.append(
            {
                "id": iid,
                "type": "Insight",
                "properties": {
                    "text": f"Point {n}",
                    "grounded": True,
                    "routing_tag": "surface",
                    "insight_type": "claim",
                },
            }
        )
        nodes.append(
            {
                "id": qid,
                "type": "Quote",
                "properties": {
                    "text": f"quote {n}",
                    "timestamp_start_ms": (start + 2) * 1000,
                    "timestamp_end_ms": (start + 6) * 1000,
                },
            }
        )
        edges += [
            {"type": "SUPPORTED_BY", "from": iid, "to": qid},
            {"type": "SPOKEN_BY", "from": qid, "to": "person:ann"},
        ]
    (root / "metadata" / f"{stem}.gi.json").write_text(
        json.dumps({"episode_id": episode_id, "nodes": nodes, "edges": edges}), encoding="utf-8"
    )


def _client(root: Path, *, signed_in: bool = True) -> TestClient:
    data_dir = root / "_appdata"
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    client = TestClient(app)
    if signed_in:
        user = get_or_create_user(data_dir, provider="stub", subject="s", email="u@x.com", name="U")
        token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
        client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def _slug(root: Path, episode_id: str) -> str:
    from podcast_scraper.server.app_slugs import slug_for_row
    from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

    for row in build_catalog_rows_cumulative(root):
        if row.episode_id == episode_id:
            return slug_for_row(row)
    raise AssertionError(episode_id)


def test_moments_in_timeline_order_with_segment_clips(tmp_path: Path) -> None:
    _episode(tmp_path, "0001-a", "ep1", insights=6, segments=True)
    body = _client(tmp_path).get(f"/api/app/episodes/{_slug(tmp_path, 'ep1')}/moments").json()
    moments = body["moments"]
    # 30 min → 5 moments; six candidates ~5 min apart.
    assert len(moments) == 5
    starts = [m["start_ms"] for m in moments]
    assert starts == sorted(starts)
    first = moments[0]
    # The quote starts 2 s into a segment → the clip starts at the segment, runs to >= 12 s.
    assert first["start_ms"] == 60_000
    assert first["end_ms"] - first["start_ms"] >= 12_000
    assert first["clip_text"].startswith("Sentence at 60.")
    assert first["speaker"] == "Ann"
    assert all(m["end_ms"] - m["start_ms"] <= 30_000 for m in moments)
    assert body["total_seconds"] == pytest.approx(
        sum(m["end_ms"] - m["start_ms"] for m in moments) / 1000, abs=0.1
    )


def test_config_from_app_state_changes_the_count(tmp_path: Path) -> None:
    _episode(tmp_path, "0001-a", "ep1", insights=6, segments=True)
    client = _client(tmp_path)
    client.app.state.moments_config = MomentsConfig(min_count=2, max_count=2)  # type: ignore[attr-defined]
    body = client.get(f"/api/app/episodes/{_slug(tmp_path, 'ep1')}/moments").json()
    assert len(body["moments"]) == 2


def test_without_segments_the_clip_is_the_quote(tmp_path: Path) -> None:
    _episode(tmp_path, "0001-a", "ep1", insights=6, segments=False)
    body = _client(tmp_path).get(f"/api/app/episodes/{_slug(tmp_path, 'ep1')}/moments").json()
    assert body["moments"][0]["clip_text"] == "quote 0"


def test_an_episode_without_insights_has_no_moments(tmp_path: Path) -> None:
    _episode(tmp_path, "0001-a", "ep1", insights=0, segments=True)
    body = _client(tmp_path).get(f"/api/app/episodes/{_slug(tmp_path, 'ep1')}/moments").json()
    assert body == {"episode_slug": _slug(tmp_path, "ep1"), "moments": [], "total_seconds": 0.0}


def test_signed_out_is_refused(tmp_path: Path) -> None:
    _episode(tmp_path, "0001-a", "ep1", insights=6, segments=True)
    resp = _client(tmp_path, signed_in=False).get(
        f"/api/app/episodes/{_slug(tmp_path, 'ep1')}/moments"
    )
    assert resp.status_code == 401

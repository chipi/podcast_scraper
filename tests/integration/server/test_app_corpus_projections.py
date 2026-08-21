"""Integration tests for the lean corpus projection routes (#perf).

``GET /api/app/corpus/trending-topics`` and ``GET /api/app/corpus/entity-signals`` serve the
top-N / per-entity slice the Home rail and entity card render, instead of the whole ~25 MB
corpus-enrichment payload. Exercised through ``TestClient`` over on-disk envelopes so the route
wrappers (filtering, projection, 422s, the no-enricher path) run for real.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


def _client(root: Path) -> TestClient:
    return TestClient(create_app(root, static_dir=False))


def _env(root: Path, enricher_id: str, data: object, *, status: str = "ok") -> None:
    """Write a corpus-scope enrichment envelope to ``<root>/enrichments/<enricher_id>.json``."""
    out = root / "enrichments"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{enricher_id}.json").write_text(
        json.dumps(
            {"enricher_id": enricher_id, "schema_version": 1, "status": status, "data": data}
        ),
        encoding="utf-8",
    )


def _velocity_topic(
    topic_id: str,
    *,
    label: str,
    velocity: float,
    total: int,
    monthly: dict[str, int] | None = None,
) -> dict[str, object]:
    """A temporal_velocity row carrying the bloat fields (weekly_*) the projection must drop."""
    return {
        "topic_id": topic_id,
        "topic_label": label,
        "velocity_last_over_6mo": velocity,
        "total": total,
        "monthly_counts": monthly or {"2026-03": total},
        # Fields the rail never reads — the whole reason the endpoint exists:
        "weekly_counts": {str(w): 1 for w in range(104)},
        "weekly_velocity": {str(w): 0.1 for w in range(104)},
        "ewma": {"2026-03": 1.2},
    }


# --------------------------------------------------------------------------- #
# GET /corpus/trending-topics
# --------------------------------------------------------------------------- #


def test_trending_topics_no_enricher_renders_nothing(tmp_path: Path) -> None:
    # No enrichments dir at all → the client must be able to render nothing (not the quiet state).
    body = _client(tmp_path).get("/api/app/corpus/trending-topics").json()
    assert body == {
        "has_velocity_data": False,
        "window_months": [],
        "topics": [],
        "theme_clusters": [],
    }


def test_trending_topics_returns_top_rising_sorted_by_velocity(tmp_path: Path) -> None:
    _env(
        tmp_path,
        "temporal_velocity",
        {
            "window_months": ["2026-01", "2026-02", "2026-03"],
            "topics": [
                _velocity_topic("topic:ai", label="AI", velocity=2.0, total=10),
                _velocity_topic("topic:policy", label="Foreign Policy", velocity=4.0, total=3),
                # not rising (velocity < 1.5) — excluded
                _velocity_topic("topic:steady", label="Steady", velocity=1.0, total=20),
                # below the sparse floor (total < 3) — excluded
                _velocity_topic("topic:noise", label="Noise", velocity=5.0, total=1),
            ],
        },
    )
    body = _client(tmp_path).get("/api/app/corpus/trending-topics").json()
    assert body["has_velocity_data"] is True
    assert body["window_months"] == ["2026-01", "2026-02", "2026-03"]
    ids = [t["topic_id"] for t in body["topics"]]
    # policy (4x) before ai (2x); steady + noise excluded.
    assert ids == ["topic:policy", "topic:ai"]
    assert body["topics"][0]["velocity_last_over_6mo"] == 4.0
    assert body["topics"][0]["total"] == 3


def test_trending_topics_projects_away_the_weekly_bloat(tmp_path: Path) -> None:
    # The endpoint's whole point: monthly_counts survive for the sparkline; weekly_* do not.
    _env(
        tmp_path,
        "temporal_velocity",
        {
            "window_months": ["2026-03"],
            "topics": [_velocity_topic("topic:ai", label="AI", velocity=2.0, total=10)],
        },
    )
    row = _client(tmp_path).get("/api/app/corpus/trending-topics").json()["topics"][0]
    assert row["monthly_counts"] == {"2026-03": 10}
    assert "weekly_counts" not in row
    assert "weekly_velocity" not in row
    assert "ewma" not in row


def test_trending_topics_quiet_state_when_nothing_rising(tmp_path: Path) -> None:
    # Enricher ran but nothing clears the bar → has_velocity_data True, topics empty (quiet state),
    # distinct from the no-enricher case above.
    _env(
        tmp_path,
        "temporal_velocity",
        {
            "window_months": ["2026-03"],
            "topics": [_velocity_topic("topic:flat", label="Flat", velocity=0.9, total=50)],
        },
    )
    body = _client(tmp_path).get("/api/app/corpus/trending-topics").json()
    assert body["has_velocity_data"] is True
    assert body["topics"] == []


def test_trending_topics_respects_limit_and_threshold_params(tmp_path: Path) -> None:
    _env(
        tmp_path,
        "temporal_velocity",
        {
            "window_months": ["2026-03"],
            "topics": [
                _velocity_topic(f"topic:t{i}", label=f"T{i}", velocity=2.0 + i, total=5)
                for i in range(5)
            ],
        },
    )
    client = _client(tmp_path)
    limited = client.get("/api/app/corpus/trending-topics", params={"limit": 2}).json()
    assert [t["topic_id"] for t in limited["topics"]] == ["topic:t4", "topic:t3"]
    # A stricter bar (velocity ≥ 5.5) drops everything but the top row (t4 = 6.0).
    strict = client.get("/api/app/corpus/trending-topics", params={"min_velocity": 5.5}).json()
    assert [t["topic_id"] for t in strict["topics"]] == ["topic:t4"]


def test_trending_topics_includes_theme_clusters(tmp_path: Path) -> None:
    _env(
        tmp_path,
        "temporal_velocity",
        {
            "window_months": ["2026-03"],
            "topics": [_velocity_topic("topic:ai", label="AI", velocity=2.0, total=5)],
        },
    )
    _env(
        tmp_path,
        "topic_theme_clusters",
        {
            "clusters": [
                {
                    "graph_compound_parent_id": "tc:ai",
                    "canonical_label": "Artificial Intelligence",
                    "members": [{"topic_id": "topic:ai"}, {"topic_id": "topic:ml"}],
                }
            ]
        },
    )
    body = _client(tmp_path).get("/api/app/corpus/trending-topics").json()
    assert len(body["theme_clusters"]) == 1
    cluster = body["theme_clusters"][0]
    assert cluster["canonical_label"] == "Artificial Intelligence"
    assert {m["topic_id"] for m in cluster["members"]} == {"topic:ai", "topic:ml"}


def test_trending_topics_ignores_non_ok_envelope(tmp_path: Path) -> None:
    _env(
        tmp_path,
        "temporal_velocity",
        {
            "window_months": ["2026-03"],
            "topics": [_velocity_topic("topic:ai", label="AI", velocity=9.0, total=9)],
        },
        status="error",
    )
    body = _client(tmp_path).get("/api/app/corpus/trending-topics").json()
    assert body["has_velocity_data"] is False
    assert body["topics"] == []


# --------------------------------------------------------------------------- #
# GET /corpus/entity-signals
# --------------------------------------------------------------------------- #


def _person_corpus(root: Path) -> None:
    _env(
        root,
        "guest_coappearance",
        {
            "pairs": [
                {
                    "person_a_id": "person:jane",
                    "person_b_id": "person:bob",
                    "person_a_name": "Jane",
                    "person_b_name": "Bob",
                    "episode_count": 3,
                },
                {
                    "person_a_id": "person:carol",
                    "person_b_id": "person:jane",
                    "person_a_name": "Carol",
                    "person_b_name": "Jane",
                    "episode_count": 1,
                },
                {
                    "person_a_id": "person:carol",
                    "person_b_id": "person:bob",
                    "person_a_name": "Carol",
                    "person_b_name": "Bob",
                    "episode_count": 9,
                },
            ]
        },
    )
    _env(
        root,
        "grounding_rate",
        {
            "persons": [
                {
                    "person_id": "person:jane",
                    "person_name": "Jane",
                    "total_insights": 10,
                    "grounded_insights": 7,
                    "rate": 0.7,
                },
                {
                    "person_id": "person:bob",
                    "person_name": "Bob",
                    "total_insights": 4,
                    "grounded_insights": 1,
                    "rate": 0.25,
                },
            ]
        },
    )


def test_entity_signals_person_filters_to_the_focused_person(tmp_path: Path) -> None:
    _person_corpus(tmp_path)
    body = (
        _client(tmp_path)
        .get("/api/app/corpus/entity-signals", params={"kind": "person", "id": "person:jane"})
        .json()
    )
    signals = body["signals"]
    # Only pairs touching Jane survive — the carol↔bob pair is dropped.
    pairs = signals["guest_coappearance"]["pairs"]
    assert len(pairs) == 2
    for p in pairs:
        assert "person:jane" in (p["person_a_id"], p["person_b_id"])
    # Only Jane's grounding row survives.
    assert [r["person_id"] for r in signals["grounding_rate"]["persons"]] == ["person:jane"]


def test_entity_signals_person_empty_when_no_rows_touch_entity(tmp_path: Path) -> None:
    _person_corpus(tmp_path)
    body = (
        _client(tmp_path)
        .get("/api/app/corpus/entity-signals", params={"kind": "person", "id": "person:nobody"})
        .json()
    )
    assert body["signals"] == {}


def test_entity_signals_topic_filters_velocity_and_cooccurrence(tmp_path: Path) -> None:
    _env(
        tmp_path,
        "temporal_velocity",
        {
            "window_months": ["2026-03"],
            "topics": [
                _velocity_topic("topic:ai", label="AI", velocity=2.0, total=10),
                _velocity_topic("topic:ml", label="ML", velocity=1.0, total=5),
            ],
        },
    )
    _env(
        tmp_path,
        "topic_cooccurrence_corpus",
        {
            "pairs": [
                {
                    "topic_a_id": "topic:ai",
                    "topic_b_id": "topic:ml",
                    "topic_a_label": "AI",
                    "topic_b_label": "ML",
                    "episode_count": 4,
                    "lift": 2.0,
                },
                {
                    "topic_a_id": "topic:policy",
                    "topic_b_id": "topic:econ",
                    "episode_count": 5,
                    "lift": 3.0,
                },
            ]
        },
    )
    body = (
        _client(tmp_path)
        .get("/api/app/corpus/entity-signals", params={"kind": "topic", "id": "topic:ai"})
        .json()
    )
    signals = body["signals"]
    # Only the AI velocity row.
    assert [t["topic_id"] for t in signals["temporal_velocity"]["topics"]] == ["topic:ai"]
    # Only the pair touching AI.
    pairs = signals["topic_cooccurrence_corpus"]["pairs"]
    assert len(pairs) == 1
    assert "topic:ai" in (pairs[0]["topic_a_id"], pairs[0]["topic_b_id"])


def test_entity_signals_normalizes_graph_id_prefix(tmp_path: Path) -> None:
    _person_corpus(tmp_path)
    # A graph-prefixed id ("kg:person:jane") must still match the bare "person:jane" rows.
    body = (
        _client(tmp_path)
        .get("/api/app/corpus/entity-signals", params={"kind": "person", "id": "kg:person:jane"})
        .json()
    )
    assert len(body["signals"]["guest_coappearance"]["pairs"]) == 2


def test_entity_signals_person_kind_omits_topic_enrichers(tmp_path: Path) -> None:
    # A person card must not carry topic enrichers even when they exist on disk (leanness).
    _person_corpus(tmp_path)
    _env(
        tmp_path,
        "temporal_velocity",
        {
            "window_months": ["2026-03"],
            "topics": [_velocity_topic("topic:ai", label="AI", velocity=2.0, total=9)],
        },
    )
    body = (
        _client(tmp_path)
        .get("/api/app/corpus/entity-signals", params={"kind": "person", "id": "person:jane"})
        .json()
    )
    assert "temporal_velocity" not in body["signals"]


def test_entity_signals_rejects_bad_kind(tmp_path: Path) -> None:
    r = _client(tmp_path).get("/api/app/corpus/entity-signals", params={"kind": "show", "id": "x"})
    assert r.status_code == 422

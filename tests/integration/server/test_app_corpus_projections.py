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
    # #1931: velocity no longer FILTERS by default, so a steady topic is included rather than
    # dropped — what the rail asserts is the ORDER. (These fixtures carry no trend_score, so the
    # route falls back to velocity ordering, which is the stale-artifact path.)
    assert ids[:2] == ["topic:policy", "topic:ai"]
    assert "topic:steady" in ids, "a steady topic is a candidate now, just a lower-ranked one"
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


def test_trending_topics_quiet_state_when_nothing_clears_min_total(tmp_path: Path) -> None:
    """Quiet state = the enricher ran but nothing is substantial enough to show.

    Rewritten for #1931. This used to assert that ``velocity=0.9, total=50`` produced an EMPTY
    rail — a topic mentioned fifty times, hidden because its acceleration ratio was below 1.5.
    That is precisely the defect #1931 fixed: velocity is a ratio, and a sustained topic sits at
    ~1.0 by construction. Executed against the live corpus, that gate passed 2 of 602 topics.

    The quiet state is real and still needs covering — it is now driven by ``min_total``, which
    is the substance gate, not by an acceleration ratio.
    """
    _env(
        tmp_path,
        "temporal_velocity",
        {
            "window_months": ["2026-03"],
            "topics": [_velocity_topic("topic:thin", label="Thin", velocity=6.0, total=1)],
        },
    )
    body = _client(tmp_path).get("/api/app/corpus/trending-topics").json()
    assert body["has_velocity_data"] is True
    assert body["topics"] == [], "a single-mention topic is not trending, whatever its ratio"


def test_a_heavily_discussed_topic_is_no_longer_hidden_by_its_ratio(tmp_path: Path) -> None:
    """The inverse of the case above, and the #1931 regression in one assertion."""
    _env(
        tmp_path,
        "temporal_velocity",
        {
            "window_months": ["2026-03"],
            "topics": [_velocity_topic("topic:busy", label="Busy", velocity=0.9, total=50)],
        },
    )
    body = _client(tmp_path).get("/api/app/corpus/trending-topics").json()
    assert [t["topic_label"] for t in body["topics"]] == ["Busy"], (
        "50 mentions with a below-1.0 acceleration ratio is the corpus's most-discussed shape; "
        "hiding it is what the old min_velocity default did"
    )


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
    # #1927: grounding is NOT a person signal any more, even though the fixture still writes a
    # per-person envelope. The metric was per-Person and scored exactly 1.0 for all 689 people in
    # the real corpus, because an insight is grounded exactly when a supporting quote exists and
    # the quote carries the speaker — so an ungrounded insight has no speaker to attribute it to
    # and the denominator could only ever equal the numerator. A constant is not a signal. It is
    # per-EPISODE now (Show rail); the person card must not offer it.
    assert "grounding_rate" not in signals, (
        "per-person grounding is back on the person card — it can only ever read 100%"
    )


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


# --------------------------------------------------------------------------- #
# Operator plane: GET /api/corpus/entity-signals (?path=-scoped viewer sibling)
# --------------------------------------------------------------------------- #


def test_operator_entity_signals_filters_identically_to_the_consumer(tmp_path: Path) -> None:
    """The viewer's ?path=-scoped endpoint reuses the SAME filter as the consumer route.

    Same person corpus, same focus → the same filtered rows (only pairs/rows touching Jane survive),
    proving the operator plane shares :func:`filtered_entity_signals` rather than re-implementing it.
    """
    _person_corpus(tmp_path)
    body = (
        _client(tmp_path)
        .get(
            "/api/corpus/entity-signals",
            params={"path": str(tmp_path), "kind": "person", "id": "person:jane"},
        )
        .json()
    )
    signals = body["signals"]
    pairs = signals["guest_coappearance"]["pairs"]
    assert len(pairs) == 2
    for p in pairs:
        assert "person:jane" in (p["person_a_id"], p["person_b_id"])
    # #1927 — and the operator plane drops it too, which is the parity this test exists for.
    assert "grounding_rate" not in signals


def test_operator_entity_signals_empty_corpus_returns_no_signals(tmp_path: Path) -> None:
    """An entity with no touching rows (here: a corpus with no envelopes) → empty signals, not 404."""
    body = (
        _client(tmp_path)
        .get(
            "/api/corpus/entity-signals",
            params={"path": str(tmp_path), "kind": "person", "id": "person:jane"},
        )
        .json()
    )
    assert body["signals"] == {}


def test_operator_entity_signals_rejects_bad_kind(tmp_path: Path) -> None:
    r = _client(tmp_path).get(
        "/api/corpus/entity-signals", params={"path": str(tmp_path), "kind": "show", "id": "x"}
    )
    assert r.status_code == 422


def test_trending_rail_default_path_with_post_shrinkage_values(tmp_path: Path) -> None:
    """#1931 regression — the DEFAULT path, with values shaped like a post-shrinkage artifact.

    The rail filtered on ``velocity_last_over_6mo >= 1.5`` before ranking on ``trend_score``, so
    it kept selecting with the signal #1931 had just proved unusable. Executed against the live
    1,066-episode artifact, that gate passed **2 of 602** topics and excluded every one of the
    six the fix was built to surface — each scores velocity 0.25-0.33, because the ratio calls
    the corpus's most-discussed topics "cooling" (fewer mentions last month than their own
    6-month average).

    Every existing trending test passed an explicit ``min_velocity``, so none exercised the
    shipped default. This one does, with no params — the way the player calls it.
    """
    _env(
        tmp_path,
        "temporal_velocity",
        {
            "window_months": ["2026-08", "2026-09"],
            "topics": [
                # The shape that matters: heavily discussed, ratio says cooling, high trend_score.
                {
                    "topic_id": "topic:hot",
                    "topic_label": "open source ai models",
                    "velocity_last_over_6mo": 0.25,
                    "trend_score": 14.0,
                    "total": 16,
                    "monthly_counts": {"2026-08": 9, "2026-09": 7},
                },
                {
                    "topic_id": "topic:warm",
                    "topic_label": "ai regulation",
                    "velocity_last_over_6mo": 0.25,
                    "trend_score": 11.6,
                    "total": 11,
                    "monthly_counts": {"2026-08": 6, "2026-09": 5},
                },
                # Sparse and spiky: the profile the old gate preferred.
                {
                    "topic_id": "topic:spike",
                    "topic_label": "fiscal dominance",
                    "velocity_last_over_6mo": 2.25,
                    "trend_score": 0.7,
                    "total": 3,
                    "monthly_counts": {"2026-08": 0, "2026-09": 3},
                },
            ],
        },
    )
    client = _client(tmp_path)
    body = client.get("/api/app/corpus/trending-topics").json()

    labels = [t["topic_label"] for t in body["topics"]]
    assert labels, "the default path returned an empty rail — the #1931 regression"
    assert labels[0] == "open source ai models", (
        "a heavily-discussed topic whose velocity ratio reads 'cooling' must still lead the rail"
    )
    assert "ai regulation" in labels
    assert labels.index("open source ai models") < labels.index("fiscal dominance"), (
        "a 3-mention spike must not outrank a 16-mention sustained topic"
    )


def test_trending_rail_min_velocity_still_available_when_asked(tmp_path: Path) -> None:
    """The parameter survives — a caller can still ask for accelerating-only topics."""
    _env(
        tmp_path,
        "temporal_velocity",
        {
            "window_months": ["2026-09"],
            "topics": [
                {
                    "topic_id": "topic:hot",
                    "topic_label": "sustained",
                    "velocity_last_over_6mo": 0.25,
                    "trend_score": 14.0,
                    "total": 16,
                    "monthly_counts": {"2026-09": 7},
                },
                {
                    "topic_id": "topic:accel",
                    "topic_label": "accelerating",
                    "velocity_last_over_6mo": 2.25,
                    "trend_score": 0.7,
                    "total": 3,
                    "monthly_counts": {"2026-09": 3},
                },
            ],
        },
    )
    client = _client(tmp_path)
    body = client.get(
        "/api/app/corpus/trending-topics", params={"min_velocity": 1.5}
    ).json()
    assert [t["topic_label"] for t in body["topics"]] == ["accelerating"]

"""Integration tests for the lean corpus projection routes (#perf).

``GET /api/app/corpus/trending-topics`` and ``GET /api/app/corpus/entity-signals`` serve the
top-N / per-entity slice the Home rail and entity card render, instead of the whole ~25 MB
corpus-enrichment payload. Exercised through ``TestClient`` over on-disk envelopes so the route
wrappers (filtering, projection, 422s, the no-enricher path) run for real.
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


def _client(root: Path) -> TestClient:
    data_dir = root / "_appdata"
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="stub", subject="s", email="u@x.com", name="U")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


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

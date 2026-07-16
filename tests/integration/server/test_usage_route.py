"""The operator usage API — GET /api/usage. Token/cost rollup over disk telemetry, sliceable.

Self-contained (no Loki/Langfuse): the endpoint rolls up ``llm_cost`` events a run wrote under the
corpus, de-duplicated by request_id, grouped by the requested dimension.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = pytest.mark.integration


def _write_events(run_dir, events) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"2026-07-15 10:00:00 INFO cost: {json.dumps(e)}" for e in events]
    (run_dir / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ev(**kw):
    base = {
        "event_type": "llm_cost",
        "provider": "openai",
        "model": "gpt-5.4-mini",
        "operation": "gi",
        "stage": "gi",
        "request_id": None,
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "cached_input_tokens": 200,
        "estimated_cost_usd": 0.01,
    }
    base.update(kw)
    return base


@pytest.fixture()
def client(tmp_path):
    _write_events(
        tmp_path / "feeds" / "s" / "run_1",
        [
            _ev(request_id="a", model="gpt-5.4-mini", estimated_cost_usd=0.01),
            _ev(request_id="b", model="gpt-5.4-nano", estimated_cost_usd=0.002),
            _ev(request_id="b", model="gpt-5.4-nano", estimated_cost_usd=0.002),  # dup → once
        ],
    )
    return TestClient(create_app(output_dir=tmp_path))


def test_get_usage_rolls_up_and_dedups(client) -> None:
    r = client.get("/api/usage?group_by=model")
    assert r.status_code == 200
    data = r.json()
    assert data["total"]["calls"] == 2, "duplicate request_id counted once"
    assert data["total"]["cached_input_tokens"] == 400
    models = {g["model"] for g in data["groups"]}
    assert models == {"gpt-5.4-mini", "gpt-5.4-nano"}
    assert "episode_id" in data["dimensions"]


def test_get_usage_slices_by_operation(client) -> None:
    data = client.get("/api/usage?group_by=operation").json()
    assert data["group_by"] == ["operation"]
    assert data["groups"][0]["operation"] == "gi"

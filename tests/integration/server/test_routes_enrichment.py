"""Integration tests for the enrichment HTTP route module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app
from podcast_scraper.server.routes import enrichment as enrichment_route

pytestmark = pytest.mark.integration


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """A corpus directory with the minimal layout the routes expect."""
    (tmp_path / "metadata").mkdir()
    return tmp_path


@pytest.fixture()
def app(corpus: Path) -> FastAPI:
    """An ``enable_jobs_api=True`` app rooted at the test corpus."""
    return create_app(corpus, static_dir=False, enable_jobs_api=True)


# ---------------------------------------------------------------------------
# Router mount + auth gating
# ---------------------------------------------------------------------------


def test_enrichment_router_not_mounted_when_jobs_api_disabled(corpus: Path) -> None:
    """Without the jobs_api gate the router isn't registered → 404."""
    app = create_app(corpus, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/enrichment/status", params={"path": str(corpus)})
    assert r.status_code == 404


def test_enrichment_status_returns_no_status_payload_for_fresh_corpus(
    app: FastAPI, corpus: Path
) -> None:
    client = TestClient(app)
    r = client.get("/api/enrichment/status", params={"path": str(corpus)})
    assert r.status_code == 200
    body = r.json()
    assert body.get("available") is False
    assert "no status yet" in body.get("reason", "")


def test_enrichment_status_returns_500_when_jobs_api_disabled_on_state(
    tmp_path: Path,
) -> None:
    """Mounting the router but turning jobs_api off → 500 (matches pipeline routes)."""
    app = FastAPI()
    app.state.output_dir = tmp_path
    app.state.jobs_api_enabled = False
    app.include_router(enrichment_route.router, prefix="/api")
    client = TestClient(app)
    r = client.get("/api/enrichment/status", params={"path": str(tmp_path)})
    assert r.status_code == 500
    assert "jobs_api" in r.json().get("detail", "").lower()


def test_enrichment_status_rejects_missing_corpus_path_when_no_anchor(
    tmp_path: Path,
) -> None:
    """When the app has no default anchor and the request omits ``?path=``, → 400."""
    app = FastAPI()
    app.state.output_dir = None  # no anchor
    app.state.jobs_api_enabled = True
    app.include_router(enrichment_route.router, prefix="/api")
    client = TestClient(app)
    r = client.get("/api/enrichment/status")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /api/jobs/enrichment — POST
# ---------------------------------------------------------------------------


def test_submit_enrichment_job_returns_202_with_job_id(app: FastAPI, corpus: Path) -> None:
    client = TestClient(app)
    r = client.post(
        "/api/jobs/enrichment",
        params={"path": str(corpus)},
        json={"corpus_only": True},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["job_id"]
    assert body["status"] in ("running", "queued")
    assert body["corpus_path"]


def test_submit_enrichment_job_accepts_empty_body(app: FastAPI, corpus: Path) -> None:
    client = TestClient(app)
    r = client.post("/api/jobs/enrichment", params={"path": str(corpus)})
    assert r.status_code == 202


def test_submit_enrichment_job_persists_command_type_in_jobs_list(
    app: FastAPI, corpus: Path
) -> None:
    client = TestClient(app)
    submit = client.post(
        "/api/jobs/enrichment",
        params={"path": str(corpus)},
        json={"only": ["topic_cooccurrence"]},
    )
    assert submit.status_code == 202
    job_id = submit.json()["job_id"]
    # The new enrichment job appears in the shared /api/jobs registry
    # with command_type = corpus_enrichment.
    listing = client.get("/api/jobs", params={"path": str(corpus)})
    assert listing.status_code == 200
    jobs = listing.json().get("jobs") or []
    matching = [j for j in jobs if j["job_id"] == job_id]
    assert matching, f"enrichment job {job_id} missing from /api/jobs"
    assert matching[0]["command_type"] == "corpus_enrichment"


# ---------------------------------------------------------------------------
# /api/enrichment/health — GET + POST re-enable
# ---------------------------------------------------------------------------


def test_get_health_empty_for_fresh_corpus(app: FastAPI, corpus: Path) -> None:
    client = TestClient(app)
    r = client.get("/api/enrichment/health", params={"path": str(corpus)})
    assert r.status_code == 200
    assert r.json() == {"enrichers": {}}


def test_get_health_with_enricher_id_filter_missing(app: FastAPI, corpus: Path) -> None:
    client = TestClient(app)
    r = client.get(
        "/api/enrichment/health",
        params={"path": str(corpus), "enricher_id": "unknown_enricher"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False


def test_post_re_enable_persists_to_health_file(app: FastAPI, corpus: Path) -> None:
    from podcast_scraper.enrichment.health import HealthRegistry
    from podcast_scraper.enrichment.paths import enrichment_health_path

    # Seed an auto-disabled record on disk.
    reg = HealthRegistry(corpus)
    h = reg.get("x")
    h.auto_disabled = True
    h.auto_disabled_reason = "burned in prod"
    h.consecutive_failures = 5
    reg.save()

    client = TestClient(app)
    r = client.post(
        "/api/enrichment/health/x/re-enable",
        params={"path": str(corpus)},
        json={"reason": "operator confirmed transient HF outage"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["enricher_id"] == "x"
    assert body["auto_disabled"] is False
    assert body["consecutive_failures"] == 0

    # Persisted to disk.
    persisted = json.loads(enrichment_health_path(corpus).read_text())
    assert persisted["enrichers"]["x"]["auto_disabled"] is False


def test_post_re_enable_works_with_empty_body(app: FastAPI, corpus: Path) -> None:
    client = TestClient(app)
    r = client.post(
        "/api/enrichment/health/topic_similarity/re-enable",
        params={"path": str(corpus)},
    )
    # Even when no body, the route should accept defaults.
    assert r.status_code == 200


def test_post_re_enable_rejects_oversized_reason(app: FastAPI, corpus: Path) -> None:
    """HealthReEnableRequest.reason has a 500-char cap (Pydantic Field max_length)."""
    client = TestClient(app)
    too_long = "x" * 501
    r = client.post(
        "/api/enrichment/health/topic_similarity/re-enable",
        params={"path": str(corpus)},
        json={"reason": too_long},
    )
    assert r.status_code == 422  # FastAPI validation error


def test_post_re_enable_emits_jsonl_event(app: FastAPI, corpus: Path) -> None:
    """Manual recovery must leave an enrichment.health.re_enabled audit row."""
    client = TestClient(app)
    r = client.post(
        "/api/enrichment/health/x/re-enable",
        params={"path": str(corpus)},
        json={"reason": "operator override"},
    )
    assert r.status_code == 200
    jsonl = corpus / "enrichments" / "run.jsonl"
    assert jsonl.is_file()
    events = [json.loads(line) for line in jsonl.read_text().splitlines() if line]
    re_enabled = [e for e in events if e["event_type"] == "enrichment.health.re_enabled"]
    assert re_enabled, "missing enrichment.health.re_enabled event"
    assert re_enabled[-1]["enricher_id"] == "x"
    assert re_enabled[-1]["reason"] == "operator override"


# ---------------------------------------------------------------------------
# /api/enrichment/run-summary
# ---------------------------------------------------------------------------


def test_get_run_summary_returns_no_run_for_fresh_corpus(app: FastAPI, corpus: Path) -> None:
    client = TestClient(app)
    r = client.get("/api/enrichment/run-summary", params={"path": str(corpus)})
    assert r.status_code == 200
    body = r.json()
    assert body.get("available") is False


def test_get_run_summary_returns_payload_after_executor_run(app: FastAPI, corpus: Path) -> None:
    """Run the executor in-process so the run_summary file exists,
    then query the route."""
    import asyncio

    from podcast_scraper.enrichment.executor import EnrichmentExecutor
    from podcast_scraper.enrichment.protocol import EnricherSet
    from podcast_scraper.enrichment.registry import EnricherRegistry

    executor = EnrichmentExecutor(
        corpus_root=corpus,
        registry=EnricherRegistry(),
        enricher_set=EnricherSet(),
    )
    asyncio.run(executor.run())

    client = TestClient(app)
    r = client.get("/api/enrichment/run-summary", params={"path": str(corpus)})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "per_enricher" in body


# ---------------------------------------------------------------------------
# /api/enrichment/events
# ---------------------------------------------------------------------------


def test_get_events_returns_empty_for_fresh_corpus(app: FastAPI, corpus: Path) -> None:
    client = TestClient(app)
    r = client.get("/api/enrichment/events", params={"path": str(corpus)})
    assert r.status_code == 200
    body = r.json()
    assert body == {"events": [], "count": 0}


def test_get_events_filters_by_enricher_id_and_event_type(app: FastAPI, corpus: Path) -> None:
    """Pre-populate run.jsonl with mixed events, then verify filtering."""
    jsonl = corpus / "enrichments" / "run.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {"event_type": "enrichment.run.started", "run_id": "r1"},
        {
            "event_type": "enrichment.enricher.started",
            "enricher_id": "topic_cooccurrence",
            "run_id": "r1",
        },
        {
            "event_type": "enrichment.enricher.completed",
            "enricher_id": "topic_cooccurrence",
            "status": "ok",
            "run_id": "r1",
        },
        {
            "event_type": "enrichment.enricher.started",
            "enricher_id": "topic_consensus",
            "run_id": "r1",
        },
    ]
    jsonl.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    client = TestClient(app)

    # Filter by enricher_id.
    r = client.get(
        "/api/enrichment/events",
        params={"path": str(corpus), "enricher_id": "topic_cooccurrence"},
    )
    assert r.status_code == 200
    ids = [e.get("enricher_id") for e in r.json()["events"]]
    assert set(ids) == {"topic_cooccurrence"}

    # Filter by event_type.
    r = client.get(
        "/api/enrichment/events",
        params={"path": str(corpus), "event_type": "enrichment.run.started"},
    )
    assert r.status_code == 200
    types = [e["event_type"] for e in r.json()["events"]]
    assert types == ["enrichment.run.started"]


def test_get_events_respects_limit(app: FastAPI, corpus: Path) -> None:
    jsonl = corpus / "enrichments" / "run.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text(
        "\n".join(
            json.dumps({"event_type": "enrichment.enricher.started", "i": i}) for i in range(20)
        )
        + "\n",
        encoding="utf-8",
    )
    client = TestClient(app)
    r = client.get("/api/enrichment/events", params={"path": str(corpus), "limit": 5})
    assert r.status_code == 200
    assert r.json()["count"] == 5


# ---------------------------------------------------------------------------
# /api/enrichment/metrics
# ---------------------------------------------------------------------------


def test_get_metrics_returns_empty_payload_for_fresh_corpus(app: FastAPI, corpus: Path) -> None:
    client = TestClient(app)
    r = client.get("/api/enrichment/metrics", params={"path": str(corpus)})
    assert r.status_code == 200
    assert r.json() == {"window": "24h", "per_enricher": {}}


def test_get_metrics_with_window_param(app: FastAPI, corpus: Path) -> None:
    client = TestClient(app)
    r = client.get("/api/enrichment/metrics", params={"path": str(corpus), "window": "1h"})
    assert r.status_code == 200
    assert r.json()["window"] == "1h"

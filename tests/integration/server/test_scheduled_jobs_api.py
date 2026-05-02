"""Integration tests for ``GET /api/scheduled-jobs`` (#708)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("apscheduler")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    return tmp_path


def _put_operator_yaml(corpus: Path, body: str) -> None:
    (corpus / "viewer_operator.yaml").write_text(body, encoding="utf-8")


def test_endpoint_404s_when_jobs_api_disabled(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False)
    with TestClient(app) as client:
        r = client.get("/api/scheduled-jobs", params={"path": str(corpus)})
    assert r.status_code == 404


def test_endpoint_returns_empty_when_no_schedule(corpus: Path) -> None:
    _put_operator_yaml(corpus, "max_episodes: 1\n")
    app = create_app(corpus, static_dir=False, enable_jobs_api=True)
    with TestClient(app) as client:
        r = client.get("/api/scheduled-jobs", params={"path": str(corpus)})
    assert r.status_code == 200
    body = r.json()
    assert body["jobs"] == []
    # ``scheduler_running`` is False when there are no enabled schedules.
    assert body["scheduler_running"] is False
    assert body["timezone"]


def test_endpoint_lists_configured_jobs_with_next_run(corpus: Path) -> None:
    _put_operator_yaml(
        corpus,
        "scheduled_jobs:\n"
        "  - name: morning\n"
        "    cron: '0 4 * * *'\n"
        "  - name: evening\n"
        "    cron: '0 20 * * *'\n"
        "    enabled: false\n",
    )
    app = create_app(corpus, static_dir=False, enable_jobs_api=True)
    with TestClient(app) as client:
        r = client.get("/api/scheduled-jobs", params={"path": str(corpus)})
    assert r.status_code == 200
    body = r.json()
    by_name = {j["name"]: j for j in body["jobs"]}
    assert set(by_name) == {"morning", "evening"}
    assert by_name["morning"]["enabled"] is True
    assert by_name["morning"]["cron"] == "0 4 * * *"
    assert by_name["morning"]["next_run_at"] is not None
    assert by_name["evening"]["enabled"] is False
    assert by_name["evening"]["next_run_at"] is None
    assert body["scheduler_running"] is True


def test_operator_config_put_triggers_scheduler_reload(corpus: Path) -> None:
    """PUT /api/operator-config picks up new schedules without app restart."""
    _put_operator_yaml(corpus, "max_episodes: 1\n")
    app = create_app(
        corpus,
        static_dir=False,
        enable_jobs_api=True,
        enable_operator_config_api=True,
    )
    with TestClient(app) as client:
        # Initial state: no schedule.
        r0 = client.get("/api/scheduled-jobs", params={"path": str(corpus)})
        assert r0.status_code == 200
        assert r0.json()["jobs"] == []

        new_yaml = (
            "max_episodes: 1\n"
            "scheduled_jobs:\n"
            "  - name: added-via-put\n"
            "    cron: '0 4 * * *'\n"
        )
        put = client.put(
            "/api/operator-config",
            params={"path": str(corpus)},
            json={"content": new_yaml},
        )
        assert put.status_code == 200, put.text

        # Same corpus, scheduler should now know about the new schedule.
        r1 = client.get("/api/scheduled-jobs", params={"path": str(corpus)})
        assert r1.status_code == 200
        body = r1.json()
        names = [j["name"] for j in body["jobs"]]
        assert names == ["added-via-put"]
        assert body["scheduler_running"] is True


def test_endpoint_requires_corpus_path_when_no_default(corpus: Path) -> None:
    from fastapi import FastAPI

    from podcast_scraper.server.routes import scheduled_jobs as scheduled_jobs_route

    app = FastAPI()
    app.state.output_dir = None
    app.state.jobs_api_enabled = True
    app.include_router(scheduled_jobs_route.router, prefix="/api")
    client = TestClient(app)
    r = client.get("/api/scheduled-jobs")
    assert r.status_code == 400
    assert "Corpus path" in r.json().get("detail", "")

"""Integration tests for GET /api/enrichment/stats + the force lever (RFC-118 §5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = pytest.mark.integration


class _FakeProc:
    """A finished child: submit tests check the job row, never a real ``enrich`` run."""

    pid = 91501

    async def wait(self) -> int:
        return 0


async def _fake_spawn(argv, corpus_root: Path, log_abs: Path):  # noqa: ARG001
    # Without a factory the route spawned a real ``python -m podcast_scraper.cli enrich`` in
    # the TestClient's per-request event loop; the loop closed with the child's transport still
    # open and its GC raised "Event loop is closed" at teardown (2026-10-28).
    log_abs.parent.mkdir(parents=True, exist_ok=True)
    log_abs.write_bytes(b"fake-log\n")
    return _FakeProc()


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    meta = tmp_path / "metadata"
    meta.mkdir(exist_ok=True)
    (meta / "e1.metadata.json").write_text(
        json.dumps({"episode": {"episode_id": "e1"}}), encoding="utf-8"
    )
    (meta / "e1.gi.json").write_text("{}", encoding="utf-8")
    app = create_app(output_dir=tmp_path, enable_jobs_api=True)
    app.state.jobs_subprocess_factory = _fake_spawn
    return TestClient(app)


def test_stats_goes_quiet_when_corpus_scope_current(client: TestClient, tmp_path: Path) -> None:
    from podcast_scraper.enrichment.eval.admission import known_enricher_manifests

    enrich_dir = tmp_path / "enrichments"
    enrich_dir.mkdir(exist_ok=True)
    for m in known_enricher_manifests().values():
        if m.scope.value != "corpus":
            continue
        (enrich_dir / m.writes).write_text(
            json.dumps(
                {
                    "derived": True,
                    "computed_at": "2099-01-01T00:00:00Z",
                    "enricher_id": m.id,
                    "enricher_version": m.version,
                    "schema_version": "1.0",
                    "status": "ok",
                    "data": {},
                }
            ),
            encoding="utf-8",
        )
    body = client.get("/api/enrichment/stats").json()
    assert body["reenrich_recommended"] is False
    assert body["reenrich_reasons"] == []


def test_force_flag_reaches_the_job_argv(client: TestClient, tmp_path: Path) -> None:
    # A submit with nothing to resolve an enricher from is a 400 (no profile, no enrichers block).
    (tmp_path / "viewer_operator.yaml").write_text("profile: airgapped_thin\n", encoding="utf-8")
    r = client.post("/api/jobs/enrichment", json={"force": True, "corpus_only": True})
    assert r.status_code == 202, r.text
    reg = tmp_path / ".viewer" / "jobs.jsonl"
    rows = [json.loads(ln) for ln in reg.read_text(encoding="utf-8").splitlines() if ln.strip()]
    mine = [row for row in rows if row.get("command_type") == "corpus_enrichment"]
    assert mine and "--force" in str(mine[-1]["argv_summary"])


def test_default_submit_has_no_force(client: TestClient, tmp_path: Path) -> None:
    # A submit with nothing to resolve an enricher from is a 400 (no profile, no enrichers block).
    (tmp_path / "viewer_operator.yaml").write_text("profile: airgapped_thin\n", encoding="utf-8")
    r = client.post("/api/jobs/enrichment", json={"corpus_only": True})
    assert r.status_code == 202, r.text
    reg = tmp_path / ".viewer" / "jobs.jsonl"
    rows = [json.loads(ln) for ln in reg.read_text(encoding="utf-8").splitlines() if ln.strip()]
    mine = [row for row in rows if row.get("command_type") == "corpus_enrichment"]
    assert mine and "--force" not in str(mine[-1]["argv_summary"])

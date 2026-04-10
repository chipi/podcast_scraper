"""Integration tests for GET /api/corpus/stats, documents, and runs/summary (dashboard)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


def _episode_doc(
    *,
    feed_id: str = "myfeed",
    episode_title: str = "Hello",
    published: str = "2024-03-10T00:00:00",
) -> dict:
    return {
        "feed": {"feed_id": feed_id, "title": "My Show"},
        "episode": {
            "episode_id": "ep1",
            "title": episode_title,
            "published_date": published,
        },
        "summary": {"title": "Sum", "bullets": ["a"]},
    }


def test_corpus_stats_publish_month_histogram(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "a.metadata.json").write_text(
        json.dumps(_episode_doc(published="2024-01-05T00:00:00")),
        encoding="utf-8",
    )
    (meta / "b.metadata.json").write_text(
        json.dumps(_episode_doc(episode_title="B", published="2024-01-20T00:00:00")),
        encoding="utf-8",
    )
    (meta / "c.metadata.json").write_text(
        json.dumps(
            _episode_doc(
                episode_title="C",
                published="2024-02-01T00:00:00",
                feed_id="otherfeed",
            ),
        ),
        encoding="utf-8",
    )

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/stats", params={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == str(tmp_path.resolve())
    hist = body["publish_month_histogram"]
    assert hist.get("2024-01") == 2
    assert hist.get("2024-02") == 1
    assert body["catalog_episode_count"] == 3
    assert body["catalog_feed_count"] == 2
    assert body["digest_topics_configured"] >= 2


def test_corpus_manifest_and_run_summary_documents(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "1.0.0",
        "feeds": [{"stable_feed_dir": "f1", "episodes_processed": 3}],
    }
    (tmp_path / "corpus_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    summary = {"schema_version": "1.0.0", "overall_ok": True, "feeds": []}
    (tmp_path / "corpus_run_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)

    m = client.get("/api/corpus/documents/manifest", params={"path": str(tmp_path)})
    assert m.status_code == 200
    assert m.json()["feeds"][0]["stable_feed_dir"] == "f1"

    s = client.get("/api/corpus/documents/run-summary", params={"path": str(tmp_path)})
    assert s.status_code == 200
    assert s.json()["overall_ok"] is True

    empty = tmp_path / "empty_sub"
    empty.mkdir()
    miss = client.get("/api/corpus/documents/manifest", params={"path": str(empty)})
    assert miss.status_code == 404


def test_corpus_runs_summary_discovers_run_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "feeds" / "pod" / "run_2024"
    run_dir.mkdir(parents=True)
    run_payload = {
        "schema_version": "1.0.0",
        "run_id": "rid-1",
        "created_at": "2024-06-01T12:00:00Z",
        "metrics": {
            "run_duration_seconds": 120.5,
            "episodes_scraped_total": 5,
            "errors_total": 0,
            "gi_artifacts_generated": 5,
            "kg_artifacts_generated": 5,
            "time_scraping": 10.0,
            "time_parsing": 2.0,
            "time_normalizing": 1.0,
            "time_io_and_waiting": 30.0,
            "episode_statuses": [
                {"status": "ok"},
                {"status": "ok"},
                {"status": "failed"},
            ],
        },
    }
    (run_dir / "run.json").write_text(json.dumps(run_payload), encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/runs/summary", params={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.json()
    assert len(body["runs"]) == 1
    row = body["runs"][0]
    assert row["run_id"] == "rid-1"
    assert row["run_duration_seconds"] == 120.5
    assert row["episode_outcomes"]["ok"] == 2
    assert row["episode_outcomes"]["failed"] == 1
    assert "feeds/pod" in row["relative_path"]


def test_corpus_runs_summary_legacy_string_episode_statuses(tmp_path: Path) -> None:
    """Older run.json used default=str for dataclass episode rows — parse status from repr."""
    run_dir = tmp_path / "run_legacy"
    run_dir.mkdir()
    ok_row = (
        "EpisodeStatus(episode_id='e1', episode_number=1, status='ok', "
        "error_type=None, error_message=None, stage=None, retry_count=0)"
    )
    fail_row = ok_row.replace("'ok'", "'failed'")
    run_payload = {
        "schema_version": "1.0.0",
        "run_id": "legacy",
        "created_at": "2024-01-01T00:00:00Z",
        "metrics": {
            "run_duration_seconds": 1.0,
            "episode_statuses": [ok_row, fail_row],
        },
    }
    (run_dir / "run.json").write_text(json.dumps(run_payload), encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/runs/summary", params={"path": str(tmp_path)})
    assert r.status_code == 200
    row = r.json()["runs"][0]
    assert row["episode_outcomes"]["ok"] == 1
    assert row["episode_outcomes"]["failed"] == 1

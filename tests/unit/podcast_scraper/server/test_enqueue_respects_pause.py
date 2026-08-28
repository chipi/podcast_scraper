"""New submissions must respect the operator pause flag (#1785 stop endpoint).

2026-08-28 incident: the pause flag only held promotion of already-QUEUED jobs, so the
03:00 scheduler cron — submitting into an idle queue — started immediately and ran a full
sweep the operator had explicitly braked (on a not-yet-deployed skip fix; another window
re-ingested). A free slot is not consent to run: while paused, every new submission lands
queued and waits for resume.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from podcast_scraper.server.jobs import enqueue_enrichment_job, enqueue_pipeline_job
from podcast_scraper.server.queue_sweeper import pause_drain, resume_drain

pytestmark = [pytest.mark.unit]


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "viewer_operator.yaml").write_text("max_episodes: 1\n", encoding="utf-8")
    return tmp_path


def test_pipeline_submission_lands_queued_while_paused(corpus: Path) -> None:
    pause_drain(corpus)
    rec = enqueue_pipeline_job(corpus, corpus / "viewer_operator.yaml")
    assert rec["status"] == "queued", (
        "a submission during pause was started immediately — the scheduler-cron bypass "
        "(2026-08-28 03:00 fire) is back"
    )


def test_pipeline_submission_runs_when_not_paused(corpus: Path) -> None:
    pause_drain(corpus)
    resume_drain(corpus)
    rec = enqueue_pipeline_job(corpus, corpus / "viewer_operator.yaml")
    assert rec["status"] == "running"


def test_enrichment_submission_lands_queued_while_paused(corpus: Path) -> None:
    pause_drain(corpus)
    rec = enqueue_enrichment_job(corpus, operator_yaml=corpus / "viewer_operator.yaml")
    assert rec["status"] == "queued"

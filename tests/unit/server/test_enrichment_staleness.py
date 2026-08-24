"""Unit tests for ``compute_enrichment_staleness`` (RFC-118 §5).

The freshness verdicts must come from on-disk facts alone — no enrichment run, no
models — and every reason must be typed so the operator UI / MCP can explain WHY.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from podcast_scraper.server.enrichment_staleness import (
    compute_enrichment_staleness,
    REASON_ARTIFACTS_NEWER,
    REASON_LAST_RUN_FAILED,
    REASON_NEVER_RAN,
    REASON_VERSION_CHANGED,
)

pytestmark = pytest.mark.unit


def _corpus_with_artifacts(tmp_path: Path, *, artifact_age_s: float = 3600.0) -> Path:
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "e1.metadata.json").write_text(
        json.dumps({"episode": {"episode_id": "e1"}}), encoding="utf-8"
    )
    gi = meta_dir / "e1.gi.json"
    gi.write_text("{}", encoding="utf-8")
    old = time.time() - artifact_age_s
    for p in (gi, meta_dir / "e1.metadata.json"):
        os.utime(p, (old, old))
    return tmp_path


def _write_envelope(
    corpus: Path,
    writes: str,
    *,
    version: str,
    status: str = "ok",
    computed_at: str | None = None,
) -> None:
    out = corpus / "enrichments" / writes
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "derived": True,
                "computed_at": computed_at or "2099-01-01T00:00:00Z",
                "enricher_id": writes.replace(".json", ""),
                "enricher_version": version,
                "schema_version": "1.0",
                "status": status,
                "data": {},
            }
        ),
        encoding="utf-8",
    )


def _row(fields, enricher_id: str):
    return next(r for r in fields.enrichers if r.enricher_id == enricher_id)


def test_empty_corpus_everything_never_ran(tmp_path):
    fields = compute_enrichment_staleness(_corpus_with_artifacts(tmp_path))
    assert fields.reenrich_recommended is True
    assert REASON_NEVER_RAN in fields.reenrich_reasons
    assert all(r.stale for r in fields.enrichers)


def test_current_corpus_scope_envelope_is_fresh(tmp_path):
    corpus = _corpus_with_artifacts(tmp_path)
    from podcast_scraper.enrichment.enrichers.topic_consensus import TopicConsensusEnricher

    m = TopicConsensusEnricher.manifest
    _write_envelope(corpus, m.writes, version=m.version, status="ok")
    row = _row(compute_enrichment_staleness(corpus), m.id)
    assert row.stale is False
    assert row.reasons == []


def test_version_change_is_reported(tmp_path):
    corpus = _corpus_with_artifacts(tmp_path)
    from podcast_scraper.enrichment.enrichers.topic_consensus import TopicConsensusEnricher

    m = TopicConsensusEnricher.manifest
    _write_envelope(corpus, m.writes, version="0.0.1", status="ok")
    row = _row(compute_enrichment_staleness(corpus), m.id)
    assert row.stale and REASON_VERSION_CHANGED in row.reasons


def test_failed_last_run_is_reported(tmp_path):
    # The efdca585 shape: the envelope exists but the last outcome was a timeout.
    corpus = _corpus_with_artifacts(tmp_path)
    from podcast_scraper.enrichment.enrichers.topic_consensus import TopicConsensusEnricher

    m = TopicConsensusEnricher.manifest
    _write_envelope(corpus, m.writes, version=m.version, status="timeout")
    fields = compute_enrichment_staleness(corpus)
    row = _row(fields, m.id)
    assert row.stale and REASON_LAST_RUN_FAILED in row.reasons
    assert fields.reenrich_recommended is True


def test_newer_artifacts_are_reported(tmp_path):
    corpus = _corpus_with_artifacts(tmp_path, artifact_age_s=0.0)  # artifacts touched NOW
    from podcast_scraper.enrichment.enrichers.topic_consensus import TopicConsensusEnricher

    m = TopicConsensusEnricher.manifest
    _write_envelope(
        corpus, m.writes, version=m.version, status="ok", computed_at="2020-01-01T00:00:00Z"
    )
    row = _row(compute_enrichment_staleness(corpus), m.id)
    assert row.stale and REASON_ARTIFACTS_NEWER in row.reasons


def test_statusless_envelope_is_never_ran_not_failed(tmp_path):
    # Review M1: a legacy/manual envelope with NO status field is unknown provenance —
    # reporting it as failed would prompt a spurious full re-derive.
    corpus = _corpus_with_artifacts(tmp_path)
    from podcast_scraper.enrichment.enrichers.topic_consensus import TopicConsensusEnricher

    m = TopicConsensusEnricher.manifest
    out = corpus / "enrichments" / m.writes
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"enricher_version": m.version, "computed_at": "2099-01-01T00:00:00Z"}),
        encoding="utf-8",
    )
    row = _row(compute_enrichment_staleness(corpus), m.id)
    assert REASON_NEVER_RAN in row.reasons
    assert REASON_LAST_RUN_FAILED not in row.reasons


def test_failed_overall_run_summary_rolls_up(tmp_path):
    corpus = _corpus_with_artifacts(tmp_path)
    enrich_dir = corpus / "enrichments"
    enrich_dir.mkdir(parents=True, exist_ok=True)
    (enrich_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "finished_at": "2099-01-01T00:00:00Z",
                "per_enricher": {},
            }
        ),
        encoding="utf-8",
    )
    fields = compute_enrichment_staleness(corpus)
    assert fields.last_run_status == "failed"
    assert REASON_LAST_RUN_FAILED in fields.reenrich_reasons


def test_episode_scope_rows_do_not_drive_the_rollup(tmp_path):
    # Fresh corpus-scope envelopes; episode-scope never ran. The table shows it,
    # but the recommendation stays quiet — per-episode staleness self-heals in-run.
    corpus = _corpus_with_artifacts(tmp_path)
    from podcast_scraper.enrichment.eval.admission import known_enricher_manifests

    for m in known_enricher_manifests().values():
        if m.scope.value == "corpus":
            _write_envelope(corpus, m.writes, version=m.version, status="ok")
    fields = compute_enrichment_staleness(corpus)
    episode_rows = [r for r in fields.enrichers if r.scope == "episode"]
    assert episode_rows and all(r.stale for r in episode_rows)
    assert fields.reenrich_recommended is False

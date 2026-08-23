"""MCP corpus admin tools (RFC-118 §5): corpus_status / reenrich / reindex.

The write tools must ENQUEUE (queued row in the shared registry), never spawn —
RUNNING is a promise only the API server can keep.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.mcp.context import CorpusContext
from podcast_scraper.mcp.tools import admin

pytestmark = pytest.mark.unit


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "e1.metadata.json").write_text(
        json.dumps({"episode": {"episode_id": "e1"}}), encoding="utf-8"
    )
    (meta / "e1.gi.json").write_text("{}", encoding="utf-8")
    return tmp_path


@pytest.fixture
def ctx(corpus: Path) -> CorpusContext:
    return CorpusContext.from_path(corpus)


def _registry_rows(corpus: Path) -> list[dict]:
    reg = corpus / ".viewer" / "jobs.jsonl"
    if not reg.is_file():
        return []
    return [json.loads(ln) for ln in reg.read_text(encoding="utf-8").splitlines() if ln.strip()]


class TestCorpusStatus:
    def test_reports_enrichment_and_index_facts(self, ctx, corpus):
        out = admin.corpus_status(ctx)
        assert out["enrichment"]["reenrich_recommended"] is True  # nothing ever ran
        assert "never_ran" in out["enrichment"]["reenrich_reasons"]
        assert out["index"]["present"] is False
        assert out["delta_backbone"]["fingerprint_manifest_present"] is False


class TestReenrich:
    def test_enqueues_queued_row(self, ctx, corpus):
        out = admin.reenrich(ctx, force=True)
        assert out["status"] == "queued"
        rows = _registry_rows(corpus)
        assert len(rows) == 1
        assert rows[0]["command_type"] == "corpus_enrichment"
        assert "--force" in str(rows[0]["argv_summary"])

    def test_without_force_no_force_flag(self, ctx, corpus):
        admin.reenrich(ctx, force=False)
        rows = _registry_rows(corpus)
        assert rows and "--force" not in str(rows[0]["argv_summary"])


class TestReindex:
    def test_enqueues_queued_row(self, ctx, corpus):
        out = admin.reindex(ctx, rebuild=True)
        assert out["status"] == "queued"
        rows = _registry_rows(corpus)
        assert len(rows) == 1
        assert rows[0]["command_type"] == "corpus_reindex"
        summary = str(rows[0]["argv_summary"])
        assert "--rebuild" in summary
        # MUST be the main-CLI ``index`` verb: the Docker job factory re-prefixes the
        # stored tail with ``python -m podcast_scraper.cli``, so a bare-module argv
        # (podcast_scraper.search.reindex) would die in the container's argparse.
        assert "podcast_scraper.cli" in summary and '"index"' in summary
        assert "podcast_scraper.search.reindex" not in summary
        # Parity with POST /api/index/rebuild: the queued child re-derives clusters too.
        assert "--with-clusters" in summary

    def test_identical_queued_reindex_coalesces(self, ctx, corpus):
        first = admin.reindex(ctx)
        second = admin.reindex(ctx)
        assert first["job_id"] == second["job_id"]
        assert len(_registry_rows(corpus)) == 1

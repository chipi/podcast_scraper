#!/usr/bin/env python3
"""E2E tests for append / resume (GitHub #444).

Uses Path 1 (transcript in feed) plus local transformers/spaCy like the basic E2E critical path,
but is **not** marked ``critical_path``: the scenario runs the pipeline **twice**, doubling ML work.
It runs under ``make test-e2e`` (full E2E suite) and is skipped when summarization models are not
pre-cached (same pattern as ``test_basic_e2e``).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

import podcast_scraper.cli as cli
from podcast_scraper import config

pytestmark = [pytest.mark.e2e, pytest.mark.module_workflow]


@pytest.mark.e2e
class TestAppendResumeCLIE2E:
    """Subprocess-equivalent CLI coverage for stable append dir + second-run skip."""

    def test_cli_append_second_invocation_reuses_run_dir_and_skips_episode(self, e2e_server):
        """Two ``cli.main`` runs with ``--append`` share one ``run_append_*`` tree (GitHub #444)."""
        from tests.integration.ml_model_cache_helpers import (
            require_transformers_model_cached,
        )

        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)

        rss_url = e2e_server.urls.feed("podcast1_with_transcript")

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                rss_url,
                "--output-dir",
                tmpdir,
                "--max-episodes",
                "1",
                "--append",
                "--auto-speakers",
                "--generate-summaries",
                "--summary-model",
                config.TEST_DEFAULT_SUMMARY_MODEL,
                "--summary-reduce-model",
                config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL,
                "--generate-metadata",
            ]

            assert cli.main(argv) == 0, "first append run should succeed"

            run_dirs = sorted(Path(tmpdir).glob("run_append_*"))
            assert len(run_dirs) == 1, f"expected one stable append run dir, got {run_dirs}"
            run_root = run_dirs[0]

            transcripts_after_first = list(run_root.glob("transcripts/*.txt"))
            assert len(transcripts_after_first) >= 1, "first run should write a transcript"

            assert cli.main(argv) == 0, "second append run should succeed"

            run_dirs_again = sorted(Path(tmpdir).glob("run_append_*"))
            assert len(run_dirs_again) == 1
            assert run_dirs_again[0] == run_root, "second run must reuse the same directory"

            transcripts_after_second = list(run_root.glob("transcripts/*.txt"))
            assert len(transcripts_after_second) == len(
                transcripts_after_first
            ), "append skip should not add duplicate transcript files"

            index_path = run_root / "index.json"
            assert index_path.is_file(), "index.json should exist after finalize"
            idx = json.loads(index_path.read_text(encoding="utf-8"))
            assert idx.get("schema_version") == "1.1.0"
            assert idx.get("pipeline_append") is True
            assert idx.get("episodes_processed", 0) >= 1

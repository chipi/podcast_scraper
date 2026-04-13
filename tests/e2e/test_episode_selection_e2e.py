#!/usr/bin/env python3
"""E2E tests for episode selection (GitHub #521).

Uses ``podcast1_episode_selection`` on the E2E mock HTTP server: three short episodes,
newest-first in the RSS document, all Path 1 (transcript URLs). Same pattern as
``test_basic_e2e`` / ``test_append_resume_e2e`` (``cli.main``, transformers/spaCy skip
when models are not cached).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

import podcast_scraper.cli as cli
from podcast_scraper import config

pytestmark = [pytest.mark.e2e, pytest.mark.module_workflow]


def _assert_metadata_episode_title_contains(output_root: Path, needle: str) -> None:
    meta_files = list(output_root.rglob("*.metadata.json"))
    assert meta_files, f"expected at least one metadata file under {output_root}"
    titles = []
    for path in meta_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        titles.append(data.get("episode", {}).get("title", ""))
    assert any(needle in t for t in titles), f"expected {needle!r} in one of {titles!r}"


def _selection_argv_base(rss_url: str, out: str) -> list[str]:
    return [
        rss_url,
        "--output-dir",
        out,
        "--max-episodes",
        "1",
        "--no-transcribe-missing",
        "--auto-speakers",
        "--generate-summaries",
        "--summary-model",
        config.TEST_DEFAULT_SUMMARY_MODEL,
        "--summary-reduce-model",
        config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL,
        "--generate-metadata",
    ]


@pytest.mark.e2e
class TestEpisodeSelectionCLIE2E:
    """CLI E2E coverage for ``episode_order``, offset, and date bounds."""

    @pytest.mark.critical_path
    def test_cli_episode_order_newest_first_item(self, e2e_server):
        from tests.integration.ml_model_cache_helpers import (
            require_transformers_model_cached,
        )

        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)
        rss_url = e2e_server.urls.feed("podcast1_episode_selection")
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = _selection_argv_base(rss_url, tmpdir)
            assert cli.main(argv) == 0
            _assert_metadata_episode_title_contains(Path(tmpdir), "E2E Selection Marker NEWEST")

    def test_cli_episode_order_oldest_processes_last_in_document(self, e2e_server):
        from tests.integration.ml_model_cache_helpers import (
            require_transformers_model_cached,
        )

        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)
        rss_url = e2e_server.urls.feed("podcast1_episode_selection")
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = _selection_argv_base(rss_url, tmpdir) + [
                "--episode-order",
                "oldest",
            ]
            assert cli.main(argv) == 0
            _assert_metadata_episode_title_contains(Path(tmpdir), "E2E Selection Marker OLDEST")

    def test_cli_episode_offset_skips_after_order(self, e2e_server):
        from tests.integration.ml_model_cache_helpers import (
            require_transformers_model_cached,
        )

        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)
        rss_url = e2e_server.urls.feed("podcast1_episode_selection")
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = _selection_argv_base(rss_url, tmpdir) + [
                "--episode-offset",
                "1",
            ]
            assert cli.main(argv) == 0
            _assert_metadata_episode_title_contains(Path(tmpdir), "E2E Selection Marker MIDDLE")

    def test_cli_since_until_filters_to_middle_episode(self, e2e_server):
        from tests.integration.ml_model_cache_helpers import (
            require_transformers_model_cached,
        )

        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)
        rss_url = e2e_server.urls.feed("podcast1_episode_selection")
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = _selection_argv_base(rss_url, tmpdir) + [
                "--since",
                "2025-09-10",
                "--until",
                "2025-09-20",
            ]
            assert cli.main(argv) == 0
            _assert_metadata_episode_title_contains(Path(tmpdir), "E2E Selection Marker MIDDLE")

    def test_cli_config_episode_order_overridden_by_cli(self, e2e_server):
        """Config file says oldest; CLI ``--episode-order newest`` wins."""
        from tests.integration.ml_model_cache_helpers import (
            require_transformers_model_cached,
        )

        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)
        rss_url = e2e_server.urls.feed("podcast1_episode_selection")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            payload = {
                "rss": rss_url,
                "output_dir": tmpdir,
                "max_episodes": 1,
                "episode_order": "oldest",
                "transcribe_missing": False,
                "auto_speakers": True,
                "generate_summaries": True,
                "summary_model": config.TEST_DEFAULT_SUMMARY_MODEL,
                "summary_reduce_model": config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL,
                "generate_metadata": True,
            }
            Path(config_path).write_text(json.dumps(payload), encoding="utf-8")
            argv = [
                "--config",
                config_path,
                "--episode-order",
                "newest",
            ]
            assert cli.main(argv) == 0
            _assert_metadata_episode_title_contains(Path(tmpdir), "E2E Selection Marker NEWEST")

    def test_cli_config_episode_offset_overridden_by_cli(self, e2e_server):
        """Config ``episode_offset: 0``; CLI ``--episode-offset 2`` skips to oldest in feed."""
        from tests.integration.ml_model_cache_helpers import (
            require_transformers_model_cached,
        )

        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)
        rss_url = e2e_server.urls.feed("podcast1_episode_selection")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            payload = {
                "rss": rss_url,
                "output_dir": tmpdir,
                "max_episodes": 1,
                "episode_order": "newest",
                "episode_offset": 0,
                "transcribe_missing": False,
                "auto_speakers": True,
                "generate_summaries": True,
                "summary_model": config.TEST_DEFAULT_SUMMARY_MODEL,
                "summary_reduce_model": config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL,
                "generate_metadata": True,
            }
            Path(config_path).write_text(json.dumps(payload), encoding="utf-8")
            argv = [
                "--config",
                config_path,
                "--episode-offset",
                "2",
            ]
            assert cli.main(argv) == 0
            _assert_metadata_episode_title_contains(Path(tmpdir), "E2E Selection Marker OLDEST")

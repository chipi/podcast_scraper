#!/usr/bin/env python3
"""E2E tests for multi-episode feed processing with real Whisper.

Extracted from tests/integration/infrastructure/test_e2e_server.py —
these tests use real Whisper for transcription, so they belong in E2E
per the 3-tier ML/AI testing policy.
"""

import os
import tempfile
from pathlib import Path

import pytest

from podcast_scraper import Config, config as config_module, run_pipeline


@pytest.mark.e2e
@pytest.mark.ml_models
class TestMultiEpisodeWhisperProcessing:
    """Multi-episode feed processing with real Whisper transcription."""

    @pytest.mark.critical_path
    def test_multi_episode_processing_fast(self, e2e_server):
        """Multi-episode feed processes 1 episode with real Whisper (fast variant)."""
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=True,
                whisper_model=config_module.TEST_DEFAULT_WHISPER_MODEL,
                generate_metadata=True,
                metadata_format="json",
            )

            count, summary = run_pipeline(cfg)

            assert count == 1, f"Should process 1 multi-episode episode (fast mode), got {count}"

            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) == 1
            ), f"Should have exactly 1 metadata file, got {len(metadata_files)}"

    @pytest.mark.slow
    def test_multi_episode_processing_full(self, e2e_server):
        """Multi-episode feed processes all 5 episodes with real Whisper (full variant)."""
        test_mode = os.environ.get("E2E_TEST_MODE", "multi_episode").lower()
        expected_episodes = 1 if test_mode == "fast" else 5

        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=5,
                transcribe_missing=True,
                whisper_model=config_module.TEST_DEFAULT_WHISPER_MODEL,
                generate_metadata=True,
                metadata_format="json",
            )

            count, summary = run_pipeline(cfg)

            if test_mode == "fast":
                assert count == expected_episodes, (
                    f"Should process {expected_episodes} episode(s) "
                    f"(mode: {test_mode}), got {count}"
                )
            else:
                assert count >= 2, (
                    f"Should process at least 2 episodes (with transcripts) "
                    f"in multi-episode mode, got {count}"
                )

            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            if test_mode == "fast":
                assert len(metadata_files) == expected_episodes, (
                    f"Should have exactly {expected_episodes} metadata "
                    f"file(s), got {len(metadata_files)}"
                )
            else:
                assert len(metadata_files) >= 2, (
                    f"Should have at least 2 metadata file(s) in "
                    f"multi-episode mode, got {len(metadata_files)}"
                )

            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            if test_mode == "fast":
                assert len(transcript_files) >= expected_episodes, (
                    f"Should have at least {expected_episodes} transcript "
                    f"file(s), got {len(transcript_files)}"
                )
            else:
                assert len(transcript_files) >= 2, (
                    f"Should have at least 2 transcript file(s) in "
                    f"multi-episode mode, got {len(transcript_files)}"
                )

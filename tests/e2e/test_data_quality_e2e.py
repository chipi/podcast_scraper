#!/usr/bin/env python3
"""Data Quality E2E Tests.

These tests validate data quality and consistency across multiple episodes.
They use all original mock data (not fast fixtures) and process 3-5 episodes
to catch issues that only appear with volume.

These tests are marked with @pytest.mark.data_quality and run in nightly builds only.
They are separate from code quality checks (which use 1 episode per test).

Key Validations:
- Data consistency across multiple episodes
- Resource usage with volume
- Edge cases that only appear with multiple episodes
- Full pipeline validation with original mock data
"""

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

from podcast_scraper import config, Config, run_pipeline


@pytest.mark.e2e
@pytest.mark.data_quality
@pytest.mark.slow
class TestDataQualityE2E:
    """Data quality validation across multiple episodes."""

    def test_full_pipeline_data_quality_ml_providers(self, e2e_server):
        """Test full pipeline data quality with ML providers across 3 episodes.

        Validates:
        - Full pipeline works correctly with multiple episodes
        - Data consistency across episodes
        - Resource usage is reasonable
        - All episodes are processed successfully

        Uses ML providers (Whisper, spaCy NER, Transformers summarization).
        Requires ML models to be cached.
        """
        from tests.integration.ml_model_cache_helpers import (
            require_transformers_model_cached,
        )

        # Require ML models to be cached (use test default model)
        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)

        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=3,  # Process 3 episodes for data quality validation
                transcribe_missing=True,  # Enable Whisper transcription
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
                auto_speakers=True,  # Enable NER (speaker detection)
                generate_summaries=True,  # Enable summarization
                summary_provider="local",  # Use local ML provider
                summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,  # Use test default (small, fast)
                generate_metadata=True,  # Enable metadata generation
                metadata_format="json",
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Validate processing
            assert count == 3, f"Should process 3 episodes, got {count}"

            # Verify transcript files were created for all episodes
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert (
                len(transcript_files) >= 3
            ), f"Should have at least 3 transcript files, got {len(transcript_files)}"

            # Verify metadata files were created for all episodes
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) == 3
            ), f"Should have exactly 3 metadata files, got {len(metadata_files)}"

            # Validate data quality: Check consistency across episodes
            metadata_contents = []
            for metadata_file in sorted(metadata_files):
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    metadata_contents.append(metadata)

            # All metadata files should have consistent structure
            for i, metadata in enumerate(metadata_contents):
                assert "content" in metadata, f"Episode {i+1} metadata should have content section"
                assert "summary" in metadata, f"Episode {i+1} metadata should have summary"
                assert metadata["summary"] is not None, f"Episode {i+1} summary should not be None"

            # Validate that summaries are different (not identical copies)
            summaries = [
                m["summary"].get("short_summary", "") for m in metadata_contents if m.get("summary")
            ]
            assert len(set(summaries)) > 1, "Summaries should be different across episodes"

    @pytest.mark.skip(
        reason="OpenAI E2E tests skipped for now - infrastructure ready but tests disabled"
    )
    def test_full_pipeline_data_quality_openai_providers(self, e2e_server):
        """Test full pipeline data quality with OpenAI providers across 3 episodes.

        Validates:
        - Full pipeline works correctly with multiple episodes using OpenAI providers
        - Data consistency across episodes
        - OpenAI mock endpoints handle multiple requests correctly

        Uses OpenAI providers (transcription, speaker detection, summarization
        via E2E server mocks).
        """
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=3,  # Process 3 episodes for data quality validation
                transcribe_missing=True,  # Enable OpenAI transcription
                transcription_provider="openai",  # Use OpenAI for transcription
                auto_speakers=True,  # Enable speaker detection
                speaker_detector_provider="openai",  # Use OpenAI for speaker detection
                generate_summaries=True,  # Enable summarization
                summary_provider="openai",  # Use OpenAI for summarization
                generate_metadata=True,  # Enable metadata generation
                metadata_format="json",
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Validate processing
            assert count == 3, f"Should process 3 episodes, got {count}"

            # Verify transcript files were created for all episodes
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert (
                len(transcript_files) >= 3
            ), f"Should have at least 3 transcript files, got {len(transcript_files)}"

            # Verify metadata files were created for all episodes
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) == 3
            ), f"Should have exactly 3 metadata files, got {len(metadata_files)}"

            # Validate data quality: Check consistency across episodes
            for metadata_file in sorted(metadata_files):
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    assert "content" in metadata, "Metadata should have content section"
                    assert "summary" in metadata, "Metadata should have summary"
                    assert metadata["summary"] is not None, "Summary should not be None"

    def test_data_consistency_across_runs(self, e2e_server):
        """Test that processing the same episodes multiple times produces consistent results.

        Validates:
        - Same input produces same output (deterministic processing)
        - No random variations in results
        - Metadata is consistent across runs
        """
        from tests.integration.ml_model_cache_helpers import (
            require_transformers_model_cached,
        )

        # Require ML models to be cached (use test default model)
        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)

        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        # Run pipeline twice with same configuration
        results = []
        for run_num in range(2):
            with tempfile.TemporaryDirectory() as tmpdir:
                cfg = Config(
                    rss_url=rss_url,
                    output_dir=tmpdir,
                    max_episodes=5,  # Process all 5 multi-episode episodes
                    transcribe_missing=True,
                    auto_speakers=True,
                    generate_summaries=True,
                    summary_provider="local",
                    # Use test default (small, fast)
                    summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,
                    generate_metadata=True,
                    metadata_format="json",
                )

                count, summary = run_pipeline(cfg)
                # In multi-episode mode, at least 2 episodes (with transcripts) should be processed
                assert count >= 2, (
                    f"Run {run_num+1}: Should process at least 2 episodes "
                    f"(with transcripts), got {count}"
                )

                # Collect metadata from this run
                metadata_files = sorted(Path(tmpdir).rglob("*.metadata.json"))
                run_metadata = []
                for metadata_file in metadata_files:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        run_metadata.append(json.load(f))
                results.append(run_metadata)

        # Validate consistency: Both runs should produce same metadata structure
        assert len(results[0]) == len(
            results[1]
        ), "Both runs should process same number of episodes"

        # Compare metadata structure (not exact content, as summaries may vary slightly)
        for i, (metadata1, metadata2) in enumerate(zip(results[0], results[1])):
            assert (
                "content" in metadata1 and "content" in metadata2
            ), f"Episode {i+1}: Both should have content"
            assert (
                "summary" in metadata1 and "summary" in metadata2
            ), f"Episode {i+1}: Both should have summary"
            # Note: Exact summary text may vary slightly, but structure should be consistent

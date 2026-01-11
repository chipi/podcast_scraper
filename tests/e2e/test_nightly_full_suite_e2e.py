"""Nightly comprehensive E2E test suite with production ML models.

This test suite runs comprehensive tests with:
- All 5 podcasts (p01-p05) with all 3 episodes each (15 total episodes)
- Production ML models: PROD_DEFAULT_WHISPER_MODEL, PROD_DEFAULT_SUMMARY_MODEL,
  PROD_DEFAULT_SUMMARY_REDUCE_MODEL
- Full pipeline validation (transcription → NER → summarization → metadata)

These tests are marked with @pytest.mark.nightly (NOT @pytest.mark.e2e) to
separate them from regular E2E tests. They run only in nightly builds.

See GitHub Issue #174 for implementation details.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper import config
from tests.conftest import create_test_config


def create_nightly_config(output_dir: str, rss_url: str):
    """Create config for nightly tests with production models.

    Uses production ML models:
    - Whisper: PROD_DEFAULT_WHISPER_MODEL (production quality)
    - Summary MAP: PROD_DEFAULT_SUMMARY_MODEL (production quality)
    - Summary REDUCE: PROD_DEFAULT_SUMMARY_REDUCE_MODEL (production quality)
    - NER: DEFAULT_NER_MODEL (same for tests and production)

    Args:
        output_dir: Output directory for test results
        rss_url: RSS feed URL for the podcast

    Returns:
        Config object with production model settings
    """
    return create_test_config(
        rss_url=rss_url,
        output_dir=output_dir,
        transcribe_missing=True,
        whisper_model=config.PROD_DEFAULT_WHISPER_MODEL,  # Production Whisper model
        generate_metadata=True,
        generate_summaries=True,
        summary_model=config.PROD_DEFAULT_SUMMARY_MODEL,  # Production MAP model
        summary_reduce_model=config.PROD_DEFAULT_SUMMARY_REDUCE_MODEL,  # Production REDUCE model
        auto_speakers=True,
        speaker_detector_provider="spacy",
        summary_provider="transformers",
        transcription_provider="whisper",
        max_episodes=None,  # Process all episodes
    )


@pytest.mark.nightly
@pytest.mark.slow
@pytest.mark.ml_models
class TestNightlyFullSuite:
    """Comprehensive nightly test suite with production models.

    Tests all 5 podcasts (p01-p05) with all 3 episodes each (15 total episodes).
    Uses production ML models for realistic testing.

    Each podcast is tested independently via parametrization, allowing parallel
    execution and clear failure attribution.
    """

    @pytest.mark.parametrize(
        "podcast_name",
        [
            "podcast1",  # p01 - Mountain Biking
            "podcast2",  # p02 - Software Engineering
            "podcast3",  # p03 - Scuba Diving
            "podcast4",  # p04 - Photography
            "podcast5",  # p05 - Investing
        ],
    )
    def test_nightly_podcast_full_pipeline(self, podcast_name, e2e_server, tmpdir):
        """Test full pipeline for a single podcast with all episodes.

        Args:
            podcast_name: Podcast identifier (podcast1-podcast5)
            e2e_server: E2E HTTP server fixture
            tmpdir: Temporary directory for output
        """
        # Create config with podcast-specific RSS URL (dynamic port from e2e_server)
        rss_url = e2e_server.urls.feed(podcast_name)
        cfg = create_nightly_config(str(tmpdir), rss_url)

        # Run pipeline
        from podcast_scraper import workflow

        saved, summary = workflow.run_pipeline(cfg)

        # Verify results
        assert saved >= 0, f"Pipeline should process episodes for {podcast_name}"
        assert isinstance(summary, str), "Pipeline should return summary string"

        # Verify transcript files were created (all 3 episodes)
        # Files are in run_id subdirectory: run_*/transcripts/
        transcript_files = list(Path(tmpdir).rglob("transcripts/*.txt"))
        # Filter to only main transcript files (not .cleaned.txt)
        main_transcripts = [f for f in transcript_files if not f.name.endswith(".cleaned.txt")]
        assert len(main_transcripts) >= 3, (
            f"Should create at least 3 transcript files for {podcast_name}, "
            f"found {len(main_transcripts)}"
        )

        # Verify metadata files were created (if metadata generation enabled)
        if cfg.generate_metadata:
            metadata_files = list(Path(tmpdir).rglob("metadata/*.metadata.json"))
            assert len(metadata_files) >= 3, (
                f"Should create at least 3 metadata files for {podcast_name}, "
                f"found {len(metadata_files)}"
            )

        # Verify metrics file was created
        metrics_files = list(Path(tmpdir).rglob("metrics.json"))
        assert len(metrics_files) >= 1, f"Metrics file should exist for {podcast_name}"

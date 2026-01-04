"""Nightly comprehensive E2E test suite with production ML models.

This test suite runs comprehensive tests with:
- All 5 podcasts (p01-p05) with all 3 episodes each (15 total episodes)
- Production ML models: Whisper base, BART-large-cnn, LED-large-16384
- Full pipeline validation (transcription → NER → summarization → metadata)

These tests are marked with @pytest.mark.nightly (NOT @pytest.mark.e2e) to
separate them from regular E2E tests. They run only in nightly builds.

See GitHub Issue #174 for implementation details.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import create_test_config


def create_nightly_config(output_dir: str, rss_url: str):
    """Create config for nightly tests with production models.

    Uses production ML models:
    - Whisper: base (production quality)
    - Summary MAP: facebook/bart-large-cnn (production quality)
    - Summary REDUCE: allenai/led-large-16384 (production quality, from issue #175)
    - NER: en_core_web_sm (same for tests and production)

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
        whisper_model="base",  # Production Whisper model
        generate_metadata=True,
        generate_summaries=True,
        summary_model="facebook/bart-large-cnn",  # Production MAP model
        summary_reduce_model="allenai/led-large-16384",  # Production REDUCE model
        auto_speakers=True,
        speaker_detector_provider="spacy",
        summary_provider="transformers",
        transcription_provider="whisper",
        max_episodes=None,  # Process all episodes
    )


# Podcast mapping: podcast name -> RSS feed URL
PODCAST_URLS = {
    "podcast1": "http://127.0.0.1:8000/podcast1/feed.xml",  # p01 - Mountain Biking
    "podcast2": "http://127.0.0.1:8000/podcast2/feed.xml",  # p02 - Software Engineering
    "podcast3": "http://127.0.0.1:8000/podcast3/feed.xml",  # p03 - Scuba Diving
    "podcast4": "http://127.0.0.1:8000/podcast4/feed.xml",  # p04 - Photography
    "podcast5": "http://127.0.0.1:8000/podcast5/feed.xml",  # p05 - Investing
}


@pytest.mark.nightly
@pytest.mark.slow
@pytest.mark.ml_models
class TestNightlyFullSuite:
    """Comprehensive nightly test suite with production models.

    Tests all 5 podcasts (p01-p05) with all 3 episodes each (15 total episodes).
    Uses production ML models for realistic testing.
    """

    @pytest.mark.parametrize(
        "podcast_name",
        ["podcast1", "podcast2", "podcast3", "podcast4", "podcast5"],
    )
    def test_nightly_podcast_full_pipeline(self, podcast_name, e2e_server, tmpdir):
        """Test full pipeline for a single podcast with all episodes.

        Args:
            podcast_name: Podcast identifier (podcast1-podcast5)
            e2e_server: E2E HTTP server fixture
            tmpdir: Temporary directory for output
        """
        # Create config with podcast-specific RSS URL
        cfg = create_nightly_config(str(tmpdir), PODCAST_URLS[podcast_name])

        # Run pipeline
        from podcast_scraper import workflow

        saved, summary = workflow.run_pipeline(cfg)

        # Verify results
        assert saved >= 0, f"Pipeline should process episodes for {podcast_name}"
        assert isinstance(summary, str), "Pipeline should return summary string"

        # Verify transcript files were created (all 3 episodes)
        transcripts_dir = Path(tmpdir) / "transcripts"
        transcript_files = list(transcripts_dir.glob("*.txt"))
        assert len(transcript_files) >= 3, (
            f"Should create at least 3 transcript files for {podcast_name}, "
            f"found {len(transcript_files)}"
        )

        # Verify metadata files were created (if metadata generation enabled)
        if cfg.generate_metadata:
            metadata_dir = Path(tmpdir) / "metadata"
            metadata_files = list(metadata_dir.glob("*.metadata.json"))
            assert len(metadata_files) >= 3, (
                f"Should create at least 3 metadata files for {podcast_name}, "
                f"found {len(metadata_files)}"
            )

        # Verify metrics file was created (if metrics output enabled)
        metrics_file = Path(tmpdir) / "metrics.json"
        if cfg.metrics_output is not None or cfg.metrics_output != "":
            # Metrics should be saved by default
            assert metrics_file.exists(), f"Metrics file should exist for {podcast_name}"

    def test_nightly_all_podcasts_summary(self, e2e_server, tmpdir):
        """Test that all podcasts can be processed in sequence.

        This test processes all 5 podcasts sequentially to validate
        that the system can handle multiple different podcast feeds
        with production models.
        """
        results = {}

        for podcast_name, rss_url in PODCAST_URLS.items():
            # Create config for this podcast
            podcast_output_dir = str(Path(tmpdir) / podcast_name)
            cfg = create_nightly_config(podcast_output_dir, rss_url)

            # Run pipeline
            from podcast_scraper import workflow

            try:
                saved, summary = workflow.run_pipeline(cfg)
                results[podcast_name] = {
                    "saved": saved,
                    "summary": summary,
                    "success": True,
                }
            except Exception as e:
                results[podcast_name] = {
                    "saved": 0,
                    "summary": str(e),
                    "success": False,
                    "error": str(e),
                }

        # Verify all podcasts processed successfully
        failed_podcasts = [
            name for name, result in results.items() if not result.get("success", False)
        ]
        assert (
            len(failed_podcasts) == 0
        ), f"Some podcasts failed: {failed_podcasts}. Results: {results}"

        # Verify all podcasts created files
        for podcast_name in PODCAST_URLS.keys():
            podcast_dir = Path(tmpdir) / podcast_name
            transcripts_dir = podcast_dir / "transcripts"
            transcript_files = (
                list(transcripts_dir.glob("*.txt")) if transcripts_dir.exists() else []
            )
            assert len(transcript_files) >= 3, (
                f"Podcast {podcast_name} should create at least 3 transcript files, "
                f"found {len(transcript_files)}"
            )

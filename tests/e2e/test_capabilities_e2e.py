"""E2E tests for provider capability contract in full pipeline scenarios."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from podcast_scraper import config, run_pipeline
from podcast_scraper.providers.capabilities import get_provider_capabilities, is_local_provider


@pytest.mark.e2e
@pytest.mark.ml_models
class TestCapabilitiesE2E:
    """E2E tests for capability contract in full pipeline."""

    def test_pipeline_uses_capability_based_decisions(self, e2e_server):
        """Test that pipeline makes decisions based on capabilities."""
        feed_url = e2e_server.urls.feed("podcast1_with_transcript")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = config.Config(
                rss_url=feed_url,
                output_dir=tmpdir,
                max_episodes=1,
                generate_summaries=True,
                summary_provider="transformers",
                summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,
                generate_metadata=True,
                metadata_format="json",
                auto_speakers=False,
                transcribe_missing=False,
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Verify pipeline completed
            assert count > 0

            # Verify metadata was created
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert len(metadata_files) > 0

            # Verify summary uses normalized schema (capability-based)
            if metadata_files:
                import json

                with open(metadata_files[0], "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                if "summary" in metadata and metadata["summary"]:
                    # Summary should have normalized schema fields
                    assert "bullets" in metadata["summary"], "Summary should use normalized schema"
                    assert isinstance(metadata["summary"]["bullets"], list)
                    assert len(metadata["summary"]["bullets"]) > 0

    def test_local_vs_api_provider_detection_e2e(self, e2e_server):
        """Test that pipeline correctly detects local vs API providers."""
        feed_url = e2e_server.urls.feed("podcast1_with_transcript")

        # Test with local provider
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_local = config.Config(
                rss_url=feed_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcription_provider="whisper",
                transcribe_missing=True,
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            )

            from podcast_scraper.transcription.factory import create_transcription_provider

            local_provider = create_transcription_provider(cfg_local)
            assert is_local_provider(local_provider), "Whisper provider should be detected as local"

        # Test with API provider (if available)
        # This would require API keys, so we'll just verify the capability detection works
        cfg_api = config.Config(
            rss_url=feed_url,
            transcription_provider="openai",
            openai_api_key="test-key",
        )

        # Even without valid API key, we can check capability detection
        api_provider = create_transcription_provider(cfg_api)
        caps = get_provider_capabilities(api_provider)
        # API providers should support JSON mode
        assert caps.supports_json_mode, "OpenAI provider should support JSON mode"


if __name__ == "__main__":
    pytest.main([__file__])

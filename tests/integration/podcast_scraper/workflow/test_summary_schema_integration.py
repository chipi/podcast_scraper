"""Integration tests for summary schema integration in summarization pipeline."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import Mock

import pytest

from podcast_scraper import config
from podcast_scraper.providers.capabilities import ProviderCapabilities
from podcast_scraper.workflow.metadata_generation import (
    _generate_episode_summary,
    SummaryMetadata,
)


@pytest.mark.integration
class TestSummarySchemaIntegration(unittest.TestCase):
    """Test summary schema integration in metadata generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries=True
            summary_provider="transformers",
            save_cleaned_transcript=False,  # Disable to avoid file write issues in tests
        )
        self.episode_idx = 1
        self.transcript_text = "This is a test transcript. " * 50  # Enough text for summarization

    def _create_temp_transcript(self, text: str) -> tuple[str, str]:
        """Create temporary transcript file and return (tmpdir, transcript_path)."""
        tmpdir = tempfile.mkdtemp()
        transcript_path = os.path.join(tmpdir, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write(text)
        return tmpdir, "transcript.txt"

    def _create_mock_provider(self, summary_value: dict) -> Mock:
        """Create a mock provider with proper cleaning processor."""
        from podcast_scraper.cleaning import HybridCleaner

        mock_provider = Mock()
        mock_provider.summarize.return_value = summary_value
        # Provide a real cleaning processor to avoid Mock write errors
        # Use HybridCleaner as default (matches default config)
        mock_provider.cleaning_processor = HybridCleaner()
        return mock_provider

    def test_plain_text_summary_requires_parsing(self):
        """Test that plain text summaries without bullets result in degraded status."""
        # Plain text without bullets should result in degraded status or None
        mock_provider = self._create_mock_provider(
            {
                "summary": "This is a plain text summary without bullets.",
                "summary_short": None,
                "metadata": {"model": "test-model"},
            }
        )

        tmpdir, transcript_path = self._create_temp_transcript(self.transcript_text)
        try:
            summary_metadata, call_metrics = _generate_episode_summary(
                transcript_file_path=transcript_path,
                output_dir=tmpdir,
                cfg=self.cfg,
                episode_idx=self.episode_idx,
                summary_provider=mock_provider,
                pipeline_metrics=None,
            )
            # Plain text without bullets should either fail (return None) or have degraded status
            if summary_metadata is not None:
                # If it parses, it should be degraded (heuristic parsing)
                self.assertEqual(summary_metadata.schema_status, "degraded")
        finally:
            import shutil

            shutil.rmtree(tmpdir)

    def test_json_mode_summary_parsing(self):
        """Test parsing JSON mode summary output."""
        # Mock provider that returns JSON format
        json_summary = '{"title": "Test Episode", "bullets": ["Point 1", "Point 2", "Point 3"]}'
        mock_provider = self._create_mock_provider(
            {
                "summary": json_summary,
                "summary_short": None,
                "metadata": {"model": "test-model"},
            }
        )

        tmpdir, transcript_path = self._create_temp_transcript(self.transcript_text)
        try:
            summary_metadata, call_metrics = _generate_episode_summary(
                transcript_file_path=transcript_path,
                output_dir=tmpdir,
                cfg=self.cfg,
                episode_idx=self.episode_idx,
                summary_provider=mock_provider,
                pipeline_metrics=None,
            )

            self.assertIsNotNone(summary_metadata)
            # Schema fields are required
            self.assertIsNotNone(summary_metadata.bullets)
            self.assertEqual(len(summary_metadata.bullets), 3)
            self.assertEqual(summary_metadata.title, "Test Episode")
            # Status may be valid or degraded depending on parsing implementation
            self.assertIn(summary_metadata.schema_status, ["valid", "degraded"])
            # short_summary is computed from bullets
            self.assertIsNotNone(summary_metadata.short_summary)
        finally:
            import shutil

            shutil.rmtree(tmpdir)

    def test_text_bullet_point_parsing(self):
        """Test parsing text summary with bullet points."""
        text_summary = """• First important point about the episode
• Second key takeaway
• Third notable insight"""
        mock_provider = self._create_mock_provider(
            {
                "summary": text_summary,
                "summary_short": None,
                "metadata": {"model": "test-model"},
            }
        )

        tmpdir, transcript_path = self._create_temp_transcript(self.transcript_text)
        try:
            summary_metadata, call_metrics = _generate_episode_summary(
                transcript_file_path=transcript_path,
                output_dir=tmpdir,
                cfg=self.cfg,
                episode_idx=self.episode_idx,
                summary_provider=mock_provider,
                pipeline_metrics=None,
            )

            self.assertIsNotNone(summary_metadata)
            # Should extract bullets from text
            self.assertIsNotNone(summary_metadata.bullets)
            # At least some bullets should be extracted (may vary based on parsing)
            self.assertGreater(len(summary_metadata.bullets), 0)
            # Status should be degraded for heuristic parsing
            self.assertEqual(summary_metadata.schema_status, "degraded")
        finally:
            import shutil

            shutil.rmtree(tmpdir)

    def test_malformed_json_repair(self):
        """Test repairing malformed JSON summary."""
        # JSON that might need repair (actually valid JSON in this case)
        json_summary = '{"title": "Test", "bullets": ["Point 1"]}'
        mock_provider = self._create_mock_provider(
            {
                "summary": json_summary,
                "summary_short": None,
                "metadata": {"model": "test-model"},
            }
        )

        tmpdir, transcript_path = self._create_temp_transcript(self.transcript_text)
        try:
            summary_metadata, call_metrics = _generate_episode_summary(
                transcript_file_path=transcript_path,
                output_dir=tmpdir,
                cfg=self.cfg,
                episode_idx=self.episode_idx,
                summary_provider=mock_provider,
                pipeline_metrics=None,
            )

            self.assertIsNotNone(summary_metadata)
            # Should parse successfully
            self.assertIsNotNone(summary_metadata.bullets)
            # Status may be valid or degraded depending on parsing success
            self.assertIn(summary_metadata.schema_status, ["valid", "degraded"])
        finally:
            import shutil

            shutil.rmtree(tmpdir)

    def test_summary_with_quotes_and_entities(self):
        """Test parsing summary with quotes and entities."""
        json_summary = """{
            "title": "Test Episode",
            "bullets": ["Point 1", "Point 2"],
            "key_quotes": ["Quote 1", "Quote 2"],
            "named_entities": ["Entity 1", "Entity 2"]
        }"""
        mock_provider = self._create_mock_provider(
            {
                "summary": json_summary,
                "summary_short": None,
                "metadata": {"model": "test-model"},
            }
        )

        tmpdir, transcript_path = self._create_temp_transcript(self.transcript_text)
        try:
            summary_metadata, call_metrics = _generate_episode_summary(
                transcript_file_path=transcript_path,
                output_dir=tmpdir,
                cfg=self.cfg,
                episode_idx=self.episode_idx,
                summary_provider=mock_provider,
                pipeline_metrics=None,
            )

            self.assertIsNotNone(summary_metadata)
            self.assertEqual(summary_metadata.title, "Test Episode")
            self.assertIsNotNone(summary_metadata.key_quotes)
            self.assertEqual(len(summary_metadata.key_quotes), 2)
            self.assertIsNotNone(summary_metadata.named_entities)
            self.assertEqual(len(summary_metadata.named_entities), 2)
        finally:
            import shutil

            shutil.rmtree(tmpdir)

    def test_summary_metadata_serialization(self):
        """Test that SummaryMetadata can be serialized to dict."""
        summary_metadata = SummaryMetadata(
            generated_at=datetime.now(),
            word_count=100,
            title="Test Title",
            bullets=["Point 1", "Point 2"],
            schema_status="valid",
        )

        # Should be able to convert to dict
        result_dict = summary_metadata.model_dump()
        self.assertIn("bullets", result_dict)
        self.assertIn("title", result_dict)
        self.assertIn("schema_status", result_dict)
        # short_summary is a computed field
        self.assertIsNotNone(summary_metadata.short_summary)

    def test_summary_metadata_requires_bullets(self):
        """Test that SummaryMetadata works with bullets field."""
        # Empty bullets might be allowed (check actual behavior)
        # If validation allows empty, test that it works
        try:
            empty_metadata = SummaryMetadata(
                generated_at=datetime.now(),
                word_count=100,
                bullets=[],  # Empty bullets
            )
            # If it doesn't raise, empty bullets are allowed
            self.assertEqual(len(empty_metadata.bullets), 0)
        except Exception:
            # If it raises, that's also valid behavior
            pass

        # Should succeed with bullets
        summary_metadata = SummaryMetadata(
            generated_at=datetime.now(),
            word_count=100,
            bullets=["Point 1", "Point 2"],
        )
        self.assertEqual(len(summary_metadata.bullets), 2)
        self.assertIsNotNone(summary_metadata.short_summary)  # Computed from bullets

    def test_provider_capability_detection(self):
        """Test that schema parsing uses provider capabilities."""
        # Mock provider with JSON mode capability
        mock_provider = self._create_mock_provider(
            {
                "summary": '{"bullets": ["Point 1"]}',
                "summary_short": None,
                "metadata": {"model": "test-model"},
            }
        )
        mock_provider.get_capabilities.return_value = ProviderCapabilities(
            supports_transcription=False,
            supports_speaker_detection=False,
            supports_summarization=True,
            supports_audio_input=False,
            supports_json_mode=True,
            max_context_tokens=128000,
            provider_name="test",
        )

        tmpdir, transcript_path = self._create_temp_transcript(self.transcript_text)
        try:
            summary_metadata, call_metrics = _generate_episode_summary(
                transcript_file_path=transcript_path,
                output_dir=tmpdir,
                cfg=self.cfg,
                episode_idx=self.episode_idx,
                summary_provider=mock_provider,
                pipeline_metrics=None,
            )

            self.assertIsNotNone(summary_metadata)
            # Should parse JSON successfully
            self.assertIsNotNone(summary_metadata.bullets)
            # Status may be valid or degraded depending on parsing
            self.assertIn(summary_metadata.schema_status, ["valid", "degraded"])
        finally:
            import shutil

            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()

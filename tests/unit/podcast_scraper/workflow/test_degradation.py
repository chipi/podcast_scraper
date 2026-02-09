#!/usr/bin/env python3
"""Unit tests for graceful degradation policy."""

import unittest

import pytest

from podcast_scraper.workflow.degradation import DegradationPolicy, handle_stage_failure

pytestmark = [pytest.mark.unit]


class TestDegradationPolicy(unittest.TestCase):
    """Tests for DegradationPolicy."""

    def test_default_policy(self):
        """Test default policy values."""
        policy = DegradationPolicy()
        self.assertTrue(policy.save_transcript_on_summarization_failure)
        self.assertTrue(policy.save_summary_on_entity_extraction_failure)
        self.assertIsNone(policy.fallback_provider_on_failure)
        self.assertTrue(policy.continue_on_stage_failure)

    def test_handle_stage_failure_summarization_save_transcript(self):
        """Test handling summarization failure with save_transcript enabled."""
        policy = DegradationPolicy(save_transcript_on_summarization_failure=True)
        error = Exception("Summarization failed")

        result = handle_stage_failure("summarization", error, policy, episode_idx=1)
        self.assertTrue(result)  # Should continue

    def test_handle_stage_failure_summarization_no_save(self):
        """Test handling summarization failure with save_transcript disabled."""
        policy = DegradationPolicy(
            save_transcript_on_summarization_failure=False,
            continue_on_stage_failure=False,
        )
        error = Exception("Summarization failed")

        result = handle_stage_failure("summarization", error, policy, episode_idx=1)
        self.assertFalse(result)  # Should stop

    def test_handle_stage_failure_entity_extraction_save_summary(self):
        """Test handling entity extraction failure with save_summary enabled."""
        policy = DegradationPolicy(save_summary_on_entity_extraction_failure=True)
        error = Exception("Entity extraction failed")

        result = handle_stage_failure("entity_extraction", error, policy, episode_idx=1)
        self.assertTrue(result)  # Should continue

    def test_handle_stage_failure_entity_extraction_no_save(self):
        """Test handling entity extraction failure with save_summary disabled."""
        policy = DegradationPolicy(
            save_summary_on_entity_extraction_failure=False,
            continue_on_stage_failure=False,
        )
        error = Exception("Entity extraction failed")

        result = handle_stage_failure("entity_extraction", error, policy, episode_idx=1)
        self.assertFalse(result)  # Should stop

    def test_handle_stage_failure_transcription(self):
        """Test handling transcription failure (always fatal)."""
        policy = DegradationPolicy()
        error = Exception("Transcription failed")

        result = handle_stage_failure("transcription", error, policy, episode_idx=1)
        self.assertFalse(result)  # Should always stop

    def test_handle_stage_failure_metadata(self):
        """Test handling metadata failure (non-fatal)."""
        policy = DegradationPolicy(continue_on_stage_failure=True)
        error = Exception("Metadata generation failed")

        result = handle_stage_failure("metadata", error, policy, episode_idx=1)
        self.assertTrue(result)  # Should continue

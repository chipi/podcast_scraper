#!/usr/bin/env python3
"""E2E tests for preprocessing pipeline.

These tests verify that preprocessing functions correctly clean transcripts:
- Credit stripping (strip_credits)
- Boilerplate removal (strip_garbage_lines)
- Timestamp removal (clean_transcript with remove_timestamps=True)
- Speaker normalization (clean_transcript with normalize_speakers=True)
- Artifact removal (remove_summarization_artifacts)

All tests use real preprocessing functions and verify output quality.
"""

import os
import sys

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import preprocessing


@pytest.mark.e2e
@pytest.mark.critical_path
class TestPreprocessingPipeline:
    """E2E tests for preprocessing pipeline functions."""

    def test_strip_credits(self):
        """Test credit stripping removes credit blocks correctly."""
        # Sample transcript with credit block
        text_with_credits = """This is the main content of the episode.

Produced by John Doe.
Music by Jane Smith.
Edited by Bob Johnson.

More episode content here."""
        cleaned = preprocessing.strip_credits(text_with_credits)

        # Verify credits are removed
        assert "Produced by" not in cleaned, "Credit line should be removed"
        assert "Music by" not in cleaned, "Credit line should be removed"
        assert "Edited by" not in cleaned, "Credit line should be removed"

        # Verify main content is preserved
        assert "main content" in cleaned, "Main content should be preserved"
        assert "More episode content" in cleaned, "Content after credits should be preserved"

    def test_strip_garbage_lines(self):
        """Test garbage line removal strips boilerplate correctly."""
        # Sample transcript with garbage lines
        text_with_garbage = """This is the main content.

Read more at example.com
Back to top
Article continues below

More episode content here."""
        cleaned = preprocessing.strip_garbage_lines(text_with_garbage)

        # Verify garbage lines are removed
        assert "Read more" not in cleaned, "Garbage line should be removed"
        assert "Back to top" not in cleaned, "Garbage line should be removed"
        assert "Article continues" not in cleaned, "Garbage line should be removed"

        # Verify main content is preserved
        assert "main content" in cleaned, "Main content should be preserved"
        assert "More episode content" in cleaned, "Content after garbage should be preserved"

    def test_remove_timestamps(self):
        """Test timestamp removal from transcripts."""
        # Sample transcript with timestamps
        text_with_timestamps = """[00:12:34] This is the first segment.

[00:15:45] This is the second segment.

[1:23:45] This is a longer timestamp."""
        cleaned = preprocessing.clean_transcript(
            text_with_timestamps, remove_timestamps=True, normalize_speakers=False
        )

        # Verify timestamps are removed
        assert "[00:12:34]" not in cleaned, "Timestamp should be removed"
        assert "[00:15:45]" not in cleaned, "Timestamp should be removed"
        assert "[1:23:45]" not in cleaned, "Timestamp should be removed"

        # Verify content is preserved
        assert "first segment" in cleaned, "Content should be preserved"
        assert "second segment" in cleaned, "Content should be preserved"

    def test_normalize_speakers(self):
        """Test speaker normalization removes generic speaker tags."""
        # Sample transcript with generic speaker tags
        text_with_speakers = """SPEAKER 1: This is the first speaker.

Speaker 2: This is the second speaker.

Host: This is the host.

John Doe: This is a real person name (should be preserved)."""
        cleaned = preprocessing.clean_transcript(
            text_with_speakers, remove_timestamps=False, normalize_speakers=True
        )

        # Verify generic speaker tags are removed
        assert "SPEAKER 1:" not in cleaned, "Generic speaker tag should be removed"
        assert "Speaker 2:" not in cleaned, "Generic speaker tag should be removed"
        assert "Host:" not in cleaned, "Generic speaker tag should be removed"

        # Verify real names are preserved
        assert "John Doe:" in cleaned, "Real speaker name should be preserved"

        # Verify content is preserved
        assert "first speaker" in cleaned, "Content should be preserved"
        assert "second speaker" in cleaned, "Content should be preserved"

    def test_remove_summarization_artifacts(self):
        """Test artifact removal strips ML-specific artifacts."""
        # Sample text with artifacts
        text_with_artifacts = """This is normal content.

TextColor- This is an artifact.
MUSIC plays in the background.
[INAUDIBLE] was mentioned.
SPEAKER 1: said something."""
        cleaned = preprocessing.remove_summarization_artifacts(text_with_artifacts)

        # Verify artifacts are removed
        assert "TextColor" not in cleaned, "TextColor artifact should be removed"
        assert "MUSIC" not in cleaned, "MUSIC artifact should be removed"
        assert "[INAUDIBLE]" not in cleaned, "INAUDIBLE artifact should be removed"
        assert "SPEAKER 1:" not in cleaned, "SPEAKER artifact should be removed"

        # Verify normal content is preserved
        assert "normal content" in cleaned, "Normal content should be preserved"

    def test_clean_for_summarization_full_pipeline(self):
        """Test complete preprocessing pipeline (clean_for_summarization)."""
        # Sample transcript with all types of issues
        text_with_issues = """[00:12:34] SPEAKER 1: This is the main content.

Produced by John Doe.
Music by Jane Smith.

Read more at example.com
Back to top

TextColor- This is an artifact.
MUSIC plays.

[00:15:45] More content here."""
        cleaned = preprocessing.clean_for_summarization(text_with_issues)

        # Verify all preprocessing steps were applied
        assert "[00:12:34]" not in cleaned, "Timestamps should be removed"
        assert "[00:15:45]" not in cleaned, "Timestamps should be removed"
        assert "SPEAKER 1:" not in cleaned, "Generic speaker tags should be removed"
        assert "Produced by" not in cleaned, "Credits should be removed"
        assert "Music by" not in cleaned, "Credits should be removed"
        assert "Read more" not in cleaned, "Garbage lines should be removed"
        assert "Back to top" not in cleaned, "Garbage lines should be removed"
        assert "TextColor" not in cleaned, "Artifacts should be removed"
        assert "MUSIC" not in cleaned, "Artifacts should be removed"

        # Verify main content is preserved
        assert "main content" in cleaned, "Main content should be preserved"
        assert "More content" in cleaned, "Content should be preserved"

    def test_preprocessing_preserves_real_speaker_names(self):
        """Test that preprocessing preserves real speaker names detected via NER."""
        # Sample transcript with real names (should be preserved)
        text_with_names = """John Doe: This is John speaking.

Jane Smith: This is Jane speaking.

SPEAKER 1: This is a generic speaker (should be removed)."""
        cleaned = preprocessing.clean_transcript(
            text_with_names, remove_timestamps=False, normalize_speakers=True
        )

        # Verify real names are preserved
        assert "John Doe:" in cleaned, "Real speaker name should be preserved"
        assert "Jane Smith:" in cleaned, "Real speaker name should be preserved"

        # Verify generic speaker is removed
        assert "SPEAKER 1:" not in cleaned, "Generic speaker tag should be removed"

        # Verify content is preserved
        assert "John speaking" in cleaned, "Content should be preserved"
        assert "Jane speaking" in cleaned, "Content should be preserved"

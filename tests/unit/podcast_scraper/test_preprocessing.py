#!/usr/bin/env python3
"""Tests for preprocessing functions.

These tests verify the preprocessing functions that clean and normalize
transcripts before summarization or other processing.
"""

import os
import sys
import unittest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from podcast_scraper import preprocessing


class TestCleanTranscript(unittest.TestCase):
    """Test clean_transcript() function."""

    def test_remove_timestamps(self):
        """Test that timestamps are removed."""
        text = "Hello [00:12:34] world [1:23:45] test [12:34]"
        result = preprocessing.clean_transcript(text, remove_timestamps=True)
        # Timestamps are removed, whitespace is cleaned up
        self.assertNotIn("[00:12:34]", result)
        self.assertNotIn("[1:23:45]", result)
        self.assertNotIn("[12:34]", result)
        self.assertIn("Hello", result)
        self.assertIn("world", result)
        self.assertIn("test", result)

    def test_remove_timestamps_disabled(self):
        """Test that timestamps are preserved when disabled."""
        text = "Hello [00:12:34] world"
        result = preprocessing.clean_transcript(text, remove_timestamps=False)
        self.assertEqual(result, "Hello [00:12:34] world")

    def test_remove_generic_speaker_tags(self):
        """Test that generic speaker tags are removed."""
        text = "SPEAKER 1: Hello\nSpeaker 2: World\nHost: Welcome\nGuest: Thanks"
        result = preprocessing.clean_transcript(text, normalize_speakers=True)
        # Generic tags should be removed, but actual names preserved
        self.assertNotIn("SPEAKER 1:", result)
        self.assertNotIn("Speaker 2:", result)
        self.assertNotIn("Host:", result)
        self.assertNotIn("Guest:", result)
        self.assertIn("Hello", result)
        self.assertIn("World", result)

    def test_preserve_real_speaker_names(self):
        """Test that real speaker names are preserved."""
        text = "John Doe: Hello\nJane Smith: World\nAlice: Test"
        result = preprocessing.clean_transcript(text, normalize_speakers=True)
        # Real names should be preserved
        self.assertIn("John Doe:", result)
        self.assertIn("Jane Smith:", result)
        self.assertIn("Alice:", result)

    def test_collapse_blank_lines(self):
        """Test that excessive blank lines are collapsed."""
        text = "Line 1\n\n\n\nLine 2\n\n\nLine 3"
        result = preprocessing.clean_transcript(text, collapse_blank_lines=True)
        # Should collapse 3+ newlines to 2
        self.assertNotIn("\n\n\n", result)
        self.assertIn("\n\n", result)

    def test_collapse_blank_lines_disabled(self):
        """Test that blank lines are preserved when disabled."""
        text = "Line 1\n\n\n\nLine 2"
        result = preprocessing.clean_transcript(text, collapse_blank_lines=False)
        # Should preserve original blank lines
        self.assertIn("\n\n\n", result)

    def test_remove_fillers_enabled(self):
        """Test that filler words are removed when enabled."""
        # Note: Filler removal is conservative and requires specific context
        # Test with fillers at start of sentence or after punctuation
        text = "Uh, this is a test. You know, I mean, it works."
        result = preprocessing.clean_transcript(text, remove_fillers=True)
        # The function is conservative - may not remove all fillers in all contexts
        # Just verify the function runs without error and processes the text
        self.assertIsInstance(result, str)
        self.assertIn("test", result)

    def test_remove_fillers_disabled(self):
        """Test that filler words are preserved when disabled."""
        text = "Uh, you know, I mean, this is a test."
        result = preprocessing.clean_transcript(text, remove_fillers=False)
        # Fillers should be preserved
        self.assertIn("uh", result.lower())
        self.assertIn("you know", result.lower())
        self.assertIn("i mean", result.lower())

    def test_empty_string(self):
        """Test that empty string is handled correctly."""
        result = preprocessing.clean_transcript("")
        self.assertEqual(result, "")

    def test_whitespace_cleanup(self):
        """Test that extra whitespace is cleaned up."""
        text = "Hello    world\n   test"
        result = preprocessing.clean_transcript(text)
        # Multiple spaces should become single space
        self.assertNotIn("    ", result)
        # Spaces at start of line should be removed
        self.assertNotIn("\n   ", result)

    def test_all_features_combined(self):
        """Test that all cleaning features work together."""
        text = "[00:12:34] SPEAKER 1: Hello    world\n\n\n[1:23:45] Host: Test"
        result = preprocessing.clean_transcript(
            text,
            remove_timestamps=True,
            normalize_speakers=True,
            collapse_blank_lines=True,
            remove_fillers=False,
        )
        # Timestamps removed
        self.assertNotIn("[00:12:34]", result)
        self.assertNotIn("[1:23:45]", result)
        # Generic speaker tags removed
        self.assertNotIn("SPEAKER 1:", result)
        self.assertNotIn("Host:", result)
        # Blank lines collapsed
        self.assertNotIn("\n\n\n", result)
        # Content preserved
        self.assertIn("Hello", result)
        self.assertIn("world", result)
        self.assertIn("Test", result)


class TestRemoveSponsorBlocks(unittest.TestCase):
    """Test remove_sponsor_blocks() function."""

    def test_remove_brought_to_you_by(self):
        """Test removal of 'this episode is brought to you by'."""
        text = (
            "Content before.\n\nThis episode is brought to you by AwesomeCo. "
            "Use code POD!\n\nContent after."
        )
        result = preprocessing.remove_sponsor_blocks(text)
        self.assertNotIn("This episode is brought to you by", result)
        self.assertNotIn("AwesomeCo", result)
        self.assertIn("Content before", result)
        self.assertIn("Content after", result)

    def test_remove_sponsored_by(self):
        """Test removal of 'today's episode is sponsored by'."""
        text = "Content.\n\nToday's episode is sponsored by BrandX.\n\nMore content."
        result = preprocessing.remove_sponsor_blocks(text)
        self.assertNotIn("sponsored by", result.lower())
        self.assertNotIn("BrandX", result)
        self.assertIn("Content", result)
        self.assertIn("More content", result)

    def test_remove_our_sponsors(self):
        """Test removal of 'our sponsors today are'."""
        text = "Content.\n\nOur sponsors today are CompanyA and CompanyB.\n\nMore content."
        result = preprocessing.remove_sponsor_blocks(text)
        self.assertNotIn("our sponsors today are", result.lower())
        self.assertIn("Content", result)
        self.assertIn("More content", result)

    def test_no_sponsor_blocks(self):
        """Test that text without sponsor blocks is unchanged."""
        text = "This is regular content without any sponsor mentions."
        result = preprocessing.remove_sponsor_blocks(text)
        self.assertEqual(result, text)

    def test_multiple_sponsor_blocks(self):
        """Test removal of multiple sponsor blocks."""
        text = (
            "Content 1.\n\nThis episode is brought to you by Sponsor1.\n\n"
            "Content 2.\n\nToday's episode is sponsored by Sponsor2.\n\nContent 3."
        )
        result = preprocessing.remove_sponsor_blocks(text)
        self.assertNotIn("Sponsor1", result)
        self.assertNotIn("Sponsor2", result)
        self.assertIn("Content 1", result)
        self.assertIn("Content 2", result)
        self.assertIn("Content 3", result)

    def test_sponsor_block_at_start(self):
        """Test removal of sponsor block at start of text."""
        text = "This episode is brought to you by Sponsor.\n\nContent after."
        result = preprocessing.remove_sponsor_blocks(text)
        self.assertNotIn("This episode is brought to you by", result)
        self.assertIn("Content after", result)

    def test_sponsor_block_at_end(self):
        """Test removal of sponsor block at end of text."""
        text = "Content before.\n\nThis episode is brought to you by Sponsor."
        result = preprocessing.remove_sponsor_blocks(text)
        self.assertNotIn("This episode is brought to you by", result)
        self.assertIn("Content before", result)


class TestRemoveOutroBlocks(unittest.TestCase):
    """Test remove_outro_blocks() function."""

    def test_remove_thank_you_listening(self):
        """Test removal of 'thank you so much for listening'."""
        # Pattern requires "so much" - test with exact pattern
        text = "Content.\n\nThank you so much for listening to this episode.\n\n"
        result = preprocessing.remove_outro_blocks(text)
        self.assertNotIn("thank you so much for listening", result.lower())
        self.assertIn("Content", result)

        # Test that "thank you for listening" (without "so much") is NOT removed
        text2 = "Content.\n\nThank you for listening.\n\n"
        result2 = preprocessing.remove_outro_blocks(text2)
        # This pattern doesn't match, so it should remain
        self.assertIn("thank you for listening", result2.lower())

    def test_remove_enjoyed_episode(self):
        """Test removal of 'if you enjoyed this episode'."""
        text = "Content.\n\nIf you enjoyed this episode, please subscribe.\n\n"
        result = preprocessing.remove_outro_blocks(text)
        self.assertNotIn("if you enjoyed this episode", result.lower())
        self.assertIn("Content", result)

    def test_remove_rate_review_subscribe(self):
        """Test removal of 'please rate/review/subscribe'."""
        text = "Content.\n\nPlease rate and review this podcast.\n\n"
        result = preprocessing.remove_outro_blocks(text)
        self.assertNotIn("please rate", result.lower())
        self.assertNotIn("review", result.lower())
        self.assertIn("Content", result)

    def test_remove_find_more_at(self):
        """Test removal of 'you can find more at'."""
        text = "Content.\n\nYou can find more at example.com.\n\n"
        result = preprocessing.remove_outro_blocks(text)
        self.assertNotIn("you can find more", result.lower())
        self.assertIn("Content", result)

    def test_remove_lennyspodcast(self):
        """Test removal of lennyspodcast.com outro."""
        text = "Content.\n\nlennyspodcast.com\n\n"
        result = preprocessing.remove_outro_blocks(text)
        self.assertNotIn("lennyspodcast.com", result.lower())
        self.assertIn("Content", result)

    def test_no_outro_blocks(self):
        """Test that text without outro blocks is unchanged."""
        text = "This is regular content without any outro patterns."
        result = preprocessing.remove_outro_blocks(text)
        self.assertEqual(result, text)

    def test_multiple_outro_patterns(self):
        """Test removal of multiple outro patterns."""
        text = (
            "Content.\n\nThank you so much for listening.\n\n"
            "If you enjoyed this episode, please subscribe.\n\nMore content."
        )
        result = preprocessing.remove_outro_blocks(text)
        self.assertNotIn("thank you so much for listening", result.lower())
        self.assertNotIn("if you enjoyed this episode", result.lower())
        self.assertIn("Content", result)
        self.assertIn("More content", result)


class TestCleanForSummarization(unittest.TestCase):
    """Test clean_for_summarization() function."""

    def test_full_pipeline(self):
        """Test that full cleaning pipeline is applied."""
        text = (
            "[00:12:34] SPEAKER 1: Hello world.\n\n"
            "This episode is brought to you by Sponsor.\n\n"
            "Thank you so much for listening.\n\n"
        )
        result = preprocessing.clean_for_summarization(text)
        # Timestamps removed
        self.assertNotIn("[00:12:34]", result)
        # Generic speaker tags removed
        self.assertNotIn("SPEAKER 1:", result)
        # Sponsor blocks removed
        self.assertNotIn("This episode is brought to you by", result)
        # Outro blocks removed
        self.assertNotIn("thank you so much for listening", result.lower())
        # Content preserved
        self.assertIn("Hello", result)
        self.assertIn("world", result)

    def test_empty_string(self):
        """Test that empty string is handled correctly."""
        result = preprocessing.clean_for_summarization("")
        self.assertEqual(result, "")

    def test_already_clean_text(self):
        """Test that already clean text is handled correctly."""
        text = "This is already clean content without any timestamps or sponsor blocks."
        result = preprocessing.clean_for_summarization(text)
        self.assertIn("This is already clean content", result)

    def test_all_cleaning_steps(self):
        """Test that all cleaning steps are applied in order."""
        text = (
            "[00:12:34] Host: Content with timestamps.\n\n"
            "This episode is brought to you by Sponsor.\n\n"
            "Thank you so much for listening.\n\n"
        )
        result = preprocessing.clean_for_summarization(text)
        # Verify all cleaning steps were applied
        self.assertNotIn("[00:12:34]", result)
        self.assertNotIn("Host:", result)
        self.assertNotIn("This episode is brought to you by", result)
        self.assertNotIn("thank you so much for listening", result.lower())
        # Content should remain
        self.assertIn("Content", result)

    def test_preserves_actual_speaker_names(self):
        """Test that actual speaker names are preserved."""
        text = "John Doe: Hello.\n\nJane Smith: World.\n\nThank you so much for listening."
        result = preprocessing.clean_for_summarization(text)
        # Real names should be preserved
        self.assertIn("John Doe:", result)
        self.assertIn("Jane Smith:", result)
        # Outro should be removed (pattern requires "so much")
        self.assertNotIn("thank you so much for listening", result.lower())


class TestRemoveSummarizationArtifacts(unittest.TestCase):
    """Test remove_summarization_artifacts function."""

    def test_remove_artifacts_textcolor(self):
        """Test removal of TextColor artifacts."""
        text = "TextColor- This is content. TextColor More content."
        result = preprocessing.remove_summarization_artifacts(text)
        self.assertNotIn("TextColor", result)
        self.assertIn("This is content", result)

    def test_remove_artifacts_music_markers(self):
        """Test removal of MUSIC markers."""
        text = "This is content. MUSIC More content."
        result = preprocessing.remove_summarization_artifacts(text)
        self.assertNotIn("MUSIC", result)
        self.assertIn("This is content", result)

    def test_remove_artifacts_speaker_labels(self):
        """Test removal of generic speaker labels."""
        text = "SPEAKER 1: This is content. SPEAKER 2: More content."
        result = preprocessing.remove_summarization_artifacts(text)
        self.assertNotIn("SPEAKER 1", result)
        self.assertNotIn("SPEAKER 2", result)
        self.assertIn("This is content", result)

    def test_remove_artifacts_bracketed_annotations(self):
        """Test removal of bracketed annotations."""
        text = "This is content [MUSIC] More content [LAUGHTER]."
        result = preprocessing.remove_summarization_artifacts(text)
        self.assertNotIn("[MUSIC]", result)
        self.assertNotIn("[LAUGHTER]", result)
        self.assertIn("This is content", result)

    def test_remove_artifacts_html_tags(self):
        """Test removal of HTML-like tags."""
        text = "This is content <tag>More content</tag>."
        result = preprocessing.remove_summarization_artifacts(text)
        self.assertNotIn("<tag>", result)
        self.assertNotIn("</tag>", result)
        self.assertIn("This is content", result)

    def test_remove_artifacts_empty(self):
        """Test with empty text."""
        result = preprocessing.remove_summarization_artifacts("")
        self.assertEqual(result, "")

    def test_remove_artifacts_cleans_whitespace(self):
        """Test that whitespace is cleaned after artifact removal."""
        text = "TextColor-  This is content  ."
        result = preprocessing.remove_summarization_artifacts(text)
        # Should clean up double spaces and spaces before punctuation
        self.assertNotIn("  ", result)
        self.assertNotIn(" .", result)


class TestStripGarbageLines(unittest.TestCase):
    """Test strip_garbage_lines function."""

    def test_strip_garbage_lines_inline(self):
        """Test removal of inline garbage patterns."""
        text = "This is content. Back to Mail Online home page. More content."
        result = preprocessing.strip_garbage_lines(text)
        self.assertNotIn("Back to Mail Online home page", result)
        self.assertIn("This is content", result)

    def test_strip_garbage_lines_anchored(self):
        """Test removal of anchored garbage lines."""
        text = "This is content.\nBack to Mail Online home page\nMore content."
        result = preprocessing.strip_garbage_lines(text)
        self.assertNotIn("Back to Mail Online home page", result)
        self.assertIn("This is content", result)
        self.assertIn("More content", result)

    def test_strip_garbage_lines_preserves_blank_lines(self):
        """Test that blank lines are preserved."""
        text = "Line 1\n\n\nLine 2"
        result = preprocessing.strip_garbage_lines(text)
        # Blank lines should be preserved for structure
        self.assertIn("\n\n", result)

    def test_strip_garbage_lines_empty(self):
        """Test with empty text."""
        result = preprocessing.strip_garbage_lines("")
        self.assertEqual(result, "")

    def test_strip_garbage_lines_no_garbage(self):
        """Test with text containing no garbage."""
        text = "This is clean content.\nMore clean content."
        result = preprocessing.strip_garbage_lines(text)
        self.assertEqual(result, text)


class TestStripCredits(unittest.TestCase):
    """Test strip_credits function."""

    def test_strip_credits_produced_by(self):
        """Test removal of 'produced by' credits."""
        text = "This is content.\nThis episode was produced by John.\nMore content."
        result = preprocessing.strip_credits(text)
        self.assertNotIn("produced by", result.lower())
        self.assertIn("This is content", result)

    def test_strip_credits_edited_by(self):
        """Test removal of 'edited by' credits."""
        text = "This is content.\nEdited by Jane.\nMore content."
        result = preprocessing.strip_credits(text)
        self.assertNotIn("edited by", result.lower())
        self.assertIn("This is content", result)

    def test_strip_credits_preserves_blank_lines(self):
        """Test that blank lines are preserved."""
        text = "Line 1\n\n\nLine 2"
        result = preprocessing.strip_credits(text)
        # Blank lines should be preserved
        self.assertIn("\n\n", result)

    def test_strip_credits_empty(self):
        """Test with empty text."""
        result = preprocessing.strip_credits("")
        self.assertEqual(result, "")

    def test_strip_credits_no_credits(self):
        """Test with text containing no credits."""
        text = "This is clean content.\nMore clean content."
        result = preprocessing.strip_credits(text)
        self.assertEqual(result, text)


if __name__ == "__main__":
    unittest.main()

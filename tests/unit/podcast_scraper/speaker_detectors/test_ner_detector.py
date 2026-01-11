#!/usr/bin/env python3
"""Unit tests for MLProvider speaker detection (via factory).

These tests verify the NER-based speaker detection provider implementation
using the unified MLProvider returned by the factory.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from parent conftest explicitly to avoid conflicts
import importlib.util
from pathlib import Path

parent_tests_dir = Path(__file__).parent.parent.parent.parent
if str(parent_tests_dir) not in sys.path:
    sys.path.insert(0, str(parent_tests_dir))

parent_conftest_path = parent_tests_dir / "conftest.py"
spec = importlib.util.spec_from_file_location("parent_conftest", parent_conftest_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load conftest from {parent_conftest_path}")
parent_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_conftest)

create_test_config = parent_conftest.create_test_config
create_test_episode = parent_conftest.create_test_episode

from podcast_scraper.speaker_detectors.factory import create_speaker_detector  # noqa: E402


class TestNERSpeakerDetector(unittest.TestCase):
    """Tests for MLProvider speaker detection (via factory)."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(
            auto_speakers=True, ner_model="en_core_web_sm", speaker_detector_provider="spacy"
        )

    def test_init(self):
        """Test MLProvider speaker detection initialization."""
        detector = create_speaker_detector(self.cfg)
        self.assertEqual(detector.cfg, self.cfg)
        self.assertIsNone(detector._spacy_nlp)
        self.assertIsNone(detector._spacy_heuristics)
        self.assertFalse(detector._spacy_initialized)

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    def test_initialize_success(self, mock_get_model):
        """Test successful initialization."""
        mock_nlp = Mock()
        mock_get_model.return_value = mock_nlp

        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        self.assertEqual(detector._spacy_nlp, mock_nlp)
        self.assertTrue(detector._spacy_initialized)
        mock_get_model.assert_called_once_with(self.cfg)

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    def test_initialize_auto_speakers_disabled(self, mock_get_model):
        """Test initialization when auto_speakers is disabled."""
        cfg = create_test_config(auto_speakers=False)
        detector = create_speaker_detector(cfg)
        detector.initialize()

        self.assertIsNone(detector._spacy_nlp)
        self.assertFalse(detector._spacy_initialized)
        mock_get_model.assert_not_called()

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    def test_initialize_already_initialized(self, mock_get_model):
        """Test initialization when already initialized."""
        mock_nlp = Mock()
        mock_get_model.return_value = mock_nlp

        detector = create_speaker_detector(self.cfg)
        detector.initialize()
        mock_get_model.reset_mock()

        # Call again
        detector.initialize()

        # Should not call get_ner_model again
        mock_get_model.assert_not_called()
        self.assertEqual(detector._spacy_nlp, mock_nlp)

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    def test_initialize_model_load_fails(self, mock_get_model):
        """Test initialization when model loading fails.

        MLProvider sets initialized even if model is None.
        """
        mock_get_model.return_value = None

        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        self.assertIsNone(detector._spacy_nlp)
        # MLProvider marks as initialized even if model is None (allows graceful degradation)
        # The actual behavior is checked when detect_speakers is called
        self.assertTrue(detector._spacy_initialized)

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.detect_speaker_names")
    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    def test_detect_speakers(self, mock_get_model, mock_detect):
        """Test detect_speakers method."""
        mock_nlp = Mock()
        mock_get_model.return_value = mock_nlp
        mock_detect.return_value = (["Alice", "Bob"], {"Alice"}, True)

        detector = create_speaker_detector(self.cfg)
        speaker_names, detected_hosts, success = detector.detect_speakers(
            "Episode Title", "Episode Description", {"Alice"}
        )

        self.assertEqual(speaker_names, ["Alice", "Bob"])
        self.assertEqual(detected_hosts, {"Alice"})
        self.assertTrue(success)
        # Now nlp is required parameter (cache removal refactoring)
        mock_detect.assert_called_once_with(
            episode_title="Episode Title",
            episode_description="Episode Description",
            nlp=mock_nlp,  # Required parameter
            cfg=self.cfg,
            known_hosts=None,
            cached_hosts={"Alice"},
            heuristics=None,
        )

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.detect_speaker_names")
    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    def test_detect_speakers_auto_initializes(self, mock_get_model, mock_detect):
        """Test detect_speakers auto-initializes if not initialized."""
        mock_nlp = Mock()
        mock_get_model.return_value = mock_nlp
        mock_detect.return_value = ([], set(), False)

        detector = create_speaker_detector(self.cfg)
        # Don't call initialize() manually
        detector.detect_speakers("Title", "Description", set())

        # Should have initialized
        self.assertTrue(detector._spacy_initialized)
        mock_get_model.assert_called_once()

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.detect_hosts_from_feed")
    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    def test_detect_hosts(self, mock_get_model, mock_detect_hosts):
        """Test detect_hosts method."""
        mock_nlp = Mock()
        mock_get_model.return_value = mock_nlp
        mock_detect_hosts.return_value = {"Host1", "Host2"}

        detector = create_speaker_detector(self.cfg)
        hosts = detector.detect_hosts("Feed Title", "Feed Description", ["Author1"])

        self.assertEqual(hosts, {"Host1", "Host2"})
        mock_detect_hosts.assert_called_once_with(
            feed_title="Feed Title",
            feed_description="Feed Description",
            feed_authors=["Author1"],
            nlp=mock_nlp,
        )

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.analyze_episode_patterns")
    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    def test_analyze_patterns(self, mock_get_model, mock_analyze):
        """Test analyze_patterns method."""
        mock_nlp = Mock()
        mock_get_model.return_value = mock_nlp
        mock_heuristics = {"pattern": "value"}
        mock_analyze.return_value = mock_heuristics

        episodes = [create_test_episode(idx=1, title="Episode 1")]
        detector = create_speaker_detector(self.cfg)
        result = detector.analyze_patterns(episodes, {"Host1"})

        self.assertEqual(result, mock_heuristics)
        self.assertEqual(detector._spacy_heuristics, mock_heuristics)
        mock_analyze.assert_called_once()

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    def test_analyze_patterns_no_nlp(self, mock_get_model):
        """Test analyze_patterns when NLP model is not available."""
        mock_get_model.return_value = None

        detector = create_speaker_detector(self.cfg)
        detector.initialize()
        result = detector.analyze_patterns([], set())

        self.assertIsNone(result)

    def test_nlp_property(self):
        """Test nlp property."""
        detector = create_speaker_detector(self.cfg)
        self.assertIsNone(detector._spacy_nlp)

        mock_nlp = Mock()
        detector._spacy_nlp = mock_nlp
        self.assertEqual(detector._spacy_nlp, mock_nlp)

    def test_heuristics_property(self):
        """Test heuristics property."""
        detector = create_speaker_detector(self.cfg)
        self.assertIsNone(detector._spacy_heuristics)

        mock_heuristics = {"pattern": "value"}
        detector._spacy_heuristics = mock_heuristics
        self.assertEqual(detector._spacy_heuristics, mock_heuristics)

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    def test_cleanup(self, mock_get_model):
        """Test cleanup method."""
        mock_nlp = Mock()
        mock_get_model.return_value = mock_nlp

        detector = create_speaker_detector(self.cfg)
        detector.initialize()
        detector._spacy_heuristics = {"pattern": "value"}

        detector.cleanup()

        self.assertIsNone(detector._spacy_nlp)
        self.assertIsNone(detector._spacy_heuristics)
        self.assertFalse(detector._spacy_initialized)

    def test_cleanup_not_initialized(self):
        """Test cleanup when not initialized."""
        detector = create_speaker_detector(self.cfg)
        detector.cleanup()  # Should not raise

    def test_clear_cache(self):
        """Test clear_cache method (no-op after cache removal)."""
        detector = create_speaker_detector(self.cfg)
        # clear_cache() is now a no-op, should not raise
        detector.clear_cache()

    def test_is_initialized_property(self):
        """Test is_initialized property."""
        detector = create_speaker_detector(self.cfg)
        self.assertFalse(detector._spacy_initialized)

        detector._spacy_nlp = Mock()
        detector._spacy_initialized = True
        self.assertTrue(detector._spacy_initialized)


if __name__ == "__main__":
    unittest.main()

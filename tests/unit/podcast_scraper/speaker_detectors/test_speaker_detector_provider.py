#!/usr/bin/env python3
"""Tests for speaker detector provider (Stage 3).

These tests verify that the speaker detector provider pattern works correctly.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

# Mock ML dependencies before importing modules that require them
# Unit tests run without ML dependencies installed
with patch.dict("sys.modules", {"spacy": MagicMock()}):
    from podcast_scraper import config, models
    from podcast_scraper.speaker_detectors.factory import create_speaker_detector


class TestSpeakerDetectorFactory(unittest.TestCase):
    """Test speaker detector factory."""

    def test_create_ner_detector(self):
        """Test that factory creates NERSpeakerDetector for 'ner'."""
        cfg = config.Config(rss_url="https://example.com/feed.xml", speaker_detector_provider="ner")
        detector = create_speaker_detector(cfg)
        self.assertIsNotNone(detector)
        self.assertEqual(detector.__class__.__name__, "NERSpeakerDetector")

    def test_create_invalid_detector(self):
        """Test that factory raises ValueError for invalid detector type."""
        # Create a config with invalid detector type
        # Note: Config validation should prevent this, but test factory error handling
        with self.assertRaises(ValueError) as context:
            # We can't actually create a config with invalid detector due to validation
            # So we'll test the factory directly with a mock config
            from podcast_scraper.speaker_detectors.factory import create_speaker_detector

            class MockConfig:
                speaker_detector_provider = "invalid"

            create_speaker_detector(MockConfig())  # type: ignore[arg-type]

        self.assertIn("Unsupported speaker detector type", str(context.exception))

    def test_factory_returns_detector_instance(self):
        """Test that factory returns a detector instance."""
        cfg = config.Config(rss_url="https://example.com/feed.xml", speaker_detector_provider="ner")
        detector = create_speaker_detector(cfg)
        # Verify it has the expected methods
        self.assertTrue(hasattr(detector, "detect_speakers"))
        self.assertTrue(hasattr(detector, "analyze_patterns"))


class TestNERSpeakerDetector(unittest.TestCase):
    """Test NERSpeakerDetector implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ner",
            auto_speakers=True,
            ner_model="en_core_web_sm",
        )

    def test_detector_initialization(self):
        """Test that detector can be initialized."""
        from podcast_scraper.speaker_detectors.ner_detector import NERSpeakerDetector

        detector = NERSpeakerDetector(self.cfg)
        self.assertFalse(detector.is_initialized)
        self.assertIsNone(detector.nlp)

    @patch("podcast_scraper.speaker_detectors.ner_detector.speaker_detection.get_ner_model")
    def test_detector_initialize_loads_model(self, mock_get_model):
        """Test that initialize() loads the spaCy model."""
        from podcast_scraper.speaker_detectors.ner_detector import NERSpeakerDetector

        mock_nlp = Mock()
        mock_get_model.return_value = mock_nlp

        detector = NERSpeakerDetector(self.cfg)
        detector.initialize()

        self.assertTrue(detector.is_initialized)
        self.assertEqual(detector.nlp, mock_nlp)
        mock_get_model.assert_called_once_with(self.cfg)

    @patch("podcast_scraper.speaker_detectors.ner_detector.speaker_detection.detect_speaker_names")
    @patch("podcast_scraper.speaker_detectors.ner_detector.speaker_detection.get_ner_model")
    def test_detector_detect_speakers(self, mock_get_model, mock_detect):
        """Test that detect_speakers() calls detect_speaker_names()."""
        from podcast_scraper.speaker_detectors.ner_detector import NERSpeakerDetector

        mock_nlp = Mock()
        mock_get_model.return_value = mock_nlp
        mock_detect.return_value = (["John Doe", "Jane Smith"], {"John Doe"}, True)

        detector = NERSpeakerDetector(self.cfg)
        detector.initialize()

        result = detector.detect_speakers(
            episode_title="Episode with John Doe",
            episode_description="Description",
            known_hosts={"John Doe"},
        )

        self.assertEqual(result[0], ["John Doe", "Jane Smith"])
        self.assertEqual(result[1], {"John Doe"})
        self.assertTrue(result[2])
        mock_detect.assert_called_once()

    @patch(
        "podcast_scraper.speaker_detectors.ner_detector.speaker_detection.detect_hosts_from_feed"
    )
    @patch("podcast_scraper.speaker_detectors.ner_detector.speaker_detection.get_ner_model")
    def test_detector_detect_hosts(self, mock_get_model, mock_detect_hosts):
        """Test that detect_hosts() calls detect_hosts_from_feed()."""
        from podcast_scraper.speaker_detectors.ner_detector import NERSpeakerDetector

        mock_nlp = Mock()
        mock_get_model.return_value = mock_nlp
        mock_detect_hosts.return_value = {"John Doe"}

        detector = NERSpeakerDetector(self.cfg)
        detector.initialize()

        result = detector.detect_hosts(
            feed_title="Podcast Title",
            feed_description="Description",
            feed_authors=None,
        )

        self.assertEqual(result, {"John Doe"})
        mock_detect_hosts.assert_called_once()

    @patch(
        "podcast_scraper.speaker_detectors.ner_detector.speaker_detection.analyze_episode_patterns"
    )
    @patch("podcast_scraper.speaker_detectors.ner_detector.speaker_detection.get_ner_model")
    def test_detector_analyze_patterns(self, mock_get_model, mock_analyze):
        """Test that analyze_patterns() calls analyze_episode_patterns()."""
        from podcast_scraper.speaker_detectors.ner_detector import NERSpeakerDetector

        mock_nlp = Mock()
        mock_get_model.return_value = mock_nlp
        mock_analyze.return_value = {"title_position_preference": "end"}

        detector = NERSpeakerDetector(self.cfg)
        detector.initialize()

        # Create Episode using proper structure
        # Bandit: Safe XML construction for test data only
        # Bandit: Safe XML construction for test data only
        from xml.etree import ElementTree as ET  # nosec B405  # nosec B405

        item = ET.Element("item")
        title_elem = ET.SubElement(item, "title")
        title_elem.text = "Episode 1"
        episodes = [
            models.Episode(
                idx=1,
                title="Episode 1",
                title_safe="episode-1",
                item=item,
                transcript_urls=[],
            )
        ]

        result = detector.analyze_patterns(episodes=episodes, known_hosts={"Host"})

        self.assertIsNotNone(result)
        self.assertEqual(result.get("title_position_preference"), "end")
        # Verify heuristics are cached
        self.assertEqual(detector.heuristics, result)
        mock_analyze.assert_called_once()

    def test_detector_analyze_patterns_no_model(self):
        """Test that analyze_patterns() returns None if model not available."""
        # Bandit: Safe XML construction for test data only
        # Bandit: Safe XML construction for test data only
        from xml.etree import ElementTree as ET  # nosec B405  # nosec B405

        from podcast_scraper.speaker_detectors.ner_detector import NERSpeakerDetector

        # Create detector with auto_speakers disabled
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ner",
            auto_speakers=False,
        )
        detector = NERSpeakerDetector(cfg)
        detector.initialize()  # This won't load a model

        # Create Episode using proper structure
        item = ET.Element("item")
        title_elem = ET.SubElement(item, "title")
        title_elem.text = "Episode 1"
        episodes = [
            models.Episode(
                idx=1,
                title="Episode 1",
                title_safe="episode-1",
                item=item,
                transcript_urls=[],
            )
        ]

        result = detector.analyze_patterns(episodes=episodes, known_hosts=set())

        self.assertIsNone(result)


class TestSpeakerDetectorProtocol(unittest.TestCase):
    """Test that NERSpeakerDetector implements SpeakerDetector protocol."""

    def test_detector_implements_protocol(self):
        """Test that NERSpeakerDetector implements SpeakerDetector protocol."""
        from podcast_scraper.speaker_detectors.ner_detector import NERSpeakerDetector

        cfg = config.Config(rss_url="https://example.com/feed.xml", speaker_detector_provider="ner")
        detector = NERSpeakerDetector(cfg)

        # Check that detector has required protocol methods
        self.assertTrue(hasattr(detector, "detect_speakers"))
        self.assertTrue(hasattr(detector, "analyze_patterns"))
        # Protocol requires detect_speakers(episode_title, episode_description, known_hosts)
        import inspect

        sig = inspect.signature(detector.detect_speakers)
        params = list(sig.parameters.keys())
        self.assertIn("episode_title", params)
        self.assertIn("episode_description", params)
        self.assertIn("known_hosts", params)

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
        """Test that factory creates MLProvider for 'spacy'."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml", speaker_detector_provider="spacy"
        )
        detector = create_speaker_detector(cfg)
        self.assertIsNotNone(detector)
        # Verify it's the unified ML provider
        self.assertEqual(detector.__class__.__name__, "MLProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(detector, "detect_speakers"))
        self.assertTrue(hasattr(detector, "detect_hosts"))
        self.assertTrue(hasattr(detector, "analyze_patterns"))
        self.assertTrue(hasattr(detector, "clear_cache"))

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
        cfg = config.Config(
            rss_url="https://example.com/feed.xml", speaker_detector_provider="spacy"
        )
        detector = create_speaker_detector(cfg)
        # Verify it has the expected methods
        self.assertTrue(hasattr(detector, "detect_speakers"))
        self.assertTrue(hasattr(detector, "analyze_patterns"))


class TestMLProviderSpeakerDetectionViaFactory(unittest.TestCase):
    """Test MLProvider speaker detection capability via factory."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="spacy",
            auto_speakers=True,  # Enable for speaker detection tests
            ner_model=config.DEFAULT_NER_MODEL,
        )

    def test_detector_creation_via_factory(self):
        """Test that detector can be created via factory."""
        detector = create_speaker_detector(self.cfg)
        self.assertIsNotNone(detector)
        # Verify it's the unified ML provider
        self.assertEqual(detector.__class__.__name__, "MLProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(detector, "detect_speakers"))
        self.assertTrue(hasattr(detector, "detect_hosts"))
        self.assertTrue(hasattr(detector, "analyze_patterns"))
        self.assertTrue(hasattr(detector, "clear_cache"))

    def test_detector_initialization_state(self):
        """Test that detector tracks initialization state."""
        detector = create_speaker_detector(self.cfg)
        # Initially not initialized (needs initialize() call)
        self.assertFalse(detector._spacy_initialized)

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    def test_detector_initialize_loads_model(self, mock_get_model):
        """Test that initialize() loads the spaCy model via factory."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            speaker_detector_provider=self.cfg.speaker_detector_provider,
            auto_speakers=True,
        )

        mock_nlp = Mock()
        mock_get_model.return_value = mock_nlp

        detector = create_speaker_detector(cfg)
        detector.initialize()

        self.assertTrue(detector._spacy_initialized)
        self.assertEqual(detector.nlp, mock_nlp)
        mock_get_model.assert_called_once()

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.detect_speaker_names")
    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    def test_detector_detect_speakers(self, mock_get_model, mock_detect):
        """Test that detect_speakers() calls detect_speaker_names() via factory."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            speaker_detector_provider=self.cfg.speaker_detector_provider,
            auto_speakers=True,
        )

        mock_nlp = Mock()
        mock_get_model.return_value = mock_nlp
        mock_detect.return_value = (["John Doe", "Jane Smith"], {"John Doe"}, True)

        detector = create_speaker_detector(cfg)
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

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.detect_hosts_from_feed")
    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    def test_detector_detect_hosts(self, mock_get_model, mock_detect_hosts):
        """Test that detect_hosts() calls detect_hosts_from_feed()."""
        # Use factory instead of direct import

        mock_nlp = Mock()
        mock_get_model.return_value = mock_nlp
        mock_detect_hosts.return_value = {"John Doe"}

        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        result = detector.detect_hosts(
            feed_title="Podcast Title",
            feed_description="Description",
            feed_authors=None,
        )

        self.assertEqual(result, {"John Doe"})
        mock_detect_hosts.assert_called_once()

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.analyze_episode_patterns")
    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    def test_detector_analyze_patterns(self, mock_get_model, mock_analyze):
        """Test that analyze_patterns() calls analyze_episode_patterns()."""
        # Use factory instead of direct import

        mock_nlp = Mock()
        mock_get_model.return_value = mock_nlp
        mock_analyze.return_value = {"title_position_preference": "end"}

        detector = create_speaker_detector(self.cfg)
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
        """Test that analyze_patterns() raises RuntimeError if auto_speakers is False."""
        # Bandit: Safe XML construction for test data only
        # Bandit: Safe XML construction for test data only
        from xml.etree import ElementTree as ET  # nosec B405  # nosec B405

        # Use factory instead of direct import
        # Create detector with auto_speakers disabled
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="spacy",
            auto_speakers=False,
        )
        detector = create_speaker_detector(cfg)

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

        with self.assertRaises(RuntimeError) as context:
            detector.analyze_patterns(episodes=episodes, known_hosts=set())

        self.assertIn("auto_speakers is False", str(context.exception))


class TestSpeakerDetectorProtocol(unittest.TestCase):
    """Test that MLProvider implements SpeakerDetector protocol (via factory)."""

    def test_detector_implements_protocol(self):
        """Test that MLProvider implements SpeakerDetector protocol."""

        cfg = config.Config(
            rss_url="https://example.com/feed.xml", speaker_detector_provider="spacy"
        )
        detector = create_speaker_detector(cfg)

        # Check that detector has required protocol methods
        self.assertTrue(hasattr(detector, "detect_speakers"))
        self.assertTrue(hasattr(detector, "detect_hosts"))
        self.assertTrue(hasattr(detector, "analyze_patterns"))
        self.assertTrue(hasattr(detector, "clear_cache"))

        # Protocol requires detect_speakers(episode_title, episode_description, known_hosts)
        import inspect

        sig = inspect.signature(detector.detect_speakers)
        params = list(sig.parameters.keys())
        self.assertIn("episode_title", params)
        self.assertIn("episode_description", params)
        self.assertIn("known_hosts", params)

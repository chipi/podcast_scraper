#!/usr/bin/env python3
"""Integration tests for providers with real ML models.

These tests verify that providers can load and use real ML models:
- Whisper transcription models (tiny model for speed)
- spaCy NER models (en_core_web_sm for speaker detection)
- Transformer summarization models (small models like bart-base or distilbart)

These tests are marked with @pytest.mark.ml_models
because they require:
- ML dependencies installed (openai-whisper, spacy, transformers, torch)
- Real model downloads (first run only, then cached)
- Longer execution time (model loading and inference)

Note: These tests use the smallest/fastest models available to keep execution time reasonable.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import models, whisper_integration

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Import cache helpers from same directory
import sys
from pathlib import Path

from conftest import create_test_config  # noqa: E402

integration_dir = Path(__file__).parent
if str(integration_dir) not in sys.path:
    sys.path.insert(0, str(integration_dir))
from ml_model_cache_helpers import (  # noqa: E402
    require_spacy_model_cached,
    require_transformers_model_cached,
    require_whisper_model_cached,
)

# Check if ML dependencies are available
WHISPER_AVAILABLE = False
SPACY_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

try:
    import whisper  # noqa: F401

    WHISPER_AVAILABLE = True
except ImportError:
    pass

try:
    import spacy  # noqa: F401

    SPACY_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: F401

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


@pytest.mark.integration
@pytest.mark.ml_models
@unittest.skipIf(not WHISPER_AVAILABLE, "Whisper dependencies not available")
class TestWhisperProviderRealModel(unittest.TestCase):
    """Test Whisper provider with real model loading."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(
            transcribe_missing=True,
            whisper_model="tiny",  # Smallest model for speed
            language="en",
        )

    def test_whisper_model_loading(self):
        """Test that Whisper model can be loaded."""
        # Require model to be cached (fail fast if not)
        # Note: cfg.whisper_model defaults to "tiny.en" (from test defaults)
        # Check for "tiny.en" which is what's preloaded
        require_whisper_model_cached("tiny.en")

        # Load real Whisper model
        model = whisper_integration.load_whisper_model(self.cfg)

        # Verify model was loaded
        self.assertIsNotNone(model, "Whisper model should be loaded")

        # Verify model has expected attributes
        self.assertTrue(hasattr(model, "device"), "Model should have device attribute")
        self.assertTrue(hasattr(model, "transcribe"), "Model should have transcribe method")

    def test_whisper_provider_with_real_model(self):
        """Test Whisper provider initialization with real model."""
        # Require model to be cached (fail fast if not)
        require_whisper_model_cached("tiny.en")

        from podcast_scraper.transcription.factory import create_transcription_provider

        # Create provider (real factory)
        provider = create_transcription_provider(self.cfg)

        # Initialize provider (loads real model)
        provider.initialize()  # type: ignore[attr-defined]

        # Verify provider is initialized
        self.assertTrue(provider._initialized)  # type: ignore[attr-defined]
        self.assertIsNotNone(provider._model)  # type: ignore[attr-defined]

        # Verify model is actually loaded (not mocked)
        model = provider._model  # type: ignore[attr-defined]
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "transcribe"))


@pytest.mark.integration
@pytest.mark.ml_models
@unittest.skipIf(not SPACY_AVAILABLE, "spaCy dependencies not available")
class TestSpacyProviderRealModel(unittest.TestCase):
    """Test spaCy NER provider with real model loading."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(
            auto_speakers=True,
            ner_model="en_core_web_sm",  # Smallest spaCy model
            language="en",
        )

    def test_spacy_model_loading(self):
        """Test that spaCy model can be loaded."""
        # Require model to be cached (fail fast if not)
        require_spacy_model_cached("en_core_web_sm")

        from podcast_scraper import speaker_detection

        # Load real spaCy model
        nlp = speaker_detection.get_ner_model(self.cfg)

        # Verify model was loaded
        self.assertIsNotNone(nlp, "spaCy model should be loaded")

        # Verify model has expected attributes
        self.assertTrue(hasattr(nlp, "pipe"), "Model should have pipe method")

    def test_ner_detector_with_real_model(self):
        """Test NER detector with real spaCy model."""
        # Require model to be cached (fail fast if not)
        require_spacy_model_cached("en_core_web_sm")

        from podcast_scraper.speaker_detectors.factory import create_speaker_detector

        # Create detector (real factory)
        detector = create_speaker_detector(self.cfg)

        # Initialize detector (loads real model)
        detector.initialize()  # type: ignore[attr-defined]

        # Verify detector is initialized
        self.assertTrue(hasattr(detector, "_nlp"))
        self.assertIsNotNone(detector._nlp)  # type: ignore[attr-defined]

        # Test that detector can actually use the model
        # (detect_speakers uses the model internally)
        result = detector.detect_speakers(  # type: ignore[attr-defined]
            episode_title="Test Episode",
            episode_description="This is a test episode with John Smith and Jane Doe.",
            known_hosts={"John Smith"},
        )

        # Verify result is valid
        # detect_speakers returns Tuple[list[str], Set[str], bool]
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        speakers, hosts, success = result
        self.assertIsInstance(speakers, list)
        self.assertIsInstance(hosts, set)
        self.assertIsInstance(success, bool)


@pytest.mark.integration
@pytest.mark.ml_models
@unittest.skipIf(not TRANSFORMERS_AVAILABLE, "Transformers dependencies not available")
class TestTransformersProviderRealModel(unittest.TestCase):
    """Test Transformers summarization provider with real model loading."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries=True
            summary_model="facebook/bart-base",  # Small model for speed
            summary_device="cpu",  # Use CPU to avoid GPU requirements
            language="en",
        )

    def test_transformers_model_loading(self):
        """Test that Transformers model can be loaded."""
        from podcast_scraper import summarizer

        # Load real transformer model
        model_name = summarizer.select_summary_model(self.cfg)

        # Require model to be cached (fail fast if not)
        require_transformers_model_cached(model_name, None)

        model = summarizer.SummaryModel(
            model_name=model_name,
            device="cpu",
            cache_dir=None,
        )

        # Verify model was loaded
        self.assertIsNotNone(model, "Transformer model should be loaded")
        self.assertIsNotNone(model.model, "Model should have model attribute")
        self.assertIsNotNone(model.tokenizer, "Model should have tokenizer attribute")
        self.assertIsNotNone(model.pipeline, "Model should have pipeline attribute")

    def test_summarization_provider_with_real_model(self):
        """Test summarization provider with real transformer model."""
        from podcast_scraper import summarizer
        from podcast_scraper.summarization.factory import create_summarization_provider

        # Require model to be cached (fail fast if not)
        model_name = summarizer.select_summary_model(self.cfg)
        require_transformers_model_cached(model_name, None)

        # Create provider (real factory)
        provider = create_summarization_provider(self.cfg)

        # Initialize provider (loads real model)
        provider.initialize()  # type: ignore[attr-defined]

        # Verify provider is initialized
        self.assertTrue(provider._initialized)  # type: ignore[attr-defined]
        self.assertIsNotNone(provider._map_model)  # type: ignore[attr-defined]

        # Verify model is actually loaded (not mocked)
        map_model = provider._map_model  # type: ignore[attr-defined]
        self.assertIsNotNone(map_model)
        self.assertIsNotNone(map_model.pipeline)  # type: ignore[attr-defined]

        # Test that provider can actually use the model
        # (summarize uses the model internally)
        test_text = (
            "This is a test transcript. It contains multiple sentences. "
            "The purpose is to test summarization. We want to verify the model works."
        )
        result = provider.summarize(  # type: ignore[attr-defined]
            text=test_text,
            episode_title="Test Episode",
            episode_description="A test episode",
            params=None,
        )

        # Verify result is valid
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        # Result should contain summary
        self.assertIn("summary", result)
        self.assertIsInstance(result["summary"], str)
        self.assertGreater(len(result["summary"]), 0)


@pytest.mark.integration
@pytest.mark.ml_models
@unittest.skipIf(
    not (WHISPER_AVAILABLE and SPACY_AVAILABLE and TRANSFORMERS_AVAILABLE),
    "Not all ML dependencies available",
)
class TestAllProvidersRealModels(unittest.TestCase):
    """Test all providers together with real models."""

    def setUp(self):
        """Set up test fixtures."""
        import tempfile

        self.temp_dir = tempfile.mkdtemp()
        self.cfg = create_test_config(
            output_dir=self.temp_dir,
            transcribe_missing=True,
            whisper_model="tiny",  # Smallest Whisper model
            auto_speakers=True,
            ner_model="en_core_web_sm",  # Smallest spaCy model
            generate_summaries=True,
            generate_metadata=True,
            summary_model="facebook/bart-base",  # Small transformer model
            summary_device="cpu",
            language="en",
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_all_providers_initialize_with_real_models(self):
        """Test that all providers can be initialized with real models."""
        from podcast_scraper import summarizer
        from podcast_scraper.speaker_detectors.factory import create_speaker_detector
        from podcast_scraper.summarization.factory import create_summarization_provider
        from podcast_scraper.transcription.factory import create_transcription_provider

        # Require all models to be cached (fail fast if not)
        require_whisper_model_cached("tiny.en")
        require_spacy_model_cached("en_core_web_sm")
        model_name = summarizer.select_summary_model(self.cfg)
        require_transformers_model_cached(model_name, None)

        # Create all providers (real factories)
        transcription_provider = create_transcription_provider(self.cfg)
        speaker_detector = create_speaker_detector(self.cfg)
        summarization_provider = create_summarization_provider(self.cfg)

        # Initialize all providers (loads real models)
        transcription_provider.initialize()  # type: ignore[attr-defined]
        speaker_detector.initialize()  # type: ignore[attr-defined]
        summarization_provider.initialize()  # type: ignore[attr-defined]

        # Verify all providers are initialized with real models
        self.assertTrue(transcription_provider._initialized)  # type: ignore[attr-defined]
        self.assertIsNotNone(transcription_provider._model)  # type: ignore[attr-defined]

        self.assertIsNotNone(speaker_detector._nlp)  # type: ignore[attr-defined]

        self.assertTrue(summarization_provider._initialized)  # type: ignore[attr-defined]
        self.assertIsNotNone(summarization_provider._map_model)  # type: ignore[attr-defined]

        # Verify models are actually loaded (not mocked)
        # Whisper model
        whisper_model = transcription_provider._model  # type: ignore[attr-defined]
        self.assertTrue(hasattr(whisper_model, "transcribe"))

        # spaCy model
        spacy_model = speaker_detector._nlp  # type: ignore[attr-defined]
        self.assertTrue(hasattr(spacy_model, "pipe"))

        # Transformer model
        transformer_model = summarization_provider._map_model  # type: ignore[attr-defined]
        self.assertIsNotNone(transformer_model.pipeline)  # type: ignore[attr-defined]

    @pytest.mark.critical_path
    def test_critical_path_with_real_models(self):
        """Test critical path (Full Workflow) with real cached ML models: RSS → Parse → Download/Transcribe → NER → Summarization → Metadata → Files.

        This test validates the COMPLETE critical path with all core features using REAL cached ML models:
        - RSS feed parsing
        - Transcript download (from test fixtures)
        - NER speaker detection (hosts and guests) - REAL spaCy model
        - Summarization - REAL Transformers model
        - Metadata generation with all features
        - File output

        This is Priority 2 from Critical Path Testing Guide: "Critical Path with Real Models (Should Have)"
        - Validates actual ML models work in the critical path
        - Uses cached models (checked before test runs)
        - Runs in fast test suite if models are cached (marked critical_path, not slow)
        - Skips if models are not cached (to avoid network downloads)
        """
        import os
        from pathlib import Path

        from podcast_scraper import metadata, rss_parser, summarizer

        # Require all models to be cached (skip if not, to avoid network downloads)
        try:
            require_whisper_model_cached("tiny.en")
            require_spacy_model_cached("en_core_web_sm")
            model_name = summarizer.select_summary_model(self.cfg)
            require_transformers_model_cached(model_name, None)
        except AssertionError as e:
            pytest.skip(f"Models not cached: {e}")

        # Get transcript from fixtures
        fixture_root = Path(__file__).parent.parent / "fixtures"
        transcript_file = fixture_root / "transcripts" / "p01_e01_fast.txt"

        if not transcript_file.exists():
            self.skipTest(f"Transcript file not found: {transcript_file}")

        transcript_text = transcript_file.read_text(encoding="utf-8")

        # Create minimal RSS feed
        rss_xml = """<?xml version='1.0'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>Test Feed</title>
    <author>John Host</author>
    <itunes:author>Jane Host</itunes:author>
    <item>
      <title>Episode 1: Interview with Bob Guest</title>
      <description>In this episode, we talk with Bob Guest about technology and software development.</description>
      <enclosure url="https://example.com/ep1.mp3" type="audio/mpeg" />
    </item>
  </channel>
</rss>""".strip()

        # Parse RSS feed
        title, authors, items = rss_parser.parse_rss_items(rss_xml.encode("utf-8"))
        feed = models.RssFeed(
            title=title,
            authors=authors,
            items=items,
            base_url="https://example.com/feed.xml",
        )
        episodes = [
            rss_parser.create_episode_from_item(item, idx, feed.base_url)
            for idx, item in enumerate(feed.items, start=1)
        ]
        episode = episodes[0]

        # Create transcript file
        transcript_path = os.path.join(self.temp_dir, "0001 - Episode_1.txt")
        os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        # Step 1: Detect hosts from feed metadata (REAL NER model)
        from podcast_scraper.speaker_detectors.factory import create_speaker_detector

        speaker_detector = create_speaker_detector(self.cfg)
        speaker_detector.initialize()
        detected_hosts = speaker_detector.detect_hosts(
            feed_title=feed.title,
            feed_description=None,
            feed_authors=feed.authors,
        )

        # Verify hosts were detected (may or may not succeed depending on model, but should not crash)
        self.assertIsInstance(detected_hosts, set)

        # Step 2: Detect guests from episode metadata (REAL NER model)
        episode_description = rss_parser.extract_episode_description(episode.item)
        detected_speakers, detected_hosts_set, detection_succeeded = (
            speaker_detector.detect_speakers(
                episode_title=episode.title,
                episode_description=episode_description,
                known_hosts=detected_hosts,
            )
        )

        # Verify speaker detection ran (may or may not detect, but should not crash)
        self.assertIsInstance(detected_speakers, list)
        self.assertIsInstance(detected_hosts_set, set)
        self.assertIsInstance(detection_succeeded, bool)

        # Step 3: Summarize transcript (REAL Transformers model)
        from podcast_scraper.summarization.factory import create_summarization_provider

        summarization_provider = create_summarization_provider(self.cfg)
        try:
            summarization_provider.initialize()
        except Exception as e:
            # If initialization fails due to network access (missing tokenizer files),
            # skip the test with a helpful message
            error_str = str(e)
            if (
                "socket" in error_str.lower()
                or "connect" in error_str.lower()
                or "network" in error_str.lower()
            ):
                pytest.skip(
                    f"Tokenizer files not fully cached (network access blocked): {e}. "
                    f"Run 'make preload-ml-models' to ensure all model files are cached."
                )
            raise

        # Use first 1000 chars for speed (real models are slow)
        summary_result = summarization_provider.summarize(
            text=transcript_text[:1000],
            episode_title=episode.title,
            episode_description=episode_description,
            params=None,
        )

        # Verify summary was generated
        self.assertIsInstance(summary_result, dict)
        self.assertIn("summary", summary_result)
        self.assertIsInstance(summary_result["summary"], str)
        self.assertGreater(len(summary_result["summary"]), 0)

        # Step 4: Generate metadata with NER and summarization
        metadata_path = metadata.generate_episode_metadata(
            feed=feed,
            episode=episode,
            feed_url="https://example.com/feed.xml",
            cfg=self.cfg,
            output_dir=self.temp_dir,
            run_suffix=None,
            transcript_file_path="0001 - Episode_1.txt",
            transcript_source="direct_download",
            whisper_model=None,
            detected_hosts=list(detected_hosts),
            detected_guests=[s for s in detected_speakers if s not in detected_hosts],
            summary_provider=summarization_provider,
        )

        # Verify metadata file was created
        self.assertIsNotNone(metadata_path)
        self.assertTrue(os.path.exists(metadata_path))

        # Verify metadata includes all features
        import json

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("detected_hosts", data["content"])
        self.assertIn("detected_guests", data["content"])
        # Summary is stored at top-level, not in content
        self.assertIn("summary", data)
        self.assertIsNotNone(data["summary"], "Summary should be generated")
        self.assertEqual(data["content"]["transcript_source"], "direct_download")

        # Cleanup
        speaker_detector.cleanup()
        summarization_provider.cleanup()


@pytest.mark.integration
@pytest.mark.critical_path
@pytest.mark.openai
class TestCriticalPathWithOpenAIProviders(unittest.TestCase):
    """Test critical path (Full Workflow) with OpenAI providers: RSS → Parse → Download/Transcribe → OpenAI Speaker Detection → OpenAI Summarization → Metadata → Files.

    This test validates the COMPLETE critical path with all core features using OpenAI providers:
    - RSS feed parsing
    - Transcript download (from test fixtures)
    - OpenAI speaker detection (hosts and guests) - Mocked API calls
    - OpenAI summarization - Mocked API calls
    - Metadata generation with all features
    - File output

    This is Priority 1 from Critical Path Testing Guide: "Critical Path (Must Have)"
    - Validates OpenAI providers work in the critical path
    - Uses mocked OpenAI API (no real API calls, fast execution)
    - Runs in fast test suite (marked critical_path, not slow)
    """

    def setUp(self):
        """Set up test fixtures."""
        import tempfile

        self.temp_dir = tempfile.mkdtemp()
        self.cfg = create_test_config(
            output_dir=self.temp_dir,
            transcription_provider="openai",
            speaker_detector_provider="openai",
            summary_provider="openai",
            openai_api_key="sk-test123",
            generate_summaries=True,
            generate_metadata=True,
            auto_speakers=True,
            screenplay_num_speakers=3,  # Allow 3 speakers so Bob Guest is included
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("podcast_scraper.transcription.openai_provider.OpenAI")
    @patch("podcast_scraper.speaker_detectors.openai_detector.OpenAI")
    @patch("podcast_scraper.summarization.openai_provider.OpenAI")
    @patch("podcast_scraper.prompt_store.render_prompt")
    def test_critical_path_with_openai_providers(
        self,
        mock_render_prompt,
        mock_summary_openai,
        mock_speaker_openai,
        mock_transcription_openai,
    ):
        """Test critical path with OpenAI providers (mocked API calls)."""
        import os
        from pathlib import Path

        from podcast_scraper import metadata, models, rss_parser

        # Get transcript from fixtures
        fixture_root = Path(__file__).parent.parent / "fixtures"
        transcript_file = fixture_root / "transcripts" / "p01_e01_fast.txt"

        if not transcript_file.exists():
            self.skipTest(f"Transcript file not found: {transcript_file}")

        transcript_text = transcript_file.read_text(encoding="utf-8")

        # Create minimal RSS feed
        rss_xml = """<?xml version='1.0'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>Test Feed</title>
    <author>John Host</author>
    <itunes:author>Jane Host</itunes:author>
    <item>
      <title>Episode 1: Interview with Bob Guest</title>
      <description>In this episode, we talk with Bob Guest about technology and software development.</description>
      <enclosure url="https://example.com/ep1.mp3" type="audio/mpeg" />
    </item>
  </channel>
</rss>""".strip()

        # Parse RSS feed
        title, authors, items = rss_parser.parse_rss_items(rss_xml.encode("utf-8"))
        feed = models.RssFeed(
            title=title,
            authors=authors,
            items=items,
            base_url="https://example.com/feed.xml",
        )
        episodes = [
            rss_parser.create_episode_from_item(item, idx, feed.base_url)
            for idx, item in enumerate(feed.items, start=1)
        ]
        episode = episodes[0]

        # Create transcript file
        transcript_path = os.path.join(self.temp_dir, "0001 - Episode_1.txt")
        os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        # Mock OpenAI clients
        mock_transcription_client = Mock()
        mock_transcription_openai.return_value = mock_transcription_client

        mock_speaker_client = Mock()
        mock_speaker_openai.return_value = mock_speaker_client

        mock_summary_client = Mock()
        mock_summary_openai.return_value = mock_summary_client

        # Mock prompt rendering for speaker detection and summarization
        mock_render_prompt.side_effect = [
            "System prompt for speaker detection",
            "User prompt with episode metadata",
            "System prompt for summarization",
            "User prompt with transcript",
        ]

        # Mock OpenAI speaker detection response
        # Note: detect_hosts uses feed_authors directly, so it won't call the API
        # detect_speakers will be called and needs the full response format
        mock_speaker_response = Mock()
        mock_speaker_response.choices = [
            Mock(
                message=Mock(
                    content='{"speakers": ["John Host", "Jane Host", "Bob Guest"], "hosts": ["John Host", "Jane Host"], "guests": ["Bob Guest"]}'
                )
            )
        ]
        mock_speaker_client.chat.completions.create.return_value = mock_speaker_response

        # Mock OpenAI summarization response
        mock_summary_response = Mock()
        mock_summary_response.choices = [
            Mock(
                message=Mock(
                    content="This is a test summary of the episode discussing technology and software development."
                )
            )
        ]
        mock_summary_client.chat.completions.create.return_value = mock_summary_response

        # Step 1: Detect hosts from feed metadata (OpenAI)
        from podcast_scraper.speaker_detectors.factory import create_speaker_detector

        speaker_detector = create_speaker_detector(self.cfg)
        speaker_detector.initialize()
        detected_hosts = speaker_detector.detect_hosts(
            feed_title=feed.title,
            feed_description=None,
            feed_authors=feed.authors,
        )

        # Verify hosts were detected
        self.assertIsInstance(detected_hosts, set)
        # OpenAI should detect hosts from the mocked response
        self.assertIn("John Host", detected_hosts)
        self.assertIn("Jane Host", detected_hosts)

        # Step 2: Detect guests from episode metadata (OpenAI)
        episode_description = rss_parser.extract_episode_description(episode.item)
        detected_speakers, detected_hosts_set, detection_succeeded = (
            speaker_detector.detect_speakers(
                episode_title=episode.title,
                episode_description=episode_description,
                known_hosts=detected_hosts,
            )
        )

        # Verify speaker detection ran
        self.assertIsInstance(detected_speakers, list)
        self.assertIsInstance(detected_hosts_set, set)
        self.assertIsInstance(detection_succeeded, bool)
        # OpenAI should detect "Bob Guest" from the mocked response
        self.assertIn("Bob Guest", detected_speakers)

        # Step 3: Summarize transcript (OpenAI)
        from podcast_scraper.summarization.factory import create_summarization_provider

        summarization_provider = create_summarization_provider(self.cfg)
        summarization_provider.initialize()

        # Mock the provider's summarize method directly
        with patch.object(
            summarization_provider,
            "summarize",
            return_value={
                "summary": "This is a test summary of the episode discussing technology and software development.",
                "summary_short": None,
                "metadata": {
                    "provider": "openai",
                    "model_used": "gpt-4o-mini",
                },
            },
        ):
            # Step 4: Generate metadata with OpenAI speaker detection and summarization
            metadata_path = metadata.generate_episode_metadata(
                feed=feed,
                episode=episode,
                feed_url="https://example.com/feed.xml",
                cfg=self.cfg,
                output_dir=self.temp_dir,
                run_suffix=None,
                transcript_file_path="0001 - Episode_1.txt",
                transcript_source="direct_download",
                whisper_model=None,
                detected_hosts=list(detected_hosts),
                detected_guests=[s for s in detected_speakers if s not in detected_hosts],
                summary_provider=summarization_provider,
            )

            # Verify metadata file was created
            self.assertIsNotNone(metadata_path)
            self.assertTrue(os.path.exists(metadata_path))

            # Verify metadata includes all features
            import json

            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertIn("detected_hosts", data["content"])
            self.assertIn("detected_guests", data["content"])
            # Summary is stored in a separate "summary" field, not in "content"
            self.assertIn("summary", data)
            self.assertIsNotNone(data["summary"], "Summary should be generated")
            self.assertIn("short_summary", data["summary"])
            self.assertIsNotNone(
                data["summary"]["short_summary"], "Summary text should be generated"
            )
            self.assertEqual(data["content"]["transcript_source"], "direct_download")

        # Verify OpenAI API was called
        self.assertTrue(mock_speaker_client.chat.completions.create.called)

        # Cleanup
        speaker_detector.cleanup()
        summarization_provider.cleanup()


if __name__ == "__main__":
    unittest.main()

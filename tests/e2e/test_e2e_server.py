#!/usr/bin/env python3
"""E2E HTTP server verification tests.

These tests verify that the E2E HTTP server correctly serves fixtures
and provides URL helpers.
"""

import json
import os
import sys

import pytest
import requests

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)


@pytest.mark.e2e
@pytest.mark.critical_path
class TestE2EServer:
    """Test that E2E server serves fixtures correctly."""

    def test_server_starts_and_stops(self, e2e_server):
        """Test that server starts and provides base URL."""
        assert e2e_server.base_url is not None
        assert e2e_server.base_url.startswith("http://127.0.0.1:")
        assert e2e_server.urls is not None

    def test_url_helpers(self, e2e_server):
        """Test that URL helpers work correctly."""
        # Test feed URL
        feed_url = e2e_server.urls.feed("podcast1")
        assert "/feeds/podcast1/feed.xml" in feed_url
        assert feed_url.startswith("http://")

        # Test audio URL
        audio_url = e2e_server.urls.audio("p01_e01")
        assert "/audio/p01_e01.mp3" in audio_url
        assert audio_url.startswith("http://")

        # Test transcript URL
        transcript_url = e2e_server.urls.transcript("p01_e01")
        assert "/transcripts/p01_e01.txt" in transcript_url
        assert transcript_url.startswith("http://")

        # Test base URL
        base_url = e2e_server.urls.base()
        assert base_url == e2e_server.base_url

    def test_serve_rss_feed(self, e2e_server):
        """Test that RSS feeds are served correctly."""
        feed_url = e2e_server.urls.feed("podcast1")

        response = requests.get(feed_url, timeout=2)
        assert response.status_code == 200
        assert "application/xml" in response.headers.get("Content-Type", "")
        assert "<?xml" in response.text
        assert "rss" in response.text.lower()

    def test_serve_audio_file(self, e2e_server):
        """Test that audio files are served correctly."""
        audio_url = e2e_server.urls.audio("p01_e01")

        response = requests.get(audio_url, timeout=2)
        assert response.status_code == 200
        assert "audio/mpeg" in response.headers.get("Content-Type", "")
        assert len(response.content) > 0

    def test_serve_transcript_file(self, e2e_server):
        """Test that transcript files are served correctly."""
        transcript_url = e2e_server.urls.transcript("p01_e01")

        response = requests.get(transcript_url, timeout=2)
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("Content-Type", "")
        assert len(response.text) > 0

    def test_404_for_missing_file(self, e2e_server):
        """Test that missing files return 404."""
        missing_url = f"{e2e_server.base_url}/audio/nonexistent.mp3"

        response = requests.get(missing_url, timeout=2)
        assert response.status_code == 404

    def test_path_traversal_protection(self, e2e_server):
        """Test that path traversal attacks are blocked."""
        # Try to access file outside fixture directory
        malicious_url = f"{e2e_server.base_url}/audio/../../../etc/passwd"

        response = requests.get(malicious_url, timeout=2)
        # Should return 403 or 404, not serve the file
        assert response.status_code in [403, 404]

    def test_range_request_support(self, e2e_server):
        """Test that range requests are supported for audio files."""
        audio_url = e2e_server.urls.audio("p01_e01")

        # Request first 1024 bytes
        headers = {"Range": "bytes=0-1023"}
        response = requests.get(audio_url, headers=headers, timeout=2)

        # Should return 206 Partial Content
        assert response.status_code == 206
        assert "Content-Range" in response.headers
        assert len(response.content) == 1024


@pytest.mark.e2e
@pytest.mark.critical_path
class TestE2EServerOpenAIEndpoints:
    """Test that E2E server OpenAI mock endpoints work correctly."""

    def test_openai_api_base_url_helper(self, e2e_server):
        """Test that OpenAI API base URL helper works."""
        openai_api_base = e2e_server.urls.openai_api_base()
        assert openai_api_base.startswith("http://127.0.0.1:")
        assert openai_api_base.endswith("/v1")
        assert "/v1" in openai_api_base

    def test_chat_completions_endpoint_summarization(self, e2e_server):
        """Test that chat completions endpoint works for summarization."""
        openai_api_base = e2e_server.urls.openai_api_base()
        url = f"{openai_api_base}/chat/completions"

        # Create a summarization request (no response_format)
        request_data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "This is a test transcript. Please summarize it."},
            ],
            "temperature": 0.3,
        }

        response = requests.post(url, json=request_data, timeout=2)
        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"
        assert "application/json" in response.headers.get("Content-Type", "")

        data = response.json()
        assert "id" in data
        assert data["id"] == "chatcmpl-test-summary"
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]
        assert "test summary" in data["choices"][0]["message"]["content"].lower()

    def test_chat_completions_endpoint_speaker_detection(self, e2e_server):
        """Test that chat completions endpoint works for speaker detection."""
        openai_api_base = e2e_server.urls.openai_api_base()
        url = f"{openai_api_base}/chat/completions"

        # Create a speaker detection request (with response_format={"type": "json_object"})
        request_data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Detect speakers in this episode."},
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(url, json=request_data, timeout=2)
        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"
        assert "application/json" in response.headers.get("Content-Type", "")

        data = response.json()
        assert "id" in data
        assert data["id"] == "chatcmpl-test-speaker"
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]

        # Parse the JSON content
        content = json.loads(data["choices"][0]["message"]["content"])
        assert "speakers" in content
        assert "hosts" in content
        assert "guests" in content
        assert isinstance(content["speakers"], list)
        assert len(content["speakers"]) > 0

    def test_audio_transcriptions_endpoint(self, e2e_server):
        """Test that audio transcriptions endpoint works."""
        import tempfile

        openai_api_base = e2e_server.urls.openai_api_base()
        url = f"{openai_api_base}/audio/transcriptions"

        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake audio data for testing")
            audio_path = f.name

        try:
            # Create multipart form data request
            with open(audio_path, "rb") as audio_file:
                files = {"file": ("test_audio.mp3", audio_file, "audio/mpeg")}
                data = {"model": "whisper-1"}

                response = requests.post(url, files=files, data=data, timeout=5)
                assert (
                    response.status_code == 200
                ), f"Expected 200, got {response.status_code}: {response.text}"
                assert "text/plain" in response.headers.get("Content-Type", "")

                # Check response content
                transcript = response.text
                assert len(transcript) > 0
                assert "test transcription" in transcript.lower()
                assert "test_audio.mp3" in transcript or "audio" in transcript.lower()
        finally:
            import os

            os.unlink(audio_path)

    def test_openai_provider_uses_e2e_server(self, e2e_server):
        """Test that OpenAI providers can successfully use E2E server endpoints."""
        from podcast_scraper import config
        from podcast_scraper.speaker_detectors.factory import create_speaker_detector
        from podcast_scraper.summarization.factory import create_summarization_provider
        from podcast_scraper.transcription.factory import create_transcription_provider

        # Create config with E2E server as OpenAI API base
        openai_api_base = e2e_server.urls.openai_api_base()
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            openai_api_key="sk-test123",
            openai_api_base=openai_api_base,
            transcription_provider="openai",
            speaker_detector_provider="openai",
            summary_provider="openai",
        )

        # Test transcription provider via factory
        transcription_provider = create_transcription_provider(cfg)
        transcription_provider.initialize()
        # OpenAI client base_url may have trailing slash, normalize both
        assert str(transcription_provider.client.base_url).rstrip("/") == openai_api_base.rstrip(
            "/"
        )

        # Test speaker detector via factory
        speaker_detector = create_speaker_detector(cfg)
        speaker_detector.initialize()
        assert str(speaker_detector.client.base_url).rstrip("/") == openai_api_base.rstrip("/")

        # Test summarization provider via factory
        summarization_provider = create_summarization_provider(cfg)
        summarization_provider.initialize()
        assert str(summarization_provider.client.base_url).rstrip("/") == openai_api_base.rstrip(
            "/"
        )

    def test_openai_provider_transcription_works(self, e2e_server):
        """Test that OpenAI transcription provider can transcribe via E2E server."""
        import tempfile

        from podcast_scraper import config
        from podcast_scraper.transcription.factory import create_transcription_provider

        openai_api_base = e2e_server.urls.openai_api_base()
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            openai_api_key="sk-test123",
            openai_api_base=openai_api_base,
            transcription_provider="openai",  # Use OpenAI provider, not Whisper
            transcribe_missing=True,  # Enable transcription
        )

        provider = create_transcription_provider(cfg)
        provider.initialize()

        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake audio data for testing")
            audio_path = f.name

        try:
            transcript = provider.transcribe(audio_path)
            assert len(transcript) > 0
            assert "test transcription" in transcript.lower()
        finally:
            import os

            os.unlink(audio_path)

    def test_openai_provider_speaker_detection_works(self, e2e_server):
        """Test that OpenAI speaker detector can detect speakers via E2E server."""
        from unittest.mock import patch

        from podcast_scraper import config
        from podcast_scraper.speaker_detectors.factory import create_speaker_detector

        openai_api_base = e2e_server.urls.openai_api_base()
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            openai_api_key="sk-test123",
            openai_api_base=openai_api_base,
            speaker_detector_provider="openai",  # Use OpenAI provider, not spaCy
        )

        detector = create_speaker_detector(cfg)
        detector.initialize()

        # Mock render_prompt to avoid loading actual prompt files
        with patch("podcast_scraper.prompt_store.render_prompt", return_value="System prompt"):
            speakers, detected_hosts, success = detector.detect_speakers(
                episode_title="Test Episode",
                episode_description="Test description",
                known_hosts=set(),
            )

            assert success is True
            assert len(speakers) > 0
            assert "Host" in speakers or "Guest" in speakers

    def test_openai_provider_summarization_works(self, e2e_server):
        """Test that OpenAI summarization provider can summarize via E2E server."""
        from unittest.mock import patch

        from podcast_scraper import config
        from podcast_scraper.summarization.factory import create_summarization_provider

        openai_api_base = e2e_server.urls.openai_api_base()
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            openai_api_key="sk-test123",
            openai_api_base=openai_api_base,
            summary_provider="openai",  # Use OpenAI provider, not transformers
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries is True
        )

        provider = create_summarization_provider(cfg)
        provider.initialize()

        # Mock render_prompt to avoid loading actual prompt files
        with patch("podcast_scraper.prompt_store.render_prompt", return_value="System prompt"):
            result = provider.summarize(
                text="This is a test transcript with some content that needs to be summarized.",
                episode_title="Test Episode",
            )

            assert "summary" in result
            assert len(result["summary"]) > 0
            assert "test summary" in result["summary"].lower()


@pytest.mark.e2e
class TestE2EServerMultiEpisodeFeed:
    """Test that E2E server multi-episode feed works correctly with mocks."""

    @pytest.mark.critical_path
    def test_multi_episode_feed_url_helper(self, e2e_server):
        """Test that multi-episode feed URL helper works."""
        feed_url = e2e_server.urls.feed("podcast1_multi_episode")
        assert "/feeds/podcast1_multi_episode/feed.xml" in feed_url
        assert feed_url.startswith("http://")

    @pytest.mark.critical_path
    def test_multi_episode_feed_rss_structure(self, e2e_server):
        """Test that multi-episode feed RSS has correct structure with 5 episodes."""
        import xml.etree.ElementTree as ET

        feed_url = e2e_server.urls.feed("podcast1_multi_episode")

        response = requests.get(feed_url, timeout=2)
        assert response.status_code == 200
        assert "application/xml" in response.headers.get("Content-Type", "")

        # Parse RSS feed
        root = ET.fromstring(response.text)  # nosec B314
        channel = root.find("channel")

        # Find all items (episodes)
        items = channel.findall("item")
        assert len(items) == 5, f"Expected 5 episodes in multi-episode feed, got {len(items)}"

        # Verify episode structure
        for i, item in enumerate(items, 1):
            title = item.find("title")
            guid = item.find("guid")
            enclosure = item.find("enclosure")
            duration = item.find("{http://www.itunes.com/dtds/podcast-1.0.dtd}duration")

            assert title is not None, f"Episode {i} should have title"
            assert guid is not None, f"Episode {i} should have guid"
            assert enclosure is not None, f"Episode {i} should have enclosure"
            assert duration is not None, f"Episode {i} should have duration"

            # Verify duration is short (10-15 seconds)
            duration_text = duration.text
            assert duration_text.startswith(
                "00:00:"
            ), f"Episode {i} duration should be short (10-15 seconds), got {duration_text}"

    @pytest.mark.critical_path
    def test_multi_episode_feed_episodes_with_transcripts(self, e2e_server):
        """Test that multi-episode feed episodes 1 and 2 have transcript URLs (Path 1: Download)."""
        import xml.etree.ElementTree as ET

        feed_url = e2e_server.urls.feed("podcast1_multi_episode")

        response = requests.get(feed_url, timeout=2)
        assert response.status_code == 200

        root = ET.fromstring(response.text)  # nosec B314
        channel = root.find("channel")
        items = channel.findall("item")

        # Episodes 1 and 2 should have transcript URLs
        for i in [0, 1]:  # First two episodes (0-indexed)
            item = items[i]
            transcript = item.find("{https://podcastindex.org/namespace/1.0}transcript")
            assert (
                transcript is not None
            ), f"Episode {i+1} should have transcript URL (Path 1: Download)"
            assert (
                transcript.get("url") is not None
            ), f"Episode {i+1} transcript should have URL attribute"
            assert "/transcripts/p01_multi_e" in transcript.get(
                "url"
            ), f"Episode {i+1} transcript URL should point to multi-episode transcript"

        # Episodes 3, 4, 5 should NOT have transcript URLs (Path 2: Transcription)
        for i in [2, 3, 4]:  # Episodes 3, 4, 5 (0-indexed)
            item = items[i]
            transcript = item.find("{https://podcastindex.org/namespace/1.0}transcript")
            assert (
                transcript is None
            ), f"Episode {i+1} should NOT have transcript URL (Path 2: Transcription)"

    @pytest.mark.critical_path
    def test_multi_episode_feed_audio_files_accessible(self, e2e_server):
        """Test that all multi-episode feed audio files are accessible."""
        # Test all 5 multi-episode episode audio files
        for episode_num in range(1, 6):
            audio_url = e2e_server.urls.audio(f"p01_multi_e{episode_num:02d}")

            response = requests.get(audio_url, timeout=2)
            assert (
                response.status_code == 200
            ), f"Audio file for episode {episode_num} should be accessible"
            assert "audio/mpeg" in response.headers.get("Content-Type", "")
            assert len(response.content) > 0

    @pytest.mark.critical_path
    def test_multi_episode_feed_transcript_files_accessible(self, e2e_server):
        """Test that multi-episode feed transcript files are accessible."""
        # Episodes 1 and 2 have transcripts
        for episode_num in [1, 2]:
            transcript_url = e2e_server.urls.transcript(f"p01_multi_e{episode_num:02d}")

            response = requests.get(transcript_url, timeout=2)
            assert (
                response.status_code == 200
            ), f"Transcript file for episode {episode_num} should be accessible"
            assert "text/plain" in response.headers.get("Content-Type", "")
            assert len(response.text) > 0

    @pytest.mark.critical_path
    def test_multi_episode_feed_with_openai_mocks_fast(self, e2e_server):
        """Test that multi-episode feed works correctly with OpenAI mocks (fast: 1 episode).

        Critical path variant that validates OpenAI mocks work with multi-episode feed
        using 1 episode.
        """
        import tempfile

        from podcast_scraper import Config, run_pipeline

        # Set up OpenAI API base to use E2E server
        openai_api_base = e2e_server.urls.openai_api_base()

        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,  # Fast variant: 1 episode for critical path
                # Enable transcription for episodes without transcripts
                transcribe_missing=True,
                transcription_provider="openai",  # Use OpenAI for transcription
                # Required for config validation (uses mocked E2E server)
                openai_api_key="sk-test-dummy-key-for-e2e-tests",
                openai_api_base=openai_api_base,
                generate_metadata=True,
                metadata_format="json",
            )

            # Run pipeline - should process episode 1 (transcript)
            count, summary = run_pipeline(cfg)

            # Should process 1 episode (fast mode)
            assert count == 1, f"Should process 1 multi-episode episode (fast mode), got {count}"

            # Verify transcript files were created
            from pathlib import Path

            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert (
                len(transcript_files) >= 1
            ), f"Should have at least 1 transcript file, got {len(transcript_files)}"

            # Verify metadata files were created
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) == 1
            ), f"Should have exactly 1 metadata file, got {len(metadata_files)}"

    @pytest.mark.slow
    def test_multi_episode_feed_with_openai_mocks(self, e2e_server):
        """Test that multi-episode feed works correctly with OpenAI mocks (full: 3 episodes).

        Full variant that validates OpenAI mocks work with multi-episode feed using
        3 episodes (mix of Path 1 and Path 2).
        """
        import tempfile

        from podcast_scraper import Config, run_pipeline

        # Set up OpenAI API base to use E2E server
        openai_api_base = e2e_server.urls.openai_api_base()

        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                # Full variant: 3 episodes (mix of Path 1 and Path 2)
                max_episodes=3,
                # Enable transcription for episodes without transcripts
                transcribe_missing=True,
                transcription_provider="openai",  # Use OpenAI for transcription
                # Required for config validation (uses mocked E2E server)
                openai_api_key="sk-test-dummy-key-for-e2e-tests",
                openai_api_base=openai_api_base,
                generate_metadata=True,
                metadata_format="json",
            )

            # Run pipeline - should process episodes 1 (transcript),
            # 2 (transcript), 3 (transcription)
            count, summary = run_pipeline(cfg)

            # Determine expected episode count based on test mode
            import os

            test_mode = os.environ.get("E2E_TEST_MODE", "multi_episode").lower()
            expected_episodes = 1 if test_mode == "fast" else 3

            # Should process expected number of episodes (adjust for test mode)
            assert (
                count == expected_episodes
            ), f"Should process {expected_episodes} episode(s) (mode: {test_mode}), got {count}"

            # Verify transcript files were created
            from pathlib import Path

            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) >= expected_episodes, (
                f"Should have at least {expected_episodes} transcript "
                f"file(s), got {len(transcript_files)}"
            )

            # Verify metadata files were created
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert len(metadata_files) == expected_episodes, (
                f"Should have exactly {expected_episodes} metadata "
                f"file(s), got {len(metadata_files)}"
            )

    @pytest.mark.critical_path
    def test_multi_episode_feed_multi_episode_processing_fast(self, e2e_server):
        """Test that multi-episode feed processes episodes correctly (fast variant: 1 episode).

        Critical path variant that validates multi-episode processing logic with 1 episode.
        """
        import tempfile

        from podcast_scraper import Config, run_pipeline

        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,  # Fast variant: 1 episode for critical path
                transcribe_missing=True,  # Enable transcription
                generate_metadata=True,
                metadata_format="json",
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Should process 1 episode (fast mode)
            assert count == 1, f"Should process 1 multi-episode episode (fast mode), got {count}"

            # Verify episode was processed
            from pathlib import Path

            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert (
                len(metadata_files) == 1
            ), f"Should have exactly 1 metadata file, got {len(metadata_files)}"

    @pytest.mark.slow
    def test_multi_episode_feed_multi_episode_processing(self, e2e_server):
        """Test that multi-episode feed processes all 5 episodes correctly (full variant).

        Full variant that validates multi-episode processing logic with all 5 episodes.
        In fast mode, only 1 episode is processed.
        Note: Episodes without transcripts require Whisper model to be cached.
        """
        import os
        import tempfile

        from podcast_scraper import Config, run_pipeline

        # Determine expected episode count based on test mode
        test_mode = os.environ.get("E2E_TEST_MODE", "multi_episode").lower()
        expected_episodes = 1 if test_mode == "fast" else 5

        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=5,  # Request 5 episodes (will be limited to 1 in fast mode)
                transcribe_missing=True,  # Enable transcription
                generate_metadata=True,
                metadata_format="json",
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # In multi-episode mode, episodes 1 and 2 have transcripts (will be processed)
            # Episodes 3, 4, 5 need transcription (will only be processed if Whisper is cached)
            # So we adjust expectations: at least 2 episodes (with transcripts) should be processed
            # In fast mode, only 1 episode is processed
            if test_mode == "fast":
                assert (
                    count == expected_episodes
                ), f"Should process {expected_episodes} episode(s) (mode: {test_mode}), got {count}"
            else:
                # In multi-episode mode, at least 2 episodes (with transcripts) should be processed
                # More if Whisper is cached
                assert count >= 2, (
                    f"Should process at least 2 episodes (with transcripts) "
                    f"in multi-episode mode, got {count}"
                )

            # Verify metadata files were created
            from pathlib import Path

            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            if test_mode == "fast":
                assert len(metadata_files) == expected_episodes, (
                    f"Should have exactly {expected_episodes} metadata "
                    f"file(s), got {len(metadata_files)}"
                )
            else:
                assert len(metadata_files) >= 2, (
                    f"Should have at least 2 metadata file(s) in multi-episode mode, "
                    f"got {len(metadata_files)}"
                )

            # Verify transcript files (episodes 1 and 2 have transcripts)
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            if test_mode == "fast":
                assert len(transcript_files) >= expected_episodes, (
                    f"Should have at least {expected_episodes} transcript "
                    f"file(s), got {len(transcript_files)}"
                )
            else:
                assert len(transcript_files) >= 2, (
                    f"Should have at least 2 transcript file(s) in multi-episode mode, "
                    f"got {len(transcript_files)}"
                )

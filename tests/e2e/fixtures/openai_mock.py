"""OpenAI API mock fixture for E2E tests.

This module provides a mock OpenAI client that returns realistic responses
for all OpenAI provider methods. This prevents real API calls in E2E tests,
avoiding costs, rate limits, and flakiness.

Note:
    OpenAI providers are intentionally mocked in E2E tests. E2E tests should
    use real implementations for internal components (HTTP client, ML models)
    but mock external paid services (OpenAI API).
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest


class MockOpenAIClient:
    """Mock OpenAI client that returns realistic responses."""

    def __init__(self):
        """Initialize mock OpenAI client with realistic responses."""
        # Mock chat.completions.create for summarization and speaker detection
        self.chat = MagicMock()
        self.chat.completions = MagicMock()

        # Mock audio.transcriptions.create for transcription
        self.audio = MagicMock()
        self.audio.transcriptions = MagicMock()

    def _create_summarization_response(self, text: str) -> Mock:
        """Create a realistic summarization response."""
        # Generate a simple summary based on text length
        summary_length = min(200, len(text) // 10)
        summary = f"This is a test summary of the transcript. {text[:summary_length]}..."

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=summary))]
        return mock_response

    def _create_transcription_response(self, audio_path: str) -> str:
        """Create a realistic transcription response."""
        # Return a simple transcript based on audio file name
        return (
            f"This is a test transcription of {audio_path}. "
            "The audio contains spoken content that has been transcribed."
        )

    def _create_speaker_detection_response(self, episode_title: str) -> str:
        """Create a realistic speaker detection response."""
        # Return JSON with detected speakers
        import json

        response_data = {
            "speakers": ["Host", "Guest"],
            "hosts": ["Host"],
            "guests": ["Guest"],
        }
        return json.dumps(response_data)


@pytest.fixture(autouse=False)
def openai_mock():
    """Mock OpenAI client for E2E tests (disabled by default).

    This fixture patches the OpenAI client initialization in all OpenAI providers,
    replacing real API calls with mock responses. However, it bypasses the HTTP
    layer, so it's disabled by default for E2E tests.

    E2E tests should use the E2E server's OpenAI mock endpoints instead (via
    configure_openai_mock_server fixture), which tests the full HTTP flow.

    The mock returns realistic responses:
    - Summarization: Returns a summary based on input text
    - Transcription: Returns a transcript based on audio file name
    - Speaker detection: Returns detected speakers in JSON format

    Note:
        This fixture is autouse=False, so it's NOT automatically applied.
        Only use it explicitly if you need direct mocking (not recommended for E2E).
    """
    mock_client = MockOpenAIClient()

    # Patch OpenAI client initialization in all providers
    with (
        patch("podcast_scraper.summarization.openai_provider.OpenAI") as mock_summary_openai,
        patch("podcast_scraper.transcription.openai_provider.OpenAI") as mock_transcription_openai,
        patch("podcast_scraper.speaker_detectors.openai_detector.OpenAI") as mock_speaker_openai,
    ):

        # Configure mock clients to return our mock client
        mock_summary_openai.return_value = mock_client
        mock_transcription_openai.return_value = mock_client
        mock_speaker_openai.return_value = mock_client

        # Configure chat.completions.create for summarization
        def summarize_side_effect(*args, **kwargs):
            messages = kwargs.get("messages", [])
            user_message = next((m for m in messages if m.get("role") == "user"), {})
            text = user_message.get("content", "")
            return mock_client._create_summarization_response(text)

        mock_client.chat.completions.create.side_effect = summarize_side_effect

        # Configure audio.transcriptions.create for transcription
        def transcribe_side_effect(*args, **kwargs):
            audio_file = kwargs.get("file")
            if hasattr(audio_file, "name"):
                audio_path = audio_file.name
            else:
                audio_path = "unknown_audio.mp3"
            return mock_client._create_transcription_response(audio_path)

        mock_client.audio.transcriptions.create.side_effect = transcribe_side_effect

        # Configure chat.completions.create for speaker detection
        def speaker_side_effect(*args, **kwargs):
            messages = kwargs.get("messages", [])
            user_message = next((m for m in messages if m.get("role") == "user"), {})
            episode_title = user_message.get("content", "")[:100]  # Extract title from prompt
            return Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content=mock_client._create_speaker_detection_response(episode_title)
                        )
                    )
                ]
            )

        # Override for speaker detection (needs JSON response)
        def create_with_json(*args, **kwargs):
            # Check if this is a speaker detection call (has response_format)
            if kwargs.get("response_format") == {"type": "json_object"}:
                return speaker_side_effect(*args, **kwargs)
            # Otherwise use summarization side effect
            return summarize_side_effect(*args, **kwargs)

        mock_client.chat.completions.create.side_effect = create_with_json

        yield mock_client

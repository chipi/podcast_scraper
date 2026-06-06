"""Integration test for the Deepgram provider against the E2E mock server.

The mock-server round-trip: point the **real** ``deepgram-sdk`` at the E2E HTTP
server's ``/v1/listen`` endpoint (via ``deepgram_api_base``) and run a full
transcription. Unlike the unit/integration-tier tests that mock the SDK method,
this exercises the real SDK's request-building and ``ListenV1Response``
deserialization — so an SDK-contract drift (on a deepgram-sdk bump) is caught
here, mirroring the OpenAI/Gemini/Mistral E2E-server mocks.

Skips when ``deepgram-sdk`` isn't installed (it's an optional ``[llm]`` extra).
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, cast, Dict

import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

deepgram = pytest.importorskip("deepgram", reason="deepgram-sdk not installed ([llm] extra)")

from podcast_scraper import config  # noqa: E402
from podcast_scraper.transcription.factory import create_transcription_provider  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.llm]


@pytest.mark.deepgram
class TestDeepgramE2EServerIntegration:
    """The real Deepgram SDK talks to the E2E server's /v1/listen mock."""

    def test_transcription_round_trips_through_mock_server(self, e2e_server) -> None:
        cfg = config.Config(
            rss="https://example.com/feed.xml",
            transcription_provider="deepgram",
            deepgram_api_key="dg-test-key",
            deepgram_model="nova-3",
            deepgram_api_base=e2e_server.urls.deepgram_api_base(),
        )

        provider = create_transcription_provider(cfg)
        provider.initialize()
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(b"FAKE AUDIO DATA")
                audio_path = tmp.name

            try:
                raw_result, elapsed = provider.transcribe_with_segments(audio_path, language="en")
                result = cast(Dict[str, Any], raw_result)
            finally:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
        finally:
            provider.cleanup()

        # The canned mock response deserialized through the real SDK and our parser.
        assert elapsed >= 0
        assert "welcome to the show" in result["text"].lower()
        segments = result["segments"]
        assert [s.get("speaker") for s in segments] == [0, 1]
        assert segments[0]["text"] == "Welcome to the show."

        # And the native diarization renders a named screenplay end-to-end.
        screenplay = provider.format_screenplay_from_segments(segments, None, ["Maya", "Liam"])
        assert screenplay is not None
        assert "Maya: Welcome to the show." in screenplay
        assert "Liam: Thanks for having me." in screenplay

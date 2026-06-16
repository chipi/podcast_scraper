"""Real-API e2e for the Deepgram transcription provider (manual / opt-in).

Parity with the other API-provider e2e tests (USE_REAL_OPENAI_API /
USE_REAL_ANTHROPIC_API …): a path to exercise the **real Deepgram service** end
to end. It is OFF by default — CI and local runs use the mock-server round-trip
(tests/integration/infrastructure/test_deepgram_mock.py) and the SDK-mocked
integration tier instead, so nothing here ever bills a real account unless you
explicitly opt in.

Enable with::

    USE_REAL_DEEPGRAM_API=1 DEEPGRAM_API_KEY=dg-... \\
        pytest tests/e2e/test_deepgram_provider_e2e.py -m e2e

Skips (never fails) when not opted in, the key is absent, or deepgram-sdk isn't
installed — so it's safe to collect in any environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.llm, pytest.mark.deepgram, pytest.mark.network]

USE_REAL_DEEPGRAM_API = os.getenv("USE_REAL_DEEPGRAM_API", "0") == "1"

# Audio fixtures are versioned (#902); the old non-versioned path no longer exists.
_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "audio" / "v1" / "p01_multi_e01.mp3"


@pytest.mark.skipif(
    not USE_REAL_DEEPGRAM_API,
    reason="real Deepgram API disabled (set USE_REAL_DEEPGRAM_API=1 to enable)",
)
class TestDeepgramRealAPI:
    """Hit the real Deepgram service — opt-in, billed."""

    def test_real_transcription_with_native_diarization(self) -> None:
        pytest.importorskip("deepgram", reason="deepgram-sdk not installed ([llm] extra)")
        if not os.getenv("DEEPGRAM_API_KEY"):
            pytest.skip("DEEPGRAM_API_KEY not set")
        assert _FIXTURE.is_file(), f"missing fixture: {_FIXTURE}"

        from podcast_scraper import config
        from podcast_scraper.providers.deepgram.deepgram_provider import (
            DeepgramTranscriptionProvider,
        )

        # No deepgram_api_base -> the real hosted endpoint.
        cfg = config.Config(
            rss="https://example.com/feed.xml",
            transcription_provider="deepgram",
            deepgram_api_key=os.environ["DEEPGRAM_API_KEY"],
            deepgram_model="nova-3",
        )
        provider = DeepgramTranscriptionProvider(cfg)
        provider.initialize()
        try:
            result, elapsed = provider.transcribe_with_segments(str(_FIXTURE), language="en")
        finally:
            provider.cleanup()

        assert elapsed >= 0
        assert (result.get("text") or "").strip(), "real Deepgram returned empty transcript"
        segments = result.get("segments") or []
        assert segments, "real Deepgram returned no segments"
        # The two-voice fixture should diarize into >= 2 distinct speakers, and the
        # native-diarization screenplay should render at least two named lines.
        speakers = {s.get("speaker") for s in segments if s.get("speaker") is not None}
        assert len(speakers) >= 2, f"expected >= 2 diarized speakers, got {sorted(speakers)}"
        screenplay = provider.format_screenplay_from_segments(segments, None, ["Maya", "Liam"])
        assert screenplay is not None and screenplay.strip()
        assert "Maya:" in screenplay and "Liam:" in screenplay

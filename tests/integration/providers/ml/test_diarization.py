"""Integration test: diarization factory + pyannote provider wiring (mocked model).

Exercises the real dispatch path — ``create_diarization_provider(cfg)`` resolves the
local pyannote backend, constructs the provider, and the provider maps the pyannote
output into a ``DiarizationResult`` — with the pyannote model itself **mocked** (no gated
download, no HuggingFace token, no real audio). This is the integration tier: it proves
the wiring, not that diarization works on real audio.

The *real* pyannote pipeline running on actual TTS audio (host **Maya** + guest **Liam**,
two distinct voices) is covered end-to-end by ``tests/e2e/test_diarization_e2e.py`` — the
e2e tier, where real ML belongs (see scripts/tools/check_test_policy.py: no @ml_models in
integration; all real-ML tests live in tests/e2e/).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.diarization]

_PROVIDER_MOD = "podcast_scraper.providers.ml.diarization.pyannote_provider"


@patch(f"{_PROVIDER_MOD}._load_waveform")
@patch(f"{_PROVIDER_MOD}._create_pyannote_pipeline")
def test_factory_builds_local_pyannote_and_maps_two_speakers(
    mock_create_pipeline: MagicMock, mock_load_waveform: MagicMock
) -> None:
    """``create_diarization_provider`` -> local pyannote provider maps a 2-voice diarization.

    Mirrors the host+guest fixture (two distinct speakers) with the model mocked, so the
    factory dispatch *and* the provider's pyannote-output mapping run without real pyannote.
    """
    from podcast_scraper import config
    from podcast_scraper.providers.ml.diarization.factory import create_diarization_provider
    from podcast_scraper.providers.ml.diarization.pyannote_provider import (
        PyAnnoteDiarizationProvider,
    )

    # Mock the gated-model boundary: a fake pipeline whose pyannote-4.x DiarizeOutput
    # carries a two-speaker Annotation. _load_waveform is mocked so no real audio is read.
    mock_pipeline = MagicMock()
    mock_create_pipeline.return_value = mock_pipeline
    mock_load_waveform.return_value = (MagicMock(), 16000)
    host_turn = MagicMock(start=0.0, end=2.0)
    guest_turn = MagicMock(start=2.0, end=3.5)
    mock_pipeline.return_value.speaker_diarization.itertracks.return_value = [
        (host_turn, None, "SPEAKER_00"),
        (guest_turn, None, "SPEAKER_01"),
    ]

    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        diarize=True,
        hf_token="test-token",  # dummy: the real Pipeline.from_pretrained is mocked out
    )

    provider = create_diarization_provider(cfg)
    assert isinstance(provider, PyAnnoteDiarizationProvider)

    result = provider.diarize("/tmp/audio.wav", min_speakers=1, max_speakers=5)

    # Two distinct mocked voices -> two mapped speakers, labels preserved, real span covered.
    assert result.num_speakers == 2, f"expected 2 speakers, got {result.num_speakers}"
    assert {seg.speaker for seg in result.segments} == {"SPEAKER_00", "SPEAKER_01"}
    covered = sum(seg.end - seg.start for seg in result.segments)
    assert covered > 1.0, f"diarization covered only {covered:.2f}s"

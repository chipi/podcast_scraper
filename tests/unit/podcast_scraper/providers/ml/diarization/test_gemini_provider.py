"""Unit tests for the Gemini audio diarization provider (#962)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.providers.ml.diarization.gemini_provider import (
    _parse_diarization_json,
    GeminiDiarizationProvider,
)

pytestmark = pytest.mark.unit


def _payload(speakers: list[dict]) -> str:
    return json.dumps({"speakers": speakers})


def _make_provider() -> GeminiDiarizationProvider:
    """Build a provider with the SDK fully mocked at construction time."""
    fake_module = MagicMock()
    fake_module.Client.return_value = MagicMock()
    with patch.dict(
        "sys.modules",
        {"google.genai": fake_module, "google": MagicMock(genai=fake_module)},
    ):
        return GeminiDiarizationProvider(api_key="test-key", model_name="gemini-2.5-flash")


# ---------------------------------------------------------------- parser unit


def test_parse_well_formed_payload() -> None:
    raw = _payload(
        [
            {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_01"},
        ]
    )
    segments = _parse_diarization_json(raw)
    assert len(segments) == 2
    assert segments[0].speaker == "SPEAKER_00"
    assert segments[1].end == 3.0


def test_parse_strips_markdown_fences() -> None:
    raw = "```json\n" + _payload([{"start": 0.1, "end": 2.0, "speaker": "S0"}]) + "\n```"
    segments = _parse_diarization_json(raw)
    assert len(segments) == 1
    assert segments[0].speaker == "S0"


def test_parse_skips_malformed_entries() -> None:
    raw = json.dumps(
        {
            "speakers": [
                {"start": 0.0, "end": 1.0, "speaker": "S0"},
                {"start": 1.0, "speaker": "S1"},  # missing end
                {"start": 2.0, "end": "not-a-float", "speaker": "S2"},
                {"start": 5.0, "end": 4.0, "speaker": "S3"},  # end <= start
                "not-a-dict",
            ]
        }
    )
    segments = _parse_diarization_json(raw)
    assert len(segments) == 1
    assert segments[0].speaker == "S0"


def test_parse_rejects_invalid_top_level() -> None:
    with pytest.raises(ValueError):
        _parse_diarization_json("not json at all")
    with pytest.raises(ValueError):
        _parse_diarization_json(json.dumps({"wrong_key": []}))
    with pytest.raises(ValueError):
        _parse_diarization_json(json.dumps({"speakers": "not-a-list"}))


# ---------------------------------------------------------------- diarize call


def test_diarize_uploads_audio_and_parses_response() -> None:
    provider = _make_provider()
    fake_file = MagicMock(name="fake_file")
    provider.client.files.upload.return_value = fake_file
    fake_response = MagicMock()
    fake_response.text = _payload(
        [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
        ]
    )
    provider.client.models.generate_content.return_value = fake_response

    result = provider.diarize("/tmp/audio.mp3", num_speakers=2)

    provider.client.files.upload.assert_called_once_with(file="/tmp/audio.mp3")
    provider.client.models.generate_content.assert_called_once()
    assert result.num_speakers == 2
    assert len(result.segments) == 2
    assert result.model_name == "gemini-2.5-flash"


def test_diarize_cleans_up_uploaded_file_even_on_error() -> None:
    provider = _make_provider()
    fake_file = MagicMock(name="fake_file")
    fake_file.name = "files/test-id"
    provider.client.files.upload.return_value = fake_file
    provider.client.models.generate_content.side_effect = RuntimeError("API down")

    with pytest.raises(RuntimeError):
        provider.diarize("/tmp/audio.mp3")
    provider.client.files.delete.assert_called_once_with(name="files/test-id")


def test_diarize_raises_on_empty_response() -> None:
    provider = _make_provider()
    provider.client.files.upload.return_value = MagicMock()
    fake_response = MagicMock()
    fake_response.text = ""
    provider.client.models.generate_content.return_value = fake_response
    with pytest.raises(ValueError, match="empty response"):
        provider.diarize("/tmp/audio.mp3")

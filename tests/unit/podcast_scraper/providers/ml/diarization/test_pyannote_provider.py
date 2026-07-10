"""Unit tests for pyannote diarization provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.providers.ml.diarization.pyannote_provider import PyAnnoteDiarizationProvider

pytestmark = pytest.mark.unit


@patch("podcast_scraper.providers.ml.diarization.pyannote_provider._create_pyannote_pipeline")
def test_clustering_threshold_merges_into_pipeline_params(mock_create) -> None:
    # The tuning override re-instantiates the pipeline with ONLY clustering.threshold changed,
    # preserving the rest of the hyperparameters (the over-segmentation knob, GPU-free plumbing).
    mock_pipeline = MagicMock()
    mock_pipeline.parameters.return_value = {
        "segmentation": {"min_duration_off": 0.5},
        "clustering": {"method": "centroid", "threshold": 0.7},
    }
    mock_create.return_value = mock_pipeline

    PyAnnoteDiarizationProvider("token", device="cpu", clustering_threshold=0.85)

    mock_pipeline.instantiate.assert_called_once()
    applied = mock_pipeline.instantiate.call_args[0][0]
    assert applied["clustering"]["threshold"] == 0.85  # overridden
    assert applied["clustering"]["method"] == "centroid"  # preserved
    assert applied["segmentation"] == {"min_duration_off": 0.5}  # untouched


@patch("podcast_scraper.providers.ml.diarization.pyannote_provider._create_pyannote_pipeline")
def test_clustering_threshold_none_leaves_pipeline_untouched(mock_create) -> None:
    mock_pipeline = MagicMock()
    mock_create.return_value = mock_pipeline
    PyAnnoteDiarizationProvider("token", device="cpu")  # no threshold
    mock_pipeline.instantiate.assert_not_called()


@patch("podcast_scraper.providers.ml.diarization.pyannote_provider._load_waveform")
@patch("podcast_scraper.providers.ml.diarization.pyannote_provider._create_pyannote_pipeline")
def test_diarize_maps_pyannote_output(mock_create, mock_load_waveform) -> None:
    mock_pipeline = MagicMock()
    mock_create.return_value = mock_pipeline
    mock_load_waveform.return_value = (MagicMock(), 16000)

    turn_a = MagicMock(start=0.0, end=1.5)
    turn_b = MagicMock(start=1.5, end=3.0)
    # pyannote 4.x returns a DiarizeOutput whose .speaker_diarization is the
    # Annotation; the provider reads itertracks off that.
    mock_pipeline.return_value.speaker_diarization.itertracks.return_value = [
        (turn_a, None, "SPEAKER_00"),
        (turn_b, None, "SPEAKER_01"),
    ]

    provider = PyAnnoteDiarizationProvider("token", device="cpu")
    result = provider.diarize("/tmp/audio.wav", num_speakers=2)

    assert result.num_speakers == 2
    assert len(result.segments) == 2
    assert result.segments[0].speaker == "SPEAKER_00"


@patch("podcast_scraper.providers.ml.diarization.pyannote_provider._load_waveform")
@patch("podcast_scraper.providers.ml.diarization.pyannote_provider._create_pyannote_pipeline")
def test_diarize_unwraps_generator_return_pyannote_4_0_6(mock_create, mock_load_waveform) -> None:
    """pyannote 4.0.6 made ``Pipeline.__call__`` a generator function (a stray
    ``yield`` in the batch branch). For single-file input it still does
    ``return prediction`` — but in a generator that's ``StopIteration(prediction)``,
    so the caller sees a generator object instead of the DiarizeOutput.

    The provider must unwrap that and still produce the segment list, otherwise
    every diarize() call blows up with ``'generator' object has no attribute
    'itertracks'`` (the regression that broke nightly-test-e2e on 2026-06-30).
    """
    diarize_output = MagicMock()
    turn = MagicMock(start=0.0, end=2.0)
    diarize_output.speaker_diarization.itertracks.return_value = [
        (turn, None, "SPEAKER_00"),
    ]

    def fake_pipeline_call(*args, **kwargs):
        # Mirror pyannote 4.0.6: a function with `yield` anywhere is a generator,
        # so single-file input still goes through StopIteration.value, not return.
        if False:
            yield  # pragma: no cover — taints the function as a generator
        return diarize_output  # noqa: B901 — intentional generator-with-return

    mock_pipeline = MagicMock(side_effect=fake_pipeline_call)
    mock_create.return_value = mock_pipeline
    mock_load_waveform.return_value = (MagicMock(), 16000)

    provider = PyAnnoteDiarizationProvider("token", device="cpu")
    result = provider.diarize("/tmp/audio.wav", num_speakers=1)

    assert len(result.segments) == 1
    assert result.segments[0].speaker == "SPEAKER_00"


@patch("podcast_scraper.providers.ml.diarization.pyannote_provider._load_waveform")
@patch("podcast_scraper.providers.ml.diarization.pyannote_provider._create_pyannote_pipeline")
def test_diarize_rejects_non_positive_num_speakers(mock_create, mock_load_waveform) -> None:
    """num_speakers < 1 is rejected rather than silently bypassing the floor (A5)."""
    mock_create.return_value = MagicMock()
    mock_load_waveform.return_value = (MagicMock(), 16000)
    provider = PyAnnoteDiarizationProvider("token", device="cpu")
    with pytest.raises(ValueError):
        provider.diarize("/tmp/audio.wav", num_speakers=0)


@patch("podcast_scraper.providers.ml.diarization.pyannote_provider._load_waveform")
@patch("podcast_scraper.providers.ml.diarization.pyannote_provider._create_pyannote_pipeline")
def test_diarize_rejects_min_greater_than_max(mock_create, mock_load_waveform) -> None:
    """min_speakers > max_speakers is an invalid bound, not silently forwarded (A5)."""
    mock_create.return_value = MagicMock()
    mock_load_waveform.return_value = (MagicMock(), 16000)
    provider = PyAnnoteDiarizationProvider("token", device="cpu")
    with pytest.raises(ValueError):
        provider.diarize("/tmp/audio.wav", num_speakers=None, min_speakers=5, max_speakers=2)

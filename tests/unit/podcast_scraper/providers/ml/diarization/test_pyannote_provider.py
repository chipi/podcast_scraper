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
def test_clustering_min_cluster_size_and_threshold_merge(mock_create) -> None:
    # Both over-segmentation levers merge into clustering, preserving other hyperparameters.
    mock_pipeline = MagicMock()
    mock_pipeline.parameters.return_value = {
        "segmentation": {"min_duration_off": 0.5},
        "clustering": {"method": "centroid", "threshold": 0.7, "min_cluster_size": 12},
    }
    mock_create.return_value = mock_pipeline

    PyAnnoteDiarizationProvider(
        "token", device="cpu", clustering_threshold=0.8, min_cluster_size=20
    )

    applied = mock_pipeline.instantiate.call_args[0][0]
    assert applied["clustering"]["threshold"] == 0.8
    assert applied["clustering"]["min_cluster_size"] == 20  # fragments dropped
    assert applied["clustering"]["method"] == "centroid"  # preserved
    assert applied["segmentation"] == {"min_duration_off": 0.5}


@patch("podcast_scraper.providers.ml.diarization.pyannote_provider._create_pyannote_pipeline")
def test_clustering_overrides_none_leaves_pipeline_untouched(mock_create) -> None:
    mock_pipeline = MagicMock()
    mock_create.return_value = mock_pipeline
    PyAnnoteDiarizationProvider("token", device="cpu")  # no threshold, no min_cluster_size
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


def _seg(start: float, end: float, speaker: str):
    from podcast_scraper.providers.ml.diarization.base import DiarizationSegment

    return DiarizationSegment(start=start, end=end, speaker=speaker)


def test_segment_squelch_drops_phantom_subsecond_speaker() -> None:
    # A phantom over-segmentation cluster (only sub-second snippets) is dropped, while the two
    # real voices — each with a multi-second segment — survive. Mirrors the audited p03/p05 case.
    from podcast_scraper.providers.ml.diarization.pyannote_provider import _apply_segment_squelch

    segments = [
        _seg(0.0, 20.0, "SPEAKER_00"),  # real
        _seg(20.0, 40.0, "SPEAKER_01"),  # real
        _seg(12.0, 12.6, "SPEAKER_02"),  # phantom: 0.6s
        _seg(30.0, 30.3, "SPEAKER_02"),  # phantom: 0.3s
    ]
    kept = _apply_segment_squelch(segments, 1000)  # 1000ms squelch
    speakers = {s.speaker for s in kept}
    assert speakers == {"SPEAKER_00", "SPEAKER_01"}  # phantom dropped
    assert all(s.speaker != "SPEAKER_02" for s in kept)


def test_segment_squelch_keeps_real_cameo_by_longest_segment() -> None:
    # A real ~3s cameo has one contiguous segment above the gate — kept — even though its TOTAL
    # talk-time is small. The discriminator is longest segment, not total (that's the whole point).
    from podcast_scraper.providers.ml.diarization.pyannote_provider import _apply_segment_squelch

    segments = [
        _seg(0.0, 30.0, "HOST"),
        _seg(30.0, 33.0, "CAMEO"),  # one 3s turn
        _seg(45.0, 45.4, "PHANTOM"),  # 0.4s snippet
    ]
    kept = _apply_segment_squelch(segments, 1200)  # 1200ms squelch
    speakers = {s.speaker for s in kept}
    assert speakers == {"HOST", "CAMEO"}  # cameo kept, phantom dropped


def test_segment_squelch_disabled_when_none_or_zero() -> None:
    from podcast_scraper.providers.ml.diarization.pyannote_provider import _apply_segment_squelch

    segments = [_seg(0.0, 20.0, "A"), _seg(12.0, 12.3, "B")]
    assert _apply_segment_squelch(segments, None) is segments  # off → identity
    assert _apply_segment_squelch(segments, 0) is segments  # 0 → identity

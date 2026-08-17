"""Unit tests for pyannote diarization provider.

The provider imports ``torch`` / ``torchaudio`` / ``pyannote`` LAZILY, inside the functions that
need them — so importing this module needs none of the ML stack. What these tests exercise is the
provider's own logic: device coercion, clustering-override merging, output mapping, squelch.

``torch`` is stubbed rather than installed (see ``_stub_torch``). Unit tests must not depend on
non-``[dev]`` extras, and ``pytest.importorskip`` is banned here (U1) precisely because a skipped
test is a test that never runs. Stubbing keeps every assertion executing on any machine — and makes
the ``auto`` device case deterministic, which it was not while it asked the real torch whether this
particular box has CUDA.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.providers.ml.diarization.pyannote_provider import (
    _resolve_device,
    PyAnnoteDiarizationProvider,
)

pytestmark = pytest.mark.unit


def _fake_torch(*, cuda_available: bool) -> types.ModuleType:
    """The slice of ``torch`` this provider touches: ``cuda.is_available`` and ``device``."""
    mod = types.ModuleType("torch")
    mod.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    mod.device = lambda name: f"device({name})"
    return mod


@pytest.fixture(autouse=True)
def _stub_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """No CUDA by default — the branch every non-GPU machine takes."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False))


def test_resolve_device_coerces_mps_to_cpu() -> None:
    """pyannote's pipeline requests float64, which Apple Metal (MPS) rejects, so any MPS request
    must coerce to CPU — a Mac dev box diarizes on CPU (slower) instead of crashing. CUDA and CPU
    pass through unchanged, and 'auto' must never resolve to MPS."""
    assert _resolve_device("mps") == "cpu"
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda") == "cuda"
    assert _resolve_device("auto") == "cpu"  # no CUDA on this box → cpu, never mps


def test_resolve_device_auto_takes_cuda_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other half of ``auto`` — CI/DGX. Asserted explicitly rather than left to the host.

    The old assertion was ``in ("cuda", "cpu")``, which passes on every machine no matter which
    branch runs, so neither branch was actually pinned.
    """
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True))
    assert _resolve_device("auto") == "cuda"
    assert _resolve_device("mps") == "cpu"  # still never mps, CUDA present or not


def test_resolve_device_says_what_to_install_when_torch_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diarization is opt-in, so a venv without the ML stack must get a usable message, not an
    ImportError from three frames down."""
    from podcast_scraper.exceptions import ProviderDependencyError

    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    with pytest.raises(ProviderDependencyError) as excinfo:
        _resolve_device("auto")
    assert "[ml]" in str(excinfo.value)


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


def test_segment_squelch_warns_when_it_erases_every_speaker(caplog) -> None:
    # D6: when the squelch drops ALL speakers the episode's diarization collapses to empty and the
    # pipeline degrades silently — so the REASON must be logged at WARNING, not DEBUG.
    from podcast_scraper.providers.ml.diarization.pyannote_provider import _apply_segment_squelch

    segments = [_seg(12.0, 12.4, "SPEAKER_00"), _seg(30.0, 30.3, "SPEAKER_01")]  # all sub-second
    with caplog.at_level("WARNING"):
        kept = _apply_segment_squelch(segments, 1000)
    assert kept == []
    assert "NO speakers remain" in caplog.text


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

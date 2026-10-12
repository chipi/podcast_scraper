"""Per-voice embeddings are captured and stored (cross-episode voice identity, #2304).

pyannote 4.x returns one embedding per speaker and nothing kept them. These tests follow a vector
from the diarizer's output to the diagnostics sidecar, through every layer that could drop it:
the local provider, the DGX client (whose server may predate the field), the diarization cache,
the sidecar builder, and a relabel, which renumbers voices and rewrites the sidecar.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.providers.ml.diarization.base import (
    clean_embeddings,
    DiarizationResult,
    DiarizationSegment,
    EMBEDDING_DECIMALS,
    speaker_embeddings_from_output,
)
from podcast_scraper.providers.ml.diarization.cache import (
    load_cached_diarization,
    save_diarization_cache,
)
from podcast_scraper.providers.ml.diarization.pipeline import voice_embeddings_block
from podcast_scraper.workflow.episode_processor import _embeddings_carried_over

pytestmark = pytest.mark.unit


class _Annotation:
    def __init__(self, labels: List[str]) -> None:
        self._labels = labels

    def labels(self) -> List[str]:
        return list(self._labels)


class _Output:
    """The two attributes of pyannote 4.x ``DiarizeOutput`` the capture reads."""

    def __init__(self, labels: List[str], rows: List[List[float]]) -> None:
        self.speaker_diarization = _Annotation(labels)
        self.speaker_embeddings = rows


def test_embeddings_are_keyed_in_label_order_and_rounded() -> None:
    out = speaker_embeddings_from_output(
        _Output(["SPEAKER_00", "SPEAKER_01"], [[0.1234567, 1.0], [-0.5, 0.25]])
    )
    assert out == {
        "SPEAKER_00": [round(0.1234567, EMBEDDING_DECIMALS), 1.0],
        "SPEAKER_01": [-0.5, 0.25],
    }


def test_a_speaker_pyannote_could_not_embed_is_skipped_not_stored_as_noise() -> None:
    out = speaker_embeddings_from_output(
        _Output(["SPEAKER_00", "SPEAKER_01"], [[math.nan, 0.1], [0.2, 0.3]])
    )
    assert list(out) == ["SPEAKER_01"]


def test_pyannote_3_output_has_no_embeddings() -> None:
    assert speaker_embeddings_from_output(_Annotation(["SPEAKER_00"])) == {}


def test_untrusted_json_is_cleaned() -> None:
    raw = {"A": [1, 2.5], "B": "x", "C": [], "D": [1, "nan?"], "E": [float("inf")]}
    assert clean_embeddings(raw) == {"A": [1.0, 2.5]}
    assert clean_embeddings(None) == {} and clean_embeddings([1, 2]) == {}


@patch("podcast_scraper.providers.ml.diarization.pyannote_provider._load_waveform")
@patch("podcast_scraper.providers.ml.diarization.pyannote_provider._create_pyannote_pipeline")
def test_the_local_provider_keeps_vectors_only_for_voices_that_survive(
    mock_create: MagicMock, mock_load: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    torch.device = lambda name: name  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    from podcast_scraper.providers.ml.diarization.pyannote_provider import (
        PyAnnoteDiarizationProvider,
    )

    mock_load.return_value = (MagicMock(), 16000)
    output = _Output(["SPEAKER_00", "SPEAKER_01"], [[0.1, 0.2], [0.3, 0.4]])
    output.speaker_diarization.itertracks = lambda yield_label: [  # type: ignore[attr-defined]
        (MagicMock(start=0.0, end=600.0), None, "SPEAKER_00"),
        (MagicMock(start=600.0, end=600.05), None, "SPEAKER_01"),  # squelched below
    ]
    mock_create.return_value = MagicMock(return_value=output)

    result = PyAnnoteDiarizationProvider("t", device="cpu", min_segment_ms=500).diarize("a.wav")

    assert {s.speaker for s in result.segments} == {"SPEAKER_00"}
    assert result.speaker_embeddings == {"SPEAKER_00": [0.1, 0.2]}


@contextmanager
def _client_returning(payload: Dict[str, Any]):
    resp = MagicMock()
    resp.json.return_value = payload
    client = MagicMock()
    client.post.return_value = resp
    yield client


def _dgx_result(tmp_path: Path, payload: Dict[str, Any]) -> DiarizationResult:
    from podcast_scraper import config
    from podcast_scraper.providers.tailnet_dgx import diarization_provider as dp

    audio = tmp_path / "ep.wav"
    audio.write_bytes(b"\x00\x00")
    cfg = config.Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "diarization_provider": "tailnet_dgx",
            "diarize": True,
            "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            "hf_token": "hf_test",
        }
    )
    provider = dp.TailnetDgxDiarizationProvider(cfg)
    with (
        patch.object(dp, "hardened_http_client", lambda *a, **k: _client_returning(payload)),
        patch.object(dp.resilience, "probe_audio_duration_sec", return_value=None),
        patch.object(dp.guardrails, "check_pyannote_response", return_value=None),
    ):
        return provider._diarize_dgx(str(audio), num_speakers=None, min_speakers=2, max_speakers=5)


_SEGMENTS = [
    {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
    {"start": 5.0, "end": 9.0, "speaker": "SPEAKER_01"},
]


def test_the_dgx_client_reads_embeddings_from_a_new_server(tmp_path: Path) -> None:
    result = _dgx_result(
        tmp_path,
        {
            "model_name": "pyannote/speaker-diarization-community-1",
            "segments": _SEGMENTS,
            "speaker_embeddings": {"SPEAKER_00": [0.1], "SPEAKER_01": [0.2], "SPEAKER_09": [0.3]},
        },
    )
    assert result.speaker_embeddings == {"SPEAKER_00": [0.1], "SPEAKER_01": [0.2]}


def test_the_dgx_client_still_works_with_a_server_that_sends_none(tmp_path: Path) -> None:
    result = _dgx_result(tmp_path, {"model_name": "m", "segments": _SEGMENTS})
    assert result.speaker_embeddings == {}
    assert result.num_speakers == 2


def test_the_cache_keeps_the_vectors(tmp_path: Path) -> None:
    path = str(tmp_path / "c" / "d.json")
    save_diarization_cache(
        path,
        DiarizationResult(
            segments=[DiarizationSegment(0.0, 1.0, "SPEAKER_00")],
            num_speakers=1,
            model_name="m",
            speaker_embeddings={"SPEAKER_00": [0.5, -0.5]},
        ),
    )
    loaded = load_cached_diarization(path)
    assert loaded is not None and loaded.speaker_embeddings == {"SPEAKER_00": [0.5, -0.5]}


def test_a_cache_written_before_capture_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"segments": [{"start": 0, "end": 1, "speaker": "S"}]}))
    loaded = load_cached_diarization(str(path))
    assert loaded is not None and loaded.speaker_embeddings == {}


def test_the_sidecar_block_names_the_model_and_dimension() -> None:
    diar = DiarizationResult(model_name="m1", speaker_embeddings={"SPEAKER_00": [0.1, 0.2, 0.3]})
    assert voice_embeddings_block(diar) == {
        "model_name": "m1",
        "dimension": 3,
        "by_voice": {"SPEAKER_00": [0.1, 0.2, 0.3]},
    }
    assert voice_embeddings_block(DiarizationResult(model_name="m1")) is None


def test_carried_vectors_keep_the_model_that_made_them() -> None:
    diar = DiarizationResult(
        speaker_embeddings={"SPEAKER_00": [0.1]}, speaker_embeddings_model="m-original"
    )
    block = voice_embeddings_block(diar)
    assert block is not None and block["model_name"] == "m-original"


def test_a_relabel_carries_the_stored_vectors_to_the_renumbered_voices(tmp_path: Path) -> None:
    txt = tmp_path / "01 - ep.txt"
    txt.write_text("x")
    (tmp_path / "01 - ep.speakers.diagnostics.json").write_text(
        json.dumps(
            {
                "voices": [],
                "voice_embeddings": {
                    "model_name": "m-original",
                    "dimension": 1,
                    "by_voice": {"SPEAKER_03": [0.3], "SPEAKER_07": [0.7], "SPEAKER_09": [0.9]},
                },
            }
        )
    )
    # The relabel renumbers in first-appearance order: 07 appeared first, then 03.
    carried, model = _embeddings_carried_over(
        txt, {"SPEAKER_07": "SPEAKER_00", "SPEAKER_03": "SPEAKER_01"}
    )
    assert carried == {"SPEAKER_00": [0.7], "SPEAKER_01": [0.3]}
    assert model == "m-original"


def test_a_relabel_of_an_episode_without_vectors_carries_nothing(tmp_path: Path) -> None:
    txt = tmp_path / "01 - ep.txt"
    txt.write_text("x")
    assert _embeddings_carried_over(txt, {"SPEAKER_00": "SPEAKER_00"}) == ({}, "")
    (tmp_path / "01 - ep.speakers.diagnostics.json").write_text(json.dumps({"voices": []}))
    assert _embeddings_carried_over(txt, {"SPEAKER_00": "SPEAKER_00"}) == ({}, "")


def _apply(diar: DiarizationResult, tmp_path: Path) -> Dict[str, Any]:
    from podcast_scraper.config import Config
    from podcast_scraper.providers.ml.diarization import pipeline as P

    result = {
        "text": "Welcome. Thanks.",
        "segments": [
            {"start": 0, "end": 30, "text": "Welcome to the show."},
            {"start": 30, "end": 60, "text": "Thanks for having me."},
        ],
    }
    cfg = Config(output_dir=str(tmp_path), speaker_resolution_llm=False)
    return P.apply_diarization_to_result(result, "", cfg, [], precomputed_diarization=diar)


def test_the_sidecar_carries_the_vectors_of_a_fresh_diarization(tmp_path: Path) -> None:
    diar = DiarizationResult(
        segments=[
            DiarizationSegment(0, 30, "SPEAKER_00"),
            DiarizationSegment(30, 60, "SPEAKER_01"),
        ],
        num_speakers=2,
        model_name="pyannote/speaker-diarization-community-1",
        speaker_embeddings={"SPEAKER_00": [0.1, 0.2], "SPEAKER_01": [0.3, 0.4]},
    )
    diag = _apply(diar, tmp_path)["speaker_diagnostics"]
    assert diag["voice_embeddings"] == {
        "model_name": "pyannote/speaker-diarization-community-1",
        "dimension": 2,
        "by_voice": {"SPEAKER_00": [0.1, 0.2], "SPEAKER_01": [0.3, 0.4]},
    }
    # Keyed like the sidecar's own voices, so a reader can join the two.
    assert set(diag["voice_embeddings"]["by_voice"]) == {v["voice"] for v in diag["voices"]}


def test_a_diarizer_without_vectors_leaves_the_sidecar_unchanged(tmp_path: Path) -> None:
    diar = DiarizationResult(
        segments=[
            DiarizationSegment(0, 30, "SPEAKER_00"),
            DiarizationSegment(30, 60, "SPEAKER_01"),
        ],
        num_speakers=2,
    )
    assert "voice_embeddings" not in _apply(diar, tmp_path)["speaker_diagnostics"]

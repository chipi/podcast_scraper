"""Unit tests for the corpus diarization-quality validator (#876)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.evaluation.diarization_quality import (
    compute_diarization_quality_metrics,
    enforce_diarization_thresholds,
)

pytestmark = pytest.mark.unit


def _write_episode(
    root: Path,
    stem: str,
    *,
    quotes,
    speakers,
    num_speakers,
    feed: str = "f1",
    run: str = "run_20260101-000000",
) -> None:
    meta_dir = root / "feeds" / feed / run / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    gi = {"nodes": [{"type": "Quote", "properties": p} for p in quotes]}
    (meta_dir / f"{stem}.gi.json").write_text(json.dumps(gi))
    meta = {"content": {"speakers": speakers}, "diarization_num_speakers": num_speakers}
    (meta_dir / f"{stem}.metadata.json").write_text(json.dumps(meta))


def _good_quote(speaker_id: str, ts: int = 1000) -> dict:
    return {"text": "x", "speaker_id": speaker_id, "timestamp_start_ms": ts}


def test_clean_corpus_passes(tmp_path: Path) -> None:
    _write_episode(
        tmp_path,
        "0001-ep",
        quotes=[_good_quote("person:patrick-oshaughnessy"), _good_quote("person:brian-chesky")],
        speakers=[
            {"id": "host_1", "name": "Patrick O'Shaughnessy", "role": "host"},
            {"id": "guest", "name": "Brian Chesky", "role": "guest"},
        ],
        num_speakers=2,
    )
    m = compute_diarization_quality_metrics(tmp_path)
    assert m["episodes_diarized"] == 1
    assert m["quote_attribution_rate"] == 1.0
    assert m["quote_timestamp_rate"] == 1.0
    assert m["episodes_with_network_speaker"] == 0
    assert m["multispeaker_undernamed_episodes"] == 0
    passed, failures = enforce_diarization_thresholds(m)
    assert passed, failures


def test_network_speaker_name_flagged(tmp_path: Path) -> None:
    # The "Colossus" bug: a network/org name in content.speakers.
    _write_episode(
        tmp_path,
        "0001-ep",
        quotes=[_good_quote("person:patrick-oshaughnessy"), _good_quote("person:brian-chesky")],
        speakers=[
            {"id": "host_1", "name": "Colossus | Investing & Business Podcasts", "role": "host"},
            {"id": "guest", "name": "Brian Chesky", "role": "guest"},
        ],
        num_speakers=2,
    )
    m = compute_diarization_quality_metrics(tmp_path)
    assert m["episodes_with_network_speaker"] == 1
    passed, failures = enforce_diarization_thresholds(m)
    assert not passed and any("network/org" in f for f in failures)


def test_multispeaker_undernamed_flagged(tmp_path: Path) -> None:
    # The partial-naming bug: 2 speakers but only one is a real name (the other stays raw).
    _write_episode(
        tmp_path,
        "0001-ep",
        quotes=[_good_quote("person:brian-chesky"), _good_quote("person:speaker-02")],
        speakers=[{"id": "guest", "name": "Brian Chesky", "role": "guest"}],
        num_speakers=2,
    )
    m = compute_diarization_quality_metrics(tmp_path)
    assert m["multispeaker_episodes"] == 1
    assert m["multispeaker_undernamed_episodes"] == 1
    passed, failures = enforce_diarization_thresholds(m)
    assert not passed and any("named speakers" in f for f in failures)


def test_unattributed_quotes_flagged(tmp_path: Path) -> None:
    # The #545 / mis-attribution bug: quotes with no speaker_id.
    _write_episode(
        tmp_path,
        "0001-ep",
        quotes=[
            {"text": "x", "speaker_id": "", "timestamp_start_ms": 10},
            {"text": "y", "speaker_id": "person:brian-chesky", "timestamp_start_ms": 20},
        ],
        speakers=[{"id": "guest", "name": "Brian Chesky", "role": "guest"}],
        num_speakers=2,
    )
    m = compute_diarization_quality_metrics(tmp_path)
    assert m["quote_attribution_rate"] == 0.5
    passed, failures = enforce_diarization_thresholds(m)
    assert not passed and any("attribution_rate" in f for f in failures)


def test_missing_num_speakers_opt_in(tmp_path: Path) -> None:
    # num_speakers propagation gap: warned by default, enforced only when required.
    _write_episode(
        tmp_path,
        "0001-ep",
        quotes=[_good_quote("person:patrick-oshaughnessy"), _good_quote("person:brian-chesky")],
        speakers=[
            {"id": "host_1", "name": "Patrick O'Shaughnessy", "role": "host"},
            {"id": "guest", "name": "Brian Chesky", "role": "guest"},
        ],
        num_speakers=None,
    )
    m = compute_diarization_quality_metrics(tmp_path)
    assert m["episodes_missing_num_speakers"] == 1
    assert enforce_diarization_thresholds(m)[0] is True  # not enforced by default
    assert enforce_diarization_thresholds(m, require_num_speakers=True)[0] is False

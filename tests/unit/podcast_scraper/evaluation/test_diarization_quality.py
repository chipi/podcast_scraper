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
    spoken_by=None,
    feed: str = "f1",
    run: str = "run_20260101-000000",
) -> None:
    """Write one episode's gi.json + metadata.json under feeds/<feed>/<run>/metadata/.

    ``spoken_by`` SPOKEN_BY edges are added (defaults to the number of attributed quotes,
    mirroring the in-pipeline Quote->Person edges).
    """
    meta_dir = root / "feeds" / feed / run / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    if spoken_by is None:
        spoken_by = sum(1 for p in quotes if p.get("speaker_id"))
    gi = {
        "nodes": [{"type": "Quote", "properties": p} for p in quotes],
        "edges": [{"type": "SPOKEN_BY"} for _ in range(spoken_by)],
    }
    (meta_dir / f"{stem}.gi.json").write_text(json.dumps(gi))
    meta = {"content": {"speakers": speakers}, "diarization_num_speakers": num_speakers}
    (meta_dir / f"{stem}.metadata.json").write_text(json.dumps(meta))


def _good_quote(speaker_id: str, ts: int = 1000) -> dict:
    return {"text": "x", "speaker_id": speaker_id, "timestamp_start_ms": ts}


_HOST_GUEST = [
    {"id": "host_1", "name": "Patrick O'Shaughnessy", "role": "host"},
    {"id": "guest", "name": "Brian Chesky", "role": "guest"},
]


def test_clean_corpus_passes(tmp_path: Path) -> None:
    _write_episode(
        tmp_path,
        "0001-ep",
        quotes=[_good_quote("person:patrick-oshaughnessy"), _good_quote("person:brian-chesky")],
        speakers=_HOST_GUEST,
        num_speakers=2,
    )
    m = compute_diarization_quality_metrics(tmp_path)
    assert m["episodes_with_quotes"] == 1
    assert m["quote_attribution_rate"] == 1.0
    assert m["spoken_by_coverage"] == 1.0
    assert m["episodes_unattributed"] == 0
    passed, failures = enforce_diarization_thresholds(m)
    assert passed, failures


def test_unattributed_episode_flagged(tmp_path: Path) -> None:
    # The #545 failure mode: an episode has quotes but 0 attributed speakers + 0 SPOKEN_BY.
    _write_episode(
        tmp_path,
        "0002-ep",
        quotes=[
            {"text": "a", "speaker_id": "", "timestamp_start_ms": 1},
            {"text": "b", "speaker_id": "", "timestamp_start_ms": 2},
        ],
        speakers=_HOST_GUEST,
        num_speakers=2,
        spoken_by=0,
    )
    m = compute_diarization_quality_metrics(tmp_path)
    assert m["episodes_unattributed"] == 1
    assert m["quote_attribution_rate"] == 0.0
    assert m["spoken_by_coverage"] == 0.0
    passed, failures = enforce_diarization_thresholds(m)
    assert not passed
    assert any("0 attributed" in f for f in failures)
    assert any("spoken_by_coverage" in f for f in failures)


def test_partial_spoken_by_coverage_flagged(tmp_path: Path) -> None:
    # Quotes attributed but SPOKEN_BY edges missing (e.g. enrich-edges offset failure).
    _write_episode(
        tmp_path,
        "0003-ep",
        quotes=[_good_quote("person:a"), _good_quote("person:b"), _good_quote("person:c")],
        speakers=_HOST_GUEST,
        num_speakers=2,
        spoken_by=1,  # only 1 of 3 quotes got an edge
    )
    m = compute_diarization_quality_metrics(tmp_path)
    assert abs(m["spoken_by_coverage"] - (1 / 3)) < 1e-6
    passed, failures = enforce_diarization_thresholds(m)
    assert not passed and any("spoken_by_coverage" in f for f in failures)


def test_network_speaker_name_flagged(tmp_path: Path) -> None:
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


def test_unnamed_speaker_flagged(tmp_path: Path) -> None:
    # #876 partial naming: a quote attributed to an unnamed person:speaker-xx while another
    # voice is named → a diarized voice the roster could not name. This MUST fail.
    _write_episode(
        tmp_path,
        "0001-ep",
        quotes=[_good_quote("person:brian-chesky"), _good_quote("person:speaker-02")],
        speakers=[{"id": "guest", "name": "Brian Chesky", "role": "guest"}],
        num_speakers=2,
    )
    m = compute_diarization_quality_metrics(tmp_path)
    assert m["episodes_with_unnamed_speaker"] == 1
    passed, failures = enforce_diarization_thresholds(m)
    assert not passed and any("unnamed speaker" in f for f in failures)


def test_guest_dominated_quotes_all_named_passes(tmp_path: Path) -> None:
    # A 2-speaker interview where GI's quotes are all from the guest (the dominant speaker)
    # is NORMAL — every voice is named, no person:speaker-xx. Must NOT be flagged, even
    # though the quotes span only one named speaker (the old over-strict check failed this).
    _write_episode(
        tmp_path,
        "0001-ep",
        quotes=[_good_quote("person:paul-tudor-jones") for _ in range(5)],
        speakers=_HOST_GUEST,
        num_speakers=2,
    )
    m = compute_diarization_quality_metrics(tmp_path)
    assert m["multispeaker_episodes"] == 1
    assert m["multispeaker_undernamed_episodes"] == 1  # informational only
    assert m["episodes_with_unnamed_speaker"] == 0  # no unnamed voice → not a bug
    assert enforce_diarization_thresholds(m)[0] is True  # passes


def test_missing_num_speakers_opt_in(tmp_path: Path) -> None:
    _write_episode(
        tmp_path,
        "0001-ep",
        quotes=[_good_quote("person:patrick-oshaughnessy"), _good_quote("person:brian-chesky")],
        speakers=_HOST_GUEST,
        num_speakers=None,
    )
    m = compute_diarization_quality_metrics(tmp_path)
    assert m["episodes_missing_num_speakers"] == 1
    assert enforce_diarization_thresholds(m)[0] is True  # not enforced by default
    assert enforce_diarization_thresholds(m, require_num_speakers=True)[0] is False

"""Audio-archive provenance at the download choke point (#1789).

Before this, ``record_provenance`` fired only from ``archive backfill``, so a normally-ingested
corpus had ZERO provenance despite every episode's audio being archived — a reprocess could not
tell an original download from a dynamic-ad re-encode. ``record_pipeline_provenance`` closes
that gap by stamping each pipeline download as an original (byte-identical to the transcript).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.archive import backfill as bf

pytestmark = [pytest.mark.unit]


def _rows(corpus_dir: Path) -> list[dict]:
    path = corpus_dir / ".podcast_scraper" / "audio-archive-provenance.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_pipeline_download_is_stamped_as_original(tmp_path: Path) -> None:
    bf.record_pipeline_provenance(
        str(tmp_path), guid="g1", rel_key="sha256/aa/bb/x.mp3", source_url="https://cdn/ep.mp3"
    )
    rows = _rows(tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["guid"] == "g1"
    assert r["origin"] == "pipeline_download"
    assert r["byte_identical_to_transcribed_audio"] is True
    assert r["source_url"] == "https://cdn/ep.mp3"


def test_empty_guid_records_nothing(tmp_path: Path) -> None:
    bf.record_pipeline_provenance(str(tmp_path), guid="", rel_key=None, source_url="x")
    assert _rows(tmp_path) == []


def test_dedupe_download_is_not_stamped_byte_identical(tmp_path: Path) -> None:
    # H1: store_via deduped against a pre-existing (possibly re-encoded) cold object, so we must
    # NOT claim these bytes are the archived ones.
    bf.record_pipeline_provenance(
        str(tmp_path), guid="g1", rel_key="k", source_url="u", byte_identical=False
    )
    r = _rows(tmp_path)[0]
    assert r["origin"] == "pipeline_download_deduped"
    assert r["byte_identical_to_transcribed_audio"] is False


def test_pipeline_and_backfill_provenance_coexist_and_disagree(tmp_path: Path) -> None:
    # A pipeline original then a later backfill re-fetch for the same guid: both rows are
    # kept, and their byte-identical flags disagree — which is the whole point (a reprocess
    # can see the audio was re-fetched, not the original).
    bf.record_pipeline_provenance(str(tmp_path), guid="g1", rel_key="k", source_url="u1")
    outcome = bf.EpisodeOutcome(
        guid="g1", title="t", feed_title="f", outcome=bf.STORED, rel_key="k", bytes_stored=5
    )
    bf.record_provenance(str(tmp_path), outcome, source_url="u2")

    rows = _rows(tmp_path)
    assert len(rows) == 2
    origins = {r["origin"]: r["byte_identical_to_transcribed_audio"] for r in rows}
    assert origins == {"pipeline_download": True, "backfill_refetch": False}

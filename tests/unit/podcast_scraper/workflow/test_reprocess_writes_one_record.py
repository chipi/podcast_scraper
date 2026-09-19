"""A reprocess stage writes ONE record, where the episode already lives (#2075).

Measured on Ground Truths after a relabel: the run wrote a full metadata artifact into its FRESH
run directory, alone, with an empty ``transcripts/`` beside it, while the transcript it had just
relabelled stayed in the original run. One episode, two records — the thing this arc exists to
prevent — and it broke the next stage, because `rederive_only` resolves the NEWEST record, found no
transcript next to it, and refused an episode whose transcript was two directories away.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from podcast_scraper import config
from podcast_scraper.models import Episode
from podcast_scraper.workflow import run_index
from podcast_scraper.workflow.metadata_generation import _determine_metadata_path

pytestmark = pytest.mark.unit

_GUID = "substack:post:179191415"


def _episode() -> Episode:
    item = ET.Element("item")
    guid = ET.SubElement(item, "guid")
    guid.text = _GUID
    return Episode(
        idx=9,
        title="The Story of Francis Crick",
        title_safe="Crick",
        item=item,
        transcript_urls=[],
    )


def _corpus(tmp_path: Path, *, with_transcript: bool) -> Path:
    """A corpus holding one episode under an OLD run, plus an empty fresh run."""
    corpus = tmp_path / "corpus"
    old = corpus / "feeds" / "rss_feed_abc" / "run_a757c07f_20260901-002926"
    (old / "metadata").mkdir(parents=True)
    (old / "transcripts").mkdir(parents=True)
    stem = "0009 - Crick_a757c07f_20260901-002926"
    (old / "metadata" / f"{stem}.metadata.json").write_text(
        json.dumps({"episode": {"guid": _GUID}, "content": {"speakers": []}}), encoding="utf-8"
    )
    if with_transcript:
        (old / "transcripts" / f"{stem}.txt").write_text("SPEAKER_00: hello\n", encoding="utf-8")
    return corpus


def _cfg(stage: str, corpus: Path) -> config.Config:
    return config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        pipeline_stage=stage,  # type: ignore[arg-type]
        output_dir=str(corpus),
        single_feed_uses_corpus_layout=True,
    )


@pytest.mark.parametrize(
    "stage", ["relabel_only", "rediarize_only", "retranscript_only", "rederive_only"]
)
def test_a_reprocess_stage_writes_the_record_where_the_episode_lives(
    tmp_path: Path, stage: str, monkeypatch
) -> None:
    corpus = _corpus(tmp_path, with_transcript=True)
    monkeypatch.setattr(run_index, "_CORPUS_METADATA_INDEX_CACHE", {})
    fresh_run = corpus / "feeds" / "rss_feed_abc" / "run_20260917-235645"
    (fresh_run / "metadata").mkdir(parents=True)

    path = _determine_metadata_path(
        _episode(), str(fresh_run), "20260917-235645", _cfg(stage, corpus)
    )

    assert "run_a757c07f_20260901-002926" in path, "the record belongs with the episode"
    assert "run_20260917-235645" not in path, "a second record in the fresh run is the defect"


def test_a_normal_run_still_writes_into_its_own_run_directory(tmp_path: Path, monkeypatch) -> None:
    """The fix must not reach a FULL run: that one is producing the episode, not repairing it."""
    corpus = _corpus(tmp_path, with_transcript=True)
    monkeypatch.setattr(run_index, "_CORPUS_METADATA_INDEX_CACHE", {})
    fresh_run = corpus / "feeds" / "rss_feed_abc" / "run_20260917-235645"
    (fresh_run / "metadata").mkdir(parents=True)

    path = _determine_metadata_path(
        _episode(), str(fresh_run), "20260917-235645", _cfg("full", corpus)
    )

    assert "run_20260917-235645" in path


def test_the_transcript_lookup_skips_a_record_with_no_transcript_beside_it(
    tmp_path: Path, monkeypatch
) -> None:
    """The second half of the same failure: an episode whose NEWEST record sits in a run with an
    empty `transcripts/` must still resolve to the transcript in the older run."""
    corpus = _corpus(tmp_path, with_transcript=True)
    monkeypatch.setattr(run_index, "_CORPUS_METADATA_INDEX_CACHE", {})
    stray = corpus / "feeds" / "rss_feed_abc" / "run_20260917-235645"
    (stray / "metadata").mkdir(parents=True)
    (stray / "transcripts").mkdir(parents=True)
    (stray / "metadata" / "0009 - Crick_20260917-235645.metadata.json").write_text(
        json.dumps({"episode": {"guid": _GUID}}), encoding="utf-8"
    )

    found = run_index.existing_transcript_path_in_corpus(_episode(), str(corpus))

    assert found is not None
    assert found.endswith(
        ".txt"
    ), f"resolved the metadata marker instead of the transcript: {found}"
    assert "run_a757c07f_20260901-002926" in found


def test_the_lookup_still_reports_presence_when_no_run_holds_a_transcript(
    tmp_path: Path, monkeypatch
) -> None:
    """With no transcript anywhere, the metadata marker is still the honest answer — the episode
    IS present, and skip-existing needs to know that."""
    corpus = _corpus(tmp_path, with_transcript=False)
    monkeypatch.setattr(run_index, "_CORPUS_METADATA_INDEX_CACHE", {})

    found = run_index.existing_transcript_path_in_corpus(_episode(), str(corpus))

    assert found is not None and found.endswith(".metadata.json")

"""A segment label written before #2220 cannot put an organisation back on a re-derive.

The segments sidecar is the durable record of who a voice is; the speaker record and GI both read
it back on every rederive. Measured on prod: "The Brazilian Report" labels the host voice on 40
Explaining Brazil episodes, and extraction calls it an Organization 23 times and a Person never.
(Not "Andreessen Horowitz": the publisher list already catches that one, vote or no vote.)
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_scraper.gi.pipeline import _resolve_quote_speaker
from podcast_scraper.speaker_detectors.entity_kind_votes import (
    corpus_kind_votes,
    votes_for_output_dir,
    votes_from_kg_payloads,
)
from podcast_scraper.workflow.metadata_generation import _build_speaker_record

pytestmark = [pytest.mark.unit]

ORG = "The Brazilian Report"


def _org_votes(n: int = 54):
    node = {"type": "Organization", "properties": {"name": ORG, "role": "mentioned"}}
    return votes_from_kg_payloads([{"nodes": [node]}] * n)


def _corpus(root: Path) -> Path:
    """Three served episodes voting Organization, plus the run dir under test."""
    for i in range(3):
        meta = root / "feeds" / "other" / f"run_{i}" / "metadata"
        meta.mkdir(parents=True)
        (meta / f"e{i}.metadata.json").write_text(
            json.dumps({"episode": {"episode_id": f"e{i}"}, "feed": {"feed_id": "other"}})
        )
        props = {"name": ORG, "role": "mentioned"}
        node = {"id": "org:a", "type": "Organization", "properties": props}
        (meta / f"e{i}.kg.json").write_text(json.dumps({"nodes": [node]}))
    run = root / "feeds" / "brazil" / "run_x"
    (run / "transcripts").mkdir(parents=True)
    segs = [
        {"start": 0.0, "speaker": "SPEAKER_00", "speaker_label": ORG, "speaker_role": "host"},
        {"start": 5.0, "speaker": "SPEAKER_01", "speaker_label": "Ben Example"},
    ]
    (run / "transcripts" / "ep.segments.json").write_text(json.dumps(segs))
    return run


class TestTheGiQuoteSpeaker:
    def test_an_org_label_is_an_unnamed_voice(self) -> None:
        gq = SimpleNamespace(char_start=0, char_end=5)
        pid, name, voice_type = _resolve_quote_speaker(
            gq, ORG, "ep1", "hello", None, "Explaining Brazil", _org_votes()
        )
        assert pid is None and voice_type == "unknown", (pid, name, voice_type)

    def test_a_person_label_is_unchanged(self) -> None:
        gq = SimpleNamespace(char_start=0, char_end=5)
        pid, _name, voice_type = _resolve_quote_speaker(
            gq, "Ben Example", "ep1", "hello", None, "Explaining Brazil", _org_votes()
        )
        assert pid == "person:ben-example" and voice_type is None

    def test_no_votes_keeps_todays_behaviour(self) -> None:
        gq = SimpleNamespace(char_start=0, char_end=5)
        pid, _name, _vt = _resolve_quote_speaker(
            gq, ORG, "ep1", "hello", None, "Explaining Brazil", None
        )
        assert pid == "person:the-brazilian-report"


class TestTheSpeakerRecord:
    def test_an_org_label_is_not_placed(self, tmp_path: Path) -> None:
        corpus_kind_votes.cache_clear()
        run = _corpus(tmp_path)
        assert votes_for_output_dir(str(run)) is not None
        speakers, num = _build_speaker_record(
            str(run), "transcripts/ep.txt", None, None, "Explaining Brazil"
        )
        assert [s.name for s in speakers if s.placed] == ["Ben Example"]
        assert num == 2  # the voice is still counted — it is real, only its name was wrong

    def test_outside_a_corpus_nothing_changes(self, tmp_path: Path) -> None:
        corpus_kind_votes.cache_clear()
        run = tmp_path / "plain" / "run_x"
        (run / "transcripts").mkdir(parents=True)
        segs = [{"start": 0.0, "speaker": "SPEAKER_00", "speaker_label": ORG}]
        (run / "transcripts" / "ep.segments.json").write_text(json.dumps(segs))
        assert votes_for_output_dir(str(run)) is None
        speakers, _num = _build_speaker_record(str(run), "transcripts/ep.txt", None, None, None)
        assert [s.name for s in speakers if s.placed] == [ORG]

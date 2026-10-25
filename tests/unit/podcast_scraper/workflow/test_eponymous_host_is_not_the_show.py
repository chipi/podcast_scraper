"""A host the show is named after is the host, when their own voice says so.

"The Paul Avery Drive": the title-prefix rule (`names_the_show`) reads "Paul Avery" as the show,
so on prod (2026-10-01) all 40 episodes had no placed host and his quotes were episode-scoped as an
"Unidentified speaker". His voice introduced itself on all 40 (roster source `self_intro`); none of
the real show-name labels did (Machine Learning Street, Trivium China, Africa Tech Summit, Turkey
Book — all `known_hosts` / `llm_resolution`). That is the evidence used, at all three places.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_scraper.gi.pipeline import _resolve_quote_speaker
from podcast_scraper.workflow.metadata_generation import (
    _build_speaker_record,
    _self_introduced_names,
    _speaker_lists_for_graph,
)

pytestmark = [pytest.mark.unit]

SHOW = "The Paul Avery Drive"
HOST = "Paul Avery"


def _run(tmp_path: Path, label: str, source: str) -> Path:
    run = tmp_path / "run_x"
    (run / "transcripts").mkdir(parents=True)
    segs = [
        {"start": 0.0, "speaker": "SPEAKER_00", "speaker_label": label, "speaker_role": "host"},
        {"start": 9.0, "speaker": "SPEAKER_01", "speaker_label": "Rhonda Patrick"},
    ]
    (run / "transcripts" / "ep.segments.json").write_text(json.dumps(segs))
    diag = {
        "voices": [
            {"voice": "SPEAKER_00", "resolved_name": label, "source": source},
            {"voice": "SPEAKER_01", "resolved_name": "Rhonda Patrick", "source": "llm_resolution"},
        ]
    }
    (run / "transcripts" / "ep.speakers.diagnostics.json").write_text(json.dumps(diag))
    return run


class TestTheSpeakerRecord:
    def test_an_e1d_rescue_from_the_hosts_own_introduction_counts_as_self_introduced(
        self, tmp_path: Path
    ) -> None:
        # Second review n2: E1d gives source e1d_rescue; its cue (here the voice's own "I'm
        # Paul Avery") is in the decision trace, and is the same evidence as self_intro.
        run = _run(tmp_path, HOST, "e1d_rescue")
        diag_path = run / "transcripts" / "ep.speakers.diagnostics.json"
        diag = json.loads(diag_path.read_text())
        diag["decision_trace"] = {
            "voices": {
                "SPEAKER_00": [
                    {"rung": "e1d_rescue", "decision": "named", "name": HOST, "cue": "self_intro"}
                ]
            }
        }
        diag_path.write_text(json.dumps(diag))
        speakers, _ = _build_speaker_record(str(run), "transcripts/ep.txt", None, None, SHOW)
        placed = {s.name: (s.role, s.source) for s in speakers if s.placed}
        assert placed[HOST] == ("host", "self_intro")
        assert _self_introduced_names(str(run), "transcripts/ep.txt") == [HOST]

    def test_an_e1d_rescue_introduced_by_another_voice_is_not_self_introduced(
        self, tmp_path: Path
    ) -> None:
        run = _run(tmp_path, HOST, "e1d_rescue")
        diag_path = run / "transcripts" / "ep.speakers.diagnostics.json"
        diag = json.loads(diag_path.read_text())
        diag["decision_trace"] = {
            "voices": {
                "SPEAKER_00": [
                    {"rung": "e1d_rescue", "decision": "named", "name": HOST, "cue": "introduced"}
                ]
            }
        }
        diag_path.write_text(json.dumps(diag))
        speakers, _ = _build_speaker_record(str(run), "transcripts/ep.txt", None, None, SHOW)
        assert HOST not in {s.name for s in speakers if s.placed}

    def test_a_self_introduced_eponymous_host_is_placed(self, tmp_path: Path) -> None:
        run = _run(tmp_path, HOST, "self_intro")
        speakers, _ = _build_speaker_record(str(run), "transcripts/ep.txt", None, None, SHOW)
        placed = {s.name: (s.role, s.source) for s in speakers if s.placed}
        assert placed[HOST] == ("host", "self_intro")

    def test_a_host_split_into_two_voices_keeps_the_self_introduction(self, tmp_path: Path) -> None:
        # One person on two diarized voices; the voice that did NOT introduce itself speaks first.
        # The record used to take the person's source from that first voice, so the show-name
        # check dropped the whole person, the self-introducing voice included.
        run = tmp_path / "run_x"
        (run / "transcripts").mkdir(parents=True)
        segs = [
            {"start": 0.0, "speaker": "SPEAKER_01", "speaker_label": HOST, "speaker_role": "host"},
            {"start": 9.0, "speaker": "SPEAKER_00", "speaker_label": HOST, "speaker_role": "host"},
        ]
        (run / "transcripts" / "ep.segments.json").write_text(json.dumps(segs))
        diag = {
            "voices": [
                {"voice": "SPEAKER_00", "resolved_name": HOST, "source": "self_intro"},
                {"voice": "SPEAKER_01", "resolved_name": HOST, "source": "llm_resolution"},
            ]
        }
        (run / "transcripts" / "ep.speakers.diagnostics.json").write_text(json.dumps(diag))
        speakers, _ = _build_speaker_record(str(run), "transcripts/ep.txt", None, None, SHOW)
        placed = {s.name: (s.role, s.source) for s in speakers if s.placed}
        assert placed[HOST] == ("host", "self_intro")

    def test_a_show_name_the_voice_never_said_is_still_refused(self, tmp_path: Path) -> None:
        run = _run(tmp_path, "Machine Learning Street", "known_hosts")
        speakers, _ = _build_speaker_record(
            str(run), "transcripts/ep.txt", None, None, "Machine Learning Street Talk (MLST)"
        )
        assert "Machine Learning Street" not in {s.name for s in speakers if s.placed}


class TestTheGraphBoundary:
    def test_the_self_introduced_host_reaches_the_graph(self) -> None:
        sp = SimpleNamespace(name=HOST, role="host", placed=True, source="self_intro")
        hosts, _guests = _speaker_lists_for_graph([sp], SHOW)
        assert hosts == [HOST]

    def test_a_roster_named_show_still_does_not(self) -> None:
        sp = SimpleNamespace(name="Trivium China", role="host", placed=True, source="known_hosts")
        hosts, _guests = _speaker_lists_for_graph([sp], "The Trivium China Podcast")
        assert hosts == []


class TestGi:
    def test_a_self_introduced_host_keeps_a_corpus_wide_id(self) -> None:
        gq = SimpleNamespace(char_start=0, char_end=5)
        pid, name, _vt = _resolve_quote_speaker(gq, HOST, "ep1", "hello", None, SHOW, None, [HOST])
        assert pid == "person:paul-avery" and name is None

    def test_without_the_self_intro_it_is_still_episode_scoped(self) -> None:
        gq = SimpleNamespace(char_start=0, char_end=5)
        pid, _name, _vt = _resolve_quote_speaker(gq, HOST, "ep1", "hello", None, SHOW, None, [])
        assert pid is not None and pid.startswith("person:unresolved-")

    def test_names_come_from_the_rosters_own_diagnostics(self, tmp_path: Path) -> None:
        run = _run(tmp_path, HOST, "self_intro")
        assert _self_introduced_names(str(run), "transcripts/ep.txt") == [HOST]

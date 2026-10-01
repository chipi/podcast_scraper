"""A host the show is named after is the host, when their own voice says so.

"The Peter Attia Drive": the title-prefix rule (`names_the_show`) reads "Peter Attia" as the show,
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

SHOW = "The Peter Attia Drive"
HOST = "Peter Attia"


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
    def test_a_self_introduced_eponymous_host_is_placed(self, tmp_path: Path) -> None:
        run = _run(tmp_path, HOST, "self_intro")
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
        assert pid == "person:peter-attia" and name is None

    def test_without_the_self_intro_it_is_still_episode_scoped(self) -> None:
        gq = SimpleNamespace(char_start=0, char_end=5)
        pid, _name, _vt = _resolve_quote_speaker(gq, HOST, "ep1", "hello", None, SHOW, None, [])
        assert pid is not None and pid.startswith("person:unresolved-")

    def test_names_come_from_the_rosters_own_diagnostics(self, tmp_path: Path) -> None:
        run = _run(tmp_path, HOST, "self_intro")
        assert _self_introduced_names(str(run), "transcripts/ep.txt") == [HOST]

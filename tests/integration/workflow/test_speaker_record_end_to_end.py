"""The speaker record, end to end through real metadata generation (#2075).

Drives `generate_episode_metadata` over a real transcript, segments sidecar and speakers-diagnostics
sidecar, captures what the knowledge-graph builder is actually TOLD, and feeds the written artifact
to the safety readers. The unit tests cover each rule; this proves the rules are wired together:
the record written to disk and the cast handed to the graph are the same thing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from podcast_scraper.kg import speaker_coherence
from podcast_scraper.upgrade.migrations import m0009_backfill_speaker_roles as m0009
from podcast_scraper.workflow import metadata_generation as metadata

pytestmark = [pytest.mark.integration]

_tests_dir = Path(__file__).parent.parent.parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))
_spec = importlib.util.spec_from_file_location("parent_conftest", _tests_dir / "conftest.py")
assert _spec is not None and _spec.loader is not None
_pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pc)

REL = "transcripts/0001 - Episode_Title.txt"


def _seg(label: str, voice: str, role: Optional[str], text: str) -> Dict[str, Any]:
    return {
        "start": 0.0,
        "end": 1.0,
        "text": text,
        "speaker": voice,
        "speaker_label": label,
        "speaker_role": role,
    }


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    segments: Optional[List[Dict[str, Any]]],
    diagnostics: Optional[Dict[str, Any]],
    hint_hosts: List[str],
    hint_guests: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out = tmp_path / "out"
    (out / "transcripts").mkdir(parents=True)
    (out / REL).write_text("SPEAKER_00: hello\n", encoding="utf-8")
    base = str(out / REL)[: -len(".txt")]
    if segments is not None:
        Path(base + ".segments.json").write_text(json.dumps(segments), encoding="utf-8")
    if diagnostics is not None:
        Path(base + ".speakers.diagnostics.json").write_text(
            json.dumps(diagnostics), encoding="utf-8"
        )
    told: Dict[str, Any] = {}

    import podcast_scraper.kg as kg_mod

    def _spy(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        told.update(kwargs)
        return {
            "schema_version": "2.1",
            "episode_id": args[0] if args else "ep",
            "nodes": [],
            "edges": [],
        }

    monkeypatch.setattr(kg_mod, "build_artifact", _spy)
    cfg = _pc.create_test_config(
        output_dir=str(out),
        generate_metadata=True,
        metadata_format="json",
        generate_kg=True,
        kg_extraction_source="metadata_only",
    )
    path = metadata.generate_episode_metadata(
        feed=_pc.create_test_feed(),
        episode=_pc.create_test_episode(),
        feed_url=_pc.TEST_FEED_URL,
        cfg=cfg,
        output_dir=str(out),
        run_suffix=None,
        transcript_file_path=REL,
        transcript_source="whisper_transcription",
        whisper_model="base",
        detected_hosts=hint_hosts,
        detected_guests=hint_guests,
    )
    assert path, "no artifact written"
    data: Dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    assert told, "the knowledge-graph builder was never called — the test missed the seam"
    return data, told


def _record(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {s["name"]: s for s in data["content"]["speakers"]}


class TestTheDailyShape:
    """One host placed, two feed hosts refused, a guest corroboration refused."""

    @pytest.fixture()
    def run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        return _run(
            tmp_path,
            monkeypatch,
            segments=[
                _seg("Michael Barbaro", "SPEAKER_02", "host", "I'm Michael Barbaro."),
                _seg("SPEAKER_00", "SPEAKER_00", None, "It is a pleasure."),
            ],
            diagnostics={
                "voices": [{"voice": "SPEAKER_02", "source": "self_intro"}],
                "tried": {"known_hosts": ["Michael Barbaro", "Natalie Kitroeff", "Rachel Abrams"]},
                "summary": {"unbound_names": ["Matina Stevis-Gridneff"]},
            },
            hint_hosts=["Michael Barbaro", "Natalie Kitroeff"],
            hint_guests=["Matina Stevis-Gridneff"],
        )

    def test_the_record_keeps_everyone_and_marks_who_was_placed(self, run) -> None:
        data, _told = run
        rec = _record(data)
        assert rec["Michael Barbaro"]["placed"] is True
        assert rec["Michael Barbaro"]["voices"] == ["SPEAKER_02"]
        for name, source in (
            ("Natalie Kitroeff", "feed_statement"),
            ("Rachel Abrams", "feed_statement"),
            ("Matina Stevis-Gridneff", "episode_metadata"),
        ):
            assert rec[name]["placed"] is False, name
            assert rec[name]["source"] == source, name
        assert data["content"]["speakers_source"] == "diarized"
        assert "detected_hosts" not in data["content"]
        assert "detected_guests" not in data["content"]

    def test_the_graph_is_told_only_the_placed_voice(self, run) -> None:
        _data, told = run
        assert list(told.get("detected_hosts") or []) == ["Michael Barbaro"]
        assert list(told.get("detected_guests") or []) == []

    def test_the_written_artifact_is_safe_for_the_spoke_check_and_m0009(self, run) -> None:
        data, _told = run
        assert speaker_coherence.roster_names(data) == ["Michael Barbaro"]
        roles = m0009.roster_roles(data)
        assert list(roles.values()) == ["host"]


class TestTheA16zShape:
    """Diarized, nobody named: the guess used to become the episode's whole cast."""

    def test_nobody_is_cast_and_everyone_is_kept_as_named(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data, told = _run(
            tmp_path,
            monkeypatch,
            segments=[
                _seg("SPEAKER_00", "SPEAKER_00", None, "a"),
                _seg("SPEAKER_01", "SPEAKER_01", None, "b"),
            ],
            diagnostics={"voices": []},
            hint_hosts=["Erik Torenberg", "Ben Horowitz"],
            hint_guests=["Garry Tan"],
        )
        assert (
            list(told.get("detected_hosts") or []),
            list(told.get("detected_guests") or []),
        ) == ([], [])
        rec = _record(data)
        assert set(rec) == {"Erik Torenberg", "Ben Horowitz", "Garry Tan"}
        assert all(e["placed"] is False for e in rec.values())
        assert data["content"]["speakers_source"] == "hint"
        assert speaker_coherence.roster_names(data) == []
        assert m0009.roster_roles(data) == {}


class TestTheNeverDiarizedShape:
    """Operator decision 2026-09-17: no voices at all means nobody is cast."""

    def test_a_publisher_transcript_episode_casts_nobody(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data, told = _run(
            tmp_path,
            monkeypatch,
            segments=None,
            diagnostics=None,
            hint_hosts=["Joe Weisenthal"],
            hint_guests=["Austan Goolsbee"],
        )
        assert (
            list(told.get("detected_hosts") or []),
            list(told.get("detected_guests") or []),
        ) == ([], [])
        assert [(e["name"], e["placed"]) for e in data["content"]["speakers"]] == [
            ("Joe Weisenthal", False),
            ("Austan Goolsbee", False),
        ]

"""Can each guard actually SEE the thing it claims to check? (#2065 arc, class C5)

Every serious miss in the host/guest arc was one shape: a check that could not observe what it
claimed to observe. Not "the logic mishandled bad input" — the logic never ran, or ran against
nothing, and returned a plausible answer anyway.

    incident                              the blindness
    ---------------------------------------------------------------------------------
    guard read SPOKEN_BY from kg.json     that key lives only in gi.json (0 of 287)
    "82->56 violations, 0 introduced"     the check shares the migration's predicate, so a
                                          wrong demotion scores as a violation FIXED
    undo reported "0 refused"             the staging corpus had no segments sidecars at all
    _speaker_lists_for_graph got dicts    it reads getattr(sp, "name") -> everything empty, so
                                          every comparison silently compared nothing
    enrich-edges omitted feed_title       names_the_show returns False on an empty title BY
                                          DESIGN, so the guard it was calling was inert

Four of those were mine, and every one passed its own tests.

THE ASSERTION SHAPE. A positive case proves a guard fires. A negative case proves it does not
misfire. Neither proves the guard is CONNECTED. So each test here feeds a guard the signal it
depends on and asserts the answer CHANGES:

    assert guard(with_signal) != guard(without_signal)

A guard whose output is identical with and without its input is inert, whatever it returns. That
is the property none of the five incidents above would have survived.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


class TestTheShowNameGuardNeedsATitle:
    """`names_the_show` is a no-op without `feed_title` — so every caller must pass one."""

    def test_the_answer_changes_when_the_title_is_supplied(self) -> None:
        from types import SimpleNamespace

        from podcast_scraper.workflow.metadata_generation import _speaker_lists_for_graph

        roster = [SimpleNamespace(name="Africa Tech Summit", role="host")]
        without, _ = _speaker_lists_for_graph(roster, [], [])
        with_title, _ = _speaker_lists_for_graph(
            roster, [], [], feed_title="Africa Tech Summit Podcast"
        )
        assert without != with_title, "the title is the signal; if it changes nothing, it is inert"


class TestTheVoiceGuardReadsTheLayerThatHasTheEdges:
    """`SPOKEN_BY` is in gi.json. A guard reading kg.json sees nothing, on every real artifact."""

    def test_the_answer_changes_when_the_gi_payload_is_supplied(self) -> None:
        from podcast_scraper.upgrade.migrations.m0009_backfill_speaker_roles import (
            voices_in_episode,
        )

        kg = {
            "nodes": [{"id": "person:x", "type": "Person", "properties": {"name": "X"}}],
            "edges": [{"type": "HOSTS", "from": "person:x", "to": "podcast:p"}],
        }
        gi = {
            "nodes": [{"id": "person:x", "type": "Person", "properties": {"name": "X"}}],
            "edges": [{"type": "SPOKEN_BY", "from": "quote:q", "to": "person:x"}],
        }
        assert voices_in_episode({}, kg) == set(), "kg alone carries no voice evidence"
        assert voices_in_episode(gi, kg) == {"person:x"}

    def test_a_real_kg_artifact_has_no_spoken_by(self) -> None:
        # Pins the fact the broken guard assumed away. If a future kg writer starts emitting
        # SPOKEN_BY this fails, and the guard above can be simplified — deliberately.
        import inspect

        from podcast_scraper.kg import pipeline as kg_pipeline

        src = inspect.getsource(kg_pipeline)
        assert (
            "SPOKEN_BY" not in src
        ), "kg/pipeline does not emit SPOKEN_BY; a guard that reads it from kg.json is inert"


class TestTheVoiceCountNeedsProof:
    """`voices_heard` must read the sidecar, not the field that cannot tell count from fallback."""

    def test_the_answer_changes_when_the_sidecar_exists(self, tmp_path: Path) -> None:
        from podcast_scraper.upgrade.migrations.m0009_backfill_speaker_roles import voices_heard

        (tmp_path / "metadata").mkdir(parents=True)
        (tmp_path / "transcripts").mkdir(parents=True)
        meta = {
            "content": {
                "speakers": [{"name": "A", "role": "host"}],
                "diarization_num_speakers": 2,
                "transcript_file_path": "transcripts/e1.txt",
            }
        }
        path = tmp_path / "metadata" / "e1.metadata.json"
        path.write_text(json.dumps(meta), encoding="utf-8")
        assert voices_heard(meta, path) is None, "no sidecar is no evidence, not the field"

        (tmp_path / "transcripts" / "e1.segments.json").write_text(
            json.dumps([{"speaker": f"SPEAKER_0{i}"} for i in range(4)]), encoding="utf-8"
        )
        assert voices_heard(meta, path) == 4, "the sidecar is the signal"


class TestTheMigrationCannotGradeItself:
    """A metric sharing the migration's predicate cannot see a wrongful demotion.

    Demoting a REAL host removes the very condition `check_no_show_as_speaker` looks for, so it
    scores as a violation FIXED. The count can only ever go down for this class of damage.
    """

    def test_a_wrongful_demotion_reduces_the_violation_count(self) -> None:
        from podcast_scraper.kg.speaker_coherence import check_no_show_as_speaker

        meta = {
            "feed": {"title": "Lex Fridman Podcast"},
            "content": {"speakers": [{"name": "Lex Fridman", "role": "host"}]},
        }

        def kg(role: str) -> dict:
            return {
                "nodes": [
                    {
                        "id": "person:lex-fridman",
                        "type": "Person",
                        "properties": {"name": "Lex Fridman", "role": role},
                    }
                ],
                "edges": [],
            }

        before = check_no_show_as_speaker(meta, kg("host"))
        after = check_no_show_as_speaker(meta, kg("mentioned"))
        assert len(after) < len(before), (
            "documents the blindness: wrongly demoting a real host READS as an improvement, "
            "which is why the role-transition table is the instrument, not this count"
        )


class TestTheseTestsWouldHaveCaughtTheRealDefects:
    """Guard-removal proofs. A negative test that still passes with its guard deleted is decoration.

    This session produced four such tests, so the matrix does not get to assert its own value —
    each capability check above is re-run here against a DELIBERATELY DISABLED guard, and must
    fail. If one of these `pytest.raises(AssertionError)` blocks stops raising, the capability
    check above it has stopped checking anything.
    """

    def test_disabling_the_show_name_guard_breaks_the_title_capability_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace

        from podcast_scraper.speaker_detectors import hosts
        from podcast_scraper.workflow.metadata_generation import _speaker_lists_for_graph

        # Simulate the #2064 regression: the predicate exists but never fires.
        monkeypatch.setattr(hosts, "names_the_show", lambda *a, **k: False)

        roster = [SimpleNamespace(name="Africa Tech Summit", role="host")]
        without, _ = _speaker_lists_for_graph(roster, [], [])
        with_title, _ = _speaker_lists_for_graph(
            roster, [], [], feed_title="Africa Tech Summit Podcast"
        )
        with pytest.raises(AssertionError):
            assert without != with_title, "(this is the assertion the capability check makes)"

    def test_reading_voices_from_the_wrong_layer_breaks_the_voice_capability_check(self) -> None:
        # The exact shipped defect: read SPOKEN_BY from the kg payload instead of the gi sibling.
        def broken_voices_in_episode(gi_payload: dict, kg_payload: dict) -> set:
            out = set()
            for edge in kg_payload.get("edges") or []:  # <- wrong layer
                if edge.get("type") == "SPOKEN_BY":
                    out.add(str(edge.get("to") or ""))
            return out

        kg = {
            "nodes": [{"id": "person:x", "type": "Person", "properties": {"name": "X"}}],
            "edges": [{"type": "HOSTS", "from": "person:x", "to": "podcast:p"}],
        }
        gi = {
            "nodes": [{"id": "person:x", "type": "Person", "properties": {"name": "X"}}],
            "edges": [{"type": "SPOKEN_BY", "from": "quote:q", "to": "person:x"}],
        }
        with pytest.raises(AssertionError):
            assert broken_voices_in_episode(gi, kg) == {
                "person:x"
            }, "a kg-only reader returns the empty set on every real artifact"

    def test_trusting_the_field_breaks_the_voice_count_capability_check(
        self, tmp_path: Path
    ) -> None:
        # The pre-#2070 behaviour: return `diarization_num_speakers` whatever its provenance.
        def broken_voices_heard(meta: dict, _path: Path) -> int | None:
            return (meta.get("content") or {}).get("diarization_num_speakers")

        meta = {
            "content": {
                "speakers": [{"name": "A", "role": "host"}],
                "diarization_num_speakers": 2,
                "transcript_file_path": "transcripts/e1.txt",
            }
        }
        path = tmp_path / "metadata" / "e1.metadata.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(AssertionError):
            assert (
                broken_voices_heard(meta, path) is None
            ), "with no sidecar the answer must be None; the field is not evidence"

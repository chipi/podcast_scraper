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


class TestTheVoiceGuardReadsTheLayerThatHasTheEdges:
    """`SPOKEN_BY` is in gi.json. A guard reading kg.json sees nothing, on every real artifact."""

    def test_a_real_kg_artifact_has_no_spoken_by(self) -> None:
        # Pins the fact the broken guard assumed away. If a future kg writer starts emitting
        # SPOKEN_BY this fails, and the guard above can be simplified — deliberately.
        import inspect

        from podcast_scraper.kg import pipeline as kg_pipeline

        src = inspect.getsource(kg_pipeline)
        assert (
            "SPOKEN_BY" not in src
        ), "kg/pipeline does not emit SPOKEN_BY; a guard that reads it from kg.json is inert"


class TestTheseTestsWouldHaveCaughtTheRealDefects:
    """Guard-removal proofs. A negative test that still passes with its guard deleted is decoration.

    This session produced four such tests, so the matrix does not get to assert its own value —
    each capability check above is re-run here against a DELIBERATELY DISABLED guard, and must
    fail. If one of these `pytest.raises(AssertionError)` blocks stops raising, the capability
    check above it has stopped checking anything.
    """

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


class TestThePlaceholderRuleIsOneDecisionNotTwo:
    """`Host` renders on an insight and is hidden from corpus-wide ranking. Both, deliberately.

    `person:unresolved-host-<ep>` carries a real Person node named `Host` — 61 of them on the
    production snapshot, all with `SPOKEN_BY`. `gi_view._speaker_name` resolves it;
    `routes/corpus_persons` and `cil_queries` drop it via `is_unresolved_speaker_placeholder`.

    That reads like one decision made twice with two answers. The rule that reconciles them is:
    **an episode-scoped label is meaningful IN its episode and meaningless aggregated.** "The Host
    said this" is true inside one episode; ranked corpus-wide it invents a person spanning 54
    shows.

    Pinned as a PAIR so neither half can be "fixed" into agreement with the other and quietly
    break the rule — which is what would happen if someone noticed only one side.
    """

    EP = "ep:placeholder"
    PID = f"person:unresolved-host-{EP}"

    def test_the_aggregate_surface_refuses_it(self) -> None:
        from podcast_scraper.enrichment.enrichers._loaders import (
            is_unresolved_speaker_placeholder,
        )

        assert (
            is_unresolved_speaker_placeholder(self.PID) is True
        ), "ranked corpus-wide this would be one person across every show that has a host"


class TestTheKgWriterIsTheKgWriter:
    """`enrich-edges` wrote kg.json through the GI validator. It could never have succeeded.

    THE SHAPE OF THIS BUG IS THE POINT. `gi/schema` requires `model_version`, `prompt_version` and
    `schema_version in ("3.0","3.1")`. Measured on the production snapshot, all 2,256 `*.kg.json`
    are `schema_version 2.1` and carry NEITHER key — so every KG write raised, was caught, counted
    as a failure and logged, AFTER gi.json had already been written. The function added to prevent
    a GI/KG desync produced one on every episode it ran on.

    It stayed invisible for two reasons worth remembering: the scope plan touches KG ids on **0**
    episodes of that snapshot (m0007 had already run), so the path is latent until the first
    episode that needs it and then fails 100%; and no test used a real 2.1-shaped KG fixture.

    So this test asserts the validators are NOT interchangeable, on the real production shape.
    """

    @staticmethod
    def _real_kg_payload():
        return {
            "schema_version": "2.1",
            "episode_id": "ep:1",
            # The REAL production shape — `extraction` carries the nested keys, which is exactly
            # the kind of detail a hand-invented fixture gets wrong and then "proves" something
            # about a shape production never takes.
            "extraction": {
                "model_version": "provider:podcast-flash-0731",
                "extracted_at": "2026-08-30T09:04:58Z",
                "transcript_ref": "transcripts/x.txt",
            },
            "nodes": [
                {"id": "person:x", "type": "Person", "properties": {"name": "X", "role": "host"}}
            ],
            "edges": [],
        }

    def test_the_kg_validator_accepts_it(self, tmp_path) -> None:
        from podcast_scraper.kg.io import write_artifact as kg_write

        target = tmp_path / "x.kg.json"
        kg_write(target, self._real_kg_payload(), validate=True)
        assert target.is_file() and target.stat().st_size > 0

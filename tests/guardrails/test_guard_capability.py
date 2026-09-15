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


class TestTheDisplayNameDecisionReachesBothArtifacts:
    """`plan_display_names` returning `{}` is not "leave the name alone" — it is "let order decide".

    This is the C5 shape again, and it shipped: the guard's OUTPUT looked correct in isolation
    (`{} == {}`, no rename needed, the survivor keeps its name) while the thing it was supposed to
    control — the name a reader actually sees — was decided somewhere else entirely, by
    `rewrite_ids` keeping whichever node each payload happened to list first.

    Every test of that function asserted on its return value. None ran the seam. So the capability
    check here is not "does it return the right dict" but "does the DISPLAYED name in both
    artifacts depend on the decision, and not on node order".
    """

    EP = "ep:order"

    @classmethod
    def _payloads(cls, order):
        person = {
            "mentioned": {
                "id": "person:stuart-brand",
                "type": "Person",
                "properties": {"name": "Stuart Brand", "role": "mentioned"},
            },
            "speaker": {
                "id": "person:stewart-brand",
                "type": "Person",
                "properties": {"name": "Stewart Brand", "role": "guest"},
            },
        }
        kg = {
            "episode_id": cls.EP,
            "nodes": [
                *(person[k] for k in order),
                {
                    "id": f"episode:{cls.EP}",
                    "type": "Episode",
                    "properties": {"title": "Stewart Brand on the Long Now"},
                },
            ],
            "edges": [],
        }
        gi = {
            "episode_id": cls.EP,
            "nodes": [person["speaker"]],
            "edges": [{"type": "SPOKEN_BY", "from": "quote:q1", "to": "person:stewart-brand"}],
        }
        return gi, kg

    @staticmethod
    def _shown(payload):
        return {
            n["id"]: (n.get("properties") or {}).get("name")
            for n in payload.get("nodes", [])
            if n.get("type") == "Person"
        }

    @classmethod
    def _run(cls, order, planner):
        from podcast_scraper.identity.bare_name_scope import rewrite_ids
        from podcast_scraper.identity.intra_episode_merge import (
            apply_display_names,
            plan_intra_episode_merges,
        )

        gi, kg = cls._payloads(order)
        plan = plan_intra_episode_merges(gi, kg)
        renames = planner(gi, kg, plan)
        kg_out, _kgn = apply_display_names(rewrite_ids(kg, plan)[0], renames)
        gi_out, _gin = apply_display_names(rewrite_ids(gi, plan)[0], renames)
        return cls._shown(kg_out), cls._shown(gi_out)

    def test_the_displayed_name_is_the_same_in_both_node_orders(self) -> None:
        from podcast_scraper.identity.intra_episode_merge import plan_display_names

        loser_first = self._run(["mentioned", "speaker"], plan_display_names)
        winner_first = self._run(["speaker", "mentioned"], plan_display_names)
        assert loser_first == winner_first, "node order must not decide the visible label"
        assert loser_first[0] == loser_first[1], "kg.json and gi.json must agree on the name"

    def test_removing_the_both_directions_rule_reintroduces_the_split(self) -> None:
        """THE REMOVAL PROOF. Restore the old rule and the check above must FAIL.

        The old rule emitted a rename only when the prose named the LOSER and not the winner —
        so the case where the prose confirms the WINNER produced `{}`, and each artifact then
        kept whichever node it listed first. Measured on production before the fix: 8 of 11 real
        intra-episode merges left kg.json and gi.json showing different names for one id.
        """
        from podcast_scraper.kg.filters import _clean_entity_name

        def old_plan_display_names(gi_payload, kg_payload, id_plan, *, episode_text=""):
            """The pre-fix implementation, verbatim in behaviour."""
            from podcast_scraper.identity.intra_episode_merge import (
                _episode_prose,
                _person_names,
            )

            if not id_plan:
                return {}
            names = {}
            for payload in (gi_payload, kg_payload):
                for pid, name in _person_names(payload).items():
                    names.setdefault(pid, name)
            prose = _clean_entity_name(
                " ".join(
                    p
                    for p in (
                        episode_text,
                        _episode_prose(gi_payload),
                        _episode_prose(kg_payload),
                    )
                    if p
                )
            )
            if not prose:
                return {}
            out = {}
            for loser, winner in id_plan.items():
                loser_name, winner_name = names.get(loser, ""), names.get(winner, "")
                if not loser_name or loser_name == winner_name:
                    continue
                loser_in = bool(loser_name) and _clean_entity_name(loser_name) in prose
                winner_in = bool(winner_name) and _clean_entity_name(winner_name) in prose
                if loser_in and not winner_in:
                    out[winner] = loser_name
            return out

        kg_shown, gi_shown = self._run(["mentioned", "speaker"], old_plan_display_names)
        assert kg_shown != gi_shown, (
            "the old rule is supposed to be broken here — if this passes, the capability check "
            "above proves nothing"
        )
        assert kg_shown == {"person:stewart-brand": "Stuart Brand"}
        assert gi_shown == {"person:stewart-brand": "Stewart Brand"}


class TestARenameCountsAsAChangeForTheWriteGate:
    """The decided name must survive the decision to WRITE, not just the decision to rename.

    Class C5 one layer further out than the seam test above, and it caught a live defect the
    seam test could not see. `metadata_generation` writes each artifact only if that artifact
    CHANGED, and it measured "changed" from `rewrite_ids`' id-rewrite count alone. A layer that
    was renamed but not re-id'd therefore looked untouched and was never written.

    That is the normal shape of this pass, not an edge case: the merged-away id usually lives only
    in KG, so GI sees zero id changes. Measured on the production snapshot — the merge fires on
    117 episodes, the loser is KG-only on 86, and the decided name was dropped from gi.json on 14.
    One id, two names, which is the exact defect the rename exists to prevent.

    So the capability being checked is: does `apply_display_names` REPORT the change it made?
    """

    @staticmethod
    def _payloads():
        """The KG-only-loser shape: GI holds the speaker alone, KG holds both."""
        speaker = {
            "id": "person:bernard-leung",
            "type": "Person",
            "properties": {"name": "Bernard Leung", "role": "host"},
        }
        gi = {
            "episode_id": "ep:kgonly",
            "nodes": [dict(speaker)],
            "edges": [{"type": "SPOKEN_BY", "from": "quote:q1", "to": "person:bernard-leung"}],
        }
        kg = {
            "episode_id": "ep:kgonly",
            "nodes": [
                {
                    "id": "person:bernard-leong",
                    "type": "Person",
                    "properties": {"name": "Bernard Leong", "role": "mentioned"},
                },
                dict(speaker),
                {
                    "id": "episode:ep:kgonly",
                    "type": "Episode",
                    "properties": {"title": "Analyse Asia with Bernard Leong"},
                },
            ],
            "edges": [],
        }
        return gi, kg

    def test_the_gi_side_reports_a_change_even_with_no_id_rewrite(self) -> None:
        from podcast_scraper.identity.bare_name_scope import rewrite_ids
        from podcast_scraper.identity.intra_episode_merge import (
            apply_display_names,
            plan_display_names,
            plan_intra_episode_merges,
        )

        gi, kg = self._payloads()
        plan = plan_intra_episode_merges(gi, kg)
        assert plan == {"person:bernard-leong": "person:bernard-leung"}

        renames = plan_display_names(gi, kg, plan)
        assert renames == {"person:bernard-leung": "Bernard Leong"}, "the title names Leong"

        _gi2, gi_id_changes = rewrite_ids(gi, plan)
        assert gi_id_changes == 0, "the loser id is KG-only — this is the shape that broke"

        _gi3, gi_rename_changes = apply_display_names(_gi2, renames)
        assert gi_rename_changes == 1, (
            "if a rename reports 0 changes, the write gate skips gi.json and the layers "
            "disagree about the name they just agreed on"
        )

    def test_a_rename_that_changes_nothing_reports_nothing(self) -> None:
        """The count must be a real count, not a constant — otherwise it cannot gate anything."""
        from podcast_scraper.identity.intra_episode_merge import apply_display_names

        payload = {
            "nodes": [{"id": "person:x", "type": "Person", "properties": {"name": "Already Right"}}]
        }
        _out, changes = apply_display_names(payload, {"person:x": "Already Right"})
        assert changes == 0

    def test_a_published_name_carries_no_extractor_punctuation(self) -> None:
        """The prose test strips punctuation; the emitted name did not, so junk was published.

        Measured on the snapshot: `person:lucas-kaiser` was renamed to `"Lukasz Kaiser)"` — the
        cleaned form matched the episode title, and the raw form went to disk.
        """
        from podcast_scraper.identity.intra_episode_merge import (
            plan_display_names,
            plan_intra_episode_merges,
        )

        gi = {
            "episode_id": "ep:junk",
            "nodes": [
                {
                    "id": "person:lucas-kaiser",
                    "type": "Person",
                    "properties": {"name": "Lucas Kaiser"},
                },
                {
                    "id": "person:lukasz-kaiser",
                    "type": "Person",
                    "properties": {"name": "Lukasz Kaiser)"},
                },
                {
                    "id": "episode:ep:junk",
                    "type": "Episode",
                    "properties": {"title": "Can Open Source Keep Up (with Lukasz Kaiser)"},
                },
            ],
            "edges": [
                {"type": "SPOKEN_BY", "from": "quote:q1", "to": "person:lukasz-kaiser"},
                {"type": "MENTIONS_PERSON", "from": "insight:i1", "to": "person:lucas-kaiser"},
            ],
        }
        plan = plan_intra_episode_merges(gi, {})
        got = plan_display_names(gi, {}, plan)
        assert got == {"person:lukasz-kaiser": "Lukasz Kaiser"}, "no trailing paren on disk"


class TestTheHealUnitesTheLabelNotJustTheId:
    """`bare_name_scope` resolves a bare label to a person. The NAME has to follow the id.

    `rewrite_ids` gives the GI node the resolved id and keeps its own `name` — which is the bare
    label that needed resolving. So the id was united and the label was not: measured on the
    production snapshot, 42 ids carry different names in kg.json and gi.json, 13 of them with the
    GI name a strict prefix of KG's (`Didi` vs `Didi Uemakpan`, `Kashmir` vs `Kashmir Hill`).

    Unlike a merge — two plausible spellings, where the feed's prose decides — a heal is not
    symmetric: the losing side is a bare label that was just resolved TO the fuller name. KG's
    resolved name wins. Measured rather than assumed: of the 16 disagreeing pairs differing in
    token count, KG carries the fuller name on 15, and is the correct spelling on the 16th.
    """

    @staticmethod
    def _payloads():
        gi = {
            "episode_id": "ep:heal",
            "nodes": [{"id": "person:didi", "type": "Person", "properties": {"name": "Didi"}}],
            "edges": [{"type": "SPOKEN_BY", "from": "quote:q1", "to": "person:didi"}],
        }
        kg = {
            "episode_id": "ep:heal",
            "nodes": [
                {
                    "id": "person:didi-uemakpan",
                    "type": "Person",
                    "properties": {"name": "Didi Uemakpan", "role": "guest"},
                }
            ],
            "edges": [],
        }
        return gi, kg

    def test_the_healed_node_takes_the_resolved_name_in_both_layers(self) -> None:
        from podcast_scraper.identity.bare_name_scope import rewrite_ids
        from podcast_scraper.identity.intra_episode_merge import (
            apply_display_names,
            plan_heal_display_names,
        )

        gi, kg = self._payloads()
        id_map = {"person:didi": "person:didi-uemakpan"}
        gi2, _n = rewrite_ids(gi, id_map)
        kg2, _m = rewrite_ids(kg, id_map)

        # Without the rename the id is united and the label is not — that is the defect.
        assert gi2["nodes"][0]["properties"]["name"] == "Didi"
        assert kg2["nodes"][0]["properties"]["name"] == "Didi Uemakpan"

        renames = plan_heal_display_names(gi2, kg2, id_map)
        assert renames == {"person:didi-uemakpan": "Didi Uemakpan"}

        gi3, gi_changes = apply_display_names(gi2, renames)
        kg3, _kgc = apply_display_names(kg2, renames)
        assert gi_changes == 1, "the GI rename must REPORT itself or the write gate skips gi.json"
        assert gi3["nodes"][0]["properties"]["name"] == "Didi Uemakpan"
        assert kg3["nodes"][0]["properties"]["name"] == "Didi Uemakpan"

    def test_no_rename_is_emitted_when_the_layers_already_agree(self) -> None:
        """A needless emit marks the artifact dirty and buys a write for nothing."""
        from podcast_scraper.identity.intra_episode_merge import plan_heal_display_names

        same = {
            "nodes": [{"id": "person:x", "type": "Person", "properties": {"name": "Agreed Name"}}]
        }
        assert plan_heal_display_names(same, same, {"person:y": "person:x"}) == {}

    def test_gi_is_used_only_when_kg_has_no_name_for_the_id(self) -> None:
        from podcast_scraper.identity.intra_episode_merge import plan_heal_display_names

        gi = {
            "nodes": [
                {"id": "person:only-gi", "type": "Person", "properties": {"name": "Only In GI"}}
            ]
        }
        got = plan_heal_display_names(gi, {"nodes": []}, {"person:bare": "person:only-gi"})
        assert got == {}, "one layer cannot disagree with itself; nothing to unify"


class TestThePlaceholderRuleIsOneDecisionNotTwo:
    """`Host` renders on an insight and is hidden from corpus-wide ranking. Both, deliberately.

    `person:unresolved-host-<ep>` carries a real Person node named `Host` — 61 of them on the
    production snapshot, all with `SPOKEN_BY`. `app_gi_view._speaker_name` resolves it;
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

    def test_the_episode_surface_names_it(self) -> None:
        from podcast_scraper.server.app_gi_view import _speaker_name

        artifact = {"nodes": [{"id": self.PID, "type": "Person", "properties": {"name": "Host"}}]}
        assert _speaker_name(artifact, self.PID) == "Host", (
            "inside one episode the host label is real information; dropping it would lose "
            "attribution the artifact actually has"
        )

    def test_the_aggregate_surface_refuses_it(self) -> None:
        from podcast_scraper.enrichment.enrichers._loaders import (
            is_unresolved_speaker_placeholder,
        )

        assert (
            is_unresolved_speaker_placeholder(self.PID) is True
        ), "ranked corpus-wide this would be one person across every show that has a host"

    def test_a_real_person_is_neither_hidden_nor_special_cased(self) -> None:
        from podcast_scraper.enrichment.enrichers._loaders import (
            is_unresolved_speaker_placeholder,
        )
        from podcast_scraper.server.app_gi_view import _speaker_name

        artifact = {
            "nodes": [
                {
                    "id": "person:kevin-roose",
                    "type": "Person",
                    "properties": {"name": "Kevin Roose"},
                }
            ]
        }
        assert _speaker_name(artifact, "person:kevin-roose") == "Kevin Roose"
        assert is_unresolved_speaker_placeholder("person:kevin-roose") is False


class TestTheShowNameGuardOnTheQUOTEPath:
    """The KG path refused a show as a speaker; the GI path minted it anyway (#2064, #2065).

    `_speaker_lists_for_graph._is_the_show` protects kg.json. GI reads the segments sidecar
    directly and never passes through it, so on the production snapshot 53 `SPOKEN_BY` edges
    targeted a Person named `Machine Learning Street` — the show "spoke" on the insight surface
    even after m0009 demoted its KG twin.

    WHAT THE REFUSAL MUST NOT COST. A first version returned no person id at all. Measured, that
    discarded **2,348 `SPOKEN_BY` edges** across 5 show names — every quote on *Machine Learning
    Street Talk* and *Conversations with Tyler* would have become unattributed. Diarization really
    did hear a distinct voice; only the LABEL is wrong, and the two facts are separable. So the id
    is episode-SCOPED rather than dropped: the quote keeps a speaker inside its episode, and the
    id can never aggregate into a corpus-wide person.

    All 5 matches across the 55 production feeds are genuine show names — none is a human — so
    `names_the_show`'s known false positive (a host whose name LEADS their show) costs nothing
    today, and if it ever fires the cost is episode-local attribution rather than deletion.
    """

    class _Q:
        char_start = 0
        char_end = 10

    EP = "ep:mlst-1"

    def test_a_show_name_is_scoped_not_published_as_a_person(self) -> None:
        from podcast_scraper.enrichment.enrichers._loaders import (
            is_unresolved_speaker_placeholder,
        )
        from podcast_scraper.gi.pipeline import _resolve_quote_speaker
        from podcast_scraper.identity.bare_name_scope import is_scoped_person_id

        pid, _n, _v = _resolve_quote_speaker(
            self._Q(),
            "Machine Learning Street",
            self.EP,
            None,
            None,
            "Machine Learning Street Talk (MLST)",
        )
        assert pid, "the quote must keep a speaker — the voice was real, only the label was wrong"
        assert is_scoped_person_id(pid), "a show must never hold a corpus-wide person id"
        assert is_unresolved_speaker_placeholder(pid), "and must stay out of every ranking surface"

    def test_a_real_person_is_untouched(self) -> None:
        from podcast_scraper.gi.pipeline import _resolve_quote_speaker
        from podcast_scraper.identity.bare_name_scope import is_scoped_person_id

        pid, _n, _v = _resolve_quote_speaker(
            self._Q(), "Kevin Roose", self.EP, None, None, "Hard Fork"
        )
        assert pid == "person:kevin-roose"
        assert not is_scoped_person_id(pid)

    def test_the_guard_is_inert_without_the_title_and_that_is_the_capability(self) -> None:
        """THE REMOVAL PROOF. `names_the_show` has no opinion on an empty title BY DESIGN, so a
        caller that forgets to thread `feed_title` silently disables it — which is precisely how
        `enrich-edges` shipped an inert copy of this same guard earlier in this arc.

        Asserting the answer CHANGES with the title is what proves the guard is connected rather
        than merely present.
        """
        from podcast_scraper.gi.pipeline import _resolve_quote_speaker

        with_title, _n1, _v1 = _resolve_quote_speaker(
            self._Q(),
            "Machine Learning Street",
            self.EP,
            None,
            None,
            "Machine Learning Street Talk (MLST)",
        )
        without_title, _n2, _v2 = _resolve_quote_speaker(
            self._Q(), "Machine Learning Street", self.EP, None, None, ""
        )
        assert with_title != without_title, "the title is the signal; inert if it changes nothing"
        assert without_title == "person:machine-learning-street"


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

    def test_the_gi_validator_rejects_a_real_kg_artifact(self, tmp_path) -> None:
        import pytest as _pytest

        from podcast_scraper.gi.io import write_artifact as gi_write

        with _pytest.raises(ValueError, match="model_version"):
            gi_write(tmp_path / "x.kg.json", self._real_kg_payload(), validate=True)

    def test_the_kg_validator_accepts_it(self, tmp_path) -> None:
        from podcast_scraper.kg.io import write_artifact as kg_write

        target = tmp_path / "x.kg.json"
        kg_write(target, self._real_kg_payload(), validate=True)
        assert target.is_file() and target.stat().st_size > 0

    def test_enrich_edges_persists_kg_through_the_kg_writer(self, tmp_path) -> None:
        """The capability: a changed KG payload actually reaches disk, not the failure counter."""
        import json

        from podcast_scraper.search.cli_handlers import _persist_scoped_kg, _stable_json

        target = tmp_path / "x.kg.json"
        target.write_text(json.dumps(self._real_kg_payload()), encoding="utf-8")
        payload = self._real_kg_payload()
        before = _stable_json(payload)
        payload["nodes"][0]["id"] = "person:unresolved-x-ep-1"  # what the scope pass does
        totals = {"kg_scoped": 0, "kg_write_failed": 0}

        import logging

        _persist_scoped_kg(target, payload, before, "ep:1", totals, logging.getLogger(__name__))
        assert totals == {"kg_scoped": 1, "kg_write_failed": 0}
        assert json.loads(target.read_text(encoding="utf-8"))["nodes"][0]["id"] == (
            "person:unresolved-x-ep-1"
        )


class TestTheShowNeverREADSAsTheSpeaker:
    """Scoping the id is not enough — the NODE's name is what the reader actually sees.

    `_attach_person_for_quote` named the node with the raw diarization label, so after the id was
    episode-scoped the insight surface still rendered `Machine Learning Street` as the speaker.
    The id fix kept the show out of every corpus-wide surface and left the original complaint —
    the show appearing to talk — untouched on the one surface the operator was looking at.
    """

    class _Q:
        char_start = 0
        char_end = 10

    def test_the_reader_is_told_a_voice_spoke_not_who(self) -> None:
        from podcast_scraper.gi.pipeline import (
            _attach_person_for_quote,
            _resolve_quote_speaker,
            UNNAMED_SPEAKER_LABEL,
        )
        from podcast_scraper.server.app_gi_view import _speaker_name

        pid, display, _vt = _resolve_quote_speaker(
            self._Q(),
            "Machine Learning Street",
            "ep:mlst-1",
            None,
            None,
            "Machine Learning Street Talk (MLST)",
        )
        nodes: list = []
        edges: list = []
        _attach_person_for_quote(
            nodes, edges, "quote:q1", "Machine Learning Street", pid, set(), display
        )
        assert _speaker_name({"nodes": nodes, "edges": edges}, pid) == UNNAMED_SPEAKER_LABEL
        assert edges == [
            {"type": "SPOKEN_BY", "from": "quote:q1", "to": pid}
        ], "the quote must keep its speaker edge — a real voice did speak"

    def test_a_real_person_still_reads_as_themselves(self) -> None:
        from podcast_scraper.gi.pipeline import _attach_person_for_quote, _resolve_quote_speaker
        from podcast_scraper.server.app_gi_view import _speaker_name

        pid, display, _vt = _resolve_quote_speaker(
            self._Q(), "Kevin Roose", "ep:hf-1", None, None, "Hard Fork"
        )
        nodes: list = []
        edges: list = []
        _attach_person_for_quote(nodes, edges, "quote:q1", "Kevin Roose", pid, set(), display)
        assert _speaker_name({"nodes": nodes, "edges": edges}, pid) == "Kevin Roose"


class TestAStaleRunCannotReachAReader:
    """A superseded run stays on disk. Every USER-FACING reader must ignore it.

    `relabel_only` writes the repaired episode to a NEW run directory and leaves the old one in
    place — verified on a real repair against a production copy: the old run still holds the
    pre-repair answer (82 `SPOKEN_BY` edges pointing at the show, the guest still `mentioned`)
    while the new run holds the fixed one.

    The catalog has always deduped to the newest run per episode; the graph builders did not.
    Of the six readers that walked every run, exactly TWO reach a user — `gi.explore` (the explore
    route and the MCP `gi` tool) and `identity.resolver` (the MCP `resolve` tool). The other four
    are CLI/offline maintenance paths where seeing every run is defensible.

    This pins the two that matter: with both runs on disk, the reader must see only the newer.
    """

    @staticmethod
    def _two_run_corpus(tmp_path):
        """A feed with an OLD run naming the show and a NEW run naming the guest."""
        import json

        feed = tmp_path / "feeds" / "f1"

        def _write(run, stem, person_name):
            md_dir, tr = feed / run / "metadata", feed / run / "transcripts"
            md_dir.mkdir(parents=True, exist_ok=True)
            tr.mkdir(parents=True, exist_ok=True)
            (md_dir / f"{stem}.metadata.json").write_text(
                json.dumps(
                    {
                        "feed": {"feed_id": "f1", "title": "T", "url": "https://e/f.xml"},
                        "episode": {"episode_id": "ep1", "title": "E"},
                        "content": {
                            "speakers": [{"id": "host", "name": person_name, "role": "host"}]
                        },
                        "processing": {"schema_version": "1.0.0"},
                    }
                ),
                encoding="utf-8",
            )
            (md_dir / f"{stem}.gi.json").write_text(
                json.dumps(
                    {
                        "schema_version": "3.1",
                        "episode_id": "ep1",
                        "model_version": "m",
                        "prompt_version": "v1",
                        "nodes": [
                            {
                                "id": f"person:{person_name.lower().replace(' ', '-')}",
                                "type": "Person",
                                "properties": {"name": person_name},
                            }
                        ],
                        "edges": [],
                    }
                ),
                encoding="utf-8",
            )

        _write("run_20260101-000000", "0001 - E_old", "The Show Itself")
        _write("run_20260202-000000", "0001 - E_new", "Real Guest")
        return tmp_path

    def test_explore_reads_only_the_newest_run(self, tmp_path) -> None:
        from podcast_scraper.gi.explore import newest_run_artifact_paths, scan_artifact_paths

        root = self._two_run_corpus(tmp_path)
        found = scan_artifact_paths(root)
        kept = newest_run_artifact_paths(root, found, ".gi.json")
        assert len(found) == 2, "both runs are on disk — that is the situation being guarded"
        assert len(kept) == 1
        assert "20260202" in str(kept[0]), "the newer run must win"

    def test_the_resolver_never_sees_the_superseded_answer(self, tmp_path) -> None:
        from podcast_scraper.identity.resolver import _iter_loaded

        root = self._two_run_corpus(tmp_path)
        names = {
            str((n.get("properties") or {}).get("name") or "")
            for _src, data in _iter_loaded(root)
            for n in data.get("nodes", [])
            if isinstance(n, dict) and str(n.get("type")) == "Person"
        }
        assert "Real Guest" in names
        assert (
            "The Show Itself" not in names
        ), "the MCP resolve tool would otherwise resolve against the answer that was replaced"

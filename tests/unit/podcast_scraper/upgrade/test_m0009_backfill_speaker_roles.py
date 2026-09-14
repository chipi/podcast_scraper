"""m0009 — promote Person nodes to their diarization-roster role (#2062).

The corpus shipped with 89.5% of Person nodes tagged ``mentioned`` and 0.6% ``guest``, because the
graph was built from the pre-diarization hint rather than the roster (330-episode production sample,
2026-09-13). The roster's answer was never lost — it is on disk in each episode's
``content.speakers`` — so this is a pure re-read, no GPU and no LLM.

What these tests pin is mostly what the migration REFUSES to do. Promoting a role is easy; the ways
a backfill quietly corrupts a corpus are demotion, duplication, and deciding something it has no
evidence for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0009_backfill_speaker_roles import (
    BackfillSpeakerRolesMigration,
    demote_non_persons,
    promote_person_roles,
    roster_roles,
    voices_heard,
    voices_in_episode,
)

pytestmark = pytest.mark.unit

HOST = "Kevin Roose"
GUEST = "Brian Chesky"
MENTIONED = "Elon Musk"


def _metadata(speakers) -> Dict[str, Any]:
    return {"content": {"speakers": speakers}}


def _kg(persons) -> Dict[str, Any]:
    return {
        "schema_version": "2.1",
        "nodes": [
            {"id": pid, "type": "Person", "properties": {"name": name, "role": role}}
            for pid, name, role in persons
        ],
        "edges": [],
    }


class TestReadingTheRoster:
    def test_host_and_guest_are_keyed_by_person_id(self) -> None:
        roles = roster_roles(
            _metadata([{"name": HOST, "role": "host"}, {"name": GUEST, "role": "guest"}])
        )
        assert roles == {"person:kevin-roose": "host", "person:brian-chesky": "guest"}

    def test_a_role_this_pass_cannot_write_is_ignored(self) -> None:
        # "mentioned" on a roster entry carries no speaker information to promote WITH.
        assert roster_roles(_metadata([{"name": HOST, "role": "mentioned"}])) == {}

    def test_case_and_spacing_differences_do_not_create_two_keys(self) -> None:
        roles = roster_roles(
            _metadata([{"name": "kevin  roose", "role": "host"}, {"name": HOST, "role": "guest"}])
        )
        assert roles == {"person:kevin-roose": "host"}  # first voice wins

    def test_missing_speakers_yields_nothing(self) -> None:
        assert roster_roles({}) == {}
        assert roster_roles({"content": {}}) == {}


class TestPromotion:
    def test_a_mentioned_guest_becomes_a_guest(self) -> None:
        kg = _kg([("person:brian-chesky", GUEST, "mentioned")])
        promoted, _demoted, unmatched = promote_person_roles(kg, {"person:brian-chesky": "guest"})
        assert promoted == 1
        assert kg["nodes"][0]["properties"]["role"] == "guest"
        assert unmatched == []

    def test_a_person_with_no_role_at_all_is_promoted(self) -> None:
        kg: Dict[str, Any] = {
            "nodes": [{"id": "person:kevin-roose", "type": "Person", "properties": {}}]
        }
        promoted, _demoted, _unmatched = promote_person_roles(kg, {"person:kevin-roose": "host"})
        assert promoted == 1
        assert kg["nodes"][0]["properties"]["role"] == "host"

    def test_a_node_is_matched_by_name_when_its_id_predates_the_slug_rule(self) -> None:
        kg = _kg([("legacy-id-7", GUEST, "mentioned")])
        promoted, _demoted, _unmatched = promote_person_roles(kg, {"person:brian-chesky": "guest"})
        assert promoted == 1

    def test_a_person_the_roster_never_placed_is_left_alone(self) -> None:
        # The trap: promoting everyone would publish someone merely discussed as a speaker.
        kg = _kg([("person:elon-musk", MENTIONED, "mentioned")])
        promoted, _demoted, _unmatched = promote_person_roles(kg, {"person:brian-chesky": "guest"})
        assert promoted == 0
        assert kg["nodes"][0]["properties"]["role"] == "mentioned"


class TestWhatItRefusesToDo:
    """Demotion is evidence-driven: only a role the roster actively contradicts."""

    def test_a_speaker_the_roster_confirms_is_not_re_decided(self) -> None:
        # The roster names them, so they DID speak. Which of host/guest they are was settled by
        # something with more context than this pass has; it does not re-litigate that.
        kg = _kg([("person:brian-chesky", GUEST, "guest")])
        promoted, demoted, _unmatched = promote_person_roles(kg, {"person:brian-chesky": "host"})
        assert (promoted, demoted) == (0, 0)
        assert kg["nodes"][0]["properties"]["role"] == "guest"

    def test_an_existing_host_the_roster_confirms_is_not_rewritten(self) -> None:
        kg = _kg([("person:kevin-roose", HOST, "host")])
        promoted, demoted, _unmatched = promote_person_roles(kg, {"person:kevin-roose": "guest"})
        assert (promoted, demoted) == (0, 0)
        assert kg["nodes"][0]["properties"]["role"] == "host"

    def test_an_empty_roster_is_refused_rather_than_demoting_everyone(self) -> None:
        # The guard that stops this pass wiping an un-diarized episode's speakers.
        kg = _kg([("person:kevin-roose", HOST, "host")])
        with pytest.raises(ValueError):
            promote_person_roles(kg, {})

    def test_an_unmatched_roster_name_is_reported_not_inserted(self) -> None:
        # A roster voice with NO node in the graph at all — the transcript named someone the
        # extractor never recorded. Report them; never invent a node, because a name with no
        # extraction behind it is a dangling reference rather than evidence.
        #
        # This test used to use the "bernt børnich" / "bernt bornich" pair. Since #2062 those are
        # recognised as one human and PROMOTED, which is the point of the variant matching — so a
        # genuinely absent person is needed to exercise the reporting path.
        kg = _kg([("person:eric-olander", "Eric Olander", "host")])
        before = len(kg["nodes"])
        promoted, _demoted, unmatched = promote_person_roles(
            kg, {"person:eric-olander": "host", "person:jorge-heine": "guest"}
        )
        assert promoted == 0  # Eric already carries the role the roster gives him
        assert len(kg["nodes"]) == before
        assert unmatched == ["person:jorge-heine"]

    def test_non_person_nodes_are_untouched(self) -> None:
        kg: Dict[str, Any] = {
            "nodes": [
                {"id": "org:acme", "type": "Organization", "properties": {"name": "Acme"}},
                {"id": "object:x", "type": "Object", "properties": {"name": "X"}},
            ]
        }
        promoted, _demoted, _unmatched = promote_person_roles(
            kg, {"org:acme": "host", "object:x": "guest"}
        )
        assert promoted == 0
        assert all("role" not in (n.get("properties") or {}) for n in kg["nodes"])


class TestOnDisk:
    def _corpus(self, tmp_path: Path) -> Path:
        d = tmp_path / "feeds" / "f1" / "run1" / "metadata"
        d.mkdir(parents=True)
        stem = "0001 - Episode"
        (d / f"{stem}.metadata.json").write_text(
            json.dumps(
                _metadata([{"name": HOST, "role": "host"}, {"name": GUEST, "role": "guest"}])
            ),
            encoding="utf-8",
        )
        (d / f"{stem}.kg.json").write_text(
            json.dumps(
                _kg(
                    [
                        ("person:kevin-roose", HOST, "mentioned"),
                        ("person:brian-chesky", GUEST, "mentioned"),
                        ("person:elon-musk", MENTIONED, "mentioned"),
                    ]
                )
            ),
            encoding="utf-8",
        )
        return d / f"{stem}.kg.json"

    def _roles(self, path: Path) -> Dict[str, str]:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return {(n["properties"]).get("name"): (n["properties"]).get("role") for n in doc["nodes"]}

    def test_the_artifact_on_disk_is_corrected(self, tmp_path: Path) -> None:
        kg_path = self._corpus(tmp_path)
        res = BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=tmp_path))
        assert res.details["persons_promoted"] == 2
        assert self._roles(kg_path) == {HOST: "host", GUEST: "guest", MENTIONED: "mentioned"}

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        kg_path = self._corpus(tmp_path)
        before = kg_path.read_text(encoding="utf-8")
        res = BackfillSpeakerRolesMigration().apply(
            MigrationContext(corpus_root=tmp_path, dry_run=True)
        )
        assert res.details["persons_promoted"] == 2
        assert kg_path.read_text(encoding="utf-8") == before

    def test_running_twice_changes_nothing_the_second_time(self, tmp_path: Path) -> None:
        kg_path = self._corpus(tmp_path)
        BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=tmp_path))
        after_first = kg_path.read_text(encoding="utf-8")
        res = BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=tmp_path))
        assert res.details["persons_promoted"] == 0
        assert kg_path.read_text(encoding="utf-8") == after_first

    def test_an_episode_with_no_metadata_sibling_is_skipped_not_failed(
        self, tmp_path: Path
    ) -> None:
        kg_path = self._corpus(tmp_path)
        kg_path.with_name(kg_path.name.replace(".kg.json", ".metadata.json")).unlink()
        res = BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=tmp_path))
        assert res.applied is True
        assert res.details["no_roster"] == 1

    def test_an_unparsable_artifact_is_recorded_not_raised(self, tmp_path: Path) -> None:
        kg_path = self._corpus(tmp_path)
        kg_path.write_text("{not json", encoding="utf-8")
        res = BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=tmp_path))
        assert res.applied is True
        assert res.details["unparsable"]


class TestDemotingSomeoneWhoNeverSpoke:
    """On a complete roster, 19.4% of prod host nodes and 14.3% of guest nodes did not speak.

    They came from the same pre-diarization hint as the missing guests, so promoting the real
    speakers without demoting these would leave the episode claiming two hosts, one of whom was
    never in the room.
    """

    def test_a_host_the_roster_never_heard_becomes_mentioned(self) -> None:
        kg = _kg(
            [
                ("person:sarah-guo", "Sarah Guo", "host"),
                ("person:elad-gil", "Elad Gil", "mentioned"),
            ]
        )
        promoted, demoted, _unmatched = promote_person_roles(
            kg, {"person:elad-gil": "host"}, voices_heard=1
        )
        assert (promoted, demoted) == (1, 1)
        assert kg["nodes"][0]["properties"]["role"] == "mentioned"

    def test_a_guest_the_roster_never_heard_becomes_mentioned(self) -> None:
        kg = _kg(
            [
                ("person:someone-on-tape", "Someone On Tape", "guest"),
                ("person:elad-gil", "Elad Gil", "mentioned"),
            ]
        )
        _promoted, demoted, _unmatched = promote_person_roles(
            kg, {"person:elad-gil": "host"}, voices_heard=1
        )
        assert demoted == 1
        assert kg["nodes"][0]["properties"]["role"] == "mentioned"

    def test_the_show_itself_stops_being_a_host(self) -> None:
        # "The China-Global South Project" shipped as a host Person on real episodes.
        kg = _kg(
            [
                ("person:the-china-global-south-project", "The China-Global South Project", "host"),
                ("person:eric-olander", "Eric Olander", "mentioned"),
            ]
        )
        _promoted, demoted, _unmatched = promote_person_roles(
            kg, {"person:eric-olander": "host"}, voices_heard=1
        )
        assert demoted == 1

    def test_someone_already_mentioned_is_not_touched(self) -> None:
        kg = _kg(
            [
                ("person:elon-musk", MENTIONED, "mentioned"),
                ("person:elad-gil", "Elad Gil", "host"),
            ]
        )
        promoted, demoted, _unmatched = promote_person_roles(
            kg, {"person:elad-gil": "host"}, voices_heard=1
        )
        assert (promoted, demoted) == (0, 0)
        assert kg["nodes"][0]["properties"]["role"] == "mentioned"

    def test_promote_and_demote_happen_in_the_same_pass(self) -> None:
        kg = _kg(
            [
                ("person:kevin-roose", HOST, "mentioned"),
                ("person:sarah-guo", "Sarah Guo", "host"),
            ]
        )
        promoted, demoted, _unmatched = promote_person_roles(
            kg, {"person:kevin-roose": "host"}, voices_heard=1
        )
        assert (promoted, demoted) == (1, 1)
        roles = {n["properties"]["name"]: n["properties"]["role"] for n in kg["nodes"]}
        assert roles == {HOST: "host", "Sarah Guo": "mentioned"}


class TestSpellingVariantsAreNotTreatedAsStrangers:
    """m0009 matched by EXACT slug, so an ASR variant of a real speaker was demoted (#2062).

    Found by an advisor review and then measured on the 287-artifact production staging copy:
    of 70 demotions, **5 stripped a real speaker** whose only crime was being misheard —

        host 'Bernard Leong'     roster ['Bernard Leung', 'Steve Clayton']
        host 'Alexandra Karppi'  roster ['Adam Reichardt', 'Alexander Carpi']

    The migration's own docstring already listed "an ASR variant of a real speaker" as one of the
    things that makes a node phantom, which was backwards: a variant means we misheard the NAME,
    not that the human was absent. Demoting them replaces a wrong spelling with a wrong ROLE, and
    the host of the episode stops being its host.

    Matching now uses `kg.speaker_coherence.same_person` — the same predicate the coherence guard
    uses, so "is this the same person" has ONE answer in the codebase — in both directions: a
    variant-named node is PROMOTED rather than reported unmatched, and is never demoted.
    """

    def test_a_misheard_host_is_not_demoted(self) -> None:
        kg = _kg([("person:bernard-leong", "Bernard Leong", "host")])
        promoted, demoted, _unmatched = promote_person_roles(
            kg, {"person:bernard-leung": "host", "person:steve-clayton": "guest"}
        )
        assert demoted == 0, "the episode's real host was stripped of the role"
        assert kg["nodes"][0]["properties"]["role"] == "host"

    def test_a_misheard_guest_is_promoted_not_reported_missing(self) -> None:
        # The other half of the 14.6%: these were reported as "needs a re-enrich" when the person
        # was sitting right there under a slightly different spelling.
        kg = _kg([("person:alexander-carpi", "Alexander Carpi", "mentioned")])
        promoted, demoted, unmatched = promote_person_roles(
            kg, {"person:alexandra-karppi": "guest"}
        )
        assert promoted == 1
        assert kg["nodes"][0]["properties"]["role"] == "guest"
        assert unmatched == []

    def test_a_genuine_stranger_is_still_demoted(self) -> None:
        # The rule must not become "never demote": "Sarah Guo" on an Elad Gil episode is not a
        # spelling variant of anybody, and that is 65 of the 70 real cases.
        kg = _kg(
            [
                ("person:sarah-guo", "Sarah Guo", "host"),
                ("person:elad-gil", "Elad Gil", "mentioned"),
            ]
        )
        _promoted, demoted, _unmatched = promote_person_roles(
            kg, {"person:elad-gil": "host"}, voices_heard=1
        )
        assert demoted == 1
        assert kg["nodes"][0]["properties"]["role"] == "mentioned"

    def test_the_show_name_is_still_demoted(self) -> None:
        kg = _kg(
            [
                ("person:the-china-global-south-project", "The China-Global South Project", "host"),
                ("person:eric-olander", "Eric Olander", "mentioned"),
            ]
        )
        _promoted, demoted, _unmatched = promote_person_roles(
            kg, {"person:eric-olander": "host"}, voices_heard=1
        )
        assert demoted == 1

    def test_a_diacritic_variant_is_matched(self) -> None:
        kg = _kg([("person:bernt-bornich", "Bernt Bornich", "mentioned")])
        promoted, demoted, _unmatched = promote_person_roles(kg, {"person:bernt-brnich": "guest"})
        assert (promoted, demoted) == (1, 0)

    def test_an_honorific_variant_is_matched(self) -> None:
        kg = _kg([("person:alexander-douglas", "Alexander Douglas", "mentioned")])
        promoted, demoted, _unmatched = promote_person_roles(
            kg, {"person:dr-alexander-douglas": "guest"}
        )
        assert (promoted, demoted) == (1, 0)


class TestDemotionRequiresACompleteRoster:
    """Absence from a PARTIAL roster is silence, not denial (#2062 / advisor H4).

    `content.speakers` lists only the voices diarization NAMED. Measured on production: 62.4% of
    episodes are partial — diarization heard more voices than it named — and on those the roster
    says nothing at all about the voices it left anonymous.

    m0009's demotion reads "not on the roster" as "never spoke". On a complete roster that is
    sound. On a partial one it is a guess, and the guess is sometimes wrong in the most damaging
    direction:

        host 'Ryan Knutson'   voices=8 named=2   <- the show's actual host, demoted

    Of 65 demotions on the 287-artifact sample, 57 sat on partial or unknown rosters. Most of those
    are genuinely wrong nodes — an org as a host ("Mercatus Center at George Mason University",
    "The China-Global South Project", "China Plus"), a mangled multi-name string — but they cannot
    be told apart from a real speaker the roster simply never named, and this migration is
    irreversible on production.

    So demotion is gated on a COMPLETE roster. Promotion is not: promoting a node the roster names
    adds information and is safe whatever the roster omits. The org-as-host cases are a separate
    defect (#2064, fixed at source in the roster) and a re-run picks them up afterwards; a demoted
    real host needs a re-enrichment to recover, which is the worse failure to choose.
    """

    def _kg_with_a_stranger(self):
        return _kg(
            [
                ("person:elad-gil", "Elad Gil", "mentioned"),
                ("person:sarah-guo", "Sarah Guo", "host"),
            ]
        )

    def test_a_complete_roster_still_demotes(self) -> None:
        meta = {
            "content": {
                "speakers": [{"name": "Elad Gil", "role": "host"}],
                "diarization_num_speakers": 1,
            }
        }
        kg = self._kg_with_a_stranger()
        promoted, demoted, _u = promote_person_roles(
            kg, roster_roles(meta), voices_heard=voices_heard(meta)
        )
        assert (promoted, demoted) == (1, 1)

    def test_a_partial_roster_promotes_but_does_not_demote(self) -> None:
        # Diarization heard 8 voices and named 2: it is silent about the other 6.
        kg = self._kg_with_a_stranger()
        promoted, demoted, _u = promote_person_roles(
            kg, {"person:elad-gil": "host"}, voices_heard=8
        )
        assert promoted == 1, "promotion is always safe — it adds information"
        assert demoted == 0, "a node the roster is SILENT about must not be demoted"

    def test_the_real_host_keeps_the_role_on_a_partial_roster(self) -> None:
        kg = _kg([("person:ryan-knutson", "Ryan Knutson", "host")])
        _p, demoted, _u = promote_person_roles(
            kg, {"person:heather-haddon": "guest"}, voices_heard=8
        )
        assert demoted == 0
        assert kg["nodes"][0]["properties"]["role"] == "host"

    def test_an_unknown_voice_count_defaults_to_no_demotion(self) -> None:
        # The safe default: a caller that does not know must not get demotion by accident. This
        # one deliberately omits the keyword — that is the assertion.
        kg = self._kg_with_a_stranger()
        _p, demoted, _u = promote_person_roles(kg, {"person:elad-gil": "host"})
        assert demoted == 0


class TestReadingTheVoiceCountFromTheArtifact:
    """`voices_heard` is only the DENOMINATOR — how many humans spoke.

    Whether the roster accounts for them is decided against the episode graph, in
    `promote_person_roles`, because a roster can name a voice with something no one can place.
    """

    def test_the_diarization_count_is_returned(self) -> None:
        meta = {
            "content": {"speakers": [{"name": "A", "role": "host"}], "diarization_num_speakers": 3}
        }
        assert voices_heard(meta) == 3

    def test_a_missing_voice_count_is_unknowable(self) -> None:
        # 6.6% of production episodes carry no `diarization_num_speakers`, and when diarization
        # names nobody the pipeline substitutes the pre-diarization HINT into `content.speakers`
        # wholesale — indistinguishable on disk from a real roster. Unknowable means unsafe.
        meta = {"content": {"speakers": [{"name": "A", "role": "host"}]}}
        assert voices_heard(meta) is None

    def test_a_nonsense_voice_count_is_unknowable(self) -> None:
        for bad in (0, -1, "two", None, True):
            assert voices_heard({"content": {"diarization_num_speakers": bad}}) is None

    def test_no_content_at_all_is_unknowable(self) -> None:
        assert voices_heard({}) is None


class TestTheRosterItselfCanBeWrong:
    """m0009 reads `content.speakers` as ground truth — and on existing artifacts it is not (#2064).

    The roster on disk was written by the pipeline BEFORE the show-name and role-word fixes, so it
    still says `host = "Africa Tech Summit"` or `host = "Host"`. m0009 promotes from that roster, so
    without a guard it does not merely fail to repair those episodes — it PROMOTES the bad name into
    a `host` role, entrenching the defect in the one pass that is irreversible on production.

    Measured on the 330-episode sample: 25 roster entries would have been treated as real speakers —
    13 show names (`Africa Tech Summit`, `Machine Learning Street`, `Conversations with Tyler`) and
    12 role-word placeholders (`Host`).

    The two predicates are the same ones the pipeline uses, so the migration and the pipeline cannot
    disagree about what counts as a person.
    """

    def test_a_show_name_in_the_roster_is_not_a_speaker(self) -> None:
        meta = {
            "feed": {"title": "Africa Tech Summit Podcast"},
            "content": {
                "speakers": [
                    {"name": "Africa Tech Summit", "role": "host"},
                    {"name": "Mukami Wairaina", "role": "guest"},
                ]
            },
        }
        assert roster_roles(meta) == {"person:mukami-wairaina": "guest"}

    def test_a_role_word_in_the_roster_is_not_a_speaker(self) -> None:
        meta = {
            "feed": {"title": "The Flip"},
            "content": {
                "speakers": [
                    {"name": "Host", "role": "host"},
                    {"name": "Wale Afolabi", "role": "guest"},
                ]
            },
        }
        assert roster_roles(meta) == {"person:wale-afolabi": "guest"}

    def test_a_real_host_whose_name_is_in_the_title_survives(self) -> None:
        meta = {
            "feed": {"title": "Invest Like the Best with Patrick O'Shaughnessy"},
            "content": {"speakers": [{"name": "Patrick O'Shaughnessy", "role": "host"}]},
        }
        assert roster_roles(meta) == {"person:patrick-oshaughnessy": "host"}

    def test_no_feed_title_still_rejects_role_words(self) -> None:
        # The show-name check needs a title; the placeholder check does not.
        meta = {
            "content": {
                "speakers": [
                    {"name": "Host", "role": "host"},
                    {"name": "Ada Lovelace", "role": "guest"},
                ]
            }
        }
        assert roster_roles(meta) == {"person:ada-lovelace": "guest"}

    def test_an_episode_whose_roster_is_entirely_bad_yields_nothing(self) -> None:
        # And an empty roster means m0009 skips the episode rather than demoting everyone.
        meta = {
            "feed": {"title": "Conversations with Tyler"},
            "content": {"speakers": [{"name": "Conversations with Tyler", "role": "host"}]},
        }
        assert roster_roles(meta) == {}


class TestARosterEntryNobodyCanFindIsNotEvidence:
    """A roster that NAMES every voice can still ACCOUNT FOR none of them (#2064 follow-up).

    `roster_is_complete` compared the voice count against the RAW length of `content.speakers`.
    That made "diarization named every voice it heard" and "we know who those voices were" the
    same question. They are not, and production has both ways of coming apart:

      * an entry the guards throw away still padded the count — a roster of
        ``['Host', 'Annie Jacobsen']`` on 2 voices counted as COMPLETE while `roster_roles`
        returned one usable name, so the seat `Host` claims to fill was in fact empty and the
        real occupant was demoted out of it;
      * an entry that is not a person at all is caught by no name-shape predicate. On
        *The Pragmatic Engineer*, "How AWS S3 is built", diarization heard 2 voices and named
        ``['Gergely Orosz', 'Developer Survey']``. The episode description reads "In this
        episode, I sit down with Mai-Lan Tomsen Bukovec, VP of Data and Analytics at AWS"; the
        graph already carried her ``GUESTS_ON``. She was demoted to `mentioned` on the strength
        of a roster whose second seat is a survey.

    The fix is not another name predicate — "Developer Survey" is not a shape you can screen for.
    It is to require that the roster ACCOUNT FOR the voices before its silence may deny anyone: a
    roster entry that resolves to no Person node in this episode's graph is an unresolved name,
    not a witness. The migration already refuses to insert a node for such an entry; it must
    equally refuse to count it.

    Measured over the 287 production artifacts: 13 demotions become 11, and the two that go are
    the only wrong one (Mai-Lan Tomsen Bukovec) plus one org the roster could not speak to.
    """

    def test_the_guest_is_not_unseated_by_a_roster_entry_that_is_a_survey(self) -> None:
        # The real artifact: substack:post:185094534.
        meta = {
            "feed": {"title": "The Pragmatic Engineer"},
            "content": {
                "speakers": [
                    {"name": "Gergely Orosz", "role": "host"},
                    {"name": "Developer Survey", "role": "guest"},
                ],
                "diarization_num_speakers": 2,
            },
        }
        kg = _kg(
            [
                ("person:gergely-orosz", "Gergely Orosz", "mentioned"),
                ("person:mai-lan-tomsen-bukovec", "Mai-Lan Tomsen Bukovec", "guest"),
            ]
        )
        promoted, demoted, unmatched = promote_person_roles(
            kg, roster_roles(meta), voices_heard=voices_heard(meta)
        )
        assert promoted == 1, "the host the roster DOES account for is still promoted"
        assert demoted == 0, "a roster seat filled by a survey cannot unseat the real guest"
        assert "person:developer-survey" in unmatched
        roles = {n["id"]: n["properties"]["role"] for n in kg["nodes"]}
        assert roles["person:mai-lan-tomsen-bukovec"] == "guest"

    def test_a_discarded_entry_no_longer_pads_the_voice_count(self) -> None:
        # 2 voices, roster ['Host', 'Annie Jacobsen'] -> one usable name, so one voice is
        # unaccounted for and the roster may not deny anyone.
        meta = {
            "feed": {"title": "Conversations with Tyler"},
            "content": {
                "speakers": [
                    {"name": "Host", "role": "host"},
                    {"name": "Annie Jacobsen", "role": "guest"},
                ],
                "diarization_num_speakers": 2,
            },
        }
        kg = _kg(
            [
                ("person:annie-jacobsen", "Annie Jacobsen", "mentioned"),
                ("person:tyler-cowen", "Tyler Cowen", "host"),
            ]
        )
        promoted, demoted, _u = promote_person_roles(
            kg, roster_roles(meta), voices_heard=voices_heard(meta)
        )
        assert promoted == 1
        assert demoted == 0, "the discarded 'Host' entry must not count as a voice accounted for"

    def test_a_roster_that_accounts_for_every_voice_still_demotes(self) -> None:
        # No Priors: the roster names Elad Gil and Glenn Fogel, both in the graph, 2 voices heard.
        # Sarah Guo is the co-host who sat this one out — this demotion is the one we want.
        meta = {
            "feed": {"title": "No Priors"},
            "content": {
                "speakers": [
                    {"name": "Elad Gil", "role": "host"},
                    {"name": "Glenn Fogel", "role": "guest"},
                ],
                "diarization_num_speakers": 2,
            },
        }
        kg = _kg(
            [
                ("person:elad-gil", "Elad Gil", "mentioned"),
                ("person:glenn-fogel", "Glenn Fogel", "mentioned"),
                ("person:sarah-guo", "Sarah Guo", "host"),
            ]
        )
        promoted, demoted, _u = promote_person_roles(
            kg, roster_roles(meta), voices_heard=voices_heard(meta)
        )
        assert (promoted, demoted) == (2, 1)
        roles = {n["id"]: n["properties"]["role"] for n in kg["nodes"]}
        assert roles["person:sarah-guo"] == "mentioned"

    def test_an_unknown_voice_count_still_denies_demotion(self) -> None:
        meta = {"content": {"speakers": [{"name": "Elad Gil", "role": "host"}]}}
        kg = _kg([("person:sarah-guo", "Sarah Guo", "host")])
        _p, demoted, _u = promote_person_roles(
            kg, roster_roles(meta), voices_heard=voices_heard(meta)
        )
        assert demoted == 0


class TestANonPersonCannotHoldAMicrophone:
    """Demoting the show's own name needs no roster at all (#2064).

    Seven of the thirteen production demotions are a show name or a role word sitting in the
    `host` seat — `Machine Learning Street`, `Africa Tech Summit`, `Conversations with Tyler`.
    Under the old rule they were demoted as a side effect of the roster being counted as
    complete, which is the rule this change tightens. They must not fall out with it: that a
    show is not a person is a coherence fact about the NODE, independent of who spoke.
    """

    def test_the_show_is_demoted_even_when_the_roster_accounts_for_nothing(self) -> None:
        meta = {
            "feed": {"title": "Machine Learning Street Talk"},
            "content": {
                "speakers": [
                    {"name": "Machine Learning Street", "role": "host"},
                    {"name": "Jeremy Berman", "role": "guest"},
                ],
                "diarization_num_speakers": 2,
            },
        }
        kg = _kg(
            [
                ("person:machine-learning-street", "Machine Learning Street", "host"),
                ("person:jeremy-berman", "Jeremy Berman", "mentioned"),
            ]
        )
        _p, demoted, _u = promote_person_roles(
            kg,
            roster_roles(meta),
            voices_heard=voices_heard(meta),
            feed_title="Machine Learning Street Talk",
        )
        assert demoted == 1
        roles = {n["id"]: n["properties"]["role"] for n in kg["nodes"]}
        assert roles["person:machine-learning-street"] == "mentioned"

    def test_a_role_word_node_is_demoted_without_any_roster_evidence(self) -> None:
        meta = {
            "feed": {"title": "Some Show"},
            "content": {
                "speakers": [{"name": "Annie Jacobsen", "role": "guest"}],
                "diarization_num_speakers": 4,
            },
        }
        kg = _kg(
            [
                ("person:host", "Host", "host"),
                ("person:annie-jacobsen", "Annie Jacobsen", "mentioned"),
            ]
        )
        _p, demoted, _u = promote_person_roles(
            kg, roster_roles(meta), voices_heard=voices_heard(meta), feed_title="Some Show"
        )
        assert demoted == 1, "a role word never held a microphone, roster or no roster"

    def test_a_real_person_is_not_swept_up_by_the_rule(self) -> None:
        meta = {
            "feed": {"title": "Some Show"},
            "content": {
                "speakers": [{"name": "Annie Jacobsen", "role": "guest"}],
                "diarization_num_speakers": 4,
            },
        }
        kg = _kg([("person:tyler-cowen", "Tyler Cowen", "host")])
        _p, demoted, _u = promote_person_roles(
            kg, roster_roles(meta), voices_heard=voices_heard(meta), feed_title="Some Show"
        )
        assert demoted == 0


class TestTheShowNamePredicateIsImperfectAndSaysSo:
    """`names_the_show` has a known false positive; it is REPORTED, not silently guarded.

    The predicate matches a multi-token PREFIX of the feed title, so it also classifies a host
    whose name LEADS their own show as the show itself::

        names_the_show('Lex Fridman', 'Lex Fridman Podcast')            -> True
        names_the_show('Dan Carlin',  "Dan Carlin's Hardcore History")  -> True

    The title cannot settle it: 'Lex Fridman Podcast' and 'Latent Space: The AI Engineer Podcast'
    are structurally identical and 'Latent.Space' really IS the show.

    "A SHOW DOES NOT SPEAK" LOOKS LIKE THE MISSING EVIDENCE AND IS NOT — this was tried and
    measured. `SPOKEN_BY` inherits the roster's mistakes: the roster is what put the show in the
    host seat (#2064), so the show ends up with quotes attributed to it. 'Machine Learning Street'
    has a voice on 4 of its 6 episodes. Sparing every node with a voice rescued 4 real show names
    this pass exists to demote, in order to protect a host shape that occurs nowhere in the corpus
    — all 8 title/name hits across the 55 production feeds are genuine show names.

    So the demotion stands and the ambiguous ones are listed in `suspect_demotions` for a human to
    read before this runs on a corpus with an eponymous show.
    """

    @staticmethod
    def _gi(spoken_name: str, node_id: str = "person:spk"):
        return {
            "nodes": [{"id": node_id, "type": "Person", "properties": {"name": spoken_name}}],
            "edges": [{"type": "SPOKEN_BY", "from": "quote:q1", "to": node_id}],
        }

    @staticmethod
    def _kg(name: str, role: str, node_id: str = "person:spk"):
        return {
            "schema_version": "2.1",
            "nodes": [
                {"id": node_id, "type": "Person", "properties": {"name": name, "role": role}}
            ],
            "edges": [{"type": "HOSTS", "from": node_id, "to": "podcast:p1"}],
        }

    def test_voices_are_read_from_the_gi_sibling_not_the_kg(self) -> None:
        # SPOKEN_BY is absent from all 287 production kg.json files and present in 259 gi.json.
        # A guard that read the KG payload's edges was a no-op on every real artifact.
        kg = self._kg("Lex Fridman", "host")
        assert voices_in_episode({}, kg) == set()
        assert voices_in_episode(self._gi("Lex Fridman"), kg) == {"person:spk"}

    def test_a_voice_is_matched_across_the_two_ID_FAMILIES(self) -> None:
        # #2056's shape: the GI id comes from the roster, the KG id from the extractor.
        kg = self._kg("Aaron Levie", "host", node_id="person:aaron-levie")
        gi = self._gi("Aaron Levy", node_id="person:aaron-levy")
        assert voices_in_episode(gi, kg) == {"person:aaron-levie"}

    def test_a_show_with_a_voice_is_still_demoted_but_flagged(self) -> None:
        kg = self._kg("Machine Learning Street", "host")
        voices = voices_in_episode(self._gi("Machine Learning Street"), kg)
        suspects: list[str] = []
        assert demote_non_persons(kg, "Machine Learning Street Talk", voices, suspects) == 1
        assert kg["nodes"][0]["properties"]["role"] == "mentioned"
        assert suspects and "Machine Learning Street" in suspects[0]

    def test_a_show_with_no_voice_is_demoted_and_not_flagged(self) -> None:
        kg = self._kg("Africa Tech Summit", "host")
        suspects: list[str] = []
        assert demote_non_persons(kg, "Africa Tech Summit Podcast", set(), suspects) == 1
        assert suspects == [], "no ambiguity here — nothing for a human to read"

    def test_a_role_word_is_demoted_and_never_flagged(self) -> None:
        # "Host" is not a human even with a voice; there is no judgement call to escalate.
        kg = self._kg("Host", "host", node_id="person:host")
        suspects: list[str] = []
        assert demote_non_persons(kg, "Some Show", {"person:host"}, suspects) == 1
        assert suspects == []

    def test_apply_reports_the_suspect_class(self, tmp_path: Path) -> None:
        # Through the real wiring: apply() must find the .gi.json sibling itself. An earlier
        # version passed every function-level test while being a no-op in the migration.
        d = tmp_path / "metadata"
        d.mkdir(parents=True)
        (d / "e1.metadata.json").write_text(
            json.dumps(
                {
                    "feed": {"title": "Machine Learning Street Talk"},
                    "content": {
                        "speakers": [{"name": "Jeremy Berman", "role": "guest"}],
                        "diarization_num_speakers": 2,
                    },
                }
            )
        )
        (d / "e1.kg.json").write_text(json.dumps(self._kg("Machine Learning Street", "host")))
        (d / "e1.gi.json").write_text(json.dumps(self._gi("Machine Learning Street")))
        res = BackfillSpeakerRolesMigration().apply(
            MigrationContext(corpus_root=tmp_path, dry_run=True)
        )
        assert res.details["suspect_demotions"], "the ambiguous demotion must reach the operator"
        assert "HAND-READ REQUIRED" in res.message


class TestANameThatCannotBeSluggedDoesNotKillTheRun:
    """A non-Latin name must not abort the migration mid-corpus (advisor S2).

    `identity.slugify.person_id` raises when NFKD -> ASCII leaves nothing::

        person_id('张川红')          -> ValueError: Slug is empty after normalisation
        person_id('Владимир Путин')  -> ValueError

    `apply()` catches only OSError/JSONDecodeError, so the runner stops at the first such name and
    records nothing for 0009. Files already rewritten stay rewritten (idempotent, so harmless), but
    every re-run dies on the same file — there is no resume point.

    Zero occurrences in the 287-artifact sample, which is 15% of production; Round Table China,
    China Plus, ChinaTalk and The Naked Pravda are in the other 85%.
    """

    def test_a_cjk_roster_name_is_skipped_not_raised(self) -> None:
        meta = {
            "feed": {"title": "Round Table China"},
            "content": {
                "speakers": [
                    {"name": "张川红", "role": "guest"},
                    {"name": "Heyang Zhang", "role": "host"},
                ],
                "diarization_num_speakers": 2,
            },
        }
        roles = roster_roles(meta)  # must not raise
        assert "person:heyang-zhang" in roles
        assert all("张" not in k for k in roles), "the unsluggable name is dropped, not crashed on"

    def test_a_cyrillic_node_name_is_skipped_not_raised(self) -> None:
        kg = _kg([("person:x", "Владимир Путин", "host")])
        # must not raise
        promoted, demoted, _u = promote_person_roles(
            kg, {"person:heyang-zhang": "host"}, voices_heard=1
        )
        assert (promoted, demoted) == (0, 0)


class TestAnAmbiguousMatchIsNotAMatch:
    """A node that plausibly matches TWO roster entries matches neither (advisor S4).

    `_fuzzy_roster_hit` iterated the roster and returned the FIRST `same_person` hit. `same_person`
    treats a token subset as the same human — ``same_person('John', 'John Smith')`` is True — so a
    bare-mononym node was promoted to whichever John the dict happened to yield first, landing a
    role on the wrong person while the real full-name node kept its own.

    That is the #2056 duplicate symptom manufactured BY the repair, which is the worst class of
    bug this migration can have: it is irreversible, and the coherence checks cannot see it because
    they share the migration's own predicate.

    The safe answer to "which of these two?" is neither. An ambiguous node is left exactly as it
    was, and — importantly — its roster keys are NOT marked as matched, so route (b) does not then
    read the absence as "this person never spoke" and demote someone.
    """

    def test_a_mononym_matching_two_roster_entries_is_left_alone(self) -> None:
        kg = _kg([("person:john", "John", "mentioned")])
        roles = {"person:john-smith": "host", "person:john-doe": "guest"}
        promoted, demoted, _u = promote_person_roles(kg, roles, voices_heard=2)
        assert (promoted, demoted) == (0, 0)
        assert kg["nodes"][0]["properties"]["role"] == "mentioned"

    def test_an_unambiguous_mononym_still_resolves(self) -> None:
        # One candidate is not a guess. `Twiggy` against a roster holding only `Twiggy Lawson`
        # is the case `same_person`'s subset rule exists for.
        kg = _kg([("person:twiggy", "Twiggy", "mentioned")])
        promoted, _d, _u = promote_person_roles(
            kg, {"person:twiggy-lawson": "guest"}, voices_heard=1
        )
        assert promoted == 1
        assert kg["nodes"][0]["properties"]["role"] == "guest"

    def test_an_ambiguous_node_does_not_make_the_roster_look_unspoken(self) -> None:
        # The subtle half: if the ambiguous node consumed both roster keys, or neither key were
        # marked matched, route (b) could conclude the roster is unaccounted for and demote a real
        # speaker elsewhere in the episode.
        kg = _kg(
            [
                ("person:john", "John", "mentioned"),
                ("person:john-smith", "John Smith", "mentioned"),
                ("person:john-doe", "John Doe", "mentioned"),
            ]
        )
        roles = {"person:john-smith": "host", "person:john-doe": "guest"}
        promoted, demoted, unmatched = promote_person_roles(kg, roles, voices_heard=2)
        assert demoted == 0
        assert promoted == 2, "both exact-id matches still promote"
        assert unmatched == [], "the roster is fully accounted for by the two real nodes"

    def test_ambiguity_is_reported_not_silent(self) -> None:
        kg = _kg([("person:john", "John", "mentioned")])
        roles = {"person:john-smith": "host", "person:john-doe": "guest"}
        ambiguous: list[str] = []
        promote_person_roles(kg, roles, voices_heard=2, ambiguous=ambiguous)
        assert ambiguous and "John" in ambiguous[0]

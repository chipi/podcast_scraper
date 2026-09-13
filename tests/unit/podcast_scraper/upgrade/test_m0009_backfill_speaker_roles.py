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
    promote_person_roles,
    roster_roles,
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
    """39.5% of prod host nodes, and 40% of guest nodes, never spoke in their episode.

    They came from the same pre-diarization hint as the missing guests, so promoting the real
    speakers without demoting these would leave the episode claiming two hosts, one of whom was
    never in the room.
    """

    def test_a_host_the_roster_never_heard_becomes_mentioned(self) -> None:
        kg = _kg([("person:sarah-guo", "Sarah Guo", "host")])
        promoted, demoted, _unmatched = promote_person_roles(kg, {"person:elad-gil": "host"})
        assert (promoted, demoted) == (0, 1)
        assert kg["nodes"][0]["properties"]["role"] == "mentioned"

    def test_a_guest_the_roster_never_heard_becomes_mentioned(self) -> None:
        kg = _kg([("person:someone-on-tape", "Someone On Tape", "guest")])
        _promoted, demoted, _unmatched = promote_person_roles(kg, {"person:elad-gil": "host"})
        assert demoted == 1
        assert kg["nodes"][0]["properties"]["role"] == "mentioned"

    def test_the_show_itself_stops_being_a_host(self) -> None:
        # "The China-Global South Project" shipped as a host Person on real episodes.
        kg = _kg(
            [("person:the-china-global-south-project", "The China-Global South Project", "host")]
        )
        _promoted, demoted, _unmatched = promote_person_roles(kg, {"person:eric-olander": "host"})
        assert demoted == 1

    def test_someone_already_mentioned_is_not_touched(self) -> None:
        kg = _kg([("person:elon-musk", MENTIONED, "mentioned")])
        promoted, demoted, _unmatched = promote_person_roles(kg, {"person:elad-gil": "host"})
        assert (promoted, demoted) == (0, 0)
        assert kg["nodes"][0]["properties"]["role"] == "mentioned"

    def test_promote_and_demote_happen_in_the_same_pass(self) -> None:
        kg = _kg(
            [
                ("person:kevin-roose", HOST, "mentioned"),
                ("person:sarah-guo", "Sarah Guo", "host"),
            ]
        )
        promoted, demoted, _unmatched = promote_person_roles(kg, {"person:kevin-roose": "host"})
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
        kg = _kg([("person:sarah-guo", "Sarah Guo", "host")])
        _promoted, demoted, _unmatched = promote_person_roles(kg, {"person:elad-gil": "host"})
        assert demoted == 1
        assert kg["nodes"][0]["properties"]["role"] == "mentioned"

    def test_the_show_name_is_still_demoted(self) -> None:
        kg = _kg(
            [("person:the-china-global-south-project", "The China-Global South Project", "host")]
        )
        _promoted, demoted, _unmatched = promote_person_roles(kg, {"person:eric-olander": "host"})
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

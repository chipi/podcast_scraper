"""The episode's speaker record: placed voices and people only named, in one list (#2075).

Operator decision 2026-09-17: ONE speaker record per episode, `content.speakers`, and every surface
that shows people is written from it. Each entry says whether a voice was matched to that person
(`placed`). These tests pin the record's rules and the readers that must honour the flag.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from podcast_scraper.builders import context_digest_builder as digest
from podcast_scraper.kg import speaker_coherence
from podcast_scraper.search.cli_handlers import _speaker_infos
from podcast_scraper.upgrade.migrations import m0009_backfill_speaker_roles as m0009
from podcast_scraper.workflow import metadata_generation as mg

REL = "transcripts/0001 - Episode.txt"


def _seg(label: str, voice: str, role: Optional[str]) -> Dict[str, Any]:
    return {
        "start": 0.0,
        "end": 1.0,
        "text": "x",
        "speaker": voice,
        "speaker_label": label,
        "speaker_role": role,
    }


def _episode(
    tmp: Path,
    *,
    segments: Optional[List[Dict[str, Any]]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> str:
    (tmp / "transcripts").mkdir(parents=True, exist_ok=True)
    (tmp / REL).write_text("x\n", encoding="utf-8")
    base = str(tmp / REL)[: -len(".txt")]
    if segments is not None:
        Path(base + ".segments.json").write_text(json.dumps(segments), encoding="utf-8")
    if diagnostics is not None:
        Path(base + ".speakers.diagnostics.json").write_text(
            json.dumps(diagnostics), encoding="utf-8"
        )
    return str(tmp)


def _record(tmp: Path, hosts=None, guests=None, feed_title="A Show", **kw) -> List[mg.SpeakerInfo]:
    out = _episode(tmp, **kw)
    speakers, _n = mg._build_speaker_record(out, REL, hosts, guests, feed_title)
    return speakers


def _by_name(speakers: List[mg.SpeakerInfo]) -> Dict[str, mg.SpeakerInfo]:
    return {s.name: s for s in speakers}


class TestPlacedVoices:
    def test_a_named_voice_is_placed_with_its_voices_and_method(self, tmp_path: Path) -> None:
        rec = _by_name(
            _record(
                tmp_path,
                segments=[
                    _seg("Michael Barbaro", "SPEAKER_02", "host"),
                    _seg("Michael Barbaro", "SPEAKER_05", "host"),
                    _seg("Matina Stevis-Gridneff", "SPEAKER_00", "guest"),
                ],
                diagnostics={
                    "voices": [
                        {"voice": "SPEAKER_02", "source": "self_intro"},
                        {"voice": "SPEAKER_00", "source": "llm_resolution"},
                    ]
                },
            )
        )
        host = rec["Michael Barbaro"]
        assert (host.role, host.placed, host.voices, host.source) == (
            "host",
            True,
            ["SPEAKER_02", "SPEAKER_05"],
            "self_intro",
        )
        guest = rec["Matina Stevis-Gridneff"]
        assert (guest.role, guest.placed, guest.source) == ("guest", True, "llm_resolution")

    def test_a_placed_voice_with_no_diagnostics_says_roster(self, tmp_path: Path) -> None:
        rec = _by_name(_record(tmp_path, segments=[_seg("Ann Lee", "SPEAKER_00", "host")]))
        assert rec["Ann Lee"].source == "roster"

    def test_placed_ids_keep_the_existing_scheme(self, tmp_path: Path) -> None:
        speakers = _record(
            tmp_path,
            segments=[
                _seg("Ann Lee", "SPEAKER_00", "host"),
                _seg("Bo Ng", "SPEAKER_01", "host"),
                _seg("Cy Oh", "SPEAKER_02", "guest"),
            ],
        )
        assert [s.id for s in speakers] == ["host_1", "host_2", "guest"]


class TestPeopleOnlyNamedAreKeptAsUnplaced:
    def test_a_refused_host_is_kept_not_lost(self, tmp_path: Path) -> None:
        """The Daily: the feed states three hosts; the roster placed one and refused the rest.

        Before the record, Natalie Kitroeff vanished from every field on the episode.
        """
        rec = _by_name(
            _record(
                tmp_path,
                segments=[_seg("Michael Barbaro", "SPEAKER_02", "host")],
                diagnostics={
                    "tried": {
                        "known_hosts": ["Michael Barbaro", "Natalie Kitroeff", "Rachel Abrams"]
                    }
                },
            )
        )
        assert rec["Michael Barbaro"].placed is True
        for name in ("Natalie Kitroeff", "Rachel Abrams"):
            assert (rec[name].role, rec[name].placed, rec[name].voices, rec[name].source) == (
                "host",
                False,
                [],
                "feed_statement",
            )

    def test_a_guest_corroboration_refused_is_kept(self, tmp_path: Path) -> None:
        rec = _by_name(
            _record(
                tmp_path,
                segments=[_seg("Tyler Cowen", "SPEAKER_00", "host")],
                diagnostics={"summary": {"unbound_names": ["Diarmaid MacCulloch"]}},
            )
        )
        d = rec["Diarmaid MacCulloch"]
        assert (d.role, d.placed, d.source) == ("guest", False, "episode_metadata")

    def test_diarized_but_nobody_named_keeps_the_guess_only_as_unplaced(
        self, tmp_path: Path
    ) -> None:
        """The a16z shape: diarized, no voice named, the guess used to become the whole cast."""
        speakers = _record(
            tmp_path,
            hosts=["Erik Torenberg", "Ben Horowitz"],
            guests=["Garry Tan"],
            segments=[
                _seg("SPEAKER_00", "SPEAKER_00", None),
                _seg("SPEAKER_01", "SPEAKER_01", None),
            ],
        )
        assert speakers and all(s.placed is False for s in speakers)
        assert {(s.name, s.role, s.source) for s in speakers} == {
            ("Erik Torenberg", "host", "hint"),
            ("Ben Horowitz", "host", "hint"),
            ("Garry Tan", "guest", "hint"),
        }

    def test_never_diarized_keeps_the_guess_only_as_unplaced(self, tmp_path: Path) -> None:
        speakers = _record(tmp_path, hosts=["Joe Weisenthal"], guests=["Austan Goolsbee"])
        assert [(s.name, s.placed) for s in speakers] == [
            ("Joe Weisenthal", False),
            ("Austan Goolsbee", False),
        ]

    def test_unplaced_ids_cannot_collide_with_placed_ids(self, tmp_path: Path) -> None:
        speakers = _record(
            tmp_path,
            segments=[_seg("Ann Lee", "SPEAKER_00", "host")],
            diagnostics={"tried": {"known_hosts": ["Ann Lee", "Bo Ng", "Cy Oh"]}},
        )
        ids = [s.id for s in speakers]
        assert ids == ["host", "unplaced_1", "unplaced_2"]
        assert len(set(ids)) == len(ids)


class TestOnePersonOnce:
    def test_a_title_variant_of_a_placed_voice_is_not_listed_again(self, tmp_path: Path) -> None:
        """Measured: `Dr. Rafael Prieto-Curiel` unbound beside a placed `Rafael Prieto-Curiel`."""
        speakers = _record(
            tmp_path,
            segments=[_seg("Rafael Prieto-Curiel", "SPEAKER_01", "guest")],
            diagnostics={"summary": {"unbound_names": ["Dr. Rafael Prieto-Curiel"]}},
        )
        assert [(s.name, s.placed) for s in speakers] == [("Rafael Prieto-Curiel", True)]

    def test_the_same_name_from_two_sources_is_listed_once_with_the_stronger(
        self, tmp_path: Path
    ) -> None:
        speakers = _record(
            tmp_path,
            hosts=["Katie Martin"],
            diagnostics={"tried": {"known_hosts": ["Katie Martin"]}},
        )
        assert [(s.name, s.source) for s in speakers] == [("Katie Martin", "feed_statement")]

    def test_internal_whitespace_is_collapsed(self, tmp_path: Path) -> None:
        """`Amanda  Aronchik` shipped as a second person beside `Amanda Aronchik`."""
        speakers = _record(tmp_path, hosts=["Amanda  Aronchik", "Amanda Aronchik"])
        assert [s.name for s in speakers] == ["Amanda Aronchik"]


class TestNothingThatIsNotAPersonEnters:
    def test_placeholders_publishers_and_the_show_are_refused(self, tmp_path: Path) -> None:
        speakers = _record(
            tmp_path,
            feed_title="Conversations with Tyler",
            hosts=["Host", "Conversations with Tyler", "Tyler Cowen"],
            guests=["unknown_guest_1", "Pushkin Industries"],
            diagnostics={"summary": {"unbound_names": ["Host", "SPEAKER_03"]}},
        )
        assert [s.name for s in speakers] == ["Tyler Cowen"]


class TestTheModel:
    def test_a_pre_1_2_0_entry_still_validates_and_reads_as_unknown(self) -> None:
        legacy = mg.SpeakerInfo.model_validate({"id": "host", "name": "Ann Lee", "role": "host"})
        assert legacy.placed is None and legacy.voices == [] and legacy.source is None

    def test_the_computed_host_and_guest_projections_are_gone(self) -> None:
        content = mg.ContentMetadata(
            media_type="audio/mpeg",
            speakers=[
                mg.SpeakerInfo(id="unplaced_1", name="Never Heard", role="host", placed=False)
            ],
        )
        dumped = content.model_dump()
        assert "detected_hosts" not in dumped and "detected_guests" not in dumped

    def test_the_schema_version_announces_the_record(self) -> None:
        assert mg.SCHEMA_VERSION == "1.2.0"


# ---------------------------------------------------------------- the readers

RECORD = [
    {"id": "host", "name": "Michael Barbaro", "role": "host", "placed": True},
    {"id": "unplaced_1", "name": "Natalie Kitroeff", "role": "host", "placed": False},
    {"id": "unplaced_2", "name": "Diarmaid MacCulloch", "role": "guest", "placed": False},
]
LEGACY = [
    {"id": "host", "name": "Michael Barbaro", "role": "host"},
    {"id": "guest", "name": "Matina Stevis-Gridneff", "role": "guest"},
]


def _ns(entries):
    return [SimpleNamespace(**{k: v for k, v in e.items() if k != "id"}) for e in entries]


class TestTheGraphIsCastOnlyFromPlacedVoices:
    def test_an_unplaced_person_is_never_a_host_or_guest(self) -> None:
        hosts, guests = mg._speaker_lists_for_graph(_ns(RECORD), [], [], "The Daily")
        assert hosts == ["Michael Barbaro"]
        assert guests == []

    def test_a_record_with_nobody_placed_never_falls_back_to_the_guess(self) -> None:
        only_named = [e for e in RECORD if e["placed"] is False]
        hosts, guests = mg._speaker_lists_for_graph(
            _ns(only_named), ["Natalie Kitroeff"], ["Diarmaid MacCulloch"], "The Daily"
        )
        assert (hosts, guests) == ([], [])

    def test_a_pre_record_artifact_keeps_its_old_behaviour(self) -> None:
        hosts, guests = mg._speaker_lists_for_graph(_ns(LEGACY), [], [], "The Daily")
        assert (hosts, guests) == (["Michael Barbaro"], ["Matina Stevis-Gridneff"])


class TestEnrichEdgesKeepsTheFlag:
    def test_the_flag_survives_conversion(self) -> None:
        infos = _speaker_infos(RECORD + LEGACY[1:])
        assert [i.placed for i in infos] == [True, False, False, None]

    def test_and_so_enrich_edges_casts_only_placed_voices(self) -> None:
        hosts, guests = mg._speaker_lists_for_graph(_speaker_infos(RECORD), [], [], "The Daily")
        assert (hosts, guests) == (["Michael Barbaro"], [])


class TestTheSpokeCheckStillCatchesThem:
    def test_unplaced_people_are_not_voices_the_roster_heard(self) -> None:
        names = speaker_coherence.roster_names({"content": {"speakers": RECORD}})
        assert names == ["Michael Barbaro"]

    def test_legacy_entries_are_still_counted(self) -> None:
        names = speaker_coherence.roster_names({"content": {"speakers": LEGACY}})
        assert names == ["Michael Barbaro", "Matina Stevis-Gridneff"]


class TestM0009NeverPromotesAPersonOnlyNamed:
    def test_unplaced_people_get_no_role_to_write(self) -> None:
        roles = m0009.roster_roles(
            {"feed": {"title": "The Daily"}, "content": {"speakers": RECORD}}
        )
        assert list(roles.values()) == ["host"]
        assert not any("kitroeff" in pid or "macculloch" in pid for pid in roles)

    def test_legacy_entries_are_unchanged(self) -> None:
        roles = m0009.roster_roles(
            {"feed": {"title": "The Daily"}, "content": {"speakers": LEGACY}}
        )
        assert sorted(roles.values()) == ["guest", "host"]


class TestTheContextDigestListsOnlyPlacedHostsAndGuests:
    def test_record(self) -> None:
        block = digest._basic_block({"content": {"speakers": RECORD}})
        assert (block["hosts"], block["guests"]) == (["Michael Barbaro"], [])

    def test_pre_record_artifact_uses_its_old_fields(self) -> None:
        block = digest._basic_block(
            {"content": {"detected_hosts": ["Ann Lee"], "detected_guests": ["Bo Ng"]}}
        )
        assert (block["hosts"], block["guests"]) == (["Ann Lee"], ["Bo Ng"])

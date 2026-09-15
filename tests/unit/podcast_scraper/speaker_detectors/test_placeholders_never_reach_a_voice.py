"""A placeholder is the absence of a name. It was being published as one (#2075).

Every provider returns `DEFAULT_SPEAKER_NAMES` (`["Host", "unknown_guest_1"]`) as its FAILURE
value. Those strings flowed onward as though detection had succeeded — into detected_hosts /
detected_guests, into `metadata_named`, and so into the closed candidate list the resolver may
match a voice against. The resolver then did exactly what it is built to do.

Measured on the production snapshot: the literal string `Host` is published as a speaker on 59
episodes and `unknown_guest_1` on 17, all `source=llm_resolution`; on 29 episodes a placeholder is
the ONLY roster name, so the episode reads as attributed when nobody was identified at all.
"""

from __future__ import annotations

from podcast_scraper.speaker_detectors.constants import DEFAULT_SPEAKER_NAMES
from podcast_scraper.speaker_detectors.normalization import (
    filter_default_speaker_names,
    is_default_speaker_name,
)
from podcast_scraper.workflow.metadata_generation import _build_speakers_from_detected_names


class TestTheConstantIsStillThere:
    def test_the_providers_failure_contract_is_unchanged(self) -> None:
        # Seven providers return this tuple on failure and `qa_flags` reads it; deleting it was
        # the obvious fix and the wrong one. The leak is fixed at the candidate boundary instead.
        assert DEFAULT_SPEAKER_NAMES == ["Host", "unknown_guest_1"]
        assert is_default_speaker_name("Host") is True
        assert is_default_speaker_name("unknown_guest_1") is True


class TestTheHintPathRefusesThem:
    def test_a_placeholder_host_builds_no_speaker(self) -> None:
        assert _build_speakers_from_detected_names(["Host"], []) == []

    def test_a_placeholder_guest_builds_no_speaker(self) -> None:
        assert _build_speakers_from_detected_names([], ["unknown_guest_1"]) == []

    def test_a_real_person_alongside_a_placeholder_survives_alone(self) -> None:
        speakers = _build_speakers_from_detected_names(["Host", "Russ Roberts"], ["Ada Lovelace"])
        assert [(s.name, s.role) for s in speakers] == [
            ("Russ Roberts", "host"),
            ("Ada Lovelace", "guest"),
        ]

    def test_the_id_reflects_the_filtered_count(self) -> None:
        # Two hosts in, one real: the survivor is "host", not "host_1" of a phantom pair.
        speakers = _build_speakers_from_detected_names(["Host", "Russ Roberts"], [])
        assert [s.id for s in speakers] == ["host"]


class TestTheFilterItself:
    def test_it_removes_only_placeholders(self) -> None:
        assert filter_default_speaker_names(["Host", "Ada Lovelace", "unknown_guest_1"]) == [
            "Ada Lovelace"
        ]

    def test_a_real_person_whose_name_merely_contains_host_survives(self) -> None:
        assert filter_default_speaker_names(["Hosting Smith"]) == ["Hosting Smith"]

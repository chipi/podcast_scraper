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
from podcast_scraper.workflow.metadata_generation import _unplaced_speakers


class TestTheConstantIsStillThere:
    def test_the_providers_failure_contract_is_unchanged(self) -> None:
        # Seven providers return this tuple on failure and `qa_flags` reads it; deleting it was
        # the obvious fix and the wrong one. The leak is fixed at the candidate boundary instead.
        assert DEFAULT_SPEAKER_NAMES == ["Host", "unknown_guest_1"]
        assert is_default_speaker_name("Host") is True
        assert is_default_speaker_name("unknown_guest_1") is True


class TestTheHintPathRefusesThem:
    """The pre-listening hint enters the speaker record as unplaced people (#2075).

    It used to build the roster directly; the guard moved with it. A placeholder is a provider's
    FAILURE value, not a person anyone named, so it must not appear in the record at all — not even
    as someone only named.
    """

    @staticmethod
    def _names(hosts, guests):
        return [
            (s.name, s.role)
            for s in _unplaced_speakers(
                [], diagnostics={}, detected_hosts=hosts, detected_guests=guests, feed_title=None
            )
        ]

    def test_a_placeholder_host_builds_no_entry(self) -> None:
        assert self._names(["Host"], []) == []

    def test_a_placeholder_guest_builds_no_entry(self) -> None:
        # `unknown_guest_1` is not a bare speaker label, so a single predicate let it through.
        assert self._names([], ["unknown_guest_1"]) == []

    def test_a_real_person_alongside_a_placeholder_survives_alone(self) -> None:
        assert self._names(["Host", "Russ Roberts"], ["Ada Lovelace"]) == [
            ("Russ Roberts", "host"),
            ("Ada Lovelace", "guest"),
        ]


class TestTheFilterItself:
    def test_it_removes_only_placeholders(self) -> None:
        assert filter_default_speaker_names(["Host", "Ada Lovelace", "unknown_guest_1"]) == [
            "Ada Lovelace"
        ]

    def test_a_real_person_whose_name_merely_contains_host_survives(self) -> None:
        assert filter_default_speaker_names(["Hosting Smith"]) == ["Hosting Smith"]

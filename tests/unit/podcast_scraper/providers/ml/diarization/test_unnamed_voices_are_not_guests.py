"""An unnamed voice is not a guest just because the episode HAS a guest (#2062).

FOUND BY A FRESH DGX INGEST on 2026-09-13. Two episodes of "The Journal." were transcribed and
diarized for real; one came back with ten voices, of which the roster named three:

    SPEAKER_04  Ryan Knutson     role=host   named=True
    SPEAKER_01  Jessica Mendoza  role=host   named=True
    SPEAKER_09  Heather Haddon   role=guest  named=True
    SPEAKER_00/02/03/05/06/07/08  (unnamed)  role=guest  <- all seven

Seven anonymous voices — archival tape, a caller, a narrator, a diarization over-split — were each
labelled a GUEST of the episode. The rule read ``guest if (guest_names or has_guest_intro)``, and
``guest_names`` is EPISODE-wide: once any guest evidence exists anywhere in the episode, every
leftover voice inherits it, including after the real guest has already been matched to their voice.

ON PRODUCTION (330-episode feed-stratified sample): 221 of 538 unnamed voices (41.1%) carry
``role="guest"``, across 32.3% of episodes.

HOW FAR IT REACHES, measured rather than assumed. It does NOT reach the graph: the metadata builder
skips any ``speaker_label`` beginning with "speaker", so across the same sample all 515
``content.speakers`` entries carry a real name and no ``kg.json`` Person named ``SPEAKER_NN`` is a
guest. What it corrupts is the ``.speakers.diagnostics.json`` sidecar and the roster object behind
it — the operator-facing record of who the diarizer heard, and the artifact an audit reads to answer
"did we identify this episode's guest?". An episode whose guest we never named reports seven of
them, so the one number that matters is unavailable exactly when it is needed.

THE RULE THIS PINS. Guest evidence is CONSUMED, not shared. N unclaimed guest names can explain at
most N anonymous voices; past that there is no evidence left and the honest label is ``unknown``.
An unnamed voice costs a ``SPEAKER_01`` on screen. A wrongly-labelled one tells the listener a
stranger was the guest.
"""

from __future__ import annotations

from typing import Dict

import pytest

from podcast_scraper.providers.ml.diarization.roster import (
    _name_guest_voices,
    SpeakerRole,
)

pytestmark = pytest.mark.unit


def _roles(out: Dict[str, SpeakerRole]) -> Dict[str, str]:
    return {v: r.role for v, r in out.items()}


class TestEvidenceIsConsumedNotShared:
    def test_leftovers_are_unknown_once_the_named_guest_is_placed(self) -> None:
        # The prod shape: one real guest, self-introduced, plus a crowd of anonymous voices.
        out = _name_guest_voices(
            voices_by_total=["SPEAKER_09", "SPEAKER_00", "SPEAKER_02", "SPEAKER_03"],
            assigned={},
            voice_intro={"SPEAKER_09": "Heather Haddon"},
            guest_names=["Heather Haddon"],
            host_names_lower=set(),
            used_lower=set(),
        )
        roles = _roles(out)
        assert roles["SPEAKER_09"] == "guest"
        for v in ("SPEAKER_00", "SPEAKER_02", "SPEAKER_03"):
            assert roles[v] == "unknown", f"{v} was called a guest with no evidence of its own"

    def test_two_unclaimed_names_can_explain_two_voices(self) -> None:
        # Not over-correcting: real evidence still gets used.
        out = _name_guest_voices(
            voices_by_total=["SPEAKER_00", "SPEAKER_01"],
            assigned={},
            voice_intro={},
            guest_names=["Ada Lovelace", "Alan Turing"],
            host_names_lower=set(),
            used_lower=set(),
        )
        assert set(_roles(out).values()) == {"guest"}

    def test_one_unclaimed_name_explains_one_voice_not_three(self) -> None:
        out = _name_guest_voices(
            voices_by_total=["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
            assigned={},
            voice_intro={},
            guest_names=["Ada Lovelace"],
            host_names_lower=set(),
            used_lower=set(),
        )
        roles = _roles(out)
        assert sum(1 for r in roles.values() if r == "guest") <= 1
        assert sum(1 for r in roles.values() if r == "unknown") >= 2

    def test_no_guest_evidence_at_all_leaves_everything_unknown(self) -> None:
        # #1170: a host-only-intro show must not manufacture guests.
        out = _name_guest_voices(
            voices_by_total=["SPEAKER_00", "SPEAKER_01"],
            assigned={},
            voice_intro={},
            guest_names=[],
            host_names_lower=set(),
            used_lower=set(),
        )
        assert set(_roles(out).values()) == {"unknown"}

    def test_a_name_already_used_is_not_evidence_again(self) -> None:
        # The guest was matched on an earlier pass; their name cannot vouch for a second voice.
        out = _name_guest_voices(
            voices_by_total=["SPEAKER_00"],
            assigned={},
            voice_intro={},
            guest_names=["Ada Lovelace"],
            host_names_lower=set(),
            used_lower={"ada lovelace"},
        )
        assert _roles(out)["SPEAKER_00"] == "unknown"


class TestNamingStillWorks:
    def test_a_self_introduced_guest_is_still_named(self) -> None:
        out = _name_guest_voices(
            voices_by_total=["SPEAKER_01"],
            assigned={},
            voice_intro={"SPEAKER_01": "Heather Haddon"},
            guest_names=[],
            host_names_lower=set(),
            used_lower=set(),
        )
        assert out["SPEAKER_01"].named is True
        assert out["SPEAKER_01"].name == "Heather Haddon"
        assert out["SPEAKER_01"].role == "guest"

    def test_the_forced_one_name_one_voice_match_still_happens(self) -> None:
        out = _name_guest_voices(
            voices_by_total=["SPEAKER_01"],
            assigned={},
            voice_intro={},
            guest_names=["Ada Lovelace"],
            host_names_lower=set(),
            used_lower=set(),
        )
        assert out["SPEAKER_01"].name == "Ada Lovelace"
        assert out["SPEAKER_01"].named is True

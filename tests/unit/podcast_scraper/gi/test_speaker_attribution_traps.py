"""Attribution traps: a name in the TEXT is not a name on the MIC.

Same bug class as the speaker-detection traps one layer down, and the same asymmetry: an insight
attributed to the wrong person is worse than an insight attributed to nobody. Here the failure mode
is a third party NAMED INSIDE someone else's turn — the hosts spend four minutes discussing Elon
Musk, and every sentence in that stretch is spoken by Kevin Roose.

``_NAMED_TURN_RE`` is anchored to line-start, which is what keeps mid-prose names out of the turn
map. That anchor is load-bearing and invisible; loosening it (to catch, say, an inline "Kevin:")
would silently turn every discussed celebrity into a speaker. These tests exist so that loosening
fails loudly.

The last test pins the cleaning-profile invariant. ``cleaning_v4`` anonymises speakers ("Kevin
Roose:" -> "A:"), which is right for summarisation and fatal for GI: it destroys the only speaker
signal in the transcript, and attribution degrades to None everywhere without erroring.
``cleaning_v5`` is v4 minus that step. Nothing else enforces the difference.
"""

from __future__ import annotations

from podcast_scraper.gi.speakers import build_unverified_named_turns, speaker_for_char
from podcast_scraper.preprocessing.profiles import apply_profile_with_stats

# A host discussing a third party. Every char of Kevin's turn belongs to Kevin — including the
# stretch where he says "Elon Musk". This is the production shape that misattributed the corpus.
TRANSCRIPT = """Kevin Roose: This week OpenAI announced a loosened partnership with Microsoft.
Elon Musk is suing them, and Sam Altman says the company can still scale. I think that's wrong.

Casey Newton: I disagree. The trial is a distraction.

Dr. Adam Rodman: Doctors are already using chatbots for differential diagnosis, and the
cyberchondria worry is overstated.
"""


def _speaker_of(needle: str) -> object:
    turns = build_unverified_named_turns(TRANSCRIPT)
    return speaker_for_char(TRANSCRIPT.index(needle), turns)


def test_a_name_spoken_inside_a_turn_does_not_steal_the_turn() -> None:
    """THE TRAP. Kevin says "Elon Musk". Musk is not the speaker — Kevin is."""
    assert _speaker_of("Elon Musk is suing them") == "Kevin Roose"
    assert _speaker_of("Sam Altman says the company") == "Kevin Roose"
    assert _speaker_of("I think that's wrong") == "Kevin Roose"


def test_the_real_speakers_still_attribute() -> None:
    """A detector so timid it attributes nobody would pass the trap above and be useless."""
    assert _speaker_of("I disagree") == "Casey Newton"
    assert _speaker_of("cyberchondria worry") == "Dr. Adam Rodman"


def test_only_real_turn_markers_become_speakers() -> None:
    """Musk and Altman are named in the transcript. Neither may appear as a speaker."""
    speakers = {label for _off, label in build_unverified_named_turns(TRANSCRIPT)}
    assert speakers == {"Kevin Roose", "Casey Newton", "Dr. Adam Rodman"}
    assert "Elon Musk" not in speakers
    assert "Sam Altman" not in speakers


def test_prose_colons_and_publishers_are_not_speakers() -> None:
    """``Note:`` is not a person and ``Bloomberg:`` is not a mouth."""
    noisy = (
        "Note: the following is sponsored.\nBloomberg: a network, not a person.\nKevin Roose: Hi.\n"
    )
    speakers = {label for _off, label in build_unverified_named_turns(noisy)}
    assert speakers == {"Kevin Roose"}


def test_cleaning_v4_destroys_attribution_and_v5_preserves_it() -> None:
    """The invariant that made GI attribute nobody for the entire corpus.

    v4 anonymises speakers — correct for summarisation (it cut speaker-name leak into summaries from
    ~80% to <10%), fatal for GI. The failure is SILENT: no error, just ``speaker: None`` on every
    insight. If someone ever points GI at v4 again, this fails instead of shipping a corpus with no
    speakers.
    """
    v4, _ = apply_profile_with_stats(TRANSCRIPT, "cleaning_v4")
    v5, _ = apply_profile_with_stats(TRANSCRIPT, "cleaning_v5")

    # v4 must anonymise EVERY speaker. Writing this fixture is what exposed that it didn't: the
    # honorific in "Dr. Adam Rodman:" defeated the pattern, so the guest's name — the one name a
    # summary must never parrot — sailed through while the hosts' names were scrubbed.
    assert not build_unverified_named_turns(v4), (
        "cleaning_v4 is expected to anonymise speakers away — if it no longer does, GI's profile "
        "choice needs revisiting, and so does the summariser's speaker-leak rate."
    )
    assert "Rodman" not in v4 and "Kevin" not in v4

    v5_speakers = {label for _off, label in build_unverified_named_turns(v5)}
    assert "Kevin Roose" in v5_speakers, "cleaning_v5 must preserve the names GI attributes from"
    assert "Dr. Adam Rodman" in v5_speakers

"""A publisher transcript's speakers survive parsing, and its hosts are named, not guessed (#2075).

Found verifying `retranscript_only` on one Odd Lots episode on the DGX. Synthetic cues; the shapes
are the measured ones:

* its SRT writes the speaker as a line prefix (`Speaker 3: …`), which `parse_srt` never read — the
  feed lists the SRT first, so every ingest lost the speakers the file carried;
* one host's voice says "And I'm Joe Weisenthal" at the open and, garbled, "And I'm Jill
  Wiesenthal" at the close: counted as two people it was suppressed as a montage, and the
  introduction reader then gave the GUEST's name to that host's voice;
* the other host's "I'm Tracy Allaway" was snapped only for host-candidate voices, so she was
  published under the ASR spelling, discarded by the resolver as unstated, and cast as a guest.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.roster import (
    _intro_people,
    _snap_near_identical_host,
)
from podcast_scraper.transcript_formats import parse_srt

pytestmark = pytest.mark.unit


class TestSrtSpeakerPrefix:
    SRT = (
        "1\n00:00:18,079 --> 00:00:20,619\nSpeaker 3: Welcome back. I'm Ada Brook.\n\n"
        "2\n00:00:22,400 --> 00:00:23,570\nSpeaker 4: And I'm Ben Carver.\n\n"
        "3\n00:00:24,000 --> 00:00:25,000\nNote: this cue has no speaker.\n"
    )

    def test_the_prefix_becomes_the_segment_speaker_and_leaves_the_text(self) -> None:
        plain, segs = parse_srt(self.SRT)
        assert [s.get("speaker") for s in segs] == ["Speaker 3", "Speaker 4", None]
        assert segs[0]["text"].startswith("Welcome")
        assert "Speaker 3:" not in plain

    def test_a_prose_colon_is_not_a_speaker(self) -> None:
        _plain, segs = parse_srt(self.SRT)
        assert segs[2]["text"].startswith("Note: this cue")


class TestIntroPeople:
    def test_an_asr_respelling_of_one_host_is_one_person(self) -> None:
        assert _intro_people(["Joe Weisenthal", "Jill Wiesenthal"]) == 1

    def test_a_real_montage_of_two_hosts_is_two(self) -> None:
        assert _intro_people(["Kevin Roose", "Casey Newton"]) == 2

    def test_short_surnames_are_not_collapsed(self) -> None:
        assert _intro_people(["Ada Pape", "Ben Page"]) == 2


class TestSnapNearIdenticalHost:
    HOSTS = ["Joe Weisenthal", "Tracy Alloway"]

    def test_one_letter_from_a_stated_host_is_that_host(self) -> None:
        assert _snap_near_identical_host("Tracy Allaway", self.HOSTS) == "Tracy Alloway"

    @pytest.mark.parametrize(
        ("name", "hosts"),
        [
            ("Kevin Ross", ["Kevin Roose"]),  # two edits: a real, different surname
            ("Tracey Alloway", ["Tracy Alloway"]),  # first name must be exact
            ("Ada Pope", ["Ada Pape"]),  # a surname under 5 letters is too short to trust one edit
            ("Tracy", ["Tracy Alloway"]),  # a mononym is not snapped here
        ],
    )
    def test_anything_further_is_left_alone(self, name: str, hosts: list) -> None:
        assert _snap_near_identical_host(name, hosts) == name


def test_the_odd_lots_shape_names_both_hosts_and_never_gives_the_guests_name_to_one() -> None:
    """Through ``resolve_speaker_roster``, in the measured shape: the publisher's voices mix the
    network bumper into a host and the guest, and one host's intro is garbled at the close."""
    from podcast_scraper.providers.ml.diarization.base import (
        DiarizationResult,
        DiarizationSegment,
    )
    from podcast_scraper.providers.ml.diarization.roster import resolve_speaker_roster

    guest_answer = "So I would put it in a much more macro context of the price of money. " * 12
    host_q = "Have you become more optimistic about the durability of the expansion? " * 3
    turns = [
        ("S0", "Network Audio Studios."),
        ("S1", "Podcasts. Radio. News."),
        (
            "S2",
            "Well, hello and welcome to another episode of the Odd Lots podcast. I'm Ada Allaway.",
        ),
        ("S3", "And I'm Ben Weisenthal."),
        ("S2", "We're going to be speaking with Regional Fed President Cal Schmidt. Welcome back."),
        ("S1", guest_answer),
        ("S0", host_q),
        ("S1", guest_answer),
        ("S2", "This has been another episode of the Odd Lots podcast. I'm Ada Allaway."),
        ("S3", "And I'm Bill Wiesenthal. You can follow me online."),
    ]
    segs, t = [], 30.0
    for voice, text in turns:
        dur = max(4.0, len(text) / 15)
        segs.append(DiarizationSegment(start=t, end=t + dur, speaker=voice))
        t += dur
    voice_texts: dict = {}
    for v, text in turns:
        voice_texts[v] = (voice_texts.get(v, "") + " " + text).strip()
    roster = resolve_speaker_roster(
        DiarizationResult(segments=segs, num_speakers=4),
        " ".join(x for _, x in turns),
        known_hosts=["Ben Weisenthal", "Ada Alloway"],
        voice_texts=voice_texts,
        ordered_turns=turns,
    )
    by = {v: (r.name, r.role) for v, r in roster.by_voice.items()}
    assert by["S2"] == ("Ada Alloway", "host"), by
    assert by["S3"] == ("Ben Weisenthal", "host"), by
    assert all(r.name != "Cal Schmidt" for v, r in roster.by_voice.items() if v != "S1"), by


class TestTheIntroducedGuestIsNamed:
    """Odd Lots, measured: the guest (who runs the symposium) says "Well, welcome to Jackson Hole",
    so the conversation flagged his voice a host and the introduction reader skipped it; and the
    host's spoken "Jeff Schmidt" sat in the guest pool beside the stated `Jeffrey Schmid`, so the
    extra copy was forced onto a host's question voice."""

    def test_a_conversation_host_stops_blocking_once_every_stated_host_is_seated(self) -> None:
        from podcast_scraper.providers.ml.diarization.roster import _hosts_not_already_seated

        intro = {"S2": "Ada Alloway", "S3": "Ben Weisenthal"}
        hosts = ["Ada Alloway", "Ben Weisenthal"]
        assert _hosts_not_already_seated({"S1", "S2"}, intro, hosts) == {"S2"}

    def test_it_still_blocks_while_a_stated_host_is_unseated(self) -> None:
        from podcast_scraper.providers.ml.diarization.roster import _hosts_not_already_seated

        intro = {"S2": "Ada Alloway"}
        hosts = ["Ada Alloway", "Ben Weisenthal"]
        assert _hosts_not_already_seated({"S1", "S2"}, intro, hosts) == {"S1", "S2"}

    @pytest.mark.parametrize(
        ("spoken", "stated", "expected"),
        [
            ("Jeff Schmidt", ["Jeffrey Schmid"], "Jeffrey Schmid"),
            ("Jeff Schmidt", ["Jeffrey Schmid", "Jeffrey Schmit"], "Jeff Schmidt"),  # ambiguous
            ("Eric Schmidt", ["Jeffrey Schmid"], "Eric Schmidt"),  # different given name
            ("Ada Pope", ["Ada Pape"], "Ada Pope"),  # surname too short to trust
        ],
    )
    def test_snap_spoken_variant(self, spoken: str, stated: list, expected: str) -> None:
        from podcast_scraper.providers.ml.diarization.roster import _snap_spoken_variant

        assert _snap_spoken_variant(spoken, stated) == expected

    def test_through_the_roster_the_guest_is_named_and_nobody_else_gets_his_name(self) -> None:
        from podcast_scraper.providers.ml.diarization.base import (
            DiarizationResult,
            DiarizationSegment,
        )
        from podcast_scraper.providers.ml.diarization.roster import resolve_speaker_roster

        guest = "So I would put it in a much more macro context of the price of money. " * 12
        question = "Have you become more optimistic about the durability of the expansion? " * 3
        turns = [
            ("S0", "Network Audio Studios."),
            ("S1", "Podcasts. Radio. News."),
            (
                "S2",
                "Hello and welcome to another episode of the Odd Lots podcast. I'm Ada Alloway.",
            ),
            ("S3", "And I'm Ben Weisenthal."),
            ("S2", "We're going to be speaking with Regional Fed President Jeff Schmidt."),
            ("S2", "Thank you so much for coming back on the show."),
            ("S1", "Well, welcome to Jackson Hole. I mean, this is amazing. " + guest),
            ("S0", question),
            ("S1", guest),
            ("S3", "Well, Jeff Schmidt, thank you so much for coming back."),
            ("S1", "I love doing this. Thank you."),
        ]
        segs, t = [], 30.0
        for voice, text in turns:
            dur = max(4.0, len(text) / 15)
            segs.append(DiarizationSegment(start=t, end=t + dur, speaker=voice))
            t += dur
        voice_texts: dict = {}
        for v, text in turns:
            voice_texts[v] = (voice_texts.get(v, "") + " " + text).strip()
        roster = resolve_speaker_roster(
            DiarizationResult(segments=segs, num_speakers=4),
            " ".join(x for _, x in turns),
            known_hosts=["Ada Alloway", "Ben Weisenthal"],
            detected_guests=["Jeffrey Schmid"],
            metadata_named=["Jeffrey Schmid"],
            voice_texts=voice_texts,
            ordered_turns=turns,
        )
        by = {v: (r.name, r.role) for v, r in roster.by_voice.items()}
        assert by["S1"] == ("Jeffrey Schmid", "guest"), by
        assert [v for v, (n, _r) in by.items() if n == "Jeffrey Schmid"] == ["S1"], by

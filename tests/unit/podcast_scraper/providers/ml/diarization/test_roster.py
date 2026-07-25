"""Unit tests for the unified speaker-roster resolver (#876).

Exercises every edge case from the pipeline tech review: host+guest, solo, panel (>2),
co-hosted, host-without-self-intro (feed fallback), network-author stripping, the
guest-name-never-on-host regression, and name/voice count mismatches.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import (
    build_speaker_diagnostics,
    resolve_speaker_roster,
)

pytestmark = pytest.mark.unit


def _diar(segs: List[Tuple[str, float, float]], num_speakers: int) -> DiarizationResult:
    return DiarizationResult(
        segments=[DiarizationSegment(start=s, end=e, speaker=spk) for spk, s, e in segs],
        num_speakers=num_speakers,
        model_name="test",
    )


def test_host_guest_basic() -> None:
    # HOST owns the intro; GUEST talks the most overall. Self-intro names the host.
    diar = _diar([("HOST", 0, 60), ("GUEST", 60, 400), ("HOST", 400, 420)], 2)
    r = resolve_speaker_roster(
        diar,
        "Hello and welcome. I'm Patrick O'Shaughnessy. My guest is Brian Chesky.",
        detected_guests=["Brian Chesky"],
    )
    assert r.by_voice["HOST"].name == "Patrick O'Shaughnessy"
    assert r.by_voice["HOST"].role == "host" and r.by_voice["HOST"].source == "self_intro"
    assert r.by_voice["GUEST"].name == "Brian Chesky"
    assert r.by_voice["GUEST"].role == "guest"
    assert r.num_speakers == 2 and r.named_count() == 2


def test_host_is_opener_even_when_guest_out_talks_intro() -> None:
    # #1169 host/guest swap: the HOST opens the episode ("welcome back…") but the GUEST
    # gives a long early answer and out-talks the host WITHIN the 90s intro window
    # (guest 60s vs host 30s). The host is the OPENER (earliest turn), not the
    # intro-window talk-time leader — otherwise the known-host name lands on the guest.
    diar = _diar(
        [
            ("SPEAKER_00", 0, 20),
            ("SPEAKER_01", 20, 80),
            ("SPEAKER_00", 80, 95),
            ("SPEAKER_01", 95, 400),
        ],
        2,
    )
    r = resolve_speaker_roster(
        diar,
        "Welcome back to the show. Today my guest is Brian Chesky.",
        known_hosts=["Patrick O'Shaughnessy"],
        detected_guests=["Brian Chesky"],
    )
    assert r.by_voice["SPEAKER_00"].role == "host"
    assert r.by_voice["SPEAKER_00"].name == "Patrick O'Shaughnessy"
    assert r.by_voice["SPEAKER_01"].role == "guest"
    assert r.by_voice["SPEAKER_01"].name == "Brian Chesky"


def test_solo_monologue_named_by_self_intro() -> None:
    diar = _diar([("SPEAKER_00", 0, 300)], 1)
    r = resolve_speaker_roster(diar, "I'm Patrick O'Shaughnessy and today we talk markets.")
    assert r.by_voice["SPEAKER_00"].name == "Patrick O'Shaughnessy"
    assert r.by_voice["SPEAKER_00"].role == "host"


def test_solo_no_intro_stays_raw() -> None:
    diar = _diar([("SPEAKER_00", 0, 300)], 1)
    r = resolve_speaker_roster(diar, "Today we discuss markets.")
    role = r.by_voice["SPEAKER_00"]
    assert role.name == "SPEAKER_00" and role.named is False and role.source == "raw"


def test_panel_names_nobody_it_cannot_place() -> None:
    """ADR-110 — CHANGED BEHAVIOUR, deliberately.

    HOST + three guest voices, two guest names. This test used to assert that Alice and Bob were
    handed to the two loudest guest voices, in talk-time order. Nothing tied either name to either
    voice: the second-loudest speaker simply got the second name.

    That is the invention mechanism behind every wrong name we have shipped, and it was caught in
    the act on FT Unhedged — Robert Armstrong painted onto the wrong voice, and Katie Martin, the
    show's lead host, onto a voice with 4% of the talk.

    With two names and three unplaced voices there is a CHOICE, so there is a GUESS. We do not
    guess. The names go unused and the voices stay raw; the LLM resolution (ADR-110) is what places
    them, from what they actually said.
    """
    diar = _diar(
        [("HOST", 0, 50), ("G1", 50, 220), ("G2", 220, 380), ("G3", 380, 430), ("HOST", 430, 440)],
        4,
    )
    r = resolve_speaker_roster(
        diar,
        "I'm Patrick O'Shaughnessy, here with our panel.",
        detected_guests=["Alice", "Bob"],
    )
    assert r.by_voice["HOST"].name == "Patrick O'Shaughnessy"
    named_guests = {v.name for v in r.by_voice.values() if v.role == "guest" and v.named}
    assert named_guests == set(), "a name was painted on a voice with no evidence tying it there"


def test_one_name_one_voice_is_FORCED_and_so_is_not_a_guess() -> None:
    """The other side of the rule. One name left, one voice left — there is no choice to make."""
    diar = _diar([("HOST", 0, 60), ("GUEST", 60, 300)], 2)
    r = resolve_speaker_roster(
        diar,
        "I'm Patrick O'Shaughnessy.",
        detected_guests=["Brian Chesky"],
    )
    assert r.by_voice["GUEST"].name == "Brian Chesky"
    assert r.by_voice["GUEST"].source == "forced"


def test_host_selfintro_no_guests_leftover_is_unknown() -> None:
    # Regression (#1170 harden): a host self-introduces but NO guests are detected.
    # A leftover unnamed voice (backchannel / phantom / short interjection) must be
    # role="unknown", NOT "guest" — the episode-wide self-intro dict includes the
    # host's own intro, and that must not paint unrelated voices as guests.
    diar = _diar([("HOST", 0, 300), ("OTHER", 300, 320)], 2)
    r = resolve_speaker_roster(
        diar,
        "I'm Patrick O'Shaughnessy and today we talk markets.",
        voice_texts={
            "HOST": "I'm Patrick O'Shaughnessy and today we talk markets.",
            "OTHER": "yeah mm-hmm right",
        },
    )
    assert r.by_voice["HOST"].role == "host"
    other = r.by_voice["OTHER"]
    assert other.named is False
    assert other.role == "unknown", f"leftover voice should be unknown, got {other.role!r}"


def test_co_hosted_via_known_hosts() -> None:
    # Two intro-dominant voices + two known host names → both named as hosts.
    diar = _diar([("H1", 0, 50), ("H2", 50, 90), ("GUEST", 90, 400), ("H1", 400, 420)], 3)
    r = resolve_speaker_roster(
        diar,
        "Welcome back everyone.",
        known_hosts=["Anna Adams", "Ben Baker"],
        detected_guests=["Grace Green"],
    )
    host_names = {v.name for v in r.by_voice.values() if v.role == "host"}
    assert host_names == {"Anna Adams", "Ben Baker"}
    assert r.by_voice["GUEST"].name == "Grace Green"


def test_host_not_self_introduced_falls_back_to_feed() -> None:
    # No "I'm …" in the transcript, but the feed/NER gave us the host name.
    diar = _diar([("HOST", 0, 60), ("GUEST", 60, 300)], 2)
    r = resolve_speaker_roster(
        diar,
        "So let's get into it.",
        host_candidates=["Patrick O'Shaughnessy"],
        detected_guests=["Brian Chesky"],
    )
    assert r.by_voice["HOST"].name == "Patrick O'Shaughnessy"
    assert r.by_voice["HOST"].source == "feed"
    assert r.by_voice["GUEST"].name == "Brian Chesky"


def test_network_author_names_stripped() -> None:
    # All host candidates look like networks → no host name; host voice stays raw.
    diar = _diar([("HOST", 0, 60), ("GUEST", 60, 300)], 2)
    r = resolve_speaker_roster(
        diar,
        "Let's begin.",
        host_candidates=["Colossus", "Colossus | Investing & Business Podcasts"],
        detected_guests=["Brian Chesky"],
    )
    assert r.by_voice["HOST"].named is False
    assert "Colossus" not in {v.name for v in r.by_voice.values()}
    assert r.by_voice["GUEST"].name == "Brian Chesky"


def test_guest_name_never_painted_on_host() -> None:
    # Regression for the headline bug: with no host name available, the host voice keeps its
    # raw label and the guest name lands on the guest voice — never the host.
    diar = _diar([("HOST", 0, 80), ("GUEST", 80, 400)], 2)
    r = resolve_speaker_roster(diar, "Let's get into it.", detected_guests=["Brian Chesky"])
    assert r.by_voice["HOST"].name == "HOST"  # raw, not "Brian Chesky"
    assert r.by_voice["GUEST"].name == "Brian Chesky"


def test_three_candidate_names_and_one_voice_names_NOBODY() -> None:
    """ADR-110 — CHANGED BEHAVIOUR, deliberately.

    This used to assign the FIRST detected name, because the list happened to be in that order.
    Show notes name the people an episode is ABOUT alongside the people in the room, so "the first
    one" is a coin toss between a guest and a lawsuit defendant. With three candidates and one
    voice there is a choice, and a choice made without evidence is a guess.

    A `SPEAKER_01` costs us an unnamed voice. A wrong name puts words in a real person's mouth.
    """
    diar = _diar([("HOST", 0, 60), ("GUEST", 60, 300)], 2)
    r = resolve_speaker_roster(
        diar,
        "I'm Patrick O'Shaughnessy.",
        detected_guests=["Brian Chesky", "Unused Person", "Also Unused"],
    )
    assert not r.by_voice["GUEST"].named
    assert {v.name for v in r.by_voice.values() if v.named} == {"Patrick O'Shaughnessy"}


def test_empty_diarization_returns_empty_roster() -> None:
    r = resolve_speaker_roster(_diar([], 0), "I'm Patrick O'Shaughnessy.")
    assert r.by_voice == {} and r.num_speakers == 0


def test_guest_named_from_own_self_intro_when_not_detected() -> None:
    # #876 partial-naming: a guest whose voice self-introduces ("Hi, I'm Nic Harrigan") is named
    # from its OWN turns even when it is NOT in the detected-guest list — previously stayed raw.
    diar = _diar([("HOST", 0, 60), ("SPEAKER_01", 60, 400), ("HOST", 400, 420)], 2)
    r = resolve_speaker_roster(
        diar,
        "Welcome. I'm Noah Kravitz.",
        detected_guests=[],  # guest NOT detected upstream
        voice_texts={
            "HOST": "Welcome to the show. I'm Noah Kravitz and today we go deep.",
            "SPEAKER_01": "Thanks for having me. Hi, I'm Nic Harrigan and I work on quantum.",
        },
    )
    assert r.by_voice["HOST"].name == "Noah Kravitz"
    guest = r.by_voice["SPEAKER_01"]
    assert guest.name == "Nic Harrigan"  # named from its own self-introduction
    assert guest.role == "guest" and guest.source == "self_intro"


def test_own_self_intro_ignored_without_voice_texts() -> None:
    # Backward-compat: with no voice_texts, an undetected guest still stays raw (old behaviour).
    diar = _diar([("HOST", 0, 60), ("SPEAKER_01", 60, 400)], 2)
    r = resolve_speaker_roster(diar, "Welcome. I'm Noah Kravitz.", detected_guests=[])
    assert r.by_voice["SPEAKER_01"].named is False


def test_speaker_diagnostics_explains_what_tried_and_why_unresolved() -> None:
    diar = _diar([("HOST", 0, 60), ("SPEAKER_01", 60, 400)], 2)
    voice_texts = {"HOST": "Welcome. I'm Noah Kravitz.", "SPEAKER_01": "No introduction here."}
    r = resolve_speaker_roster(
        diar, "Welcome. I'm Noah Kravitz.", detected_guests=[], voice_texts=voice_texts
    )
    diag = build_speaker_diagnostics(
        diar,
        r,
        transcript_text="Welcome. I'm Noah Kravitz.",
        voice_texts=voice_texts,
        detected_guests=[],
        known_hosts=[],
    )
    assert diag["summary"] == {
        "num_speakers": 2,
        "named": 1,
        "unresolved": 1,
        "by_voice_type": {"person": 1, "unidentified": 1},
        "show_centric": False,
        "expected_unresolved": 1,
        # SPEAKER_01 is substantive, and NOBODY NAMES THEM — that is tape, not a failure.
        "truly_unknown": 0,
        # ...but 85% of the episode is still attributable to nobody, and THAT is worth an alarm
        # even when it is not our fault. `unbound_names` is empty: no metadata name went unplaced,
        # so there is nobody to go and find.
        "unattributed_talk_share": 0.85,
        "unattributed_alarm": True,
        "unbound_names": [],
    }
    assert diag["tried"]["host_self_intro"] == "Noah Kravitz"
    by_voice = {v["voice"]: v for v in diag["voices"]}
    assert by_voice["HOST"]["named"] is True and by_voice["HOST"]["source"] == "self_intro"
    assert by_voice["HOST"]["voice_type"] == "person"
    assert by_voice["SPEAKER_01"]["named"] is False
    assert (
        by_voice["SPEAKER_01"]["voice_type"] == "unidentified"
    )  # substantive, but nobody names them
    # NOBODY names SPEAKER_01, so there was nothing to fail at: `expected`, not a miss. That
    # distinction is what keeps `truly_unknown` meaningful as a defect count.
    assert by_voice["SPEAKER_01"]["expected"] is True
    assert by_voice["SPEAKER_01"]["reason"]  # a non-empty "why it failed" explanation


def test_speaker_diagnostics_show_centric_host_is_expected() -> None:
    # On a show-centric feed an unnamed host is the EXPECTED outcome, not a miss.
    diar = _diar([("HOST", 0, 60), ("GUEST", 60, 400)], 2)
    r = resolve_speaker_roster(diar, "Welcome back.", detected_guests=[])  # host unnamed
    diag = build_speaker_diagnostics(diar, r, transcript_text="Welcome back.", show_centric=True)
    by_voice = {v["voice"]: v for v in diag["voices"]}
    assert r.by_voice["HOST"].role == "host" and r.by_voice["HOST"].named is False
    assert by_voice["HOST"]["expected"] is True
    assert "show-centric" in by_voice["HOST"]["reason"]
    # Nobody names the guest either — that is tape, not a miss.
    assert by_voice["GUEST"]["expected"] is True
    assert diag["summary"]["show_centric"] is True
    # BOTH are expected: the show-centric host (renders "Host") and the guest nobody names.
    # `truly_unknown` is now the honest "we should have named this and did not" residual.
    assert diag["summary"]["expected_unresolved"] == 2
    assert diag["summary"]["truly_unknown"] == 0


def test_voice_type_cameo_commercial_and_unknown() -> None:
    # HOST named; SPEAKER_01 is a long unnamed voice that NOBODY NAMES -> "unidentified" (the
    # tape / vox-pop of a narrated piece); SPEAKER_02 a brief cameo (<20s); SPEAKER_03 speaks only
    # inside an ad region (commercial).
    diar = _diar(
        [
            ("HOST", 0, 60),
            ("SPEAKER_01", 60, 400),  # 340s, and nobody names them -> unidentified
            ("SPEAKER_02", 400, 408),  # 8s -> cameo
            ("SPEAKER_03", 500, 560),  # 60s but all inside the ad region -> commercial
        ],
        4,
    )
    r = resolve_speaker_roster(
        diar,
        "Welcome. I'm Noah Kravitz.",
        ad_intervals=[(495.0, 570.0)],
    )
    assert r.by_voice["HOST"].voice_type == "person"
    assert r.by_voice["SPEAKER_01"].voice_type == "unidentified"
    assert r.by_voice["SPEAKER_02"].voice_type == "cameo"
    assert r.by_voice["SPEAKER_03"].voice_type == "commercial"
    # Friendly display labels for the non-person voices (id-bearing label stays raw).
    assert r.display_label_for("SPEAKER_02") == "Brief speaker"
    assert r.display_label_for("SPEAKER_03") == "Advertisement"
    assert r.display_label_for("SPEAKER_01") == "Unidentified speaker"
    assert r.label_for("SPEAKER_01") == "SPEAKER_01"  # id-bearing label never swapped
    assert r.label_for("SPEAKER_02") == "SPEAKER_02"


def test_a_voice_we_FAILED_to_name_keeps_the_raw_id_as_a_defect_marker() -> None:
    """The distinction the corpus audit made possible.

    "Nobody named them" and "we failed to name them" are not the same thing, and until the audit
    existed we could not tell them apart — so both rendered as a bare SPEAKER_07.

    Here a guest name is DECLARED and goes unclaimed. A name existed; we did not attach it. That
    voice keeps its raw id, because the raw id is the defect marker: it means "we should have named
    this and did not". Showing that marker on a voice nobody could have named turns a signal into
    noise, and a signal nobody trusts stops being a signal.
    """
    diar = _diar(
        [
            ("HOST", 0, 60),
            ("SPEAKER_01", 60, 400),
            ("SPEAKER_02", 400, 800),
        ],
        3,
    )
    r = resolve_speaker_roster(
        diar,
        "Welcome. I'm Noah Kravitz.",
        known_hosts=["Noah Kravitz"],
        detected_guests=["Ada Lovelace", "Alan Turing"],  # two names, and a voice left over
    )
    leftover = [v for v, role in r.by_voice.items() if not role.named]
    for v in leftover:
        assert r.by_voice[v].voice_type != "unidentified", (
            f"{v} was typed 'unidentified', but a declared guest name was still going spare — "
            "we FAILED to name it, and the raw id has to say so"
        )
        assert r.display_label_for(v) == v


def test_unnamed_host_displays_as_host() -> None:
    # A show-centric feed never names the host; the intro-dominant unnamed voice is role=host and
    # renders as "Host" (not SPEAKER_00), while its id-bearing label stays raw (#1056 / Step C).
    diar = _diar([("HOST", 0, 60), ("GUEST", 60, 400)], 2)
    r = resolve_speaker_roster(diar, "Welcome back to the show.", detected_guests=[])
    host = r.by_voice["HOST"]
    assert host.role == "host" and host.named is False
    assert r.display_label_for("HOST") == "Host"
    assert r.label_for("HOST") == "HOST"  # id-bearing label untouched


def test_voice_type_commercial_needs_ad_intervals() -> None:
    # Without ad_intervals the same in-ad voice is only cameo/unknown (no commercial guess).
    diar = _diar([("HOST", 0, 60), ("SPEAKER_03", 500, 560)], 2)
    r = resolve_speaker_roster(diar, "I'm Noah Kravitz.")
    # 60s, no ad info, and nobody names them -> unidentified (tape), not a defect
    assert r.by_voice["SPEAKER_03"].voice_type == "unidentified"


def test_self_intro_single_name_accepted_on_own_turns() -> None:
    """A voice that says 'I'm Brandon' in its own turns IS Brandon (no anchor needed)."""
    from podcast_scraper.providers.ml.diarization.roster import _self_intros_by_voice

    vt = {"SPEAKER_00": "Welcome. I'm Brandon. I develop RNA therapeutics and love it."}
    assert _self_intros_by_voice(vt) == {"SPEAKER_00": "Brandon"}


def test_self_intro_rejects_nationality_mononym() -> None:
    """'I'm American' must NOT name a voice (the guard the single-name path preserves)."""
    from podcast_scraper.providers.ml.diarization.roster import _self_intros_by_voice

    vt = {"SPEAKER_00": "I'm American and I think this is a great question, honestly."}
    assert _self_intros_by_voice(vt) == {}


# --- host GREETS the guest by name: "Kara Swisher, welcome back" (#1226 follow-up) -----------
# The deterministic introduction reader only read the cue-FIRST form ("joined by X"); a host who
# greets a just-arrived guest name-first ("Jody Rosen, welcome to the show") named nobody. Two
# detected guests + one guest voice defeats the forced-single-name path, so the greeting is the
# ONLY signal that can name the voice.


def test_guest_named_by_host_greeting_name_first() -> None:
    # Clean case: the greeting sits on the HOST's own turn. The host-gated intro reader names the
    # voice that speaks NEXT (the greeted guest).
    diar = _diar([("HOST", 0, 20), ("HOST", 20, 40), ("GUEST", 40, 340), ("HOST", 340, 360)], 2)
    r = resolve_speaker_roster(
        diar,
        "Welcome back. I'm Patrick O'Shaughnessy.",
        known_hosts=["Patrick O'Shaughnessy"],
        detected_guests=["Kara Swisher", "Andrew Yang"],  # 2 names -> no forced-single naming
        voice_texts={
            "HOST": "Welcome back. I'm Patrick O'Shaughnessy. Kara Swisher, welcome to the show.",
            "GUEST": "Thanks, it is great to be here. My new project is about longevity.",
        },
        ordered_turns=[
            ("HOST", "Welcome back. I'm Patrick O'Shaughnessy."),
            ("HOST", "Kara Swisher, welcome to the show."),
            ("GUEST", "Thanks, it is great to be here. My new project is about longevity."),
            ("HOST", "Tell us all about it."),
        ],
    )
    assert r.by_voice["HOST"].name == "Patrick O'Shaughnessy"
    assert r.by_voice["GUEST"].name == "Kara Swisher"
    assert r.by_voice["GUEST"].role == "guest"


def test_contaminated_greeting_reclaimed_off_guest_cluster() -> None:
    # Contamination (the v2.2 / community-1 failure): the host's greeting "Kara Swisher, welcome
    # back" was mis-merged into the GUEST's own voice cluster. Un-fixed, the greeting reader would
    # see the GUEST introducing "Kara" and name whoever speaks next -> the HOST, painting the host
    # with the guest's name. The reclamation moves the name-anchored greeting back to the host, and
    # the host-gated reader then names the guest voice correctly. The HOST must never become Kara.
    diar = _diar([("HOST", 0, 20), ("GUEST", 20, 40), ("GUEST", 40, 340), ("HOST", 340, 360)], 2)
    r = resolve_speaker_roster(
        diar,
        "Welcome back. I'm Patrick O'Shaughnessy.",
        known_hosts=["Patrick O'Shaughnessy"],
        detected_guests=["Kara Swisher", "Andrew Yang"],
        voice_texts={
            "HOST": "Welcome back. I'm Patrick O'Shaughnessy. Big news today.",
            "GUEST": (
                "Kara Swisher, welcome back, we are delighted to have you. "
                "Thanks, it is great to be here. My new project is about longevity."
            ),
        },
        ordered_turns=[
            ("HOST", "Welcome back. I'm Patrick O'Shaughnessy. Big news today."),
            ("GUEST", "Kara Swisher, welcome back, we are delighted to have you."),
            ("GUEST", "Thanks, it is great to be here. My new project is about longevity."),
            ("HOST", "Tell us all about it."),
        ],
    )
    # The host is NEVER painted with the guest's name (the safety invariant).
    assert r.by_voice["HOST"].name == "Patrick O'Shaughnessy"
    # The guest voice is recovered by the reclaimed greeting.
    assert r.by_voice["GUEST"].name == "Kara Swisher"


@pytest.mark.parametrize("opener", ["But", "Well", "Anyway", "So", "Now"])
def test_greeted_names_reject_a_swept_up_discourse_opener(opener) -> None:
    """R1/#876 fu on the intro-reader path: ``_greeted_names`` fed the introduction reader a

    2-word "name" whenever the ASR capitalised a sentence-opener before a real name ("So Nick,
    welcome"). It shares the greeting regexes with ``guests_introduced_by_the_host`` and now shares
    its ordinary-word guard, so the opener class yields NO name here either. Parametrised over
    openers beyond the incident strings to pin the class, with a positive control.
    """
    from podcast_scraper.providers.ml.diarization.roster import _greeted_names

    assert _greeted_names(f"{opener} Nick, welcome to the show.") == []
    # positive control: a real two-word greeted name still comes through.
    assert _greeted_names("Jody Rosen, welcome to the show.") == ["Jody Rosen"]


def test_guest_with_asr_close_name_does_not_steal_the_host_identity() -> None:
    # N1: a guest self-introducing with a name ASR-close to a configured host's ("Kevin Ross" vs
    # "Kevin Roose", edit distance 2) must NOT be snapped onto the host. Applied to every voice,
    # canonicalization published the guest AS the host and demoted the real host to guest. The fix
    # gates canonicalization to the host-candidate voices (the first len(known_hosts) to speak).
    diar = _diar([("HOST", 0, 30), ("GUEST", 30, 380), ("HOST", 380, 400)], 2)
    r = resolve_speaker_roster(
        diar,
        "Welcome back to the show.",
        known_hosts=["Kevin Roose"],
        voice_texts={
            "HOST": "Welcome back to the show. We have a fantastic guest lined up for you today.",
            "GUEST": "So I'm Kevin Ross and I build developer tools for a living, a decade now.",
        },
        ordered_turns=[
            ("HOST", "Welcome back to the show. We have a fantastic guest lined up for you today."),
            ("GUEST", "So I'm Kevin Ross and I build developer tools for a living, a decade now."),
        ],
    )
    assert r.by_voice["GUEST"].name == "Kevin Ross"  # keeps its OWN name
    assert r.by_voice["GUEST"].role == "guest"
    assert r.by_voice["HOST"].role == "host"  # the real host is not stolen
    assert r.by_voice["HOST"].name == "Kevin Roose"


def test_asr_mangled_co_host_still_canonicalizes() -> None:
    # The N1 fix must NOT cost a co-host its correct spelling: a second host that opens the show
    # (within the first len(known_hosts) voices) still gets its ASR-mangled self-intro snapped.
    diar = _diar([("HOST1", 0, 30), ("HOST2", 30, 60), ("HOST1", 60, 300), ("HOST2", 300, 400)], 2)
    r = resolve_speaker_roster(
        diar,
        "Welcome to the show.",
        known_hosts=["Kevin Roose", "Casey Newton"],
        voice_texts={
            "HOST1": "Welcome to the show. I'm Kevin Russo.",
            "HOST2": "And I'm Casey Noon. Big show for everyone today, lots to get through.",
        },
        ordered_turns=[
            ("HOST1", "Welcome to the show. I'm Kevin Russo."),
            ("HOST2", "And I'm Casey Noon. Big show for everyone today, lots to get through."),
        ],
    )
    assert r.by_voice["HOST1"].name == "Kevin Roose"
    assert r.by_voice["HOST2"].name == "Casey Newton"


def test_short_montage_is_suppressed_but_a_long_dominant_voice_is_kept() -> None:
    # 1a/#1330: a SHORT cold-open montage merges several hosts' garbled self-intros into one 13s
    # cluster ("I'm Kevin Russo… I'm Casey Noon…" on Hard Fork) — not a person, suppress it. But a
    # LONG dominant voice with the SAME double self-intro is the real host whose cluster absorbed a
    # merged cold-open clip (the real Kevin Roose measured 1500s) — keep it, named from its own
    # leading self-intro. Talk time is what tells the clip from the speaker. Synthetic names.
    from podcast_scraper.providers.ml.diarization.roster import _self_intro_voice_names

    montage = "I'm Ada Brightwell, tech columnist. I'm Ben Coalcrest from Platformer."
    diar = _diar([("MONT", 0, 13), ("HOST", 13, 900), ("HOST", 950, 1500)], 2)
    voice_texts = {
        "MONT": montage,  # 13s clip: two self-intros -> montage, suppressed
        "HOST": "I'm Ada Brightwell. " + montage,  # 1437s dominant voice absorbed the same clip
    }
    out = _self_intro_voice_names(diar, voice_texts, [], known_hosts=[], ad_voices=set())
    assert "MONT" not in out  # the short montage clip is suppressed
    assert out.get("HOST") == "Ada Brightwell"  # the long dominant voice is kept (first self-intro)


def test_detected_guest_is_not_forced_when_its_surname_is_already_on_the_roster() -> None:
    # 2a/#876: an interviewer names the guest under a title ("Professor Fenwick"); the metadata
    # lists the SAME person as "Alan Fenwick". The forced one-name-one-voice path treated "Alan
    # Fenwick" as still-unclaimed and painted it onto a leftover BUMPER voice ("We'll be right
    # back"), fabricating a second Fenwick (The Daily ep 0002 did this with "Robert Pape"). A
    # detected-guest name whose SURNAME already appears on the roster is not spare.
    diar = _diar([("HOST", 0, 40), ("GUEST", 40, 600), ("BUMPER", 600, 631)], 3)
    guest = (
        "Thank you for having me. I'm Professor Fenwick and I study political violence at length."
    )
    r = resolve_speaker_roster(
        diar,
        "Welcome. I'm Dana Reyes.",
        known_hosts=["Dana Reyes"],
        detected_guests=["Alan Fenwick"],
        metadata_named=["Alan Fenwick"],
        voice_texts={
            "HOST": "Welcome. I'm Dana Reyes. Professor Fenwick, thanks so much for coming on.",
            "GUEST": guest,
            "BUMPER": "We'll be right back.",
        },
        ordered_turns=[
            ("HOST", "Welcome. I'm Dana Reyes. Professor Fenwick, thanks so much for coming on."),
            ("GUEST", guest),
            ("BUMPER", "We'll be right back."),
        ],
    )
    names = [role.name for role in r.by_voice.values()]
    assert "Alan Fenwick" not in names  # not force-fabricated onto the bumper
    assert r.by_voice["BUMPER"].name == "BUMPER"  # the bumper stays unnamed (safe direction)


def test_a_quoted_greeting_by_a_non_host_never_force_names_a_voice() -> None:
    # N2: guests_introduced_by_the_host must trust only the HOST's turns. A non-host voice that
    # QUOTES a greeting ("...and then Sarah Chen, thanks so much for coming to my defense...") must
    # not harvest that name into the guest pool, where the forced one-name-one-voice match would
    # then paint "Sarah Chen" onto that unrelated voice.
    diar = _diar([("HOST", 0, 30), ("SPK", 30, 380), ("HOST", 380, 400)], 2)
    r = resolve_speaker_roster(
        diar,
        "Welcome to the show.",
        known_hosts=["Alex Rivera"],
        voice_texts={
            "HOST": "Welcome to the show. I'm Alex Rivera and today we get into a wild legal saga.",
            "SPK": (
                "So the trial was chaos. And then Sarah Chen, thanks so much for coming to my "
                "defense, she stood up and we ended up winning the whole thing that afternoon."
            ),
        },
        ordered_turns=[
            (
                "HOST",
                "Welcome to the show. I'm Alex Rivera and today we get into a wild legal saga.",
            ),
            ("SPK", "And then Sarah Chen, thanks so much for coming to my defense, we won it all."),
        ],
    )
    assert r.by_voice["SPK"].name != "Sarah Chen"
    assert r.by_voice["SPK"].named is False


def test_a_cold_open_guest_opener_does_not_name_the_host_voice() -> None:
    # N5: a cold-open GUEST clip that speaks first, performs the guest role ("thanks for having
    # me"), and utters a name-first phrase ("Jane Doe is with us") must NOT be trusted as a host
    # hint — otherwise the intro reader names the NEXT (real host) voice from that phrase. With no
    # known_hosts there is no host_pool to rescue the misname, so this is where it bites.
    guest = "Thanks for having me. Honestly Jane Doe is with us, thrilled to be here."
    host = "Right, let us get straight into the biggest tech stories this week everybody."
    diar = _diar([("GUEST", 0, 40), ("HOST", 40, 360), ("GUEST", 360, 400)], 2)
    r = resolve_speaker_roster(
        diar,
        "podcast",
        voice_texts={"GUEST": guest, "HOST": host},
        ordered_turns=[("GUEST", guest), ("HOST", host)],
    )
    assert "Jane Doe" not in (r.by_voice["HOST"].name or "")

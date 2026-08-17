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
        "voice_census": {
            "person": {"count": 1, "talk_s": 60.0, "talk_share": 0.15},
            "unidentified": {"count": 1, "talk_s": 340.0, "talk_share": 0.85},
        },
        # Labeling OUTPUT (ADR-135/#1220): both voices are real (no cameo/commercial), so both are
        # exposed to GI/KG — HOST named, SPEAKER_01 an unidentified Voice.
        "exposed": {
            "speakers": 2,
            "named": 1,
            "voices": 1,
            "voices_unknown": 0,
            "voices_unidentified": 1,
        },
        "show_centric": False,
        "expected_unresolved": 1,
        # SPEAKER_01 is substantive, and NOBODY NAMES THEM — that is tape, not a failure.
        "truly_unknown": 0,
        # 85% of the episode is attributed to nobody — recorded as a trace
        # (`unattributed_talk_share`) so the sidecar carries the full picture — but it is all
        # `unidentified` TAPE, nobody we could have named, so the DEFECT share is 0 and the alarm
        # does NOT fire (ADR-139 / Pattern B). A
        # vox-pop nobody introduces is not our failure. `unbound_names` is empty: nobody to go find.
        "unattributed_talk_share": 0.85,
        "unattributed_defect_share": 0.0,
        # None = the caller did not say whether speaker detection ran (#1647). The alarm's
        # extra trip needs `False` specifically, so an unstated caller keeps the old behaviour
        # instead of retroactively alarming on every episode built before the ledger existed.
        "detection_stage_ran": None,
        "unattributed_alarm": False,
        "labeling_profile": "naming-4",
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


def test_alarm_fires_when_nothing_was_named_and_detection_never_ran() -> None:
    """The #1646 case: every voice unidentified because the stage was SKIPPED (#1647).

    ADR-139 deliberately excludes ``unidentified`` talk from the defect share — "nobody in the
    episode says who they are" is not our failure. That reasoning holds only if detection
    actually looked. When the stage is skipped, every voice degrades to ``unidentified``, the
    defect share collapses to 0.0, and the share-based alarm reads False on an episode that
    lost 100 % of its insights. That is exactly what happened across 72 % of the corpus.

    The real Latent Space episode this reproduces: 4 voices, 0 named, 29 insights all
    unsurfaceable, and ``unattributed_alarm: false``.
    """
    diar = _diar([("SPEAKER_00", 0, 200), ("SPEAKER_01", 200, 600)], 2)
    voice_texts = {"SPEAKER_00": "No introduction.", "SPEAKER_01": "Also none."}
    r = resolve_speaker_roster(
        diar, "A transcript with no self-intros.", detected_guests=[], voice_texts=voice_texts
    )

    skipped = build_speaker_diagnostics(
        diar, r, voice_texts=voice_texts, detected_guests=[], known_hosts=[], detection_ran=False
    )
    assert skipped["summary"]["named"] == 0
    assert skipped["summary"]["detection_stage_ran"] is False
    # The defect share is still 0.0 — the point is that the alarm no longer depends on it alone.
    assert skipped["summary"]["unattributed_defect_share"] == 0.0
    assert skipped["summary"]["unattributed_alarm"] is True

    # Same episode, but detection DID run and genuinely found nobody: unchanged, no alarm.
    # A narrated desk (Planet Money) must not start alarming for doing nothing wrong.
    looked = build_speaker_diagnostics(
        diar, r, voice_texts=voice_texts, detected_guests=[], known_hosts=[], detection_ran=True
    )
    assert looked["summary"]["detection_stage_ran"] is True
    assert looked["summary"]["unattributed_alarm"] is False

    # And an unstated caller keeps the pre-ledger behaviour rather than alarming retroactively.
    unstated = build_speaker_diagnostics(
        diar, r, voice_texts=voice_texts, detected_guests=[], known_hosts=[]
    )
    assert unstated["summary"]["detection_stage_ran"] is None
    assert unstated["summary"]["unattributed_alarm"] is False


def test_pattern_b_bounds_defect_to_spare_name_count() -> None:
    # ADR-139 / Pattern B: 2 unbound metadata names can explain at most 2 missed voices. The 2
    # most-substantive unnamed voices are `unknown` (defect); the rest are `unidentified` TAPE, so a
    # narrated desk's random inserts stop reading as "we should have named this".
    diar = _diar(
        [("HOST", 0, 60), ("A", 60, 360), ("B", 360, 560), ("C", 560, 600), ("D", 600, 630)], 5
    )
    voice_texts = {
        "HOST": "Welcome. I'm Noah Kravitz.",
        "A": "a long substantive stretch of discussion about the topic at length here",
        "B": "more substantive discussion continuing on for a good while as well",
        "C": "a brief interjection from the field",
        "D": "another short clip of tape",
    }
    r = resolve_speaker_roster(
        diar,
        "Welcome. I'm Noah Kravitz.",
        detected_guests=[],
        voice_texts=voice_texts,
        metadata_named=["Alice Anderson", "Bob Brown"],  # 2 spare, unbindable to these voices
    )
    vt = {v: role.voice_type for v, role in r.by_voice.items()}
    # top-2 by talk are the defects worth chasing...
    assert vt["A"] == "unknown" and vt["B"] == "unknown"
    # ...the excess beyond the 2 spare names is tape, not our failure.
    assert vt["C"] == "unidentified" and vt["D"] == "unidentified"


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


def test_exposed_output_excludes_cameo_and_commercial_noise_1220() -> None:
    """ADR-135/#1220: the labeling OUTPUT surface is what reaches GI/KG after cleanup.

    Raw diarization here has FOUR voices, but two are noise (cameo + commercial) that never
    become graph nodes. ``summary.exposed`` reports only the two real speakers — HOST (named
    Person) and SPEAKER_01 (an unidentified Voice) — so the sidecar states the clean
    named-vs-Voice rate on its own, without opening the graph.
    """
    diar = _diar(
        [
            ("HOST", 0, 60),
            ("SPEAKER_01", 60, 400),  # unidentified — real, unnamed -> Voice
            ("SPEAKER_02", 400, 408),  # cameo -> dropped
            ("SPEAKER_03", 500, 560),  # commercial -> dropped
        ],
        4,
    )
    r = resolve_speaker_roster(diar, "Welcome. I'm Noah Kravitz.", ad_intervals=[(495.0, 570.0)])
    diag = build_speaker_diagnostics(
        diar, r, transcript_text="Welcome. I'm Noah Kravitz.", detected_guests=[], known_hosts=[]
    )
    s = diag["summary"]
    # INPUT still counts all four raw voices...
    assert s["num_speakers"] == 4
    assert s["by_voice_type"] == {
        "person": 1,
        "unidentified": 1,
        "cameo": 1,
        "commercial": 1,
    }
    # ...but the OUTPUT exposed to GI/KG is only the two real speakers.
    assert s["exposed"] == {
        "speakers": 2,
        "named": 1,
        "voices": 1,
        "voices_unknown": 0,
        "voices_unidentified": 1,
    }


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


def test_two_distinct_guests_sharing_a_surname_are_both_nameable() -> None:
    # 2a negative control: ONLY a honorific-form roster name ("Professor Pape") claims its surname.
    # Two distinct guests who merely share a surname — "Robert Pape" (named by his own self-intro)
    # and a genuinely different "Karen Pape" — must both be nameable; the shared surname must not
    # suppress the second (the reviewer's HIGH finding on an over-broad surname claim).
    diar = _diar([("HOST", 0, 40), ("G1", 40, 400), ("G2", 400, 760)], 3)
    g1 = "Thank you. I'm Robert Pape and I study political violence for a living, a whole career."
    r = resolve_speaker_roster(
        diar,
        "Welcome. I'm Dana Reyes.",
        known_hosts=["Dana Reyes"],
        detected_guests=["Robert Pape", "Karen Pape"],
        metadata_named=["Robert Pape", "Karen Pape"],
        voice_texts={
            "HOST": "Welcome. I'm Dana Reyes.",
            "G1": g1,
            "G2": "Thanks for having me. I work on domestic policy and social movements at length.",
        },
        ordered_turns=[("HOST", "Welcome. I'm Dana Reyes."), ("G1", g1), ("G2", "Thanks. Policy.")],
    )
    names = {role.name for role in r.by_voice.values()}
    assert "Robert Pape" in names  # named by his own self-intro
    assert "Karen Pape" in names  # a distinct same-surname guest is NOT suppressed


def test_surname_token_edge_cases() -> None:
    from podcast_scraper.providers.ml.diarization.roster import _surname_token

    assert _surname_token("Robert Pape") == "pape"
    assert _surname_token("Professor Pape") == "pape"
    assert _surname_token("Robert Pape Jr.") == "pape"  # generational suffix dropped
    assert _surname_token("Li Xu") == "xu"  # short romanised surname kept (>=2 chars)
    assert _surname_token("R. Pape") == "pape"  # a leading initial is not the surname
    assert _surname_token("Cher") is None  # mononym has no surname
    assert _surname_token("") is None


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


def test_asr_mangled_guest_snaps_to_metadata_stated_name() -> None:
    # ADR-130 (IMPLEMENT 1): the guest snap, symmetric to the host snap. Turbo rendered the guest
    # "David Duvenaud" as "David Duvino" in his self-introduction; the episode metadata states the
    # correct spelling (title/description -> metadata_named). The provider-agnostic final pass snaps
    # the mangled published name to the stated guest by the same fuzzy rule the host path uses.
    diar = _diar([("HOST", 0, 30), ("GUEST", 30, 380), ("HOST", 380, 400)], 2)
    r = resolve_speaker_roster(
        diar,
        "Welcome. I'm Kevin Roose.",
        known_hosts=["Kevin Roose"],
        detected_guests=["David Duvenaud"],
        metadata_named=["David Duvenaud"],
        voice_texts={
            "HOST": "Welcome. I'm Kevin Roose and today we talk AI safety for the whole hour.",
            "GUEST": "Thanks for having me. So I'm David Duvino and I research AI alignment daily.",
        },
        ordered_turns=[
            ("HOST", "Welcome. I'm Kevin Roose and today we talk AI safety for the whole hour."),
            (
                "GUEST",
                "Thanks for having me. So I'm David Duvino and I research AI alignment daily.",
            ),
        ],
    )
    assert r.by_voice["GUEST"].name == "David Duvenaud"  # snapped to the stated spelling
    assert r.by_voice["GUEST"].role == "guest"
    assert r.by_voice["HOST"].name == "Kevin Roose"


def test_guest_snap_never_steals_a_name_another_voice_already_holds() -> None:
    # ADR-130 one-name-one-voice: a guest self-introducing "Kevin Ross" (ASR-close to the host
    # "Kevin Roose") must NOT be snapped onto the host's name — the host already holds it. The snap
    # is reference-bounded AND claim-aware, so the guest keeps its own (distinct) name.
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
    assert r.by_voice["HOST"].name == "Kevin Roose"
    assert r.by_voice["GUEST"].name == "Kevin Ross"  # its own name, not stolen onto the host


def test_stated_snap_helpers_are_reference_bounded() -> None:
    # ADR-130 unit: the snap can only ever return a name present in the stated set (never invents),
    # and _recover_stated_names respects one-name-one-voice.
    from dataclasses import replace

    from podcast_scraper.providers.ml.diarization.roster import (
        _canonicalize_to_stated_name,
        _recover_stated_names,
        SpeakerRole,
    )

    # snaps a mangled name to the stated spelling...
    assert _canonicalize_to_stated_name("David Duvino", ["David Duvenaud"]) == "David Duvenaud"
    # ...but a name too far from every stated name is left unchanged (never invented).
    assert _canonicalize_to_stated_name("Zebediah Quux", ["David Duvenaud"]) == "Zebediah Quux"

    by_voice = {
        "V0": SpeakerRole(name="Kevin Roose", role="host", named=True, source="self_intro"),
        "V1": SpeakerRole(name="David Duvino", role="guest", named=True, source="self_intro"),
    }
    _recover_stated_names(by_voice, ["Kevin Roose", "David Duvenaud"])
    assert by_voice["V1"].name == "David Duvenaud"  # guest recovered
    assert by_voice["V0"].name == "Kevin Roose"  # host untouched
    # a name already held by another voice is not reused: snapping V1 onto "Kevin Roose" is refused.
    by_voice2 = {
        "V0": SpeakerRole(name="Kevin Roose", role="host", named=True, source="self_intro"),
        "V1": replace(
            SpeakerRole(name="Kevin Ross", role="guest", named=True, source="self_intro"),
            name="Kevin Ross",
        ),
    }
    _recover_stated_names(by_voice2, ["Kevin Roose"])
    assert by_voice2["V1"].name == "Kevin Ross"


def test_snap_guards_exact_match_and_known_host_role_gate() -> None:
    # ADR-130 hardening (Fable-5 review SEV-4 + SEV-3b).
    from podcast_scraper.providers.ml.diarization.roster import (
        _recover_stated_names,
        SpeakerRole,
    )

    # SEV-4: a name that EXACTLY matches a stated ref is never re-snapped onto an earlier near-ref.
    # "Robert Pape" (exact) must stay put even though "Rob Pape" precedes it in the ref list.
    by_voice = {
        "V0": SpeakerRole(name="Robert Pape", role="guest", named=True, source="self_intro"),
    }
    _recover_stated_names(by_voice, ["Rob Pape", "Robert Pape"])
    assert by_voice["V0"].name == "Robert Pape"

    # SEV-3b: a NON-host voice is never snapped onto a KNOWN-HOST's spelling, even when that host
    # name is unclaimed (host on leave / cross-promo). The guest keeps its own name; the N1 gate
    # restricting host-identity canonicalization to host-candidate voices is preserved by this pass.
    by_voice2 = {
        "V0": SpeakerRole(name="Patrick", role="host", named=True, source="self_intro"),
        "V1": SpeakerRole(name="Kevin Ross", role="guest", named=True, source="self_intro"),
    }
    # "Kevin Roose" is a known host, unclaimed (only Patrick is on the roster).
    _recover_stated_names(
        by_voice2, ["Patrick", "Kevin Roose"], known_hosts=["Patrick", "Kevin Roose"]
    )
    assert by_voice2["V1"].name == "Kevin Ross"  # NOT snapped onto the host's name

    # ...but a genuinely mangled CO-HOST (role already "host") still snaps to the known-host name.
    by_voice3 = {
        "V0": SpeakerRole(name="Casey Noon", role="host", named=True, source="self_intro"),
    }
    _recover_stated_names(by_voice3, ["Casey Newton"], known_hosts=["Casey Newton"])
    assert by_voice3["V0"].name == "Casey Newton"


def test_mononym_snaps_to_a_uniquely_matching_stated_person() -> None:
    # Audit fix 3: a bare first name ("Kevin", from a self-intro that only caught the first name)
    # snaps to the stated person when EXACTLY one reference carries it, and abstains when ambiguous.
    from podcast_scraper.providers.ml.diarization.roster import _canonicalize_to_stated_name

    assert _canonicalize_to_stated_name("Kevin", ["Kevin Roose", "Kara Swisher"]) == "Kevin Roose"
    # two Kevins -> ambiguous -> unchanged (never guess which)
    assert _canonicalize_to_stated_name("Kevin", ["Kevin Roose", "Kevin Systrom"]) == "Kevin"
    # a mononym matching nobody's first name -> unchanged
    assert _canonicalize_to_stated_name("Z6", ["Kevin Roose"]) == "Z6"


def test_relaxed_first_name_snaps_only_with_a_strong_surname() -> None:
    # Audit fix 2a: "Arietta Laika" (first edit 2 vs "Arijeta", surname edit 1 vs "Lajka") snaps —
    # but a near first name with a DIFFERENT surname does not, so a real person is never renamed.
    from podcast_scraper.providers.ml.diarization.roster import _canonicalize_to_stated_name

    assert _canonicalize_to_stated_name("Arietta Laika", ["Arijeta Lajka"]) == "Arijeta Lajka"
    assert _canonicalize_to_stated_name("Sam Alton", ["Sam Bright"]) == "Sam Alton"


def test_over_split_same_person_both_voices_get_canonical_name() -> None:
    # Audit fix 2a: diarization split one guest into two clusters — the dominant one mangled
    # ("Arietta Laika"), the small one correct ("Arijeta Lajka"). Both should carry the stated
    # spelling; a genuinely different person (different role) is NOT merged (see snap-guard test).
    from podcast_scraper.providers.ml.diarization.roster import (
        _recover_stated_names,
        SpeakerRole,
    )

    by_voice = {
        "V0": SpeakerRole(name="Arietta Laika", role="guest", named=True, source="self_intro"),
        "V1": SpeakerRole(name="Arijeta Lajka", role="guest", named=True, source="self_intro"),
    }
    _recover_stated_names(by_voice, ["Arijeta Lajka"])
    assert by_voice["V0"].name == "Arijeta Lajka"
    assert by_voice["V1"].name == "Arijeta Lajka"

    # DIFFERENT role (host holds the name) -> the guest is NOT merged onto it (one name, one voice).
    by_voice2 = {
        "V0": SpeakerRole(name="Arijeta Lajka", role="host", named=True, source="self_intro"),
        "V1": SpeakerRole(name="Arietta Laika", role="guest", named=True, source="self_intro"),
    }
    _recover_stated_names(by_voice2, ["Arijeta Lajka"])
    assert by_voice2["V1"].name == "Arietta Laika"  # not merged across roles


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

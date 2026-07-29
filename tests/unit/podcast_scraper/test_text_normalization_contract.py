"""ADR-137 — text-normalization contract: per-fix isolation + the seam consistency invariant.

Three fixes for narrated-desk / lowercase-turbo speaker naming, each tested in isolation at the
function level, plus the case-invariance invariant: the same content, truecased and lowercased,
produces the same output for the metadata-anchored seams.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.base import (
    DiarizationResult,
    DiarizationSegment,
)
from podcast_scraper.providers.ml.diarization.labeling_profile import (
    get_profile,
    list_profiles,
    NAMING_3_LEGACY,
    NAMING_4,
)
from podcast_scraper.providers.ml.diarization.roster import (
    _canonicalize_to_stated_name,
    _distinct_intros_map_to_multiple_stated,
    _metadata_anchored_self_intro,
    _voice_named_by_the_introduction,
    build_speaker_diagnostics,
    resolve_speaker_roster,
)
from podcast_scraper.text_normalization import (
    first_names_match,
    normalize_for_match,
    normalize_name_for_match,
)

# --- Foundation ----------------------------------------------------------------------------------


def test_normalize_for_match_folds_case_and_whitespace():
    assert normalize_for_match("  I'm  Rich   GELFOND ") == "i'm rich gelfond"
    # idempotent
    once = normalize_for_match("The  DAILY")
    assert normalize_for_match(once) == once


def test_first_names_match_nickname_initial_and_rejects():
    assert first_names_match("rich", "Richard")
    assert first_names_match("Bob", "Robert")
    assert first_names_match("R", "Robert")  # initial
    assert first_names_match("Katie", "Katherine")
    assert not first_names_match("Eric", "Erica")  # different people
    assert not first_names_match("rich", "Michael")


def test_first_names_match_distinct_names_sharing_a_short_form_do_not_match():
    # review: the nickname table must NOT merge two DIFFERENT formal names that merely share an
    # ambiguous short form. Alexander (m) / Alexandra (f) and Jonathan / John are distinct people;
    # each keeps its own short form but they never match EACH OTHER. (Eric/Erica above is a trivial
    # true-negative because neither is in the table — these exercise a real in-table collision.)
    assert not first_names_match("Alexander", "Alexandra")
    assert not first_names_match("Jonathan", "John")
    # each still matches its own shared short form (ambiguity is acceptable there)
    assert first_names_match("Alexander", "Alex")
    assert first_names_match("Alexandra", "Alex")
    assert first_names_match("Jonathan", "Jon")
    assert first_names_match("John", "Jon")
    # spelling variants of the SAME name are correctly kept matching
    assert first_names_match("Stephen", "Steven")
    assert first_names_match("Katherine", "Catherine")
    # the surname path is safe for the ambiguous shared tokens: the FORMAL names never cross-match
    assert not first_names_match("Patrick", "Patricia")
    assert not first_names_match("Edward", "Theodore")


# --- Fix 2: nickname feeds the ADR-128 canonicalizer ---------------------------------------------


def test_fix2_canonicalizer_snaps_nickname_to_stated_name():
    # "Rich Gelfond" (spoken) -> "Richard Gelfond" (stated): nickname first name + exact surname.
    assert _canonicalize_to_stated_name("Rich Gelfond", ["Richard Gelfond"]) == "Richard Gelfond"
    assert _canonicalize_to_stated_name("Bob Iger", ["Robert Iger"]) == "Robert Iger"


def test_fix2_canonicalizer_does_not_rename_a_different_person():
    # Shares a nickname-able first name but a different surname -> never snapped.
    assert _canonicalize_to_stated_name("Rich Barton", ["Richard Gelfond"]) == "Rich Barton"


# --- Fix 3: case-blind, metadata-anchored self-introduction --------------------------------------


def test_fix3_lowercase_self_intro_binds_to_stated_name():
    # turbo ASR (lowercase) + nickname + ASR-mangled surname all handled, anchored to metadata.
    assert (
        _metadata_anchored_self_intro("well i'm rich gelfond, ceo of imax", ["Richard Gelfond"])
        == "Richard Gelfond"
    )
    assert (
        _metadata_anchored_self_intro("my name is eric schmidt and", ["Eric Schmitt"])
        == "Eric Schmitt"  # schmidt≈schmitt (edit 1)
    )


def test_fix3_guards_against_false_binds():
    # No metadata -> never invents.
    assert _metadata_anchored_self_intro("i'm rich gelfond", []) is None
    # "I'm American" must not bind a stated person on a first-name-ish token.
    assert _metadata_anchored_self_intro("i'm american and proud", ["Amelia Anderson"]) is None
    # Right first name, wrong surname -> no bind.
    assert _metadata_anchored_self_intro("i'm rich barton here", ["Richard Gelfond"]) is None


def test_f1_third_person_this_is_never_self_binds():
    # F1 (advisor): "this is <stated>'s <thing>" is a THIRD-PERSON reference, not a self-intro. The
    # possessive folds to an edit-1 surname ("altmans" vs "altman"), so the fuzzy surname would bind
    # the WRONG name onto whoever said it. And "this is <full name>" is how a host INTRODUCES a
    # guest (or a show names itself) — never a self-introduction. #876: no name beats a wrong name.
    # Fix: "this is" is dropped from the match-form self-intro cue (its capitalized sibling
    # extract_self_introduced_host is "I'm"-only; legit capitalized "This is <mononym>" stays on the
    # metadata-vouched _THIS_IS_INTRO path).
    assert (
        _metadata_anchored_self_intro("so this is sam altman's company we use", ["Sam Altman"])
        is None
    )
    assert (
        _metadata_anchored_self_intro("okay everyone this is eric schmidt", ["Eric Schmidt"])
        is None
    )


def test_f1_self_intro_scan_is_head_bounded():
    # F1: every sibling intro scanner bounds to the first 5000 chars; this fallback must too, so a
    # late third-person mention deep in a turn cannot masquerade as an opening self-introduction.
    filler = "and then we talked for quite a while about it. " * 200  # > 5000 chars
    assert _metadata_anchored_self_intro(filler + " i'm rich gelfond", ["Richard Gelfond"]) is None
    # within the head it still recovers the legitimate self-intro (regression guard).
    assert (
        _metadata_anchored_self_intro("i'm rich gelfond, ceo of imax", ["Richard Gelfond"])
        == "Richard Gelfond"
    )


def test_f1_first_person_possessive_never_self_binds():
    # F1 residual (second advisor review): "i'm <stated>'s <role/thing>" is a THIRD-PERSON reference
    # ("i'm sam altman's biggest fan") — the speaker is NOT that person. The possessive "altman's"
    # folds to an edit-1 surname, so it would wrong-bind through the retained i'm/i am/my name is
    # cues. A trailing "'s" token is dropped from surname candidacy.
    assert (
        _metadata_anchored_self_intro("i'm sam altman's biggest fan today", ["Sam Altman"]) is None
    )
    assert (
        _metadata_anchored_self_intro("i am sam altman's former colleague", ["Sam Altman"]) is None
    )
    # a real self-intro (no possessive) still binds; a real apostrophe surname is untouched.
    assert (
        _metadata_anchored_self_intro("i'm rich gelfond here", ["Richard Gelfond"])
        == "Richard Gelfond"
    )
    # s-ending surname possessive is bare-apostrophe ("hastings'") — must also be dropped (3rd
    # advisor); Conan O'Brien (a real apostrophe name) is not a possessive and still binds.
    assert (
        _metadata_anchored_self_intro("i'm reed hastings' successor at netflix", ["Reed Hastings"])
        is None
    )
    assert _metadata_anchored_self_intro("i'm conan o'brien", ["Conan O'Brien"]) == "Conan O'Brien"


def test_f4_exact_surname_wins_over_a_fuzzy_collision():
    # 3rd advisor, finding 4: stated [Chris Smith, Chris Schmidt] + "chris schmidt" must resolve to
    # Schmidt (exact), never Smith (Smith/Schmidt share a soundex) — order-independent two-pass.
    for order in (["Chris Smith", "Chris Schmidt"], ["Chris Schmidt", "Chris Smith"]):
        assert _metadata_anchored_self_intro("i'm chris schmidt", order) == "Chris Schmidt"
        assert _metadata_anchored_self_intro("i'm chris smith", order) == "Chris Smith"


def test_f2_past_tense_recap_binds_only_at_head_from_a_host():
    # 3rd advisor, finding 2: a MID-SHOW recap "we spoke with X" must not misattribute X to the next
    # voice; a head-of-episode cold-open still binds. (alternating turns push the recap past head.)
    recap = [("h", "so what do you think"), ("g", "interesting")] * 12
    recap += [("h", "earlier we spoke with andrew ng about that"), ("c", "agreed")]
    assert (
        _voice_named_by_the_introduction(
            recap, host_hint_voices={"h"}, metadata_named=["Andrew Ng"]
        )
        == {}
    )
    cold = [("h", "today i sat down with andrew ng"), ("g", "thanks")]
    got = _voice_named_by_the_introduction(
        cold, host_hint_voices={"h"}, metadata_named=["Andrew Ng"], corroborated_named=["Andrew Ng"]
    )
    assert got.get("g") == "Andrew Ng"


def test_f3_report_verb_binds_corroborated_not_a_topical_subject():
    # 3rd advisor, finding 3: "sam altman explains it best" is a TOPICAL mention of an episode
    # subject — must not bind. A report verb on a CORROBORATED reporter (detected guest) does bind.
    topical = [("h", "on scaling laws, sam altman explains it best in his blog"), ("g", "right")]
    assert (
        _voice_named_by_the_introduction(
            topical, host_hint_voices={"h"}, metadata_named=["Sam Altman"], corroborated_named=[]
        )
        == {}
    )
    desk = [("h", "farnaz fassihi reports on the fallout"), ("g", "the situation")]
    got = _voice_named_by_the_introduction(
        desk,
        host_hint_voices={"h"},
        metadata_named=["Farnaz Fassihi"],
        corroborated_named=["Farnaz Fassihi"],
    )
    assert got.get("g") == "Farnaz Fassihi"


def test_f2c_recap_inside_a_merged_host_monologue_does_not_bind():
    # 4th advisor, 2c: a host monologue merges into ONE turn, so the turn-index head bound is met
    # trivially (i=0). A late recap inside that long turn — or an early one preceded by a temporal
    # marker — must not misattribute; only a true opening cold-open binds.
    mono = (
        "welcome, we have a great episode planned today. " * 30
        + "last month we spoke with sam altman about agi."
    )
    assert (
        _voice_named_by_the_introduction(
            [("h", mono), ("g", "hi")], host_hint_voices={"h"}, metadata_named=["Sam Altman"]
        )
        == {}
    )
    cold = [("h", "today i sat down with sam altman"), ("g", "thanks")]
    got = _voice_named_by_the_introduction(
        cold,
        host_hint_voices={"h"},
        metadata_named=["Sam Altman"],
        corroborated_named=["Sam Altman"],
    )
    assert got.get("g") == "Sam Altman"


def test_v1_report_verb_host_name_binds_only_a_host_voice():
    # 4th advisor, v1: adding known hosts to the corroborated set let "kevin roose explains in his
    # book" (a topical mention of an ABSENT co-host) paint onto a guest. A host name on the
    # report-verb path binds ONLY a host voice; the legit co-host desk hand-off still works.
    kh = {"kevin roose"}
    absent = [("h", "kevin roose reports from davos this week"), ("g", "yes")]
    assert (
        _voice_named_by_the_introduction(
            absent,
            host_hint_voices={"h"},
            known_hosts_lower=kh,
            metadata_named=["Kevin Roose"],
            corroborated_named=["Kevin Roose"],
        )
        == {}
    )
    handoff = [("h", "kevin roose walks us through it"), ("kev", "so the way this works")]
    got = _voice_named_by_the_introduction(
        handoff,
        host_hint_voices={"h", "kev"},
        conv_hosts={"kev"},
        known_hosts_lower=kh,
        metadata_named=["Kevin Roose"],
        corroborated_named=["Kevin Roose"],
    )
    assert got.get("kev") == "Kevin Roose"


# --- Fix 1: narrated-desk cue vocabulary ---------------------------------------------------------


def test_fix1_my_colleague_cue_binds_next_voice():
    turns = [
        ("HOST", "today, my colleague Claire Cain Miller on how these accounts work"),
        ("GUEST", "so the way these accounts are designed"),
    ]
    assert _voice_named_by_the_introduction(turns).get("GUEST") == "Claire Cain Miller"


def test_fix1_name_first_desk_verbs_bind_next_voice():
    # host-gated name-first verbs: "X explains", "X talks us through", "X reports".
    for cue in ("explains what happened", "talks us through the case", "reports on the fallout"):
        turns = [("HOST", f"Farnaz Fassihi {cue}"), ("GUEST", "the situation on the ground")]
        got = _voice_named_by_the_introduction(turns, host_hint_voices={"HOST"})
        assert got.get("GUEST") == "Farnaz Fassihi", cue


def test_fix1_cue_path_is_case_invariant_with_metadata():
    # ADR-137: lowercase turbo desk hand-off + metadata anchor binds identically to truecased.
    md = ["Claire Cain Miller"]
    lower = [
        ("host", "today, my colleague claire cain miller on how these accounts work"),
        ("guest", "so the way these accounts are designed"),
    ]
    true = [(v, t[0].upper() + t[1:]) for v, t in lower]  # naive truecase of the first char
    got_low = _voice_named_by_the_introduction(lower, metadata_named=md)
    got_true = _voice_named_by_the_introduction(true, metadata_named=md)
    assert got_low.get("guest") == "Claire Cain Miller"
    assert got_low == got_true  # seam invariance


def test_fix1_cue_path_bridges_asr_mangled_surname():
    # metadata "Fassihi", ASR heard "fasihi" (edit 1) -> bridged by the metadata anchor, lowercase.
    # A report-verb tail ("explains") resolves only against CORROBORATED refs (fix 3), so a real
    # desk reporter is passed as corroborated — a bare topical subject would not bind here.
    turns = [("host", "farnaz fasihi explains the situation"), ("guest", "on the ground")]
    got = _voice_named_by_the_introduction(
        turns,
        host_hint_voices={"host"},
        metadata_named=["Farnaz Fassihi"],
        corroborated_named=["Farnaz Fassihi"],
    )
    assert got.get("guest") == "Farnaz Fassihi"


def test_fix1_cue_path_no_metadata_stays_capitalization_only():
    # No metadata -> the case-blind path is inert; lowercase yields nothing (documented exception).
    turns = [("host", "today, my colleague claire cain miller on this"), ("guest", "yes")]
    assert _voice_named_by_the_introduction(turns) == {}


def test_first_name_only_intro_binds_when_unique():
    # flightcast group intro: a bare first name that UNIQUELY matches a stated name binds the guest.
    turns = [("host", "we're here with akshat of moto together"), ("g", "thanks for having me")]
    got = _voice_named_by_the_introduction(
        turns, host_hint_voices={"host"}, metadata_named=["Akshat Bubna"]
    )
    assert got.get("g") == "Akshat Bubna"


def test_first_name_only_intro_declines_when_ambiguous():
    # Two stated people share the first name -> no bind (the uniqueness guard).
    turns = [("host", "we're here with john from the team today"), ("g", "hi there")]
    got = _voice_named_by_the_introduction(
        turns, host_hint_voices={"host"}, metadata_named=["John Smith", "John Doe"]
    )
    assert got == {}


def test_f2_first_name_only_declines_on_contradicting_surname():
    # F2 (advisor): "here with akshat KANAPARTHY" names a DIFFERENT Akshat than stated "Akshat
    # Bubna" — the surname is present and does not match, so it must NOT fall through to a bare
    # first-name bind. (An affiliation like "akshat of moto" carries no surname and still binds.)
    turns = [("host", "we're here with akshat kanaparthy today"), ("g", "great to be here")]
    got = _voice_named_by_the_introduction(
        turns, host_hint_voices={"host"}, metadata_named=["Akshat Bubna"]
    )
    assert got == {}


def test_f2_first_name_only_declines_on_two_letter_surname():
    # F2 residual (second advisor review): "here with andrew NG" names Andrew Ng — NOT stated
    # "Andrew Chen". A genuine 2-letter surname (Ng/Wu/Li/Xu) must still count as a contradicting
    # surname so it abstains, rather than falling through to a bare-first-name bind.
    turns = [("host", "we're here with andrew ng today"), ("g", "great to be here")]
    got = _voice_named_by_the_introduction(
        turns, host_hint_voices={"host"}, metadata_named=["Andrew Chen"]
    )
    assert got == {}


def test_f2_first_name_only_declines_from_non_host_turn():
    # F2: the bare-first-name relaxation is trusted only from a HOST introducer's turn. The same cue
    # from a non-host voice ("…with rich investors") must not paint a stated Richard onto the next
    # speaker. Here the introducing voice is not in host_hint_voices, so no bind.
    turns = [("guest", "we're here with akshat of moto"), ("x", "hello")]
    got = _voice_named_by_the_introduction(
        turns, host_hint_voices={"host"}, metadata_named=["Akshat Bubna"]
    )
    assert got == {}


def test_first_name_only_never_applies_to_self_intro():
    # First-name-only is cue-path ONLY: a colloquial "i am rich" self-intro must not bind "Richard".
    assert (
        _metadata_anchored_self_intro("well i am rich and successful today", ["Richard Gelfond"])
        is None
    )


def test_merged_cluster_of_multiple_stated_speakers_is_detected():
    # flightcast: two guests' self-intros merged into one cluster ("I'm Lucas and I'm Axel") -> the
    # cluster maps to 2 stated people, so it's a merge (its name gets suppressed, not painted).
    md = ["Lukas Petersson", "Axel Backlund"]  # ASR heard "Lucas" for "Lukas"
    assert (
        _distinct_intros_map_to_multiple_stated("take turns. I'm Lucas and I'm Axel.", md) is True
    )
    # a single self-intro is not a merge
    assert _distinct_intros_map_to_multiple_stated("I'm Lucas here today.", md) is False
    # two self-intros but only ONE maps to a stated person -> not a named-speaker merge
    assert (
        _distinct_intros_map_to_multiple_stated("I'm Lucas and I'm Bob.", ["Lukas Petersson"])
        is False
    )


# --- The seam consistency invariant (ADR-137) ----------------------------------------------------


def test_seam_invariance_self_intro_metadata_anchored():
    stated = ["Richard Gelfond"]
    truecased = "Well, I'm Rich Gelfond, CEO of IMAX."
    lowered = truecased.lower()
    assert (
        _metadata_anchored_self_intro(truecased, stated)
        == _metadata_anchored_self_intro(lowered, stated)
        == "Richard Gelfond"
    )


def test_seam_invariance_canonicalizer():
    stated = ["Richard Gelfond"]
    assert (
        _canonicalize_to_stated_name("Rich Gelfond", stated)
        == _canonicalize_to_stated_name("rich gelfond", stated)
        == "Richard Gelfond"
    )


def test_name_helpers_are_case_and_quote_blind():
    assert normalize_name_for_match("O'Shaughnessy") == normalize_name_for_match("oshaughnessy")
    assert normalize_name_for_match("Gómez-Bombarelli") == "gomez-bombarelli"


# --- Org wart + comma tolerance + "my name is" (today's smaller fixes) --------------------------


def test_org_form_introduced_name_is_not_bound_to_a_voice():
    # The greedy capture picks up "the New York Times" after a cue; an ORG is never the introduced
    # PERSON and must not be painted onto the next (guest) voice (the Daily 0008 wart).
    turns = [("HOST", "here with me is the New York Times"), ("G", "thanks for having me on")]
    got = _voice_named_by_the_introduction(turns, host_hint_voices={"HOST"})
    assert "New York Times" not in got.values()


def test_comma_tolerated_in_name_first_intro():
    # ASR inserts a comma between the name and the verb; the desk hand-off still binds (host-gated).
    turns = [
        ("HOST", "Pentagon reporter Eric Schmitt, talks us through the escalation"),
        ("G", "the situation on the ground is"),
    ]
    got = _voice_named_by_the_introduction(turns, host_hint_voices={"HOST"})
    assert got.get("G") == "Eric Schmitt"


def _diar(segs):
    return DiarizationResult(
        segments=[DiarizationSegment(s, e, v) for v, s, e in segs],
        num_speakers=len({v for v, _, _ in segs}),
    )


def test_my_name_is_self_intro_discovered_without_metadata():
    # A network show host self-introduces "My name is X" and is NOT in the feed metadata — the
    # capitalized discovery path must catch it (not only "I'm X").
    diar = _diar([("HOST", 0, 200), ("G", 200, 400)])
    r = resolve_speaker_roster(
        diar,
        "Hello and welcome to the show. My name is Ana Rodriguez.",
        voice_texts={
            "HOST": "Hello and welcome to the show. My name is Ana Rodriguez.",
            "G": "thanks so much for having me on today",
        },
        metadata_named=[],
        known_hosts=[],
        detected_guests=[],
        llm_voice_names=None,
    )
    assert r.by_voice["HOST"].name == "Ana Rodriguez"


# --- Integration: full labeling flow (diarize -> name -> classify -> census) ---------------------


def test_labeling_integration_full_flow_and_census():
    # One episode exercising: "my name is" host discovery, lowercase nickname guest self-intro
    # (Rich -> Richard, metadata-anchored), and a random tape voice that stays `unidentified` (not a
    # defect). The census + defect alarm must reflect exactly that (ADR-137 / Pattern B).
    diar = _diar([("HOST", 0, 200), ("GUEST", 200, 700), ("TAPE", 700, 760)])
    voice_texts = {
        "HOST": "Hello and welcome to the show. My name is Ana Rodriguez.",
        "GUEST": "well i'm rich gelfond and i have run imax for decades now and here is why",
        "TAPE": "just some random person on the street sharing a quick unrelated thought here",
    }
    r = resolve_speaker_roster(
        diar,
        " ".join(voice_texts.values()),
        voice_texts=voice_texts,
        metadata_named=["Richard Gelfond"],
        known_hosts=[],
        detected_guests=[],
        llm_voice_names=None,
    )
    assert r.by_voice["HOST"].name == "Ana Rodriguez"  # my-name-is discovery
    assert r.by_voice["GUEST"].name == "Richard Gelfond"  # nickname + case-blind self-intro
    assert r.by_voice["TAPE"].voice_type == "unidentified"  # tape, not our failure

    diag = build_speaker_diagnostics(
        diar,
        r,
        transcript_text=" ".join(voice_texts.values()),
        voice_texts=voice_texts,
        detected_guests=[],
        known_hosts=[],
        metadata_named=["Richard Gelfond"],
    )
    s = diag["summary"]
    assert s["named"] == 2
    # only TAPE is unattributed, and it is `unidentified` — so the DEFECT share is 0 and no alarm.
    assert s["unattributed_defect_share"] == 0.0
    assert s["unattributed_alarm"] is False
    # the census carries the full picture with talk-time per type.
    assert s["voice_census"]["person"]["count"] == 2
    assert s["voice_census"]["unidentified"]["count"] == 1


# --- ADR-138: versioned labeling profile (registry + A/B switch) ---------------------------------


def test_labeling_profile_registry():
    assert get_profile("naming-4") is NAMING_4
    assert "naming-3-legacy" in list_profiles()
    # the two profiles genuinely differ, so A/B is meaningful
    assert NAMING_4.pattern_b_bounded_promotion and not NAMING_3_LEGACY.pattern_b_bounded_promotion
    assert NAMING_4.alarm_on_defect_share and not NAMING_3_LEGACY.alarm_on_defect_share
    with pytest.raises(ValueError):
        get_profile("does-not-exist")


def test_labeling_profile_ab_switch_changes_classification_and_alarm():
    # 2 spare names + several unnamed voices, dominated by short tape. naming-4 bounds the defect to
    # the top-2 and alarms on the defect share; the legacy profile promotes ALL of them and alarms
    # on total unattributed. The switch — and the sidecar provenance — must differ (ADR-138).
    segs = [("HOST", 0, 60), ("A", 60, 360), ("B", 360, 560), ("C", 560, 620), ("D", 620, 680)]
    diar = _diar(segs)
    voice_texts = {
        "HOST": "Welcome to the show. I'm Noah Kravitz and today we dig into a big story here.",
        "A": "a long substantive stretch of reporting about the topic that runs on for a while",
        "B": "more substantive reporting continuing on across several minutes of the episode too",
        "C": "a brief clip from the field with a quick unrelated aside that does not run long",
        "D": "another short piece of tape that also does not run for very long at all here",
    }
    md = ["Alice Anderson", "Bob Brown"]  # 2 spare, unbindable to these voices

    def run(profile):
        r = resolve_speaker_roster(
            diar,
            " ".join(voice_texts.values()),
            voice_texts=voice_texts,
            metadata_named=md,
            known_hosts=[],
            detected_guests=[],
            llm_voice_names=None,
            profile=profile,
        )
        diag = build_speaker_diagnostics(
            diar,
            r,
            transcript_text=" ".join(voice_texts.values()),
            voice_texts=voice_texts,
            detected_guests=[],
            known_hosts=[],
            metadata_named=md,
            profile=profile,
        )
        return diag["summary"]

    v4 = run(NAMING_4)
    legacy = run(NAMING_3_LEGACY)
    # naming-4 bounds the defect (top-2 unknown), legacy promotes all four
    assert v4["by_voice_type"]["unknown"] == 2
    assert legacy["by_voice_type"]["unknown"] == 4
    assert v4["by_voice_type"].get("unidentified", 0) == 2
    # provenance recorded per profile
    assert v4["labeling_profile"] == "naming-4"
    assert legacy["labeling_profile"] == "naming-3-legacy"

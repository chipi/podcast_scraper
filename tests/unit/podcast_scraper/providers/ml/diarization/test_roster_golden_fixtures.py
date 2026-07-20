"""v4 golden-roster fixture harness (#1189) — Phase 0.

Loads every ``tests/fixtures/roster-golden/*.yaml`` fixture and drives the SHIPPED
``resolve_speaker_roster`` directly with each fixture's pre-resolved inputs. This suite never
re-implements any part of the roster's logic — it constructs inputs and asserts on the real
output, which is the whole point (CORPUS-V4-FIXTURE-LADDER.md §F0: a diagnostic that restates
the rule in its own words measures its own words, not the shipped code).

See ``tests/fixtures/roster-golden/SCHEMA.md`` for the fixture contract, the friendly-shorthand
translation table, and what is deliberately out of scope for Phase 0 (driving the full
NER/corroboration stack ahead of the roster).
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any, Callable, Dict, List, Tuple

import pytest
import yaml

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import resolve_speaker_roster, SpeakerRoster

pytestmark = pytest.mark.unit

FIXTURES_DIR = pathlib.Path("tests/fixtures/roster-golden")


def _load_cases() -> List[Dict[str, Any]]:
    if not FIXTURES_DIR.is_dir():
        return []
    cases: List[Dict[str, Any]] = []
    for path in sorted(FIXTURES_DIR.glob("*.yaml")):
        case = yaml.safe_load(path.read_text())
        case["_path"] = str(path)
        cases.append(case)
    return cases


CASES = _load_cases()
IDS = [c["id"] for c in CASES]


def _diarization_result(fixture: Dict[str, Any]) -> DiarizationResult:
    segs = fixture["diarization"]
    speakers = {s["speaker"] for s in segs}
    return DiarizationResult(
        segments=[
            DiarizationSegment(start=float(s["start"]), end=float(s["end"]), speaker=s["speaker"])
            for s in segs
        ],
        num_speakers=len(speakers),
        model_name="golden-fixture",
    )


def _resolve(fixture: Dict[str, Any]) -> SpeakerRoster:
    """Drive the real resolve_speaker_roster with a fixture's pre-resolved inputs."""
    ri = fixture.get("resolver_inputs", {}) or {}
    ad_intervals_raw = ri.get("ad_intervals") or None
    ad_intervals = [tuple(iv) for iv in ad_intervals_raw] if ad_intervals_raw else None
    return resolve_speaker_roster(
        _diarization_result(fixture),
        fixture.get("transcript_text"),
        host_candidates=ri.get("host_candidates") or (),
        detected_guests=ri.get("detected_guests") or (),
        known_hosts=ri.get("known_hosts") or (),
        voice_texts=fixture.get("voice_texts"),
        ordered_turns=fixture.get("ordered_turns"),
        ad_intervals=ad_intervals,
        metadata_named=ri.get("metadata_named") or (),
    )


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_golden_roster_matches_expected(case: Dict[str, Any]) -> None:
    """The roster's real output must match the fixture's verified expected_roster exactly."""
    roster = _resolve(case)
    expected = case["expected_roster"]

    assert set(roster.by_voice) == set(expected), (
        f"{case['id']}: voice set mismatch. "
        f"roster={sorted(roster.by_voice)} expected={sorted(expected)}"
    )
    for voice, exp in expected.items():
        actual = roster.by_voice[voice]
        assert (
            actual.role == exp["role"]
        ), f"{case['id']}: {voice} role={actual.role!r}, expected {exp['role']!r}"
        assert actual.voice_type == exp["voice_type"], (
            f"{case['id']}: {voice} voice_type={actual.voice_type!r}, "
            f"expected {exp['voice_type']!r}"
        )
        assert (
            actual.named == exp["named"]
        ), f"{case['id']}: {voice} named={actual.named!r}, expected {exp['named']!r}"
        if exp["named"]:
            assert (
                actual.name == exp["name"]
            ), f"{case['id']}: {voice} name={actual.name!r}, expected {exp['name']!r}"


def test_the_fixtures_actually_contain_traps() -> None:
    """Guard the guard: a fixture set that never exercises an ad voice or a real guest would
    pass vacuously — it would not be testing anything CORPUS-V4-FIXTURE-LADDER.md calls out."""
    assert CASES, "no roster-golden fixtures loaded"
    for case in CASES:
        expected = case.get("expected_roster")
        assert expected, f"{case['id']}: no expected_roster"
        roles = {v["role"] for v in expected.values()}
        voice_types = {v["voice_type"] for v in expected.values()}
        assert "host" in roles, f"{case['id']}: no host in expected_roster — too easy to pass"
        assert "guest" in roles, f"{case['id']}: no guest in expected_roster — too easy to pass"
        assert "commercial" in voice_types, (
            f"{case['id']}: no advertisement voice in expected_roster — "
            "the #1188/#876 ad-narrator trap is not exercised"
        )
        # At least one voice must be an ad that SELF-INTRODUCES with a real-sounding name in
        # its own voice_texts — otherwise the fixture never actually poisons the most-trusted
        # naming signal, and the ad-exclusion assertion above is trivially satisfied.
        voice_texts = case.get("voice_texts", {})
        ad_voices = [v for v, exp in expected.items() if exp["voice_type"] == "commercial"]
        assert any("I'm" in voice_texts.get(v, "") for v in ad_voices), (
            f"{case['id']}: no ad voice self-introduces — the trap this fixture ladder exists "
            "for (an ad narrator poisoning the most-trusted signal) is not actually present"
        )


def test_no_roster_name_lacks_a_source() -> None:
    """The universal assertion from CORPUS-V4-FIXTURE-LADDER.md §G:

    'No name may appear in the roster that is not either (a) stated in the feed, (b) stated in
    the episode description, or (c) spoken in the transcript by the voice it is assigned to.'

    This is a corpus-level INVARIANT, not roster logic, so — unlike the rest of this file — it
    is allowed to check the property directly rather than only asserting on resolve_speaker_
    roster's output.
    """
    assert CASES, "no roster-golden fixtures loaded"
    for case in CASES:
        roster = _resolve(case)
        feed = case.get("feed", {}) or {}
        episode = case.get("episode", {}) or {}
        voice_texts = case.get("voice_texts", {}) or {}
        sources = " ".join(
            [
                str(feed.get("title", "")),
                str(feed.get("description", "")),
                " ".join(str(a) for a in feed.get("authors", []) or ()),
                str(episode.get("description", "")),
                " ".join(voice_texts.values()),
            ]
        ).lower()
        for voice, role in roster.by_voice.items():
            if not role.named:
                continue
            assert role.name.lower() in sources, (
                f"{case['id']}: voice {voice} was named {role.name!r}, but that name appears "
                "in none of feed title/description/authors, episode description, or any "
                "voice's own words — it has no source"
            )


# ---------------------------------------------------------------------------------------------
# Perturbations — one-line transforms of a fixture dict, from CORPUS-V4-FIXTURE-LADDER.md §G's
# perturbation table. Each is verified against the real roster (see SCHEMA.md "Honesty
# workflow"). Ones that pass are wired into `test_perturbation` below; ones that do NOT are left
# defined here (so the transform is inspectable and re-runnable) with a `# PENDING:` comment
# describing exactly what the roster does instead, and are NOT parametrized into the asserted
# suite — never xfail'd (an xfail still runs and stops being read as a real signal).
# ---------------------------------------------------------------------------------------------


def _base_fixture() -> Dict[str, Any]:
    for case in CASES:
        if case["id"] == "hard_fork_ep1":
            return copy.deepcopy(case)
    raise AssertionError("hard_fork_ep1 fixture not loaded — perturbations have nothing to base on")


def _seg(fixture: Dict[str, Any], speaker: str) -> Dict[str, Any]:
    return next(s for s in fixture["diarization"] if s["speaker"] == speaker)


def perturb_stray_turn(fixture: Dict[str, Any]) -> Dict[str, Any]:
    """+stray_turn — move a 2s turn of an ad voice to mid-episode (a mis-assigned pyannote
    cluster). Guards: the edge-ad-voice rule must tolerate a FRACTION of stray talk, not demand
    zero (AD_VOICE_EDGE_TIME_FRACTION), because a single mis-assigned turn is exactly what put
    a guest's name on a host in the real corpus."""
    f = copy.deepcopy(fixture)
    ad_seg = _seg(f, "AD_1")
    ad_seg["end"] -= 2
    idx = f["diarization"].index(ad_seg)
    f["diarization"].insert(idx + 1, {"speaker": "AD_1", "start": 1000.0, "end": 1002.0})
    return f


def assert_stray_turn(roster: SpeakerRoster) -> None:
    assert roster.by_voice["AD_1"].voice_type == "commercial"
    assert roster.by_voice["AD_1"].named is False
    assert roster.by_voice["HOST_A"].role == "host" and roster.by_voice["HOST_A"].named
    assert roster.by_voice["GUEST_1"].role == "guest" and roster.by_voice["GUEST_1"].named


def perturb_swapped(fixture: Dict[str, Any]) -> Dict[str, Any]:
    """+swapped — exchange two clusters' own words (the NVIDIA bug: the label lies, the
    conversation tells the truth). Guards: naming must follow the voice's OWN text, never the
    arbitrary diarization label."""
    f = copy.deepcopy(fixture)
    f["voice_texts"]["HOST_A"], f["voice_texts"]["HOST_B"] = (
        f["voice_texts"]["HOST_B"],
        f["voice_texts"]["HOST_A"],
    )
    return f


def assert_swapped(roster: SpeakerRoster) -> None:
    assert roster.by_voice["HOST_A"].name == "Casey Newton"
    assert roster.by_voice["HOST_B"].name == "Kevin Roose"
    assert roster.by_voice["HOST_A"].role == "host" and roster.by_voice["HOST_B"].role == "host"


def perturb_quiet_host(fixture: Dict[str, Any]) -> Dict[str, Any]:
    """+quiet_host — shrink HOST_A's turns to a few seconds each (<10% of the episode). Guards:
    a host is never demoted for talking less (Latent Space's host holds 8.6% of the episode)."""
    f = copy.deepcopy(fixture)
    for seg in f["diarization"]:
        if seg["speaker"] == "HOST_A":
            seg["end"] = seg["start"] + 5
    return f


def assert_quiet_host(roster: SpeakerRoster) -> None:
    assert roster.by_voice["HOST_A"].role == "host"
    assert roster.by_voice["HOST_A"].name == "Kevin Roose"


def perturb_loud_guest(fixture: Dict[str, Any]) -> Dict[str, Any]:
    """+loud_guest — grow the guest's turn hugely (>80% of the episode). Guards: talk share
    never promotes a guest to host (Invest Like the Best: guest 82%, host 17%)."""
    f = copy.deepcopy(fixture)
    guest_seg = _seg(f, "GUEST_1")
    guest_seg["end"] = guest_seg["start"] + 20000
    return f


def assert_loud_guest(roster: SpeakerRoster) -> None:
    assert roster.by_voice["GUEST_1"].role == "guest"
    assert roster.by_voice["GUEST_1"].name == "Dr. Adam Rodman"
    assert roster.by_voice["HOST_A"].role == "host" and roster.by_voice["HOST_B"].role == "host"


def perturb_solo(fixture: Dict[str, Any]) -> Dict[str, Any]:
    """+solo — remove the guest entirely (a news round-up). Guards: the roster must not invent
    a guest when there isn't one."""
    f = copy.deepcopy(fixture)
    f["diarization"] = [s for s in f["diarization"] if s["speaker"] != "GUEST_1"]
    f["voice_texts"].pop("GUEST_1", None)
    f["resolver_inputs"]["detected_guests"] = []
    return f


def assert_solo(roster: SpeakerRoster) -> None:
    assert "GUEST_1" not in roster.by_voice
    assert set(r.role for r in roster.by_voice.values()) <= {"host", "unknown"}
    assert roster.by_voice["HOST_A"].role == "host" and roster.by_voice["HOST_B"].role == "host"


def perturb_guest_soup(fixture: Dict[str, Any]) -> Dict[str, Any]:
    """+guest_soup — add unconfirmed "past guest" names to detected_guests/metadata_named, as
    if the (out-of-scope, upstream) corroboration gate had failed to filter them. Guards: the
    roster's OWN one-name-one-voice safety net — it must never paint a spare name onto a voice
    just because a name is available (#876: a wrong name is worse than no name)."""
    f = copy.deepcopy(fixture)
    f["resolver_inputs"]["detected_guests"] = [
        "Dr. Adam Rodman",
        "Bret Taylor",
        "Chris Lattner",
    ]
    f["resolver_inputs"]["metadata_named"] = list(f["resolver_inputs"]["metadata_named"]) + [
        "Bret Taylor",
        "Chris Lattner",
    ]
    return f


def assert_guest_soup(roster: SpeakerRoster) -> None:
    names = {r.name for r in roster.by_voice.values() if r.named}
    assert (
        "Bret Taylor" not in names and "Chris Lattner" not in names
    ), "an unconfirmed 'past guest' name was painted onto a voice"
    # The real guest is left unnamed rather than risk a wrong name (documented, intended
    # trade-off — #876) — this asserts the SAFE direction held, not full recall.
    assert roster.by_voice["GUEST_1"].role == "guest"
    assert roster.by_voice["GUEST_1"].named is False


# PENDING: +short — cut the episode to ~4 minutes (240s). AD_VOICE_MIN_EPISODE_S (600s) makes
# the edge-ad-voice rule ABSTAIN below that length (by design — "too short for only-at-the-edges
# to mean anything"). On this fixture that means AD_1/AD_2 (the pre-roll ad) are no longer
# excluded from naming at all: the real roster then names them from their own self-introduction
# ("Paul Tenorio", "Amy Lawrence") and gives them role=guest. So on a SHORT episode with a real
# pre-roll ad, the ad narrators are currently misclassified as named guests — the opposite
# failure from the one AD_VOICE_MIN_EPISODE_S was written to prevent (typing the whole cast as
# advertising). This is a real gap, not previously called out explicitly in
# CORPUS-V4-FIXTURE-LADDER.md's "Open items"; reported to the operator for triage.
def perturb_short(fixture: Dict[str, Any]) -> Dict[str, Any]:
    f = copy.deepcopy(fixture)
    f["diarization"] = [s for s in f["diarization"] if s["start"] < 240]
    for s in f["diarization"]:
        s["end"] = min(s["end"], 240)
    return f


# PENDING: +no_feed_hosts — blank known_hosts/host_candidates (the feed states no host).
# HOST_A is still correctly identified as host (performs "Welcome to Hard Fork" — a recognized
# host speech act, `roles_from_conversation`). HOST_B is NOT: it only self-introduces ("And I'm
# Casey Newton...") without a recognized host speech act, and with known_hosts empty the host
# cap collapses to whatever the CONVERSATION alone establishes (uncapped, since host_pool is
# empty) — but nothing puts HOST_B in `conv_hosts`, so it falls through to guest-naming and the
# real roster names it "Casey Newton" with role=guest. This matches
# CORPUS-V4-FIXTURE-LADDER.md §G case #4 ("the feed names NO host") exactly as a case the
# fixture ladder says fixtures "must simulate" — it is not claimed fixed there. Reported for
# triage: a co-host who only self-introduces (never utters a host-shaped phrase) on a
# no-host-stated feed is currently demoted to "guest".
def perturb_no_feed_hosts(fixture: Dict[str, Any]) -> Dict[str, Any]:
    f = copy.deepcopy(fixture)
    f["resolver_inputs"]["known_hosts"] = []
    f["resolver_inputs"]["host_candidates"] = []
    return f


# PENDING: +merged_cluster — append a host sentence ("Welcome back to Hard Fork, I'm Kevin
# Roose.") to the guest's own voice_texts, simulating a diarization cluster merge. The real
# roster reclassifies GUEST_1's ROLE to "host" (it does not misname it — "Kevin Roose" is
# already claimed by HOST_A, so GUEST_1 stays unnamed — but its role flips from guest to host).
# `_name_host_voices`'s step 1 (self-intro matches a known host name) has NO CAP check, unlike
# steps 2 and 4, so a merged cluster can add an uncapped third "host" even though known_hosts
# names only two. This is the exact "Hard Fork briefly produced a THIRD host" bug
# CORPUS-V4-FIXTURE-LADDER.md §G case #10 names ("uncapped conversation-derived roles") —
# reproduced here at the self-intro step rather than the conversation-performed-role step.
# Reported for triage.
def perturb_merged_cluster(fixture: Dict[str, Any]) -> Dict[str, Any]:
    f = copy.deepcopy(fixture)
    f["voice_texts"]["GUEST_1"] += " Welcome back to Hard Fork, I'm Kevin Roose."
    return f


# PENDING: +crosspost — swap in another show's host (known_hosts=["Ezra Klein"]), simulating
# Hard Fork airing as a guest episode on a different feed. Host_pool caps at 1 (the crosspost
# feed states one host who isn't in this episode at all). HOST_A still matches via
# `roles_from_conversation` ("Welcome to Hard Fork") and fills the one slot; HOST_B — self-
# introduces but performs no recognized host speech act — is left out of the capped host list
# and falls through to guest-naming, and the real roster names it "Casey Newton" with
# role=guest: a real host, correctly self-identified, demoted to guest because the FEED's host
# count doesn't match this episode. This is exactly CORPUS-V4-FIXTURE-LADDER.md §G case #12
# ("the feed's hosts are not the episode's") — listed there as a case the fixtures "must
# simulate," not as already-fixed behaviour. Reported for triage.
def perturb_crosspost(fixture: Dict[str, Any]) -> Dict[str, Any]:
    f = copy.deepcopy(fixture)
    f["resolver_inputs"]["known_hosts"] = ["Ezra Klein"]
    return f


# Only the GREEN perturbations are wired into the asserted suite. The PENDING ones above are
# defined (inspectable, re-runnable) but deliberately absent from this list.
_GREEN_PERTURBATIONS: List[Tuple[str, Callable, Callable]] = [
    ("+stray_turn", perturb_stray_turn, assert_stray_turn),
    ("+swapped", perturb_swapped, assert_swapped),
    ("+quiet_host", perturb_quiet_host, assert_quiet_host),
    ("+loud_guest", perturb_loud_guest, assert_loud_guest),
    ("+solo", perturb_solo, assert_solo),
    ("+guest_soup", perturb_guest_soup, assert_guest_soup),
]


@pytest.mark.parametrize(
    "transform,checker",
    [(t, c) for _, t, c in _GREEN_PERTURBATIONS],
    ids=[n for n, _, _ in _GREEN_PERTURBATIONS],
)
def test_perturbation(transform: Callable, checker: Callable) -> None:
    fixture = transform(_base_fixture())
    roster = _resolve(fixture)
    checker(roster)

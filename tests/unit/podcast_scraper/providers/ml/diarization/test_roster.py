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


def test_panel_extra_guest_kept_raw() -> None:
    # HOST + three guest voices, but only two guest names → third guest stays raw, never
    # painted with a wrong name.
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
    assert named_guests == {"Alice", "Bob"}
    # exactly one guest voice left unnamed (the third)
    raw_guests = [v for v in r.by_voice.values() if v.role == "guest" and not v.named]
    assert len(raw_guests) == 1


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


def test_more_names_than_voices_drops_extras() -> None:
    diar = _diar([("HOST", 0, 60), ("GUEST", 60, 300)], 2)
    r = resolve_speaker_roster(
        diar,
        "I'm Patrick O'Shaughnessy.",
        detected_guests=["Brian Chesky", "Unused Person", "Also Unused"],
    )
    assert r.by_voice["GUEST"].name == "Brian Chesky"
    assert {v.name for v in r.by_voice.values()} == {"Patrick O'Shaughnessy", "Brian Chesky"}


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
        "by_voice_type": {"person": 1, "unknown": 1},
        "show_centric": False,
        "expected_unresolved": 0,
        "truly_unknown": 1,  # SPEAKER_01 is a substantive person we failed to name
    }
    assert diag["tried"]["host_self_intro"] == "Noah Kravitz"
    by_voice = {v["voice"]: v for v in diag["voices"]}
    assert by_voice["HOST"]["named"] is True and by_voice["HOST"]["source"] == "self_intro"
    assert by_voice["HOST"]["voice_type"] == "person"
    assert by_voice["SPEAKER_01"]["named"] is False
    assert by_voice["SPEAKER_01"]["voice_type"] == "unknown"  # 340s talk time -> substantive
    assert by_voice["SPEAKER_01"]["expected"] is False  # a genuine miss, not expected
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
    # The substantive guest is still a genuine miss.
    assert by_voice["GUEST"]["expected"] is False
    assert diag["summary"]["show_centric"] is True
    assert diag["summary"]["expected_unresolved"] == 1
    assert diag["summary"]["truly_unknown"] == 1  # the guest


def test_voice_type_cameo_commercial_and_unknown() -> None:
    # HOST named; SPEAKER_01 is a long unnamed voice (unknown), SPEAKER_02 a brief cameo (<20s),
    # SPEAKER_03 speaks only inside an ad region (commercial).
    diar = _diar(
        [
            ("HOST", 0, 60),
            ("SPEAKER_01", 60, 400),  # 340s -> unknown (substantive)
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
    assert r.by_voice["SPEAKER_01"].voice_type == "unknown"
    assert r.by_voice["SPEAKER_02"].voice_type == "cameo"
    assert r.by_voice["SPEAKER_03"].voice_type == "commercial"
    # Friendly display labels for the non-person voices (id-bearing label stays raw).
    assert r.display_label_for("SPEAKER_02") == "Brief speaker"
    assert r.display_label_for("SPEAKER_03") == "Advertisement"
    assert r.display_label_for("SPEAKER_01") == "SPEAKER_01"  # substantive unknown keeps raw id
    assert r.label_for("SPEAKER_02") == "SPEAKER_02"  # id-bearing label never swapped


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
    assert r.by_voice["SPEAKER_03"].voice_type == "unknown"  # 60s, no ad info -> substantive

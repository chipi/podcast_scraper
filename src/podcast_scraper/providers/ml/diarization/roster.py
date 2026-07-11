"""Unified speaker-roster resolution — the single source of truth for "who said what" (#876).

Replaces the previously-scattered name sets — feed-level ``cached_hosts``, the diarization-time
self-introduction host, and the guest-only ``detected_speaker_names`` — with **one roster
resolved once**, after transcription + diarization, when the most signal is available. The same
roster feeds the screenplay labels, ``content.speakers`` metadata, GI quote ``speaker_id`` and
``diarization_num_speakers``, so they can no longer disagree (the "screenplay says Patrick,
metadata says Colossus" class of bug).

Resolution, per diarized **voice** (``SPEAKER_xx``):

- **Host voice(s)** = the intro-dominant speaker(s). Named, most-trusted first, from: the
  transcript self-introduction (``I'm …``) → config ``known_hosts`` → filtered feed authors/NER.
  Co-hosts are supported when ≥2 host names are available and a second voice owns a meaningful
  share of the intro.
- **Guest voice(s)** = the remaining voices by total speaking time, named from the detected
  guest list (de-duplicated against host names).
- Network/organisation names ("Colossus") are filtered once, here.
- A guest's name is **never** assigned to a host voice; unmatched voices keep their raw label.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ....speaker_detectors.hosts import (
    extract_self_introduced_host,
    has_org_markers,
    is_network_or_org_author,
)
from .base import DiarizationResult

INTRO_WINDOW_SECONDS = 90.0
# A non-primary voice is also treated as a host when it owns at least this share of the intro
# speaking time AND a host name is available for it (co-hosted shows).
CO_HOST_INTRO_SHARE = 0.30
# An unnamed voice with less than this much total speaking time is a one-off "cameo" — a brief
# interjection not worth naming (measured: ~60% of unresolved voices, ~4% of unknown talk time).
CAMEO_MAX_TALK_S = 20.0
# An unnamed voice whose turns sit mostly inside ad regions is an ad read, not a person.
COMMERCIAL_AD_FRACTION = 0.6

# voice_type values (the *nature* of a voice, distinct from the host/guest role):
VOICE_PERSON = "person"  # a named real person
VOICE_CAMEO = "cameo"  # unnamed, trivially brief
VOICE_COMMERCIAL = "commercial"  # unnamed, mostly inside ad regions
VOICE_UNKNOWN = "unknown"  # unnamed, substantive — a real person we failed to name
# Friendly display labels for the non-person types (surfaces render these instead of SPEAKER_xx).
# ``unknown`` (a substantive person we failed to name) deliberately keeps its raw id, not a label.
VOICE_TYPE_LABELS = {VOICE_CAMEO: "Brief speaker", VOICE_COMMERCIAL: "Advertisement"}
# An unnamed but intro-dominant voice is the host — many show-centric feeds (news desks) never
# name the host, and "Host" is the correct outcome there, not a bare SPEAKER_NN failure.
UNNAMED_HOST_LABEL = "Host"


def friendly_voice_label(voice_type: Optional[str]) -> Optional[str]:
    """Human label for a cameo/commercial voice ("Brief speaker" / "Advertisement"), else None.

    The single source of truth for rendering an *unnamed-but-typed* voice on any surface, so the
    player transcript, the diagnostics, and the roster never disagree. Returns None for a real
    name, a substantive ``unknown`` voice, or an unrecognised type — the caller keeps its raw id.
    """
    return VOICE_TYPE_LABELS.get(voice_type or "")


def friendly_speaker_label(role: Optional[str], voice_type: Optional[str]) -> Optional[str]:
    """Display label for an UNNAMED voice: "Host" for an unnamed host, else the cameo/commercial
    label, else None (a substantive unknown keeps its raw ``SPEAKER_NN`` id). Shared by the roster
    and the segment view so the surface label is derived one way only."""
    if role == "host":
        return UNNAMED_HOST_LABEL
    return friendly_voice_label(voice_type)


@dataclass(frozen=True)
class SpeakerRole:
    """Resolved identity for one diarized voice."""

    name: str  # display label — a real person name, or the raw ``SPEAKER_xx`` when unknown
    role: str  # "host" | "guest" | "unknown"
    named: bool  # True when ``name`` is a real name (not a raw diarization id)
    source: str  # provenance: self_intro | known_hosts | feed | guest | raw
    voice_type: str = VOICE_PERSON  # person | cameo | commercial | unknown (see constants)


@dataclass(frozen=True)
class SpeakerRoster:
    """The full set of resolved voices for an episode."""

    by_voice: Dict[str, SpeakerRole]
    num_speakers: int

    def label_for(self, voice_id: str) -> str:
        """Display label for a diarized voice id (falls back to the raw id when unknown).

        This is the **id-bearing** label (a real name or the raw ``SPEAKER_xx``) — do NOT swap in
        the friendly type label here, or the person-node id would change. Use
        :meth:`display_label_for` for a human surface.
        """
        role = self.by_voice.get(voice_id)
        return role.name if role else voice_id

    def display_label_for(self, voice_id: str) -> str:
        """Human-facing label: a real name, else "Brief speaker" / "Advertisement" for a
        cameo/commercial voice, else the raw id. For rendering only — never for id generation."""
        role = self.by_voice.get(voice_id)
        if role is None:
            return voice_id
        friendly = friendly_speaker_label(role.role, role.voice_type) if not role.named else None
        return friendly or role.name

    def named_count(self) -> int:
        """Number of voices resolved to a real name (not a raw ``SPEAKER_xx``)."""
        return sum(1 for r in self.by_voice.values() if r.named)


def _talk_time(
    diarization: DiarizationResult, *, window_end: Optional[float] = None
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for seg in diarization.segments:
        if window_end is not None and seg.start >= window_end:
            continue
        end = seg.end if window_end is None else min(seg.end, window_end)
        if end <= seg.start:
            continue
        totals[seg.speaker] = totals.get(seg.speaker, 0.0) + (end - seg.start)
    return totals


def _ad_overlap_by_voice(
    diarization: DiarizationResult, ad_intervals: Sequence[Tuple[float, float]]
) -> Dict[str, float]:
    """Seconds of each voice's speaking time that fall inside an ad region."""
    out: Dict[str, float] = {}
    for seg in diarization.segments:
        ov = 0.0
        for a_start, a_end in ad_intervals:
            ov += max(0.0, min(seg.end, a_end) - max(seg.start, a_start))
        if ov > 0:
            out[seg.speaker] = out.get(seg.speaker, 0.0) + ov
    return out


def _classify_voice_types(
    by_voice: Dict[str, "SpeakerRole"],
    diarization: DiarizationResult,
    ad_intervals: Optional[Sequence[Tuple[float, float]]],
) -> Dict[str, "SpeakerRole"]:
    """Tag every *unnamed* voice as cameo / commercial / unknown; named voices are ``person``.

    Lets surfaces show "Brief speaker" / "Advertisement" instead of ``SPEAKER_03`` and lets
    corpus enrichers drop the noise, while the id-bearing raw label is untouched. ``ad_intervals``
    is optional — without it, commercial is not attempted (only cameo vs unknown by talk time).
    """
    talk = _talk_time(diarization)
    ad_by_voice = _ad_overlap_by_voice(diarization, ad_intervals) if ad_intervals else {}
    out: Dict[str, SpeakerRole] = {}
    for v, role in by_voice.items():
        if role.named:
            out[v] = replace(role, voice_type=VOICE_PERSON)
            continue
        total = talk.get(v, 0.0)
        ad_frac = (ad_by_voice.get(v, 0.0) / total) if total else 0.0
        if ad_intervals and ad_frac >= COMMERCIAL_AD_FRACTION:
            vt = VOICE_COMMERCIAL
        elif total < CAMEO_MAX_TALK_S:
            vt = VOICE_CAMEO
        else:
            vt = VOICE_UNKNOWN
        out[v] = replace(role, voice_type=vt)
    return out


def _dedupe(names: Sequence[str], *, reject) -> List[str]:
    """Trim, drop names where ``reject(name)`` is True, de-dup case-insensitively (order kept)."""
    out: List[str] = []
    seen = set()
    for raw in names or ():
        s = (raw or "").strip()
        if not s or reject(s):
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _clean_person_names(names: Sequence[str]) -> List[str]:
    """Clean trusted person names (self-intro / known_hosts / guests).

    Drops only explicit org markers — a single-token name here is a real person (Oprah,
    Sting), not a network, so the mononym rule is NOT applied (#876).
    """
    return _dedupe(names, reject=has_org_markers)


def _clean_author_candidates(names: Sequence[str]) -> List[str]:
    """Clean feed RSS-author host candidates with the full network/org filter (incl. mononym)."""
    return _dedupe(names, reject=is_network_or_org_author)


def _host_name_pool(
    transcript_text: Optional[str],
    known_hosts: Sequence[str],
    host_candidates: Sequence[str],
) -> List[Tuple[str, str]]:
    """Ordered ``(name, source)`` host-name candidates, most-trusted first."""
    pool: List[Tuple[str, str]] = []
    seen = set()

    def _add(name: str, source: str) -> None:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            pool.append((name, source))

    for n in _clean_person_names([extract_self_introduced_host(transcript_text) or ""]):
        _add(n, "self_intro")
    for n in _clean_person_names(known_hosts):
        _add(n, "known_hosts")
    for n in _clean_author_candidates(host_candidates):
        _add(n, "feed")
    return pool


def _self_intros_by_voice(voice_texts: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Per-voice self-introductions ``{voice: name}`` — a voice that says "I'm <First Last>"
    in its *own* turns IS that person. The most reliable per-voice signal, so it names the
    guests/co-hosts that the opening-host self-intro + position-ordered detected-guest list
    miss (the #876 "partial-naming" case: "Hi, I'm Nic Harrigan" rendering as SPEAKER_1).

    Requires a first+last name (≥2 tokens) — guarding the guest path against "I'm American"-style
    false positives; the single main host is still covered by the opening-intro pool.
    """
    out: Dict[str, str] = {}
    for voice, text in (voice_texts or {}).items():
        name = extract_self_introduced_host(text, intro_chars=5000)
        if name and len(name.split()) >= 2:
            out[voice] = name
    return out


def _name_host_voices(
    host_voices: Sequence[str],
    host_pool: Sequence[Tuple[str, str]],
    voice_intro: Dict[str, str],
    used_lower: set,
) -> Dict[str, SpeakerRole]:
    """Name host voices: own self-introduction first, else the ordered host-name pool."""
    out: Dict[str, SpeakerRole] = {}
    hi = 0
    for v in host_voices:
        iname = voice_intro.get(v)
        if iname and iname.lower() not in used_lower:
            used_lower.add(iname.lower())
            out[v] = SpeakerRole(name=iname, role="host", named=True, source="self_intro")
            continue
        while hi < len(host_pool) and host_pool[hi][0].lower() in used_lower:
            hi += 1
        if hi < len(host_pool):
            name, source = host_pool[hi]
            used_lower.add(name.lower())
            hi += 1
            out[v] = SpeakerRole(name=name, role="host", named=True, source=source)
        else:
            out[v] = SpeakerRole(name=v, role="host", named=False, source="raw")
    return out


def _name_guest_voices(
    voices_by_total: Sequence[str],
    assigned: Dict[str, SpeakerRole],
    voice_intro: Dict[str, str],
    guest_names: Sequence[str],
    host_names_lower: set,
    used_lower: set,
) -> Dict[str, SpeakerRole]:
    """Name the remaining voices: own self-introduction first, else the detected-guest list by
    talk-time; unmatched voices stay raw (never painted with someone else's name)."""
    out: Dict[str, SpeakerRole] = {}
    gi = 0
    for v in voices_by_total:
        if v in assigned:
            continue
        iname = voice_intro.get(v)
        if iname and iname.lower() not in used_lower and iname.lower() not in host_names_lower:
            used_lower.add(iname.lower())
            out[v] = SpeakerRole(name=iname, role="guest", named=True, source="self_intro")
        elif gi < len(guest_names):
            out[v] = SpeakerRole(name=guest_names[gi], role="guest", named=True, source="guest")
            gi += 1
        else:
            # Paint a leftover unnamed voice as "guest" only with positive GUEST
            # evidence: detected guest names, or a self-intro from a NON-host voice.
            # ``voice_intro`` is episode-wide and includes the host's own intro, so a
            # host-only-intro show must leave leftovers "unknown", not "guest" (#1170).
            has_guest_intro = any(vid not in assigned for vid in voice_intro)
            role = "guest" if (guest_names or has_guest_intro) else "unknown"
            out[v] = SpeakerRole(name=v, role=role, named=False, source="raw")
    return out


def resolve_speaker_roster(
    diarization: DiarizationResult,
    transcript_text: Optional[str],
    *,
    host_candidates: Sequence[str] = (),
    detected_guests: Sequence[str] = (),
    known_hosts: Sequence[str] = (),
    voice_texts: Optional[Dict[str, str]] = None,
    ad_intervals: Optional[Sequence[Tuple[float, float]]] = None,
    intro_window_s: float = INTRO_WINDOW_SECONDS,
) -> SpeakerRoster:
    """Resolve every diarized voice to a ``SpeakerRole`` (see module docstring).

    ``voice_texts`` maps each diarized voice id to the concatenation of *its own* turns; when
    supplied it lets a voice be named from its own self-introduction (#876). Omitted → the
    previous host-pool + ordered-guest behaviour (fully backward-compatible).

    ``ad_intervals`` (``(start_s, end_s)`` ad regions) lets an unnamed voice that speaks mostly
    inside ads be typed ``commercial``; omitted → only cameo vs unknown by talk time.
    """
    if not diarization.segments:
        return SpeakerRoster(by_voice={}, num_speakers=diarization.num_speakers or 0)

    intro = _talk_time(diarization, window_end=intro_window_s)
    total = _talk_time(diarization)
    voices_by_total = sorted(total, key=lambda v: total[v], reverse=True)
    voices_by_intro = sorted(intro, key=lambda v: intro[v], reverse=True)

    host_pool = _host_name_pool(transcript_text, known_hosts, host_candidates)

    # Which voices are hosts: the intro-dominant voice, plus co-hosts when more host names are
    # available and another voice owns a meaningful share of the intro.
    host_voices: List[str] = []
    if voices_by_intro:
        host_voices.append(voices_by_intro[0])
        intro_total = sum(intro.values()) or 1.0
        for v in voices_by_intro[1:]:
            if len(host_voices) >= max(1, len(host_pool)):
                break
            if intro[v] / intro_total >= CO_HOST_INTRO_SHARE:
                host_voices.append(v)

    # A voice that introduces itself in its own turns is named from that, most-trusted (#876).
    voice_intro = _self_intros_by_voice(voice_texts)
    host_names_lower = {n.lower() for n, _ in host_pool}
    used_lower: set[str] = set()

    by_voice = _name_host_voices(host_voices, host_pool, voice_intro, used_lower)

    intro_names_lower = {n.lower() for n in voice_intro.values()}
    guest_names = [
        g
        for g in _clean_person_names(detected_guests)
        if g.lower() not in host_names_lower and g.lower() not in intro_names_lower
    ]
    by_voice.update(
        _name_guest_voices(
            voices_by_total, by_voice, voice_intro, guest_names, host_names_lower, used_lower
        )
    )
    by_voice = _classify_voice_types(by_voice, diarization, ad_intervals)
    return SpeakerRoster(by_voice=by_voice, num_speakers=diarization.num_speakers or len(by_voice))


def _why_unresolved(voice: str, per_voice_intro: Dict[str, str], guests_available: bool) -> str:
    """Best-effort reason a voice stayed raw (for the diagnostics sidecar)."""
    if voice in per_voice_intro:
        return "self-introduction found but it collided with a host/used name"
    if guests_available:
        return "no first+last self-introduction in own turns; detected-guest names exhausted"
    return "no self-introduction in own turns and no guests were detected for this episode"


def build_speaker_diagnostics(
    diarization: DiarizationResult,
    roster: SpeakerRoster,
    *,
    transcript_text: Optional[str] = None,
    voice_texts: Optional[Dict[str, str]] = None,
    detected_guests: Sequence[str] = (),
    known_hosts: Sequence[str] = (),
    show_centric: bool = False,
) -> Dict[str, Any]:
    """Per-episode speaker-resolution diagnostics — *what we tried, what we resolved, and why
    each voice that stayed raw failed*. Written as a sidecar so an operator can see why a
    speaker is unrecognized without re-running the pipeline.

    ``show_centric`` marks feeds where the host is deliberately unnamed (news desks): an unnamed
    host is then flagged ``expected`` (rendered "Host"), not a detection failure.
    """
    talk = _talk_time(diarization)
    per_voice_intro = _self_intros_by_voice(voice_texts)
    guests_available = bool(_clean_person_names(detected_guests))
    named = sum(1 for r in roster.by_voice.values() if r.named)
    type_counts: Dict[str, int] = {}
    for r in roster.by_voice.values():
        type_counts[r.voice_type] = type_counts.get(r.voice_type, 0) + 1

    voices: List[Dict[str, Any]] = []
    expected_unnamed = 0
    for v, role in roster.by_voice.items():
        entry: Dict[str, Any] = {
            "voice": v,
            "resolved_name": role.name,
            "role": role.role,
            "named": role.named,
            "source": role.source,
            "voice_type": role.voice_type,
            "talk_time_s": round(talk.get(v, 0.0), 1),
        }
        if not role.named:
            # A show-centric feed's unnamed host is the expected outcome, not a failure — it
            # renders "Host". So are cameo/commercial voices (noise, not people we missed).
            expected = (show_centric and role.role == "host") or role.voice_type in (
                VOICE_CAMEO,
                VOICE_COMMERCIAL,
            )
            entry["expected"] = expected
            entry["reason"] = (
                "show-centric feed — host name not expected"
                if (show_centric and role.role == "host")
                else _why_unresolved(v, per_voice_intro, guests_available)
            )
            if expected:
                expected_unnamed += 1
        voices.append(entry)

    unresolved = len(roster.by_voice) - named
    return {
        "summary": {
            "num_speakers": roster.num_speakers,
            "named": named,
            "unresolved": unresolved,
            # Of the unresolved voices, how many are noise (cameo/commercial) vs a real person
            # we failed to name (unknown) — so an operator can tell "worth chasing" from "junk".
            "by_voice_type": type_counts,
            "show_centric": show_centric,
            # Unresolved voices that are the EXPECTED outcome (show-centric host, cameo, ad) vs a
            # genuine miss — ``truly_unknown`` is the real "we failed to name a person" residual.
            "expected_unresolved": expected_unnamed,
            "truly_unknown": unresolved - expected_unnamed,
        },
        "tried": {
            "host_self_intro": (
                extract_self_introduced_host(transcript_text) if transcript_text else None
            ),
            "known_hosts": list(known_hosts),
            "detected_guests": list(detected_guests),
            "per_voice_self_intro": {v: per_voice_intro.get(v) for v in roster.by_voice},
        },
        "voices": voices,
    }

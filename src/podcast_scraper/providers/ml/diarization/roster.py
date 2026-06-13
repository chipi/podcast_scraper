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

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class SpeakerRole:
    """Resolved identity for one diarized voice."""

    name: str  # display label — a real person name, or the raw ``SPEAKER_xx`` when unknown
    role: str  # "host" | "guest" | "unknown"
    named: bool  # True when ``name`` is a real name (not a raw diarization id)
    source: str  # provenance: self_intro | known_hosts | feed | guest | raw


@dataclass(frozen=True)
class SpeakerRoster:
    """The full set of resolved voices for an episode."""

    by_voice: Dict[str, SpeakerRole]
    num_speakers: int

    def label_for(self, voice_id: str) -> str:
        """Display label for a diarized voice id (falls back to the raw id when unknown)."""
        role = self.by_voice.get(voice_id)
        return role.name if role else voice_id

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


def resolve_speaker_roster(
    diarization: DiarizationResult,
    transcript_text: Optional[str],
    *,
    host_candidates: Sequence[str] = (),
    detected_guests: Sequence[str] = (),
    known_hosts: Sequence[str] = (),
    intro_window_s: float = INTRO_WINDOW_SECONDS,
) -> SpeakerRoster:
    """Resolve every diarized voice to a ``SpeakerRole`` (see module docstring)."""
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

    by_voice: Dict[str, SpeakerRole] = {}
    for i, v in enumerate(host_voices):
        if i < len(host_pool):
            name, source = host_pool[i]
            by_voice[v] = SpeakerRole(name=name, role="host", named=True, source=source)
        else:
            by_voice[v] = SpeakerRole(name=v, role="host", named=False, source="raw")

    # Guests: remaining voices by total speaking time, named from detected guests (minus hosts).
    host_names_lower = {n.lower() for n, _ in host_pool}
    guest_names = [
        g for g in _clean_person_names(detected_guests) if g.lower() not in host_names_lower
    ]
    gi = 0
    for v in voices_by_total:
        if v in by_voice:
            continue
        if gi < len(guest_names):
            by_voice[v] = SpeakerRole(
                name=guest_names[gi], role="guest", named=True, source="guest"
            )
            gi += 1
        else:
            # Unmatched voice: a guest role if guests exist, else unknown — kept raw, never
            # painted with someone else's name.
            role = "guest" if guest_names else "unknown"
            by_voice[v] = SpeakerRole(name=v, role=role, named=False, source="raw")

    return SpeakerRoster(by_voice=by_voice, num_speakers=diarization.num_speakers or len(by_voice))

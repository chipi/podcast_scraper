"""Unified speaker-roster resolution — the single source of truth for "who said what" (#876).

Replaces the previously-scattered name sets — feed-level ``cached_hosts``, the diarization-time
self-introduction host, and the guest-only ``detected_speaker_names`` — with **one roster
resolved once**, after transcription + diarization, when the most signal is available. The same
roster feeds the screenplay labels, ``content.speakers`` metadata, GI quote ``speaker_id`` and
``diarization_num_speakers``, so they can no longer disagree (the "screenplay says Patrick,
metadata says Colossus" class of bug).

Resolution, per diarized **voice** (``SPEAKER_xx``):

- **Host voice(s)** = the **opening** speaker (whoever starts the episode — the host doing the
  intro), not the intro-window talk-time leader, which the guest wins whenever they answer at
  length early (#1169). Named, most-trusted first, from: the transcript self-introduction
  (``I'm …``) → config ``known_hosts`` → filtered feed authors/NER. Co-hosts are supported when
  ≥2 host names are available and a second voice owns a meaningful share of the intro.
- **Guest voice(s)** = the remaining voices by total speaking time, named from the detected
  guest list (de-duplicated against host names).
- Network/organisation names ("Colossus") are filtered once, here.
- A guest's name is **never** assigned to a host voice; unmatched voices keep their raw label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import AbstractSet, Any, Dict, List, Optional, Sequence, Set, Tuple

from ....speaker_detectors.hosts import (
    _clean_stated_name as _clean_intro_name,
    _GUEST_GREETED as _GUEST_GREETED_RE,
    _GUEST_INTRODUCED_BY_HOST as _GUEST_INTRODUCED_BY_HOST_RE,
    _GUEST_INTRODUCED_NAME_FIRST as _GUEST_INTRODUCED_NAME_FIRST_RE,
    _NAME_RE as _INTRO_NAME_RE,
    CUE_FIRST_BODY,
    CUE_FIRST_PAST_BODY,
    distinct_self_introductions,
    extract_self_introduced_host,
    GREETED_TAIL,
    guests_introduced_by_the_host,
    has_org_markers,
    HONORIFIC_TITLES,
    is_network_or_org_author,
    is_plausible_mononym,
    is_publishable_speaker_name,
    looks_like_a_person_name,
    NAME_FIRST_REPORT_TAIL,
    NAME_FIRST_TAIL,
    roles_from_conversation,
)
from ....text_normalization import (
    first_names_match,
    normalize_for_match,
    normalize_name_for_match,
)
from .base import DiarizationResult
from .labeling_profile import DEFAULT_LABELING_PROFILE, LabelingProfile
from .labeling_strategy import (
    _DEEPGRAM,
    DiarizationLabelingStrategy,
    labeling_strategy_for,
)

INTRO_WINDOW_SECONDS = 90.0
# A non-primary voice is also treated as a host when it owns at least this share of the intro
# speaking time AND a host name is available for it (co-hosted shows).
CO_HOST_INTRO_SHARE = 0.30
# An unnamed voice with less than this much total speaking time is a one-off "cameo" — a brief
# interjection not worth naming (measured: ~60% of unresolved voices, ~4% of unknown talk time).
CAMEO_MAX_TALK_S = 20.0
# A SHORT voice that introduces itself as several people is a cold-open montage clip whose entire
# run fits inside the opening ("I'm Kevin Russo… I'm Casey Noon…" measured 13s) — set to the intro
# window because that is where such a montage lives. Above it, a voice with two self-intros is a
# real dominant speaker whose cluster merely ABSORBED a merged cold-open clip (real hosts measured
# 400–1500s); it keeps its name, resolved from its own leading self-intro. Only the brief clip is
# suppressed. Residual gap (accepted, unlikely for a cold open): a pure montage COMPILATION longer
# than this window would keep its first-intro name rather than be suppressed.
MONTAGE_CLIP_MAX_TALK_S = INTRO_WINDOW_SECONDS
# An unnamed voice whose turns sit mostly inside ad regions is an ad read, not a person.
COMMERCIAL_AD_FRACTION = 0.6

# A voice that speaks only at the very top (or very bottom) of the episode and then is never heard
# again is an AD READ, whatever it calls itself — and it does not need `ad_intervals` to be spotted.
#
# It has to be spotted without them, because the ad-pattern list only knows *sponsor* language
# ("brought to you by", "dot com slash promo") and modern house ads carry none of it. Hard Fork's
# pre-roll is two Athletic journalists introducing themselves and plugging their World Cup app: zero
# pattern hits, so `ad_intervals` came back EMPTY and every ad-aware guard below was inert.
#
# The ad then walked straight through the roster's most-trusted signal. `_self_intros_by_voice`
# holds that a voice saying "I'm <First Last>" IS that person — and reading its own name is the one
# thing an ad narrator always does. So "Paul Tenorio" and "Amy Lawrence", who cover soccer and
# football for The Athletic, were crowned the hosts of a technology podcast in 10 of 10 episodes,
# taking roster slots from the real hosts and leaving a cluster free for a hallucinated "Elon Musk"
# to claim.
#
# Structure separates them with an enormous margin and no keywords at all. Measured over those 10:
#
#     hosts          26-42% of talk, spanning 96-99% of the episode
#     guests         11-22% of talk, spanning 18-42%
#     ad narrators   0.3-0.4% of talk, spanning 1%  (gone by 0:30, never return)
#
# So: under AD_VOICE_MAX_TALK_S of speech in total, AND every turn confined to the first or last
# AD_VOICE_EDGE_WINDOW_S. A host cannot satisfy that, and neither can a guest. The failure mode is
# to under-name a genuinely brief edge speaker, which costs a `SPEAKER_01` — the safe direction
# (#876), and cheap next to publishing a real person's words under an advertiser's name.
#
# All THREE must hold, and the share test is the one doing the real work. An absolute
# "short + at the edges" rule is meaningless on a short episode — in a three-minute clip every voice
# is near an edge and under a minute of talk, which would type the whole cast as advertising. Share
# is scale-free, and it is where the measured gap actually is (0.4% vs 11%).
AD_VOICE_MAX_TALK_S = 90.0
AD_VOICE_MAX_SHARE = 0.03
AD_VOICE_EDGE_WINDOW_S = 150.0
# Almost all of an ad voice's speech sits in the edge windows — but not necessarily ALL of it.
# Requiring zero turns elsewhere was too brittle to survive real diarization: pyannote mis-assigned
# a single mid-episode turn to Amy Lawrence's cluster, that one turn disqualified her from the ad
# test, and the whole failure cascaded — she was named from her own ad self-intro, she OPENED the
# episode so she took a host slot, the host cap (two known hosts) was then full, and the real
# co-host was pushed out to GUEST naming and handed Dr. Adam Rodman's name.
AD_VOICE_EDGE_TIME_FRACTION = 0.75
# Below this an episode is too short for "only at the edges" to mean anything, so the rule abstains.
AD_VOICE_MIN_EPISODE_S = 600.0

# WHO the hosts are is NOT inferred here. It is read from METADATA — the feed states it in plain
# English — and passed in as `known_hosts`. This module's job is to work out WHICH VOICE each of
# them is, not to guess who they are.
#
# There was a `HOST_MIN_SHARE` / spanning rule here, derived from Hard Fork ("a host talks a lot and
# is present start to finish"). It was wrong, because talk share and span INVERT by show format:
#
#     Invest Like the Best   the GUEST talks 82%, the host 17%
#     Latent Space           the GUEST talks 85%
#     Hard Fork              the HOSTS talk 26-39%, the guest 22%
#     Hard Fork              the episode is OPENED by a pre-roll ADVERT, not by a host
#
# Any rule keyed on "who talks most" or "who spans the episode" is therefore tuned to whichever show
# it was written against, and it promoted the guest to host on the interview-format feeds. Meanwhile
# the feed simply says: "journalists Kevin Roose and Casey Newton"; "Hosted by Ryan Knutson and
# Jessica Mendoza"; "co-hosts Elad Gil and Sarah Guo" — and Invest Like the Best puts its host in
# the show TITLE. 7 of our 10 feeds name their hosts outright; the rest carry author tags.
#
# So: metadata is the authority for WHO, diarization is the authority for WHICH VOICE, and the two
# are cross-referenced. A statistic never overrules a stated fact.
#
# Talk share remains legitimate for exactly one question — AD vs PERSON (an ad reads for 30 seconds,
# a host talks for 20 minutes, and that gap does not invert across formats). It is never used to
# separate host from guest.

# voice_type values (the *nature* of a voice, distinct from the host/guest role):
VOICE_PERSON = "person"  # a named real person
VOICE_CAMEO = "cameo"  # unnamed, trivially brief
VOICE_COMMERCIAL = "commercial"  # unnamed, mostly inside ad regions
VOICE_UNKNOWN = "unknown"  # unnamed, substantive — a real person we FAILED to name (a defect)
# ...and a real person NOBODY NAMES. Not the same thing, and until the corpus audit existed we
# could not tell them apart, so both rendered as a raw SPEAKER_07.
#
# This is TAPE: the vox-pop interviewee in a narrated documentary. Measured across the corpus, they
# speak for 20-180 seconds — far too long to be a "Brief speaker" (the cameo rule stops at 20s), and
# every one of them is substantive first-person testimony:
#
#     [Planet Money] SPEAKER_09, 151s: "I was in the shocks, which was the part that would tie off
#                                       from the harness..."
#
# Nobody in the episode ever says who that is. There is no name to be had — from the feed, from the
# description, from an introduction, or from their own mouth. So `SPEAKER_09` is not a failure
# marker here, it is just ugly: we did nothing wrong.
#
# Keeping the two apart matters because the raw id is meant to MEAN something — "we should have
# named this and did not". Showing it on a voice nobody could have named turns a defect signal into
# noise, and a defect signal nobody trusts stops being a signal.
VOICE_UNIDENTIFIED = "unidentified"  # unnamed, substantive — and NO source names them
# Voice types that are NOT real conversational speakers — dropped before GI/KG (the labeling
# OUTPUT surface excludes them). ADR-135/#1220.
_NOISE_VOICE_TYPES = frozenset({VOICE_CAMEO, VOICE_COMMERCIAL})
# Friendly display labels for the non-person types (surfaces render these instead of SPEAKER_xx).
# ``unknown`` (a person we FAILED to name) keeps its raw id — that raw id IS the defect marker.
VOICE_TYPE_LABELS = {
    VOICE_CAMEO: "Brief speaker",
    VOICE_COMMERCIAL: "Advertisement",
    VOICE_UNIDENTIFIED: "Unidentified speaker",
}
# An unnamed but intro-dominant voice is the host — many show-centric feeds (news desks) never
# name the host, and "Host" is the correct outcome there, not a bare SPEAKER_NN failure.
UNNAMED_HOST_LABEL = "Host"

# The unattributed-talk alarm threshold now lives on the profile as ``unattributed_alarm_threshold``
# (ADR-140, default 0.25) — read directly by build_speaker_diagnostics. The old module constant was
# left behind unread after that move (review A7), so it is removed to avoid two sources of truth.


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
    diarization: DiarizationResult,
    *,
    window_start: float = 0.0,
    window_end: Optional[float] = None,
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for seg in diarization.segments:
        if window_end is not None and seg.start >= window_end:
            continue
        if seg.end <= window_start:
            continue
        start = max(seg.start, window_start)
        end = seg.end if window_end is None else min(seg.end, window_end)
        if end <= start:
            continue
        totals[seg.speaker] = totals.get(seg.speaker, 0.0) + (end - start)
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


def _edge_ad_voices(diarization: DiarizationResult) -> set:
    """Voices that speak ONLY at the top/bottom of the episode, briefly, and for a trivial share.

    Keyword-free, so it catches the house ads and cross-promos the sponsor-pattern list cannot see
    (see AD_VOICE_MAX_TALK_S). All three tests must pass, and the SHARE test carries the rule: on a
    short episode every voice is near an edge and under a minute of talk, so an absolute-only rule
    would type an entire three-minute cast as advertising. Share is scale-free.

    The edge test is a FRACTION of the voice's speech, not "no turns elsewhere". Demanding zero
    stray turns did not survive contact with real diarization: pyannote mis-assigned one mid-episode
    turn to the ad narrator's cluster, that single turn cleared her of being an ad, and everything
    downstream fell over (see AD_VOICE_EDGE_TIME_FRACTION).

    A host or guest fails every one of these by a wide margin.
    """
    if not diarization.segments:
        return set()
    episode_end = max(s.end for s in diarization.segments)
    if episode_end < AD_VOICE_MIN_EPISODE_S:
        return set()  # too short for "only at the edges" to carry any information

    head = AD_VOICE_EDGE_WINDOW_S
    tail = episode_end - AD_VOICE_EDGE_WINDOW_S

    talk: Dict[str, float] = {}
    edge_talk: Dict[str, float] = {}
    for seg in diarization.segments:
        dur = max(0.0, seg.end - seg.start)
        talk[seg.speaker] = talk.get(seg.speaker, 0.0) + dur
        if seg.end <= head or seg.start >= tail:
            edge_talk[seg.speaker] = edge_talk.get(seg.speaker, 0.0) + dur

    spoken = sum(talk.values()) or 1.0
    return {
        v
        for v, secs in talk.items()
        if secs > 0
        and secs < AD_VOICE_MAX_TALK_S
        and (secs / spoken) < AD_VOICE_MAX_SHARE
        and (edge_talk.get(v, 0.0) / secs) >= AD_VOICE_EDGE_TIME_FRACTION
    }


def _voices_by_talk(diarization: DiarizationResult, ad_voices: set) -> List[str]:
    """Non-ad voices, most talkative first — the order host NAMES from metadata are matched onto.

    This is an ORDERING, not a classification. It does not decide who is a host: the feed already
    said that, and `known_hosts` carries it. It only decides which voice a given host name is
    matched to first, which is why it is safe — the count of hosts comes from the metadata, so a
    guest cannot become one by talking a lot.
    """
    talk: Dict[str, float] = {}
    for seg in diarization.segments:
        if seg.speaker in ad_voices:
            continue
        talk[seg.speaker] = talk.get(seg.speaker, 0.0) + max(0.0, seg.end - seg.start)
    return sorted(talk, key=lambda v: talk[v], reverse=True)


def _opening_voice(
    diarization: DiarizationResult,
    *,
    window_end: float,
    ad_intervals: Optional[Sequence[Tuple[float, float]]] = None,
    ad_voices: Optional[set] = None,
) -> Optional[str]:
    """The voice that OPENS the episode — the speaker of the earliest turn in the intro window
    (the host doing the intro). A turn sitting mostly inside an ad region is skipped (a pre-roll
    ad read is not the host). This mirrors ``gi.speakers``' "opening cluster -> host" rule over
    diarization time, and beats intro-window talk-time — which the guest wins whenever they
    answer at length early, swapping the roles (#1169). ``None`` when no turn qualifies.
    """
    ads = ad_intervals or ()
    skip = ad_voices or set()
    best_start: Optional[float] = None
    best_voice: Optional[str] = None
    for seg in diarization.segments:
        dur = seg.end - seg.start
        if seg.start >= window_end or dur <= 0:
            continue
        # The pre-roll ad OPENS the episode, so "whoever starts" is the ad narrator unless the ad
        # is skipped. `ad_intervals` only sees sponsor-shaped ads; `ad_voices` sees the rest.
        if seg.speaker in skip:
            continue
        in_ad = sum(
            max(0.0, min(seg.end, a_end) - max(seg.start, a_start)) for a_start, a_end in ads
        )
        if in_ad / dur >= COMMERCIAL_AD_FRACTION:
            continue
        if best_start is None or seg.start < best_start:
            best_start, best_voice = seg.start, seg.speaker
    return best_voice


def _classify_voice_types(
    by_voice: Dict[str, "SpeakerRole"],
    diarization: DiarizationResult,
    ad_intervals: Optional[Sequence[Tuple[float, float]]],
    ad_voices: Optional[set] = None,
    nameable: Optional[set] = None,
    cleaning: Optional["VoiceCleaning"] = None,
    cameo_max_talk_s: float = CAMEO_MAX_TALK_S,
) -> Dict[str, "SpeakerRole"]:
    """Tag every *unnamed* voice; named voices are ``person``.

    Lets surfaces show "Brief speaker" / "Advertisement" / "Unidentified speaker" instead of
    ``SPEAKER_03``, while the id-bearing raw label is untouched. ``ad_intervals`` is optional —
    without it, commercial is not attempted.

    ``nameable`` is the set of voices for which a name EXISTED somewhere — the voice introduced
    itself, the host introduced it, or a declared guest name was still going spare. Those are the
    ones we FAILED on, and they keep the raw ``SPEAKER_NN`` because that id is the defect marker.
    A substantive voice that is NOT nameable is ``unidentified``: nobody in the episode ever says
    who they are, so there was nothing to fail at.
    """
    talk = _talk_time(diarization)
    ad_by_voice = _ad_overlap_by_voice(diarization, ad_intervals) if ad_intervals else {}
    edge_ads = ad_voices or set()
    out: Dict[str, SpeakerRole] = {}
    for v, role in by_voice.items():
        # An edge-ad voice is commercial even when it is NAMED. Being named used to short-circuit
        # straight to `person`, and an ad narrator is always named — it reads its own name out loud.
        # That is how "Advertisement" became "Paul Tenorio, host".
        if v in edge_ads:
            out[v] = replace(role, name=v, named=False, voice_type=VOICE_COMMERCIAL)
            continue
        if role.named:
            out[v] = replace(role, voice_type=VOICE_PERSON)
            continue
        total = talk.get(v, 0.0)
        if cleaning is not None:
            # Single source of truth (ADR-137): the shared classifier already decided noise.
            is_commercial, is_cameo = v in cleaning.commercial, v in cleaning.cameo
        else:
            ad_frac = (ad_by_voice.get(v, 0.0) / total) if total else 0.0
            is_commercial = bool(ad_intervals) and ad_frac >= COMMERCIAL_AD_FRACTION
            is_cameo = total < cameo_max_talk_s
        if is_commercial:
            vt = VOICE_COMMERCIAL
        elif is_cameo:
            vt = VOICE_CAMEO
        elif nameable is not None and v not in nameable:
            # Substantive, and NO source names them: the tape / vox-pop of a narrated documentary.
            # Not a failure — there was no name to be had.
            vt = VOICE_UNIDENTIFIED
        else:
            vt = VOICE_UNKNOWN
        out[v] = replace(role, voice_type=vt)
    return out


@dataclass(frozen=True)
class VoiceCleaning:
    """Which diarized voices are NOISE (ad / cameo / commercial) vs REAL speakers.

    The deterministic cleaning classification, computed ONCE right after diarization and consumed by
    BOTH the LLM resolution call (which needs a clean intro + real-voice candidates) and the roster
    — so "which voices are noise" is defined in one place and never replicated (ADR-137). It answers
    only real-vs-noise; the finer person/unknown/unidentified split is a naming question the roster
    still draws later.
    """

    ad: frozenset
    cameo: frozenset
    commercial: frozenset
    real: frozenset


def _ad_voices_for(
    diarization: DiarizationResult,
    ordered_turns: Optional[Sequence[Tuple[str, str]]],
    voice_texts: Optional[Dict[str, str]],
    recurring_text: Optional[set],
    diarization_provider: Optional[str],
) -> set:
    """The full ad-voice set: edge ads (``_edge_ad_voices``) plus the cross-episode recurring-ad
    voices the edge rule cannot see (#1188 — a mid-roll house ad read from the same script weekly).

    Factored out so :func:`classify_voices` and the roster's standalone path compute it identically.
    """
    ad = _edge_ad_voices(diarization)
    if recurring_text:
        strategy = labeling_strategy_for(diarization_provider)
        ad = ad | strategy.recorded_voices(
            ordered_turns or [], voice_texts or {}, _talk_time(diarization), recurring_text
        )
    return ad


def classify_voices(
    diarization: DiarizationResult,
    ad_intervals: Optional[Sequence[Tuple[float, float]]] = None,
    *,
    voice_texts: Optional[Dict[str, str]] = None,
    ordered_turns: Optional[Sequence[Tuple[str, str]]] = None,
    recurring_text: Optional[set] = None,
    diarization_provider: Optional[str] = None,
    cameo_max_talk_s: float = CAMEO_MAX_TALK_S,
) -> VoiceCleaning:
    """Classify every diarized voice as ad / cameo / commercial / real (see :class:`VoiceCleaning`).

    Uses only signals available the moment diarization finishes — the SAME primitives the roster's
    typing uses (``_edge_ad_voices`` + the cross-episode recurring-ad strategy, talk-time, ad
    overlap). Naming is neither required nor used: this is the real-vs-noise cut both the LLM call
    and the roster share.
    """
    ad = _ad_voices_for(
        diarization, ordered_turns, voice_texts, recurring_text, diarization_provider
    )
    talk = _talk_time(diarization)
    ad_by_voice = _ad_overlap_by_voice(diarization, ad_intervals) if ad_intervals else {}
    cameo: Set[str] = set()
    commercial: Set[str] = set()
    for v, total in talk.items():
        if v in ad:
            continue
        ad_frac = (ad_by_voice.get(v, 0.0) / total) if total else 0.0
        if ad_intervals and ad_frac >= COMMERCIAL_AD_FRACTION:
            commercial.add(v)
        elif total < cameo_max_talk_s:
            cameo.add(v)
    real = {v for v in talk if v not in ad and v not in cameo and v not in commercial}
    return VoiceCleaning(
        ad=frozenset(ad),
        cameo=frozenset(cameo),
        commercial=frozenset(commercial),
        real=frozenset(real),
    )


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
    """Ordered ``(name, source)`` host-name candidates, most-trusted first.

    The FEED comes first. It states its hosts, and a statement outranks a guess.

    The transcript self-introduction used to lead this list, and it is the wrong thing to trust:
    ``extract_self_introduced_host`` reads the FIRST "I'm <Name>" in the transcript, and the first
    thing in the transcript is the PRE-ROLL AD. On episode 5 of the rebuild that put "I'm Paul
    Tenorio" (a soccer writer, reading an advert) at the head of the host pool, and his name was
    then painted onto a voice holding 37% of a technology podcast.

    So the self-intro is now a FALLBACK — used only when the feed names nobody, which is the case
    for 3 of our 10 feeds. There it is genuinely useful ("hello and welcome to Planet Money, I'm
    Alexi Horowitz-Gazi"). Where the feed HAS spoken, nothing in the audio may overrule it.
    """
    pool: List[Tuple[str, str]] = []
    seen = set()

    def _add(name: str, source: str) -> None:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            pool.append((name, source))

    for n in _clean_person_names(known_hosts):
        _add(n, "known_hosts")
    for n in _clean_author_candidates(host_candidates):
        _add(n, "feed")
    if not pool:  # the feed named nobody — only then does the transcript get a vote
        for n in _clean_person_names([extract_self_introduced_host(transcript_text) or ""]):
            _add(n, "self_intro")
    return pool


def _soundex(word: str) -> str:
    """Classic Soundex. Catches ASR substitutions that swap vowels but keep the consonant skeleton
    ("Roose" -> "Russo"). Blind to "Newton" -> "Noon", which edit distance catches instead — the two
    are complementary and neither alone is enough."""
    w = "".join(c for c in word.upper() if c.isalpha())
    if not w:
        return ""
    codes = {
        **dict.fromkeys("BFPV", "1"),
        **dict.fromkeys("CGJKQSXZ", "2"),
        **dict.fromkeys("DT", "3"),
        **dict.fromkeys("L", "4"),
        **dict.fromkeys("MN", "5"),
        **dict.fromkeys("R", "6"),
    }
    out, prev = w[0], codes.get(w[0], "")
    for ch in w[1:]:
        code = codes.get(ch, "")
        if code and code != prev:
            out += code
        if ch not in "HW":
            prev = code
    return (out + "000")[:4]


def _edit_distance(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# Generational suffixes that are not the surname ("Robert Pape Jr." -> surname "pape", not "jr").
_NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})


def _surname_token(name: str) -> Optional[str]:
    """The lowercased last name-token, or ``None`` for a mononym/too-short token. Used to tell that

    "Robert Pape", "Professor Pape" and "Dr. Pape" are the same person for de-duplication. A ≥2-char
    floor keeps an initial ("R.") from matching everything while still recognising short romanised
    surnames (Xu, Li, Ng); trailing generational suffixes are dropped first.
    """
    toks = [t.strip(".,'’") for t in (name or "").split()]
    toks = [t for t in toks if t and t.lower() not in _NAME_SUFFIXES]
    if len(toks) < 2:
        return None
    last = toks[-1].lower()
    return last if len(last) >= 2 else None


def _given_tokens(name: str) -> List[str]:
    """Name tokens after stripping leading honorifics ("Dr. Adam Rodman" -> ["Adam", "Rodman"])."""
    toks = [t.strip(".,'’") for t in (name or "").split()]
    while toks and toks[0].lower() in HONORIFIC_TITLES:
        toks = toks[1:]
    return toks


def _same_person(a: str, b: str) -> bool:
    """Whether two names denote the SAME person, for de-duplicating a diarizer-split guest or a
    title variant. Same surname AND (one side is title-only, OR matching given name, OR one given is
    an initial of the other). "Dr. Adam Rodman" == "Adam Rodman"; "Professor Fenwick" ==
    "Alan Fenwick"; but "Robert Pape" != "Karen Pape" — distinct people who merely share a surname.
    """
    sa, sb = _surname_token(a), _surname_token(b)
    if not sa or sa != sb:
        return False
    ga, gb = _given_tokens(a), _given_tokens(b)
    if len(ga) >= 2 and len(gb) >= 2:  # both carry a given name before the surname
        fa, fb = ga[0].lower(), gb[0].lower()
        return (
            fa == fb or (len(fa) == 1 and fb.startswith(fa)) or (len(fb) == 1 and fa.startswith(fb))
        )
    return True  # one side is title + surname only ("Professor Pape")


def _canonicalize_to_stated_person(name: str, stated: Sequence[str]) -> str:
    """Upgrade a host-introduced title-form ("Professor Pape") to the metadata-stated WHOLE name of
    the same person ("Robert Pape").

    When the host greets a guest by a title + surname the intro reader names the RIGHT voice but
    with a degraded label; the metadata already states that person's full name, and the code already
    knows they are the same (:func:`_same_person`) — it just never used that to upgrade the label.
    We adopt a stated name, never strip the title to fabricate one — this respects reject-not-strip.
    Snap only when exactly one stated person carries a real given name AND is that same person; on
    ambiguity (two people share the surname) keep the title form (#1169)."""
    matches = {s for s in stated if len(_given_tokens(s)) >= 2 and _same_person(name, s)}
    return next(iter(matches)) if len(matches) == 1 else name


def _canonicalize_to_known_host(
    name: str,
    known_hosts: Sequence[str],
    *,
    first_name_max_edit: int = 0,
    mononym_ok: bool = False,
) -> str:
    """Snap an ASR-mangled self-introduction onto the configured host name.

    A self-introduction is transcribed, so it carries the ASR's spelling: Kevin Roose introduces
    himself and Whisper writes "Kevin Russo" in one episode and "Kevin Roos" in the next. The roster
    trusts a self-intro above ``known_hosts``, so the corpus ended up with three different people
    hosting the same show, none of them spelled correctly.

    Snapping requires an EXACT first-name match plus a near surname (phonetic, or within a small
    edit distance), so a guest who merely shares a host's first name is left alone. Requiring both
    is what keeps this from quietly renaming real people.

    ``first_name_max_edit`` (audit 2a) relaxes the first name to a small edit distance — but ONLY
    when the surname matches strongly (soundex or edit ≤ 1), so "Arietta Laika" snaps to the stated
    "Arijeta Lajka" (first edit 2, surname edit 1) without letting a relaxed first name rename a
    genuinely different person. Host callers keep the default 0 (exact). ``mononym_ok`` (audit 3)
    lets a bare first name ("Kevin") snap to a stated person iff EXACTLY ONE reference carries it.
    """
    toks = name.split()
    if len(toks) == 1 and mononym_ok:
        first = toks[0].lower()
        matches = [r for r in known_hosts if r.split() and r.split()[0].lower() == first]
        # Abstain on ambiguity: only snap when a single stated person owns that first name.
        return matches[0] if len(matches) == 1 else name
    if len(toks) < 2:
        return name
    first, last = toks[0].lower(), toks[-1]
    for host in known_hosts:
        h = host.split()
        if len(h) < 2:
            continue
        # A known nickname or initial ("Rich"↔"Richard", "R."↔"Robert") is a confident given-name
        # equivalence, not a fuzzy guess, so it counts as an EXACT first-name match (ADR-139) — the
        # surname branch below still demands a strong surname match, so this cannot rename a
        # different person who merely shares a nickname.
        first_exact = h[0].lower() == first or first_names_match(h[0], first)
        first_near = first_name_max_edit > 0 and _edit_distance(h[0].lower(), first) <= (
            first_name_max_edit
        )
        if not (first_exact or first_near):
            continue
        if not first_exact:
            # Relaxed first name → demand a STRONG surname match so we cannot rename a different
            # person who merely has a near first name.
            if _soundex(last) == _soundex(h[-1]) or _edit_distance(last, h[-1]) <= 1:
                return host
            continue
        if _soundex(last) == _soundex(h[-1]) or _edit_distance(last, h[-1]) <= 3:
            return host
        # A shared surname STEM, on top of the exact first-name match. "Natalie Kitcher" is the
        # ASR's rendering of Natalie Kitroeff, a stated host of The Daily: soundex misses it and the
        # edit distance is 4 — one over the threshold. Demanding an exact first name, a shared
        # three-letter surname stem AND a bounded edit distance is a far narrower claim than any of
        # the three alone, and it leaves "Kevin Systrom" / "Casey Affleck" untouched.
        if (
            len(last) >= 4
            and len(h[-1]) >= 4
            and last[:3].lower() == h[-1][:3].lower()
            and _edit_distance(last, h[-1]) <= 5
        ):
            return host
    return name


def _canonicalize_to_stated_name(name: str, stated: Sequence[str]) -> str:
    """ADR-130: snap an ASR-mangled published name to the correctly-spelled name the episode
    metadata STATES — host OR guest — by the same fuzzy rule ``_canonicalize_to_known_host`` uses.

    Every ASR mistranscribes proper nouns (OpenAI Whisper wrote "Kevin Russo"; turbo writes
    "Kevin Roos" / "David Duvino"), and naming reads names out of the transcript. The correct
    spelling is almost always in the episode's own metadata: the feed states the hosts, the title +
    description name the guest. Host snapping already exists; this applies the same matcher to the
    FULL stated set so guests recover symmetrically. Provider-agnostic. Reference-bounded — it can
    only ever return a name in ``stated`` (never invents one); a mangling too far from every stated
    name is left unchanged. A slightly relaxed first name (audit 2a) and bare first names (audit 3)
    are accepted — both gated (strong surname / unique first name) so no real person is renamed.
    """
    return _canonicalize_to_known_host(name, stated, first_name_max_edit=2, mononym_ok=True)


def _recover_stated_names(
    by_voice: Dict[str, "SpeakerRole"],
    stated_refs: Sequence[str],
    known_hosts: Sequence[str] = (),
    *,
    fuzzy: bool = True,
) -> None:
    """ADR-130 in-place pass: snap each published name that ASR-mangled a STATED person (host OR
    guest) back to its metadata spelling. Guards that keep it from doing harm:

    ``fuzzy`` is the ADR-140 ``nickname_fuzzy_binding`` knob: this whole pass IS the nickname +
    ASR-fuzzy-surname canonicalizer, so when it is off (naming-3-legacy) the pass is skipped and
    mangled names keep their spoken spelling — the A/B lever for the recovery.

    * A name that ALREADY exactly matches a stated ref is correct and is never re-snapped — else
      two stated people sharing a first name could move an exact match onto the earlier near-ref.
    * A name another voice already holds is never reused (one name, one voice) — EXCEPT when the
      holder is the same person diarization over-split into two voice clusters (audit 2a): both a
      404s "Arietta Laika" and a 32s "Arijeta Lajka" are the one stated guest, so both take the
      canonical spelling rather than leaving the dominant cluster mangled. Two DIFFERENT people are
      still never merged, because the exception only fires when the holder's own name also
      canonicalizes to the same stated ref.
    * A NON-host voice is never snapped onto a KNOWN-HOST's spelling. This preserves the N1 gate
      (`test_guest_with_asr_close_name_does_not_steal_the_host_identity`): a guest self-introducing
      "Kevin Ross" must not be painted as the host "Kevin Roose" just because the host's voice was
      left unnamed — the host-identity canonicalization is gated to host-candidate voices upstream,
      and this final pass must not reopen that hole. A genuinely mangled co-host already carries
      ``role == "host"`` by the time this runs, so it still snaps.
    """
    if not fuzzy:
        return
    stated_lower = {r.lower() for r in stated_refs}
    known_hosts_lower = {h.lower() for h in known_hosts}
    claimed = {r.name.lower() for r in by_voice.values() if r.named}
    for v, role in list(by_voice.items()):
        if not role.named or role.name.lower() in stated_lower:
            continue
        canon = _canonicalize_to_stated_name(role.name, stated_refs)
        if canon == role.name:
            continue
        if canon.lower() in known_hosts_lower and role.role != "host":
            continue
        if canon.lower() in claimed:
            # one-name-one-voice — unless the current holder is the SAME person (its own name also
            # canonicalizes to this stated ref), i.e. a diarization over-split. Then both clusters
            # get the canonical spelling; distinct people are never merged.
            same_person = any(
                r.named
                and r.name.lower() != role.name.lower()
                and r.role == role.role  # an over-split is ONE person -> one role
                and _canonicalize_to_stated_name(r.name, [canon]) == canon
                for r in by_voice.values()
            )
            if not same_person:
                continue
        claimed.discard(role.name.lower())
        claimed.add(canon.lower())
        by_voice[v] = replace(role, name=canon)


def _vouched_by_metadata(candidate: str, metadata_named: Sequence[str]) -> Optional[str]:
    """A weak self-introduction the METADATA can vouch for, resolved to the full stated name.

    "This is Alessio" is the single most common way a host opens a show, and we threw it away: a
    one-token intro could be "I'm American", so the guest path demanded first+last. Correct — and it
    cost us every host who uses their first name. On Latent Space that is *every episode*.

    The metadata settles it. The conversation says which VOICE; the metadata says WHO. A weak intro
    binds only when the episode metadata already states that person, and only when it states exactly
    one such person — an ambiguous first name ("Chris") names nobody, and neither does a show
    talking about itself ("This is Unhedged").
    """
    cand = (candidate or "").strip().lower()
    if len(cand) < 2:
        return None
    hits = {
        n
        for n in metadata_named
        if n.lower() == cand or n.lower().split()[0] == cand  # "Alessio" -> "Alessio Fanelli"
    }
    return hits.pop() if len(hits) == 1 else None


# "This is Alessio." — the other way a host opens a show, and it is NOT a safe self-introduction on
# its own: the same construction is how a show names ITSELF ("This is Unhedged", "This is Planet
# Money"), which is precisely why `extract_self_introduced_host` only ever matched "I'm <Name>".
#
# The metadata dissolves the ambiguity. A "this is X" match binds ONLY when the episode metadata
# states X as a person — whatever its token count, because "This is Latent Space Podcast" is three
# tokens and no more a person than "Unhedged" is.
_THIS_IS_INTRO = re.compile(r"\b[Tt]his is\s+([A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+){0,3})")


# Case-blind intro detectors on match-form text (ADR-139). The capitalization-based regexes in
# hosts.py find nothing on lowercase turbo ASR; these match the SAME cue vocabulary (imported, so
# they cannot drift) on the folded form and are anchored to the stated names — never used to
# discover a name from raw text.
_NAME_WINDOW_MF = r"([a-z][a-z'\-]+(?:\s+[a-z][a-z'\-]+){0,3})"
# "this is" is deliberately EXCLUDED here (unlike the vouched capitalized _THIS_IS_INTRO): on the
# folded form it is a third-person / show-naming hazard — "this is sam altman's company" folds the
# possessive to an edit-1 surname and would bind the WRONG person to the speaker, and "this is <full
# name>" is how a host introduces a guest, not a self-intro. The capitalized self-intro sibling
# (extract_self_introduced_host) is "I'm"-only for exactly this reason; the match form matches it.
_SELF_INTRO_MATCHFORM = re.compile(rf"\b(?:i'm|i am|my name is)\s+{_NAME_WINDOW_MF}")
_CUE_FIRST_MATCHFORM = re.compile(rf"\b(?:{CUE_FIRST_BODY})\s+(?:the\s+|our\s+)?{_NAME_WINDOW_MF}")
# Past-tense cue — gated to head-of-episode + host turns in the loop (recap misattribution, F fix).
_CUE_FIRST_PAST_MATCHFORM = re.compile(
    rf"\b(?:{CUE_FIRST_PAST_BODY})\s+(?:the\s+|our\s+)?{_NAME_WINDOW_MF}"
)
_NAME_FIRST_MATCHFORM = re.compile(rf"{_NAME_WINDOW_MF}\s*,?\s+(?:{NAME_FIRST_TAIL})")
# Report-verb tail — resolved against CORROBORATED refs only (see _voice_named_by_the_introduction).
_NAME_FIRST_REPORT_MATCHFORM = re.compile(rf"{_NAME_WINDOW_MF}\s*,?\s+(?:{NAME_FIRST_REPORT_TAIL})")
_GREETED_MATCHFORM = re.compile(rf"{_NAME_WINDOW_MF}\s*,\s*(?:{GREETED_TAIL})")
# A self-introduction / hand-off is an opening act; the past-tense recap cue and the report-verb
# tails only name the next voice within the first few merged turns of an episode (3rd advisor).
_HEAD_INTRO_TURNS = 10
# A host monologue merges into ONE turn, so the past-tense cue also scans only the first chars of
# that turn (a cold-open hand-off lives in the opening sentences), and rejects a match preceded by a
# temporal recap marker ("last month we spoke with X" is a recap, not an intro). (4th advisor, 2c)
_HEAD_INTRO_CHARS = 1500
_RECAP_MARKER_RE = re.compile(
    r"last\s+(?:week|month|year|night|time)|earlier|previously|recently|yesterday"
    r"|back\s+then|a\s+while\s+ago|the\s+other\s+(?:day|week)"
)


def _stated_tokens(metadata_named: Sequence[str]) -> List[Tuple[str, List[str]]]:
    """``[(stated_name, match-form tokens)]`` for metadata names of ≥2 tokens (a mononym cannot be
    matched by first+surname, so it is excluded from the anchored matchers)."""
    out: List[Tuple[str, List[str]]] = []
    for n in metadata_named or ():
        toks = normalize_name_for_match(n).split()
        if len(toks) >= 2:
            out.append((n, toks))
    return out


# Words that can sit between a first name and its affiliation without being a surname
# ("akshat OF moto", "here WITH akshat AND vibhu") — a token from this set after the first name is
# not a contradicting surname, so the bare-first-name relaxation may still apply. Includes 2-letter
# function words so a genuine SHORT surname (Ng, Wu, Li, Xu — common on an AI-podcast corpus) is not
# mistaken for one (second advisor review): mis-classification then errs toward abstain, not a
# wrong name.
_INTRO_AFFILIATION_TOKENS = frozenset(
    {
        "of",
        "from",
        "at",
        "with",
        "and",
        "the",
        "our",
        "a",
        "an",
        "in",
        "on",
        "for",
        "to",
        "here",
        "as",
        "is",
        "by",
        "or",
        "so",
        "if",
        "up",
        "it",
        "my",
        "me",
        "us",
        "do",
        "go",
        "no",
        "he",
        "we",
    }
)


def _span_has_contradicting_surname(span: Sequence[str]) -> bool:
    """True when the token right after the first name is a purported SURNAME (name-like, not an
    affiliation word). The surname-matching path already ran and matched nothing, so a real surname
    here means the span names a DIFFERENT person who merely shares the first name — "akshat
    kanaparthy" vs stated "Akshat Bubna", "andrew ng" vs stated "Andrew Chen", "rich investors" vs
    "Richard …". Only then is the bare-first-name relaxation refused; a bare first name ("akshat")
    or an affiliation form ("akshat of moto") carries no contradicting surname and still binds.
    A 2-letter token is checked too (Ng/Wu/Li), unless it is a function word. (F2, advisor review)
    """
    return len(span) >= 2 and len(span[1]) >= 2 and span[1] not in _INTRO_AFFILIATION_TOKENS


def _match_stated_in_span(
    span: Sequence[str],
    stated: Sequence[Tuple[str, List[str]]],
    *,
    allow_first_name_only: bool = False,
) -> Optional[str]:
    """The stated name whose first name (nickname/initial-aware) matches ``span[0]`` AND whose
    surname (soundex or edit ≤ 1) matches a later span token; else ``None``. Reference-bounded and
    case-blind — the shared core of every metadata-anchored intro matcher (ADR-139).

    ``allow_first_name_only`` (cue path only) relaxes the surname requirement when the span's first
    token uniquely matches ONE stated name's first name — the flightcast group-intro case ("here
    with akshat" → "Akshat Bubna"). Gated on uniqueness (the same safety ``_vouched_by_metadata``
    uses) and NEVER enabled on the self-intro path, where a colloquial "i'm rich" would false-bind
    "Richard".
    """
    if not span:
        return None
    # Pass 1 — EXACT surname across ALL stated first, so a fuzzy soundex/edit-1 collision can never
    # win over a stated person whose surname the span matches exactly: stated [Chris Smith, Chris
    # Schmidt] + span "chris schmidt" must resolve to Schmidt, not whichever Chris is listed first
    # (Smith/Schmidt share a soundex). Order-independent. (advisor review, finding 4)
    for name, toks in stated:
        if first_names_match(toks[0], span[0]) and toks[-1] in span[1:]:
            return name
    # Pass 2 — fuzzy surname (soundex / edit ≤ 1) only if no stated name matched exactly.
    for name, toks in stated:
        if not first_names_match(toks[0], span[0]):
            continue
        slast = toks[-1]
        if any(_soundex(t) == _soundex(slast) or _edit_distance(t, slast) <= 1 for t in span[1:]):
            return name
    if allow_first_name_only and not _span_has_contradicting_surname(span):
        first_hits = [name for name, toks in stated if first_names_match(toks[0], span[0])]
        if len(first_hits) == 1:
            return first_hits[0]
    return None


def _metadata_anchored_self_intro(
    voice_text: Optional[str], metadata_named: Sequence[str]
) -> Optional[str]:
    """A case-blind self-introduction bound to a STATED name (ADR-139).

    Recovers "i'm rich gelfond" -> metadata "Richard Gelfond" on lowercase turbo ASR, where the
    capitalization-dependent :func:`extract_self_introduced_host` finds nothing. Reference-bounded:
    only ever returns a name in ``metadata_named`` (never invents one), and requires BOTH a
    first-name match (exact / nickname / initial) AND a fuzzy surname match, so it cannot bind
    "i'm american" or paint a shared first name onto the wrong person.
    """
    stated = _stated_tokens(metadata_named)
    if not stated:
        return None
    # Head-bounded like every sibling intro scanner (extract_self_introduced_host, _THIS_IS_INTRO):
    # a self-introduction is an opening act, so a late third-person mention deep in a long turn must
    # not masquerade as one.
    text = normalize_for_match((voice_text or "")[:5000])
    for m in _SELF_INTRO_MATCHFORM.finditer(text):
        # Drop possessive tokens ("altman's", and the s-ending "hastings'"): a trailing-possessive
        # word is a THIRD-PERSON reference — "i'm sam altman's biggest fan", "i'm reed hastings'
        # successor" — never the speaker's OWN surname, so it must not become a surname candidate
        # (the possessive folds to an edit-1/exact surname otherwise). Done before
        # normalize_name_for_match strips the apostrophe. Real apostrophe names ("o'brien") end in a
        # letter, not "'s" or a bare "'", so they are untouched. (F1 residual, 2nd + 3rd advisor.)
        window = " ".join(
            t for t in m.group(1).split() if not (t.endswith("'s") or t.endswith("'"))
        )
        name = _match_stated_in_span(normalize_name_for_match(window).split(), stated)
        if name:
            return name
    return None


def _self_intros_by_voice(
    voice_texts: Optional[Dict[str, str]],
    metadata_named: Sequence[str] = (),
    *,
    case_blind: bool = True,
) -> Dict[str, str]:
    """Per-voice self-introductions ``{voice: name}`` — a voice that says "I'm <First Last>"
    in its *own* turns IS that person. The most reliable per-voice signal, so it names the
    guests/co-hosts that the opening-host self-intro + position-ordered detected-guest list
    miss (the #876 "partial-naming" case: "Hi, I'm Nic Harrigan" rendering as SPEAKER_1).

    Requires a first+last name (≥2 tokens) — guarding the guest path against "I'm American"-style
    false positives; the single main host is still covered by the opening-intro pool. A ONE-token
    "I'm <X>", and ANY "this is <X>", are accepted only when ``metadata_named`` vouches for them —
    which is what rescues "This is Alessio" without admitting "I'm American" or "This is Unhedged".
    """
    out: Dict[str, str] = {}
    for voice, text in (voice_texts or {}).items():
        head = (text or "")[:5000]
        name = extract_self_introduced_host(text, intro_chars=5000)
        if name and len(name.split()) >= 2:
            out[voice] = name
            continue
        # A single-token self-intro on the voice's OWN turns ("I'm Brandon", "I'm Neeraj") names it,
        # provided the token is a plausible mononym and not the "I'm American" class — the guard the
        # ≥2-token rule used to enforce, now sharpened so no-anchor feeds don't lose real speakers.
        if name and is_plausible_mononym(name):
            out[voice] = name
            continue
        # A bare first name we couldn't vouch, or a "this is <X>" — neither stands alone. Metadata.
        candidates = [name] if name else []
        candidates += [m.group(1).strip(" .,") for m in _THIS_IS_INTRO.finditer(head)]
        for cand in candidates:
            stated = _vouched_by_metadata(cand, metadata_named)
            if stated:
                out[voice] = stated
                break
        # Case-blind fallback (ADR-139): the capitalization-based paths above find nothing on
        # lowercase turbo ASR. Match the self-intro on the folded form, anchored to a stated name.
        if case_blind and voice not in out:
            stated = _metadata_anchored_self_intro(text, metadata_named)
            if stated:
                out[voice] = stated
    return out


def _name_host_voices(
    host_voices: Sequence[str],
    host_pool: Sequence[Tuple[str, str]],
    voice_intro: Dict[str, str],
    used_lower: set,
    llm_named: Optional[set] = None,
) -> Dict[str, SpeakerRole]:
    """Name host voices: own self-introduction first, else the ordered host-name pool."""
    out: Dict[str, SpeakerRole] = {}
    hi = 0
    for v in host_voices:
        iname = voice_intro.get(v)
        if iname and iname.lower() not in used_lower:
            used_lower.add(iname.lower())
            src = "llm_resolution" if (llm_named and v in llm_named) else "self_intro"
            out[v] = SpeakerRole(name=iname, role="host", named=True, source=src)
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
    talk: Optional[Dict[str, float]] = None,
    llm_named: Optional[set] = None,
    cameo_max_talk_s: float = CAMEO_MAX_TALK_S,
) -> Dict[str, SpeakerRole]:
    """Name the remaining voices from EVIDENCE, never from position.

    A voice is named by its own self-introduction, by an on-air introduction, or by the post-
    diarization resolution (all three arrive in ``voice_intro``) — and otherwise only when the match
    is FORCED: one name left, one voice left, therefore no choice and therefore no guess.

    What used to happen instead was ``guest_names[gi]``: hand the detected names out in TALK-TIME
    ORDER. That is the invention mechanism behind every wrong name we have shipped. Nothing tied the
    name to the voice; the second-loudest speaker simply got the second name. Caught in the act on
    FT Unhedged, where it painted Robert Armstrong onto the wrong voice and put Katie Martin — the
    show's lead host — on a voice with 4% of the talk (ADR-110).

    An unnamed voice costs us a `SPEAKER_01`. A misnamed one puts words in a real person's mouth.
    """
    out: Dict[str, SpeakerRole] = {}
    # A CAMEO IS NOT A CANDIDATE. Counting a two-second "Yeah." as a voice that might be the guest
    # turns "one name, one voice" into "one name, two voices" and the forced match declines — which
    # is how removing positional painting first cost the NVIDIA guest her name, on an episode whose
    # host says "Jia Li is with us today" out loud.
    unassigned = [
        v
        for v in voices_by_total
        if v not in assigned
        and v not in voice_intro
        and (talk is None or talk.get(v, 0.0) >= cameo_max_talk_s)
    ]
    # A detected-guest name is only spare if the SAME PERSON is not already on the roster. Same
    # person = same surname AND (one side title-only, or matching given name, or one given an
    # initial of the other). A guest the interviewer named "Professor Pape" is the metadata's
    # "Robert Pape" (title-only); a community-1-split guest named "Adam Rodman" on one cluster is
    # the detected "Dr. Adam Rodman" (given-name match) — neither is spare, so the forced path
    # abstains instead of fabricating a second Pape/Rodman (#876/#1330). But a distinct guest who
    # merely shares a surname ("Robert Pape" vs "Karen Pape") IS spare — different givens.
    roster_names = [*used_lower, *voice_intro.values()]
    spare = [
        g
        for g in guest_names
        if g.lower() not in used_lower and not any(_same_person(g, r) for r in roster_names)
    ]
    # One name, one voice: the assignment is forced, so it is not a guess.
    forced = spare[0] if (len(spare) == 1 and len(unassigned) == 1) else None
    # The cameo floor exists to avoid painting a name onto a brief cutaway when
    # OTHER voices compete for it — not to drop the ONLY remaining candidate. If
    # exactly one name and one non-host/non-intro voice remain (ignoring the floor),
    # that voice IS the guest however briefly it spoke, so force it. Without this a
    # short two-voice episode leaves the guest as SPEAKER_NN.
    if forced is None and len(spare) == 1:
        remaining = [v for v in voices_by_total if v not in assigned and v not in voice_intro]
        if len(remaining) == 1:
            forced = spare[0]
            unassigned = remaining

    for v in voices_by_total:
        if v in assigned:
            continue
        iname = voice_intro.get(v)
        if iname and iname.lower() not in used_lower and iname.lower() not in host_names_lower:
            used_lower.add(iname.lower())
            src = "llm_resolution" if (llm_named and v in llm_named) else "self_intro"
            out[v] = SpeakerRole(name=iname, role="guest", named=True, source=src)
        elif forced is not None and v == unassigned[0]:
            used_lower.add(forced.lower())
            out[v] = SpeakerRole(name=forced, role="guest", named=True, source="forced")
        else:
            # Paint a leftover unnamed voice as "guest" only with positive GUEST
            # evidence: detected guest names, or a self-intro from a NON-host voice.
            # ``voice_intro`` is episode-wide and includes the host's own intro, so a
            # host-only-intro show must leave leftovers "unknown", not "guest" (#1170).
            has_guest_intro = any(vid not in assigned for vid in voice_intro)
            role = "guest" if (guest_names or has_guest_intro) else "unknown"
            out[v] = SpeakerRole(name=v, role=role, named=False, source="raw")
    return out


def _intro_names(m: "re.Match[str]") -> List[str]:
    """The person-names an introduction/greeting match named (order preserved).

    Filtered through the same ``looks_like_a_person_name`` guard the self-intro path uses: a
    capitalised run with an ordinary English word in it ("So Nick", "But Sun") is ASR noise, not a
    name, and a wrong label is worse than an unnamed voice.
    """
    return [
        n
        for n in (_clean_intro_name(x) for x in _INTRO_NAME_RE.findall(m.group("names")))
        # `looks_like_a_person_name` passes a capitalised org ("New York Times"), which the greedy
        # capture picks up from "from the New York Times, I'm…". An org is never the introduced
        # PERSON, and binding it to the guest voice buries the real name — reject it (ADR-139).
        if n and looks_like_a_person_name(n) and not is_network_or_org_author(n)
    ]


def _greeted_names(text: str) -> List[str]:
    """Names in a NAME-ANCHORED greeting/introduction ("Kara Swisher, welcome" / "X is with us").

    Deliberately excludes the cue-first form (already handled) and never the loose show-structure
    cues ("when we come back") — those are not name-anchored and reclaiming on them corrupts real
    guest speech (a wrong label is worse than an unnamed voice).
    """
    names: List[str] = []
    for rx in (_GUEST_GREETED_RE, _GUEST_INTRODUCED_NAME_FIRST_RE):
        for m in rx.finditer(text or ""):
            names += _intro_names(m)
    return names


def _nearest_host_voice(
    turns: Sequence[Tuple[str, str]], i: int, host_hint_voices: Set[str]
) -> Optional[str]:
    """The host-hint voice nearest turn ``i`` (preceding preferred), or ``None`` if none is."""
    for d in range(1, len(turns)):
        for j in (i - d, i + d):
            if 0 <= j < len(turns) and turns[j][0] in host_hint_voices:
                return turns[j][0]
    return None


def _reclaim_greeting_turns(
    ordered_turns: Sequence[Tuple[str, str]],
    host_hint_voices: Set[str],
    known_hosts: Sequence[str],
) -> List[Tuple[str, str]]:
    """Move a host's name-anchored greeting off a guest cluster it was mis-merged into (#1226 fu).

    Diarization (community-1 especially) merges the host's "Kara Swisher, welcome back" turn into
    the GUEST's own voice cluster. The greeting then reads as the guest naming themselves in the
    third person — the resolver's third-person guard refuses it, and the introduction reader, seeing
    the guest as the introducer, would name whoever speaks NEXT (a host). Reclaiming the greeting to
    a host cluster restores both: the guard is never involved (deterministic naming), and the reader
    names the greeted guest.

    Conservative by construction: only NAME-ANCHORED greeting/introduction turns, only when the
    turn sits on a NON-host cluster, only when the greeted name is not itself a stated host, and
    only when a host destination is determinable. Anything ambiguous is left untouched.
    """
    if not host_hint_voices:
        return list(ordered_turns)
    known_lower = {h.lower().strip() for h in (known_hosts or ())}
    out = list(ordered_turns)
    for i, (speaker, text) in enumerate(out):
        if speaker in host_hint_voices:
            continue  # greeting already on a host cluster — nothing to reclaim
        greeted = _greeted_names(text)
        if not greeted:
            continue
        # A stated host being "greeted" is a co-host intro / self-reference, not contamination.
        if all(n.lower() in known_lower for n in greeted):
            continue
        dest = _nearest_host_voice(out, i, host_hint_voices)
        if dest is not None:
            out[i] = (dest, text)
    return out


def _past_cue_head_name(
    text: str, stated: Sequence[Tuple[str, List[str]]], first_name_only: bool
) -> Optional[str]:
    """The stated name a past-tense hand-off cue ("we spoke with X") introduces in the OPENING of a
    turn, else None. Text-head-bounded (a cold-open lives in the first sentences) and recap-marker
    rejecting, so a mid-monologue recap in a long merged turn does not misattribute. (4th advisor,
    2c) — the turn-index head bound alone is trivially met when a host monologue merges to turn 0.
    """
    head_mf = normalize_for_match((text or "")[:_HEAD_INTRO_CHARS])
    for m in _CUE_FIRST_PAST_MATCHFORM.finditer(head_mf):
        if _RECAP_MARKER_RE.search(head_mf[max(0, m.start() - 40) : m.start()]):
            continue
        nm = _match_stated_in_span(
            normalize_name_for_match(m.group(1)).split(),
            stated,
            allow_first_name_only=first_name_only,
        )
        if nm:
            return nm
    return None


def _voice_named_by_the_introduction(
    ordered_turns: Optional[Sequence[Tuple[str, str]]],
    host_hint_voices: Optional[Set[str]] = None,
    conv_hosts: Optional[Set[str]] = None,
    known_hosts_lower: AbstractSet[str] = frozenset(),
    metadata_named: Sequence[str] = (),
    first_name_only: bool = True,
    corroborated_named: Sequence[str] = (),
) -> Dict[str, str]:
    """``{voice: name}`` for a voice the HOST introduced by name — "and now, Bobby Allen".

    The person a host introduces is the person who SPEAKS NEXT. That is conversation structure, and
    it is the only per-voice way to use an introduction: knowing that "Bobby Allen" was named
    somewhere in the episode does not say WHICH cluster he is, and handing introduced names out by
    talk order is just the talk-share mistake wearing a different hat.

    Worth 5% of the corpus's talk. Planet Money is a narrated desk that hands off constantly
    ("joined by...", "here with me is..."), and every one of those reporters came out as SPEAKER_NN.

    The cue-first form ("joined by X") is a host act on its own and is read unconditionally. The
    weaker name-first ("X is with us") and greeting ("X, welcome back") forms — a guest can utter
    a name-first sentence — name the next voice ONLY when ``host_hint_voices`` says the introducer
    is a plausible host. Given ``None`` those two forms are skipped, preserving prior behaviour.

    Only the FIRST introduction of a voice is used, and a name already claimed by another voice is
    never reused — under-naming beats naming the wrong person (#876).
    """
    if not ordered_turns:
        return {}

    # ASR segments are 14-50 characters — a fragment of a sentence. An introduction ("we spoke with
    # assistant managing editor Patrick Healy, who oversees...") spans several of them, so a regex
    # applied per SEGMENT sees only fragments and matches nothing. Merge consecutive segments by the
    # same speaker into one utterance first: that is what a conversational "turn" actually is.
    merged: List[Tuple[str, str]] = []
    for speaker, text in ordered_turns:
        if merged and merged[-1][0] == speaker:
            merged[-1] = (speaker, merged[-1][1] + " " + (text or ""))
        else:
            merged.append((speaker, text or ""))
    ordered_turns = merged

    out: Dict[str, str] = {}
    taken: set = set()
    host_voices = (host_hint_voices or set()) | (conv_hosts or set())

    def _assign(i: int, names: List[str], *, host_name_requires_host_target: bool = False) -> None:
        # whoever speaks next, that is who was just introduced
        introducer = ordered_turns[i][0]
        name = names[0]
        # The introduced GUEST is not one of the show's hosts. When the host names the guest and the
        # CO-HOST banters back before the guest actually answers ("Adam Rodman, welcome" → co-host:
        # "great to have you" → guest), the next voice is a host, and naming it paints the guest's
        # name on the host AND blocks the host's own name from the pool (#1169). Skip a host as
        # the target — UNLESS the introduction is OF a stated host ("welcome back, my co-host Kevin
        # Roose"), which legitimately names a host voice.
        name_is_host = name.lower() in known_hosts_lower
        for j in range(i + 1, min(i + 6, len(ordered_turns))):
            nxt = ordered_turns[j][0]
            if nxt == introducer or nxt in out:
                continue
            if nxt in host_voices and not name_is_host:
                continue
            # v1 (4th advisor): on the report-verb path a HOST name is usually a TOPICAL mention
            # ("kevin roose explains in his book") — bind it only to a host VOICE, never paint an
            # absent co-host's name onto a guest. The legit co-host desk hand-off still binds when
            # the next voice is a host.
            if host_name_requires_host_target and name_is_host and nxt not in host_voices:
                continue
            if name.lower() in taken:
                return
            out[nxt] = name
            taken.add(name.lower())
            return

    stated = _stated_tokens(metadata_named)
    # Corroborated refs (detected guests + known hosts) for the report-verb tails — those tails also
    # match a bare TOPICAL mention ("sam altman explains it best"), so binding them to a metadata
    # SUBJECT is the #876 error. NO fallback to the full stated set: when nothing is corroborated,
    # a report-verb tail names nobody (abstain beats a wrong bind). (3rd advisor review, fix 3)
    corroborated_stated = _stated_tokens(corroborated_named)

    def _assign_matchform(
        i: int,
        mf_text: str,
        rx: "re.Pattern[str]",
        *,
        allow_first_name_only: bool = False,
        stated_set: Optional[Sequence[Tuple[str, List[str]]]] = None,
        report_path: bool = False,
    ) -> None:
        # Case-blind, metadata-anchored (ADR-139): the capitalized regexes above find nothing on
        # lowercase turbo ASR, so match the SAME cue on the folded form and resolve the window to a
        # stated name. Reference-bounded — only ever assigns a name the metadata stated.
        refs = stated if stated_set is None else stated_set
        for m in rx.finditer(mf_text):
            name = _match_stated_in_span(
                normalize_name_for_match(m.group(1)).split(),
                refs,
                allow_first_name_only=allow_first_name_only,
            )
            if name:
                _assign(i, [name], host_name_requires_host_target=report_path)

    for i, (speaker, text) in enumerate(ordered_turns):
        is_host_hint = host_hint_voices is not None and speaker in host_hint_voices
        at_head = i < _HEAD_INTRO_TURNS
        mf = normalize_for_match(text or "") if stated else ""
        for m in _GUEST_INTRODUCED_BY_HOST_RE.finditer(text or ""):
            names = _intro_names(m)
            if names:
                _assign(i, names)
        if stated:
            # The bare-first-name relaxation is trusted ONLY from a host introducer's turn — a
            # random voice saying "…with rich investors" must not paint a stated Richard onto the
            # next speaker (F2, advisor review). Surname-anchored cue matches still run on any turn.
            _assign_matchform(
                i, mf, _CUE_FIRST_MATCHFORM, allow_first_name_only=first_name_only and is_host_hint
            )
            # Past-tense recap cue ("we spoke with X"): only a head-of-episode cold-open from a host
            # (fix 2 + 2c); the helper handles the text-head bound + recap-marker rejection.
            if is_host_hint and at_head:
                past_nm = _past_cue_head_name(text, stated, first_name_only)
                if past_nm:
                    _assign(i, [past_nm])
        if is_host_hint:
            for rx in (_GUEST_GREETED_RE, _GUEST_INTRODUCED_NAME_FIRST_RE):
                for m in rx.finditer(text or ""):
                    names = _intro_names(m)
                    if names:
                        _assign(i, names)
            if stated:
                _assign_matchform(i, mf, _GREETED_MATCHFORM)
                _assign_matchform(i, mf, _NAME_FIRST_MATCHFORM)
                # Report-verb tails ("X explains/reports") resolve ONLY against corroborated refs,
                # and a HOST name among them binds only a host voice (v1, 4th advisor).
                _assign_matchform(
                    i,
                    mf,
                    _NAME_FIRST_REPORT_MATCHFORM,
                    stated_set=corroborated_stated,
                    report_path=True,
                )
    return out


def _distinct_intros_map_to_multiple_stated(text: str, stated: Sequence[str]) -> bool:
    """True when a cluster's DISTINCT self-intros map (fuzzily) to 2+ different STATED people.

    That is a diarization MERGE of multiple named speakers — "…take turns introducing yourselves.
    I'm Lucas and I'm Axel." landing in the host's cluster (flightcast). Naming the cluster from the
    first self-intro paints a guest's name onto the wrong voice; this flags it so the name is
    suppressed instead. First-name match tolerates the ASR spelling ("Lucas" vs stated "Lukas").
    """
    distinct = distinct_self_introductions(text, intro_chars=5000)
    if len(distinct) < 2:
        return False
    stated_firsts = [(s, _given_tokens(s)[0].lower()) for s in stated if _given_tokens(s)]
    matched: Set[str] = set()
    for nm in distinct:
        toks = _given_tokens(nm)
        if not toks:
            continue
        nf = toks[0].lower()
        for s, sf in stated_firsts:
            if first_names_match(sf, nf) or _edit_distance(sf, nf) <= 1:
                matched.add(s.lower())
                break
        if len(matched) >= 2:
            return True
    return False


def _self_intro_voice_names(
    diarization: DiarizationResult,
    voice_texts: Optional[Dict[str, str]],
    intro_sources: Sequence[str],
    known_hosts: Sequence[str],
    ad_voices: Set[str],
    conv_guests: AbstractSet[str] = frozenset(),
    strategy: Optional[DiarizationLabelingStrategy] = None,
    case_blind: bool = True,
    suppress_merged: bool = True,
    cameo_max_talk_s: float = CAMEO_MAX_TALK_S,
) -> Dict[str, str]:
    """``{voice: name}`` from each voice's OWN self-introduction, ads excluded (#876).

    A garbled self-intro is snapped onto a configured host ONLY for the host-candidate voices; who
    those are is a cluster-shape question, so the ``strategy`` (ADR-134) decides. Applied to every
    voice it swaps identities: a guest self-introducing with a name ASR-close to a host's ("I'm
    Kevin Ross" vs host "Kevin Roose") was snapped onto the host (N1) — the candidate gate is what
    prevents that, so it stays load-bearing under every strategy.

    A SHORT cold-open montage that strings several hosts' garbled self-intros into one cluster ("I'm
    Kevin Russo… I'm Casey Noon…", 13s) is not a person and is suppressed (#1330); a LONG voice with
    the same double self-intro is a real dominant speaker and keeps its name.
    """
    strategy = strategy or _DEEPGRAM
    first_start: Dict[str, float] = {}
    talk: Dict[str, float] = {}
    for s in diarization.segments:
        if s.speaker in ad_voices:
            continue
        if s.speaker not in first_start or s.start < first_start[s.speaker]:
            first_start[s.speaker] = s.start
        talk[s.speaker] = talk.get(s.speaker, 0.0) + (s.end - s.start)
    texts = voice_texts or {}
    intros = _self_intros_by_voice(voice_texts, intro_sources, case_blind=case_blind)
    # A SHORT cluster with 2+ self-intros is a cold-open montage clip (#1330). A LONG one with 2+
    # self-intros that map to DIFFERENT stated people is a diarization MERGE of multiple named
    # speakers (flightcast "I'm Lucas and I'm Axel" in the host cluster) — also suppressed, so the
    # first name is not painted onto the wrong voice. The merge case is gated (ADR-140).
    montage_suppressed = {
        v
        for v in intros
        if (
            talk.get(v, 0.0) < MONTAGE_CLIP_MAX_TALK_S
            and len(distinct_self_introductions(texts.get(v, ""), intro_chars=5000)) >= 2
        )
        or (
            suppress_merged
            and _distinct_intros_map_to_multiple_stated(texts.get(v, ""), intro_sources)
        )
    }
    host_candidate_voices = strategy.host_candidate_voices(
        first_start=first_start,
        talk=talk,
        known_hosts=known_hosts,
        conv_guests=set(conv_guests),
        montage_suppressed=montage_suppressed,
        cameo_floor=cameo_max_talk_s,
    )
    out: Dict[str, str] = {}
    for v, n in intros.items():
        if v in ad_voices or v in montage_suppressed:
            continue
        if v not in host_candidate_voices:
            out[v] = n
            continue
        canon = _canonicalize_to_known_host(n, known_hosts)
        # Inside the host gate: if the surname canonicalization did not fire, the strategy may still
        # resolve a garbled host name (community-1's unique-first-name snap for "Casey Noonan").
        out[v] = canon if canon != n else (strategy.snap_extra(n, known_hosts) or n)
    return out


def _intro_reader_voice_names(
    reclaimed_turns: Sequence[Tuple[str, str]],
    host_hint_voices: Set[str],
    voice_intro: Dict[str, str],
    ad_voices: Set[str],
    known_hosts: Sequence[str],
    conv_hosts: Optional[Set[str]] = None,
    stated_persons: Sequence[str] = (),
    narrator_cue: bool = True,
    first_name_only: bool = True,
    corroborated_persons: Sequence[str] = (),
) -> Dict[str, str]:
    """``{voice: canonical name}`` for voices a host introduced by name — "and now, Bobby Allen" —
    since the person a host introduces is the one who speaks next. Complements the self-intro:
    plenty of guests never say their own name and are named FOR them; a voice that already named
    itself keeps its own word for it (skipped here).

    The introduced person is a guest unless the introduction names a stated host, so a host voice is
    not a valid target (``conv_hosts``/``known_hosts`` gate that in :func:`_voice_named_by_the_
    introduction`). A title-form greeting ("Professor Pape, thanks for coming") is upgraded to the
    stated full name of the same person when the host-ward snap does not fire (#1169).
    """
    out: Dict[str, str] = {}
    known_lower = {h.lower() for h in known_hosts}
    conv_host_set = conv_hosts or set()
    for v, n in _voice_named_by_the_introduction(
        reclaimed_turns,
        host_hint_voices,
        conv_hosts,
        known_lower,
        metadata_named=stated_persons if narrator_cue else (),
        first_name_only=first_name_only,
        corroborated_named=corroborated_persons if narrator_cue else (),
    ).items():
        if v in ad_voices or v in voice_intro:
            continue
        # F3 (advisor): the host-ward snap (nickname-as-exact + surname edit ≤ 3) can rename a GUEST
        # whose introduced name merely shares a nickname with a known host ("Rich Perkins" → host
        # "Richard Parker"). Only snap toward a known host when the target voice is itself a host
        # voice — the same guard _recover_stated_names applies (a non-host voice is never given a
        # known host's spelling). A guest still canonicalizes to the stated PERSON (its own name).
        canon = _canonicalize_to_known_host(n, known_hosts) if v in conv_host_set else n
        out[v] = canon if canon != n else _canonicalize_to_stated_person(n, stated_persons)
    return out


def _select_host_voices(
    *,
    diarization: DiarizationResult,
    voice_intro: Dict[str, str],
    host_pool: Sequence[Tuple[str, str]],
    known_hosts: Sequence[str],
    conv_hosts: Sequence[str],
    conv_guests: AbstractSet[str],
    voices_by_intro: Sequence[str],
    llm_named: AbstractSet[str],
    llm_voice_roles: Optional[Dict[str, str]],
    content_start: float,
    intro_window_s: float,
    ad_intervals: Optional[Sequence[Tuple[float, float]]],
    ad_voices: AbstractSet[str],
) -> List[str]:
    """WHICH diarized voices are the hosts — the cross-reference of metadata (who / how many) and
    the conversation (which voice performs the role). Five ordered signals, strongest first."""
    # A voice that SAYS a name NOT in the feed's host pool is positive evidence it is NOT a stated
    # host, so it must never fill a vacant host seat (No Priors → Andy Fang over absent Sarah Guo;
    # Unhedged → Joshua Franklin over absent Rob Armstrong). Only when the feed STATED hosts;
    # `llm_named` voices are excluded because their name was INFERRED, not said aloud.
    host_pool_lower = {n.lower() for n, _ in host_pool}
    stated_non_host_voices = {
        v
        for v, n in voice_intro.items()
        if host_pool and n and n.lower() not in host_pool_lower and v not in llm_named
    }
    # ADR-137 — the LLM's host/guest verdict as BOUNDED advice: it may DEMOTE a positional host
    # guess (a voice it calls "guest" is blocked from the opener/intro-fill steps) and ANCHOR a
    # no-host show (a voice it calls "host" may seat when the pool is empty). It never unseats a
    # self-intro'd known host (step 1) nor overrides a performed host (step 2).
    llm_roles = llm_voice_roles or {}
    llm_guest_voices = {v for v, r in llm_roles.items() if r == "guest"}
    llm_host_voices = {v for v, r in llm_roles.items() if r == "host"}
    positional_non_host = stated_non_host_voices | llm_guest_voices

    host_voices: List[str] = []
    known_lower = {h.lower() for h in known_hosts}

    # 1. A voice that introduces itself as one of the feed's STATED hosts IS that host (the
    #    strongest cross-reference). A stated name is claimed once — a merged cluster carrying the
    #    host's intro verbatim must not claim it twice as a third host on a two-host show (#1226).
    claimed_host_names: set = set()
    for v, n in voice_intro.items():
        nl = n.lower()
        if (
            nl in known_lower
            and nl not in claimed_host_names
            and v not in host_voices
            and v not in conv_guests
        ):
            host_voices.append(v)
            claimed_host_names.add(nl)

    # 2. A voice that PERFORMS the host's role is a host — but the feed says how MANY, so a stated
    #    count is binding (a third voice cannot host a two-host show). Uncapped when the feed names
    #    no host. The deterministic-only guard here; the LLM never overrides a performed host.
    cap = len(host_pool) if host_pool else None
    for v in conv_hosts:
        if cap is not None and len(host_voices) >= cap:
            break
        if v not in host_voices and v not in stated_non_host_voices:
            host_voices.append(v)

    # 3. The opener does the intro (the pre-roll ad is excluded). Skipped when the conversation
    #    already named a host, and never a voice the conversation heard say "thanks for having me".
    opener = _opening_voice(
        diarization,
        window_end=content_start + intro_window_s,
        ad_intervals=ad_intervals,
        ad_voices=set(ad_voices),
    )
    if opener is None and voices_by_intro:
        opener = voices_by_intro[0]
    if (
        opener is not None
        and not conv_hosts
        and opener not in host_voices
        and opener not in conv_guests
        and opener not in positional_non_host
        and len(host_voices) < max(1, len(host_pool))
    ):
        host_voices.append(opener)

    # 4. Fill any host slot the feed COUNTED but we have not matched, from the SHOW's intro voices
    #    (ads excluded). NOT by talk time — that hands a slot to a long-answering guest (#1169).
    for v in voices_by_intro:
        if len(host_voices) >= len(host_pool):
            break
        if v not in host_voices and v not in conv_guests and v not in positional_non_host:
            host_voices.append(v)

    # 5. ANCHOR a show that STATES NO HOSTS (ADR-137). With an empty pool the cues have no anchor,
    #    so a rotating/narrator host (Planet Money) is labeled a guest. A voice the LLM judged host
    #    may seat here — but ONLY if it is NAMED (in `voice_intro`): a no-stated-host show is a
    #    narrated documentary as often as a rotating-host desk show, and the LLM will call the field
    #    tape "host" too. Seating a host we can NAME (the narrator who self-introduced) is safe;
    #    anchoring an anonymous voice is exactly the vox-pop-as-host over-assignment. Not a heard
    #    guest, not an ad.
    if not host_pool:
        for v in llm_host_voices:
            if (
                v not in host_voices
                and v not in conv_guests
                and v not in ad_voices
                and v in voice_intro
            ):
                host_voices.append(v)

    return host_voices


def resolve_speaker_roster(
    diarization: DiarizationResult,
    transcript_text: Optional[str],
    *,
    host_candidates: Sequence[str] = (),
    detected_guests: Sequence[str] = (),
    known_hosts: Sequence[str] = (),
    voice_texts: Optional[Dict[str, str]] = None,
    ordered_turns: Optional[Sequence[Tuple[str, str]]] = None,
    ad_intervals: Optional[Sequence[Tuple[float, float]]] = None,
    metadata_named: Sequence[str] = (),
    llm_voice_names: Optional[Dict[str, str]] = None,
    llm_voice_roles: Optional[Dict[str, str]] = None,
    cleaning: Optional[VoiceCleaning] = None,
    recurring_text: Optional[set] = None,
    intro_window_s: float = INTRO_WINDOW_SECONDS,
    diarization_provider: Optional[str] = None,
    profile: LabelingProfile = DEFAULT_LABELING_PROFILE,
) -> SpeakerRoster:
    """Resolve every diarized voice to a ``SpeakerRole`` (see module docstring).

    ``voice_texts`` maps each diarized voice id to the concatenation of *its own* turns; when
    supplied it lets a voice be named from its own self-introduction (#876). Omitted → the
    previous host-pool + ordered-guest behaviour (fully backward-compatible).

    ``ad_intervals`` (``(start_s, end_s)`` ad regions) lets an unnamed voice that speaks mostly
    inside ads be typed ``commercial``; omitted → only cameo vs unknown by talk time.

    ``metadata_named`` is every person the episode METADATA states, *before* corroboration filters
    them. It never names a voice on its own — it only decides whether an unnamed voice is a defect
    (``unknown``) or a person nobody could have named (``unidentified``). A name we saw and could
    not place is a failure of ours, whatever our reason for declining to use it.

    ``recurring_text`` is the script this FEED repeats across its own episodes (#1188). A voice that
    is mostly reading it, and barely speaks, is a recording — the mid-roll house ad the edge rule
    cannot see. Omitted → only the edge rule, as before.
    """
    if not diarization.segments:
        return SpeakerRoster(by_voice={}, num_speakers=diarization.num_speakers or 0)

    # Ad voices are established BEFORE anything can be named from them: the pre-roll opens the
    # episode and reads its own name, so it wins both the "opening voice = host" rule and the
    # most-trusted self-introduction rule unless it is removed from contention up front.
    # Ad voices come from the shared cleaning classifier (ADR-137) when the caller computed it, so
    # "which voices are ads" is defined in ONE place so the LLM call and roster cannot disagree.
    # Standalone callers (tests, relabel-only) pass no cleaning and get the identical set from the
    # same helper — the classifier is a factoring of exactly this logic, not a behaviour change
    # (edge ads + the cross-episode recurring house-ad the edge rule misses, #1188/ADR-134).
    ad_voices = (
        set(cleaning.ad)
        if cleaning is not None
        else _ad_voices_for(
            diarization, ordered_turns, voice_texts, recurring_text, diarization_provider
        )
    )

    # The intro window is the SHOW's intro, not the advert's. Measured from 0 it was mostly ad, so a
    # co-host who speaks a minute in barely registered and never cleared CO_HOST_INTRO_SHARE — which
    # left Kevin Roose, who talks for 39% of the episode, outside `host_voices` entirely.
    content_start = min(
        (s.start for s in diarization.segments if s.speaker not in ad_voices),
        default=0.0,
    )
    intro = _talk_time(
        diarization, window_start=content_start, window_end=content_start + intro_window_s
    )
    total = _talk_time(diarization)
    voices_by_total = [v for v in sorted(total, key=lambda v: total[v], reverse=True)]
    voices_by_intro = [
        v for v in sorted(intro, key=lambda v: intro[v], reverse=True) if v not in ad_voices
    ]

    # A voice that introduces itself in its own turns is named from that, most-trusted (#876) —
    # but not if it is an ad. An ad narrator reads its own name aloud by design, which is precisely
    # what makes the most-trusted signal the easiest one to poison.
    # ...and the name it gives is the ASR's spelling, so it is snapped onto the configured host
    # when it is plainly the same person ("Kevin Russo" / "Kevin Roos" -> "Kevin Roose").
    # The metadata is handed in so a bare "This is Alessio" can be vouched for; it never names a
    # voice on its own.
    intro_sources = (
        list(metadata_named or ()) + list(known_hosts or ()) + list(detected_guests or ())
    )
    # Conversation-performed roles, computed once and reused for host selection below. A voice that
    # says "thanks for having me" is a guest even if it speaks first.
    conv_roles = roles_from_conversation(voice_texts)
    conv_guests = {v for v, r in conv_roles.items() if r == "guest"}
    conv_host_voices = {v for v, r in conv_roles.items() if r == "host"}
    # A voice that IDENTIFIES ITSELF as one of the feed's stated hosts is a host, even if a guest
    # speech act also appears in its cluster. community-1 merged an ad testimonial ("...thank you so
    # much for having me") into a co-host's cluster; that flipped him to a conv_guest, dropped his
    # host-candidacy, and his ASR-mangled self-intro ("Kevin Russo") then never canonicalized to the
    # stated host ("Kevin Roose"). Self-identifying as a stated host is the stronger signal (#1169).
    _stated_host_set = set(known_hosts)
    _names_self_intro = _self_intros_by_voice(voice_texts, intro_sources)
    conv_guests -= {
        v
        for v, n in _names_self_intro.items()
        if _canonicalize_to_known_host(n, known_hosts) in _stated_host_set
    }

    # A voice that introduces itself in its own turns is named from that, most-trusted (#876) —
    # but not if it is an ad. An ad narrator reads its own name aloud by design, which is precisely
    # what makes the most-trusted signal the easiest one to poison.
    # ...and the name it gives is the ASR's spelling, so it is snapped onto the configured host
    # when it is plainly the same person ("Kevin Russo" / "Kevin Roos" -> "Kevin Roose").
    _strategy = labeling_strategy_for(diarization_provider)
    voice_intro = _self_intro_voice_names(
        diarization,
        voice_texts,
        intro_sources,
        known_hosts,
        ad_voices,
        conv_guests,
        _strategy,
        case_blind=profile.case_blind_self_intro,
        suppress_merged=profile.suppress_merged_speaker_clusters,
        cameo_max_talk_s=profile.cameo_max_talk_s,
    )

    # WHICH voices can plausibly be hosts, for the introduction reader's gate and the greeting
    # reclamation (#1226 follow-up): a voice that self-introduced as a STATED host, plus the first
    # non-ad speaker (the opener does the intro). Kept deliberately small — it only decides who is
    # allowed to *introduce*, never who gets named.
    known_lower_hint = {h.lower() for h in known_hosts}
    host_hint_voices: Set[str] = {
        v for v, n in voice_intro.items() if n and n.lower() in known_lower_hint
    }
    # The opener usually IS a host — but not if the conversation heard it perform the guest role
    # (a cold-open guest clip before the host's welcome). Trusting such an opener as a host hint let
    # the reclamation move a greeting onto the guest and the weaker intro forms name from it (N5).
    for spk, _t in ordered_turns or []:
        if spk not in ad_voices:
            if spk not in conv_guests:
                host_hint_voices.add(spk)
            break

    # A host's name-anchored greeting mis-merged into the guest's cluster is moved back to a host,
    # so the guest is named from it deterministically rather than refused by the third-person guard.
    reclaimed_turns = _reclaim_greeting_turns(ordered_turns or [], host_hint_voices, known_hosts)

    # A voice the HOST introduced by name is that person (the introduced person speaks next),
    # guarded against the interview-CLOSE case where the next voice is the host resuming (R3/#876).
    voice_intro.update(
        _intro_reader_voice_names(
            reclaimed_turns,
            host_hint_voices,
            voice_intro,
            ad_voices,
            known_hosts,
            conv_host_voices,
            list(detected_guests or ()) + list(metadata_named or ()),
            narrator_cue=profile.narrator_cue_binding,
            first_name_only=profile.first_name_only_intro,
            # Report-verb tails resolve only against CORROBORATED names (detected guests + known
            # hosts), never a bare metadata subject like an episode's topic-person (fix 3).
            corroborated_persons=list(detected_guests or ()) + list(known_hosts or ()),
        )
    )

    # ...and the voices an LLM matched to a STATED name from their own words (ADR-110). It ranks
    # BELOW both of the above on purpose: a voice that says "I'm Peter Ludwig" needs no model's
    # opinion, and a name the host spoke aloud is a fact. The model only fills what neither covers —
    # the reporter who files under her byline, the co-host nobody ever introduces — which on desk
    # shows is most of the newsroom.
    #
    # It can only ever MATCH a name the metadata already stated (the resolver enforces the closed
    # list), so it cannot invent a speaker here; the worst it can do is misplace a real one.
    llm_named: set = set()
    for v, n in (llm_voice_names or {}).items():
        if v not in ad_voices and v not in voice_intro:
            voice_intro[v] = _canonicalize_to_known_host(n, known_hosts)
            # PROVENANCE. These must not be recorded as `self_intro`: an audit that cannot tell a
            # name the voice SAID from a name a model INFERRED cannot audit the model at all, and
            # the model is the part that needs watching.
            llm_named.add(v)

    ad_names_lower = {
        n.lower()
        for v, n in _self_intros_by_voice(voice_texts, intro_sources).items()
        if v in ad_voices and n
    }

    # A transcript-level self-introduction has NO VOICE ATTACHED TO IT.
    #
    # `extract_self_introduced_host(transcript_text)` scans the whole transcript for the first
    # "I'm <Name>" and offers it as a host name. On Latent Space — a feed that states no host — the
    # first "I'm ..." in the transcript is the GUEST introducing himself ("Yeah, I'm Peter Ludwig,
    # co-founder and CTO of Applied Intuition"). His name was handed to the host voice, and the
    # voice that actually said it was left as SPEAKER_03 with 48% of the episode.
    #
    # Per-voice self-introductions (`voice_intro`) carry the same signal AND say who said it, so
    # when we have them the transcript-level scan is strictly worse and is skipped. It survives only
    # for callers that pass no `voice_texts` at all.
    intro_source = None if voice_texts else transcript_text
    host_pool = [
        (n, s)
        for n, s in _host_name_pool(intro_source, known_hosts, host_candidates)
        if n.lower() not in ad_names_lower
    ]

    # WHICH voices are the hosts. Metadata and conversation are CROSS-REFERENCED here; neither
    # replaces the other, and neither is a statistic.
    #
    #   METADATA says WHO and HOW MANY  — `host_pool` carries the feed's own words ("journalists
    #                                     Kevin Roose and Casey Newton").
    #   The CONVERSATION says WHICH VOICE — the role is PERFORMED: the host welcomes you to the show
    #                                     and introduces the guest; the guest says thanks for having
    #                                     me. Measured on the shows whose feed states no host, this
    #                                     is decisive where talk time is worthless — on Latent Space
    #                                     the host talks 8.6% and the guest 84.5%.
    #
    # When the feed states no host at all, the conversation is the ONLY source, and it is a good
    # one: "hello and welcome to Planet Money. I'm Alexi Horowitz-Gazi" gives the role AND the name.
    conv_hosts = [v for v, r in conv_roles.items() if r == "host" and v not in ad_voices]

    host_voices = _select_host_voices(
        diarization=diarization,
        voice_intro=voice_intro,
        host_pool=host_pool,
        known_hosts=known_hosts,
        conv_hosts=conv_hosts,
        conv_guests=conv_guests,
        voices_by_intro=voices_by_intro,
        llm_named=llm_named,
        llm_voice_roles=llm_voice_roles,
        content_start=content_start,
        intro_window_s=intro_window_s,
        ad_intervals=ad_intervals,
        ad_voices=ad_voices,
    )

    host_names_lower = {n.lower() for n, _ in host_pool}
    used_lower: set[str] = set()

    by_voice = _name_host_voices(host_voices, host_pool, voice_intro, used_lower, llm_named)

    # The host also NAMES the guest out loud — "My guest today is Brian Chesky". That is a stated
    # fact from the conversation, and it complements the guests the episode description declared
    # (which the corroboration gate may have had to drop for want of an interview cue).
    intro_names_lower = {n.lower() for n in voice_intro.values()}
    declared = list(_clean_person_names(detected_guests))
    # The HOST introduces the guest, so only the resolved HOST voices' turns are trusted here.
    # Scanning EVERY voice let a guest who merely QUOTES a greeting ("...and then Sarah Chen, thanks
    # so much for coming to my defense...") harvest that name into the guest pool, where the forced
    # one-name-one-voice match then painted it onto an unrelated voice (N2). A wrong name is worse
    # than no name (#876) — with no resolved host there is no trustworthy introducer, harvest none.
    host_voice_texts = {v: (voice_texts or {}).get(v, "") for v in host_voices}
    for n in sorted(guests_introduced_by_the_host(host_voice_texts)):
        if n.lower() not in {d.lower() for d in declared}:
            declared.append(n)

    # DELIBERATELY NOT DONE HERE: an "anchor" rule, letting one confirmed guest vouch for the other
    # people the description names ("Qasar Younis and Peter Ludwig have spent the last decade...";
    # Peter self-introduces, so Qasar must speak too). A tempting rule. It was built, replayed over
    # the corpus, and REMOVED. On 160 episodes it admitted 8 names: 3 real guests (Qasar
    # Younis, Dan Gural, Marc Andreessen) and 5 that were never in the room —
    #
    #     HB Reese   the founder of Reese's, discussed by a Planet Money episode. He died in 1956.
    #     "Marc"     a bare first name, landing on a SECOND voice in the Marc Andreessen episode
    #     "Bill"     a first name, on a voice with 0.0% of the talk
    #     "Er"       an NER fragment, on a voice with 0.0% of the talk
    #
    # That is the #876 failure exactly: a person an episode DISCUSSES, painted onto a voice. A
    # description that names several people is a guest list or a topic list, and one member speaking
    # does not tell you which — the vouching has to be scoped to the PHRASE the anchor appears in,
    # and the metadata names have to be whole. Until both hold, no name.
    #
    # A wrong name is worse than no name. The only metadata-vouched naming we do is the one where
    # the VOICE ITSELF says it (`_vouched_by_metadata`, warrant (c)) — that admitted 3 names across
    # the corpus and all 3 were right.

    guest_names = [
        g
        for g in declared
        if g.lower() not in host_names_lower and g.lower() not in intro_names_lower
    ]
    # Ad voices are excluded from GUEST naming too — otherwise the pre-roll consumes a real guest's
    # name out of the pool and the guest is left as SPEAKER_0n.
    by_voice.update(
        _name_guest_voices(
            [v for v in voices_by_total if v not in ad_voices],
            by_voice,
            voice_intro,
            guest_names,
            host_names_lower,
            used_lower,
            talk=total,
            llm_named=llm_named,
            cameo_max_talk_s=profile.cameo_max_talk_s,
        )
    )
    # They still belong in the roster — as "Advertisement", not as a missing id.
    for v in ad_voices:
        by_voice.setdefault(v, SpeakerRole(name=v, role="unknown", named=False, source="raw"))

    # ADR-130 — provider-agnostic name recovery: snap any published name that ASR-mangled a STATED
    # person (host OR guest) back to the metadata spelling. Runs for every provider (repairs
    # OpenAI's manglings on the Deepgram/community-1 corpus too), reference-bounded (never invents).
    # Corroborated refs (known_hosts, detected_guests) precede metadata_named so a mangle closer to
    # a person we confirmed is in the room wins over one that merely matches someone the episode is
    # ABOUT (metadata_named carries un-corroborated subjects — see _recover_stated_names guards).
    stated_refs = list(
        dict.fromkeys(list(known_hosts) + list(detected_guests or ()) + list(metadata_named or ()))
    )
    if stated_refs:
        _recover_stated_names(
            by_voice, stated_refs, known_hosts, fuzzy=profile.nickname_fuzzy_binding
        )

    # FINAL PLAUSIBILITY GATE (ADR-134 shared core). Every naming path above — self-intro, host
    # pool, greeting reader, strategy snap, LLM, metadata — writes into `by_voice`, and each has its
    # own filters; community-1's finer clustering surfaced turn-boundary openers the ASR capitalised
    # ("But Sun", "So Nick", a bare "But") slipping through one path or another. Rather than reaudit
    # every path, refuse to PUBLISH a name that is not a plausible speaker name: demote it back to
    # the raw SPEAKER_NN (a defect marker), because a wrong label is worse than an unnamed voice.
    # Placed before the leftover/nameable accounting and `_classify_voice_types` so a demoted voice
    # is counted as an unnamed defect and re-typed correctly.
    for _v, _role in list(by_voice.items()):
        if _role.named and not is_publishable_speaker_name(_role.name):
            by_voice[_v] = replace(_role, name=_v, named=False, source="raw")

    # Which unnamed voices did we FAIL on, and which could nobody have named?
    #
    # A voice is "nameable" when a name existed for it and we did not attach it: it introduced
    # itself, the host introduced it, or a declared guest name was left over unclaimed. Those keep
    # the raw SPEAKER_NN, because that id is the defect marker — it means "we should have named this
    # and did not".
    #
    # A substantive voice that is NOT nameable is `unidentified`: nobody in the episode ever says
    # who they are. Showing a defect marker there turns a signal into noise.
    leftover_names = [g for g in guest_names if g.lower() not in used_lower]

    # ...and the names the METADATA STATED that never reached a voice — INCLUDING the ones
    # corroboration threw away.
    #
    # This is the hole that let us launder our own failures. `unidentified` claims *no source names
    # them*, and the episode description is a source. But `guest_names` only ever held the names
    # that SURVIVED corroboration, so when corroboration rejected a real guest ("the episode text
    # names them but never introduces them as speaking" — #876), the name vanished, no name was left
    # going spare, and the roster concluded nobody could have been named. Two correct safety rules,
    # compounding into a false innocence:
    #
    #     [Physical AI] description: "Qasar Younis and Peter Ludwig have spent the last decade..."
    #                   SPEAKER_01, 35% of the episode -> "Unidentified speaker"  <- A LIE.
    #
    # We could not place him. That is a DEFECT, and it is counted as one.
    #
    # This deliberately errs pessimistic: a description that merely *mentions* someone who never
    # speaks is indistinguishable here from one that names a guest we failed to bind, so both land
    # in the defect bucket. Over-counting our own failures is the safe direction — the whole reason
    # this exists is that the old rule under-counted them. ``unbound_stated_names`` in the
    # diagnostics names exactly who we could not place, so the residual stays auditable.
    stated_unbound = [
        n for n in _clean_person_names(metadata_named or ()) if n.lower() not in used_lower
    ]

    nameable = set(voice_intro)
    # M unbound names can explain at most M missed voices (ADR-139 / Pattern B). Promote only the M
    # most-substantive unnamed voices that no name already points to; the rest are genuine TAPE
    # (`unidentified`), not our failure. Without this bound, a single unbound producer credit turned
    # every 30-second vox-pop on a narrated desk (Planet Money: 4 credits → all 16 voices flagged)
    # into a "we should have named this" defect. The dominant miss the pessimistic rule protects — a
    # 35%-of-episode guest the description named ([Physical AI]) — still lands, because talk-time
    # sorts it to the top of the promotion list.
    spare_name_count = len(
        {n.lower() for n in leftover_names} | {n.lower() for n in stated_unbound}
    )
    if spare_name_count:
        promotable = sorted(
            (v for v, r in by_voice.items() if not r.named and v not in voice_intro),
            key=lambda v: total.get(v, 0.0),
            reverse=True,
        )
        # naming-4 bounds the promotion to the M most-substantive voices (Pattern B); the legacy
        # profile promotes EVERY unnamed voice (the old pessimistic rule), for a clean A/B.
        promoted = (
            promotable if not profile.pattern_b_bounded_promotion else promotable[:spare_name_count]
        )
        nameable |= set(promoted)

    by_voice = _classify_voice_types(
        by_voice,
        diarization,
        ad_intervals,
        ad_voices,
        nameable=nameable,
        cleaning=cleaning,
        cameo_max_talk_s=profile.cameo_max_talk_s,
    )
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
    metadata_named: Sequence[str] = (),
    show_centric: bool = False,
    profile: LabelingProfile = DEFAULT_LABELING_PROFILE,
    detection_ran: Optional[bool] = None,
) -> Dict[str, Any]:
    """Per-episode speaker-resolution diagnostics — *what we tried, what we resolved, and why
    each voice that stayed raw failed*. Written as a sidecar so an operator can see why a
    speaker is unrecognized without re-running the pipeline.

    ``show_centric`` marks feeds where the host is deliberately unnamed (news desks): an unnamed
    host is then flagged ``expected`` (rendered "Host"), not a detection failure.

    ``detection_ran`` says whether the speaker-detection stage executed (#1647). It changes how
    an unnamed voice is read: "nobody in the episode says who they are" is only a legitimate
    finding if we looked. ``None`` means the caller did not say, and is treated as "assume we
    looked" so existing behaviour is unchanged.
    """
    talk = _talk_time(diarization)
    per_voice_intro = _self_intros_by_voice(
        voice_texts, list(metadata_named or ()) + list(known_hosts or ()) + list(detected_guests)
    )
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
            # renders "Host". So are cameo/commercial voices (noise, not people we missed), and so
            # is an `unidentified` voice: nobody in the episode ever says who they are, so there was
            # nothing to fail at. `truly_unknown` is the "we should have named this and did not"
            # residual, and counting the tape in it made the defect number meaningless.
            expected = (show_centric and role.role == "host") or role.voice_type in (
                VOICE_CAMEO,
                VOICE_COMMERCIAL,
                VOICE_UNIDENTIFIED,
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

    # COUNT THE VOICE, NOT THE HEAD. Three cameos and one 50%-of-the-episode guest are both
    # "1 unresolved voice" by headcount, and they are nothing alike: the first is noise, the second
    # means we lost a PRINCIPAL and half the episode cannot be attributed to anybody.
    #
    # So the per-episode alarm is a SHARE OF TALK, and it is the number to look at first. A large
    # one is never a rounding error — it says a specific voice went missing, and ``unbound_names``
    # usually says whose.
    total_talk = sum(talk.values()) or 1.0
    # TOTAL unattributed — every voice attributed to nobody, for ANY reason. A trace for the sidecar
    # and operator, NOT the alarm signal.
    unattributed_s = sum(
        talk.get(v, 0.0)
        for v, role in roster.by_voice.items()
        if not role.named and role.voice_type in (VOICE_UNKNOWN, VOICE_UNIDENTIFIED)
    )
    # The DEFECT share — talk we FAILED to attribute (`unknown`: a nameable voice we missed). It is
    # the alarm signal; `unidentified` tape (a vox-pop nobody names) is NOT our failure and never
    # trips it. Counting it fired the alarm on narrated desks (Planet Money) for doing nothing wrong
    # (ADR-139 / Pattern B).
    defect_s = sum(
        talk.get(v, 0.0)
        for v, role in roster.by_voice.items()
        if not role.named and role.voice_type == VOICE_UNKNOWN
    )
    defect_share = defect_s / total_talk
    unattributed_share = unattributed_s / total_talk
    used_lower = {str(r.name).lower() for r in roster.by_voice.values() if r.named}
    unbound_names = [
        n for n in _clean_person_names(metadata_named or ()) if n.lower() not in used_lower
    ]

    # ---- Labeling OUTPUT: the clean speaker surface handed to GI/KG (ADR-135/#1220) ----------
    # Everything above is the raw diarization INPUT — it counts every voice pyannote heard,
    # including ad/cameo noise. This is what SURVIVES cleanup: cameo/commercial are dropped,
    # leaving the real conversational speakers that become Person (named) / Voice (unresolved)
    # nodes. Recorded here so the per-episode sidecar answers "what did labeling actually expose
    # downstream?" on its own — without opening the graph (grounding attrition may still drop a
    # speaker with no grounded quote, so the graph is a cross-check, not the same number).
    exposed = [r for r in roster.by_voice.values() if r.voice_type not in _NOISE_VOICE_TYPES]
    exposed_named = sum(1 for r in exposed if r.named)
    exposed_out = {
        "speakers": len(exposed),
        "named": exposed_named,
        # Unnamed exposed voices — these become ``Voice`` nodes downstream.
        "voices": len(exposed) - exposed_named,
        # Of those Voices: a real person we failed to name (defect) vs one nobody ever names.
        "voices_unknown": sum(1 for r in exposed if not r.named and r.voice_type == VOICE_UNKNOWN),
        "voices_unidentified": sum(
            1 for r in exposed if not r.named and r.voice_type == VOICE_UNIDENTIFIED
        ),
    }

    # Full labeling census (v2.4 sidecar directive): count AND talk-seconds AND talk-share per voice
    # type, so an operator reads "how many named / unknown / unidentified / cameo / commercial, and
    # how much of the episode each holds" straight from the sidecar — no deeper dig. `named` folds
    # into `person`. `unknown` is the defect worth chasing; `unidentified` is tape we accept.
    census: Dict[str, Dict[str, float]] = {}
    for v, r in roster.by_voice.items():
        c = census.setdefault(r.voice_type, {"count": 0, "talk_s": 0.0})
        c["count"] += 1
        c["talk_s"] += talk.get(v, 0.0)
    for c in census.values():
        c["talk_share"] = round(c["talk_s"] / total_talk, 4)
        c["talk_s"] = round(c["talk_s"], 1)

    return {
        "summary": {
            "num_speakers": roster.num_speakers,
            "named": named,
            "unresolved": unresolved,
            # Of the unresolved voices, how many are noise (cameo/commercial) vs a real person
            # we failed to name (unknown) — so an operator can tell "worth chasing" from "junk".
            "by_voice_type": type_counts,
            # Full census with talk-time per type (the number to report from, not deep-dive).
            "voice_census": census,
            # The labeling OUTPUT surface (post cameo/commercial cleanup) exposed to GI/KG.
            "exposed": exposed_out,
            "show_centric": show_centric,
            # Unresolved voices that are the EXPECTED outcome (show-centric host, cameo, ad) vs a
            # genuine miss — ``truly_unknown`` is the real "we failed to name a person" residual.
            "expected_unresolved": expected_unnamed,
            "truly_unknown": unresolved - expected_unnamed,
            # How much of the EPISODE is attributed to nobody, for ANY reason (defect + tape). A
            # trace, not the alarm — kept so the sidecar carries the full picture.
            "unattributed_talk_share": round(unattributed_share, 4),
            # The share that is OUR failure — a nameable voice (`unknown`) we missed. This is the
            # alarm basis; `unidentified` tape does not count (ADR-139 / Pattern B).
            "unattributed_defect_share": round(defect_share, 4),
            # Did the speaker-detection stage run at all? (#1647) Without this the two rows
            # below are unreadable: zero named voices means something different when we looked
            # and found nobody than when we never looked.
            "detection_stage_ran": detection_ran,
            # naming-4 alarms on the DEFECT share; the legacy profile alarms on TOTAL unattributed
            # (the pre-Pattern-B behaviour). Threshold + basis both come from the profile (ADR-140).
            #
            # PLUS an unconditional trip (#1646/#1647): nothing named AND detection never ran.
            # The share-based basis cannot catch that case — it deliberately excludes
            # ``unidentified`` talk as "tape nobody names, not our failure" (ADR-139), which is
            # correct ONLY when detection actually looked. On an episode where the stage was
            # skipped, every voice degrades to ``unidentified`` and the defect share falls to
            # 0.0, so the alarm read FALSE on episodes that lost 100 % of their insights. That
            # is exactly what happened across 72 % of the corpus under #1646.
            "unattributed_alarm": bool(
                (defect_share if profile.alarm_on_defect_share else unattributed_share)
                >= profile.unattributed_alarm_threshold
                or (named == 0 and detection_ran is False and roster.num_speakers > 0)
            ),
            # Which labeling profile produced this episode (reprocess key + A/B provenance).
            "labeling_profile": profile.version,
            # Names the metadata stated and we could not place. When the alarm fires, this is
            # usually the answer to "who did we lose".
            "unbound_names": unbound_names,
        },
        "tried": {
            "host_self_intro": (
                extract_self_introduced_host(transcript_text) if transcript_text else None
            ),
            "known_hosts": list(known_hosts),
            "detected_guests": list(detected_guests),
            "metadata_named": list(metadata_named or ()),
            "per_voice_self_intro": {v: per_voice_intro.get(v) for v in roster.by_voice},
        },
        "voices": voices,
    }

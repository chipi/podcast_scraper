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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ....speaker_detectors.hosts import (
    _clean_stated_name as _clean_intro_name,
    _GUEST_INTRODUCED_BY_HOST as _GUEST_INTRODUCED_BY_HOST_RE,
    _NAME_RE as _INTRO_NAME_RE,
    extract_self_introduced_host,
    guests_introduced_by_the_host,
    has_org_markers,
    is_network_or_org_author,
    roles_from_conversation,
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

# When this much of an episode's TALK cannot be attributed to anybody, we did not lose a bit of
# noise — we lost a PRINCIPAL. A vox-pop cutaway costs a few percent; a guest we failed to bind
# costs a quarter of the episode or more. Measured on the real corpus, the misses cluster at 26-50%
# while the genuinely-nameless tape of a narrated show sits well below it.
UNATTRIBUTED_TALK_ALARM = 0.25


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
        ad_frac = (ad_by_voice.get(v, 0.0) / total) if total else 0.0
        if ad_intervals and ad_frac >= COMMERCIAL_AD_FRACTION:
            vt = VOICE_COMMERCIAL
        elif total < CAMEO_MAX_TALK_S:
            vt = VOICE_CAMEO
        elif nameable is not None and v not in nameable:
            # Substantive, and NO source names them: the tape / vox-pop of a narrated documentary.
            # Not a failure — there was no name to be had.
            vt = VOICE_UNIDENTIFIED
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


def _canonicalize_to_known_host(name: str, known_hosts: Sequence[str]) -> str:
    """Snap an ASR-mangled self-introduction onto the configured host name.

    A self-introduction is transcribed, so it carries the ASR's spelling: Kevin Roose introduces
    himself and Whisper writes "Kevin Russo" in one episode and "Kevin Roos" in the next. The roster
    trusts a self-intro above ``known_hosts``, so the corpus ended up with three different people
    hosting the same show, none of them spelled correctly.

    Snapping requires an EXACT first-name match plus a near surname (phonetic, or within a small
    edit distance), so a guest who merely shares a host's first name is left alone. Requiring both
    is what keeps this from quietly renaming real people.
    """
    toks = name.split()
    if len(toks) < 2:
        return name
    first, last = toks[0].lower(), toks[-1]
    for host in known_hosts:
        h = host.split()
        if len(h) < 2 or h[0].lower() != first:
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


def _self_intros_by_voice(
    voice_texts: Optional[Dict[str, str]], metadata_named: Sequence[str] = ()
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
        # A bare first name, or a "this is <X>" — neither stands up alone. The metadata decides.
        candidates = [name] if name else []
        candidates += [m.group(1).strip(" .,") for m in _THIS_IS_INTRO.finditer(head)]
        for cand in candidates:
            stated = _vouched_by_metadata(cand, metadata_named)
            if stated:
                out[voice] = stated
                break
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
        and (talk is None or talk.get(v, 0.0) >= CAMEO_MAX_TALK_S)
    ]
    spare = [g for g in guest_names if g.lower() not in used_lower]
    # One name, one voice: the assignment is forced, so it is not a guess.
    forced = spare[0] if (len(spare) == 1 and len(unassigned) == 1) else None

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


def _voice_named_by_the_introduction(
    ordered_turns: Optional[Sequence[Tuple[str, str]]],
) -> Dict[str, str]:
    """``{voice: name}`` for a voice the HOST introduced by name — "and now, Bobby Allen".

    The person a host introduces is the person who SPEAKS NEXT. That is conversation structure, and
    it is the only per-voice way to use an introduction: knowing that "Bobby Allen" was named
    somewhere in the episode does not say WHICH cluster he is, and handing introduced names out by
    talk order is just the talk-share mistake wearing a different hat.

    Worth 5% of the corpus's talk. Planet Money is a narrated desk that hands off constantly
    ("joined by...", "here with me is..."), and every one of those reporters came out as SPEAKER_NN.

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
    for i, (_speaker, text) in enumerate(ordered_turns):
        for m in _GUEST_INTRODUCED_BY_HOST_RE.finditer(text or ""):
            names = [
                n
                for n in (_clean_intro_name(x) for x in _INTRO_NAME_RE.findall(m.group("names")))
                if n
            ]
            if not names:
                continue
            # whoever speaks next, that is who was just introduced
            introducer = ordered_turns[i][0]
            for j in range(i + 1, min(i + 6, len(ordered_turns))):
                nxt = ordered_turns[j][0]
                if nxt == introducer or nxt in out:
                    continue
                name = names[0]
                if name.lower() in taken:
                    break
                out[nxt] = name
                taken.add(name.lower())
                break
    return out


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
    intro_window_s: float = INTRO_WINDOW_SECONDS,
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
    """
    if not diarization.segments:
        return SpeakerRoster(by_voice={}, num_speakers=diarization.num_speakers or 0)

    # Ad voices are established BEFORE anything can be named from them: the pre-roll opens the
    # episode and reads its own name, so it wins both the "opening voice = host" rule and the
    # most-trusted self-introduction rule unless it is removed from contention up front.
    ad_voices = _edge_ad_voices(diarization)

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
    voice_intro = {
        v: _canonicalize_to_known_host(n, known_hosts)
        for v, n in _self_intros_by_voice(voice_texts, intro_sources).items()
        if v not in ad_voices
    }

    # A voice the HOST introduced by name — "and now, Bobby Allen" — is that person, because the
    # person a host introduces is the person who speaks next. It complements the self-introduction:
    # plenty of guests never say their own name, and are named FOR them. A voice that already
    # introduced itself keeps its own word for it.
    for v, n in _voice_named_by_the_introduction(ordered_turns).items():
        if v not in ad_voices and v not in voice_intro:
            voice_intro[v] = _canonicalize_to_known_host(n, known_hosts)

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
    conv_roles = roles_from_conversation(voice_texts)
    conv_hosts = [v for v, r in conv_roles.items() if r == "host" and v not in ad_voices]
    conv_guests = {v for v, r in conv_roles.items() if r == "guest"}

    host_voices: List[str] = []
    known_lower = {h.lower() for h in known_hosts}

    # 1. A voice that introduces itself as one of the feed's STATED hosts IS that host. Both sources
    #    agree — this is the cross-reference, and it is the strongest evidence available.
    for v, n in voice_intro.items():
        if n.lower() in known_lower and v not in host_voices and v not in conv_guests:
            host_voices.append(v)

    # 2. A voice that PERFORMS the host's role is a host, even if the feed never named them.
    #
    #    But the feed says how MANY. When it named its hosts, that count is binding: a third voice
    #    cannot host a two-host show, however host-like it sounds. Diarization merges a host's turn
    #    into a guest's cluster often enough that the guest's cluster "performs" a host act, and on
    #    Hard Fork that produced a third host. Where the feed named nobody, there is no count to
    #    respect and the conversation is the only source — so it is uncapped there.
    cap = len(host_pool) if host_pool else None
    for v in conv_hosts:
        if cap is not None and len(host_voices) >= cap:
            break
        if v not in host_voices:
            host_voices.append(v)

    # 3. The opener — the host does the intro (the pre-roll ad is excluded, or it wins this).
    #    Skipped when the conversation already identified a host, and never applied to a voice the
    #    conversation identified as a GUEST ("thanks for having me").
    opener = _opening_voice(
        diarization,
        window_end=content_start + intro_window_s,
        ad_intervals=ad_intervals,
        ad_voices=ad_voices,
    )
    if opener is None and voices_by_intro:
        opener = voices_by_intro[0]
    if (
        opener is not None
        and not conv_hosts
        and opener not in host_voices
        and opener not in conv_guests
        and len(host_voices) < max(1, len(host_pool))
    ):
        host_voices.append(opener)

    # 4. Fill any host slots the feed COUNTED but we have not matched yet, from the voices present
    #    in the SHOW'S INTRO (ads excluded). The hosts open the show; that is what an intro is.
    #
    #    NOT by talk time. Filling by "who talks most" hands a host slot straight to the guest —
    #    on Invest Like the Best the guest talks 82% and the host 17%, and even on a co-hosted show
    #    a long first answer outweighs both hosts. Talk share does not identify a host, in any
    #    format. And a voice the conversation heard say "thanks for having me" is never a host.
    for v in voices_by_intro:
        if len(host_voices) >= len(host_pool):
            break
        if v not in host_voices and v not in conv_guests:
            host_voices.append(v)

    host_names_lower = {n.lower() for n, _ in host_pool}
    used_lower: set[str] = set()

    by_voice = _name_host_voices(host_voices, host_pool, voice_intro, used_lower, llm_named)

    # The host also NAMES the guest out loud — "My guest today is Brian Chesky". That is a stated
    # fact from the conversation, and it complements the guests the episode description declared
    # (which the corroboration gate may have had to drop for want of an interview cue).
    intro_names_lower = {n.lower() for n in voice_intro.values()}
    declared = list(_clean_person_names(detected_guests))
    for n in sorted(guests_introduced_by_the_host(voice_texts)):
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
        )
    )
    # They still belong in the roster — as "Advertisement", not as a missing id.
    for v in ad_voices:
        by_voice.setdefault(v, SpeakerRole(name=v, role="unknown", named=False, source="raw"))

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
    if leftover_names or stated_unbound:
        # A name is still going spare, so ANY unnamed voice could have taken it — we cannot claim
        # nobody could be named.
        nameable |= {v for v, r in by_voice.items() if not r.named}

    by_voice = _classify_voice_types(
        by_voice, diarization, ad_intervals, ad_voices, nameable=nameable
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
) -> Dict[str, Any]:
    """Per-episode speaker-resolution diagnostics — *what we tried, what we resolved, and why
    each voice that stayed raw failed*. Written as a sidecar so an operator can see why a
    speaker is unrecognized without re-running the pipeline.

    ``show_centric`` marks feeds where the host is deliberately unnamed (news desks): an unnamed
    host is then flagged ``expected`` (rendered "Host"), not a detection failure.
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
    unattributed_s = sum(
        talk.get(v, 0.0)
        for v, role in roster.by_voice.items()
        if not role.named and role.voice_type in (VOICE_UNKNOWN, VOICE_UNIDENTIFIED)
    )
    unattributed_share = unattributed_s / total_talk
    used_lower = {str(r.name).lower() for r in roster.by_voice.values() if r.named}
    unbound_names = [
        n for n in _clean_person_names(metadata_named or ()) if n.lower() not in used_lower
    ]

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
            # How much of the EPISODE nobody can be held to — the headline defect number.
            "unattributed_talk_share": round(unattributed_share, 4),
            "unattributed_alarm": unattributed_share >= UNATTRIBUTED_TALK_ALARM,
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

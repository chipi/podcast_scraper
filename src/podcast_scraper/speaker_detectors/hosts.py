"""Host detection from feed metadata and transcript intro."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Set

from .entities import extract_person_entities as _extract_person_entities_direct

logger = logging.getLogger(__name__)

# RSS author tags are often the network/publisher, not the host — e.g. "Colossus",
# "Colossus | Investing & Business Podcasts", "NPR". Real hosts are personal "First Last"
# names. Reject org/network-looking tags so host detection falls through to transcript-intro
# NER / config ``known_hosts`` instead of mislabelling the host on every episode (#876).
_NONPERSON_AUTHOR_MARKERS = re.compile(
    r"[|/&@]|\d|"
    r"\b(?:podcasts?|media|networks?|productions?|studios?|radio|fm|news|inc|llc|ltd|"
    r"co|company|corp|shows?|entertainment|audio|broadcasting|group|labs?|"
    # News-outlet suffixes — a publisher, not a person ("The New York Times", "Financial
    # Times", "Wall Street Journal", "Chicago Tribune"). Standalone-surname words (Post, Press)
    # are left out here and caught by KNOWN_NETWORKS to avoid flagging people like "Emily Post".
    r"times|journal|tribune|gazette|herald|chronicle|magazine|quarterly|newspaper|gmbh|plc)\b",
    re.IGNORECASE,
)


# Known podcast networks / publishers that appear as a spoken bumper ("This is Unhedged,
# I'm Pushkin. I'm Katie Martin…") or an RSS author tag, but are NOT a person. A bare
# mononym is not enough to reject a self-introduced name (real hosts go by one name —
# Oprah, Sting), so host-intro extraction needs this explicit list to skip the network
# bumper and fall through to the actual host. Lowercased; matched against the whole name
# and its first token. (#876 — "Pushkin" leaked as the Unhedged host.)
KNOWN_NETWORKS: frozenset[str] = frozenset(
    {
        "pushkin",
        "wondery",
        "gimlet",
        "npr",
        "iheart",
        "iheartradio",
        "spotify",
        "audible",
        "stitcher",
        "radiotopia",
        "earwolf",
        "headgum",
        "ringer",
        "the ringer",
        "vox",
        "crooked media",
        "maximum fun",
        "maximumfun",
        "barstool",
        "cadence13",
        "megaphone",
        "acast",
        "patreon",
        "substack",
        "bloomberg",
        "kaleidoscope",
        # Multi-token news publishers not caught by the org-marker suffixes (Post/Guardian/etc.).
        "the new york times",
        "new york times",
        "the washington post",
        "washington post",
        "the guardian",
        "the economist",
        "the atlantic",
        "reuters",
        "associated press",
        "the wall street journal",
        "financial times",
        "pushkin industries",
        # Firms that publish a show under their own brand. Same shape as the news publishers
        # above — two real-looking tokens, no generic org marker — so nothing else catches
        # them. Added from OBSERVED corpus damage (#1652), not speculation:
        # ``person:andreessen-horowitz`` was the corpus's TOP-RANKED Person (54 episodes,
        # 723 insights), and on one a16z episode it was the only "person" present while the
        # actual speakers went unresolved. Whole-name match only — the first-token check
        # cannot fire on "andreessen", so a real person like Marc Andreessen is untouched.
        "andreessen horowitz",
        "a16z",
    }
)


def is_known_network(name: str) -> bool:
    """True when ``name`` (whole or its first token) is a known podcast network/publisher.

    Used to skip a network *bumper* in a host self-introduction ("I'm Pushkin") and to flag a
    network name that leaked into ``content.speakers`` even when it carries no generic org
    markers (``Pushkin`` has none — :func:`has_org_markers` returns False for it). #876.
    """
    n = (name or "").strip().lower()
    if not n:
        return False
    if n in KNOWN_NETWORKS:
        return True
    first = n.split()[0] if n.split() else ""
    return first in KNOWN_NETWORKS


def has_org_markers(name: str) -> bool:
    """True when ``name`` contains explicit network/organisation markers.

    The marker-only half of :func:`is_network_or_org_author` (``|``, ``&``, digits, words like
    ``Podcasts``/``Media``/``Network``) — WITHOUT the mononym rule. Use this for names from
    trusted person sources (a transcript self-introduction, config ``known_hosts``, or a
    detected guest), where a single-token name is a real person (Oprah, Sting), not a network.
    """
    n = (name or "").strip()
    if not n:
        return True
    return bool(_NONPERSON_AUTHOR_MARKERS.search(n))


def is_network_or_org_author(name: str) -> bool:
    """True when an RSS author tag looks like a network/organisation, not a host person.

    Any of these → reject: org/network markers (see :func:`has_org_markers`); or a single
    mononym token (real hosts are ``First Last``; this also catches all-caps acronyms like
    NPR/BBC). The mononym rule is specific to RSS **author tags** (where a lone token is almost
    always the network); apply :func:`has_org_markers` instead to trusted person names. Mononym
    person-hosts can still be supplied via config ``known_hosts`` (#876).
    """
    n = (name or "").strip()
    if not n:
        return True
    if has_org_markers(n):
        return True
    # A known network/publisher in an author tag is the PUBLISHER, not a host — and multi-token
    # brands ("Andreessen Horowitz", "The New York Times") carry no generic org marker and are
    # not mononyms, so nothing else here rejects them (#1652). This check was already applied
    # to self-introductions and to host/guest metadata via ``looks_like_publisher``; the RSS
    # author path was the one place that skipped it, which is how ``person:andreessen-horowitz``
    # became the corpus's top-ranked Person.
    if is_known_network(n):
        return True
    if len(n.split()) < 2:  # mononym ("Colossus", "NPR") — not a "First Last" host name
        return True
    return False


# Name suffixes that legitimately follow a comma. Without these, "Martin Luther King, Jr."
# splits into a person and the orphan token "Jr.".
_NAME_SUFFIXES = frozenset(
    {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "phd", "ph.d.", "md", "m.d.", "esq", "esq."}
)

# Comma, semicolon, ampersand, or a standalone "and" — the separators RSS author tags actually
# use. Word-bounded so "Alexander" is not cut at its "and".
_AUTHOR_SEPARATORS = re.compile(r"\s*(?:,|;|&|\band\b)\s*", re.IGNORECASE)


def split_author_names(author: str) -> list[str]:
    """Split one RSS author tag into individual person names (#1652).

    Publishers routinely put a whole cast in a single ``<itunes:author>``:
    ``"Brandon Anderson, RJ Honicky, and Latent.Space"``. Kept whole, that string can never
    match a diarized voice — the roster compares per name — so the known-hosts fallback silently
    does nothing for every multi-author feed.

    Deliberately conservative, because a bad split INVENTS a person, which is worse than
    failing to find one:

    - name suffixes are re-attached (``"Martin Luther King, Jr."`` stays one name);
    - fragments that are not plausible names are dropped by the caller's
      :func:`is_network_or_org_author` check, which already rejects mononyms — so an
      over-eager split degrades to "no host", the safe direction (#876), never to a fake one;
    - a tag with no separator is returned unchanged.
    """
    text = (author or "").strip()
    if not text:
        return []

    parts = [part.strip() for part in _AUTHOR_SEPARATORS.split(text)]
    merged: list[str] = []
    for part in parts:
        if not part:
            continue
        if merged and part.lower().rstrip(".") in {s.rstrip(".") for s in _NAME_SUFFIXES}:
            # "Jr." belongs to the name before it, not to a new person.
            merged[-1] = f"{merged[-1]}, {part}"
            continue
        merged.append(part)
    return merged


def normalize_host_names(names: Iterable[str]) -> Set[str]:
    """The single gate every host-name source must pass through (#1652).

    Four independent code paths can seed ``known_hosts`` — the deterministic feed parse, the
    LLM provider's ``detect_hosts``, episode-level ``<itunes:author>`` tags, and config
    ``known_hosts``. Each one had grown its own idea of cleaning, and the two that had none
    were the two that shipped a composite into the corpus:

    - the provider path returned ``"Erik Torenberg, Ben Horowitz, Travis Kalanick"`` as one
      string on *The a16z Show*;
    - the episode-authors fallback returned the same composite from ``<itunes:author>`` — and
      that is the path that actually fired on the acceptance run, which a fix applied only to
      the provider path did not touch.

    A composite is worse than no host at all: the roster compares per name, so it can never
    match a diarized voice (silently disabling the anchor) while still minting a ``Person``
    node for a human who does not exist. Centralising the rule is the point — a fifth seeding
    path added later cannot forget to call something it has to go through anyway.

    Conservative in the same direction as :func:`split_author_names`: an over-eager split
    degrades to "no host" (#876), never to an invented person.
    """
    out: Set[str] = set()
    for raw in names or ():
        text = str(raw or "").strip()
        if not text:
            continue
        # "Jane Roe <jane@example.com>" — the feed-author path stripped the address, the other
        # paths did not, so the same person arrived under two different spellings.
        if "<" in text and ">" in text:
            text = text.split("<")[0].strip()
        for candidate in split_author_names(text):
            candidate = candidate.strip()
            if not candidate or is_network_or_org_author(candidate):
                continue
            out.add(candidate)
    return out


def looks_like_publisher(name: str) -> bool:
    """True when a name is a network / publisher / organisation rather than a person.

    Combines the known-network denylist with the generic org-marker + news-outlet-suffix regex.
    Unlike :func:`is_network_or_org_author` this does NOT apply the mononym rule, so a
    single-token real person (Oprah, Sting) is kept — use it to strip publishers from
    already-resolved person surfaces (key people, host/guest roles) without dropping people.
    """
    return is_known_network(name) or has_org_markers(name)


# Host self-introduction in the transcript intro, e.g. "I'm Patrick O'Shaughnessy" or
# "My name is Ana Rodriguez". The name sub-pattern allows apostrophes/hyphens so it captures full
# surnames ("O'Shaughnessy", "Jean-Luc") but NOT periods — a period ends the self-intro sentence, so
# excluding it stops the match from absorbing the next sentence ("…O'Shaughnessy. My guest").
# "my name is" is a safe discovery cue (no network bumper says it, unlike "this is X" =
# "This is Planet Money", which stays metadata-gated in `_THIS_IS_INTRO`).
_HOST_SELF_INTRO = re.compile(
    r"\b(?:I'?m|[Mm]y name is)\s+([A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+){0,3})"
)


def extract_self_introduced_host(
    transcript_text: Optional[str], *, intro_chars: int = 2000
) -> Optional[str]:
    """Return the host's name from a transcript-intro self-introduction (``I'm <Name>``).

    Diarization yields anonymous speaker turns, and for network-published shows the host's
    name is *not* in the feed metadata (the author tag is the network — see
    :func:`is_network_or_org_author`). The host almost always self-introduces in the
    first ~90s ("Hello and welcome, I'm Patrick O'Shaughnessy"), so this lets us marry the
    transcript-derived host name to the diarized host speaker (#876). Only the intro is
    scanned so a guest who later says "I'm …" isn't mistaken for the host. Returns ``None``
    when no self-introduction is found.
    """
    if not transcript_text:
        return None
    # Scan ALL self-introductions in the intro, not just the first: network shows open with a
    # publisher bumper in the same "I'm <X>" shape ("This is Unhedged… I'm Pushkin. I'm Katie
    # Martin"), so the first match is often the network, not the host. Skip known-network
    # bumpers and return the first match that is a real person name (#876 — "Pushkin" leak).
    for match in _HOST_SELF_INTRO.finditer(transcript_text[:intro_chars]):
        name = match.group(1).strip(" .,")
        if len(name) < 2:
            continue
        if is_known_network(name):
            continue
        # "I'm Coming Out" is not a self-introduction. The regex takes any capitalised run and the
        # ASR capitalises freely; The Daily had a voice recorded as introducing itself as
        # "Coming Out". A single-token match is still allowed here (a mononym host — Oprah, Sting),
        # so the guard only fires on a multi-token run containing an ordinary English word.
        if len(name.split()) >= 2 and not looks_like_a_person_name(name):
            continue
        # A single-token capture must be a plausible mononym, not a sentence-opener the ASR
        # capitalised at a turn boundary. "I'm But it …" (a disfluency) captured a bare "But" and,
        # because the loop returns on the FIRST hit, shadowed a real later "I'm <Name>". This is the
        # guard `distinct_self_introductions` already applies; without it here the two sibling
        # scanners disagreed. ``continue`` (not ``return None``) keeps scanning for the real intro.
        if len(name.split()) == 1 and not is_plausible_mononym(name):
            continue
        return name
    return None


def distinct_self_introductions(
    transcript_text: Optional[str], *, intro_chars: int = 2000
) -> List[str]:
    """Every DISTINCT person-name a voice introduces itself as ("I'm <Name>"), same filtering as
    :func:`extract_self_introduced_host` (network bumpers + ordinary-word runs skipped).

    One physical speaker introduces itself once. Two or more distinct self-introductions in a single
    diarization cluster is the signature of a MERGED cluster — a cold-open montage that strings
    several hosts' intros together ("I'm Kevin Russo… I'm Casey Noon…") collapses into one voice.
    The caller uses ``len(...) >= 2`` to refuse naming such a cluster after any one of them.
    """
    seen: List[str] = []
    lowered: Set[str] = set()
    for match in _HOST_SELF_INTRO.finditer((transcript_text or "")[:intro_chars]):
        name = match.group(1).strip(" .,")
        if len(name) < 2 or is_known_network(name):
            continue
        toks = name.split()
        # A multi-token run must look like a person; a single token must be a plausible mononym, not
        # a bare honorific ("Dr", the truncated "I'm Dr. Jane Smith" capture) — else "I'm Dr. X …
        # I'm X" would count as two distinct speakers and wrongly read as a montage.
        if len(toks) >= 2 and not looks_like_a_person_name(name):
            continue
        if len(toks) == 1 and not is_plausible_mononym(name):
            continue
        if name.lower() not in lowered:
            lowered.add(name.lower())
            seen.append(name)
    return seen


def _extract_person_entities(text: str, nlp: Any) -> list[tuple[str, float]]:
    """Resolve extract_person_entities via public wrapper when loaded (patchable in tests)."""
    try:
        from podcast_scraper.providers.ml import speaker_detection

        return speaker_detection.extract_person_entities(text, nlp)
    except ImportError:
        return _extract_person_entities_direct(text, nlp)


def _log(logger_method: str, message: str, *args: object) -> None:
    """Emit log via wrapper module logger when available (patchable in tests)."""
    try:
        from podcast_scraper.providers.ml import speaker_detection

        getattr(speaker_detection.logger, logger_method)(message, *args)
    except ImportError:
        getattr(logger, logger_method)(message, *args)


def detect_hosts_from_transcript_intro(
    transcript_text: str,
    nlp: Optional[Any] = None,
    intro_duration_seconds: int = 120,
    words_per_second: float = 2.5,
) -> Set[str]:
    """Detect host names from transcript intro patterns (first 60-120 seconds)."""
    if not transcript_text or not nlp:
        return set()

    intro_word_count = int(intro_duration_seconds * words_per_second)
    words = transcript_text.split()[:intro_word_count]
    intro_text = " ".join(words)

    # The cue ("I'm" / "welcome to") is matched case-insensitively, but the NAME capture is scoped
    # case-SENSITIVE with (?-i:...): under a blanket re.IGNORECASE the [A-Z][a-z]+ classes matched
    # any letter, so "I'm going to explain how this works" captured "going to explain..." as a host
    # name (N3). Same fix the module's _NAME pattern already uses elsewhere.
    intro_patterns = [
        r"I'?m\s+((?-i:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*))",
        r"This is\s+[^.]+\s+I'?m\s+((?-i:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*))",
        r"Welcome to\s+[^.]+\s+I'?m\s+((?-i:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*))",
    ]

    detected_names = set()
    for pattern in intro_patterns:
        matches = re.finditer(pattern, intro_text, re.IGNORECASE)
        for match in matches:
            name = match.group(1).strip()
            if name and len(name) > 2 and name.lower() not in ["the", "this", "that"]:
                detected_names.add(name)

    if nlp:
        intro_persons = _extract_person_entities(intro_text, nlp)
        for name, _ in intro_persons:
            detected_names.add(name)

    return detected_names


# The feed STATES its hosts. Read the statement — do not just run NER over the paragraph.
#
#   Hard Fork      "journalists Kevin Roose and Casey Newton explore..."
#   The Journal    "Hosted by Ryan Knutson and Jessica Mendoza."
#   No Priors      "co-hosts Elad Gil and Sarah Guo talk to..."
#   Odd Lots       "Bloomberg's Joe Weisenthal and Tracy Alloway explore..."
#   Invest Like…   in the TITLE: "Invest Like the Best with Patrick O'Shaughnessy"
#
# Bare NER over the description is not good enough, and Latent Space is the proof: its description
# lists PAST GUESTS (Bret Taylor, Chris Lattner, George Hotz...), and NER offered every one of them
# as a host. The phrase is the signal, not the entity.
# A name is a run of Capitalised words, and that capitalisation is the whole signal. The
# `(?-i:...)` keeps the character classes case-SENSITIVE even where the surrounding pattern is
# compiled with re.IGNORECASE for its lowercase cue words ("joined by", "is with us"). Without it,
# IGNORECASE makes `[A-Z]` match a-z too, so this pattern matches every multi-word lowercase phrase
# in the transcript — which both crowned non-names as guests AND made the conversation scan
# backtrack catastrophically (a 77k-char episode spun for minutes in guests_introduced_by_the_host).
# The token-run and the name-list are BOUNDED ({1,5} / {0,9}) rather than unbounded (+/*): two
# nested unbounded quantifiers over a long capitalized run are O(n²) on the finditer scan (a
# 60k-char voice measured 3.3s, 120k → 13s), and a real person-name is <=6 tokens / an intro <=10
# people — anything longer is org/ASR noise the has_org_markers + looks_like_a_person_name guards
# reject downstream. Atomic groups would be exact but are 3.11-only (floor is 3.10). Bounding makes
# every consumer (_NAMES sites, _NAME_RE) linear with identical matches on real intros.
_NAME = r"(?-i:[A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+){1,5})"
_NAMES = rf"{_NAME}(?:\s*(?:,|and|&)\s*{_NAME}){{0,9}}"
# Presenting verbs — what a show's own description says its hosts DO.
_PRESENTS = r"(?:explore|explain|discuss|talk|cover|host|present|bring)s?\b"
_HOST_PHRASES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        rf"\bhosted\s+by\s+(?P<names>{_NAMES})",
        rf"\bco-?hosts?\s+(?P<names>{_NAMES})",
        rf"\bjournalists?\s+(?P<names>{_NAMES})",
        # "Joe Weisenthal and Tracy Alloway explore..." / "Katie Martin, Robert Armstrong and other
        # markets nerds at the Financial Times explain..." — names, then a presenting verb. The
        # filler between them is bounded so the verb belongs to THESE names.
        rf"(?P<names>{_NAMES})[\w\s,'’\-]{{0,60}}?\s+{_PRESENTS}",
        rf"\bwith\s+(?P<names>{_NAME})\s*$",  # the show title: "... with Patrick O'Shaughnessy"
    )
]
_NAME_RE = re.compile(_NAME)


def hosts_from_feed_statement(
    feed_title: Optional[str], feed_description: Optional[str]
) -> Set[str]:
    """Hosts the feed EXPLICITLY names ("Hosted by X and Y"), rather than every person it mentions.

    This is the authoritative source: the show says who presents it. Only used for the names inside
    the host phrase, so a description that also lists past guests cannot smuggle them in.
    """
    title_lower = (feed_title or "").lower()
    out: Set[str] = set()
    for is_title, text in ((True, feed_title or ""), (False, feed_description or "")):
        if not text.strip():
            continue
        for pat in _HOST_PHRASES:
            m = pat.search(text)
            if not m:
                continue
            for raw in _NAME_RE.findall(m.group("names")):
                clean = _clean_stated_name(raw)
                if len(clean.split()) < 2 or has_org_markers(clean):
                    continue
                # A publisher/platform is never the host, even inside a host phrase (#1652
                # applied this to RSS author tags; the statement path was the last place that
                # skipped it). Real case from the #1657 acceptance run: The a16z Show's episode
                # blurb runs two sentences together with no full stop —
                # "...Listen to the a16z Show on Spotify Listen to the a16z Show on Apple
                # Podcasts Follow our host:" — so "Spotify Listen" is a capitalised run across
                # the sentence boundary, and the NOUN "host" 45 chars later satisfied the
                # presenting-verb pattern. Rejecting known platforms kills it at the name.
                if is_known_network(clean):
                    logger.debug(
                        "host statement named '%s', which is a platform/publisher, not a host",
                        clean,
                    )
                    continue
                # In the DESCRIPTION, a capitalised run that echoes the show's own name is the show,
                # not a person: "At Planet Money, we explore...". In the TITLE it is the opposite —
                # that is where the host lives ("Invest Like the Best with Patrick O'Shaughnessy"),
                # so the same guard there would throw the host away.
                if not is_title and clean.lower() in title_lower:
                    continue
                out.add(clean)
    return out


# A capitalised run is not automatically a name: it can start with a preposition ("At Planet
# Money"), or be prefixed by the publisher's possessive ("Bloomberg's Joe Weisenthal").
_LEADING_JUNK = re.compile(r"^(?:At|In|On|By|With|From|The)\s+", re.IGNORECASE)
# "Bloomberg's Joe Weisenthal", "Red Hat's Chris Wright" — the employer, then the person. Non-greedy
# so it strips through the FIRST possessive only, leaving "Patrick O'Shaughnessy" (no "'s ") alone.
_POSSESSIVE_PREFIX = re.compile(r"^.*?['’]s\s+")


def _clean_stated_name(name: str) -> str:
    clean = (name or "").strip()
    clean = _POSSESSIVE_PREFIX.sub("", clean)
    clean = _LEADING_JUNK.sub("", clean)
    return clean.strip()


# When the feed states no host, the CONVERSATION does. The role is performed, not measured: the host
# welcomes you to the show and introduces the guest; the guest thanks them for having him.
#
# Measured on the three feeds that state no host — and it is decisive where talk time is worthless:
#
#   Latent Space   Alex Lupsasca talks 84.5% and performs NO host act. Brandon talks 8.6% and
#                  says "welcome to the AI for Science podcast". Brandon is the host.
#   Planet Money   "hello and welcome to Planet Money. I'm Alexi Horowitz-Gazi" — host + his name.
#   NVIDIA         the cluster LABELLED "Nicolas Cerisier" says "I'm Noah Kravitz. My guest is
#                  Nicolas Serissier" — the shipped labels were swapped, and the conversation
#                  is what says so.
#
# The host usually announces himself and names his guest in one breath, which yields both roles and
# both names from a single utterance.
_HOST_SPEECH_ACTS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bwelcome (?:back )?to (?:the |my |our )?\w+",
        r"\bi'?m your host\b",
        r"\b(?:my|our) guests? (?:today )?(?:is|are)\b",
        r"\b(?:joining|with) (?:me|us) (?:today|now|this week)\b",
        r"\bthanks? (?:so much )?for (?:coming on|joining me|joining us|being here)\b",
        r"\bthis week on (?:the )?\w+",
    )
]
# NOTE (#1228) — a "floor-managing" host act (a co-host who only self-introduces on a no-host feed
# but directs the show, "Let's get into this week's news") was TRIED as a recall lever and REVERTED.
# On the prod-v2 corpus (90 eps, `relabel_corpus.py --llm none`) the tightened, nameability-gated
# pattern promoted ZERO voices, while the untightened form regressed real episodes (crowned an
# anonymous voice a host on Latent Space; painted host "Natalie Kitroeff" onto guest Robert Pape on
# The Daily — show-directing boilerplate like "we'll be right back" smears across diarization
# clusters). Inert on real data + precision-dangerous ⇒ not worth the code path (#876). The
# co-host-on-a-no-host-feed case stays the documented precision boundary (roster leaves the role
# unknown rather than risk a wrong name); revisit only with the #1189 human-GT fixtures.
_GUEST_SPEECH_ACTS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        # "thanks/thank you [so much | very much] for having me" — the intensifier is optional AND
        # may be "very much", not only "so much". "Thank you very much for having me" (The Daily's
        # guest Robert Pape) matched NEITHER old fixed pattern, so the dominant guest was never
        # flagged and community-1's clustering then crowned him a host (#1169).
        r"\b(?:thanks?|thank you)(?:\s+(?:so|very)\s+much)? for having me\b",
        r"\b(?:glad|happy|great|good) to be (?:here|on|back)\b",
    )
]
# The host hands the floor to someone, BY NAME. "My guest today is Brian Chesky" is only one of the
# ways they do it, and knowing only that phrasing left 5.2% of the corpus's talk anonymous —
# measured by `scripts/audit/attribution_ceiling.py`. Planet Money is full of it: a narrated desk
# where the host introduces reporter after reporter ("joined by", "here with me is") and every one
# of them came out as SPEAKER_NN.
#
# The host also often names TWO, each behind their employer's possessive: "My guests today are Red
# Hat's Chris Wright and NVIDIA's Justin Boitano" — which a single greedy capture turned into one
# person with that entire string as their name.
# The cue vocabularies are factored into shared bodies (ADR-139) so the case-blind, metadata-
# anchored variants (roster.py `_voice_named_by_the_introduction`) are built from the SAME words and
# cannot drift from these capitalized forms.
#
# Narrated-desk hand-off: The Daily / Planet Money / The Journal introduce a colleague in the third
# person — "today, my colleague Claire Cain Miller…". The possessive + "colleague" anchor keeps it
# from a bare topical mention. Role-title hand-offs ("Pentagon reporter Eric Schmitt talks us
# through…") are caught by the name-first verb tail, which is host-gated and safe to keep looser.
CUE_FIRST_BODY = (
    r"(?:my|our)\s+guests?\s+(?:today\s+)?(?:is|are)"
    r"|joined\s+(?:today\s+)?by"
    r"|joining\s+(?:me|us)(?:\s+(?:today|now|this\s+week))?\s+(?:is|are)"
    r"|(?:i'?m|we'?re)\s+(?:here\s+)?(?:joined\s+)?with"
    r"|(?:please\s+)?welcome\s+(?:back\s+)?"
    r"|here\s+with\s+me\s+(?:is|are)"
    r"|(?:my|our)\s+colleague"
)
# Past-tense hand-off ("i sat down with X", "we spoke with X"). A real introduction ONLY as a
# head-of-episode cold-open; mid-show it describes a PAST conversation and would misattribute the
# named person to whatever voice happens to speak next (a recap is not an intro). Kept separate so
# the roster can gate it to the first turns AND a host introducer (3rd advisor review).
CUE_FIRST_PAST_BODY = r"(?:i|we)\s+(?:spoke|talked|sat\s+down)\s+with"
_GUEST_INTRODUCED_BY_HOST = re.compile(
    rf"\b(?:{CUE_FIRST_BODY})\s+(?:the\s+|our\s+)?(?P<names>{_NAMES})",
    re.IGNORECASE,
)

# ...and the same introduction with the NAME FIRST. Every cue above expects "cue, then name"
# ("joined by Jia Li"), and hosts phrase it the other way round just as often:
#
#     [NVIDIA AI Podcast] "Welcome to the NVIDIA AI podcast. I'm Noah Kravitz.
#                          Jia Li is with us today."      <- introduced, and we heard nothing
#
# The cue still has to be there — the name alone proves nothing, or every person an episode
# discusses becomes a speaker. It is the cue that makes it an introduction.
# Name-first tail (ADR-139). The last two lines are narrated-desk report verbs — "…Farnaz Fassihi
# explain…", "Eric Schmitt talks us through…", "Sydney Baloue reports…". Host-gated (only read on a
# host-hint voice), so a topical "X explains that…" in a guest's own answer does not reclaim a name.
# Intro tails ("Jia Li is with us", "…joins me"): a first-person address, safe to resolve against
# the full stated set.
NAME_FIRST_TAIL = (
    r"(?:is|are)\s+(?:here\s+)?with\s+(?:me|us)"
    r"|(?:is|are)\s+(?:my|our)\s+guests?"
    r"|(?:is|are)\s+joining\s+(?:me|us)"
    r"|joins?\s+(?:me|us)"
)
# Narrated-desk REPORT verbs ("Farnaz Fassihi explains…", "Sydney Baloue reports…"). These ALSO
# match a purely TOPICAL mention on a host's own sentence ("Sam Altman explains it best in his
# blog"), so on the case-blind match-form path they are resolved only against CORROBORATED refs
# (detected guests + known hosts) — never a bare metadata SUBJECT (3rd advisor review).
NAME_FIRST_REPORT_TAIL = (
    r"explains?|reports?|tells\s+us|walks\s+us\s+through|talks\s+us\s+through"
    r"|takes\s+us\s+(?:through|inside)|breaks\s+(?:it\s+|this\s+)?down"
)
_GUEST_INTRODUCED_NAME_FIRST = re.compile(
    # Tolerate an ASR comma between the name and the verb ("Eric Schmitt, talks us through…").
    rf"(?P<names>{_NAMES})\s*,?\s+(?:{NAME_FIRST_TAIL}|{NAME_FIRST_REPORT_TAIL})",
    re.IGNORECASE,
)

# The host greets a just-introduced guest BY NAME: "Jody Rosen, welcome to the show",
# "Nic Harrigan, thanks so much for coming on". Name-then-greeting — the mirror of the cue-first
# forms, and the ordering a narrated interview show (The Daily) actually uses to bring a guest in.
GREETED_TAIL = (
    r"welcome\b"
    r"|thanks?(?:\s+so\s+much)?\s+for\s+(?:coming|joining|being)"
    r"|thank\s+you(?:\s+so\s+much)?\s+for\s+(?:coming|joining|being)"
)
_GUEST_GREETED = re.compile(
    rf"(?P<names>{_NAMES})\s*,\s*(?:{GREETED_TAIL})",
    re.IGNORECASE,
)

# "I'm Coming Out", "I'm Not Sure" — the self-introduction regex matches any capitalised run, and
# the ASR capitalises plenty of things that are not people. Found in The Daily, where a voice was
# recorded as introducing itself as "Coming Out".
_NOT_A_NAME_TOKEN = frozenset(
    {
        "coming",
        "going",
        "not",
        "sorry",
        "sure",
        "just",
        "here",
        "there",
        "really",
        "gonna",
        "trying",
        "talking",
        "telling",
        "saying",
        "looking",
        "thinking",
        "working",
        "wondering",
        "curious",
        "afraid",
        "worried",
        "excited",
        "glad",
        "happy",
        "good",
        "great",
        "fine",
        "okay",
        "back",
        "out",
        "in",
        "so",
        "very",
        "always",
        "still",
        "also",
        "the",
        "a",
        "an",
        # Sentence-opening discourse markers the ASR capitalises at a turn boundary and the greeting
        # regexes then sweep into a 2-word "name" ("So Nick, welcome" -> "So Nick", "But Sun, thanks
        # for coming" -> "But Sun"). They are ordinary English words, so they belong to this set by
        # its own contract. Any-position match means a real surname colliding with one ("Andrew
        # Look") is also dropped — accepted per "a wrong label is worse than an unnamed voice".
        "but",
        "and",
        "well",
        "now",
        "then",
        "because",
        "plus",
        "anyway",
        "look",
        "yeah",
        # Turn-opener contractions (BUG 5): the ASR capitalises the first word of a turn, and a
        # screenplay-line speaker-label reader that just grabs whatever precedes the colon can pick
        # up the contraction itself as the "speaker" ("I'm: You'll never find a harder worker...").
        # Listed in both straight-quote and curly-quote (’) spellings since transcript punctuation
        # restoration can emit either, and neither ``looks_like_a_person_name`` nor
        # ``is_publishable_speaker_name`` normalises the apostrophe before this lookup (``.strip()``
        # only trims the ends of the token, not an internal one).
        "i'm",
        "i’m",
        "i've",
        "i’ve",
        "i'll",
        "i’ll",
        "i'd",
        "i’d",
        "you're",
        "you’re",
        "you'll",
        "you’ll",
        "you've",
        "you’ve",
        "you'd",
        "you’d",
        "we're",
        "we’re",
        "we've",
        "we’ve",
        "we'll",
        "we’ll",
        "they're",
        "they’re",
        "they've",
        "they’ve",
        "it's",
        "it’s",
        "that's",
        "that’s",
        "there's",
        "there’s",
        "here's",
        "here’s",
        "let's",
        "let’s",
        "don't",
        "don’t",
        "doesn't",
        "doesn’t",
        "didn't",
        "didn’t",
        "can't",
        "can’t",
        "won't",
        "won’t",
        "wouldn't",
        "wouldn’t",
        "shouldn't",
        "shouldn’t",
        "couldn't",
        "couldn’t",
        "isn't",
        "isn’t",
        "aren't",
        "aren’t",
        "wasn't",
        "wasn’t",
        "weren't",
        "weren’t",
        "haven't",
        "haven’t",
        "hasn't",
        "hasn’t",
    }
)


def looks_like_a_person_name(name: str) -> bool:
    """A capitalised run is not a name if any of its tokens is an ordinary English word.

    "I'm Coming Out" is not a person. Requires First-Last shape and no stop-token.
    """
    toks = (name or "").split()
    if len(toks) < 2:
        return False
    return not any(t.lower().strip(".,'’") in _NOT_A_NAME_TOKEN for t in toks)


# Capitalised single words that follow "I'm <Cap>" but are NOT names — the "I'm American" class.
# The self-intro regex is case-SENSITIVE, so lowercase adjectives ("I'm ready") never reach here;
# the residual risk is demonyms / religion / politics, which do get capitalised.
_NOT_A_MONONYM = frozenset(
    {
        "american",
        "british",
        "canadian",
        "australian",
        "irish",
        "scottish",
        "english",
        "welsh",
        "german",
        "french",
        "italian",
        "spanish",
        "portuguese",
        "chinese",
        "japanese",
        "korean",
        "indian",
        "russian",
        "mexican",
        "brazilian",
        "dutch",
        "swedish",
        "norwegian",
        "danish",
        "european",
        "african",
        "asian",
        "latino",
        "latina",
        "hispanic",
        "jewish",
        "christian",
        "catholic",
        "protestant",
        "muslim",
        "hindu",
        "buddhist",
        "atheist",
        "republican",
        "democrat",
        "democratic",
        "conservative",
        "liberal",
        "progressive",
        "independent",
    }
)


# Honorifics. The self-intro regex `\bI'?m\s+([A-Z][\w'’\-]+…)` stops at the period in "I'm Dr.
# Jane Smith", capturing the bare title "Dr" — which must never become a speaker name, and must not
# count as a distinct self-introduction (else "I'm Dr. X … I'm X" reads as a two-person montage).
HONORIFIC_TITLES = frozenset(
    {
        "dr",
        "doctor",
        "mr",
        "mrs",
        "ms",
        "miss",
        "prof",
        "professor",
        "sir",
        "dame",
        "lord",
        "lady",
        "rev",
        "reverend",
        "fr",
        "father",
        "sen",
        "senator",
        "rep",
        "gov",
        "governor",
        "pres",
        "president",
        "judge",
        "justice",
        "capt",
        "captain",
        "gen",
        "sgt",
        "col",
    }
)


def is_plausible_mononym(token: Optional[str]) -> bool:
    """True if a one-token self-intro ("I'm Brandon") is a plausible name, not "I'm American".

    Accepts a capitalised alphabetic token (apostrophes/hyphens allowed) that is neither an
    ordinary word (:data:`_NOT_A_NAME_TOKEN`), a demonym/religion/politics label
    (:data:`_NOT_A_MONONYM`), nor a bare honorific (:data:`HONORIFIC_TITLES`, the "I'm Dr." case).
    Used to let a voice's own single-name self-introduction name it on feeds with no host anchor —
    without re-admitting the false positives the guard exists for.
    """
    t = (token or "").strip(" .,")
    if not re.fullmatch(r"[A-Z][A-Za-z'’\-]+", t):
        return False
    tl = t.lower()
    return tl not in _NOT_A_NAME_TOKEN and tl not in _NOT_A_MONONYM and tl not in HONORIFIC_TITLES


def is_publishable_speaker_name(name: Optional[str]) -> bool:
    """Final reject filter for a name about to be painted on a diarized voice (ADR-134 shared core).

    Every extraction path (self-intro, host-pool, greeting reader, strategy snap, LLM, metadata)
    converges on the roster; a name that carries a sentence-opener the ASR capitalised at a turn
    boundary ("But Sun", "So Nick", bare "But") is not a person, and a wrong label is worse than an
    unnamed voice. This is the last gate before publish, so no single path can bypass it.

    Deliberately WEAKER than :func:`is_plausible_mononym` for a one-token name: it rejects only a
    token that is a *known* non-name word, and does NOT require a capitalised first letter — else a
    real lowercase handle already vouched by a trusted source ("swyx") would be thrown away. The
    contract is "drop the garbage", not "re-validate every accepted name".
    """
    nm = name or ""
    toks = nm.split()
    if len(toks) >= 2:
        return looks_like_a_person_name(nm)
    if len(toks) == 1:
        tl = toks[0].lower().strip(".,'’")
        return (
            tl not in _NOT_A_NAME_TOKEN and tl not in _NOT_A_MONONYM and tl not in HONORIFIC_TITLES
        )
    return False


def roles_from_conversation(voice_texts: Optional[Dict[str, str]]) -> Dict[str, str]:
    """``{voice: "host" | "guest"}`` for the voices that PERFORM one of the two roles.

    Complements the metadata; it does not replace it. Used when the feed states no host, and as a
    cross-check when it does. Silent about voices that perform neither — those stay unknown, which
    is the safe direction (#876).
    """
    out: Dict[str, str] = {}
    for voice, text in (voice_texts or {}).items():
        if not text:
            continue
        if any(p.search(text) for p in _HOST_SPEECH_ACTS):
            out[voice] = "host"
        elif any(p.search(text) for p in _GUEST_SPEECH_ACTS):
            out[voice] = "guest"
    return out


def guests_introduced_by_the_host(voice_texts: Optional[Dict[str, str]]) -> Set[str]:
    """Names the host introduces as guests ("My guest today is Brian Chesky").

    Splits a multi-guest introduction into people. "My guests today are Red Hat's Chris Wright and
    NVIDIA's Justin Boitano" is two guests, each behind an employer's possessive — and it was being
    recorded as ONE person with that entire string as their name.

    Reads the introduction in BOTH directions. Every cue we knew put the name after it ("joined by
    Jia Li"), and hosts say it the other way round just as often — "Jia Li is with us today" — so a
    whole class of on-air introduction was going in the bin while the episode sat at 75% of its talk
    attributable to nobody. An on-air introduction is a stated fact from the conversation and cannot
    invent anybody, which is exactly what makes it worth reading properly.
    """
    out: Set[str] = set()
    for text in (voice_texts or {}).values():
        matches = list(_GUEST_INTRODUCED_BY_HOST.finditer(text or ""))
        matches += list(_GUEST_INTRODUCED_NAME_FIRST.finditer(text or ""))
        matches += list(_GUEST_GREETED.finditer(text or ""))
        for m in matches:
            for raw in _NAME_RE.findall(m.group("names")):
                name = _clean_stated_name(raw)
                # Same person-name guard the self-intro and intro-reader paths apply: a run with an
                # ordinary English word in it ("So Nick") is ASR noise the greeting regex swept up.
                if (
                    len(name.split()) >= 2
                    and not has_org_markers(name)
                    and looks_like_a_person_name(name)
                ):
                    out.add(name)
    return out


def detect_hosts_from_feed(
    feed_title: Optional[str],
    feed_description: Optional[str],
    feed_authors: Optional[List[str]] = None,
    nlp: Optional[Any] = None,
) -> Set[str]:
    """Detect host names from feed-level metadata.

    Order of authority: the feed's own HOST STATEMENT ("Hosted by ..."), then non-organisation
    author tags, then NER over the title/description as a last resort. NER is last because it cannot
    tell a host from anyone else the description happens to mention — on Latent Space it returns a
    list of past guests, and on Planet Money it returns the word "Wanna".
    """
    stated = hosts_from_feed_statement(feed_title, feed_description)
    if stated:
        logger.debug("Hosts stated by the feed: %s", sorted(stated))
        return stated

    hosts: Set[str] = set()

    if feed_authors:
        for author in feed_authors:
            if author and author.strip():
                author_clean = author.strip()
                if "<" in author_clean and ">" in author_clean:
                    author_clean = author_clean.split("<")[0].strip()
                # One RSS author tag routinely names SEVERAL people (#1652). Latent Space ships
                # ``"Brandon Anderson, RJ Honicky, and Latent.Space"`` in a single
                # ``<itunes:author>``. Kept whole it can never match a voice — the roster
                # compares per-name — so the known-hosts fallback was inert for every
                # multi-author feed. That is the fallback that would otherwise have limited
                # #1646's damage on exactly those shows.
                for candidate in split_author_names(author_clean):
                    if not candidate:
                        continue
                    if is_network_or_org_author(candidate):
                        logger.debug(
                            "RSS author '%s' looks like a network/organisation, not a host; "
                            "treating as publisher metadata rather than host",
                            candidate,
                        )
                    else:
                        hosts.add(candidate)
        if hosts:
            logger.debug(
                "Detected hosts from RSS author tags (author/itunes:author/itunes:owner): %s",
                list(hosts),
            )
            return hosts
        if feed_authors:
            _log(
                "info",
                "All RSS author(s) treated as organisation(s); host detection will use "
                "NER from feed title/description, episode-level authors, or config known_hosts",
            )

    # Last resort: NER over the TITLE only, and only for real First-Last names.
    #
    # NOT the description. NER cannot tell a host from anyone else a paragraph mentions, and the
    # description is exactly where the other people are: Latent Space lists its PAST GUESTS (Bret
    # Taylor, Chris Lattner, George Hotz), and NER offered all of them as hosts of the show. Planet
    # Money's description opens "Wanna see a trick?" and NER offered "Wanna".
    #
    # A title does not list guests. And when the feed neither states its hosts nor carries a
    # personal author tag, the right answer is NO HOSTS — the roster then leaves those voices
    # unnamed, the safe direction (#876). Guessing is what put an advertiser's name on a podcast.
    if nlp and feed_title:
        for name, _score in _extract_person_entities(feed_title, nlp):
            clean = (name or "").strip()
            if len(clean.split()) >= 2 and not has_org_markers(clean):
                hosts.add(clean)
        if hosts:
            logger.debug("Detected hosts via NER from the feed TITLE: %s", sorted(hosts))

    return hosts

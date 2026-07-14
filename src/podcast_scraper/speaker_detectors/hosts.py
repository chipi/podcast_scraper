"""Host detection from feed metadata and transcript intro."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

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
    if len(n.split()) < 2:  # mononym ("Colossus", "NPR") — not a "First Last" host name
        return True
    return False


def looks_like_publisher(name: str) -> bool:
    """True when a name is a network / publisher / organisation rather than a person.

    Combines the known-network denylist with the generic org-marker + news-outlet-suffix regex.
    Unlike :func:`is_network_or_org_author` this does NOT apply the mononym rule, so a
    single-token real person (Oprah, Sting) is kept — use it to strip publishers from
    already-resolved person surfaces (key people, host/guest roles) without dropping people.
    """
    return is_known_network(name) or has_org_markers(name)


# Host self-introduction in the transcript intro, e.g. "I'm Patrick O'Shaughnessy".
# The name sub-pattern allows apostrophes/hyphens so it captures full surnames
# ("O'Shaughnessy", "Jean-Luc") but NOT periods — a period ends the self-intro sentence, so
# excluding it stops the match from absorbing the next sentence ("…O'Shaughnessy. My guest").
_HOST_SELF_INTRO = re.compile(r"\bI'?m\s+([A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+){0,3})")


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
        return name
    return None


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

    intro_patterns = [
        r"I'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"This is\s+[^.]+\s+I'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"Welcome to\s+[^.]+\s+I'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
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
_NAME = r"[A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+)+"
_NAMES = rf"{_NAME}(?:\s*(?:,|and|&)\s*{_NAME})*"
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
_GUEST_SPEECH_ACTS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bthanks? (?:so much )?for having me\b",
        r"\bthank you for having me\b",
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
_GUEST_INTRODUCED_BY_HOST = re.compile(
    r"\b(?:"
    r"(?:my|our)\s+guests?\s+(?:today\s+)?(?:is|are)"
    r"|joined\s+(?:today\s+)?by"
    r"|joining\s+(?:me|us)(?:\s+(?:today|now|this\s+week))?\s+(?:is|are)"
    r"|(?:i'?m|we'?re)\s+(?:here\s+)?(?:joined\s+)?with"
    r"|(?:i|we)\s+(?:spoke|talked|sat\s+down)\s+with"
    r"|(?:please\s+)?welcome\s+(?:back\s+)?"
    r"|here\s+with\s+me\s+(?:is|are)"
    r")"
    rf"\s+(?:the\s+|our\s+)?(?P<names>{_NAMES})",
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
    """
    out: Set[str] = set()
    for text in (voice_texts or {}).values():
        for m in _GUEST_INTRODUCED_BY_HOST.finditer(text or ""):
            for raw in _NAME_RE.findall(m.group("names")):
                name = _clean_stated_name(raw)
                if len(name.split()) >= 2 and not has_org_markers(name):
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
                if author_clean:
                    if is_network_or_org_author(author_clean):
                        logger.debug(
                            "RSS author '%s' looks like a network/organisation, not a host; "
                            "treating as publisher metadata rather than host",
                            author_clean,
                        )
                    else:
                        hosts.add(author_clean)
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

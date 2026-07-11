"""Host detection from feed metadata and transcript intro."""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Set

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


def detect_hosts_from_feed(
    feed_title: Optional[str],
    feed_description: Optional[str],
    feed_authors: Optional[List[str]] = None,
    nlp: Optional[Any] = None,
) -> Set[str]:
    """Detect host names from feed-level metadata."""
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

    if nlp:
        if feed_title:
            title_persons = _extract_person_entities(feed_title, nlp)
            hosts.update(name for name, _ in title_persons)
        if feed_description:
            desc_persons = _extract_person_entities(feed_description, nlp)
            hosts.update(name for name, _ in desc_persons)
        if hosts:
            logger.debug("Detected hosts via NER from feed metadata: %s", list(hosts))

    return hosts

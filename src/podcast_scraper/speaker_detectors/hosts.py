"""Host detection from feed metadata and transcript intro."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..kg.speaker_coherence import same_person
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
    r"times|journal|tribune|gazette|herald|chronicle|magazine|quarterly|newspaper|gmbh|plc|"
    # INSTITUTIONS. A think tank, university or committee is not a person, and none of the
    # commercial markers above catch one: "Mercatus Center at George Mason University" has no
    # pipe, no digit, no "Media"/"Network". It was therefore eligible to be a HOST — 41 Person
    # nodes on the production snapshot carry it with role="host", and the feed host detector
    # still emits it for *Conversations with Tyler* today, so a `relabel_only` repair would swap
    # the show's name for this one rather than for a person.
    #
    # Measured before adding, because a rule invented from one example is how this arc kept
    # going wrong: across 4,307 roster entries and 13,642 Person nodes, these tokens match 5
    # distinct names — Mercatus Center…, Rindman University, Alexander Committee, PC Alexander
    # Committee, Boston College — and every one is an organisation. No real person is caught.
    r"centers?|centres?|universit(?:y|ies)|colleges?|institutes?|foundations?|"
    r"committees?|councils?|associations?|societies|society|museums?|librar(?:y|ies)|"
    # A BROADCAST BRAND suffix ("China Plus", the author of Biz Talk and Round Table China). Across
    # 17,949 person names on the production snapshot (kg Person nodes + rosters) the only name
    # carrying the token is `China Plus` itself.
    r"plus)\b",
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


#: A trailing "with <Name>" in a TITLE names the host, not the show — "Invest Like the Best with
#: Patrick O'Shaughnessy". Stripped before comparing a candidate against the show's name, or the
#: host would look like part of it.
_TITLE_WITH_SUFFIX = re.compile(r"\s+with\s+.+$", re.IGNORECASE)


#: A leading article is not part of a show's name for comparison purposes. Without dropping it,
#: "Trivium China" was not recognised as the prefix of "The Trivium China Podcast" and the show
#: seated itself as its own host.
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+")


def _fold_title(text: Optional[str]) -> str:
    """Lowercase, drop punctuation and any leading article, collapse whitespace."""
    folded = " ".join(re.sub(r"[^\w\s]", " ", str(text or "").lower()).split())
    return _LEADING_ARTICLE.sub("", folded)


def names_the_show(candidate: str, feed_title: Optional[str]) -> bool:
    """True when *candidate* is the SHOW's own name rather than a person on it (#2064).

    Measured on production: 19 speaker entries across 279 episodes are a show seated as a host —
    "Africa Tech Summit", "Trivium China", "Conversations with Tyler", "Machine Learning Street".
    None is caught by :func:`is_network_or_org_author`, because none carries an org marker, is a
    known network, or is a mononym. The feed's own title is the one piece of evidence that tells a
    show apart from a person, and it is already on the artifact — so this is a comparison, not a
    wordlist that needs feeding forever.

    The trailing ``with <Name>`` is removed from the title first, because that is exactly where a
    real host lives: without it "Patrick O'Shaughnessy" would look like part of "Invest Like the
    Best with Patrick O'Shaughnessy" and the fix would throw away the 18 legitimate cases in the
    same sample.

    A leading article is dropped from both sides first, so "Trivium China" is recognised as
    the prefix of "The Trivium China Podcast" — the show's own name, not a presenter.

    Matches the whole show name or a multi-token PREFIX of it ("Machine Learning Street" of
    "Machine Learning Street Talk"). A prefix rather than any substring, so a host whose name
    happens to appear late in a title is untouched. No title means no opinion: absence of evidence
    is not evidence that the candidate is the show.
    """
    cand = _fold_title(candidate)
    title = _fold_title(feed_title)
    if not cand or not title:
        return False
    show = _fold_title(_TITLE_WITH_SUFFIX.sub("", str(feed_title or ""))) or title
    if cand == title or cand == show:
        return True
    cand_tokens = cand.split()
    if len(cand_tokens) < 2:
        return False
    return show.split()[: len(cand_tokens)] == cand_tokens


def normalize_host_names(names: Iterable[str], *, feed_title: Optional[str] = None) -> Set[str]:
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
            # THE DANGLING BRACKET THE SPLIT LEFT BEHIND, and nothing else. "…(with Aaron Levie)"
            # splits to "Aaron Levie)", and that rode all the way to publication: 14 production
            # voices carry one, every one `source=known_hosts`, so "Aaron Levie)" and "Aaron Levie"
            # are two different people to every downstream id, slug and graph join.
            #
            # DELIBERATELY NOT `_sanitize_person_name`, which was the first attempt. It strips ALL
            # non-word characters, and that does two unacceptable things here: it rewrites a real
            # name ("Martin Luther King, Jr." -> "Martin Luther King Jr"), and — far worse — it
            # launders an ORGANISATION past the filter that was correctly rejecting it
            # ("Colossus | Investing & Business Podcasts" -> "Colossus Investing", which then reads
            # as a person). Reject, do not strip: trim the unmatched bracket the split created and
            # leave every other character alone.
            candidate = candidate.strip()
            if candidate.endswith(")") and "(" not in candidate:
                candidate = candidate[:-1].strip()
            if candidate.startswith("(") and ")" not in candidate:
                candidate = candidate[1:].strip()
            if not candidate or is_network_or_org_author(candidate):
                continue
            # #2064: the SHOW is not a person on it. Only checkable when the caller knows the
            # title, which is why it is a keyword rather than a silent no-op.
            if names_the_show(candidate, feed_title):
                logger.debug(
                    "host candidate '%s' names the show '%s' — not a person", candidate, feed_title
                )
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
#
# THE ROLE PHRASE IS THE COMMONEST OPENING IN THE CORPUS AND THIS PATTERN COULD NOT SEE IT.
# "Hello, and welcome to the NVIDIA AI podcast. I'm your host, Noah Kravitz" — `your` is lowercase,
# so the capitalised run never starts and the scanner returned nothing. Measured over the 136
# production episodes that end with no named speaker: the pattern below matched 0 of them before
# the role phrase was allowed, and 25 of NVIDIA's 31 after.
#
# "the host never self-introduces" was therefore a property of THIS REGEX, not of the corpus — and
# it was reported as a fact about the data for most of a day.
# "I am" is the same statement as "I'm" and was not in the alternation. Macro Musings opens
# "Welcome to Macro Musings. I am your host, David Beckworth" on 37 voices, none of which this
# scanner could see. Same guards apply to it as to the contraction — this widens the FORM, not the
# evidence.
_HOST_SELF_INTRO = re.compile(
    r"\b(?:I'?m|I am|[Mm]y name is)\s+"
    r"(?:(?:your|the)\s+(?:co-?)?host,?\s+)?"
    r"([A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+){0,3})"
)

# "it's <Full Name> with <Show>" — the branded open. Ground Truths: "Hello, it's Eric Topol with
# Ground Truths."
#
# DELIBERATELY MUCH NARROWER THAN THE PATTERNS ABOVE, because a bare "it's <Cap>" is not a
# self-introduction at all — "It's Monday", "It's Christmas", "It's OpenAI" all match that shape.
# Three conditions together make it one, and none of them is optional:
#   - a FIRST-LAST name, not a mononym (the weakest possible evidence in the weakest form);
#   - a following "with"/"from"/"for", which is what turns a statement of fact into a byline;
#   - the thing fronted must be THIS SHOW, which is why the caller has to supply `feed_title`.
# That is the same reason `_THIS_IS_INTRO` stays metadata-gated: "this is Planet Money" is a
# station ident, and so is most of what "it's X" produces without these three.
#
# THE SHOW CONDITION IS NOT DECORATION — IT IS THE WHOLE GUARD. `it's <Name> from <Company>` is the
# shape of a SPONSOR READ, and the first sweep of this pattern over the corpus found one:
# "Hi, it's Michael Sullivan from Wirecutter, the product recommendation service from the New York
# Times" on Hard Fork. An ad narrator says their own name by design, which is what makes the
# most-trusted signal the easiest to poison. Eric Topol fronts "Ground Truths" and that IS the
# show; Michael Sullivan fronts Wirecutter and that is not. With no feed title there is no way to
# tell them apart, so with no feed title this form does not fire at all.
_HOST_BRANDED_INTRO = re.compile(
    r"\b[Ii]t'?s\s+([A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+){1,2})\s+(?:with|from|for)\s+"
    r"([A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*){0,3})"
)

# "with me, <Name>" / "and me, <Name>" — the British broadcast idiom for naming ONESELF.
# "Welcome to The Rest Is Politics: Leading with me, Alastair Campbell" is spoken BY Campbell.
# 11 of that feed's 12 unnamed episodes open this way; the pattern above matches none of them.
#
# THE COMMA IS REQUIRED AND "joining me" IS EXCLUDED, deliberately: "joining me, <Name>" introduces
# somebody ELSE, and admitting it would paint a guest's name onto the host's voice — the exact
# direction of error this module exists to prevent.
_HOST_WITH_ME_INTRO = re.compile(
    r"\b(?:with|and)\s+me,\s+([A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+){0,2})"
)


def _branded_intro_matches(head: str, feed_title: Optional[str]) -> List["re.Match[str]"]:
    """`it's <Name> with <Show>` matches, but only where the fronted thing IS this show."""
    if not feed_title:
        return []
    return [m for m in _HOST_BRANDED_INTRO.finditer(head) if names_the_show(m.group(2), feed_title)]


def extract_self_introduced_host(
    transcript_text: Optional[str],
    *,
    intro_chars: int = 2000,
    feed_title: Optional[str] = None,
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
    head = transcript_text[:intro_chars]
    # Both forms, in one pass with the SAME guards below. Two scanners with two guard sets is how
    # the sibling scanners drifted apart before (#876).
    matches = (
        list(_HOST_SELF_INTRO.finditer(head))
        + list(_HOST_WITH_ME_INTRO.finditer(head))
        + _branded_intro_matches(head, feed_title)
    )
    for match in matches:
        # Collapse runs of whitespace: word-level ASR segments join as "Amanda  Aronchik" — a
        # different person id from "Amanda Aronchik" on every other surface.
        name = " ".join(match.group(1).split()).strip(" .,")
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
    transcript_text: Optional[str],
    *,
    intro_chars: int = 2000,
    feed_title: Optional[str] = None,
) -> List[str]:
    """Every DISTINCT person-name a voice introduces itself as ("I'm <Name>"), same filtering as
    :func:`extract_self_introduced_host` (network bumpers + ordinary-word runs skipped).

    One physical speaker introduces itself once. Two or more distinct self-introductions in a single
    diarization cluster is the signature of a MERGED cluster — a cold-open montage that strings
    several hosts' intros together ("I'm Kevin Russo… I'm Casey Noon…") collapses into one voice.
    The caller uses ``len(...) >= 2`` to refuse naming such a cluster after any one of them.

    Reads BOTH intro forms, like its sibling. Omitting ``_HOST_WITH_ME_INTRO`` here was exactly the
    drift that docstring warns about: a co-host presented as "...and with me, Casey Newton" was
    invisible to every caller of this function while the sibling saw them, so a two-host desk show
    looked like a one-host show with a guest.
    """
    seen: List[str] = []
    lowered: Set[str] = set()
    head = (transcript_text or "")[:intro_chars]
    matches = (
        list(_HOST_SELF_INTRO.finditer(head))
        + list(_HOST_WITH_ME_INTRO.finditer(head))
        + _branded_intro_matches(head, feed_title)
    )
    for match in matches:
        # Collapse runs of whitespace: word-level ASR segments join as "Amanda  Aronchik" — a
        # different person id from "Amanda Aronchik" on every other surface.
        name = " ".join(match.group(1).split()).strip(" .,")
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
#: Patterns safe to run over a TITLE as well as a description.
_HOST_PHRASES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        rf"\bhosted\s+by\s+(?P<names>{_NAMES})",
        rf"\bco-?hosts?\s+(?P<names>{_NAMES})",
        rf"\bjournalists?\s+(?P<names>{_NAMES})",
        rf"\bwith\s+(?P<names>{_NAME})\s*$",  # the show title: "... with Patrick O'Shaughnessy"
    )
]

#: DESCRIPTION-ONLY. "Joe Weisenthal and Tracy Alloway explore..." / "Katie Martin, Robert Armstrong
#: and other markets nerds at the Financial Times explain..." — names, then a presenting verb, with
#: bounded filler so the verb belongs to THESE names.
#:
#: It must never run over a TITLE (#2064). A title is a NAME, not a sentence, and `_PRESENTS`
#: contains ordinary words that end show names — so "Machine Learning Street Talk" parsed as the
#: names "Machine Learning Street" followed by the verb "Talk", and the show became the host of
#: itself on 6 production episodes. The host does appear in a title, but in a different shape:
#: "... with Patrick O'Shaughnessy", which the `with` pattern above already reads.
_HOST_PHRASES_DESCRIPTION_ONLY = [
    re.compile(rf"(?P<names>{_NAMES})[\w\s,'’\-]{{0,60}}?\s+{_PRESENTS}", re.IGNORECASE)
]
_NAME_RE = re.compile(_NAME)


_ARTICLE_BEFORE = re.compile(r"\b(?:the|of)\s+$", re.IGNORECASE)
_PLACE_PREPOSITION = re.compile(r"(?:At|From|In|On)\s+", re.IGNORECASE)


def hosts_from_feed_statement(
    feed_title: Optional[str], feed_description: Optional[str]
) -> Set[str]:
    """Hosts the feed EXPLICITLY names ("Hosted by X and Y"), rather than every person it mentions.

    This is the authoritative source: the show says who presents it. Only used for the names inside
    the host phrase, so a description that also lists past guests cannot smuggle them in.
    """
    return _feed_statement(feed_title, feed_description)[0]


def _feed_statement(
    feed_title: Optional[str], feed_description: Optional[str]
) -> Tuple[Set[str], bool]:
    """``(hosts, rejected)`` — ``rejected`` when a host phrase matched but its names were refused.

    A refused statement means the feed's own words point at something that is not a person
    (`Americas Online`, the tail of "Council of the Americas Online team brings"). The caller must
    then name NO host rather than fall back to the author tag or a later match: those fallbacks
    added real names (Carin Zissis on 52 voices) that the single-seat host rule then placed on
    guests' answers, clips and other presenters' episodes (#2075, advisor review). This work may
    remove a wrong host; it must not add one.
    """
    title_lower = (feed_title or "").lower()
    out: Set[str] = set()
    rejected = False
    for is_title, text in ((True, feed_title or ""), (False, feed_description or "")):
        if not text.strip():
            continue
        patterns = _HOST_PHRASES if is_title else _HOST_PHRASES + _HOST_PHRASES_DESCRIPTION_ONLY
        for pat in patterns:
            m = pat.search(text)
            if not m:
                continue
            # A person's name does not follow "the" or "of": "…Council of the Americas Online team
            # brings…" made `Americas Online` the host of Latin America in Focus.
            after_article = bool(_ARTICLE_BEFORE.search(text[: m.start("names")]))
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
                # The rejections this work ADDS run only on a name the checks above let through,
                # and they refuse the whole statement (no host, no fallback):
                # - the tail of a longer proper noun (after "the"/"of");
                # - a place or body: "At Carnegie India, our diverse lineup of experts will host…";
                # - a nationality: "hosted by Anglo Canadian transplant to Colombia…".
                # The LAST token only: `_NOT_A_MONONYM` holds demonyms and religion/politics
                # labels, several of which are ordinary given names ("Christian"). Checking every
                # token would throw away the whole statement of a feed hosted by Christian Schmidt.
                # "Anglo Canadian transplant", the case this catches, ends on the demonym.
                if (
                    after_article
                    or _PLACE_PREPOSITION.match(raw.strip())
                    or clean.split()[-1].lower().strip(".,'’") in _NOT_A_MONONYM
                ):
                    rejected = True
                    continue
                out.add(clean)
    return (set() if rejected else out), rejected


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
    # "I'm chatting with X" / "we're speaking with X". The form above only covers a bare
    # "I'm with" / "I'm joined with"; the conversational verbs were absent, and they carry a
    # measurable share of real introductions — 30 correct / 0 wrong over the decidable cases on
    # the production snapshot (5th advisor review).
    r"|(?:i'?m|we'?re)\s+(?:here\s+)?(?:chatting|talking|speaking|sitting\s+down)\s+(?:with|to)"
    # "with us today is X" — the mirror of "here with me is X", which was covered while this was
    # not. 3 correct / 1 wrong; small, and it costs nothing to read.
    r"|with\s+us\s+(?:today\s+)?(?:is|are)"
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
    # "X is here to talk about …" — the host says why the guest came. Distinct from
    # "is here with me": the purpose clause, not the presence clause. 12 correct / 1 wrong.
    r"|(?:is|are)\s+here\s+to\b"
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
        # FUNCTION WORDS — a closed class: pronouns, determiners, auxiliaries, prepositions. No
        # person-name token is one of them, and an ASR stretch that capitalises every word turns
        # prose into a "name" that contains one: `Super Willing To Be`, `One Factor That`, `México
        # She`, `Karin Zesis This` (Latin America in Focus). Measured over the full roster on
        # 4,543 production and validation episode files: the only other change is the show name
        # `Conversations with Tyler` leaving a voice — no person's name is lost. Name particles
        # (`van`, `de`, `al`, `bin`) and words that are also surnames (`do`, `an`) are left out.
        "i",
        "me",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "him",
        "us",
        "them",
        "that",
        "this",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "am",
        "does",
        "did",
        "has",
        "have",
        "had",
        "to",
        "of",
        "on",
        "at",
        "for",
        "with",
        "from",
        "by",
        "into",
        "about",
        "or",
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
        # Found by sweeping every published 2+ token speaker name in the production snapshot for a
        # leading ordinary word this set did not already hold. Exactly four occur, on 8 names:
        # "As Peter" (4), "Before Gene" (2), "Their Erica" (1), "Your  Host  Luisa  Leni" (1).
        # Measured rather than brainstormed, because a wordlist grown by imagination is the
        # treadmill `names_the_show` was written to get off.
        "as",
        "before",
        "after",
        # POSSESSIVE DETERMINERS AS A CLOSED CLASS. "their"/"your" are the two that appear, and the
        # rest of the class is listed with them because it IS a closed class — no possessive
        # determiner is ever a given name, so this is a categorical statement rather than a guess
        # about what might turn up next.
        "my",
        "our",
        "your",
        "his",
        "her",
        "their",
        "its",
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
    if tl in _NOT_A_NAME_TOKEN or tl in _NOT_A_MONONYM or tl in HONORIFIC_TITLES:
        return False
    # A HYPHENATED COMPOUND is judged by its parts. "Pan-African" reached the graph as a guest with
    # 363s of talk time on a freshly ingested episode: `african` is in the demonym list, but
    # `pan-african` is not, and the compound was compared whole. The same shape covers
    # "Afro-Caribbean", "Anglo-Irish", "Sino-American". Checking the parts against the list we
    # already have beats adding one entry per compound, which is a list that needs feeding forever.
    #
    # A hyphenated real surname ("Crebo-Rediker", "Smith-Jones") is unaffected: neither part is a
    # demonym or an ordinary word, so it still passes.
    if "-" in tl:
        parts = [x for x in tl.split("-") if x]
        if any(x in _NOT_A_MONONYM or x in _NOT_A_NAME_TOKEN for x in parts):
            return False
    return True


def drop_non_person_names(names: Iterable[str], feed_title: Optional[str] = None) -> List[str]:
    """Remove publishers and the show's own name from a list of candidate PEOPLE.

    A NAME THAT REACHES `known_hosts` BECOMES A NAME A VOICE MAY BE CALLED, so an organisation in
    that list is an organisation on somebody's quotes. Measured on the production snapshot, 150
    published speaker entries across 144 episodes are a publisher or the show itself, and the paths
    they arrive by are exactly the ones this guards:

        128  source=known_hosts     "Andreessen Horowitz" 60, "Conversations with Tyler" 23,
                                    "Machine Learning Street" 18, "Trivium China" 10
         50  source=llm_resolution  the same strings, reached through the closed candidate list
          5  source=self_intro      one-off ASR garbage ("Boston College", "Rindman University")

    Both predicates already existed and neither was applied here: ``_clean_person_names`` checks
    only ``has_org_markers``, which catches 12 of the 150, and ``is_publishable_speaker_name``
    accepts all 150.

    DELIBERATELY NOT ``is_network_or_org_author``. That one rejects every mononym, and a one-token
    name in this list is a real person — Oprah, Sting, the handle "swyx" — which is the contract
    ``_clean_person_names`` documents and #876 depends on. ``looks_like_publisher`` and
    ``names_the_show`` leave all three alone and still catch every org in the sample.

    No *feed_title* means no opinion about the show's name — absence of evidence is not evidence
    that the candidate is the show.
    """
    out: List[str] = []
    for raw in names or ():
        name = str(raw or "").strip()
        if not name:
            continue
        if looks_like_publisher(name):
            continue
        if feed_title and names_the_show(name, feed_title):
            continue
        out.append(name)
    return out


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
    # A PUBLISHER IS NOT A SPEAKER, whatever path produced it. This gate is the last thing between
    # a name and a voice, and it was passing all 150 organisation names in the sample — including
    # the five that arrive as a transcript self-introduction ("Boston College", "Rindman
    # University"), which no candidate-list filter upstream can see.
    if looks_like_publisher(nm):
        return False
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


#: "<HOST> is joined by <GUEST>" / "<HOST> speaks with <GUEST>" — the host sits BEFORE the cue.
#: Two names may share the slot ("Yoko Li and Justine Moore speak with ...").
#:
#: EXACTLY TWO TOKENS. Nothing here anchors the START of the name, so a three-token run takes
#: whatever capitalised word precedes it — a job title ("a16z Partners Martin Casado speak
#: with..."), or the tail of the episode title running into the description ("...the Future of
#: Forecasting" + "Theo Jaffee speaks with..."). Measured over the 2,256-episode production
#: snapshot, three-token captures were 2 for 2 WRONG and produced no correct host that the
#: two-token form missed, so the extra token buys a defect class and nothing else. A genuine
#: three-part name still reaches the roster through every other path; it just cannot be minted
#: here, where there is no evidence for where the name begins.
_EPISODE_HOST_CUE = re.compile(
    r"\b([A-Z][a-z'\u2019\-]{2,}\s+[A-Z][a-z'\u2019.\-]{1,})"
    r"(?:\s+and\s+([A-Z][a-z'\u2019\-]{2,}\s+[A-Z][a-z'\u2019.\-]{1,}))?"
    r"\s+(?:(?:is|are)\s+joined\s+by|speaks?\s+with)\b"
)


def hosts_from_episode_description(
    episode_title: Optional[str], episode_description: Optional[str], feed_title: Optional[str]
) -> Set[str]:
    """Hosts named by the EPISODE's own description — the other side of the interview cue.

    THE HOST IS THE NAME BEFORE THE CUE. Guest detection reads what follows "is joined by" /
    "speaks with"; the name in front of it is the person doing the joining-with, i.e. the host.
    That half was being discarded, and on the shows where the feed's author tag is an
    ORGANISATION it is the only place a host is named at all.

    Measured over the 136 production episodes that end with no named speaker: **25 yield a host
    here** — 22 of a16z's 48, plus The Rest Is Politics and MLST. Extractions verified by hand:
    ``Elena Burger``, ``Ben Horowitz``, ``Theo Jaffee``, ``Tim Scarfe``,
    ``Alastair Campbell`` + ``Rory Stewart``.

    WHY A FEED-LEVEL HOST IS NOT ENOUGH ON THESE SHOWS. a16z rotates its host per episode — the
    feed cannot state one, and its author tag is "Andreessen Horowitz", which the org filter
    correctly discards. A per-episode host is the only correct answer for that shape.

    The show's own name and any publisher/org are refused, so "Planet Money is joined by..." can
    never mint a person.
    """
    # A SENTENCE BREAK BETWEEN TITLE AND DESCRIPTION, not a space. Joined with a bare space, a
    # title ending in a capitalised word runs straight into the description's first sentence and
    # the cue matches across the seam: "...the Future of Forecasting" + "Theo Jaffee speaks with"
    # gave a host called "Forecasting Theo Jaffee" on a16z.
    text = ". ".join(p.strip() for p in (episode_title or "", episode_description or "") if p)
    if not text.strip():
        return set()
    out: Set[str] = set()
    folded_show = _fold_title(feed_title)
    for match in _EPISODE_HOST_CUE.finditer(text):
        # THE SENTENCE CAN RUN THE OTHER WAY, and then the name in front of the cue is the GUEST.
        # EconTalk: "Listen as journalist Stephen Witt speaks with EconTalk's Russ Roberts about
        # how Jensen pivoted..." — Witt is the guest and Roberts the host, and the plain
        # before-the-cue rule seats the guest as host. The show naming ITSELF right after the cue
        # is what marks the inversion, and it is the only evidence in the sentence that does.
        if folded_show and folded_show in _fold_title(text[match.end() : match.end() + 60]):
            continue
        for cand in match.groups():
            name = (cand or "").strip()
            if not name or len(name.split()) < 2:
                continue
            # AND THE CAPTURE MUST LOOK LIKE A PERSON. The seam is only the loudest case; the same
            # run happens inside one description ("...the future of Forecasting Theo Jaffee speaks
            # with..."), and the token run the regex takes is as long as the capitals allow. This
            # is the guard every sibling extractor already applies, and omitting it here is how a
            # topic word ended up published as a host — the #876 failure exactly.
            if not looks_like_a_person_name(name):
                continue
            if looks_like_publisher(name):
                continue
            if feed_title and names_the_show(name, feed_title):
                continue
            out.add(name)
    return out


def recurrent_hosts_across_episodes(
    self_intros_by_episode: Iterable[Iterable[str]],
    *,
    feed_title: Optional[str] = None,
    min_episodes: int = 3,
    min_share: float = 0.25,
) -> Set[str]:
    """Names that SELF-INTRODUCE across many of a feed's episodes — i.e. the show's presenter.

    A GUEST APPEARS ONCE; A HOST APPEARS EVERY WEEK. That is the one property separating them that
    does not depend on phrasing, and no per-episode rule can see it.

    Measured over the production snapshot: at ``>=3 episodes AND >=25%`` of a feed's transcribed
    episodes, **28 of 55 feeds yield a name and every name is that show's presenter or a standing
    co-host** — zero guests, zero networks, zero sponsors, zero show names. Russ Roberts on 41 of
    41 EconTalk episodes, Noah Kravitz on 98 of 109 NVIDIA, Jessica Mendoza and Ryan Knutson on The
    Journal, Kevin Roose and Casey Newton on Hard Fork.

    Several arrive in the ASR's spelling rather than the published one (see the merge below), so
    "correct person" is the claim here, not "correct string".

    BOTH THRESHOLDS ARE LOAD-BEARING. The count alone admits a recurring guest on a long feed; the
    share alone admits anybody on a feed with three episodes.

    SELF-INTRODUCTIONS ONLY. The caller must pass names a voice used for ITSELF
    (:func:`distinct_self_introductions`), never names merely mentioned — a show that discusses
    the same person weekly would otherwise make them its host. Pass EVERY intro in the episode,
    not just the first: a show's second host introduces themselves second, and reading one name
    per episode is why a two-host desk show looked like one host plus a guest.

    WHAT THE CALLER MUST DO WITH THE RESULT: put it in ``known_hosts``, nothing more. It is a
    CANDIDATE. It may bind to a voice through that episode's own evidence — its self-introduction,
    or the LLM resolver under its existing guards — and it must never bind by talk share or by
    elimination, both of which measured below the safety bar (49-92%).

    THAT IS NOT YET TRUE OF THE CODE, AND THIS DOCSTRING USED TO CLAIM IT WAS. ``known_hosts`` also
    reaches ``_host_name_pool`` and then ``_name_host_voices``, which walks the pool with an integer
    index and assigns ``host_pool[hi]`` to the i-th seated host voice with NO per-voice evidence at
    all. Measured on the production snapshot: 1,070 voices carry ``source=known_hosts`` and 114 runs
    have two or more of them, so on a show whose opener is the guest's cold-open soundbite a name
    this function supplies can land on the guest. 128 of those 1,070 are rescued on this branch by
    the widened self-intro reader; 942 are not.

    So a name added here is only as safe as that pool rule. Until ``_name_host_voices`` requires
    per-voice evidence, treat every addition as reaching a positional assignment, and do not read
    this paragraph as a guarantee — read it as the reason the pool rule has to change.
    """
    episodes = [list(names or ()) for names in self_intros_by_episode]
    total = len([e for e in episodes if e])
    if total <= 0:
        return set()
    counts: Dict[str, int] = {}
    for names in episodes:
        for name in {str(n).strip() for n in names if str(n).strip()}:
            counts[name] = counts.get(name, 0) + 1
    # ONE HOST, ONE COUNT. A self-introduction is transcribed, so a co-host arrives spelled several
    # ways across a season. Hard Fork's Casey Newton is "Casey Newn" 20 times, "Casey Noon" 18 and
    # "Casey Newton" 7; The Journal's Ryan Knutson is "Ryan Knudson" 11 and "Ryan Knutson" 10.
    # Counted separately no spelling clears the share threshold and a real co-host is invisible;
    # merged, he clears it easily and the feed gets ONE candidate rather than three near-duplicate
    # people — the defect this branch exists to remove.
    #
    # THE WINNING SPELLING IS OFTEN WRONG, and this cannot fix that. Frequency does not separate a
    # mangle from the truth (Casey's most common rendering is a mangle), and nothing here has
    # access to written text to check against. It does not need to: a variant that matches a name
    # the feed or the config already states is dropped by the caller before it reaches the roster,
    # and ADR-130's `_recover_stated_names` snaps a published mangle back to the stated spelling.
    # What survives to publication mangled is a host on a feed that states no host anywhere — where
    # the alternative is not a correct name, it is `SPEAKER_01` on every episode.
    merged: List[Tuple[str, int]] = []
    for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        for i, (canonical, tally) in enumerate(merged):
            if same_person(canonical, name):
                merged[i] = (canonical, tally + n)
                break
        else:
            merged.append((name, n))
    out: Set[str] = set()
    for name, n in merged:
        if n < min_episodes or (n / total) < min_share:
            continue
        if is_known_network(name) or has_org_markers(name):
            continue
        # THE SHOW SAYS ITS OWN NAME EVERY EPISODE — that is recurrence, not a presenter. "The
        # Trivium China Podcast" opens with "Trivium" on 10 of 10 episodes and no other guard here
        # catches it: it is one token, carries no org marker, and is in no network list. Measured:
        # this is the single false positive across all 55 feeds.
        if feed_title and names_the_show(name, feed_title):
            continue
        # FIRST-LAST REQUIRED, unlike the per-episode self-intro path which allows a mononym host.
        # A recurring MONONYM is the weakest possible evidence and it misfired here: "Brandon" on
        # Latent Space cleared 25% (15 of 53) and is not one of that show's hosts. A real mononym
        # presenter can still be supplied through config `known_hosts`, which is what it is for.
        if len(name.split()) < 2 or not looks_like_a_person_name(name):
            continue
        out.add(name)
    return out


def _publisher_by_usage(author: str, feed_description: Optional[str]) -> bool:
    """An author tag that is a PUBLISHER by how the feed itself uses it, not by a wordlist.

    Measured on the production feeds, all unmarked by `_NONPERSON_AUTHOR_MARKERS`:

    * a name beginning with "The" — no person is "The X": `The Brazilian Report` (Explaining
      Brazil), `The China-Global South Project`;
    * a name the feed's description puts after "at" / "from" / "of" — a place or a body, not a
      presenter: "At Carnegie India, our diverse lineup of experts will host…".

    "by" is deliberately NOT read: "a podcast by Jane Doe" is how a host is credited.
    """
    if re.match(r"(?i)the\s", author.strip()):
        return True
    if not feed_description:
        return False
    return bool(
        re.search(
            rf"(?i)\b(?:at|from|of)\s+(?:the\s+)?{re.escape(author.strip())}\b", feed_description
        )
    )


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
    stated, statement_rejected = _feed_statement(feed_title, feed_description)
    if stated:
        logger.debug("Hosts stated by the feed: %s", sorted(stated))
        return stated
    if statement_rejected:
        logger.debug("The feed's host statement names no person; naming no host")
        return set()

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
                    # #2064: an <itunes:author> equal to the show's own name is the SHOW. It trips
                    # none of the checks below (no org marker, not a known network, not a mononym),
                    # so "Africa Tech Summit" and "Trivium China" were accepted as host people —
                    # and `_validate_hosts_with_first_episode` then confirmed them, because a
                    # show's name is always spoken in its own opening.
                    if names_the_show(candidate, feed_title):
                        logger.debug(
                            "RSS author '%s' names the show '%s', not a person on it",
                            candidate,
                            feed_title,
                        )
                        continue
                    if is_network_or_org_author(candidate) or _publisher_by_usage(
                        candidate, feed_description
                    ):
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

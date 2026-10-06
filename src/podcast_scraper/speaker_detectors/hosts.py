"""Host detection from feed metadata and transcript intro."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from ..kg.speaker_coherence import same_person
from ..languages import primary_language, TARGET_LANGUAGE
from . import naming_vocabulary
from .entities import extract_person_entities as _extract_person_entities_direct
from .entity_kind_votes import KindVotes

logger = logging.getLogger(__name__)

#: A STATED name — the feed's prose, which a publisher wrote down. Wider than `_NAME` on purpose
#: and used ONLY by the feed-statement patterns: a middle initial ("Stephen J. Dubner"), an accented
#: capital ("Dara Ó Briain") and a lowercase particle ("Cobus van Staden") are parts of a written
#: name. `_NAME` (the intro reader's run over ASR text) is untouched: there a lone capital is a
#: sentence opener the ASR capitalised.
_STATED_UC = r"[A-ZÀ-ÖØ-Þ]"
#: Its lowercase counterpart, and the reason it has to exist: three regexes below spelled a name
#: as `[A-Z][a-z]+`, which is ASCII-only, and the corpus now holds `Lucía`, `Élodie`, `Inês`,
#: `Solís` and `Lefèvre`. Measured on the fifteen-episode cast: `Lucía Herrera` yielded the first
#: name "Luc", `Inês Carvalho` yielded "In", and `Élodie Chevalier` yielded "Chevalier" — her
#: given name skipped entirely and the SURNAME returned in its place. The ranges mirror
#: `_STATED_UC`'s: `ß-ö` then `ø-ÿ`, stepping over ÷ exactly as that one steps over ×.
_STATED_LC_CHARS = r"a-zß-öø-ÿ"
_STATED_LC = rf"[{_STATED_LC_CHARS}]"

#: ENGLISH IS MAIN'S PATTERN TEXT, BYTE FOR BYTE (operator, 2026-10-05: English must not change).
#: The Unicode classes above serve the five non-English rows; every English pattern built in this
#: module reproduces main's text exactly, pinned by `test_english_patterns_are_mains.py`.
_ASCII_UC = r"[A-Z]"
_ASCII_LC = r"[a-z]"


def _uc(language: str) -> str:
    """The capital class for *language*: main's ASCII for English, Unicode for the rest."""
    return _ASCII_UC if language == TARGET_LANGUAGE else _STATED_UC


def _lc(language: str) -> str:
    """The lowercase class for *language*: main's ASCII for English, Unicode for the rest."""
    return _ASCII_LC if language == TARGET_LANGUAGE else _STATED_LC


def _alt(words: str) -> str:
    """*words* as a regex alternative: bare when it is one word, grouped when it has a ``|``.

    A single word needs no group, and leaving it bare is what keeps English byte-identical to
    main (`\\band\\b`, not `\\b(?:and)\\b`); a row with alternatives ("y|e") must be grouped.
    """
    return f"(?:{words})" if "|" in words else words


# RSS author tags are often the network/publisher, not the host — e.g. "Colossus",
# "Colossus | Investing & Business Podcasts", "NPR". Real hosts are personal "First Last"
# names. Reject org/network-looking tags so host detection falls through to transcript-intro
# NER / config ``known_hosts`` instead of mislabelling the host on every episode (#876).
# The marker WORDS live in `naming_vocabulary.NONPERSON_AUTHOR_WORDS`, where each row is the
# measured English base plus that market's own outlet words. The three measurements behind the
# base are recorded there: the news-outlet suffixes (standalone-surname words like Post and Press
# left out, caught by KNOWN_NETWORKS instead, so "Emily Post" is not flagged); the INSTITUTION
# tokens, added only after measuring across 4,307 roster entries and 13,642 Person nodes, where
# they match 5 distinct names and every one is an organisation ("Mercatus Center at George Mason
# University" has no pipe, no digit and no "Media"/"Network", so it was eligible to be a HOST and
# carries role="host" on 41 Person nodes today); and `plus`, a broadcast-brand suffix whose only
# carrier across 17,949 person names is "China Plus" itself.
#
# The pipe/slash/ampersand/at and the digit class stay HERE: punctuation is not vocabulary.
_NONPERSON_AUTHOR_MARKERS_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    lang: re.compile(r"[|/&@]|\d|" + r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)
    for lang, words in naming_vocabulary.NONPERSON_AUTHOR_WORDS.items()
}
_NONPERSON_AUTHOR_MARKERS = _NONPERSON_AUTHOR_MARKERS_BY_LANGUAGE[TARGET_LANGUAGE]


# Known podcast networks / publishers that appear as a spoken bumper ("This is Unhedged,
# I'm Pushkin. I'm Katie Martin…") or an RSS author tag, but are NOT a person. A bare
# mononym is not enough to reject a self-introduced name (real hosts go by one name —
# Oprah, Sting), so host-intro extraction needs this explicit list to skip the network
# bumper and fall through to the actual host. Lowercased; matched against the whole name
# and its first token. (#876 — "Pushkin" leaked as the Unhedged host.)
_KNOWN_NETWORKS_BY_LANGUAGE = naming_vocabulary.KNOWN_NETWORKS
KNOWN_NETWORKS = _KNOWN_NETWORKS_BY_LANGUAGE[TARGET_LANGUAGE]


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
_NAME_SUFFIXES_BY_LANGUAGE = naming_vocabulary.NAME_SUFFIXES
_NAME_SUFFIXES = _NAME_SUFFIXES_BY_LANGUAGE[TARGET_LANGUAGE]

# Comma, semicolon, ampersand, or a standalone "and" — the separators RSS author tags actually
# use. Word-bounded so "Alexander" is not cut at its "and".
#: The conjunction is the only language-dependent part; the comma, semicolon and ampersand
#: are punctuation. Word-bounded so "Alexander" is not cut at its "and".
_AUTHOR_SEPARATORS_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    lang: re.compile(rf"\s*(?:,|;|&|\b{_alt(conj)}\b)\s*", re.IGNORECASE)
    for lang, conj in naming_vocabulary.NAME_LIST_CONJUNCTION.items()
}
_AUTHOR_SEPARATORS = _AUTHOR_SEPARATORS_BY_LANGUAGE[TARGET_LANGUAGE]


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
_TITLE_WITH_SUFFIX_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    lang: re.compile(p, re.IGNORECASE) for lang, p in naming_vocabulary.TITLE_WITH_SUFFIX.items()
}
_TITLE_WITH_SUFFIX = _TITLE_WITH_SUFFIX_BY_LANGUAGE[TARGET_LANGUAGE]


#: A leading article is not part of a show's name for comparison purposes. Without dropping it,
#: "Trivium China" was not recognised as the prefix of "The Trivium China Podcast" and the show
#: seated itself as its own host.
_LEADING_ARTICLE_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    lang: re.compile(p) for lang, p in naming_vocabulary.LEADING_ARTICLE.items()
}
_LEADING_ARTICLE = _LEADING_ARTICLE_BY_LANGUAGE[TARGET_LANGUAGE]


def _fold_title(text: Optional[str]) -> str:
    """Lowercase, drop punctuation and any leading article, collapse whitespace."""
    folded = " ".join(re.sub(r"[^\w\s]", " ", str(text or "").lower()).split())
    return _LEADING_ARTICLE.sub("", folded)


def _title_possessive_of(candidate: str, feed_title: Optional[str]) -> bool:
    """ "Azeem Azhar's Exponential View" is Azeem Azhar's show, not a show called "Azeem Azhar".

    The prefix rule of :func:`names_the_show` read the host of every "<Person>'s <Show>" feed as the
    show itself and refused the author tag (gold development set, 2026-10-03).
    """
    cand = str(candidate or "").strip()
    title = str(feed_title or "").strip()
    if not cand or not title:
        return False
    return bool(re.match(rf"{re.escape(cand)}['’]s\b", title, re.IGNORECASE))


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
    if _title_possessive_of(candidate, feed_title):
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
# "my name's" is "my name is" (Past Present Future opens "Hello, my name's David Runciman" on every
# episode, and its host was unnamed on all of them), and "your host for today, <Name>" is "your
# host, <Name>" (#2224 follow-up). Both widen the FORM; the guards below are unchanged.
#: The NAME SHAPE stays here — how many tokens of evidence a self-introduction needs is not a
#: language question. Only the cue words come from `naming_vocabulary`.
def _self_intro_name(lang: str) -> str:
    return rf"{_uc(lang)}[\w'’\-]+(?:\s+{_uc(lang)}[\w'’\-]+){{0,3}}"


_HOST_SELF_INTRO_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    lang: re.compile(t % {"names": _self_intro_name(lang)})
    for lang, t in naming_vocabulary.HOST_SELF_INTRO.items()
}
_HOST_SELF_INTRO = _HOST_SELF_INTRO_BY_LANGUAGE[TARGET_LANGUAGE]

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
#: FIRST-LAST required (a mononym is the weakest evidence in the weakest form), and the show is
#: captured so the caller can check it IS this show. Both bounds are structural.
_HOST_BRANDED_INTRO_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    lang: re.compile(
        t
        % {
            "names": rf"{_uc(lang)}[\w'’\-]+(?:\s+{_uc(lang)}[\w'’\-]+){{1,2}}",
            "show": rf"{_uc(lang)}[\w'’\-]*(?:\s+{_uc(lang)}[\w'’\-]*){{0,3}}",
        }
    )
    for lang, t in naming_vocabulary.HOST_BRANDED_INTRO.items()
}
_HOST_BRANDED_INTRO = _HOST_BRANDED_INTRO_BY_LANGUAGE[TARGET_LANGUAGE]

# "with me, <Name>" / "and me, <Name>" — the British broadcast idiom for naming ONESELF.
# "Welcome to The Rest Is Politics: Leading with me, Alastair Campbell" is spoken BY Campbell.
# 11 of that feed's 12 unnamed episodes open this way; the pattern above matches none of them.
#
# THE COMMA IS REQUIRED AND "joining me" IS EXCLUDED, deliberately: "joining me, <Name>" introduces
# somebody ELSE, and admitting it would paint a guest's name onto the host's voice — the exact
# direction of error this module exists to prevent.
_HOST_WITH_ME_INTRO_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    lang: re.compile(t % {"names": rf"{_uc(lang)}[\w'’\-]+(?:\s+{_uc(lang)}[\w'’\-]+){{0,2}}"})
    for lang, t in naming_vocabulary.HOST_WITH_ME_INTRO.items()
}
_HOST_WITH_ME_INTRO = _HOST_WITH_ME_INTRO_BY_LANGUAGE[TARGET_LANGUAGE]


# "Let's say I'm Cass Sunstein" is a hypothetical, not a self-introduction: the host was posing a
# scenario and got named after a past guest (#2224). Checked against the few words BEFORE the match.
_HYPOTHETICAL_LEAD_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    lang: re.compile(p, re.IGNORECASE) for lang, p in naming_vocabulary.HYPOTHETICAL_LEAD.items()
}
_HYPOTHETICAL_LEAD = _HYPOTHETICAL_LEAD_BY_LANGUAGE[TARGET_LANGUAGE]

# "someone beside me said, hello. My name's Greg." is REPORTED speech — the narrator quoting
# somebody else (Planet Money), and "he says, I'm Anthony Aguirre" put a caller's name on the
# guest (MLST).
# But only somebody ELSE's: "So I said, I'm Jose Pereira" and "As you rightly said, my name is
# Joshua Chimakula Ngoma" are the speaker naming himself, and stay self-introductions.
_REPORTED_LEAD_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    lang: re.compile(p, re.IGNORECASE) for lang, p in naming_vocabulary.REPORTED_LEAD.items()
}
_REPORTED_LEAD = _REPORTED_LEAD_BY_LANGUAGE[TARGET_LANGUAGE]


def _is_hypothetical(head: str, match: "re.Match[str]") -> bool:
    """True when *match* follows a hypothetical ("let's say") or somebody ELSE's reported speech."""
    before = head[max(0, match.start() - 40) : match.start()]
    if _HYPOTHETICAL_LEAD.search(before[-30:]):
        return True
    reported = _REPORTED_LEAD.search(before)
    return reported is not None and reported.group("who").lower() not in {"i", "you"}


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
        if _is_hypothetical(head, match):
            continue
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
        if _is_hypothetical(head, match):
            continue
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
    text_language: Optional[str] = None,
) -> Set[str]:
    """Detect host names from transcript intro patterns (first 60-120 seconds).

    ``text_language`` is the S2.14 guard: the cue regexes below are English (``I'm X``,
    ``Welcome to … I'm X``) and the NER model is ``en_core_web_sm``, so on non-English prose
    this does not find nothing — measured on the V.6a Spanish fixture, recall held at 2/2 while
    precision fell from 67% to 18%. A missing name is visible; a wrong one becomes a person.
    Left ``None`` the guard does not engage, which is the pre-existing behaviour for every
    caller that has no language to offer.
    """
    if not transcript_text or not nlp:
        return set()
    from ..languages_guard import refuse_unsupported_language

    if refuse_unsupported_language("transcript-intro host detection", text_language):
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
_PRESENTS_BY_LANGUAGE = naming_vocabulary.PRESENTS
_PRESENTS = _PRESENTS_BY_LANGUAGE[TARGET_LANGUAGE]
_STATED_PARTICLES_BY_LANGUAGE = naming_vocabulary.STATED_PARTICLES
_STATED_PARTICLE_BY_LANGUAGE: Dict[str, str] = {
    lang: "(?:" + "|".join(parts) + ")" for lang, parts in _STATED_PARTICLES_BY_LANGUAGE.items()
}
_STATED_PARTICLE = _STATED_PARTICLE_BY_LANGUAGE[TARGET_LANGUAGE]


def _stated_name(language: str) -> str:
    """The written-name shape for *language*.

    Only the PARTICLE varies; the capital/lowercase classes are structural (a Unicode capital is
    a Unicode capital in every language) and the bounds are the catastrophic-backtracking fix
    documented above `_NAME`, which is not a language question either.
    """
    particle = _STATED_PARTICLE_BY_LANGUAGE.get(language, _STATED_PARTICLE)
    # The INITIAL ("Stephen J. Dubner") is `_uc(language)`: main spelled it `[A-Z]`, and English
    # keeps main's text; the name's own capitals were already Unicode on main.
    return (
        rf"(?-i:{_STATED_UC}[\w'’\-]+(?:\s+(?:{_uc(language)}\.|{particle}\s+{_STATED_UC}[\w'’\-]+"
        rf"|{_STATED_UC}[\w'’\-]*)){{1,5}})"
    )


_STATED_NAME = _stated_name(TARGET_LANGUAGE)
#: A list of stated names. Between two names the feed may put an Oxford comma ("Alexandra Karppi,
#: and Nina Panikova"), a role ("and co-host Shalma Wegsman", "and science creator Michael Stevens"
#: — up to three lowercase words), or a place after a name ("Eric Olander in Vietnam and Cobus van
#: Staden in South Africa"). Filler and place are matched case-SENSITIVELY, like the names, and the
#: place is extracted as a name of its own only to be refused by the person check.
#:
#: The CONJUNCTION and the place PREPOSITION are the language-dependent parts — "and"/"in" strip
#: nothing from "Marta Solís y Diego Ferrer en Madrid" — and come from `naming_vocabulary`. The
#: comma, the ampersand and the bounds are punctuation and structure, and stay here.


def _stated_names(language: str) -> str:
    """A list of written names for *language*, with its own conjunction and place preposition."""
    name = _stated_name(language)
    conj = naming_vocabulary.NAME_LIST_CONJUNCTION.get(language) or (
        naming_vocabulary.NAME_LIST_CONJUNCTION[TARGET_LANGUAGE]
    )
    prep = naming_vocabulary.PLACE_PREPOSITION_IN.get(language) or (
        naming_vocabulary.PLACE_PREPOSITION_IN[TARGET_LANGUAGE]
    )
    place = rf"(?-i:(?:\s+{_alt(prep)}\s+{_STATED_UC}[\w\-]+(?:\s+{_STATED_UC}[\w\-]+)?)?)"
    return (
        rf"{name}{place}"
        rf"(?:\s*(?:,\s*{_alt(conj)}|,|{_alt(conj)}|&)\s*(?-i:(?:[a-z][\w\-]*\s+){{0,3}}?)"
        rf"{name}{place})"
        rf"{{0,9}}"
    )


_STATED_PLACE = rf"(?-i:(?:\s+in\s+{_STATED_UC}[\w\-]+(?:\s+{_STATED_UC}[\w\-]+)?)?)"
_STATED_NAMES = _stated_names(TARGET_LANGUAGE)
#: Words a feed puts between the cue and the name: "hosted by Johannesburg-based entrepreneur and
#: American expat Justin Norman", "Join mathematician Professor Hannah Fry". Non-greedy and
#: bounded, so a name right after the cue is taken as is.
_STATED_LEAD = r"(?:[\w\-]+\s+){0,6}?"
#: Patterns safe to run over a TITLE as well as a description. One compiled list per language,
#: from `naming_vocabulary.HOST_PHRASE_TEMPLATES` — `%`-interpolated, not `.format`-ed, because
#: the templates carry regex quantifiers like `{0,6}` that `format` would read as fields.
_HOST_PHRASES_BY_LANGUAGE: Dict[str, List["re.Pattern[str]"]] = {
    lang: [
        re.compile(t % {"lead": _STATED_LEAD, "names": _stated_names(lang)}, re.IGNORECASE)
        for t in templates
    ]
    for lang, templates in naming_vocabulary.HOST_PHRASE_TEMPLATES.items()
}
_HOST_PHRASES = _HOST_PHRASES_BY_LANGUAGE[TARGET_LANGUAGE]

#: DESCRIPTION-ONLY. "Joe Weisenthal and Tracy Alloway explore..." / "Katie Martin, Robert Armstrong
#: and other markets nerds at the Financial Times explain..." — names, then a presenting verb, with
#: bounded filler so the verb belongs to THESE names.
#:
#: It must never run over a TITLE (#2064). A title is a NAME, not a sentence, and `_PRESENTS`
#: contains ordinary words that end show names — so "Machine Learning Street Talk" parsed as the
#: names "Machine Learning Street" followed by the verb "Talk", and the show became the host of
#: itself on 6 production episodes. The host does appear in a title, but in a different shape:
#: "... with Patrick O'Shaughnessy", which the `with` pattern above already reads.
_HOST_PHRASES_DESCRIPTION_ONLY_BY_LANGUAGE: Dict[str, List["re.Pattern[str]"]] = {
    lang: [
        re.compile(
            rf"(?P<names>{_stated_names(lang)})[\w\s,'’\-]{{0,60}}?\s+{verb}",
            re.IGNORECASE,
        )
    ]
    for lang, verb in _PRESENTS_BY_LANGUAGE.items()
}
_HOST_PHRASES_DESCRIPTION_ONLY = _HOST_PHRASES_DESCRIPTION_ONLY_BY_LANGUAGE[TARGET_LANGUAGE]
_NAME_RE = re.compile(_NAME)
_STATED_NAME_RE_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    lang: re.compile(_stated_name(lang)) for lang in _STATED_PARTICLE_BY_LANGUAGE
}
_STATED_NAME_RE = _STATED_NAME_RE_BY_LANGUAGE[TARGET_LANGUAGE]
#: DESCRIPTION-ONLY. The hosts by FIRST NAME, presenting: "Tom and Dominic bring the past to life"
#: (The Rest Is History), "guests join Rory and Alastair to discuss" (The Rest Is Politics:
#: Leading). Two or more given names, each resolved against a full name the same description
#: states ("with Tom Holland & Dominic Sandbrook"); a given name with no full form adds nobody.
_FIRST_NAME_PRESENTERS_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    lang: re.compile(
        rf"(?P<names>(?-i:{_uc(lang)}{_lc(lang)}+"
        rf"(?:\s*(?:,\s*{_alt(naming_vocabulary.NAME_LIST_CONJUNCTION[lang])}|,"
        rf"|{_alt(naming_vocabulary.NAME_LIST_CONJUNCTION[lang])}|&)\s*"
        rf"{_uc(lang)}{_lc(lang)}+){{1,3}}))"
        rf"[\w\s,'’\-]{{0,40}}?\s+{verb}",
        re.IGNORECASE,
    )
    for lang, verb in _PRESENTS_BY_LANGUAGE.items()
}
_FIRST_NAME_PRESENTERS = _FIRST_NAME_PRESENTERS_BY_LANGUAGE[TARGET_LANGUAGE]
#: PAIRED WITH `_STATED_NAME_RE`, and that is why the character class matters. `_STATED_NAME_RE`
#: has always been Unicode-aware, so `_first_name_presenters` built its `full_by_first` lookup with
#: the key `"lucía"` while this regex produced `"luc"` to look up — the two could never meet for an
#: accented name, and the failure was a silent MISS rather than a wrong answer.
_FIRST_NAME_RE_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    lang: re.compile(rf"(?-i:{_uc(lang)}{_lc(lang)}+)") for lang in _PRESENTS_BY_LANGUAGE
}
_FIRST_NAME_RE = _FIRST_NAME_RE_BY_LANGUAGE[TARGET_LANGUAGE]


_ARTICLE_BEFORE_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    lang: re.compile(p, re.IGNORECASE) for lang, p in naming_vocabulary.ARTICLE_BEFORE.items()
}
_ARTICLE_BEFORE = _ARTICLE_BEFORE_BY_LANGUAGE[TARGET_LANGUAGE]
_PLACE_PREPOSITION_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    lang: re.compile(p, re.IGNORECASE) for lang, p in naming_vocabulary.PLACE_PREPOSITION.items()
}
_PLACE_PREPOSITION = _PLACE_PREPOSITION_BY_LANGUAGE[TARGET_LANGUAGE]


def hosts_from_feed_statement(
    feed_title: Optional[str],
    feed_description: Optional[str],
    language: str = TARGET_LANGUAGE,
) -> Set[str]:
    """Hosts the feed EXPLICITLY names ("Hosted by X and Y"), rather than every person it mentions.

    This is the authoritative source: the show says who presents it. Only used for the names inside
    the host phrase, so a description that also lists past guests cannot smuggle them in.
    """
    return _feed_statement(feed_title, feed_description, language)[0]


def refused_feed_statement_names(
    feed_title: Optional[str],
    feed_description: Optional[str],
    language: str = TARGET_LANGUAGE,
) -> List[str]:
    """What a host phrase named that the person check refused ("Two Carnegie Mellon"), in order.

    Never a host; kept so the show metadata can show what the statement said and why it was not
    used.
    """
    return _feed_statement(feed_title, feed_description, language)[1]


def _feed_statement(
    feed_title: Optional[str],
    feed_description: Optional[str],
    language: str = TARGET_LANGUAGE,
) -> Tuple[Set[str], List[str]]:
    """``(hosts, refused)`` — every person a host phrase names, and what a phrase also named that
    is not a person.

    EVERY MATCH OF EVERY PHRASE IS READ, AND EVERY NAME IS JUDGED ON ITS OWN. Until 2026-10-03 the
    first match per pattern was taken, and one refused name threw the whole statement away and
    blocked the author tag behind it (#2075: "Americas Online" on Latin America in Focus). That
    protection was for the old positional host rule, which painted a pool name on whatever voice
    was seated; the seat logic now names a pool entry only from a self-introduction or a forced
    one-name-one-seat answer, so the pool may again hold what the feed says. Measured on the gold
    development set (2026-10-03): the junk statements of record ("Two Carnegie Mellon", "Norman
    Conquest", "Americas Online", "Anglo Canadian", "Carnegie India") are refused by the person
    check individually and the real names beside them ("Tom Holland", "Dominic Sandbrook", "Emily
    Hart", the author tags "Dan Saffer", "Nik Martelaro", "Carin Zissis", "Richard McColl") are
    kept.
    """
    out: Set[str] = set()
    refused: List[str] = []
    # EVERY pattern below is the FEED's language, not the analysis language. Nothing translates
    # feed metadata (D-44 moves the transcript, never the description), so a Spanish show's
    # "...es presentado por la anfitriona Lucía Herrera" has to be read in Spanish or the show
    # names nobody at all. Before these rows existed it named nobody: all five non-English feeds
    # returned an empty host set from a description that states its host in the first sentence.
    lang = _primary_subtag(language) or TARGET_LANGUAGE
    host_phrases = _HOST_PHRASES_BY_LANGUAGE.get(lang, [])
    description_only = _HOST_PHRASES_DESCRIPTION_ONLY_BY_LANGUAGE.get(lang, [])
    stated_name_re = _STATED_NAME_RE_BY_LANGUAGE.get(lang, _STATED_NAME_RE)
    article_before = _ARTICLE_BEFORE_BY_LANGUAGE.get(lang)
    place_preposition = _PLACE_PREPOSITION_BY_LANGUAGE.get(lang)
    not_a_mononym = _NOT_A_MONONYM_BY_LANGUAGE.get(lang, frozenset())
    for is_title, text in ((True, feed_title or ""), (False, feed_description or "")):
        if not text.strip():
            continue
        patterns = host_phrases if is_title else host_phrases + description_only
        for pat in patterns:
            for m in pat.finditer(text):
                # A person's name does not follow "the" or "of": "…Council of the Americas Online
                # team brings…" made `Americas Online` the host of Latin America in Focus.
                after_article = article_before is not None and bool(
                    article_before.search(text[: m.start("names")])
                )
                for raw in stated_name_re.findall(m.group("names")):
                    clean = _clean_stated_name(raw, language)
                    if len(clean.split()) < 2 or has_org_markers(clean):
                        continue
                    # A publisher/platform is never the host, even inside a host phrase (#1652
                    # applied this to RSS author tags; the statement path was the last place that
                    # skipped it). Real case from the #1657 acceptance run: The a16z Show's episode
                    # blurb runs two sentences together with no full stop — "...Listen to the a16z
                    # Show on Spotify Listen to the a16z Show on Apple Podcasts Follow our host:" —
                    # so "Spotify Listen" is a capitalised run across the sentence boundary, and the
                    # NOUN "host" 45 chars later satisfied the presenting-verb pattern.
                    if is_known_network(clean):
                        logger.debug(
                            "host statement named '%s', which is a platform/publisher, not a host",
                            clean,
                        )
                        continue
                    # In the DESCRIPTION, a capitalised run that echoes the show's own name is the
                    # show, not a person: "At Planet Money, we explore...". In the TITLE it is the
                    # opposite — that is where the host lives ("Invest Like the Best with Patrick
                    # O'Shaughnessy"), so the same guard there would throw the host away. A title
                    # that is the host's POSSESSIVE ("Azeem Azhar's Exponential View") names the
                    # host, not the show (`_title_possessive_of`).
                    if not is_title and _echoes_the_title(clean, feed_title):
                        continue
                    # Not a person: the tail of a longer proper noun (after "the"/"of"); a place or
                    # body ("At Carnegie India, our diverse lineup of experts will host…"); a
                    # nationality ("hosted by Anglo Canadian transplant to Colombia…" — LAST token
                    # only, `_NOT_A_MONONYM` holds demonyms that are also given names, "Christian");
                    # anything the shared person check refuses ("Two Carnegie Mellon", "Norman
                    # Conquest", "Timmerman Report", "South Africa").
                    if (
                        after_article
                        or (place_preposition is not None and place_preposition.match(raw.strip()))
                        or clean.split()[-1].lower().strip(".,'’") in not_a_mononym
                        or not is_publishable_speaker_name(clean, language=language)
                    ):
                        if clean not in refused:
                            refused.append(clean)
                        continue
                    out.add(clean)
        if not is_title:
            out |= _first_name_presenters(text, language)
    return out, refused


def _first_name_presenters(description: str, language: str = TARGET_LANGUAGE) -> Set[str]:
    """Full names for the given names a description shows presenting ("Tom and Dominic bring").

    ``language`` is the FEED's language, not the analysis language, and the difference is the
    whole point: nothing translates feed metadata, so this reads Spanish in a Spanish show's
    description even when its transcript has been translated to English.
    """
    full_by_first: Dict[str, str] = {}
    lang = _primary_subtag(language) or TARGET_LANGUAGE
    stated_name_re = _STATED_NAME_RE_BY_LANGUAGE.get(lang, _STATED_NAME_RE)
    for raw in stated_name_re.findall(description):
        clean = _clean_stated_name(raw, language)
        toks = clean.split()
        if len(toks) >= 2 and is_publishable_speaker_name(clean, language=language):
            full_by_first.setdefault(toks[0].lower(), clean)
    out: Set[str] = set()
    presenters = _FIRST_NAME_PRESENTERS_BY_LANGUAGE.get(lang, _FIRST_NAME_PRESENTERS)
    for m in presenters.finditer(description):
        first_name_re = _FIRST_NAME_RE_BY_LANGUAGE.get(lang, _FIRST_NAME_RE)
        firsts = [f for f in first_name_re.findall(m.group("names")) if f.lower() != "and"]
        if any(f.lower() in _NOT_A_NAME_TOKEN or f.lower() in _NOT_A_MONONYM for f in firsts):
            continue
        for f in firsts:
            full = full_by_first.get(f.lower())
            if full:
                out.add(full)
    return out


def _echoes_the_title(candidate: str, feed_title: Optional[str]) -> bool:
    """A stated name that is (part of) the show's own title, unless the title is the person's
    possessive: "Azeem Azhar" in "Azeem Azhar's Exponential View" is the host."""
    if _title_possessive_of(candidate, feed_title):
        return False
    return candidate.lower() in (feed_title or "").lower()


# A capitalised run is not automatically a name: it can start with a preposition ("At Planet
# Money"), or be prefixed by the publisher's possessive ("Bloomberg's Joe Weisenthal").
#
# KEYED BY LANGUAGE, because both halves are grammar. Under D-44 the canonical body is always the
# analysis language, so `_LEADING_JUNK` / `_POSSESSIVE_PREFIX` below resolve to the English rows
# and behave exactly as they shipped; the map is what makes another analysis language a data edit.
#
# NAME PARTICLES ARE DELIBERATELY ABSENT from every non-English row, and that is the one real
# decision here. English can strip a leading "The"/"From" safely because English names do not
# begin with them. Spanish, Italian, French, German and Portuguese names DO begin with exactly the
# words a naive translation would add: "de la Fuente", "Da Vinci", "De Gaulle", "Le Pen", "von
# Neumann", "da Silva", "dos Santos". Stripping those corrupts the name instead of cleaning it, so
# `de/di/da/do/dos/das/du/le/la/les/von/zu` appear in NO row — the junk list stays short rather
# than becoming symmetric with English.
_LEADING_JUNK_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    "en": re.compile(r"^(?:At|In|On|By|With|From|The)\s+", re.IGNORECASE),
    "es": re.compile(r"^(?:En|Al|Con|Desde|Para|Sobre|El|La|Los|Las)\s+", re.IGNORECASE),
    "it": re.compile(r"^(?:A|In|Su|Con|Per|Il|Lo|La|Gli|Le)\s+", re.IGNORECASE),
    "fr": re.compile(r"^(?:(?:À|A)|En|Sur|Avec|Dans|Chez|Pour)\s+", re.IGNORECASE),
    "de": re.compile(r"^(?:Bei|In|An|Auf|Mit|Aus|F(?:ü|u)r|Der|Die|Das)\s+", re.IGNORECASE),
    "pt": re.compile(r"^(?:Em|No|Na|Com|Desde|Para|Sobre|Os|As)\s+", re.IGNORECASE),
}

# "Bloomberg's Joe Weisenthal", "Red Hat's Chris Wright" — the employer, then the person. Non-greedy
# so it strips through the FIRST possessive only, leaving "Patrick O'Shaughnessy" (no "'s ") alone.
#
# `None` FOR FIVE LANGUAGES IS THE ANSWER, NOT A HOLE I DID NOT FILL. The construction this strips
# is PREFIX possession, and Spanish, Italian, French and Portuguese do not have it: they postpose
# the employer ("Joe Weisenthal de Bloomberg"), so there is nothing in front of the name to remove
# and any pattern here would only be able to damage one.
#
# GERMAN IS A GENUINE UNCOVERED CASE, recorded as `None` rather than guessed. German does front the
# genitive — "Bloombergs Joe Weisenthal" — but with a bare `s` and no apostrophe, so the English
# shape has no anchor to match on. `^\w+s\s+` would strip the first word of every name whose first
# token happens to end in s ("Hans Zimmer" -> "Zimmer"), which is worse than not cleaning. A safe
# pattern needs a capitalisation or NER anchor this stage does not have.
_POSSESSIVE_PREFIX_BY_LANGUAGE: Dict[str, Optional["re.Pattern[str]"]] = {
    # `s?`, not `s`: a PLURAL possessive has no trailing s — "The Rest Is Politics'
    # Alastair Campbell" (main, 2026-10-03). "O'Shaughnessy" stays safe because the
    # apostrophe there is not followed by whitespace.
    "en": re.compile(r"^.*?['’]s?\s+"),
    "es": None,
    "it": None,
    "fr": None,
    "de": None,
    "pt": None,
}

_LEADING_JUNK = _LEADING_JUNK_BY_LANGUAGE[TARGET_LANGUAGE]
_TRAILING_JOB_TOKENS_BY_LANGUAGE = naming_vocabulary.TRAILING_JOB_TOKENS
_TRAILING_JOB_TOKENS = _TRAILING_JOB_TOKENS_BY_LANGUAGE[TARGET_LANGUAGE]

# NO `_POSSESSIVE_PREFIX` ALIAS HERE, DELIBERATELY. One used to sit on this line and it was
# DEAD: a second `_POSSESSIVE_PREFIX` is defined much further down for `strip_role_prefix`,
# and the later module-level assignment wins — so the alias resolved to the ROLE-prefix
# pattern, not the stated-name one. `main` met the same bug from the other side and fixed it
# by naming its pattern `_STATED_POSSESSIVE_PREFIX`; reading the by-language map directly
# leaves no name to collide with. `.get(language)` below therefore has NO fallback: an
# unknown language gets no possessive stripping, which fails closed instead of applying a
# role-prefix pattern to a stated name.


def _clean_stated_name(name: str, language: str = TARGET_LANGUAGE) -> str:
    """The person inside a stated capitalised run.

    "Bloomberg's Joe Weisenthal" -> "Joe Weisenthal"; "At Planet Money" -> "Planet Money" (then
    refused as the show); "Professor Hannah Fry" -> "Hannah Fry" (a title is how someone is
    addressed; the roster snaps a self-introduction onto the pool by first name, and "Professor"
    is not one); "Senior User Experience Specialist Therese Fessenden" -> "Therese Fessenden";
    "Celestin Ntawirema CEO" -> "Celestin Ntawirema".

    ``language`` selects EVERY row this function reads: the possessive, the leading junk, the
    honorifics and both job-token passes. The earlier version keyed only the first two and said so
    — "the day a language needs them the fix is a row, not a rewrite". That day came: with English
    honorifics on a Spanish name, "Anfitrión Miguel" survived every pass and was publishable, while
    "Host Mike" was correctly refused. The rows exist now, so the passes read them.

    A language with no row strips NOTHING rather than stripping wrongly — `.get` with no English
    fallback, the same fail-closed rule as the rest of this module. "Cannot read this language"
    and "this language says nothing here" stay distinguishable.
    """
    clean = (name or "").strip()
    lang = _primary_subtag(language) or TARGET_LANGUAGE
    possessive = _POSSESSIVE_PREFIX_BY_LANGUAGE.get(lang)
    if possessive is not None:
        clean = possessive.sub("", clean)
    junk = _LEADING_JUNK_BY_LANGUAGE.get(lang)
    if junk is not None:
        clean = junk.sub("", clean)
    honorifics = _HONORIFIC_TITLES_BY_LANGUAGE.get(lang, frozenset())
    job_titles = _JOB_TITLE_TOKENS_BY_LANGUAGE.get(lang, frozenset())
    trailing_jobs = _TRAILING_JOB_TOKENS_BY_LANGUAGE.get(lang, frozenset())
    toks = clean.split()
    while len(toks) > 2 and toks[0].lower().strip(".,") in honorifics:
        toks = toks[1:]
    job = [i for i, tok in enumerate(toks[:-2]) if tok.lower().strip(".,") in job_titles]
    if job:
        toks = toks[job[-1] + 1 :]
    while len(toks) > 2 and toks[-1].lower().strip(".,") in trailing_jobs:
        toks = toks[:-1]
    return " ".join(toks).strip()


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
#: KEYED BY LANGUAGE for the reason the ad and cue vocabularies are: these are phrases, and a flat
#: list is a list in English wearing no label. Non-English rows are TRANSLATIONS of the English
#: categories — the English row is the measured one, and the three feeds quoted above are what
#: measured it. No non-English conversation has been run against these.
_HOST_SPEECH_ACTS_BY_LANGUAGE: Dict[str, Tuple["re.Pattern[str]", ...]] = {
    lang: tuple(re.compile(p, re.IGNORECASE) for p in patterns)
    for lang, patterns in {
        "en": (
            r"\bwelcome (?:back )?to (?:the |my |our )?\w+",
            r"\bi'?m your host\b",
            r"\b(?:my|our) guests? (?:today )?(?:is|are)\b",
            r"\b(?:joining|with) (?:me|us) (?:today|now|this week)\b",
            r"\bthanks? (?:so much )?for (?:coming on|joining me|joining us|being here)\b",
            # Thanking SEVERAL people for coming is the presenter's line ("Thank you
            # both very much for coming", The a16z Show). The singular form is left
            # alone: "Boris, thank you so much for being here" bleeds from the host's
            # close into the guest's cluster (Lenny's Podcast, corpus replay
            # 2026-10-03) and would hand the guest the only seat. NOT translated into
            # the five rows below, which is a judgement rather than an omission: the
            # both/all distinction is what makes the English pattern safe, and the
            # non-English rows already use number-agnostic phrasing ("gracias por
            # venir" covers one guest and several), so there is no singular form there
            # for it to separate from.
            r"\bthanks?(?: you)? (?:both|all)(?: (?:so|very) much)? for "
            r"(?:coming(?: on)?|joining (?:me|us)|being here)\b",
            r"\bthis week on (?:the )?\w+",
        ),
        "es": (
            r"\bbienvenid[oa]s? (?:de nuevo )?a (?:la |el |mi |nuestro )?\w+",
            r"\bsoy (?:tu|su|vuestro) (?:anfitri(?:ó|o)n|presentador[a]?)\b",
            r"\b(?:mi|nuestro|nuestra)s? invitad[oa]s? (?:de hoy )?(?:es|son)\b",
            r"\b(?:me|nos) acompa(?:ñ|n)a (?:hoy|ahora|esta semana)\b",
            r"\bgracias por (?:venir|acompa(?:ñ|n)arnos|estar aqu(?:í|i))\b",
            r"\besta semana en (?:la |el )?\w+",
        ),
        "it": (
            r"\bben(?:venut|tornat)[oiae]+ (?:di nuovo )?(?:a|su|nel|nella) "
            r"(?:il |la |mio |nostro )?\w+",
            r"\bsono il (?:vostro|tuo) (?:conduttore|host)\b",
            r"\b(?:il mio|il nostro|i nostri) ospit[ei] (?:di oggi )?(?:(?:è|e'|e)|sono)\b",
            r"\bcon (?:me|noi) (?:oggi|ora|questa settimana)\b",
            r"\bgrazie per (?:essere qui|essere venut[oa]|esserci)\b",
            r"\bquesta settimana (?:a|su|in) (?:il |la )?\w+",
        ),
        "fr": (
            r"\bbienvenue (?:(?:à|a) nouveau )?(?:dans|(?:à|a)|sur) (?:le |la |mon |notre )?\w+",
            r"\bje suis votre (?:h(?:ô|o)te"  # codespell:ignore te
            r"|animat(?:eur|rice)|pr(?:é|e)sentat(?:eur|rice))\b",
            r"\b(?:mon|notre|nos) invit(?:é|e)e?s? (?:du jour )?(?:est|sont)\b",
            r"\bavec (?:moi|nous) (?:aujourd'hui|maintenant|cette semaine)\b",
            r"\bmerci (?:d'(?:ê|e)tre "  # codespell:ignore tre
            r"(?:l(?:à|a)|ici|venu[e]?)|de venir)\b",
            r"\bcette semaine (?:dans|sur) (?:le |la )?\w+",
        ),
        "de": (
            r"\bwillkommen (?:zur(?:ü|u)ck )?(?:bei|zu|in|im) "
            r"(?:der |die |das |meinem |unserem )?\w+",
            r"\bich bin (?:euer|ihr|dein) (?:gastgeber(?:in)?|moderator(?:in)?)\b",
            r"\b(?:mein|unser)(?:e)? "  # codespell:ignore unser
            r"g(?:ä|a)st(?:e|in)? (?:heute )?(?:ist|sind)\b",
            r"\bbei (?:mir|uns) (?:heute|jetzt|diese woche)\b",
            r"\bdanke,? dass (?:du|sie|ihr) "  # codespell:ignore sie
            r"(?:hier|da|dabei) (?:bist|sind|seid)\b",
            r"\bdiese woche (?:bei|in|im) (?:der |die )?\w+",
        ),
        "pt": (
            r"\bbem-vind[oa]s? (?:de volta )?(?:ao|(?:à|a)|para o|para a) "
            r"(?:o |a |meu |nosso )?\w+",  # codespell:ignore meu
            r"\beu sou (?:o|a) (?:seu|sua) (?:anfitri(?:ã|a)o|apresentador[a]?)\b",
            r"\b(?:meu|minha|nosso|nossa)s? "  # codespell:ignore meu
            r"convidad[oa]s? (?:de hoje )?(?:(?:é|e)|s(?:ã|a)o)\b",
            r"\b(?:comigo|conosco) (?:hoje|agora|esta semana)\b",
            r"\bobrigad[oa] por (?:vir|estar aqui|nos acompanhar)\b",
            r"\besta semana (?:no|na|em) (?:o |a )?\w+",
        ),
    }.items()
}

_HOST_SPEECH_ACTS = _HOST_SPEECH_ACTS_BY_LANGUAGE[TARGET_LANGUAGE]
# NOTE (#1228) — a "floor-managing" host act (a co-host who only self-introduces on a no-host feed
# but directs the show, "Let's get into this week's news") was TRIED as a recall lever and REVERTED.
# On the prod-v2 corpus (90 eps, `relabel_corpus.py --llm none`) the tightened, nameability-gated
# pattern promoted ZERO voices, while the untightened form regressed real episodes (crowned an
# anonymous voice a host on Latent Space; painted host "Natalie Kitroeff" onto guest Robert Pape on
# The Daily — show-directing boilerplate like "we'll be right back" smears across diarization
# clusters). Inert on real data + precision-dangerous ⇒ not worth the code path (#876). The
# co-host-on-a-no-host-feed case stays the documented precision boundary (roster leaves the role
# unknown rather than risk a wrong name); revisit only with the #1189 human-GT fixtures.
#: The guest's half, keyed the same way. Gendered participles are spelled out in the rows that
#: have them (``obrigad[oa]``, ``encantad[oa]``, ``ravi(?:e)?``) — a row that only matched the
#: masculine form would see half the guests.
_GUEST_SPEECH_ACTS_BY_LANGUAGE: Dict[str, Tuple["re.Pattern[str]", ...]] = {
    lang: tuple(re.compile(p, re.IGNORECASE) for p in patterns)
    for lang, patterns in {
        "en": (
            # "thanks/thank you [so much | very much] for having me" — the intensifier is optional
            # AND may be "very much", not only "so much". "Thank you very much for having me" (The
            # Daily's guest Robert Pape) matched NEITHER old fixed pattern, so the dominant guest
            # was never flagged and community-1's clustering then crowned him a host (#1169).
            r"\b(?:thanks?|thank you)(?:\s+(?:so|very)\s+much)? for having me\b",
            r"\b(?:glad|happy|great|good) to be (?:here|on|back)\b",
        ),
        "es": (
            r"\bgracias(?:\s+(?:mil|muchas))? por (?:invitarme|recibirme|tenerme)\b",
            r"\b(?:encantad[oa]|content[oa]|feli(?:z|ces)) de estar (?:aqu(?:í|i)|de vuelta)\b",
            r"\bun placer estar (?:aqu(?:í|i)|contigo|con ustedes)\b",
        ),
        "it": (
            r"\bgrazie(?:\s+mille)? per (?:avermi invitato|l'invito|avermi qui)\b",
            r"\b(?:felice|content[oa]|un piacere) di essere (?:qui|qua|di nuovo qui)\b",
            r"\b(?:è|e'|e) un piacere essere (?:qui|qua)\b",
        ),
        "fr": (
            r"\bmerci(?:\s+beaucoup)? de m'(?:avoir invit(?:é|e)e?|accueillir|recevoir)\b",
            r"\b(?:ravi(?:e)?|content(?:e)?|heureu(?:x|se)) d'(?:ê|e)tre "  # codespell:ignore tre
            r"(?:l(?:à|a)|ici|de retour)\b",
            r"\bc'est un plaisir d'(?:ê|e)tre (?:l(?:à|a)|ici)\b",  # codespell:ignore tre
        ),
        "de": (
            r"\b(?:vielen |herzlichen )?dank f(?:ü|u)r die einladung\b",
            r"\b(?:freut mich|sch(?:ö|o)n|toll),? (?:hier|dabei|wieder hier) zu sein\b",
            r"\bich freue mich,? hier zu sein\b",
        ),
        "pt": (
            r"\bobrigad[oa](?:\s+(?:muito|demais))? por me (?:receber|convidar|ter aqui)\b",
            r"\b(?:feli(?:z|zes)|contente|um prazer) (?:de |em )?estar (?:aqui|de volta)\b",
            r"\b(?:é|e) um prazer estar aqui\b",
        ),
    }.items()
}

_GUEST_SPEECH_ACTS = _GUEST_SPEECH_ACTS_BY_LANGUAGE[TARGET_LANGUAGE]

_SHOW_INTRO_CUE_BY_LANGUAGE = naming_vocabulary.SHOW_INTRO_CUE
_SHOW_INTRO_CUE = _SHOW_INTRO_CUE_BY_LANGUAGE[TARGET_LANGUAGE]
#: A subtitle after the show's name ("No Priors: Artificial Intelligence | Technology").
_SHOW_TITLE_SEPARATOR = re.compile(r"\s+[:|\u2013\u2014-]\s+|:\s+")
#: A parenthesised tag after the name ("Machine Learning Street Talk (MLST)") is not said aloud.
_SHOW_TITLE_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
_SHOW_TAIL_WORDS_BY_LANGUAGE = naming_vocabulary.SHOW_TAIL_WORDS
_SHOW_TAIL_WORDS = _SHOW_TAIL_WORDS_BY_LANGUAGE[TARGET_LANGUAGE]
#: A one-word title must be at least this long: "Today" or "Daily" after "this is" is ordinary
#: speech; "Unbelievable", "Unhedged", "Decoder" are not.
_SHOW_MIN_MONONYM_LEN = 6
#: A two-token PREFIX of a longer title ("this is Roundtable" for "Round Table China") counts only
#: when it carries this many letters AND ends the clause -- "this is machine learning in the wild"
#: must not stand for "Machine Learning Street Talk".
_SHOW_PREFIX_MIN_LETTERS = 9


def show_name_pattern(feed_title: Optional[str]) -> Optional[str]:
    """A regex for the SHOW's name as a voice says it, or ``None`` when the title cannot be used.

    From the feed title: the subtitle dropped, the "with <Host>" suffix and a leading article
    dropped as :func:`names_the_show` does, a trailing "podcast"/"show" dropped (the caller makes it
    optional). Tokens are joined by ``\\W*`` because the ASR writes "Roundtable" for "Round Table"
    and "NNG" for "NN/G". The whole name, or its first two tokens when they end the clause.
    """
    if not feed_title:
        return None
    main = _SHOW_TITLE_PAREN.sub("", _SHOW_TITLE_SEPARATOR.split(str(feed_title), maxsplit=1)[0])
    folded = _fold_title(_TITLE_WITH_SUFFIX.sub("", main)) or _fold_title(main)
    toks = [t for t in folded.split() if t]
    while len(toks) > 1 and toks[-1] in _SHOW_TAIL_WORDS:
        toks.pop()
    if not toks:
        return None
    if len(toks) == 1:
        if len(toks[0]) < _SHOW_MIN_MONONYM_LEN or toks[0] in _NOT_A_NAME_TOKEN:
            return None
        return re.escape(toks[0])
    full = r"\W*".join(re.escape(t) for t in toks)
    if len(toks) > 2 and len(toks[0]) + len(toks[1]) >= _SHOW_PREFIX_MIN_LETTERS:
        prefix = re.escape(toks[0]) + r"\W*" + re.escape(toks[1])
        return rf"{full}|{prefix}(?=\W*(?:podcast|show)?\W*(?:[.,!?;:]|$))"
    return full


def performs_show_intro(text: Optional[str], feed_title: Optional[str]) -> bool:
    """True when the voice presents THIS show by name ("you're listening to Why This Universe")."""
    show = show_name_pattern(feed_title)
    if not text or not show:
        return False
    return bool(
        re.search(
            rf"\b{_SHOW_INTRO_CUE}\s+(?:the\s+|our\s+)?(?:{show})(?:\s*(?:podcast|show))?(?!\w)",
            text,
            re.IGNORECASE,
        )
    )


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
_CUE_FIRST_BODY_BY_LANGUAGE = naming_vocabulary.CUE_FIRST_BODY
CUE_FIRST_BODY = _CUE_FIRST_BODY_BY_LANGUAGE[TARGET_LANGUAGE]
# Past-tense hand-off ("i sat down with X", "we spoke with X"). A real introduction ONLY as a
# head-of-episode cold-open; mid-show it describes a PAST conversation and would misattribute the
# named person to whatever voice happens to speak next (a recap is not an intro). Kept separate so
# the roster can gate it to the first turns AND a host introducer (3rd advisor review).
_CUE_FIRST_PAST_BODY_BY_LANGUAGE = naming_vocabulary.CUE_FIRST_PAST_BODY
CUE_FIRST_PAST_BODY = _CUE_FIRST_PAST_BODY_BY_LANGUAGE[TARGET_LANGUAGE]
_GUEST_INTRODUCED_BY_HOST_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    lang: re.compile(
        rf"\b(?:{body})\s+(?:the\s+|our\s+)?(?P<names>{_NAMES})",
        re.IGNORECASE,
    )
    for lang, body in _CUE_FIRST_BODY_BY_LANGUAGE.items()
}
_GUEST_INTRODUCED_BY_HOST = _GUEST_INTRODUCED_BY_HOST_BY_LANGUAGE[TARGET_LANGUAGE]

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
_NAME_FIRST_TAIL_BY_LANGUAGE = naming_vocabulary.NAME_FIRST_TAIL
NAME_FIRST_TAIL = _NAME_FIRST_TAIL_BY_LANGUAGE[TARGET_LANGUAGE]
# Narrated-desk REPORT verbs ("Farnaz Fassihi explains…", "Sydney Baloue reports…"). These ALSO
# match a purely TOPICAL mention on a host's own sentence ("Sam Altman explains it best in his
# blog"), so on the case-blind match-form path they are resolved only against CORROBORATED refs
# (detected guests + known hosts) — never a bare metadata SUBJECT (3rd advisor review).
_NAME_FIRST_REPORT_TAIL_BY_LANGUAGE = naming_vocabulary.NAME_FIRST_REPORT_TAIL
NAME_FIRST_REPORT_TAIL = _NAME_FIRST_REPORT_TAIL_BY_LANGUAGE[TARGET_LANGUAGE]
_GUEST_INTRODUCED_NAME_FIRST_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    # Tolerate an ASR comma between the name and the verb ("Eric Schmitt, talks us through…").
    # The trailing \b is load-bearing: without it `reports?` matched inside "reported", and "in
    # June, China Daily reported on…" read as a hand-off — the newspaper was published as the guest.
    lang: re.compile(
        rf"(?P<names>{_NAMES})\s*,?\s+"
        rf"(?:{tail}|{_NAME_FIRST_REPORT_TAIL_BY_LANGUAGE[lang]})\b",
        re.IGNORECASE,
    )
    for lang, tail in _NAME_FIRST_TAIL_BY_LANGUAGE.items()
}
_GUEST_INTRODUCED_NAME_FIRST = _GUEST_INTRODUCED_NAME_FIRST_BY_LANGUAGE[TARGET_LANGUAGE]

# The host greets a just-introduced guest BY NAME: "Jody Rosen, welcome to the show",
# "Nic Harrigan, thanks so much for coming on". Name-then-greeting — the mirror of the cue-first
# forms, and the ordering a narrated interview show (The Daily) actually uses to bring a guest in.
_GREETED_TAIL_BY_LANGUAGE = naming_vocabulary.GREETED_TAIL
GREETED_TAIL = _GREETED_TAIL_BY_LANGUAGE[TARGET_LANGUAGE]
_GUEST_GREETED_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    lang: re.compile(rf"(?P<names>{_NAMES})\s*,\s*(?:{tail})", re.IGNORECASE)
    for lang, tail in _GREETED_TAIL_BY_LANGUAGE.items()
}
_GUEST_GREETED = _GUEST_GREETED_BY_LANGUAGE[TARGET_LANGUAGE]

# "I'm Coming Out", "I'm Not Sure" — the self-introduction regex matches any capitalised run, and
# the ASR capitalises plenty of things that are not people. Found in The Daily, where a voice was
# recorded as introducing itself as "Coming Out".
_NOT_A_NAME_TOKEN_BY_LANGUAGE = naming_vocabulary.NOT_A_NAME_TOKEN
_NOT_A_NAME_TOKEN = _NOT_A_NAME_TOKEN_BY_LANGUAGE[TARGET_LANGUAGE]


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
_NOT_A_MONONYM_BY_LANGUAGE = naming_vocabulary.NOT_A_MONONYM
_NOT_A_MONONYM = _NOT_A_MONONYM_BY_LANGUAGE[TARGET_LANGUAGE]


# Honorifics. The self-intro regex `\bI'?m\s+([A-Z][\w'’\-]+…)` stops at the period in "I'm Dr.
# Jane Smith", capturing the bare title "Dr" — which must never become a speaker name, and must not
# count as a distinct self-introduction (else "I'm Dr. X … I'm X" reads as a two-person montage).
_HONORIFIC_TITLES_BY_LANGUAGE = naming_vocabulary.HONORIFIC_TITLES
HONORIFIC_TITLES = _HONORIFIC_TITLES_BY_LANGUAGE[TARGET_LANGUAGE]


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


def drop_non_person_names(
    names: Iterable[str],
    feed_title: Optional[str] = None,
    kind_votes: Optional[KindVotes] = None,
) -> List[str]:
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

    *kind_votes* (``entity_kind_votes.KindVotes``) adds the corpus's own extraction labels: a name
    extraction decisively calls an organisation ("The Brazilian Report", "Americas Online") is not
    a candidate either (#2220). ``None`` — no corpus, or votes unreadable — changes nothing.
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
        # A real KindVotes only. Anything else (a test double, a stale field) answering truthy
        # would silently drop every candidate — measured: a MagicMock result emptied the guests.
        if isinstance(kind_votes, KindVotes) and kind_votes.calls_organisation(name):
            logger.info(
                "speaker candidate %r dropped: the corpus's KG extraction calls it an "
                "organisation (#2220)",
                name,
            )
            continue
        out.append(name)
    return out


_ROLE_OR_FILLER_TOKENS_BY_LANGUAGE = naming_vocabulary.ROLE_OR_FILLER_TOKENS
_ROLE_OR_FILLER_TOKENS = _ROLE_OR_FILLER_TOKENS_BY_LANGUAGE[TARGET_LANGUAGE]
_ORG_TAIL_TOKENS_BY_LANGUAGE = naming_vocabulary.ORG_TAIL_TOKENS
_ORG_TAIL_TOKENS = _ORG_TAIL_TOKENS_BY_LANGUAGE[TARGET_LANGUAGE]
_PLACE_TAIL_TOKENS_BY_LANGUAGE = naming_vocabulary.PLACE_TAIL_TOKENS
_PLACE_TAIL_TOKENS = _PLACE_TAIL_TOKENS_BY_LANGUAGE[TARGET_LANGUAGE]
_NUMBER_WORDS_BY_LANGUAGE = naming_vocabulary.NUMBER_WORDS
_NUMBER_WORDS = _NUMBER_WORDS_BY_LANGUAGE[TARGET_LANGUAGE]
_LEADING_ROLE_WORDS_BY_LANGUAGE = naming_vocabulary.LEADING_ROLE_WORDS
_LEADING_ROLE_WORDS = _LEADING_ROLE_WORDS_BY_LANGUAGE[TARGET_LANGUAGE]
_JOB_TITLE_TOKENS_BY_LANGUAGE = naming_vocabulary.JOB_TITLE_TOKENS
_JOB_TITLE_TOKENS = _JOB_TITLE_TOKENS_BY_LANGUAGE[TARGET_LANGUAGE]
#: Products and companies that introduce themselves in ads ("I'm Gemini", "Hey, it's Claude").
_BRAND_MONONYMS_BY_LANGUAGE = naming_vocabulary.BRAND_MONONYMS
_BRAND_MONONYMS = _BRAND_MONONYMS_BY_LANGUAGE[TARGET_LANGUAGE]
#: Stray brackets are an artefact the canonicaliser strips ("Aaron Levie)"), so they do not reject.
_DESCRIPTOR_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    lang: re.compile(pattern, re.IGNORECASE)
    for lang, pattern in naming_vocabulary.DESCRIPTOR_PATTERNS.items()
}
_DESCRIPTOR_SUFFIX = _DESCRIPTOR_BY_LANGUAGE[TARGET_LANGUAGE]
_NOT_IN_A_NAME = re.compile(r"[?{}<>!/@#|]")
_ROLE_PREFIX_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    #: The job or role in FRONT of a self-introduction ("I'm deputy editor Eilish Hart"), which
    #: `strip_role_prefix` removes. The English row keeps its optional "your"; the others take
    #: the possessive that language actually uses in the formula ("su", "vostro", "votre",
    #: "euer", "o seu").
    "en": re.compile(
        r"^(?:your\s+)?(?:(?:co-?)?host|(?:(?:deputy|senior|executive|managing|contributing)\s+)?"
        r"editor|producer|correspondent|reporter)\s*,?\s+",
        re.IGNORECASE,
    ),
    "es": re.compile(
        r"^(?:(?:su|tu|vuestro)\s+)?(?:co)?(?:anfitri(?:ó|o)n|anfitriona|presentador[a]?"
        r"|conductor[a]?|(?:(?:sub|jefe de)\s+)?redactor[a]?|editor[a]?|productor[a]?"
        r"|corresponsal|reporter[oa])\s*,?\s+",
        re.IGNORECASE,
    ),
    "it": re.compile(
        r"^(?:(?:il|la)\s+(?:vostro|tuo)\s+)?(?:co)?(?:conduttore|conduttrice|presentatore"
        r"|presentatrice|(?:vice\s+)?redattore|redattrice|produttore|produttrice"
        r"|corrispondente|giornalista)\s*,?\s+",
        re.IGNORECASE,
    ),
    "fr": re.compile(
        r"^(?:votre\s+)?(?:co)?(?:animateur|animatrice|pr(?:é|e)sentateur|pr(?:é|e)sentatrice"
        r"|(?:r(?:é|e)dacteur|r(?:é|e)dactrice)(?:\s+en\s+chef)?|producteur|productrice"
        r"|correspondant[e]?|reporter)\s*,?\s+",  # codespell:ignore correspondant
        re.IGNORECASE,
    ),
    "de": re.compile(
        r"^(?:(?:euer|ihr|dein)\s+)?(?:ko)?(?:gastgeber(?:in)?|moderator(?:in)?"
        r"|(?:chef)?redakteur(?:in)?|produzent(?:in)?|korrespondent(?:in)?|reporter(?:in)?)"
        r"\s*,?\s+",
        re.IGNORECASE,
    ),
    "pt": re.compile(
        r"^(?:(?:o|a)\s+(?:seu|sua)\s+)?(?:co)?(?:anfitri(?:ã|a)o|anfitri(?:ã|a)"
        r"|apresentador[a]?|(?:sub)?(?:editor[a]?|redator[a]?)|produtor[a]?|correspondente"
        r"|rep(?:ó|o)rter)\s*,?\s+",
        re.IGNORECASE,
    ),
}
_ROLE_PREFIX = _ROLE_PREFIX_BY_LANGUAGE[TARGET_LANGUAGE]
#: English is main's pattern verbatim (ASCII). The other languages get the accent-aware capital,
#: so "Écran's Marie Dupont" (accented INITIAL) is stripped on a French feed — and on an English
#: one, exactly as on main, it is not. ("Télérama's" is stripped by both: its initial is ASCII.)
_POSSESSIVE_PREFIX_ROLE_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    lang: re.compile(rf"^(?:{_uc(lang)}[\w&.\-]*\s+){{0,3}}{_uc(lang)}[\w&.\-]*['’]s\s+")
    for lang in _ROLE_PREFIX_BY_LANGUAGE
}
_POSSESSIVE_PREFIX = _POSSESSIVE_PREFIX_ROLE_BY_LANGUAGE[TARGET_LANGUAGE]


def strip_role_prefix(name: str, language: Optional[str] = TARGET_LANGUAGE) -> str:
    """``"Your Host Luisa Leni"`` -> ``"Luisa Leni"``; ``"Planet Money's Kenny Malone"`` ->
    ``"Kenny Malone"``. A self-introduction often carries the job or the show in front of the
    person ("I'm deputy editor Eilish Hart"), and the reader keeps it. Returns *name* unchanged when
    nothing would remain.

    ENGLISH IS EXACTLY MAIN: with no language, an English one, or one without rows, only the two
    English patterns run, as before. Another supported language runs its own two AFTER them — a
    strip only removes a prefix, so adding a language's prefixes cannot strip less than English.
    """
    out = (name or "").strip()
    patterns: List["re.Pattern[str]"] = [_POSSESSIVE_PREFIX, _ROLE_PREFIX]
    lang = _primary_subtag(language)
    if lang and lang != TARGET_LANGUAGE and lang in _ROLE_PREFIX_BY_LANGUAGE:
        patterns += [_POSSESSIVE_PREFIX_ROLE_BY_LANGUAGE[lang], _ROLE_PREFIX_BY_LANGUAGE[lang]]
    for pattern in patterns:
        stripped = pattern.sub("", out, count=1).strip()
        # A person survives the strip as at least two words; "Kay's Anatomy" -> "Anatomy" does not.
        if stripped and stripped != out and len(stripped.split()) >= 2:
            out = stripped
    return out


def _primary_subtag(language: Optional[str]) -> str:
    """``"es-ES"`` -> ``"es"``. NORMALISED HERE, not at each call site.

    Feeds carry a full BCP-47 tag and these maps are keyed by the primary subtag, so every caller
    would otherwise have to remember the split — and the one that forgot would silently get the
    English row for a Spanish feed, which is the exact failure this threading exists to close.

    Through ``normalize_language_tag`` first: callers pass the RAW feed tag, and splitting on
    ``-`` alone turned ``en_US`` / ``English`` / ``eng`` into keys no map has — an English feed the
    language gate accepts got no host-statement patterns at all, where main always read English.
    A tag the normaliser cannot place keeps the old split.
    """
    return primary_language(language)


def _reject_vocab(which: str, language: Optional[str]) -> FrozenSet[str]:
    """The English row UNION the feed language's row, for a REJECT filter.

    UNION, NOT REPLACEMENT, and the direction matters. This gate only ever says "no", so adding a
    language's words can reject more junk and can never accept something English would have
    refused — with ``language="en"`` it is byte-identical to the English row, which is why the
    English path cannot move. Replacement would be the unsafe choice: a non-English feed's
    description routinely carries an English show name beside its own role words
    ("Sesiones de Sendero", "Tech Summit"), and dropping the English row would let those through.
    """
    base = _NAMING_VOCABULARY_MAPS.get(which, {}).get(TARGET_LANGUAGE) or frozenset()
    row = _NAMING_VOCABULARY_MAPS.get(which, {}).get(_primary_subtag(language)) or frozenset()
    return frozenset(base) | frozenset(row)


def is_publishable_speaker_name(
    name: Optional[str],
    *,
    require_person_shape: bool = True,
    language: Optional[str] = TARGET_LANGUAGE,
) -> bool:
    """Final reject filter for a name about to be painted on a diarized voice (ADR-134 shared core).

    Every extraction path (self-intro, host-pool, greeting reader, strategy snap, LLM, metadata)
    converges on the roster; a name that carries a sentence-opener the ASR capitalised at a turn
    boundary ("But Sun", "So Nick", bare "But") is not a person, and a wrong label is worse than an
    unnamed voice. This is the last gate before publish, so no single path can bypass it.

    Deliberately WEAKER than :func:`is_plausible_mononym` for a one-token name: it rejects only a
    token that is a *known* non-name word, and does NOT require a capitalised first letter — else a
    real lowercase handle already vouched by a trusted source ("swyx") would be thrown away. The
    contract is "drop the garbage", not "re-validate every accepted name".

    ``require_person_shape=False`` skips only the last, ordinary-English-word check on a multi-word
    name (:func:`looks_like_a_person_name`), keeping every explicit junk rule. A corpus cleanup uses
    it: that check also refuses real names whose parts are common words ("Ethan He", "Michael I.
    Jordan"), and removing published names on it would delete real people.
    """
    nm = name or ""
    # A PUBLISHER IS NOT A SPEAKER, whatever path produced it. This gate is the last thing between
    # a name and a voice, and it was passing all 150 organisation names in the sample — including
    # the five that arrive as a transcript self-introduction ("Boston College", "Rindman
    # University"), which no candidate-list filter upstream can see.
    if looks_like_publisher(nm):
        return False
    # Punctuation a person's name never carries: "Premier Unbelievable?".
    if _NOT_IN_A_NAME.search(nm):
        return False
    # A hyphenated descriptor is a phrase about a person, not their name: "Pulitzer Prize-winning"
    # (Freakonomics, 2026-10-03 — the only such published name across the prod corpus).
    # English UNION the language's row, like every reject list here: a Spanish description still
    # writes an English show's "Emmy-winning", and the Spanish row adds "galardonada".
    lang_descriptor = _DESCRIPTOR_BY_LANGUAGE.get(_primary_subtag(language))
    if _DESCRIPTOR_SUFFIX.search(nm) or (
        lang_descriptor is not None and lang_descriptor.search(nm)
    ):
        return False
    toks = nm.split()
    lowered = [t.lower().strip(".,'’") for t in toks]
    # THE VOCABULARY IS PER-LANGUAGE (2026-10-03). Measured before it was: this gate rejected
    # "Host Mike", "Host" and "Tech Summit" and ACCEPTED "Anfitrión Miguel", "Anfitrión" and
    # "Cumbre Tecnología" — so a Spanish feed minted a person called "Anfitrión Miguel", and a
    # bare role word became a person. That is §5.2's phantom-person failure reached through the
    # DESCRIPTION, which is never translated.
    role_filler = _reject_vocab("role_or_filler_tokens", language)
    org_tail = _reject_vocab("org_tail_tokens", language)
    place_tail = _reject_vocab("place_tail_tokens", language)
    numbers = _reject_vocab("number_words", language)
    leading_roles = _reject_vocab("leading_role_words", language)
    job_titles = _reject_vocab("job_title_tokens", language)
    # Role and filler words disqualify a ONE-word name, or a name made of nothing else. Inside a
    # longer name they are real surnames and given names: Christopher Guest, Ok Taecyeon.
    if lowered and all(t in role_filler for t in lowered):
        return False
    if len(toks) >= 2:
        if lowered[-1] in org_tail or any(t.endswith(("'s", "’s")) for t in toks):
            return False
        if lowered[-1] in place_tail or lowered[0] in numbers:
            return False
        # "Host Mike", "Guest Host Tim": every word before the last is a role word.
        if all(t in leading_roles for t in lowered[:-1]):
            return False
        if len(toks) >= 3 and any(t in job_titles for t in lowered):
            return False
        if len(toks) >= 6:
            return False
        return looks_like_a_person_name(nm) if require_person_shape else True
    if len(toks) == 1:
        tl = lowered[0]
        # "GE", "AI", "IDF": an abbreviation standing in for a name, not a name (the same rule the
        # roster applies to publisher voice labels, `_looks_like_initials`).
        letters = toks[0].replace(".", "")
        if letters.isalpha() and letters.isupper() and len(letters) <= 3:
            return False
        return (
            tl not in _NOT_A_NAME_TOKEN
            and tl not in _NOT_A_MONONYM
            and tl not in HONORIFIC_TITLES
            and tl not in _BRAND_MONONYMS
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
#: The TWO-TOKEN capture is the measured structure (see above) and stays here; the hand-off verb
#: and the "and" joining two hosts come from `naming_vocabulary`.
def _episode_host_name(lang: str) -> str:
    lc_chars = "a-z" if lang == TARGET_LANGUAGE else _STATED_LC_CHARS
    return rf"{_uc(lang)}[{lc_chars}'\u2019\-]{{2,}}" rf"\s+{_uc(lang)}[{lc_chars}'\u2019.\-]{{1,}}"


#: Each `EPISODE_HOST_CUE` row is already ONE `(?:...)` group, so it is not wrapped again \u2014
#: which is also what keeps the English row byte-identical to main.
_EPISODE_HOST_CUE_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    lang: re.compile(
        rf"\b({_episode_host_name(lang)})"
        rf"(?:\s+{_alt(naming_vocabulary.NAME_LIST_CONJUNCTION[lang])}\s+"
        rf"({_episode_host_name(lang)}))?"
        rf"\s+{tail}\b"
    )
    for lang, tail in naming_vocabulary.EPISODE_HOST_CUE.items()
}
_EPISODE_HOST_CUE = _EPISODE_HOST_CUE_BY_LANGUAGE[TARGET_LANGUAGE]


#: A host role word immediately before a name in the episode's own prose.
_HOST_ROLE_BEFORE_NAME_BY_LANGUAGE: Dict[str, "re.Pattern[str]"] = {
    #: The role word immediately BEFORE a name ("...with co-hosts Tom and Dominic"). Anchored to
    #: the end of the preceding run, so only the word order changes between languages — Romance
    #: and German both put the role in front of the name here, same as English.
    "en": re.compile(r"\b(?:(?:co-?|guest )?hosts?|presenters?)\s*,?\s*$", re.IGNORECASE),
    "es": re.compile(
        r"\b(?:co)?(?:anfitri(?:ó|o)n(?:es|as)?|anfitriona|presentador(?:es|as|a)?"
        r"|conductor(?:es|as|a)?)\s*,?\s*$",
        re.IGNORECASE,
    ),
    "it": re.compile(
        r"\b(?:co)?(?:conduttor(?:e|i|ice|ici)|presentator(?:e|i|ice|ici))\s*,?\s*$",
        re.IGNORECASE,
    ),
    "fr": re.compile(
        r"\b(?:co)?(?:animateur(?:s)?|animatrice(?:s)?|pr(?:é|e)sentateur(?:s)?"
        r"|pr(?:é|e)sentatrice(?:s)?)\s*,?\s*$",
        re.IGNORECASE,
    ),
    "de": re.compile(
        r"\b(?:ko)?(?:gastgeber(?:in|innen)?|moderator(?:in|en|innen)?"
        r"|pr(?:ä|a)sentator(?:in|en|innen)?)\s*,?\s*$",
        re.IGNORECASE,
    ),
    "pt": re.compile(
        r"\b(?:co)?(?:anfitri(?:ã|a)o(?:s)?|anfitri(?:ã|a)(?:s)?|apresentador(?:es|as|a)?)"
        r"\s*,?\s*$",
        re.IGNORECASE,
    ),
}
_HOST_ROLE_BEFORE_NAME = _HOST_ROLE_BEFORE_NAME_BY_LANGUAGE[TARGET_LANGUAGE]


#: Every per-language naming map, by the name callers use. One place to add a map, so the registry
#: below cannot fall behind the data.
_NAMING_VOCABULARY_MAPS: Dict[str, Dict[str, Any]] = {
    "leading_role_words": _LEADING_ROLE_WORDS_BY_LANGUAGE,
    "show_tail_words": _SHOW_TAIL_WORDS_BY_LANGUAGE,
    "job_title_tokens": _JOB_TITLE_TOKENS_BY_LANGUAGE,
    "trailing_job_tokens": _TRAILING_JOB_TOKENS_BY_LANGUAGE,
    "number_words": _NUMBER_WORDS_BY_LANGUAGE,
    "role_or_filler_tokens": _ROLE_OR_FILLER_TOKENS_BY_LANGUAGE,
    "org_tail_tokens": _ORG_TAIL_TOKENS_BY_LANGUAGE,
    "place_tail_tokens": _PLACE_TAIL_TOKENS_BY_LANGUAGE,
    "presents": _PRESENTS_BY_LANGUAGE,
    "show_intro_cue": _SHOW_INTRO_CUE_BY_LANGUAGE,
    "host_role_before_name": _HOST_ROLE_BEFORE_NAME_BY_LANGUAGE,
    "role_prefix": _ROLE_PREFIX_BY_LANGUAGE,
    "possessive_prefix": _POSSESSIVE_PREFIX_BY_LANGUAGE,
    "leading_junk": _LEADING_JUNK_BY_LANGUAGE,
    "host_speech_acts": _HOST_SPEECH_ACTS_BY_LANGUAGE,
    "guest_speech_acts": _GUEST_SPEECH_ACTS_BY_LANGUAGE,
    # Added 2026-10-03, when the remaining English-only collections were keyed. These are the
    # RECALL side — what a feed or a host actually says — where the maps above are mostly the
    # PRECISION side. Both halves have to move together: a language with reject rows but no cue
    # rows reads nothing and refuses correctly, which looks like "this show names nobody".
    "honorific_titles": _HONORIFIC_TITLES_BY_LANGUAGE,
    "name_suffixes": _NAME_SUFFIXES_BY_LANGUAGE,
    "not_a_name_words": _NOT_A_NAME_TOKEN_BY_LANGUAGE,
    "not_a_mononym": _NOT_A_MONONYM_BY_LANGUAGE,
    "brand_mononyms": _BRAND_MONONYMS_BY_LANGUAGE,
    "known_networks": _KNOWN_NETWORKS_BY_LANGUAGE,
    "host_phrases": _HOST_PHRASES_BY_LANGUAGE,
    "host_phrases_description_only": _HOST_PHRASES_DESCRIPTION_ONLY_BY_LANGUAGE,
    "first_name_presenters": _FIRST_NAME_PRESENTERS_BY_LANGUAGE,
    "host_self_intro": _HOST_SELF_INTRO_BY_LANGUAGE,
    "host_branded_intro": _HOST_BRANDED_INTRO_BY_LANGUAGE,
    "host_with_me_intro": _HOST_WITH_ME_INTRO_BY_LANGUAGE,
    "episode_host_cue": _EPISODE_HOST_CUE_BY_LANGUAGE,
    "cue_first_body": _CUE_FIRST_BODY_BY_LANGUAGE,
    "cue_first_past_body": _CUE_FIRST_PAST_BODY_BY_LANGUAGE,
    "name_first_tail": _NAME_FIRST_TAIL_BY_LANGUAGE,
    "name_first_report_tail": _NAME_FIRST_REPORT_TAIL_BY_LANGUAGE,
    "greeted_tail": _GREETED_TAIL_BY_LANGUAGE,
    "guest_introduced_by_host": _GUEST_INTRODUCED_BY_HOST_BY_LANGUAGE,
    "guest_introduced_name_first": _GUEST_INTRODUCED_NAME_FIRST_BY_LANGUAGE,
    "guest_greeted": _GUEST_GREETED_BY_LANGUAGE,
    "hypothetical_lead": _HYPOTHETICAL_LEAD_BY_LANGUAGE,
    "reported_lead": _REPORTED_LEAD_BY_LANGUAGE,
    "nonperson_author_markers": _NONPERSON_AUTHOR_MARKERS_BY_LANGUAGE,
    "stated_particles": _STATED_PARTICLES_BY_LANGUAGE,
    "name_list_conjunction": naming_vocabulary.NAME_LIST_CONJUNCTION,
    "place_preposition_in": naming_vocabulary.PLACE_PREPOSITION_IN,
    "leading_article": _LEADING_ARTICLE_BY_LANGUAGE,
    "article_before": _ARTICLE_BEFORE_BY_LANGUAGE,
    "place_preposition": _PLACE_PREPOSITION_BY_LANGUAGE,
    "title_with_suffix": _TITLE_WITH_SUFFIX_BY_LANGUAGE,
    "author_separators": _AUTHOR_SEPARATORS_BY_LANGUAGE,
    # OWNED BY OTHER MODULES, registered here anyway. `roster.py`, `resolution.py` and
    # `gi/speakers.py` read these, and none of them can be imported from here without a cycle —
    # but the maps themselves are plain data in `naming_vocabulary`, so the INTERSECTION below can
    # still see them. That is the point: a language added to the host maps and not to the roster
    # ones would otherwise be advertised as complete while the diarizer reads nothing for it.
    "self_intro_words": naming_vocabulary.SELF_INTRO_WORDS,
    "this_is_intro": naming_vocabulary.THIS_IS_INTRO,
    "recap_markers": naming_vocabulary.RECAP_MARKERS,
    "descriptor_patterns": naming_vocabulary.DESCRIPTOR_PATTERNS,
    "intro_affiliation_tokens": naming_vocabulary.INTRO_AFFILIATION_TOKENS,
    "sign_off_cues": naming_vocabulary.SIGN_OFF_CUES,
    "greeting_at_open": naming_vocabulary.GREETING_AT_OPEN,
    "non_person_label_tokens": naming_vocabulary.NON_PERSON_LABEL_TOKENS,
    "generational_suffixes": naming_vocabulary.GENERATIONAL_SUFFIXES,
}

#: The languages whose naming vocabulary is COMPLETE, as an INTERSECTION.
#:
#: Fails CLOSED, same as `SPEAKER_CUE_LANGUAGES`: a language added to some maps and not others is
#: absent from this set rather than half-supported, because half-supported is the state that
#: produces a confident wrong answer. `possessive_prefix` is excluded from the intersection on
#: purpose — it is the one map whose correct value for four of the six languages is `None` (they
#: postpose the employer), so requiring a truthy row there would refuse languages that are in fact
#: complete.
NAMING_VOCABULARY_LANGUAGES: FrozenSet[str] = frozenset(
    set.intersection(
        *(
            {lang for lang, row in m.items() if row or name == "possessive_prefix"}
            for name, m in _NAMING_VOCABULARY_MAPS.items()
        )
    )
)


def naming_vocabulary_for(name: str, language: str) -> Any:
    """One language's row from one naming map, or ``None`` when it has none.

    ``None`` RATHER THAN THE ENGLISH ROW, for the reason the ad and cue accessors give: falling
    back to English makes "we cannot read this language" indistinguishable from "this language
    says nothing here", and the second is a measurement while the first is a gap. A caller that
    wants the analysis language should ask for it by name.
    """
    return (_NAMING_VOCABULARY_MAPS.get(name) or {}).get(_primary_subtag(language))


def hosts_from_episode_description(
    episode_title: Optional[str],
    episode_description: Optional[str],
    feed_title: Optional[str],
    *,
    feed_hosts: Iterable[str] = (),
    participants: Iterable[str] = (),
    language: str = TARGET_LANGUAGE,
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
        after = text[match.end() : match.end() + 60]
        if folded_show and folded_show in _fold_title(after):
            continue
        # ...and so does one of the FEED's hosts right after the cue: "Mark Zuckerberg speaks with
        # Sarah Guo and Elad Gil" names the guest first (No Priors-type, validation 2026-10-03).
        if any(_mentions_full_name(after, h) for h in feed_hosts):
            continue
        for gi, cand in enumerate(match.groups(), start=1):
            name = (cand or "").strip()
            if not name or len(name.split()) < 2:
                continue
            # "comic co-host Jordan Klepper sit down with Lara Anderson": a HOST role word right
            # before the name is the description saying who hosts, and it outranks the detector
            # having listed the person among the episode's guests (StarTalk, gold development set).
            role_before = bool(
                _HOST_ROLE_BEFORE_NAME.search(text[max(0, match.start(gi) - 24) : match.start(gi)])
            )
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
            # A person the episode states as a PARTICIPANT (its stated guests, its byline) is not
            # made the host by a cue: a guest's name in the pool seats the guest (validation).
            if not role_before and any(same_person(name, p) for p in participants):
                continue
            out.add(name)
    return out


def _mentions_full_name(text: str, name: str) -> bool:
    """Does *text* carry this person's given AND family name, in order (case-insensitive)?"""
    toks = [t.strip(".,'’") for t in (name or "").split() if t.strip(".,'’")]
    if len(toks) < 2 or not text:
        return False
    return bool(
        re.search(rf"\b{re.escape(toks[0])}\s+{re.escape(toks[-1])}\b", text, re.IGNORECASE)
    )


def _merge_people(*groups: Iterable[str]) -> List[str]:
    """One entry per human across *groups*, first spelling kept (the earlier group outranks).

    A feed statement and an author tag spell the same host differently ("Alastair Campbell" /
    "Alistair Campbell", "Robert Armstrong" / "Rob Armstrong"); kept apart they are two seats for
    one person, and the roster then splits a host over them (advisor review, 2026-10-02).
    """
    out: List[str] = []
    for group in groups:
        for raw in group or ():
            name = str(raw or "").strip()
            if not name or any(same_person(name, kept) for kept in out):
                continue
            out.append(name)
    return out


def compose_episode_hosts(
    feed_hosts: Iterable[str],
    episode_authors: Iterable[str] = (),
    *,
    episode_title: Optional[str] = None,
    episode_description: Optional[str] = None,
    feed_title: Optional[str] = None,
    more: Iterable[str] = (),
    episode_people: Iterable[str] = (),
    language: Optional[str] = TARGET_LANGUAGE,
) -> List[str]:
    """The host pool of ONE episode, as a list the roster may paint names from.

    - ``feed_hosts``: what the feed states (:func:`detect_hosts_from_feed`); first.
    - ``more``: config ``known_hosts`` and the recurrence scan; person-checked here because each of
      those paths has its own idea of what an organisation looks like.
    - the hosts THIS episode's own description names ("Erik Torenberg sits down with ...",
      "X is joined by Y") — computed since #2075 and thrown away before it reached the roster.
    - ``episode_authors``: the episode's ``<itunes:author>`` byline — ONLY when the description
      names no host. On The a16z Show the byline lists everybody in the room, so the guests were in
      the pool and the LLM's name for a guest seated them as a host (gold development set: Amjad
      Masad, Gagan Biyani, Diogo Almeida). When the description says who hosts, the byline adds
      nobody.

    Every entry passes the shared person check and is not the show's own name; spellings of one
    person are merged (:func:`_merge_people`).
    """

    def people(names: Iterable[str]) -> List[str]:
        out: List[str] = []
        for raw in names or ():
            name = str(raw or "").strip()
            if not name or not is_publishable_speaker_name(name, language=language):
                continue
            if feed_title and names_the_show(name, feed_title):
                continue
            out.append(name)
        return out

    feed_people = people(feed_hosts)
    described = sorted(
        hosts_from_episode_description(
            episode_title,
            episode_description,
            feed_title,
            feed_hosts=feed_people,
            participants=list(episode_people or ()),
            language=language or TARGET_LANGUAGE,
        )
    )
    pool = _merge_people(feed_people, people(more), described)
    if not described:
        pool = _merge_people(pool, people(episode_authors))
    return pool


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
    language: str = TARGET_LANGUAGE,
) -> Set[str]:
    """Detect host names from feed-level metadata.

    The feed's own HOST STATEMENT ("Hosted by ...", "with A and B" in the title) and its personal
    author tags are both read and UNIONED — a statement no longer hides the tag behind it, and a
    junk name in the statement no longer empties it. "Two Carnegie Mellon faculty explore" used to
    beat the author tag "Dan Saffer and Nik Martelaro" (AI and Design); "...Council of the Americas
    Online team brings..." used to block "Carin Zissis" (Latin America in Focus). One person stated
    twice under two spellings is one entry (:func:`_merge_people`).

    NER over the title is the last resort, only when neither names anybody: it cannot tell a host
    from anyone else the description happens to mention — on Latent Space it returns a list of past
    guests, and on Planet Money it returns the word "Wanna".
    """
    stated, _junk = _feed_statement(feed_title, feed_description, language)
    if stated:
        logger.debug("Hosts stated by the feed: %s", sorted(stated))

    tag_hosts: List[str] = []
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
                        continue
                    # The shared person check, so an author tag that is a brand with no org marker
                    # ("Timmerman Report", "Premier Unbelievable?", "Brilliant Experience") does
                    # not become a seat the roster may fill.
                    if not is_publishable_speaker_name(candidate, language=language):
                        logger.debug("RSS author '%s' is not a person's name", candidate)
                        continue
                    tag_hosts.append(candidate)
        if tag_hosts:
            logger.debug(
                "Detected hosts from RSS author tags (author/itunes:author/itunes:owner): %s",
                tag_hosts,
            )
        elif not stated:
            _log(
                "info",
                "All RSS author(s) treated as organisation(s); host detection will use "
                "NER from feed title/description, episode-level authors, or config known_hosts",
            )

    hosts: Set[str] = set(_merge_people(sorted(stated), tag_hosts))
    if hosts:
        return hosts

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

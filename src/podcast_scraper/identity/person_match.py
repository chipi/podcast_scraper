"""Person identity: the one home for cross-episode identity decisions (RFC-131).

This module grows in steps. First it holds the name-similarity helpers that were copied across
the code base. The roster, speaker resolution, the corpus graph, name canonicalisation and the
name report each had their own edit distance or Soundex. Later steps of RFC-131 add the
same-person rule modes, the guards (ADR-167), the spelling rule and the display-name function.

Importable from any layer without a cycle: it imports only the standard library at load time
(``kg.filters`` is bound on first use by the cluster mode).
"""

from __future__ import annotations

import difflib
import functools
import re
from typing import Callable, List, Optional

__all__ = [
    "MODES",
    "cluster_person_variants",
    "cluster_variants",
    "edit_distance",
    "host_variants",
    "name_words",
    "one_edit_apart",
    "same_person",
    "soundex",
]

_SOUNDEX_CODES = {
    **dict.fromkeys("BFPV", "1"),
    **dict.fromkeys("CGJKQSXZ", "2"),
    **dict.fromkeys("DT", "3"),
    **dict.fromkeys("L", "4"),
    **dict.fromkeys("MN", "5"),
    **dict.fromkeys("R", "6"),
}


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between ``a`` and ``b``, case-insensitive."""
    a, b = a.lower(), b.lower()
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def one_edit_apart(a: str, b: str) -> bool:
    """True for strings at Levenshtein distance exactly 1 (case-insensitive), in linear time."""
    a, b = a.lower(), b.lower()
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i + (len(a) == len(b)) :] == b[i + 1 :]


def soundex(word: str) -> str:
    """Classic American Soundex of one token.

    Catches ASR substitutions that swap vowels but keep the consonant skeleton ("Roose" ->
    "Russo"). It is blind to "Newton" -> "Noon", which edit distance catches instead; the two are
    complementary, and neither alone is enough. Letters are kept by ``str.isalpha``, so a
    non-ASCII letter can be the first letter of the code; an empty string for a token without
    letters.
    """
    w = "".join(c for c in word.upper() if c.isalpha())
    if not w:
        return ""
    out, prev = w[0], _SOUNDEX_CODES.get(w[0], "")
    for ch in w[1:]:
        code = _SOUNDEX_CODES.get(ch, "")
        if code and code != prev:
            out += code
        if ch not in "HW":
            prev = code
    return (out + "000")[:4]


# ---- same-person rule modes ---------------------------------------------------------------------
#
# One function per consumer's rule, moved here unchanged (RFC-131, S1b step 3). They differ on
# purpose today; folding them into one rule is a behaviour change, measured on its own.

# Thresholds tuned in #853 (data/eval/runs/baseline_entity_canon_v1).
# Silver eval: 190 candidate pairs from a real prod corpus
# (`.test_outputs/manual/my-manual-run-10`), Sonnet 4.6 silver labels
# (49 SAME / 134 DIFFERENT / 7 BORDERLINE).
#
# Baseline (token=0.78, overall=0.85): P=1.00, R=0.31, F1=0.47
# Tuned    (token=0.65, overall=0.70): P=1.00, R=0.49, F1=0.66
# +60% recall at preserved 100% precision. Caught: Bessent/Bessett,
# Tracy Alloway/Allaway, Joe Weisenthal/Wassenthal, Tim Geithner/Geidner,
# Henry Blodget/Blodgett, etc. The recall ceiling is structural (token-
# count mismatches like "Dr. Elena Fischer" vs "Elena Fischer") — predicate
# redesign tracked in #904, not threshold tuning.
_TOKEN_RATIO = 0.65  # per-aligned-token spelling-variant floor
_OVERALL_RATIO = 0.70  # whole-string floor
_VERSION_TOKEN_RE = re.compile(r"\d")  # a differing token containing a digit blocks merge

#: A well-formed roman numeral. Regnal numbers are how monarchs are told apart, and
#: ``_VERSION_TOKEN_RE`` only sees ARABIC digits — so ``Charles I`` and ``Charles II`` reached the
#: ratio test and merged (found by the guardrail matrix; *The Rest Is History* is in the corpus).
#:
#: Applied ONLY when BOTH differing tokens are roman-shaped AND sit in the LAST position. A bare
#: numeral test would be unsafe: ``li`` is a valid roman numeral and a common Chinese surname, and
#: ``md`` is one too (1500).
#:
#: An earlier version of this comment claimed the last-token scope was what kept ``Li`` out of
#: reach, "because Li occupies first position". That is false, and the corpus says so: of 13,642
#: person nodes, 127 end in a roman-shaped token, including ``Fei-Fei Li`` (x17), ``Jennifer Li``,
#: ``Ang Li``, ``Jia Li`` and ``Peter Attia, MD`` — Western order, surname LAST.
#:
#: What actually makes it safe is the conjunction: the guard fires only when the two names differ
#: in that position AND both spellings are roman-shaped. For real names that means ``Jia Li`` vs
#: ``Jia Mi`` — two different people, correctly refused. And the guard can only ever REFUSE, so
#: its failure mode is a false split (clutter), never a false merge (reassigned statements). One
#: human whose surname drifted between two roman-shaped spellings would be split; none observed.
_ROMAN_NUMERAL_RE = re.compile(
    r"^(?=[ivxlcdm])m{0,4}(c[md]|d?c{0,3})(x[cl]|l?x{0,3})(i[xv]|v?i{0,3})$"
)

# Nicknames where the canonical pair differs by too much for the ratio test
# (Michael→Mike, Robert→Rob/Bob) but the people are the same. Lowercase →
# lowercase, bidirectional (both directions added at module load). Tuned in
# #904 from the #853 silver-eval miss catalogue. NOT meant to be exhaustive —
# the catch is "the long-tail of recall improvements at preserved precision":
# add a pair only after you've seen the miss class in real prod data.
_NICKNAME_MAP: dict[str, set[str]] = {
    "michael": {"mike"},
    "nicholas": {"nick"},
    "elizabeth": {"liz", "beth", "betsy"},
    "jerome": {"jerry", "jay", "j."},
    "emmanuel": {"manny"},
    "richard": {"rich", "rick", "dick"},
    "robert": {"rob", "bob", "bobby"},
    "william": {"will", "bill", "billy"},
    "thomas": {"tom", "tommy"},
    "joseph": {"joe", "joey"},
    "anthony": {"tony"},
    "patrick": {"pat"},
    "stephen": {"steve"},
    "steven": {"steve"},
    "matthew": {"matt"},
    "jonathan": {"jon"},
    "christopher": {"chris"},
    "samuel": {"sam"},
    "alexander": {"alex", "al"},
    "daniel": {"dan", "danny"},
    "edward": {"ed", "eddie", "ted"},
    "andrew": {"andy", "drew"},
    "benjamin": {"ben"},
    "katherine": {"kate", "katie", "kathy"},
    "catherine": {"cathy", "kate"},
    "rebecca": {"becky"},
    "susan": {"sue"},
    "deborah": {"debbie", "deb"},
    "barbara": {"barb"},
    "patricia": {"pat", "patty", "trish"},
    "jennifer": {"jen", "jenny"},
    "margaret": {"maggie", "meg", "peggy"},
}
# Symmetric closure so lookup works either direction.
_NICKNAME_PAIRS: set[tuple[str, str]] = set()
for _full, _nicks in _NICKNAME_MAP.items():
    for _n in _nicks:
        # Strip trailing dot so "j." stored as "j" — `_nickname_token_equiv` does
        # the same dot-strip on input, so lookups round-trip.
        _n_norm = _n.rstrip(".")
        _NICKNAME_PAIRS.add((_full, _n_norm))
        _NICKNAME_PAIRS.add((_n_norm, _full))

# Title / honorific prefixes the predicate strips before comparing — handles
# `Dr. Elena Fischer` vs `Elena Fischer`, `Ayatollah Ali Khamenei` vs
# `Ali Khamenei`. Lowercase, trailing dot optional.
_TITLE_PREFIXES: frozenset[str] = frozenset(
    {
        "dr",
        "dr.",
        "mr",
        "mr.",
        "mrs",
        "mrs.",
        "ms",
        "ms.",
        "prof",
        "prof.",
        "professor",
        "sir",
        "ayatollah",
        "rabbi",
        "imam",
        "father",
        "sister",
        "brother",
        "president",
        "senator",
        "rep",
        "rep.",
        "gov",
        "gov.",
        "judge",
        "justice",
        "captain",
        "lieutenant",
        "lt",
        "lt.",
        "general",
        "gen",
        "gen.",
        "ambassador",
    }
)
_TITLE_PREFIX_STEMS: frozenset[str] = frozenset(p.rstrip(".") for p in _TITLE_PREFIXES)

# The canonical-map build compares every candidate against every same-show cluster member, so
# one name is cleaned millions of times on the production corpus. Cleaning is pure; cache it.
# Bounded so a long-lived server cannot grow it without limit. Bound on first use: importing
# kg.filters at module load runs kg/__init__, which imports the KG pipeline, and this module
# must stay importable from every layer without a cycle.
_clean_cached: Optional[Callable[[str], str]] = None
_is_acronymish: Optional[Callable[[str, str], bool]] = None


def _bind_kg_filters() -> None:
    global _clean_cached, _is_acronymish
    from podcast_scraper.kg.filters import _clean_entity_name, _is_acronymish as acronymish

    _clean_cached = functools.lru_cache(maxsize=1 << 16)(_clean_entity_name)
    _is_acronymish = acronymish


def _ratio_below(a: str, b: str, threshold: float) -> bool:
    """Exactly ``SequenceMatcher(None, a, b).ratio() < threshold``, mostly without the full match.

    ``real_quick_ratio`` and ``quick_ratio`` are upper bounds on ``ratio`` (difflib documents
    this), so a bound already under the threshold decides the answer.
    """
    matcher = difflib.SequenceMatcher(None, a, b)
    if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
        return True
    return matcher.ratio() < threshold


def _strip_title_prefix(tokens: List[str]) -> List[str]:
    """Drop a leading title token like 'Dr.' or 'Ayatollah'. Returns the rest."""
    if tokens and tokens[0].rstrip(".") in _TITLE_PREFIX_STEMS:
        return tokens[1:]
    return tokens


def _nickname_token_equiv(x: str, y: str) -> bool:
    """True if x and y are the same first-name modulo nickname (case-insensitive).

    Tolerates trailing-dot variants on both sides so `J.` and `j` and `j.`
    all canonicalise to the same lookup ("j" → set containing "jerome",
    plus "j." → same set after rstrip).
    """
    xn, yn = x.rstrip("."), y.rstrip(".")
    if xn == yn:
        return True
    return (xn, yn) in _NICKNAME_PAIRS


def _token_count_tolerant_match(a: str, b: str, kind: str) -> bool:
    """Handle token-count mismatches the strict aligned predicate rejects.

    Covers three patterns from the #853 silver-eval miss catalogue (#904
    scope):
    1. **Title-prefix strip** — `Dr. Elena Fischer` ↔ `Elena Fischer`,
       `Ayatollah Ali Khamenei` ↔ `Ali Khamenei`. After stripping a leading
       title from either side, retry the strict predicate.
    2. **Family-only reference** — `Carney` ↔ `Mark Carney`, `Trump` ↔
       `Donald Trump`. The single-token side must match the LAST token of
       the multi-token side EXACTLY (no fuzzy on last name alone — too
       risky, e.g. `Brown` vs `Mark Brown` is too thin a match).
    3. **First-name-only alias** — `Liam` ↔ `Liam Verbeek`. Symmetric to
       (2): single-token first name matches FIRST token of multi-token name
       exactly. Restricted to ``kind == "person"`` because the same pattern
       on orgs (`Adobe` ↔ `Adobe Creative Cloud`) is more likely a
       distinct sub-product than an alias.

    NOT covered (out of scope, would need richer signal):
    - Both sides multi-token but with different counts (e.g.
      ``Joe Eisenthal House`` vs ``Joe Weisenthal`` — Whisper artefact
      that inserts a spurious token mid-name).
    """
    ta, tb = a.split(), b.split()
    if len(ta) == len(tb):
        return False  # caller already handled equal-count case
    # 1. Title-prefix strip on either side (or both).
    sa, sb = _strip_title_prefix(ta), _strip_title_prefix(tb)
    if (sa != ta or sb != tb) and len(sa) == len(sb) and sa == sb:
        return True
    if kind == "person":
        short, long_ = (ta, tb) if len(ta) < len(tb) else (tb, ta)
        if len(short) == 1 and len(long_) >= 2:
            # 2. Family-only reference: short token matches long's LAST token exactly.
            #    e.g. `Carney` ↔ `Mark Carney`, `Trump` ↔ `Donald Trump`.
            if short[0] == long_[-1]:
                return True
            # NOTE: a first-name-only-merge rule (short matches long's first
            # token) was attempted in an earlier #904 draft but removed —
            # it can't distinguish ASR aliases (`Liam` ↔ `Liam Verbeek`,
            # same person) from organic same-first-name pairs (`Marco` ↔
            # `Marco Bianchi`, different people, the v2 two-Marcos test).
            # Both have identical predicate shape; differentiating needs
            # external signal (shared-episode evidence, same-show speaker
            # role, or LLM escalation). Tracked for follow-up; the Liam
            # alias case is fixture-only today and real-prod first-name-only
            # references rarely arise from ASR.
    return False


def cluster_variants(name_a: str, name_b: str, kind: str, language: Optional[str] = None) -> bool:
    """Mode ``cluster``: are two names one entity across episodes (the identity map's rule, #852).

    Conservative: spelling drift, not distinct names. ``kind`` is ``"person"`` or ``"org"``.
    ``language`` is accepted and not yet used: the titles and nicknames below are today's
    English-only lists for every language, and the per-language vocabulary is wired in a later,
    measured step (RFC-131, languages).
    """
    del language
    if _clean_cached is None or _is_acronymish is None:
        _bind_kg_filters()
    assert _clean_cached is not None and _is_acronymish is not None
    a, b = _clean_cached(name_a), _clean_cached(name_b)
    if not a or not b:
        return False
    if a == b:
        return True
    # Up-front title-prefix normalization so `President Trump` vs `Donald Trump`
    # (equal token counts after stripping a title from one side) routes through
    # the tolerant family-only-reference path instead of being rejected by the
    # token-ratio test. We compare both as-is AND title-stripped versions —
    # whichever pair matches.
    ta_full, tb_full = a.split(), b.split()
    ta_stripped, tb_stripped = _strip_title_prefix(ta_full), _strip_title_prefix(tb_full)
    if ta_stripped != ta_full or tb_stripped != tb_full:
        stripped_a, stripped_b = " ".join(ta_stripped), " ".join(tb_stripped)
        if stripped_a and stripped_b and stripped_a == stripped_b:
            return True
        # If only one side had a title prefix, recurse into the tolerant path
        # so the now-shorter name can match by family-only-reference.
        if (ta_stripped == ta_full) != (tb_stripped == tb_full):
            short, long_ = (
                (ta_stripped, tb_full) if ta_stripped != ta_full else (tb_stripped, ta_full)
            )
            # Family-only-reference only (no first-name-only-merge — see the
            # NOTE in `_token_count_tolerant_match` for why).
            if kind == "person" and len(short) == 1 and len(long_) >= 2 and short[0] == long_[-1]:
                return True
    # Spacing variants: "Data Bricks" == "Databricks", "Chat GPT" == "ChatGPT".
    if a.replace(" ", "") == b.replace(" ", "") and a.replace(" ", ""):
        return True
    ta, tb = ta_full, tb_full
    # Different token counts → tolerant matcher (title-prefix, family-only,
    # first-name-only). Runs BEFORE the acronym guard because short proper
    # nouns like `Trump`/`Liam`/`Brown` look acronym-ish to that guard but
    # are valid family-only-reference candidates when they appear as a
    # complete token in the other (longer) name. The tolerant matcher uses
    # exact-token equality, not fuzzy similarity, so the UPS-vs-USPS concern
    # the acronym guard exists to address doesn't apply on this path.
    # #904 lift: this branch was unconditional reject before #904.
    if len(ta) != len(tb):
        return _token_count_tolerant_match(a, b, kind)
    # Acronyms never fuzzy-merge (UPS != USPS) — guard applies to the
    # equal-token-count branch where we'd otherwise apply ratio similarity.
    if _is_acronymish(name_a, a) or _is_acronymish(name_b, b):
        return False
    if _ratio_below(a, b, _OVERALL_RATIO):
        # One escape hatch: same token-count, otherwise-identical names, but
        # one token differs by nickname (Mike Selig vs Michael Selig). The
        # ratio test rejects because Mike/Michael are too dissimilar — but
        # they ARE the same first name.
        if (
            kind == "person"
            and len(ta) == len(tb)
            and sum(1 for x, y in zip(ta, tb) if x != y) == 1
        ):
            diff_pairs = [(x, y) for x, y in zip(ta, tb) if x != y]
            if diff_pairs and _nickname_token_equiv(*diff_pairs[0]):
                return True
        return False
    # A PERSON'S NAME MAY DRIFT IN ONLY ONE TOKEN (#2056). Measured over 287 production
    # artifacts: every clearly-wrong same-show merge differs in BOTH tokens —
    # 'Albert Einstein'=='Bert Vogelstein', 'Jensen Huang'=='Jesse Zhang', 'Li Lun'=='Lily Liu' —
    # while almost every correct one differs in exactly one ('Bernard Leong'=='Bernard Leung',
    # 'Stewart Brand'=='Stuart Brand', 'Joe Weisenthal'=='Joe Wiesenthal'). A transcription slip
    # or typo lands on ONE token; two independent drifts is not one person spelled badly, it is
    # two people who rhyme.
    #
    # This costs one real merge in the sample — 'Alexander Carpi'/'Alexandra Karppi', a genuine
    # person whose name drifted twice — and prevents three false ones. Deliberate: a false split
    # is visible clutter, a false merge reassigns one person's statements to another.
    #
    # People only. Org names are compositional and legitimately differ in more than one token.
    if kind == "person" and sum(1 for x, y in zip(ta, tb) if x != y) > 1:
        return False
    # Token-aligned: every differing token pair must be a spelling variant, not a
    # distinct content word (audio vs media) or a version token (3, v2).
    for pos, (x, y) in enumerate(zip(ta, tb)):
        if x == y:
            continue
        if _VERSION_TOKEN_RE.search(x) or _VERSION_TOKEN_RE.search(y):
            return False  # numeric/version distinction
        if (
            kind == "person"
            and pos == len(ta) - 1
            and _ROMAN_NUMERAL_RE.match(x)
            and _ROMAN_NUMERAL_RE.match(y)
        ):
            return False  # regnal numbers: Charles I is not Charles II
        if _ratio_below(x, y, _TOKEN_RATIO) and not _nickname_token_equiv(x, y):
            return False  # distinct words, not a spelling variant
    return True


def name_words(name: str) -> int:
    """How many words the cluster rule compares: the cleaned name without a leading title.

    ``President Trump`` is one word here, because the rule reads it as ``Trump`` against a
    longer name (family-only reference).
    """
    if _clean_cached is None:
        _bind_kg_filters()
    assert _clean_cached is not None
    return len(_strip_title_prefix(_clean_cached(name).split()))


def cluster_person_variants(name_a: str, name_b: str, language: Optional[str] = None) -> bool:
    """Mode ``cluster_person``: mode ``cluster`` for people, except that two ONE-WORD names join
    only when they are one spelling (case and accents folded) or differ only in spacing.

    A single token carries too little to tell a misspelling from another name: on the production
    corpus (RFC-131 S4, N12) the fuzzy one-word joins were mostly two people, Hitler/Schiller,
    Einstein/Wittgenstein, Claude/Claudius, Yushun/Yushan (two guests of one show), against a
    few right ones (Tutankhamun/Tutankhamen). A false split is visible clutter; a false merge
    puts one person's words under another's name.
    """
    if _clean_cached is None:
        _bind_kg_filters()
    assert _clean_cached is not None
    a, b = _clean_cached(name_a), _clean_cached(name_b)
    if a and b and len(a.split()) == 1 and len(b.split()) == 1:
        return a == b  # cleaning already folds case and accents (Erdoğan -> erdogan)
    return cluster_variants(name_a, name_b, "person", language)


def host_variants(a: str, b: str, language: Optional[str] = None) -> bool:
    """Mode ``host_variant``: two host names of one feed that are spellings of one person (#1190).

    Shared surname (exact or one edit apart) and a near first name (up to two edits), and not
    identical. Conservative: it never merges a different surname or a first name more than a
    typo apart. ``language`` is accepted and not yet used (RFC-131, languages).
    """
    del language
    at, bt = a.split(), b.split()
    if len(at) < 2 or len(bt) < 2 or a.lower() == b.lower():
        return False
    surname_ok = at[-1].lower() == bt[-1].lower() or edit_distance(at[-1], bt[-1]) <= 1
    first_ok = edit_distance(at[0], bt[0]) <= 2
    return surname_ok and first_ok


#: The rule modes, by name. A consumer names its mode rather than importing a function.
MODES = {
    "cluster": cluster_variants,
    "cluster_person": cluster_person_variants,
    "host_variant": host_variants,
}


def same_person(
    a: str, b: str, *, mode: str, kind: str = "person", language: Optional[str] = None
) -> bool:
    """Are two names one person (or, for ``cluster``, one entity) under rule ``mode``?"""
    if mode == "cluster":
        return cluster_variants(a, b, kind, language)
    if mode == "cluster_person":
        return cluster_person_variants(a, b, language)
    if mode == "host_variant":
        return host_variants(a, b, language)
    raise ValueError(f"unknown same-person mode: {mode!r} (known: {sorted(MODES)})")

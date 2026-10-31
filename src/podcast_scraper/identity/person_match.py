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
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

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


@dataclass(frozen=True)
class _Vocab:
    """One language's name-matching vocabulary (``naming_vocabulary``; RFC-131 S5b, L1/L4/L5)."""

    titles: FrozenSet[str]  # title stems, trailing dot stripped
    nick_pairs: FrozenSet[Tuple[str, str]]
    folds: Tuple[Tuple[str, str], ...]
    family_first: bool


#: Built on first use (importing ``speaker_detectors`` at load would pull its package in). The
#: value None for a language means no vocabulary row: equal-after-normalisation only (L2).
_VOCABS: Dict[str, Optional[_Vocab]] = {}


def _pairs(table: Dict[str, FrozenSet[str]]) -> FrozenSet[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    for full, nicks in table.items():
        for nick in nicks:
            # Stored without the trailing dot; `_nickname_token_equiv` strips it on input too.
            n = nick.rstrip(".")
            out.update({(full, n), (n, full)})
    return frozenset(out)


def _vocab_for(language: Optional[str]) -> Optional[_Vocab]:
    """The vocabulary for ``language``: English for None (today's rule), None for no row."""
    from ..speaker_detectors import naming_vocabulary as nv

    code = "en" if language is None else nv.primary_subtag(language)
    if code in _VOCABS:
        return _VOCABS[code]
    titles = nv.NAME_MATCH_TITLES.get(code)
    if titles is None and code in nv.FAMILY_NAME_FIRST:
        titles = frozenset()  # no title list yet; the name order is known
    vocab: Optional[_Vocab] = None
    if titles is not None:
        folds = nv.NAME_FOLDS.get(code, ())
        if code == "en":
            stems = frozenset(
                t.rstrip(".") for t in titles
            )  # exactly the list the rule always used
        else:
            if _clean_cached is None:
                _bind_kg_filters()
            assert _clean_cached is not None
            stems = frozenset(_clean_cached(_fold(t, folds)).rstrip(".") for t in titles)
        vocab = _Vocab(
            stems, _pairs(nv.NAME_NICKNAMES.get(code, {})), folds, code in nv.FAMILY_NAME_FIRST
        )
    _VOCABS[code] = vocab
    return vocab


def _en() -> _Vocab:
    vocab = _vocab_for(None)
    assert vocab is not None
    return vocab


def _fold(name: str, folds: Tuple[Tuple[str, str], ...]) -> str:
    for src, dst in folds:
        name = name.replace(src, dst)
    return name


def _ratio_below(a: str, b: str, threshold: float) -> bool:
    """Exactly ``SequenceMatcher(None, a, b).ratio() < threshold``, mostly without the full match.

    ``real_quick_ratio`` and ``quick_ratio`` are upper bounds on ``ratio`` (difflib documents
    this), so a bound already under the threshold decides the answer.
    """
    matcher = difflib.SequenceMatcher(None, a, b)
    if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
        return True
    return matcher.ratio() < threshold


def _strip_title_prefix(tokens: List[str], vocab: Optional[_Vocab] = None) -> List[str]:
    """Drop a leading title token like 'Dr.' or 'Ayatollah'. Returns the rest."""
    if tokens and tokens[0].rstrip(".") in (vocab or _en()).titles:
        return tokens[1:]
    return tokens


def _nickname_token_equiv(x: str, y: str, vocab: Optional[_Vocab] = None) -> bool:
    """True if x and y are the same first-name modulo nickname (case-insensitive).

    Tolerates trailing-dot variants on both sides so `J.` and `j` and `j.`
    all canonicalise to the same lookup ("j" → set containing "jerome",
    plus "j." → same set after rstrip).
    """
    xn, yn = x.rstrip("."), y.rstrip(".")
    if xn == yn:
        return True
    return (xn, yn) in (vocab or _en()).nick_pairs


def _token_count_tolerant_match(a: str, b: str, kind: str, vocab: Optional[_Vocab] = None) -> bool:
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
    vocab = vocab or _en()
    sa, sb = _strip_title_prefix(ta, vocab), _strip_title_prefix(tb, vocab)
    if (sa != ta or sb != tb) and len(sa) == len(sb) and sa == sb:
        return True
    if kind == "person":
        short, long_ = (ta, tb) if len(ta) < len(tb) else (tb, ta)
        if len(short) == 1 and len(long_) >= 2:
            # 2. Family-only reference: short token matches long's LAST token exactly.
            #    e.g. `Carney` ↔ `Mark Carney`, `Trump` ↔ `Donald Trump`.
            if short[0] == (long_[0] if vocab.family_first else long_[-1]):
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
    ``language`` is accepted and ignored: this mode is the org rule, the read-time map's and the
    within-episode merge's, and keeps the English vocabulary for every language. The per-language
    rules are mode ``cluster_person`` (RFC-131 S5b).
    """
    del language
    return _cluster(name_a, name_b, kind, _en())


def _cluster(name_a: str, name_b: str, kind: str, vocab: _Vocab) -> bool:
    """Mode ``cluster``'s rule with one language's vocabulary."""
    if _cluster_rule(name_a, name_b, kind, vocab):
        return kind != "org" or _org_drift_within_one_edit(name_a, name_b)
    return kind == "org" and _org_same_without_article(name_a, name_b)


def _org_same_without_article(name_a: str, name_b: str) -> bool:
    """ "The Pragmatic Engineer" and "Pragmatic Engineer": EQUAL (case, spacing, hyphens aside) once
    a leading "the" goes, with two or more words left (RFC-131 S5o part 2).

    Exact on purpose: compared through the fuzzy rule instead, "The Nation" joined "Notion" on the
    production corpus. One word left ("The Times" / "Times") is too little to tell an outlet from
    a word. English only: orgs carry no language in mode ``cluster``, so another language's
    article is not dropped, which can only split.
    """
    if _clean_cached is None:
        _bind_kg_filters()
    assert _clean_cached is not None
    wa, wb = (
        _clean_cached(name_a).replace("-", " ").split(),
        _clean_cached(name_b).replace("-", " ").split(),
    )
    sa = wa[1:] if wa[:1] == ["the"] else wa
    sb = wb[1:] if wb[:1] == ["the"] else wb
    if (sa, sb) == (wa, wb) or len(sa) < 2 or len(sb) < 2:
        return False
    return "".join(sa) == "".join(sb)


def _org_drift_within_one_edit(name_a: str, name_b: str) -> bool:
    """An organisation's two names are one only when they differ by spacing or hyphens, or every
    differing word is at most ONE edit apart (RFC-131 S5o).

    The ratio test alone let names that rhyme or share a stem join: on the production corpus 61 of
    230 org joins were two entities (places onto places, products onto a company with the same
    prefix), 53 of them one word against one word. A one-letter slip is a transcription error; two
    letters is as often another name. The joins and their read are in the eval-data repository
    (``data/eval/name-consolidation/s5o/``).
    """
    assert _clean_cached is not None
    a, b = _clean_cached(name_a), _clean_cached(name_b)
    if a.replace(" ", "").replace("-", "") == b.replace(" ", "").replace("-", ""):
        return True
    ta, tb = a.replace("-", " ").split(), b.replace("-", " ").split()
    return len(ta) == len(tb) and all(edit_distance(x, y) <= 1 for x, y in zip(ta, tb) if x != y)


def _cluster_rule(name_a: str, name_b: str, kind: str, vocab: _Vocab) -> bool:
    """The cluster rule's spelling-drift test (#852, #904, #2056)."""
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
    ta_stripped, tb_stripped = _strip_title_prefix(ta_full, vocab), _strip_title_prefix(
        tb_full, vocab
    )
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
            family = long_[0] if vocab.family_first else long_[-1]
            if kind == "person" and len(short) == 1 and len(long_) >= 2 and short[0] == family:
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
        return _token_count_tolerant_match(a, b, kind, vocab)
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
            if diff_pairs and _nickname_token_equiv(*diff_pairs[0], vocab):
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
        if _ratio_below(x, y, _TOKEN_RATIO) and not _nickname_token_equiv(x, y, vocab):
            return False  # distinct words, not a spelling variant
    return True


def name_words(name: str) -> int:
    """How many words the cluster rule compares: the cleaned name without a leading title.

    "President <Surname>" is one word here, because the rule reads it as the bare surname against
    a longer name (family-only reference).
    """
    if _clean_cached is None:
        _bind_kg_filters()
    assert _clean_cached is not None
    return len(_strip_title_prefix(_clean_cached(name).split()))


def cluster_person_variants(name_a: str, name_b: str, language: Optional[str] = None) -> bool:
    """Mode ``cluster_person``: mode ``cluster`` for people, except that two ONE-WORD names join
    only when they are one spelling (case and accents folded) or differ only in spacing.

    A single token carries too little to tell a misspelling from another name: on the production
    corpus (RFC-131 S4, N12) 16 of the 23 fuzzy one-word joins were two people (two historical
    figures whose names rhyme; two guests of one show one letter apart), against 5 right ones
    (an alternative transliteration). The cases are listed by id in the eval-data repository
    (``data/eval/name-consolidation/s4/``). A false split is visible clutter; a false merge puts
    one person's words under another's name.

    ``language`` (RFC-131 S5b, the candidates' shared episode language): None or ``en`` is the
    English vocabulary, today's rule. A code with a ``naming_vocabulary.NAME_MATCH_TITLES`` row
    uses that row's titles, nicknames, folds and name order. Any other code, ``und`` included,
    joins only names equal after normalisation (L2): a missing vocabulary may split, never merge.
    """
    if _clean_cached is None:
        _bind_kg_filters()
    assert _clean_cached is not None
    vocab = _vocab_for(language)
    if vocab is None:
        return _equal_after_normalisation(name_a, name_b)
    name_a, name_b = _fold(name_a, vocab.folds), _fold(name_b, vocab.folds)
    a, b = _clean_cached(name_a), _clean_cached(name_b)
    if a and b and len(a.split()) == 1 and len(b.split()) == 1:
        return a == b  # cleaning already folds case and accents ("Oğuz" -> "oguz")
    if _cluster(name_a, name_b, "person", vocab):
        return True
    return _hyphen_or_initial_variant(a, b, vocab)


def _equal_after_normalisation(name_a: str, name_b: str) -> bool:
    """L2: one name, written with different case, accents, spacing or hyphens."""
    assert _clean_cached is not None
    a, b = _clean_cached(name_a), _clean_cached(name_b)
    if not a or not b:
        return False
    return bool(a.replace("-", "").replace(" ", "") == b.replace("-", "").replace(" ", ""))


def _is_initial(token: str) -> bool:
    bare = token.rstrip(".")
    return len(bare) == 1 and bare.isalpha()


def _hyphen_or_initial_variant(a: str, b: str, vocab: _Vocab) -> bool:
    """Two same-person forms the cluster rule refuses on word count (RFC-131 S5a), on cleaned names.

    Only ever adds to mode ``cluster``, so no pair it joins is refused:

    - **A hyphen read as a space** ("Orrell-Vanders" / "Orell Vanders"): retried through the
      cluster rule, so its one-drifting-word limit still applies.
    - **One middle initial on one side** ("Corwen T. Halloway" / "Corwen Halloway"): dropped only
      when the rest is IDENTICAL. A fuzzy spelling or nickname on top joined two different people
      on the production corpus (a bare first name onto a historical figure with a middle initial);
      two initials, or an initial on both sides, are not dropped either.

    The cases are listed in the eval-data repository (``data/eval/name-consolidation/s5/``).
    """
    ta, tb = a.replace("-", " ").split(), b.replace("-", " ").split()
    if ("-" in a or "-" in b) and _cluster(" ".join(ta), " ".join(tb), "person", vocab):
        return True
    ia = [i for i in range(1, len(ta) - 1) if _is_initial(ta[i])]
    ib = [i for i in range(1, len(tb) - 1) if _is_initial(tb[i])]
    if sorted((len(ia), len(ib))) != [0, 1]:
        return False
    rest_a = [t for i, t in enumerate(ta) if i not in ia]
    rest_b = [t for i, t in enumerate(tb) if i not in ib]
    return len(rest_a) >= 2 and rest_a == rest_b


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

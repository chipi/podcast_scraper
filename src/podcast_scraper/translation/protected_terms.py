"""Names the episode's own metadata states, kept out of the translator's hands (tier 2, 2026-10-10).

TranslateGemma translates proper nouns as words: "La Boca" (the Buenos Aires neighbourhood and
club) came back "Mouth", the podcast "Central" came back "flagship", a town as an adjective. On the
first non-English set about one in seven of the judge's low units was a name taken as a word. The
model's prompt is its own fixed template and cannot carry a glossary, so names are PROTECTED
instead: each one is replaced by a numbered placeholder before the request and put back after.

The terms come from what the episode states about itself -- the show's title, and capitalised
spans in the episode title and description that do not start a sentence -- so a term is a name
the publisher wrote, not one guessed from the transcript. Matching is case-sensitive and on whole
words: "Boca" is protected, "boca" (mouth) is not.

`restore` refuses (returns None) when a placeholder is missing, duplicated or altered; the caller
then translates the unit unprotected, as before, so protection can never make a unit worse.
Which placeholder form the model copies through verbatim is measured on the DGX, not assumed:
`PLACEHOLDER` is the one constant to change.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: `{n}` is the term's index. Bracketed, digit-only, no letters for the model to translate.
PLACEHOLDER = "[{n}]"
_PLACEHOLDER_RE = re.compile(r"\[(\d{1,3})\]")

#: Lower-case words a name may carry inside it ("Ciudad de México", "Rio de Janeiro").
_JOINERS = frozenset("de del da do dos das la las los el y e van von der den di".split())
#: Capitalised words that open a sentence or clause on their own, never a name.
_STOP = frozenset(
    "el la los las un una unos unas y o pero en de del con por para sobre este esta estos estas "
    "hoy ayer desde cuando porque como que si "
    "o a os as um uma e mas em no na nos nas com por para sobre este esta hoje quando "
    "le la les un une des et ou mais en dans du de avec pour sur ce cette aujourd'hui "
    "the a an and or but in on of with for this that today".split()
)
_WORD = re.compile(r"[\w'’\-]+|[^\w\s]")


def _capitalised(word: str) -> bool:
    return bool(word) and word[0].isupper()


#: Languages that capitalise every noun: a capitalised run there is no evidence of a name
#: (SWR's metadata yields "Organisierte Kriminalität", "Staatsangehöriger"), so only the show's
#: title is protected. The same set the naming code refuses one-word self-introductions in.
NOUN_CAPITALISING_LANGUAGES = frozenset({"de"})


def candidate_terms(
    *texts: Optional[str], show_title: Optional[str] = None, language: Optional[str] = None
) -> List[str]:
    """Names stated in the episode's own metadata, longest first, without duplicates.

    A name is a run of capitalised words (lower-case joiners allowed inside, a number allowed at
    the end: "Superclásico 94"). A single capitalised word that opens a sentence is not counted:
    it is capitalised because it comes first, which says nothing about whether it is a name."""
    terms: List[str] = []
    if show_title and show_title.strip():
        terms.append(show_title.strip())
    if (language or "").split("-")[0].lower() in NOUN_CAPITALISING_LANGUAGES:
        return terms
    for text in texts:
        for sentence in re.split(r"(?<=[.!?:;])\s+|\n+", text or ""):
            tokens = _WORD.findall(sentence)
            i = 0
            while i < len(tokens):
                if not _capitalised(tokens[i]) or not tokens[i][0].isalpha():
                    i += 1
                    continue
                j, words = i + 1, [tokens[i]]
                while j < len(tokens):
                    if _capitalised(tokens[j]):
                        words.append(tokens[j])
                    elif (
                        tokens[j] in _JOINERS
                        and j + 1 < len(tokens)
                        and _capitalised(tokens[j + 1])
                    ):
                        words.append(tokens[j])
                    elif tokens[j].isdigit() and len(words) >= 1:
                        words.append(tokens[j])
                        j += 1
                        break
                    else:
                        break
                    j += 1
                opening = i == 0 or (i == 1 and not tokens[0][0].isalnum())
                if opening and len(words) > 2 and words[1] in _JOINERS:
                    words = words[2:]  # "Hablamos de Boca": the opener is not part of it
                    opening = False
                if words[0].lower() in _STOP:
                    words = words[1:]
                    opening = False
                if (
                    words
                    and not (opening and len(words) == 1)
                    and len(" ".join(words)) >= 3
                    and max(len(w) for w in words) <= 25
                ):
                    terms.append(" ".join(words))
                i = j
    seen, out = set(), []
    for t in sorted(terms, key=len, reverse=True):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def terms_in(text: str, terms: Iterable[str]) -> List[str]:
    """The terms that occur in ``text`` as whole words, longest first."""
    return [t for t in terms if _pattern(t).search(text)]


def _pattern(term: str) -> "re.Pattern[str]":
    """Whole-word occurrences of ``term``; a ONE-word term not where a sentence starts, since a
    capitalised word there is capitalised for its position ("Apoya", "Filho" in a transcript)."""
    if " " in term:
        return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")
    return re.compile(rf"(?<![.!?¿¡:;\n]\s)(?<![.!?¿¡:;\n])(?<!^)(?<!\w){re.escape(term)}(?!\w)")


def protect(lines: Sequence[str], terms: Sequence[str]) -> Tuple[List[str], Dict[str, str]]:
    """``lines`` with each term replaced by a numbered placeholder, and the placeholder map."""
    mapping: Dict[str, str] = {}
    out = []
    for line in lines:
        for term in terms:
            pattern = _pattern(term)
            if not pattern.search(line):
                continue
            ph = next((p for p, t in mapping.items() if t == term), None)
            if ph is None:
                ph = PLACEHOLDER.format(n=len(mapping) + 1)
                mapping[ph] = term
            line = pattern.sub(ph, line)
        out.append(line)
    return out, mapping


def restore(
    lines: Sequence[str], sent: Sequence[str], mapping: Dict[str, str]
) -> Optional[List[str]]:
    """The reply's lines with the terms put back, or None when any line lost, gained or changed a
    placeholder relative to what was sent on that line."""
    out = []
    for got, src in zip(lines, sent):
        if sorted(_PLACEHOLDER_RE.findall(got)) != sorted(_PLACEHOLDER_RE.findall(src)):
            return None
        out.append(_PLACEHOLDER_RE.sub(lambda m: mapping.get(m.group(0), m.group(0)), got))
    return out if len(out) == len(lines) else None

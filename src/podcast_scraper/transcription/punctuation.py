"""Is a transcript unpunctuated, and how to ask Whisper for one that is not (#2284).

On prod (census 2026-10-05) 121 of 2,378 transcripts had no sentence punctuation and no capital
letters at all -- 118 of them from our DGX Whisper (faster-whisper large-v3-turbo, 6.8% of its
output). In 113 of the 121 the style holds from the first word: the server decodes sequentially,
conditioning each 30-second window on the previous one's text, so a lowercase first window sets
the style for the whole file. A punctuated ``prompt`` (Whisper's initial prompt) sets it instead:
on the shortest affected episode the same audio went from 0 to 92 sentence ends per 1,000 words.

Everything that needs sentences downstream (cleaning by selection, quotes, summaries) degrades
on such a transcript, so it is detected here rather than passed on.
"""

from __future__ import annotations

import re
from typing import Optional

from ..languages import primary_language

#: Below this many words a transcript is not judged: a music bed, a trailer or a one-line
#: episode can be legitimately short and sparse.
MIN_WORDS_TO_JUDGE = 300
#: A conversational transcript carries ~50-100 sentence ends per 1,000 words; the unpunctuated
#: prod transcripts carry 0-2. Same threshold the cleaning pass uses (``span_selection``). Sentence
#: ends ALONE decide: capitals do not -- Whisper often keeps capitalising names and acronyms
#: ("Bolivia", "AI", "U.S") while dropping every full stop; on prod (2026-10-05) 5 of the 121
#: unpunctuated transcripts had 3-10% capitalised words and no sentences.
MIN_SENTENCE_ENDS_PER_1000_WORDS = 5.0
#: Only text that is mostly Latin script is judged: the sentence-end and capital-letter signals
#: mean nothing for scripts without them.
MIN_LATIN_LETTER_SHARE = 0.5

#: Sent as Whisper's initial prompt to set a punctuated, capitalised style. Neutral on purpose:
#: Whisper reads the prompt as preceding speech, so it must not suggest a topic or a name.
PUNCTUATION_PROMPT = (
    "Hello, and welcome. In this episode, we talk about the topic at hand, with questions, "
    "answers, and full sentences."
)

_SPEAKER_LABEL = re.compile(r"(?m)^[^\n:.!?]{1,40}:\s")
#: A sentence end is . ! ? followed by whitespace, a closing quote/bracket, the end -- or directly
#: by a capital: some publisher transcripts drop the space ("factor.Now,"; In Moscow's Shadows).
_SENTENCE_END = re.compile(r"[.!?。！？](?:[\s\"'”)]|$|(?=[A-Z]))")
_WORD = re.compile(r"\S+")


def _body(text: str) -> str:
    """The spoken words: a screenplay's ``Name:`` labels are capitalised but are not speech."""
    return _SPEAKER_LABEL.sub("", text)


def sentence_ends_per_1000_words(text: str) -> float:
    """Sentence ends per 1,000 words (0.0 for empty text)."""
    body = _body(text)
    words = len(_WORD.findall(body))
    if not words:
        return 0.0
    return len(_SENTENCE_END.findall(body)) * 1000.0 / words


def is_unpunctuated(text: Optional[str]) -> bool:
    """True when ``text`` is long enough to judge and carries (almost) no sentence ends.

    On the prod corpus this flags the 121 known-defective transcripts plus every new one, and
    nothing else (2026-10-05, 2,397 transcripts of 300+ words).
    """
    if not text:
        return False
    body = _body(text)
    words = _WORD.findall(body)
    if len(words) < MIN_WORDS_TO_JUDGE:
        return False
    letters = [c for c in body if c.isalpha()]
    if not letters:
        return False
    latin = sum(1 for c in letters if c.isascii() or "À" <= c <= "ɏ")
    if latin / len(letters) < MIN_LATIN_LETTER_SHARE:
        return False
    return sentence_ends_per_1000_words(text) < MIN_SENTENCE_ENDS_PER_1000_WORDS


def echoes_prompt(text: Optional[str], prompt: str = PUNCTUATION_PROMPT) -> bool:
    """Did Whisper repeat the prompt as if it had been said? It sometimes does; such a
    transcript starts with words nobody spoke."""
    if not text:
        return False
    head = [w.strip(",.!?").lower() for w in _WORD.findall(prompt)[:6]]
    said = [w.strip(",.!?").lower() for w in _WORD.findall(_body(text))[:6]]
    return len(head) == len(said) and head == said


def prompt_suits_language(language: Optional[str]) -> bool:
    """The prompt is English; sent with another language it can push Whisper to translate."""
    return not language or primary_language(language) == "en"

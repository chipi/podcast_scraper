"""#1188 — a voice that says the same thing every week is a RECORDING, not a person.

The edge-ad rule (`_edge_ad_voices`) catches the pre-roll: short, and confined to the top or tail
of the episode. A MID-ROLL house ad defeats it by sitting in the middle — and the narrator reads
its own name aloud, which is the roster's most-trusted signal. So `Jonathan Knight` (NYT Games)
walked into 7 of 10 Hard Fork episodes as a named person, with `GUESTS_ON` edges.

No keyword can see it: modern house ads carry no sponsor language at all. But a signal exists that
no single episode can see — the ad is read from the same script every week, and the show is not:

    Jonathan Knight    1.3-1.7% of talk, 77-100% of his words repeat across the feed  <- a recording
    Jack Clark         0.6% of talk,     80% repeat                                   <- a recording
    Robert Armstrong   3.4-3.6% of talk, 70-77% repeat  (he reads the CREDITS)        <- a real host

The share is what separates them, and it is the same 3% bar the edge rule already uses. A co-host
reading the credits is above it; an advertisement is not.

Repetition alone is NOT enough to call something an advertisement — a show's own intro and outro
repeat too. But it is enough to say the voice is not contributing knowledge about THIS episode, and
that is the only question the roster is asking.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, Sequence, Set

# Long enough that ordinary phrasing does not collide across episodes by chance, short enough to
# survive the ASR spelling a word differently in two takes of the same ad.
SHINGLE_WORDS = 12

# A voice that is mostly reading a script the show repeats is being replayed, not interviewed.
RECORDED_MIN_REPEATED_FRACTION = 0.5

# ...but only when it barely speaks. The credits repeat verbatim every week and a real host reads
# them; at 3.4% of the talk Robert Armstrong is not an advertisement. Same bar as the edge rule.
RECORDED_MAX_SHARE = 0.03


def _words(text: str) -> list:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).split()


def recurring_shingles(transcripts: Sequence[str], min_episodes: int = 3) -> Set[str]:
    """Word sequences that appear in at least ``min_episodes`` of a feed's OWN episodes.

    This is the cross-episode evidence a single episode cannot have. Ads, credits, the show's intro
    and its outro all land here — and none of them is knowledge about the episode in hand.
    """
    seen: Dict[str, Set[int]] = defaultdict(set)
    for idx, text in enumerate(transcripts):
        words = _words(text)
        for i in range(len(words) - SHINGLE_WORDS + 1):
            seen[" ".join(words[i : i + SHINGLE_WORDS])].add(idx)
    bar = max(min_episodes, len(transcripts) // 2)
    return {sh for sh, eps in seen.items() if len(eps) >= bar}


def repeated_fraction(text: str, shingles: Set[str]) -> float:
    """How much of this text is script the show says every week (0.0 - 1.0)."""
    words = _words(text)
    if len(words) < SHINGLE_WORDS or not shingles:
        return 0.0
    hit = [False] * len(words)
    for i in range(len(words) - SHINGLE_WORDS + 1):
        if " ".join(words[i : i + SHINGLE_WORDS]) in shingles:
            for j in range(i, i + SHINGLE_WORDS):
                hit[j] = True
    return sum(hit) / len(words)


def recorded_voices(
    voice_texts: Dict[str, str],
    talk: Dict[str, float],
    shingles: Set[str],
) -> Set[str]:
    """Voices that are a REPLAYED SCRIPT rather than a person in the room.

    Both tests must hold. Repetition alone would strip a co-host who reads the credits; a small
    share alone is just a brief speaker. Together they describe an advertisement and nothing else.
    """
    if not shingles or not voice_texts:
        return set()
    total = sum(talk.values()) or 1.0
    out: Set[str] = set()
    for voice, text in voice_texts.items():
        share = talk.get(voice, 0.0) / total
        if share >= RECORDED_MAX_SHARE:
            continue
        if repeated_fraction(text, shingles) >= RECORDED_MIN_REPEATED_FRACTION:
            out.add(voice)
    return out


def shingles_from_transcript_files(paths: Iterable) -> Set[str]:
    """Build the feed's recurring-script index from its transcripts on disk."""
    texts = []
    for p in paths:
        try:
            texts.append(p.read_text(encoding="utf-8"))
        except OSError:
            continue
    return recurring_shingles(texts) if len(texts) >= 3 else set()

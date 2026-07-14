"""ADR-110 — ask who speaks AFTER we can hear them.

`detect_speakers(title, description, known_hosts)` is asked "who are the speakers?" *before the
audio is downloaded*. Its interface cannot take a transcript. So an LLM shown only the show notes
returns the people they MENTION — which is how `Elon Musk`, named in a Hard Fork description solely
as the man *suing* OpenAI, was returned as a speaker and published as the author of a real guest's
words (#876). `corroborate_guests` then checked that guess against the same show notes it was
guessed from, which is circular, so it fell back to a regex looking for an interview cue — and desk
shows never write one. Measured on 50 episodes through the prod detector, that gate deleted 70
proposed names, 69 of them whole and correct, including Rob Armstrong, the co-host of FT Unhedged.

This module asks the question where the answer lives: after diarization, against **each voice's own
turns**. The model must point at a VOICE and it may only choose from the names the metadata already
STATED — so it cannot invent a speaker, only match one, or decline.

Declining is a first-class answer. A voice nobody names stays unnamed (`unknown`), because a wrong
name is worse than no name.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# How much of each voice we show the model. The opening turns are where people are introduced and
# introduce themselves; further in, everyone is just talking about the topic and the signal is gone.
VOICE_SAMPLE_CHARS = 1200

# A voice with less than this much to say cannot be identified from its words, and asking the model
# to try invites a guess. Cameos and backchannel ("Yeah." "Right.") live here.
MIN_SAMPLE_CHARS = 80


# How much of the transcript to show around each MENTION of a candidate name. This is the retrieval
# step, and it is what tells a speaker apart from a subject: "Elon Musk is suing OpenAI" and "Jia Li
# is with us today" both MENTION a person, and only the sentence says which one is in the room.
MENTION_CONTEXT_CHARS = 220
MAX_MENTIONS_PER_NAME = 4


def _speaker_sample(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())[:VOICE_SAMPLE_CHARS]


def retrieve_mentions(
    name: str, ordered_turns: Sequence[tuple], context_chars: int = MENTION_CONTEXT_CHARS
) -> List[str]:
    """Every passage where this NAME is spoken, with who said it and who spoke NEXT.

    The retrieval half of the problem. A name's presence in a transcript proves nothing — a podcast
    discusses people constantly — but the SENTENCE AROUND IT is decisive, and so is the turn that
    follows it: the person a host introduces is the person who speaks next.

    Matching is exact on the full name and on the surname, so "Jia Li" is found whether the host
    says "Jia Li is with us" or "welcome, Li". No embeddings: identity is not a similarity question,
    and a fuzzy match here is how you assign a voice to the wrong person.
    """
    tokens = [t for t in re.split(r"\s+", name.strip()) if t]
    if not tokens:
        return []
    surname = tokens[-1]
    pattern = re.compile(rf"\b(?:{re.escape(name)}|{re.escape(surname)})\b", re.IGNORECASE)

    out: List[str] = []
    for i, (voice, text) in enumerate(ordered_turns):
        body = str(text or "")
        for m in pattern.finditer(body):
            lo = max(0, m.start() - context_chars // 2)
            hi = min(len(body), m.end() + context_chars // 2)
            passage = re.sub(r"\s+", " ", body[lo:hi]).strip()
            nxt = ordered_turns[i + 1][0] if i + 1 < len(ordered_turns) else None
            out.append(
                f'said by {voice}: "...{passage}..."'
                + (f" -> the NEXT voice to speak is {nxt}" if nxt and nxt != voice else "")
            )
            if len(out) >= MAX_MENTIONS_PER_NAME:
                return out
    return out


def build_resolution_prompt(
    stated_names: Sequence[str],
    voice_texts: Dict[str, str],
    known_hosts: Sequence[str] = (),
    ordered_turns: Optional[Sequence[tuple]] = None,
) -> str:
    """The question that HAS an answer: which of these named people is each of these voices?

    The candidate list is closed. The model picks from it or says ``null`` — it is never asked to
    produce a name, so it cannot produce one that was never stated. Alongside the closed list it is
    given the RETRIEVED EVIDENCE for each name: every passage in which that name is actually spoken.
    """
    hosts = ", ".join(known_hosts) if known_hosts else "(not stated)"

    roster_lines = []
    for n in stated_names:
        mentions = retrieve_mentions(n, ordered_turns or [])
        roster_lines.append(f"  - {n}")
        if mentions:
            for passage in mentions:
                roster_lines.append(f"      * {passage}")
        else:
            roster_lines.append(
                "      * NEVER SPOKEN ALOUD in this episode — the show notes name them and the "
                "conversation does not."
            )
    roster = "\n".join(roster_lines) or "  (none)"

    voices = []
    for voice, text in voice_texts.items():
        sample = _speaker_sample(text)
        if len(sample) < MIN_SAMPLE_CHARS:
            continue
        voices.append(f'  {voice}: "{sample}"')
    voice_block = "\n".join(voices)

    return f"""You are matching diarized voices to the people an episode's metadata names.

PEOPLE THE EPISODE METADATA NAMES (the ONLY names you may use), each followed by every passage in
the transcript where that name is actually SPOKEN:
{roster}

Known hosts of the show: {hosts}

VOICES, each shown with the opening of its OWN speech:
{voice_block}

For each voice, decide which of the named people it is.

RULES — these matter more than covering every voice:
1. You may ONLY use a name from the list above. Never invent a name, and never use a name that is
   not on the list, even if a voice mentions one.
2. Many of the named people DO NOT SPEAK. Show notes name the people an episode is ABOUT as well as
   the people in the room — a lawsuit defendant, a politician, a founder who died in 1956. Read the
   retrieved passage: "Elon Musk is suing OpenAI" names a SUBJECT, "Jia Li is with us today" names
   a SPEAKER. If a voice is not clearly one of the named people, answer null. Null is CORRECT and
   expected, and is always better than a plausible guess.
3. Evidence is the voice's own words or an introduction of it. A voice that says "I'm Peter Ludwig"
   IS Peter Ludwig. The person a host introduces is usually the NEXT voice to speak — the passages
   above tell you who that is. Topic overlap is NOT evidence: a voice discussing a person is not
   that person.
4. A name marked NEVER SPOKEN ALOUD is almost certainly not in the room. Assign it only if a voice
   unmistakably speaks as that person.
5. Never assign the same name to two voices.

Return JSON only:
{{"voices": {{"SPEAKER_00": "Full Name or null", "SPEAKER_01": null}}}}"""


def _parse(raw: str) -> Dict[str, Optional[str]]:
    """Pull the mapping out of the model's answer, tolerating fences and reasoning preambles."""
    text = (raw or "").strip()
    if not text:
        return {}
    # A reasoning model may emit <think>…</think> before the JSON; scope past it.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        logger.warning("speaker resolution: unparsable response, resolving nobody")
        return {}
    voices = obj.get("voices") if isinstance(obj, dict) else None
    if not isinstance(voices, dict):
        return {}
    return {str(k): (str(v) if v not in (None, "", "null") else None) for k, v in voices.items()}


def resolve_voices_from_conversation(
    stated_names: Sequence[str],
    voice_texts: Dict[str, str],
    complete: Callable[[str], str],
    known_hosts: Sequence[str] = (),
    ordered_turns: Optional[Sequence[tuple]] = None,
) -> Dict[str, str]:
    """``{voice: name}`` for the voices the model could identify, from the CLOSED stated list.

    ``complete`` is any "prompt in, text out" callable, so this stays provider-agnostic and is
    trivially testable without a network.

    Everything the model returns is verified against the stated list before it is believed. The
    model is an *identifier*, never an author: if it answers with a name nobody stated, the answer
    is discarded, because that is precisely the failure this exists to prevent.
    """
    stated = [n for n in (stated_names or ()) if str(n).strip()]
    if not stated or not voice_texts:
        return {}

    prompt = build_resolution_prompt(stated, voice_texts, known_hosts, ordered_turns)
    try:
        raw = complete(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("speaker resolution failed (%s); no voice is named from it", exc)
        return {}

    by_stated = {n.lower(): n for n in stated}
    out: Dict[str, str] = {}
    used: set = set()
    invented: List[str] = []

    for voice, name in _parse(raw).items():
        if not name or voice not in voice_texts:
            continue
        canonical = by_stated.get(name.strip().lower())
        if canonical is None:
            invented.append(name)
            continue
        if canonical.lower() in used:  # rule 4 — one person, one voice
            continue
        used.add(canonical.lower())
        out[voice] = canonical

    if invented:
        logger.warning(
            "speaker resolution proposed %d name(s) the metadata never stated (%s) — DISCARDED. "
            "The model may identify a voice, never author a name.",
            len(invented),
            ", ".join(sorted(set(invented))),
        )
    if out:
        logger.info(
            "speaker resolution named %d/%d voice(s) from the conversation: %s",
            len(out),
            len(voice_texts),
            ", ".join(f"{v}={n}" for v, n in sorted(out.items())),
        )
    return out


def completion_fn_for(provider: Any) -> Optional[Callable[[str], str]]:
    """A "prompt in, text out" callable for a provider that can do one, else ``None``.

    ``None`` is the airgapped answer: the spaCy detector has no LLM, so the deterministic cue
    matcher stays in charge and nothing about those profiles changes.
    """
    fn = getattr(provider, "complete_text", None)
    return fn if callable(fn) else None

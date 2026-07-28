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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# The host/guest roles the model is allowed to assert. Anything else is discarded like an invented
# name — a prompt is not an enforcement mechanism (#876), so the vocabulary is closed in code.
_VALID_ROLES = {"host", "guest"}


@dataclass(frozen=True)
class LLMVoice:
    """One voice's LLM verdict: a matched ``name`` (from the closed stated list) and/or a host/guest
    ``role``. Either may be ``None`` — the model is allowed, and expected, to decline (ADR-135)."""

    name: Optional[str] = None
    role: Optional[str] = None  # "host" | "guest" | None


def _coerce_name(value: Any) -> Optional[str]:
    return str(value) if value not in (None, "", "null") else None


def _coerce_role(value: Any) -> Optional[str]:
    role = str(value).strip().lower() if value not in (None, "", "null") else None
    return role if role in _VALID_ROLES else None


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
            # "said by X" reads as "X is associated with this name", which is the opposite of what a
            # third-person mention means. Say what it actually is: somebody TALKING ABOUT them.
            out.append(
                f'{voice} says this ABOUT them (so {voice} is probably NOT them): "...{passage}..."'
                + (f" | the NEXT voice to speak is {nxt}" if nxt and nxt != voice else "")
            )
            if len(out) >= MAX_MENTIONS_PER_NAME:
                return out
    return out


def build_resolution_prompt(
    stated_names: Sequence[str],
    voice_texts: Dict[str, str],
    known_hosts: Sequence[str] = (),
    ordered_turns: Optional[Sequence[tuple]] = None,
    episode_title: Optional[str] = None,
    episode_description: Optional[str] = None,
    intro_block: Optional[str] = None,
) -> str:
    """Two questions with answers: which named person is each voice, and is it a host or a guest?

    The candidate list is closed. The model picks a name from it or says ``null`` — never asked to
    produce a name, so it cannot produce one that was never stated. Alongside the closed list it is
    given the RETRIEVED EVIDENCE for each name, and — for the host/guest role (ADR-135) — the
    episode title, description, and the cleaned, speaker-labeled intro, which is where a show states
    who is hosting and who is visiting.
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

    title = (episode_title or "").strip() or "(not provided)"
    description = (episode_description or "").strip() or "(not provided)"
    intro = (intro_block or "").strip() or "(not provided)"

    return f"""You are matching diarized voices to the people an episode's metadata names, and \
deciding which are HOSTS and which are GUESTS.

EPISODE TITLE: {title}

EPISODE DESCRIPTION: {description}

THE INTRO (first minutes, speaker-labeled, ads/cameos removed) — where a show usually says who hosts
and who is visiting:
{intro}

PEOPLE THE EPISODE METADATA NAMES (the ONLY names you may use), each followed by every passage in
the transcript where that name is actually SPOKEN:
{roster}

Known hosts of the show: {hosts}

VOICES, each shown with the opening of its OWN speech:
{voice_block}

For each voice, decide (a) which of the named people it is, and (b) whether it is a host or a guest.

RULES — these matter more than covering every voice:
1. NAME: you may ONLY use a name from the list above. Never invent a name, and never use a name that
   is not on the list, even if a voice mentions one.
2. Many of the named people DO NOT SPEAK. Show notes name the people an episode is ABOUT as well as
   the people in the room — a lawsuit defendant, a politician, a founder who died in 1956. Read the
   retrieved passage: "Elon Musk is suing OpenAI" names a SUBJECT, "Jia Li is with us today" names
   a SPEAKER. If a voice is not clearly one of the named people, answer null for name. Null is
   CORRECT and expected, and is always better than a plausible guess.
3. Evidence is the voice's own words or an introduction of it. A voice that says "I'm Peter Ludwig"
   IS Peter Ludwig. The person a host introduces is usually the NEXT voice to speak — the passages
   above tell you who that is. Topic overlap is NOT evidence: a voice discussing a person is not
   that person.
4. A name marked NEVER SPOKEN ALOUD is almost certainly not in the room. Assign it only if a voice
   unmistakably speaks as that person.
5. Never assign the same name to two voices.
6. ROLE: use the title, description and intro. The host welcomes listeners to the show and
   introduces the guest ("welcome to X, I'm your host…", "my guest today is…"); the guest is being
   interviewed ("thanks for having me"). A person the description presents as the interviewee is a
   GUEST even if a co-host is absent and a host seat looks empty. Answer "host", "guest", or null.
   null is correct when you cannot tell.
7. ABSTAIN on brief, anonymous voices — a member of the public in a field clip, a one-line cameo.
   Give them null name AND null role. Never invent a name or force a role onto them.

Return JSON only, one object per voice:
{{"voices": {{"SPEAKER_00": {{"name": "Full Name or null", "role": "host|guest|null"}}}}}}"""


def _parse(raw: str) -> Dict[str, LLMVoice]:
    """Pull the ``{voice: LLMVoice(name, role)}`` mapping out of the model's answer.

    Tolerates fences and reasoning preambles, and BOTH output shapes so a legacy prompt/response and
    the ADR-135 role-bearing one both parse:
      legacy  ``{"voices": {"SPEAKER_00": "Name"}}``            → name only, role None
      current ``{"voices": {"SPEAKER_00": {"name": "Name", "role": "host"}}}``
    """
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
    out: Dict[str, LLMVoice] = {}
    for k, v in voices.items():
        if isinstance(v, dict):
            out[str(k)] = LLMVoice(
                name=_coerce_name(v.get("name")), role=_coerce_role(v.get("role"))
            )
        else:
            out[str(k)] = LLMVoice(name=_coerce_name(v), role=None)
    return out


def _introduces_itself_as(text: str, name: str) -> bool:
    """Does this voice say "I'm X" / "this is X" / "my name is X" in its own turns?"""
    first = re.split(r"\s+", name.strip())[0]
    return bool(
        re.search(
            rf"\b(?:I'?m|I am|my name is|this is)\s+(?:{re.escape(name)}|{re.escape(first)})\b",
            text or "",
            re.IGNORECASE,
        )
    )


def _talks_about(text: str, name: str) -> bool:
    """Does this voice utter the name at all (in any context)?"""
    tokens = [t for t in re.split(r"\s+", name.strip()) if t]
    if not tokens:
        return False
    return bool(
        re.search(rf"\b(?:{re.escape(name)}|{re.escape(tokens[-1])})\b", text or "", re.IGNORECASE)
    )


def _refuted_by_third_person(voice_text: str, name: str) -> bool:
    """IF YOU SAY SOMEBODY'S NAME IN THE THIRD PERSON, YOU ARE NOT THEM.

    The retrieval that makes this work is also what misleads the model. It hands over passages
    labelled "said by SPEAKER_01: '...Jay Powell, chair of the Federal Reserve, made a joke...'" and
    a model reads the name sitting next to the voice as association — so on an FT Unhedged episode
    ABOUT the Fed, it gave 53.5% of the show to Jay Powell. SPEAKER_01 is Rob Armstrong, the
    co-host, discussing him.

    So this is checked, not asked for. A voice that utters a name and never introduces itself with
    it is talking ABOUT that person, and cannot BE them. Deterministic, like the closed-list rule —
    a prompt is not an enforcement mechanism (#876).
    """
    return _talks_about(voice_text, name) and not _introduces_itself_as(voice_text, name)


def resolve_voices_and_roles(
    stated_names: Sequence[str],
    voice_texts: Dict[str, str],
    complete: Callable[[str], str],
    known_hosts: Sequence[str] = (),
    ordered_turns: Optional[Sequence[tuple]] = None,
    episode_title: Optional[str] = None,
    episode_description: Optional[str] = None,
    intro_block: Optional[str] = None,
) -> Dict[str, LLMVoice]:
    """``{voice: LLMVoice(name, role)}`` — the name AND host/guest role, in one call (ADR-135).

    ``complete`` is any "prompt in, text out" callable, so this stays provider-agnostic and is
    trivially testable without a network.

    Everything the model returns is verified before it is believed. The model is an *identifier and
    a classifier*, never an author: a name nobody stated is discarded (that is the #876 failure this
    exists to prevent), a name a voice only speaks in the third person is discarded, and a role
    outside {host, guest} is dropped. Name and role are independent: a voice may keep its role even
    if its name is refuted, and vice-versa.
    """
    stated = [n for n in (stated_names or ()) if str(n).strip()]
    # Naming needs a closed candidate list, but ROLE does not — it reads title/description/intro.
    # A no-stated-host show (Planet Money) names nobody in metadata, yet its hosts self-introduce
    # on air; run in ROLE-ONLY mode so the model can still say host/guest with no candidates (names
    # stay closed and come back null). Requires real voices AND some role context.
    has_role_context = bool(episode_title or episode_description or intro_block)
    if not voice_texts or (not stated and not has_role_context):
        return {}

    prompt = build_resolution_prompt(
        stated,
        voice_texts,
        known_hosts,
        ordered_turns,
        episode_title,
        episode_description,
        intro_block,
    )
    try:
        raw = complete(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("speaker resolution failed (%s); no voice is named/roled from it", exc)
        return {}

    by_stated = {n.lower(): n for n in stated}
    out: Dict[str, LLMVoice] = {}
    used: set = set()
    invented: List[str] = []
    refuted: List[str] = []

    for voice, verdict in _parse(raw).items():
        if voice not in voice_texts:
            continue
        canonical: Optional[str] = None
        if verdict.name:
            match = by_stated.get(verdict.name.strip().lower())
            if match is None:
                invented.append(verdict.name)
            elif _refuted_by_third_person(voice_texts[voice], match):
                refuted.append(f"{voice}={match}")
            elif match.lower() in used:  # rule 5 — one person, one voice
                pass
            else:
                used.add(match.lower())
                canonical = match
        if canonical or verdict.role:
            out[voice] = LLMVoice(name=canonical, role=verdict.role)

    if invented:
        logger.warning(
            "speaker resolution proposed %d name(s) the metadata never stated (%s) — DISCARDED. "
            "The model may identify a voice, never author a name.",
            len(invented),
            ", ".join(sorted(set(invented))),
        )
    if refuted:
        logger.warning(
            "speaker resolution assigned %d name(s) to a voice that TALKS ABOUT that person in the "
            "third person and never introduces itself as them (%s) — DISCARDED. Saying somebody's "
            "name does not make you them.",
            len(refuted),
            ", ".join(sorted(set(refuted))),
        )
    if out:
        logger.info(
            "speaker resolution: %d/%d voice(s) resolved from the conversation: %s",
            len(out),
            len(voice_texts),
            ", ".join(f"{v}={lv.name or '?'}/{lv.role or '?'}" for v, lv in sorted(out.items())),
        )
    return out


def resolve_voices_from_conversation(
    stated_names: Sequence[str],
    voice_texts: Dict[str, str],
    complete: Callable[[str], str],
    known_hosts: Sequence[str] = (),
    ordered_turns: Optional[Sequence[tuple]] = None,
) -> Dict[str, str]:
    """``{voice: name}`` — the name-only view (unchanged contract). Delegates to
    :func:`resolve_voices_and_roles` and projects away the role."""
    resolved = resolve_voices_and_roles(
        stated_names, voice_texts, complete, known_hosts=known_hosts, ordered_turns=ordered_turns
    )
    return {voice: lv.name for voice, lv in resolved.items() if lv.name}


def completion_fn_for(provider: Any) -> Optional[Callable[[str], str]]:
    """A "prompt in, text out" callable for a provider that can do one, else ``None``.

    ``None`` is the airgapped answer: the spaCy detector has no LLM, so the deterministic cue
    matcher stays in charge and nothing about those profiles changes.
    """
    fn = getattr(provider, "complete_text", None)
    return fn if callable(fn) else None

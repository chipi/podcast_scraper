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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# The host/guest roles the model is allowed to assert. Anything else is discarded like an invented
# name — a prompt is not an enforcement mechanism (#876), so the vocabulary is closed in code.
_VALID_ROLES = {"host", "guest"}


@dataclass(frozen=True)
class LLMVoice:
    """One voice's LLM verdict: a matched ``name`` (from the closed stated list) and/or a host/guest
    ``role``. Either may be ``None`` — the model is allowed, and expected, to decline (ADR-137)."""

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


# A spoken "First Last". The lookbehind also accepts a lowercase letter, because publisher and ASR
# cues are often joined with no space ("Kansas City Fed PresidentJeff Schmidt").
# Zero-width, so candidate pairs OVERLAP: "Fed PresidentJeff Schmidt" must yield "Jeff Schmidt"
# even though "Fed PresidentJeff" is also a capitalised pair.
_SPOKEN_FULL_NAME = re.compile(
    r"(?:(?<![A-Za-z])|(?<=[a-z]))(?=([A-Z][a-z'’\-]+)\s+([A-Z][a-zA-Z'’\-]+))"
)
# "I'm Tracy Allaway", "I am", "my name is": the voice saying it IS the person. NOT "this is": that
# is how a host introduces a guest ("this is Matthew Cobb's seventh book"), and framed as a
# self-introduction it told the model the host was the guest (advisor review, #2075).
_SELF_INTRO_BEFORE = re.compile(r"(?:\bI['’]?m|\bI am|\bmy name is)\s*$", re.IGNORECASE)


class _Span:
    """A match position; exact and variant matches are handled alike."""

    def __init__(self, start: int, end: int) -> None:
        self._s, self._e = start, end

    def start(self) -> int:
        return self._s

    def end(self) -> int:
        return self._e


def _one_edit_apart(a: str, b: str) -> bool:
    """True for strings at Levenshtein distance exactly 1 (case-insensitive)."""
    a, b = a.lower(), b.lower()
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i + (len(a) == len(b)) :] == b[i + 1 :]


def _mentions_of(name: str, tokens: List[str], exact: "re.Pattern[str]", body: str) -> List[Any]:
    """Exact full-name / surname matches, plus the SPOKEN VARIANTS of the full name (#2075).

    Measured on an Odd Lots episode: the show notes say `Jeffrey Schmid` and `Tracy Alloway`, the
    transcript says "Jeff Schmidt" and "I'm Tracy Allaway", and the prompt told the model both were
    "NEVER SPOKEN ALOUD" — which it is instructed to read as "almost certainly not in the room".
    A variant counts only as a FULL name: the given name equal or a known nickname
    (`first_names_match`) AND a 5+-letter surname one edit away. A bare surname stays exact, so a
    passage about Eric Schmidt is never shown as evidence about Jeffrey Schmid.
    """
    from ..text_normalization import first_names_match

    found: List[Any] = [_Span(m.start(), m.end()) for m in exact.finditer(body)]
    if len(tokens) < 2 or len(tokens[-1]) < 5:
        return found
    for m in _SPOKEN_FULL_NAME.finditer(body):
        given, last = m.group(1), m.group(2)
        if not (given.lower() == tokens[0].lower() or first_names_match(tokens[0], given)):
            continue
        if not _one_edit_apart(last, tokens[-1]):
            continue
        lo, hi = m.start(1), m.end(2)
        if any(f.start() < hi and lo < f.end() for f in found):
            continue
        found.append(_Span(lo, hi))
    return sorted(found, key=lambda x: x.start())


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
        for m in _mentions_of(name, tokens, pattern, body):
            lo = max(0, m.start() - context_chars // 2)
            hi = min(len(body), m.end() + context_chars // 2)
            passage = re.sub(r"\s+", " ", body[lo:hi]).strip()
            # The next DIFFERENT voice. Turns arrive per ASR segment, so the turn after a mention is
            # usually the same speaker finishing the sentence, and the hand-off was never shown:
            # "…Kansas City Fed President Jeff Schmidt. Thank you so much for coming back on…"
            # carried no next voice at all (#2075, Odd Lots, the pipeline's own prompt).
            nxt = next((v for v, _t in ordered_turns[i + 1 :] if v != voice), None)
            # "said by X" reads as "X is associated with this name", which is the opposite of what a
            # third-person mention means. Say what it actually is: somebody TALKING ABOUT them.
            # EXCEPT a self-introduction — "And I'm Joe Weisenthal" was being presented as Joe's
            # own voice "probably NOT" being Joe (#2075, Odd Lots, measured on the DGX).
            if _SELF_INTRO_BEFORE.search(body[max(0, m.start() - 20) : m.start()]):
                line = f'{voice} INTRODUCES ITSELF as them (so {voice} IS them): "...{passage}..."'
            else:
                line = (
                    f"{voice} says this ABOUT them (so {voice} is probably NOT them): "
                    f'"...{passage}..."'
                    + (f" | the NEXT voice to speak is {nxt}" if nxt and nxt != voice else "")
                )
            if line in out:
                continue  # two matches inside one passage are one piece of evidence
            out.append(line)
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
    given the RETRIEVED EVIDENCE for each name, and — for the host/guest role (ADR-137) — the
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


def _voice_id_in(said: str, voice_texts: Dict[str, str]) -> Optional[str]:
    """The voice id the model MEANT. It writes `SPEAKER_2` for `SPEAKER_02` (vLLM on the DGX,
    measured on an Odd Lots episode), and an exact lookup then discarded every one of its answers —
    both hosts, correctly identified. Same prefix and same number is the same voice; ambiguity
    resolves to nobody."""
    if said in voice_texts:
        return said
    m = re.fullmatch(r"(.*?)(\d+)", said.strip())
    if not m:
        return None
    hits = [
        v
        for v in voice_texts
        if (mv := re.fullmatch(r"(.*?)(\d+)", v))
        and mv.group(1).lower() == m.group(1).lower()
        and int(mv.group(2)) == int(m.group(2))
    ]
    return hits[0] if len(hits) == 1 else None


def _parse(raw: str) -> Dict[str, LLMVoice]:
    """Pull the ``{voice: LLMVoice(name, role)}`` mapping out of the model's answer.

    Tolerates fences and reasoning preambles, and BOTH output shapes so a legacy prompt/response and
    the ADR-137 role-bearing one both parse:
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
    if (
        voices is None
        and isinstance(obj, dict)
        and obj
        and all(re.fullmatch(r"[A-Za-z_]*\d+", str(k)) for k in obj)
    ):
        # The model dropped the `voices` wrapper and answered with the voice map itself (vLLM on the
        # DGX, measured) — every key a voice id. Read it rather than resolving nobody.
        voices = obj
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
    """Does this voice say "I'm X" / "this is X" / "my name is X" in its own turns?

    NOT when the name is POSSESSIVE. "this is Matthew Cobb's seventh book" is a host describing the
    guest's work, and reading it as a self-introduction makes this function vouch for the model's
    answer — which is how Ground Truths published the host's name on the guest's voice and the
    guest's on the host's, each confirming the other. `dff4c116` removed the construction from the
    resolver's PROMPT; this is the same defect in the code that is supposed to check the prompt's
    output, where it actually matters (#876: a prompt is not an enforcement mechanism).
    """
    first = re.split(r"\s+", name.strip())[0]
    return bool(
        re.search(
            # The possessive lookahead spans the REST of the name phrase, not just the matched
            # token: anchored to "Matthew" alone, "this is Matthew Cobb's seventh book" still
            # matched, because the engine simply backtracked to the first-name alternative.
            rf"\b(?:I'?m|I am|my name is|this is)\s+"
            rf"(?:{re.escape(name)}|{re.escape(first)})"
            rf"(?!(?:\s+[A-Z][\w'’\-]*){{0,2}}['’]s\b)\b",
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


def refuted_by_third_person(voice_text: str, name: str) -> bool:
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
    """``{voice: LLMVoice(name, role)}`` — the name AND host/guest role, in one call (ADR-137).

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

    def _stated_match(said: str) -> Optional[str]:
        """The stated name the model meant. Exact first; else the ONE stated name its words are a
        spoken variant of — same rule as retrieval (given name or nickname, surname one edit, 5+
        letters). The model copies the spelling it read in the transcript: "Tracy Allaway" for
        stated `Tracy Alloway` was discarded as an invented name (#2075, measured)."""
        exact = by_stated.get(said.strip().lower())
        if exact is not None:
            return exact
        toks = said.split()
        if len(toks) < 2:
            return None
        from ..text_normalization import first_names_match

        hits = [
            n
            for n in stated
            if len(n.split()) >= 2
            and len(n.split()[-1]) >= 5
            and (
                n.split()[0].lower() == toks[0].lower() or first_names_match(n.split()[0], toks[0])
            )
            and _one_edit_apart(n.split()[-1], toks[-1])
        ]
        return hits[0] if len(hits) == 1 else None

    out: Dict[str, LLMVoice] = {}
    used: set = set()
    invented: List[str] = []
    refuted: List[str] = []
    # (voice, name) for each refutation, so the refusal can be USED rather than only counted —
    # see the complement pass below.
    refuted_pairs: List[Tuple[str, str]] = []

    for said_voice, verdict in _parse(raw).items():
        voice = _voice_id_in(said_voice, voice_texts)
        if voice is None:
            continue
        canonical: Optional[str] = None
        if verdict.name:
            match = _stated_match(verdict.name)
            if match is None:
                invented.append(verdict.name)
            elif refuted_by_third_person(voice_texts[voice], match):
                refuted.append(f"{voice}={match}")
                refuted_pairs.append((voice, match))
            elif match.lower() in used:  # rule 5 — one person, one voice
                pass
            else:
                used.add(match.lower())
                canonical = match
        if canonical or verdict.role:
            out[voice] = LLMVoice(name=canonical, role=verdict.role)

    # ---- COMPLEMENT PASS: a refutation is EVIDENCE, not just a veto -------------------------
    #
    # The model gets the NAME right and the VOICE wrong more often than it invents people. When it
    # says "SPEAKER_00 is Alison Gopnik" and SPEAKER_00 is the host who said "I am talking with
    # Alison Gopnik", the third-person guard correctly refuses — and then the far more useful fact,
    # that Alison is therefore the OTHER voice, was thrown away with it. On a two-voice interview
    # that is a complete answer, and today it produces an episode with no named speakers at all.
    #
    # STRICTLY TWO REAL VOICES, measured. Against episodes whose voices are already correctly
    # named on the production snapshot, taking "the one voice not refuted for this name":
    #
    #     any voice count   338 fires   90.5% correct
    #     exactly 2 voices  297 fires   98.0% correct   <- this rule
    #
    # The jump is the whole point: with three or more voices "exactly one unrefuted" is weak
    # evidence and misattributes ~1 in 10. Adding a two-token name filter moved 98.0 -> 98.2 for 23
    # lost firings and is not worth it.
    #
    # The comparison that matters is NOT 98% against a perfect answer. This fires only where the
    # model's proposal was already discarded, so the alternative is no name at all.
    #
    # Every existing guard still applies to the complement: the name must be one the metadata
    # stated (it came from `by_stated`), it must not already be used by another voice, and it must
    # not itself be third-person-refuted on the voice it is about to land on.
    #
    # NOTE ON THE VOICE COUNT: `voice_texts` here is already the REAL voices — the caller
    # (`pipeline._resolve_voices_via_llm`) is fed `real_voice_texts`, which `classify_voices` has
    # filtered by `cameo_max_talk_s` (20s). So a 9-second backchannel cluster does NOT make a
    # two-person interview look like a three-way, and this gate must not try to re-derive that
    # filter from text length. An offline replay that passes RAW clusters here will see three voices
    # on Ground Truths and wrongly conclude this gate is broken.
    if len(voice_texts) == 2 and refuted_pairs:
        for bad_voice, name in refuted_pairs:
            others = [v for v in voice_texts if v != bad_voice]
            if len(others) != 1:
                continue
            other = others[0]
            if name.lower() in used:
                continue
            existing = out.get(other)
            if existing is not None and existing.name:
                # THE TWO ANSWERS ARE SWAPPED, which is a different case from "the other voice is
                # spoken for". When this episode states exactly two people, the model put the
                # refuted name here and the OTHER stated name there, and that other name is not
                # refuted where it would move to, then the only arrangement consistent with the
                # audio is the swap. A third-person refutation is a fact about the recording; an
                # unrefuted model answer is an opinion, so the fact wins.
                #
                # Found on Ground Truths: the host says "this is Matthew Cobb's seventh book", the
                # model reads the possessive as a self-introduction, and each wrong answer props up
                # the other — the guest's name on the host's voice and the host's on the guest's.
                other_name = next((n for n in stated if n.lower() != name.lower()), None)
                if other_name is None or len(stated) != 2:
                    continue  # that voice already has a name; do not overwrite a direct answer
                if (
                    existing.name.lower() != other_name.lower()
                    or refuted_by_third_person(voice_texts[other], name)
                    or refuted_by_third_person(voice_texts[bad_voice], other_name)
                ):
                    continue  # not a swap of the two stated names — the direct answer stands
                used.add(name.lower())
                used.add(other_name.lower())
                # THE ROLE TRAVELS WITH THE NAME, not with the voice. What the model got right is
                # the person-to-role mapping ("Topol hosts, Cobb is the guest"); what it got wrong
                # is which voice is which. Leaving each role where it sat therefore keeps the half
                # of the error the swap exists to undo — measured on Ground Truths, it published
                # `Matthew Cobb` as the host and `Eric Topol`, the one name in `known_hosts`, as
                # the guest. Moving each role alongside its name makes both halves agree.
                refuted_voice = out.get(bad_voice)
                out[other] = LLMVoice(name=name, role=refuted_voice.role if refuted_voice else None)
                out[bad_voice] = LLMVoice(name=other_name, role=existing.role)
                logger.info(
                    "speaker resolution: %r was refuted on %s while %r sat on %s — the two stated "
                    "names are swapped, binding each to the voice the audio allows",
                    name,
                    bad_voice,
                    other_name,
                    other,
                )
                continue
            if refuted_by_third_person(voice_texts[other], name):
                continue  # the other voice talks about them too — no evidence either way
            used.add(name.lower())
            out[other] = LLMVoice(name=name, role=existing.role if existing else None)
            logger.info(
                "speaker resolution: %r was refuted on %s, and %s is the only other voice and is "
                "not refuted — binding it there (two-voice complement)",
                name,
                bad_voice,
                other,
            )

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

"""Transcript naming: the model reads the WHOLE diarized transcript and names every voice.

The ladder names voices from per-voice samples (the first 1,200 characters of each voice), a closed
candidate list, and guards that can only refuse a name; none of it can move a name from one voice
to another, and most wrong names are swaps. Given the full transcript, the production model (DGX
Qwen) named voices far better on the labelled naming sets (podcast-scraper-eval-data,
docs/wip/NAMING-RETHINK-2026-10-22.md, "Bet 1"): on 181 held-out two-voice episodes, unnamed
nameable voices 9.4% -> 2.8% and wrong names 6.6% -> 3.8% of named voices, names published only on
voices the model itself calls host or guest.

The prompt and the transcript layout are the ones measured there (``autoresearch/naming_rethink/
qwen_verify.py`` over ``autoresearch/naming_gate1/panel_prep.py`` files); change them only with a
new measurement. When the call fails or the answer cannot be read, the ladder's roster stands.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SOURCE = "transcript_naming"
MIN_WORDS = 40
# The labelled cases capped the metadata at these lengths (scripts/measure/naming_gold_cases.py).
FEED_DESCRIPTION_CHARS = 1500
EPISODE_DESCRIPTION_CHARS = 2500
MAX_OUTPUT_TOKENS = 3000
_NULLS = {"", "null", "none", "unknown", "n/a"}
PARTICIPANT_ROLES = ("host", "guest")

PROMPT = """You identify the speakers of one podcast episode. Below are the feed and episode
metadata, then the FULL diarized transcript: each line is "[mm:ss] SPEAKER_xx: text". Voice ids
carry no names.

For EVERY voice listed at the end, decide:
- name: the real person this voice is, or null. Use only this file: self-introductions, being
  introduced or welcomed by another voice (and who speaks next), being addressed by name, the
  metadata naming hosts and guests, sign-offs. Diarization can merge two people into one voice or
  split one person into two voices; judge a voice by most of its speech. Two voices may be the same
  person. A voice that only TALKS ABOUT a person is not that person.
- role: host | guest | ad | promo | clip | unknown (ad = an advertisement read; promo = a trailer or
  promotion for another show or product; clip = an archival or quoted recording; unknown = mixed or
  unclear).
- confidence: high | medium | low.
- evidence: a short verbatim quote with its [mm:ss] that supports the name, or "" when name is null.
A wrong name is worse than a missing one: when unsure, null.

Return JSON only:
{{"voices": {{"SPEAKER_00": {{"name": "Full Name or null", "role": "...", \
"confidence": "...", "evidence": "..."}}}}}}

{text}

VOICES TO JUDGE: {voices}
"""


def _cap(text: Optional[str], chars: int) -> Optional[str]:
    return text[:chars] if isinstance(text, str) else text


def transcript_block(
    aligned: Sequence[Tuple[Mapping[str, Any], Optional[str]]],
    feed_title: Optional[str],
    feed_description: Optional[str],
    episode_title: Optional[str],
    episode_description: Optional[str],
) -> str:
    """The metadata and the turn-merged transcript, laid out as measured."""
    lines = [
        f"FEED: {feed_title}",
        f"FEED DESCRIPTION: {_cap(feed_description, FEED_DESCRIPTION_CHARS)}",
        f"EPISODE: {episode_title}",
        f"EPISODE DESCRIPTION: {_cap(episode_description, EPISODE_DESCRIPTION_CHARS)}",
        "",
        "TRANSCRIPT (diarized voices; no names attached):",
    ]
    cur: Optional[str] = None
    buf: List[str] = []
    t0 = 0.0
    for segment, speaker in aligned:
        voice = speaker or "UNATTRIBUTED"
        if voice != cur and buf:
            lines.append(f"[{int(t0 // 60):02d}:{int(t0 % 60):02d}] {cur}: {' '.join(buf)}")
            buf = []
        if not buf:
            t0 = float(segment.get("start") or 0.0)
        cur = voice
        buf.append(str(segment.get("text") or "").strip())
    if buf:
        lines.append(f"[{int(t0 // 60):02d}:{int(t0 % 60):02d}] {cur}: {' '.join(buf)}")
    return "\n".join(lines) + "\n"


def voices_to_judge(text: str) -> List[str]:
    """Every voice with at least ``MIN_WORDS`` words in the transcript block."""
    words: Dict[str, int] = {}
    for m in re.finditer(r"^\[\d+:\d+\] (SPEAKER_\d+): (.*)$", text, re.M):
        words[m.group(1)] = words.get(m.group(1), 0) + len(m.group(2).split())
    return sorted(v for v, n in words.items() if n >= MIN_WORDS)


def build_prompt(text: str, voices: Sequence[str]) -> str:
    return PROMPT.format(text=text, voices=", ".join(voices))


def parse_answer(raw: str) -> Optional[Dict[str, Tuple[Optional[str], Optional[str]]]]:
    """``{voice: (name or None, role)}``, or None when the answer is not the expected JSON."""
    t = (raw or "").strip()
    if "</think>" in t:
        t = t.split("</think>", 1)[1]
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        obj = json.loads(t[a : b + 1])
    except json.JSONDecodeError:
        return None
    voices = obj.get("voices") if isinstance(obj, dict) else None
    if not isinstance(voices, dict):
        return None
    out: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for voice, v in voices.items():
        if not isinstance(v, dict):
            continue
        name = str(v.get("name") or "").strip()
        role = str(v.get("role") or "").strip().lower() or None
        out[str(voice)] = (None if name.lower() in _NULLS else name, role)
    return out


def apply_answer(
    roster: Any, answer: Mapping[str, Tuple[Optional[str], Optional[str]]], trace: Any = None
) -> Any:
    """The roster named by the answer: a voice the model calls host or guest takes its name and
    role; every other voice is unnamed. As measured: the model's answer replaces the ladder's names.
    """
    from ....speaker_detectors.hosts import is_publishable_speaker_name

    by_voice = {}
    for voice, role in roster.by_voice.items():
        name, said_role = answer.get(voice, (None, None))
        if name and said_role in PARTICIPANT_ROLES and is_publishable_speaker_name(name):
            new = dataclasses.replace(
                role, name=name, role=said_role, named=True, source=SOURCE, forced=False
            )
            decision = "named" if not role.named or role.name != name else "kept"
        else:
            new = dataclasses.replace(
                role, name=voice, role="unknown", named=False, source=SOURCE, forced=False
            )
            decision = "removed" if role.named else "unnamed"
        by_voice[voice] = new
        if trace is not None and (decision != "unnamed"):
            trace.voice(
                voice,
                SOURCE,
                decision,
                name=name if new.named else role.name if role.named else None,
                said_role=said_role,
            )
    return dataclasses.replace(roster, by_voice=by_voice)


def name_from_transcript(
    roster: Any,
    aligned: Sequence[Tuple[Mapping[str, Any], Optional[str]]],
    complete: Any,
    *,
    feed_title: Optional[str],
    feed_description: Optional[str],
    episode_title: Optional[str],
    episode_description: Optional[str],
    trace: Any = None,
) -> Any:
    """Ask the model; return the roster it names, or the ladder's roster when the call fails."""
    text = transcript_block(
        aligned, feed_title, feed_description, episode_title, episode_description
    )
    voices = voices_to_judge(text)
    if not voices:
        return roster
    prompt = build_prompt(text, voices)
    report: Dict[str, Any] = {"voices_asked": voices, "prompt_chars": len(prompt)}
    try:
        raw = complete(prompt)
    except Exception as exc:  # noqa: BLE001 — the ladder's roster stands
        report["error"] = f"{type(exc).__name__}: {exc}"
        if trace is not None:
            trace.input(SOURCE, report)
        logger.warning("transcript naming: call failed (%s); the ladder's names stand", exc)
        return roster
    report["raw"] = (raw or "")[:16000]
    answer = parse_answer(raw)
    if trace is not None:
        trace.input(SOURCE, report)
    if not answer:
        logger.warning("transcript naming: unreadable answer; the ladder's names stand")
        return roster
    return apply_answer(roster, answer, trace)

"""Transcript naming: the model reads the WHOLE diarized transcript and names every voice.

The ladder names voices from per-voice samples (the first 1,200 characters of each voice), a closed
candidate list, and guards that can only refuse a name; none of it can move a name from one voice
to another, and most wrong names are swaps. Given the full transcript, the production model (DGX
Qwen) named voices far better on the labelled naming sets (podcast-scraper-eval-data,
docs/wip/NAMING-RETHINK-2026-10-22.md, "Bet 1"): on 181 held-out two-voice episodes, unnamed
nameable voices 9.4% -> 2.8% and wrong names 6.6% -> 3.8% of named voices, names published only on
voices the model itself calls host or guest.

Three modes (``transcript_naming``): ``off``; ``shadow`` — the ladder publishes and the model's
answer is written beside it in the speaker-diagnostics sidecar under :data:`SIDECAR_KEY`, so the two
namers can be compared voice by voice; ``publish`` — the model's answer names the roster. Shadow
changes no published artifact other than that sidecar key.

The prompt and the transcript layout are the ones measured there (``autoresearch/naming_rethink/
qwen_verify.py`` over ``autoresearch/naming_gate1/panel_prep.py`` files), now the prompt-store
templates ``shared/speaker_naming/*_v1``; change them only with a new measurement. A
transcript too long for the provider's window loses turns from the MIDDLE, never the opening
introductions or the closing sign-offs the prompt asks about, and the sidecar records how many.
When the call fails or the answer cannot be read, the ladder's roster stands.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SOURCE = "transcript_naming"
SIDECAR_KEY = "transcript_naming"
SIDECAR_VERSION = 1
MODES = ("off", "shadow", "publish")
COST_STAGE = "speaker_naming"
PROMPT_VERSION = "v1"
STANDALONE_PROMPT = f"shared/speaker_naming/standalone_{PROMPT_VERSION}"
INSTRUCTIONS_PROMPT = f"shared/speaker_naming/instructions_{PROMPT_VERSION}"
MIN_WORDS = 40
# The labelled cases capped the metadata at these lengths (scripts/measure/naming_gold_cases.py).
FEED_DESCRIPTION_CHARS = 1500
EPISODE_DESCRIPTION_CHARS = 2500
MAX_OUTPUT_TOKENS = 3000
# Headroom between the counted prompt and the window, for chat-template tokens the count misses.
WINDOW_MARGIN_TOKENS = 256
RAW_CHARS = 16000
_CLIP_ROUNDS = 6
_NULLS = {"", "null", "none", "unknown", "n/a"}
PARTICIPANT_ROLES = ("host", "guest")
_TURN = re.compile(r"^\[\d+:\d+\] (SPEAKER_\d+): (.*)$", re.M)


def mode_of(cfg: Any) -> str:
    """The configured mode, ``off`` for anything unrecognised."""
    value = str(getattr(cfg, "transcript_naming", "off") or "off").strip().lower()
    return value if value in MODES else "off"


def _cap(text: Optional[str], chars: int) -> Optional[str]:
    return text[:chars] if isinstance(text, str) else text


def header_lines(
    feed_title: Optional[str],
    feed_description: Optional[str],
    episode_title: Optional[str],
    episode_description: Optional[str],
) -> List[str]:
    """The metadata lines above the transcript, capped as the labelled cases were."""
    return [
        f"FEED: {feed_title}",
        f"FEED DESCRIPTION: {_cap(feed_description, FEED_DESCRIPTION_CHARS)}",
        f"EPISODE: {episode_title}",
        f"EPISODE DESCRIPTION: {_cap(episode_description, EPISODE_DESCRIPTION_CHARS)}",
        "",
        "TRANSCRIPT (diarized voices; no names attached):",
    ]


def turn_lines(aligned: Sequence[Tuple[Mapping[str, Any], Optional[str]]]) -> List[str]:
    """Consecutive segments of one voice merged into one ``[mm:ss] VOICE: text`` turn."""
    lines: List[str] = []
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
    return lines


def voices_to_judge(text: str) -> List[str]:
    """Every voice with at least ``MIN_WORDS`` words in the transcript block."""
    words: Dict[str, int] = {}
    for m in _TURN.finditer(text):
        words[m.group(1)] = words.get(m.group(1), 0) + len(m.group(2).split())
    return sorted(v for v, n in words.items() if n >= MIN_WORDS)


def instructions() -> str:
    """The per-voice instructions, shared by the standalone call and the bundled fields."""
    from ....prompts.store import render_prompt

    return render_prompt(INSTRUCTIONS_PROMPT)


def build_prompt(text: str, voices: Sequence[str]) -> str:
    """The standalone prompt for one transcript block and the voices to judge."""
    from ....prompts.store import render_prompt

    rendered = render_prompt(
        STANDALONE_PROMPT, instructions=instructions(), text=text, voices=", ".join(voices)
    )
    return rendered + "\n"


def _count(text: str, count_tokens: Optional[Callable[[str], Optional[int]]]) -> int:
    from ...common.token_budget import FALLBACK_EFFECTIVE_CHARS_PER_TOKEN

    n = count_tokens(text) if count_tokens is not None else None
    return int(n) if n is not None else int(len(text) / FALLBACK_EFFECTIVE_CHARS_PER_TOKEN) + 1


def fit_prompt(
    head: Sequence[str],
    turns: Sequence[str],
    window_tokens: Optional[int],
    count_tokens: Optional[Callable[[str], Optional[int]]] = None,
) -> Tuple[Optional[str], List[str], Dict[str, Any]]:
    """``(prompt, voices, clip)``: the measured prompt, with middle turns dropped until it fits.

    ``prompt`` is None when no voice qualifies or nothing fits. ``window_tokens`` None means the
    window is unknown and the prompt is sent whole (a provider's 400 then says so).
    """
    clip: Dict[str, Any] = {"window_tokens": window_tokens, "turns": len(turns), "turns_dropped": 0}
    budget = (
        None if not window_tokens else int(window_tokens) - MAX_OUTPUT_TOKENS - WINDOW_MARGIN_TOKENS
    )
    keep = len(turns)
    for _ in range(_CLIP_ROUNDS + 1):
        if keep == len(turns):
            body = list(turns)
        else:
            front, back = (keep + 1) // 2, keep // 2
            dropped = len(turns) - keep
            body = (
                list(turns[:front])
                + [f"[... {dropped} turns omitted for length ...]"]
                + list(turns[len(turns) - back :] if back else [])
            )
        text = "\n".join(list(head) + body) + "\n"
        voices = voices_to_judge(text)
        if not voices:
            return None, [], clip
        prompt = build_prompt(text, voices)
        if budget is None:
            return prompt, voices, clip
        used = _count(prompt, count_tokens)
        clip["prompt_tokens"] = used
        if used <= budget:
            clip["turns_dropped"] = len(turns) - keep
            return prompt, voices, clip
        keep = int(keep * budget / used * 0.95)
        if keep < 2:
            break
    clip["turns_dropped"] = len(turns) - max(keep, 0)
    clip["error"] = "transcript does not fit the window"
    return None, [], clip


def parse_answer_full(raw: str) -> Optional[Dict[str, Dict[str, Optional[str]]]]:
    """``{voice: {name, role, confidence, evidence}}``, or None when not the expected JSON."""
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
    return parse_voices(voices)


def parse_voices(voices: Any) -> Optional[Dict[str, Dict[str, Optional[str]]]]:
    """The ``voices`` object of an answer (standalone, or a bundled call's speakers field)."""
    if not isinstance(voices, dict):
        return None
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for voice, v in voices.items():
        if not isinstance(v, dict):
            continue
        name = str(v.get("name") or "").strip()
        role = str(v.get("role") or "").strip().lower() or None
        conf = str(v.get("confidence") or "").strip().lower() or None
        evidence = str(v.get("evidence") or "").strip() or None
        out[str(voice)] = {
            "name": None if name.lower() in _NULLS else name,
            "role": role,
            "confidence": conf,
            "evidence": evidence,
        }
    return out


def keyed_by_voices_asked(
    answer: Optional[Dict[str, Dict[str, Optional[str]]]], voices: Sequence[str]
) -> Optional[Dict[str, Dict[str, Optional[str]]]]:
    """The answer keyed by the voice ids asked: the model writes ``SPEAKER_1`` for
    ``SPEAKER_01`` (vLLM on the DGX, 5 of 20 episodes), and an exact lookup left that voice
    unnamed. An exact key wins; a key no asked voice matches keeps its own spelling."""
    if not answer:
        return answer
    from ....speaker_detectors.resolution import _voice_id_in

    asked = dict.fromkeys(voices, "")
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for said in sorted(answer, key=lambda k: k not in asked):
        voice = _voice_id_in(said, asked) or said
        out.setdefault(voice, answer[said])
    return out


@dataclasses.dataclass
class NamingAnswer:
    """One ask: what was sent, what came back, and the parsed answer (None on failure)."""

    call: str
    voices_asked: List[str]
    answer: Optional[Dict[str, Dict[str, Optional[str]]]]
    report: Dict[str, Any]

    @property
    def ok(self) -> bool:
        return bool(self.answer)

    def pairs(self) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
        """``{voice: (name, role)}``, the shape ``apply_answer`` takes."""
        return {v: (a.get("name"), a.get("role")) for v, a in (self.answer or {}).items()}


def ask(
    aligned: Sequence[Tuple[Mapping[str, Any], Optional[str]]],
    complete: Callable[[str], str],
    *,
    feed_title: Optional[str],
    feed_description: Optional[str],
    episode_title: Optional[str],
    episode_description: Optional[str],
    window_tokens: Optional[int] = None,
    count_tokens: Optional[Callable[[str], Optional[int]]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> NamingAnswer:
    """Ask the model to name every voice (``ok`` False when skipped, failed or unreadable)."""
    head = header_lines(feed_title, feed_description, episode_title, episode_description)
    prompt, voices, clip = fit_prompt(head, turn_lines(aligned), window_tokens, count_tokens)
    report: Dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "provider": provider,
        "model": model,
        "clip": clip,
    }
    if prompt is None:
        if "error" not in clip:
            report["skipped"] = f"no voice with {MIN_WORDS} words"
            return NamingAnswer("standalone", [], None, report)
        report["error"] = clip["error"]
        logger.warning("transcript naming: %s; the ladder's names stand", clip["error"])
        return NamingAnswer("standalone", [], None, report)
    report["prompt_chars"] = len(prompt)
    try:
        raw = complete(prompt)
    except Exception as exc:  # noqa: BLE001 — the ladder's roster stands
        report["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("transcript naming: call failed (%s); the ladder's names stand", exc)
        return NamingAnswer("standalone", voices, None, report)
    report["raw"] = (raw or "")[:RAW_CHARS]
    answer = keyed_by_voices_asked(parse_answer_full(raw), voices)
    if not answer:
        report["error"] = "unreadable answer"
        logger.warning("transcript naming: unreadable answer; the ladder's names stand")
    return NamingAnswer("standalone", voices, answer or None, report)


def apply_answer(
    roster: Any,
    answer: Mapping[str, Tuple[Optional[str], Optional[str]]],
    trace: Any = None,
    *,
    language: Optional[str],
) -> Any:
    """The roster named by the answer: a voice the model calls host or guest takes its name and
    role; every other voice is unnamed. As measured: the model's answer replaces the ladder's names.

    A name the model gives the voice the ladder already gave keeps the ladder's ``source``, so the
    evidence it carried (a self-introduction the record and the graph rely on) is not lost.
    ``language`` is the feed's, so the publish gate refuses that language's role words too.
    """
    from ....speaker_detectors.hosts import is_publishable_speaker_name

    by_voice = {}
    for voice, role in roster.by_voice.items():
        name, said_role = answer.get(voice, (None, None))
        if (
            name
            and said_role in PARTICIPANT_ROLES
            and is_publishable_speaker_name(name, language=language)
        ):
            same = bool(role.named) and str(role.name).lower() == name.lower()
            new = dataclasses.replace(
                role,
                name=role.name if same else name,
                role=said_role,
                named=True,
                source=role.source if same else SOURCE,
                forced=False,
            )
            decision = "kept" if same else "named"
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


def role_view(role: Any) -> Dict[str, Any]:
    """A voice's published name (None when unnamed), role and source, for the sidecars."""
    named = bool(getattr(role, "named", False))
    return {
        "name": getattr(role, "name", None) if named else None,
        "role": getattr(role, "role", None),
        "source": getattr(role, "source", None),
    }


def outcome(ladder: Optional[str], model: Optional[str]) -> str:
    """agree | adds | removes | changes — the model's roster against the ladder's, one voice."""
    if (ladder or "").lower() == (model or "").lower():
        return "agree"
    if not ladder:
        return "adds"
    if not model:
        return "removes"
    return "changes"


def sidecar(
    mode: str, asked: NamingAnswer, ladder_roster: Any, model_roster: Optional[Any]
) -> Dict[str, Any]:
    """The sidecar entry: per voice, the ladder's final name against the model's.

    ``ladder_roster`` is what the ladder publishes; ``model_roster`` is the roster the model's
    answer gives after the same talk floor and renames (None when the call failed).
    """
    voices: Dict[str, Any] = {}
    counts = {"agree": 0, "adds": 0, "removes": 0, "changes": 0}
    for voice, role in ladder_roster.by_voice.items():
        entry: Dict[str, Any] = {
            "ladder": role_view(role),
            "model": (asked.answer or {}).get(voice),
        }
        if model_roster is not None and voice in model_roster.by_voice:
            result = role_view(model_roster.by_voice[voice])
            entry["model_result"] = result
            entry["outcome"] = outcome(entry["ladder"]["name"], result["name"])
            counts[entry["outcome"]] += 1
        voices[voice] = entry
    out: Dict[str, Any] = {
        "version": SIDECAR_VERSION,
        "mode": mode,
        "call": asked.call,
        "ok": asked.ok,
        "voices_asked": asked.voices_asked,
        "report": asked.report,
        "voices": voices,
    }
    if model_roster is not None:
        out["outcomes"] = counts
    return out


# --- the same answer inside a bundled call (bundled, mega_bundled, extraction_bundled) ---------

BUNDLED_PROMPT = f"shared/speaker_naming/bundled_field_{PROMPT_VERSION}"
BUNDLED_MODES = ("bundled", "mega_bundled", "extraction_bundled")
#: params key the metadata stage sets for a bundled call: the transcript's speaker labels.
PARAMS_LABELS = "speaker_naming_labels"


def bundled_field(labels: Sequence[str]) -> str:
    """The ``speakers`` field instructions a bundled call appends to its user prompt.

    The bundled transcript carries the ladder's labels (``Name: text``), not voice ids, so the
    model is asked who each LABEL really is (operator's option A, 2026-10-23); the answer maps
    back to voices through the speaker diagnostics.
    """
    from ....prompts.store import render_prompt

    return render_prompt(BUNDLED_PROMPT, labels=", ".join(labels))


def with_bundled_field(user_prompt: str, params: Optional[Mapping[str, Any]]) -> str:
    """``user_prompt`` with the ``speakers`` field appended when the caller asked for it."""
    labels = (params or {}).get(PARAMS_LABELS)
    if not labels:
        return user_prompt
    return f"{user_prompt}\n\n{bundled_field(list(labels))}"


def labels_from_diagnostics(diagnostics: Mapping[str, Any]) -> Dict[str, List[str]]:
    """``{transcript label: [voice, ...]}`` from a speaker-diagnostics sidecar.

    A segment's label is the voice's resolved name, or its raw id when unnamed
    (``SpeakerRoster.label_for``), which is the sidecar's ``resolved_name``.
    """
    out: Dict[str, List[str]] = {}
    for v in diagnostics.get("voices") or []:
        voice = v.get("voice")
        label = v.get("resolved_name") or voice
        if voice and label:
            out.setdefault(str(label), []).append(str(voice))
    return out


def answer_by_voice(
    speakers: Any, label_map: Mapping[str, Sequence[str]]
) -> Optional[Dict[str, Dict[str, Optional[str]]]]:
    """The bundled ``speakers`` object (keyed by label) as a per-voice answer.

    A label two voices share is left out: one answer cannot say which of them is which.
    """
    parsed = parse_voices(speakers)
    if parsed is None:
        return None
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for label, answer in parsed.items():
        voices = label_map.get(label) or []
        if len(voices) == 1:
            out[voices[0]] = answer
    return out


def bundled_sidecar(
    mode: str,
    call: str,
    speakers: Any,
    diagnostics: Mapping[str, Any],
    *,
    floor: float,
    renames: Optional[Mapping[str, str]],
    language: Optional[str],
) -> Dict[str, Any]:
    """The sidecar entry for a bundled call's answer, in the standalone entry's shape.

    The ladder side is what the sidecar already publishes; the model side applies the same rule
    as ``apply_answer`` (host or guest, publishable in the feed's language), the same talk floor
    (on the sidecar's talk seconds) and the same operator renames.
    """
    from ....speaker_detectors.hosts import is_publishable_speaker_name

    label_map = labels_from_diagnostics(diagnostics)
    answer = answer_by_voice(speakers, label_map)
    report: Dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "labels_asked": sorted(label_map),
        "labels_shared_by_voices": sorted(k for k, v in label_map.items() if len(v) > 1),
    }
    if answer is None:
        report["error"] = "no readable speakers field"
    voices_in = list(diagnostics.get("voices") or [])
    total = sum(float(v.get("talk_time_s") or 0.0) for v in voices_in) or 1.0
    voices: Dict[str, Any] = {}
    counts = {"agree": 0, "adds": 0, "removes": 0, "changes": 0}
    for v in voices_in:
        voice = str(v.get("voice"))
        ladder = {
            "name": v.get("resolved_name") if v.get("named") else None,
            "role": v.get("role"),
            "source": v.get("source"),
        }
        entry: Dict[str, Any] = {"ladder": ladder, "model": (answer or {}).get(voice)}
        if answer is not None:
            a = answer.get(voice) or {}
            name, role = a.get("name"), a.get("role")
            keep = bool(name) and role in PARTICIPANT_ROLES
            keep = keep and is_publishable_speaker_name(name, language=language)
            share = float(v.get("talk_time_s") or 0.0) / total
            if keep and floor and share < floor:
                keep = False
            result = (renames or {}).get(str(name), name) if keep else None
            entry["model_result"] = {"name": result, "role": role if keep else "unknown"}
            entry["outcome"] = outcome(ladder["name"], result)
            counts[entry["outcome"]] += 1
        voices[voice] = entry
    out: Dict[str, Any] = {
        "version": SIDECAR_VERSION,
        "mode": mode,
        "call": call,
        "ok": answer is not None,
        "voices_asked": sorted(v for vs in label_map.values() for v in vs if len(vs) == 1),
        "report": report,
        "voices": voices,
    }
    if answer is not None:
        out["outcomes"] = counts
    return out


def agree(asks: Sequence[NamingAnswer]) -> NamingAnswer:
    """One answer from ``speaker_naming_passes`` asks: a voice keeps its name only when every pass
    that answered gave it the same name and role (case-insensitive); otherwise it is null.

    One ask is returned as it is. A pass that failed or could not be read is left out, and the
    report says how many answered; when none did, the first ask (with its error) stands.
    """
    if len(asks) == 1:
        return asks[0]
    answered = [a for a in asks if a.ok]
    first = answered[0] if answered else asks[0]
    report = dict(first.report)
    report["passes"] = [a.report for a in asks if a is not first]
    if not answered:
        return NamingAnswer(first.call, first.voices_asked, None, report)
    agreed_answer: Dict[str, Dict[str, Optional[str]]] = {}
    dropped: List[str] = []
    for voice, a in (first.answer or {}).items():
        names = {str((x.answer or {}).get(voice, {}).get("name") or "").lower() for x in answered}
        roles = {(x.answer or {}).get(voice, {}).get("role") for x in answered}
        if len(names) == 1 and len(roles) == 1:
            agreed_answer[voice] = dict(a)
        else:
            role = a.get("role") if len(roles) == 1 else "unknown"
            agreed_answer[voice] = {
                "name": None,
                "role": role,
                "confidence": "low",
                "evidence": None,
            }
            if a.get("name"):
                dropped.append(voice)
    report["passes_agreed"] = {"asked": len(asks), "answered": len(answered), "dropped": dropped}
    return NamingAnswer(first.call, first.voices_asked, agreed_answer, report)

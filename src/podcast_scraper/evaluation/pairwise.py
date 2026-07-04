"""Pairwise judging — candidate vs silver, anchored preference + magnitude.

Why pairwise: scalar 0-1 G-Eval on smoke sets tends to saturate — the W27
sweep produced judge_mean in [0.925, 0.975] across every candidate, with
all separation coming from ROUGE-L. Pairwise judging asks "is A better
than B?" against the silver directly, which is far more discriminative
because judges can't hide in the top of a 5-point scale — a preference
is binary (with magnitude signalling confidence).

Anchor the magnitude 1-5 to the silver:

    1 = barely better than the other
    2 = slightly better
    3 = noticeably better
    4 = clearly better
    5 = decisively better

Anti-position-bias: for every episode the caller randomizes which slot
(A vs B) holds the candidate. The judge's ``preference`` reply names the
slot it prefers; the caller converts that back to a candidate-vs-silver
verdict using the position record. Without this, a judge that happens to
have a first-slot bias would systematically overrate whichever party
lands in slot A.

This module is transport-agnostic: it produces the user message and
parses the reply, but doesn't POST anything. Callers (e.g.
:mod:`autoresearch_track_a`) route the message through the existing
scalar transports (OllamaChatJudge / VllmChatJudge / call_openai_judge /
call_anthropic_judge) — same retry, same error-wrapping.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

# The candidate landed in slot A or slot B for this specific call.
# Tracked per-episode so the reply's ``preference`` can be resolved back
# to candidate-vs-silver rather than A-vs-B.
CandidateSlot = Literal["A", "B"]

# Reply preference the judge names — the SLOT it prefers, not the party.
SlotPreference = Literal["A", "B", "tie"]

# What the pairwise judgement resolves to after position adjustment.
PartyPreference = Literal["candidate", "silver", "tie"]


@dataclass(frozen=True)
class PairwiseVerdict:
    """One judge's pairwise call, resolved to candidate-vs-silver space.

    ``preference`` is the party the judge preferred AFTER position
    adjustment — no longer the slot. ``magnitude`` is 1 (barely better)
    to 5 (decisively better); for ``tie`` it's 0.
    """

    preference: PartyPreference
    magnitude: int
    rationale: str


PAIRWISE_RUBRIC = """\
Score how well each summary reflects the transcript on:
1. Topic coverage — main themes appear; nothing central missing.
2. Factual accuracy — no contradictions or invented facts vs transcript.
3. Conciseness — reasonable length, little fluff or repetition.

A perfect summary hits all three; a failing summary hallucinates, omits
the core story, or is unusably vague."""


def build_pairwise_user_message(
    *,
    rubric: str,
    transcript: str,
    slot_a_summary: str,
    slot_b_summary: str,
    max_transcript_chars: int = 28_000,
) -> str:
    """Build the pairwise judging prompt.

    The judge sees SLOTS ``A`` and ``B`` — no hint about which is the
    candidate. The caller records the slot mapping externally so it can
    resolve the ``preference`` reply back to candidate-vs-silver.
    """
    t = transcript if len(transcript) <= max_transcript_chars else transcript[:max_transcript_chars]
    return (
        "You are comparing two podcast episode summaries against the transcript.\n\n"
        "### Rubric\n"
        f"{rubric}\n\n"
        "### Transcript (may be truncated)\n"
        f"{t}\n\n"
        "### Summary A\n"
        f"{slot_a_summary}\n\n"
        "### Summary B\n"
        f"{slot_b_summary}\n\n"
        "Which summary is closer to the ideal by the rubric? Reply with a single JSON "
        "object only, no markdown:\n"
        '{"preference": "A" | "B" | "tie", '
        '"magnitude": <int 1-5, or 0 for tie>, '
        '"rationale": "<one short sentence>"}\n\n'
        "Magnitude anchor (only when preference is A or B):\n"
        "  1 = barely better  |  2 = slightly better  |  3 = noticeably better\n"
        "  4 = clearly better |  5 = decisively better"
    )


_JSON_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\s*|\s*```$")
_THINK_TAG_RE = re.compile(r"</think\s*>", re.IGNORECASE)


def _extract_json_object(text: str) -> str:
    """Return the first balanced ``{...}`` JSON object substring in ``text``.

    Handles reasoning-model outputs that emit a chain-of-thought free-text
    block before the JSON answer (e.g. nemotron_h, gpt-oss, qwen3.x
    thinking). Steps:

    1. If the text contains ``</think>`` (case-insensitive), everything
       before it is CoT — strip it.
    2. Strip surrounding markdown code fences ``` ... ```.
    3. Scan for the first ``{`` and walk forward tracking brace depth
       (respecting string quoting) to find the matching closing ``}``.

    Raises ``ValueError`` if no balanced object is found.
    """
    m = _THINK_TAG_RE.search(text)
    if m:
        text = text[m.end() :]
    cleaned = _JSON_FENCE_RE.sub("", text.strip())
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("no JSON object found in judge reply")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1]
    raise ValueError("unterminated JSON object in judge reply")


def parse_pairwise_verdict(
    text: str,
    *,
    candidate_slot: CandidateSlot,
) -> PairwiseVerdict:
    """Parse the judge's JSON reply and resolve the slot preference back to
    candidate-vs-silver using ``candidate_slot``.

    Position adjustment logic:
        candidate_slot == "A" AND preference == "A" → "candidate"
        candidate_slot == "A" AND preference == "B" → "silver"
        candidate_slot == "B" AND preference == "A" → "silver"
        candidate_slot == "B" AND preference == "B" → "candidate"
        preference == "tie"                         → "tie" (magnitude=0)

    Raises ``ValueError`` on malformed replies — same discipline as the
    scalar parser; a broken judge reply is a hard error, not a silent
    default.
    """
    data = json.loads(_extract_json_object(text))
    if not isinstance(data, dict):
        raise ValueError("Pairwise verdict JSON must be an object")

    raw_pref = data.get("preference")
    if raw_pref not in ("A", "B", "tie"):
        raise ValueError(f"pairwise preference must be 'A' | 'B' | 'tie', got {raw_pref!r}")
    slot_pref: SlotPreference = raw_pref

    raw_mag = data.get("magnitude")
    if raw_mag is None:
        raise ValueError("pairwise verdict missing 'magnitude'")
    try:
        magnitude = int(raw_mag)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"magnitude must be an int, got {raw_mag!r}") from exc

    if slot_pref == "tie":
        # Ties reset magnitude to 0 regardless of what the judge sent — a
        # tied preference with magnitude=5 is a contradiction; treat as 0.
        magnitude = 0
    elif not (1 <= magnitude <= 5):
        raise ValueError(f"magnitude for A/B preference must be 1-5, got {magnitude}")

    rationale = str(data.get("rationale", "")).strip()

    party: PartyPreference
    if slot_pref == "tie":
        party = "tie"
    elif slot_pref == candidate_slot:
        party = "candidate"
    else:
        party = "silver"

    return PairwiseVerdict(preference=party, magnitude=magnitude, rationale=rationale)


def pairwise_verdict_to_score(verdict: PairwiseVerdict) -> float:
    """Convert a resolved verdict to a scalar in [0, 1] for aggregation.

    Encoding:
        candidate + magnitude → 0.5 + (magnitude / 10)  ∈ [0.6, 1.0]
        silver    + magnitude → 0.5 − (magnitude / 10)  ∈ [0.0, 0.4]
        tie                    → 0.5

    Magnitude affects spread but stays bounded: even a decisive win
    (magnitude=5) doesn't drown out the pairwise signal itself, and a
    barely-better preference (magnitude=1) still moves the score off the
    tie point. Aggregated across N episodes this gives a
    candidate-quality proxy in [0, 1] that plays well with the existing
    ``rouge_weight * rougeL + (1 - rouge_weight) * X`` scoring formula.
    """
    if verdict.preference == "tie":
        return 0.5
    direction = 1.0 if verdict.preference == "candidate" else -1.0
    return 0.5 + direction * (verdict.magnitude / 10.0)


@dataclass(frozen=True)
class PairwiseOutcome:
    """Per-episode pairwise result from BOTH judges."""

    judge_a: PairwiseVerdict
    judge_b: PairwiseVerdict
    contested: bool


def _party_of(verdict: PairwiseVerdict) -> Optional[PartyPreference]:
    """Return the non-tie party the verdict picked, else None for tie."""
    return None if verdict.preference == "tie" else verdict.preference


def is_contested(judge_a: PairwiseVerdict, judge_b: PairwiseVerdict) -> bool:
    """Pairwise contested = both judges named a preference AND they picked
    OPPOSITE parties. Ties don't contest (one side just isn't strong).

    Rationale: a scalar contest fires on |a - b| > 0.25, which is a
    magnitude disagreement. Pairwise's version has to be a DIRECTION
    disagreement — magnitude disagreements are expected and don't
    invalidate the preference.
    """
    party_a = _party_of(judge_a)
    party_b = _party_of(judge_b)
    if party_a is None or party_b is None:
        return False
    return party_a != party_b


def _pick_slot(episode_id: str) -> CandidateSlot:
    """Deterministically pick the candidate's slot for an episode.

    Uses a hash on the episode id so a rerun of the same episode gets the
    same slot assignment — makes debugging + replay work — but the
    distribution across episodes is effectively 50/50 for a real dataset.
    Uses builtin ``hash`` seeded via PYTHONHASHSEED at process start;
    across runs the assignment may differ, which is exactly what we want
    for position bias — different runs, different randomization.
    """
    return "A" if hash(episode_id) & 1 == 0 else "B"


def prepare_slots(
    *,
    episode_id: str,
    candidate_summary: str,
    silver_summary: str,
) -> tuple[CandidateSlot, str, str]:
    """Assign the candidate to slot A or B; return (slot, slot_a, slot_b)."""
    slot = _pick_slot(episode_id)
    if slot == "A":
        return "A", candidate_summary, silver_summary
    return "B", silver_summary, candidate_summary


def summarize_pairwise_run(
    verdicts: list[PairwiseVerdict],
) -> dict[str, Any]:
    """Aggregate a list of per-episode verdicts (one judge, all episodes).

    Returns a dict with:
        mean_score:   scalar in [0, 1], mean of pairwise_verdict_to_score
        win_rate:     candidate_wins / non_tie_decisions (or None if all ties)
        tie_rate:     ties / total episodes
        decisive_rate: (magnitude >= 4) / total episodes
        n:            number of episodes
    """
    n = len(verdicts)
    if n == 0:
        return {
            "mean_score": None,
            "win_rate": None,
            "tie_rate": None,
            "decisive_rate": None,
            "n": 0,
        }
    scores = [pairwise_verdict_to_score(v) for v in verdicts]
    non_tie = [v for v in verdicts if v.preference != "tie"]
    wins = sum(1 for v in non_tie if v.preference == "candidate")
    return {
        "mean_score": sum(scores) / n,
        "win_rate": (wins / len(non_tie)) if non_tie else None,
        "tie_rate": (n - len(non_tie)) / n,
        "decisive_rate": sum(1 for v in verdicts if v.magnitude >= 4) / n,
        "n": n,
    }

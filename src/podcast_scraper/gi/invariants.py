"""Postconditions for a GI artifact: the stage checks its own output before claiming success.

Every bug that reached the corpus in this arc had the same shape — a stage produced empty or
impossible output and *reported success*. None of them raised. They were found by a human squinting
at a table days later:

* the evidence-align never fired, and a 10-episode run emitted **513 insights and zero quotes**
* GI had no detected-person list, so ``build_named_turns`` never ran and **every quote** shipped
  with ``speaker_id: None``
* the LLM speaker detector returned Elon Musk as a speaker on an episode he is merely sued in, and
  his name was painted onto the guest's voice — ``succeeded: True``
* the extractive-QA scorer softmaxed within each window, so **every** window scored ~1.000

Fixtures catch the traps someone was imaginative enough to invent. These invariants catch the class:
output that cannot be true regardless of which stage broke. A violation means a wire is disconnected
upstream — it does not tell you which one, but it tells you not to trust the artifact.

Checks are cheap, pure, and require no model. They are deliberately *structural* — they assert
things that are impossible in a working pipeline, never things that are merely unlikely, so a
violation is always a bug and never a judgement call. That is what makes them safe to run always.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# A quote's recorded char_start should land on its text. Allow slack for whitespace normalisation
# between the transcript GI saw and the one handed to the checker.
_OFFSET_SLACK = 64


def _nodes_of(artifact: Dict[str, Any], node_type: str) -> List[Dict[str, Any]]:
    return [n for n in artifact.get("nodes") or [] if n.get("type") == node_type]


def _supported_by(artifact: Dict[str, Any]) -> Dict[str, List[str]]:
    """``{insight_id: [quote_id, ...]}`` from SUPPORTED_BY edges."""
    out: Dict[str, List[str]] = {}
    for e in artifact.get("edges") or []:
        if e.get("type") == "SUPPORTED_BY":
            out.setdefault(str(e.get("from")), []).append(str(e.get("to")))
    return out


def check_artifact_invariants(
    artifact: Dict[str, Any],
    transcript_text: Optional[str] = None,
    turns: Optional[Sequence[Tuple[int, str]]] = None,
) -> List[str]:
    """Return the violated postconditions of *artifact*. Empty list means structurally sound.

    Args:
        artifact: The GI artifact (``nodes`` / ``edges``).
        transcript_text: The transcript GI was run on. When given, quotes are checked to actually
            occur in it — a quote that is not in the transcript is fabricated, not grounded.
        turns: ``[(char_offset, speaker_label)]`` from the transcript. When given, every attributed
            speaker must be one of these labels: a speaker who never has a turn was read off the
            episode *description*, which is exactly how a lawsuit defendant became a podcast guest.
    """
    violations: List[str] = []

    insights = _nodes_of(artifact, "Insight")
    quotes = {str(q.get("id")): q for q in _nodes_of(artifact, "Quote")}
    links = _supported_by(artifact)

    if not insights:
        return violations  # nothing extracted; other stages report that, it is not a wiring fault

    # 1. Insights with no grounding at all. A GI run whose grounder is disconnected still happily
    #    emits insights — this is the 513-insights-zero-quotes signature.
    if not quotes:
        violations.append(
            f"grounding produced NOTHING: {len(insights)} insights, 0 quotes. The grounder is "
            "disconnected (check the evidence-provider align: model_copy skips validators)"
        )
        return violations

    grounded = [i for i in insights if links.get(str(i.get("id")))]

    # 2. Every quote must actually occur in the transcript. A quote that does not is fabricated,
    #    and a char_start that does not point at its own text cannot attribute a speaker correctly.
    if transcript_text:
        norm = " ".join(transcript_text.split())
        missing = 0
        misplaced = 0
        for q in quotes.values():
            p = q.get("properties") or {}
            text = str(p.get("text") or "").strip()
            if not text:
                continue
            if text in transcript_text:
                cs = p.get("char_start")
                if isinstance(cs, int) and abs(transcript_text.find(text) - cs) > _OFFSET_SLACK:
                    misplaced += 1
            elif " ".join(text.split()) not in norm:
                missing += 1
        if missing:
            violations.append(
                f"{missing}/{len(quotes)} quotes do not occur in the transcript — they are "
                "fabricated, not grounded"
            )
        if misplaced:
            violations.append(
                f"{misplaced}/{len(quotes)} quotes have a char_start that does not point at their "
                "own text — every speaker attributed from those offsets is a coin flip"
            )

    # 3. Attribution wired at all. Quotes exist but nobody is ever attributed => the speaker map was
    #    never built (GI had no names list; or the transcript was anonymised by cleaning_v4).
    speakers = [(i.get("properties") or {}).get("speaker") for i in grounded]
    if grounded and not any(speakers):
        violations.append(
            f"attribution produced NOTHING: {len(grounded)} grounded insights, 0 with a speaker. "
            "The speaker map is empty (anonymised transcript, or no named turns were read)"
        )

    # 4. A speaker who never holds the microphone. This is the Elon Musk bug: a name that appears
    #    only in the episode description, assigned to a diarized voice cluster.
    if turns:
        known = {label.lower() for _off, label in turns}
        ghosts = sorted(
            {str(s) for s in speakers if s and str(s).lower() not in known}  # noqa: E501
        )
        if ghosts:
            violations.append(
                f"{len(ghosts)} insight speaker(s) never speak in the transcript: "
                f"{', '.join(ghosts)}. They were read off the description, not the mic"
            )

    return violations


def log_artifact_invariants(
    artifact: Dict[str, Any],
    transcript_text: Optional[str] = None,
    turns: Optional[Sequence[Tuple[int, str]]] = None,
) -> List[str]:
    """Check *artifact* and log every violation at ERROR. Returns the violations.

    Deliberately does not raise: a broken wire should be loud, but an episode that already cost a
    transcription should still emit what it has. The contract tests are what keep this from being
    a log nobody reads.
    """
    violations = check_artifact_invariants(artifact, transcript_text, turns)
    for v in violations:
        logger.error("GI INVARIANT VIOLATED [%s]: %s", artifact.get("episode_id", "?"), v)
    return violations

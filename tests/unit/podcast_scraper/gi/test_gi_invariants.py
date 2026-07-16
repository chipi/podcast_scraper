"""Each test here is a bug that actually shipped, rebuilt as the artifact it produced.

The unit fixtures elsewhere catch traps someone was imaginative enough to invent. These catch the
*class*: output that cannot be true no matter which stage broke. If a future refactor disconnects
the grounder, the speaker map, or the offsets, one of these fails — and, at runtime, the pipeline
logs it at ERROR instead of writing a confident corpus full of nothing.

``test_a_healthy_artifact_is_silent`` is the one that keeps the rest honest: a checker that flags
everything is as useless as one that flags nothing.
"""

from __future__ import annotations

from typing import Any, Dict, List

from podcast_scraper.gi.invariants import check_artifact_invariants

TRANSCRIPT = (
    "Kevin Roose: OpenAI announced a loosened partnership with Microsoft this week, "
    "and Elon Musk is suing them.\n\n"
    "Dr. Adam Rodman: Doctors are already using chatbots for differential diagnosis.\n"
)
TURNS = [(0, "Kevin Roose"), (TRANSCRIPT.index("Dr. Adam Rodman"), "Dr. Adam Rodman")]

QUOTE_TEXT = "Doctors are already using chatbots for differential diagnosis."


def _artifact(
    insights: List[Dict[str, Any]],
    quotes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": "3.0",
        "model_version": "test",
        "prompt_version": "test",
        "episode_id": "ep1",
        "nodes": insights + quotes,
        "edges": edges,
    }


def _insight(iid: str, speaker: str | None) -> Dict[str, Any]:
    return {"id": iid, "type": "Insight", "properties": {"text": "an insight", "speaker": speaker}}


def _quote(qid: str, text: str, char_start: int) -> Dict[str, Any]:
    return {"id": qid, "type": "Quote", "properties": {"text": text, "char_start": char_start}}


def _healthy() -> Dict[str, Any]:
    return _artifact(
        [_insight("i1", "Dr. Adam Rodman")],
        [_quote("q1", QUOTE_TEXT, TRANSCRIPT.index(QUOTE_TEXT))],
        [{"type": "SUPPORTED_BY", "from": "i1", "to": "q1"}],
    )


def test_a_healthy_artifact_is_silent() -> None:
    """A checker that flags a correct artifact is worse than no checker."""
    assert check_artifact_invariants(_healthy(), TRANSCRIPT, TURNS) == []


def test_insights_with_zero_quotes_is_caught() -> None:
    """THE 513-INSIGHTS-ZERO-QUOTES BUG.

    Deleting the evidence-provider align made every eval cell ground with a provider that was never
    configured. The run completed, reported success, and emitted 513 insights and not one quote.
    """
    broken = _artifact([_insight("i1", None)], [], [])
    (violation,) = check_artifact_invariants(broken, TRANSCRIPT, TURNS)
    assert "grounding produced NOTHING" in violation


def test_grounded_insights_with_zero_speakers_is_caught() -> None:
    """THE ATTRIBUTION-NEVER-RAN BUG.

    ``build_named_turns`` only attributes names it can match against a detected-person list, and GI
    never had one — so the branch never ran and every quote shipped ``speaker_id: None``. Same
    signature as pointing GI at ``cleaning_v4``, which anonymises the names away.
    """
    broken = _artifact(
        [_insight("i1", None)],
        [_quote("q1", QUOTE_TEXT, TRANSCRIPT.index(QUOTE_TEXT))],
        [{"type": "SUPPORTED_BY", "from": "i1", "to": "q1"}],
    )
    (violation,) = check_artifact_invariants(broken, TRANSCRIPT, TURNS)
    assert "attribution produced NOTHING" in violation


def test_a_speaker_who_never_speaks_is_caught() -> None:
    """THE ELON MUSK BUG.

    Musk is named in the description as the man suing OpenAI. The LLM speaker detector returned him
    as a *speaker*, he was assigned positionally to a diarized voice cluster, and the doctor's
    insights were published under his name. He has no turn in the transcript — that is checkable.
    """
    broken = _artifact(
        [_insight("i1", "Elon Musk")],
        [_quote("q1", QUOTE_TEXT, TRANSCRIPT.index(QUOTE_TEXT))],
        [{"type": "SUPPORTED_BY", "from": "i1", "to": "q1"}],
    )
    (violation,) = check_artifact_invariants(broken, TRANSCRIPT, TURNS)
    assert "never speak in the transcript" in violation
    assert "Elon Musk" in violation


def test_a_fabricated_quote_is_caught() -> None:
    """A quote is a verbatim span of the transcript or it is not a quote."""
    broken = _artifact(
        [_insight("i1", "Kevin Roose")],
        [_quote("q1", "Doctors should never be allowed to use chatbots at all.", 0)],
        [{"type": "SUPPORTED_BY", "from": "i1", "to": "q1"}],
    )
    violations = check_artifact_invariants(broken, TRANSCRIPT, TURNS)
    assert any("do not occur in the transcript" in v for v in violations)


def test_a_quote_whose_offset_points_elsewhere_is_caught() -> None:
    """THE OFFSET BUG, and the reason it matters.

    The speaker is derived by looking up which turn contains the quote's ``char_start``. If the
    offset does not point at the quote's own text, the speaker is whoever happens to be talking at
    a meaningless position — here, Kevin gets credit for the doctor's line.
    """
    broken = _artifact(
        [_insight("i1", "Kevin Roose")],
        [_quote("q1", QUOTE_TEXT, 0)],  # the quote is Rodman's; offset 0 is Kevin's turn
        [{"type": "SUPPORTED_BY", "from": "i1", "to": "q1"}],
    )
    violations = check_artifact_invariants(broken, TRANSCRIPT, TURNS)
    assert any("does not point at their own text" in v for v in violations)


def test_no_insights_is_not_a_wiring_violation() -> None:
    """An episode that yields nothing is a content outcome, not a disconnected wire."""
    assert check_artifact_invariants(_artifact([], [], []), TRANSCRIPT, TURNS) == []

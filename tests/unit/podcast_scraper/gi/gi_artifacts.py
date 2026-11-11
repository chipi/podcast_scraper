"""GI test artifacts: populated artifacts built through the real path (see conftest.py).

A module of its own, not conftest: a test that imports from ``conftest`` gets whichever
conftest pytest loaded first when one run collects several test directories.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence
from unittest.mock import MagicMock, patch

from podcast_scraper.gi import build_artifact
from podcast_scraper.gi.grounding import GroundedQuote

DEFAULT_INSIGHT = "A real insight extracted from the transcript."
DEFAULT_TRANSCRIPT = "We have evidence here in the transcript body."


def make_cfg(**overrides: Any) -> Any:
    """A cfg that drives the grounded provider path."""
    cfg = MagicMock()
    cfg.generate_gi = True
    cfg.gi_require_grounding = True
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@contextmanager
def grounding_returns(quotes: Sequence[GroundedQuote]) -> Iterator[None]:
    """Pin what the evidence stack finds, so a test can assert on Quotes deterministically."""
    with patch(
        "podcast_scraper.gi.grounding.find_grounded_quotes_via_providers",
        return_value=list(quotes),
    ):
        yield


def grounded_quote(
    text: str = "evidence",
    *,
    char_start: int = 1,
    char_end: int = 9,
    qa_score: float = 0.9,
    nli_score: float = 0.85,
) -> GroundedQuote:
    return GroundedQuote(
        char_start=char_start,
        char_end=char_end,
        text=text,
        qa_score=qa_score,
        nli_score=nli_score,
    )


def artifact_with_insights(
    episode_id: str = "ep:1",
    transcript: str = DEFAULT_TRANSCRIPT,
    *,
    insight_texts: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """An artifact with real Insight nodes and NO quotes (no evidence stack wired).

    The replacement for ``build_artifact(id, text)`` in tests that only need insights to exist.
    """
    return build_artifact(
        episode_id,
        transcript,
        prompt_version="v1",
        insight_texts=insight_texts or [DEFAULT_INSIGHT],
        **kwargs,
    )


def artifact_with_grounded_insights(
    episode_id: str = "ep:1",
    transcript: str = DEFAULT_TRANSCRIPT,
    *,
    insight_texts: Optional[List[str]] = None,
    quotes: Optional[Sequence[GroundedQuote]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """An artifact with Insight + Quote + SUPPORTED_BY, built through the REAL evidence stack.

    This is what tests should use when they need the full shape — the shape the placeholder used
    to hand out for free, now produced by the path production actually takes.
    """
    if quotes is None:
        # Offsets must slice the REAL transcript back to the quote text, or the artifact's
        # offset invariant drops the Quote and the caller silently gets an insight with no
        # evidence. Derive them from the transcript the caller actually passed rather than
        # from a fixed guess.
        needle = "evidence" if "evidence" in transcript else transcript.strip().split(" ")[0]
        start = transcript.index(needle)
        quotes = [grounded_quote(needle, char_start=start, char_end=start + len(needle))]

    with grounding_returns(quotes):
        return build_artifact(
            episode_id,
            transcript,
            cfg=kwargs.pop("cfg", None) or make_cfg(),
            prompt_version="v1",
            insight_texts=insight_texts or [DEFAULT_INSIGHT],
            quote_extraction_provider=MagicMock(),
            entailment_provider=MagicMock(),
            **kwargs,
        )

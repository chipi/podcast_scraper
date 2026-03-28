"""GIL grounding: extractive QA + NLI to find quotes that support an insight.

Uses evidence stack (extractive_qa, nli_loader) when available, or provider
extract_quotes/score_entailment when quote_extraction_provider/entailment_provider
are set. Returns list of grounded quote spans (char_start, char_end, verbatim text).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Thresholds for evidence stack (RFC-049 / scope doc)
QA_SCORE_MIN = 0.3
NLI_ENTAILMENT_MIN = 0.5


@dataclass
class QuoteCandidate:
    """A candidate quote span from QA (before NLI)."""

    char_start: int
    char_end: int
    text: str
    qa_score: float


@dataclass
class GroundedQuote:
    """A transcript span that passed QA + NLI as evidence for an insight."""

    char_start: int
    char_end: int
    text: str
    qa_score: float
    nli_score: float


def find_grounded_quotes(
    transcript: str,
    insight_text: str,
    qa_model_id: str,
    nli_model_id: str,
    qa_device: Optional[str] = None,
    nli_device: Optional[str] = None,
    qa_score_min: float = QA_SCORE_MIN,
    nli_entailment_min: float = NLI_ENTAILMENT_MIN,
) -> List[GroundedQuote]:
    """Run QA then NLI to return quotes that support the insight.

    QA question: "What evidence supports: {insight_text}?". Spans with
    score >= qa_score_min are checked with NLI(premise=quote, hypothesis=insight).
    Only spans with entailment >= nli_entailment_min are returned.

    Args:
        transcript: Full transcript text (context for QA).
        insight_text: The insight claim (for QA question and NLI hypothesis).
        qa_model_id: Model for extractive QA (alias or full HF ID).
        nli_model_id: Model for NLI (alias or full HF ID).
        qa_device: Device for QA (cpu, cuda, mps) or None for auto.
        nli_device: Device for NLI or None for auto.
        qa_score_min: Minimum QA score to consider a span.
        nli_entailment_min: Minimum entailment score to accept as grounded.

    Returns:
        List of GroundedQuote (char_start, char_end, text, qa_score, nli_score).
    """
    if not (transcript and insight_text):
        return []

    try:
        from podcast_scraper.providers.ml import extractive_qa, nli_loader
    except ImportError:
        logger.debug("Evidence stack not available (transformers/sentence_transformers)")
        return []

    question = f"What evidence supports: {insight_text.strip()}"
    try:
        span = extractive_qa.answer(
            context=transcript,
            question=question,
            model_id=qa_model_id,
            device=qa_device,
        )
    except Exception as e:
        logger.warning("Extractive QA failed for GIL grounding: %s", e)
        return []

    if span.score < qa_score_min:
        return []

    verbatim = transcript[span.start : span.end] if span.end <= len(transcript) else span.answer
    try:
        nli_score = nli_loader.entailment_score(
            premise=verbatim,
            hypothesis=insight_text.strip(),
            model_id=nli_model_id,
            device=nli_device,
        )
    except Exception as e:
        logger.warning("NLI failed for GIL grounding: %s", e)
        return []

    if nli_score < nli_entailment_min:
        return []

    return [
        GroundedQuote(
            char_start=span.start,
            char_end=span.end,
            text=verbatim,
            qa_score=span.score,
            nli_score=nli_score,
        )
    ]


def find_grounded_quotes_via_providers(
    transcript: str,
    insight_text: str,
    quote_extraction_provider: Any,
    entailment_provider: Any,
    qa_score_min: float = QA_SCORE_MIN,
    nli_entailment_min: float = NLI_ENTAILMENT_MIN,
    pipeline_metrics: Optional[Any] = None,
) -> List[GroundedQuote]:
    """Run QA then NLI using provider methods; return quotes that support the insight.

    Calls quote_extraction_provider.extract_quotes(transcript, insight_text), then
    entailment_provider.score_entailment(quote.text, insight_text) for each candidate.
    Only spans with qa_score >= qa_score_min and nli_score >= nli_entailment_min
    are returned.

    Args:
        transcript: Full transcript text.
        insight_text: The insight claim.
        quote_extraction_provider: Provider with extract_quotes(transcript, insight_text).
        entailment_provider: Provider with score_entailment(premise, hypothesis).
        qa_score_min: Minimum QA score to consider a span.
        nli_entailment_min: Minimum entailment score to accept as grounded.
        pipeline_metrics: Optional metrics; when set, evidence call counts are updated.

    Returns:
        List of GroundedQuote (char_start, char_end, text, qa_score, nli_score).
    """
    if not (transcript and insight_text):
        return []

    extract_fn = getattr(quote_extraction_provider, "extract_quotes", None)
    score_fn = getattr(entailment_provider, "score_entailment", None)
    if not callable(extract_fn) or not callable(score_fn):
        logger.debug(
            "Provider(s) missing extract_quotes or score_entailment; skipping provider grounding."
        )
        return []

    try:
        raw = extract_fn(transcript=transcript.strip(), insight_text=insight_text.strip())
    except Exception as e:
        logger.warning("extract_quotes failed for GIL grounding: %s", e)
        return []

    if pipeline_metrics is not None and hasattr(
        pipeline_metrics, "gi_evidence_extract_quotes_calls"
    ):
        pipeline_metrics.gi_evidence_extract_quotes_calls += 1

    # Validate return type: must be a list of objects with char_start, char_end, text, qa_score
    if not isinstance(raw, list):
        logger.debug(
            "extract_quotes returned non-list (%s); skipping provider grounding.",
            type(raw).__name__,
        )
        return []
    candidates = raw

    result: List[GroundedQuote] = []
    insight_stripped = insight_text.strip()
    for c in candidates:
        char_start = getattr(c, "char_start", None)
        char_end = getattr(c, "char_end", None)
        text = getattr(c, "text", None)
        qa_score = getattr(c, "qa_score", 0.0)
        if text is None or char_start is None or char_end is None:
            continue
        if qa_score < qa_score_min:
            continue
        if pipeline_metrics is not None and hasattr(
            pipeline_metrics, "gi_evidence_score_entailment_calls"
        ):
            pipeline_metrics.gi_evidence_score_entailment_calls += 1
        try:
            nli_score = score_fn(premise=text, hypothesis=insight_stripped)
        except Exception as e:
            logger.debug("score_entailment failed for quote: %s", e)
            continue
        if nli_score < nli_entailment_min:
            continue
        result.append(
            GroundedQuote(
                char_start=int(char_start),
                char_end=int(char_end),
                text=str(text),
                qa_score=float(qa_score),
                nli_score=nli_score,
            )
        )
    return result

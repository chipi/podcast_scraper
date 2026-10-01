"""GIL grounding: extractive QA + NLI to find quotes that support an insight.

``find_grounded_quotes_via_providers`` is the path used by ``gi.pipeline`` (via
configured summarization backends). ``find_grounded_quotes`` calls ML
``extractive_qa`` + ``nli_loader`` directly and remains for tests and tooling.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from podcast_scraper.utils.log_redaction import format_exception_for_log

logger = logging.getLogger(__name__)

# Thresholds for evidence stack
QA_SCORE_MIN = 0.3
NLI_ENTAILMENT_MIN = 0.5

# Appended to insight_text on provider extract_quotes retries (after first empty result).
GIL_EXTRACT_RETRY_INSIGHT_SUFFIX = (
    "\n\nReminder: quote_text must be one contiguous substring copied exactly "
    'from the transcript (verbatim wording, including small words like "And").'
)


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
    qa_score: Optional[float] = None
    nli_score: Optional[float] = None


def _normalize_apostrophes(s: str) -> str:
    return s.replace("\u2019", "'").replace("\u2018", "'")


def _best_regex_matches(matches: List[Any]) -> Any:
    if not matches:
        return None
    return max(matches, key=lambda m: (m.end() - m.start(), -m.start()))


def _subphrase_span_regex(
    transcript: str, qt: str, min_subphrase_words: int
) -> Optional[Tuple[int, int]]:
    """The reference matcher: one regex scan of the whole transcript per (start, end) sub-phrase.

    Correct, and O(W² · len(transcript)) when a long quote is not verbatim — 12 s for one
    200-word near-verbatim quote on a 67k-char transcript, measured. `_subphrase_span` returns
    the same answer from a token index; this stays as its fallback and its parity oracle.
    """
    words = re.findall(r"\S+", qt)
    if len(words) < 2:
        return None
    eff_min = min(min_subphrase_words, len(words))
    best: Optional[Tuple[int, int, int]] = None  # span_len, start, end
    max_i = len(words) - eff_min
    for i in range(0, max_i + 1):
        for j in range(len(words), i + eff_min - 1, -1):
            pattern = r"\s+".join(re.escape(w) for w in words[i:j])
            found = list(re.finditer(pattern, transcript, re.IGNORECASE))
            match = _best_regex_matches(found)
            if match is None:
                continue
            a, b = match.start(), match.end()
            span_len = b - a
            if best is None or (span_len, -a) > (best[0], -best[1]):
                best = (span_len, a, b)
            break
    if best is None:
        return None
    return (best[1], best[2])


@dataclass(frozen=True)
class _TokenIndex:
    """A transcript's whitespace tokens (``\\S+``), lowercased, with offsets and a suffix index."""

    starts: Tuple[int, ...]
    ends: Tuple[int, ...]
    lower: Tuple[str, ...]
    by_suffix: Dict[str, Tuple[int, ...]]
    length_stable: bool  # lower() never changed a token's length, so offsets carry over


@lru_cache(maxsize=4)
def _token_index(transcript: str) -> _TokenIndex:
    starts: List[int] = []
    ends: List[int] = []
    lower: List[str] = []
    stable = True
    by_suffix: Dict[str, List[int]] = {}
    for k, m in enumerate(re.finditer(r"\S+", transcript)):
        tok = m.group(0)
        low = tok.lower()
        stable = stable and len(low) == len(tok)
        starts.append(m.start())
        ends.append(m.end())
        lower.append(low)
        for s in range(len(low)):
            by_suffix.setdefault(low[s:], []).append(k)
    return _TokenIndex(
        tuple(starts),
        tuple(ends),
        tuple(lower),
        {s: tuple(v) for s, v in by_suffix.items()},
        stable,
    )


def _subphrase_span(
    transcript: str, qt: str, min_subphrase_words: int
) -> Optional[Tuple[int, int]]:
    """The longest contiguous sub-phrase of ``qt`` found in ``transcript`` — same answer as
    `_subphrase_span_regex`, from a token index instead of a regex scan per sub-phrase (#2207).

    The regex ``w_i\\s+…\\s+w_{j-1}`` (IGNORECASE, no anchors) matches where the first word is a
    SUFFIX of a transcript token, every middle word EQUALS a token, and the last word is a PREFIX
    of one. So each (quote word, transcript token) start extends along equal tokens once, instead
    of re-scanning 67k chars for every (i, j): on prod that scan held the single pipeline slot for
    ~34 minutes on one episode.

    Per start word ``i`` the reference takes the LONGEST ``j`` that matches anywhere, its
    leftmost-non-overlapping ``finditer`` matches, and the longest (then earliest) of those; then
    the longest (then earliest) over all ``i``. Reproduced exactly, overlap skipping included.
    """
    words = re.findall(r"\S+", qt)
    if len(words) < 2:
        return None
    idx = _token_index(transcript)
    q = [w.lower() for w in words]
    if not idx.length_stable or any(len(a) != len(b) for a, b in zip(q, words)):
        # A case fold that changes length breaks offset arithmetic; the regex is exact.
        return _subphrase_span_regex(transcript, qt, min_subphrase_words)

    n_tok = len(idx.lower)
    eff_min = min(min_subphrase_words, len(words))
    best: Optional[Tuple[int, int, int]] = None  # span_len, start, end
    for i in range(0, len(words) - eff_min + 1):
        # Longest L (= j - i) matching at each candidate token k.
        reach: List[Tuple[int, int]] = []  # (k, max L at k)
        for k in idx.by_suffix.get(q[i], ()):
            r = 0
            while i + 1 + r < len(q) and k + 1 + r < n_tok and idx.lower[k + 1 + r] == q[i + 1 + r]:
                r += 1
            # Words i+1 .. i+r equal tokens k+1 .. k+r, so any L <= r+1 matches with an equal
            # last word; L = r+2 needs the next word to be a PREFIX of the next token.
            max_l = r + 1
            if (
                i + r + 1 < len(q)
                and k + r + 1 < n_tok
                and idx.lower[k + r + 1].startswith(q[i + r + 1])
            ):
                max_l = r + 2
            if max_l >= max(eff_min, 2):
                reach.append((k, max_l))
        if not reach:
            continue
        length = max(ml for _k, ml in reach)
        # The reference's matches for words[i:i+length]: leftmost, non-overlapping.
        chosen: Optional[Tuple[int, int, int]] = None
        prev_end = -1
        for k, ml in sorted(reach):
            if ml < length:
                continue
            a = idx.ends[k] - len(q[i])
            if a < prev_end:
                continue
            b = idx.starts[k + length - 1] + len(q[i + length - 1])
            prev_end = b
            if chosen is None or (b - a, -a) > (chosen[0], -chosen[1]):
                chosen = (b - a, a, b)
        if chosen is not None and (best is None or (chosen[0], -chosen[1]) > (best[0], -best[1])):
            best = chosen
    if best is None:
        return None
    return (best[1], best[2])


def resolve_llm_quote_span(
    transcript: str,
    quote_text: str,
    *,
    min_subphrase_words: int = 3,
) -> Optional[Tuple[int, int, str]]:
    """Map LLM ``quote_text`` to a verbatim slice of ``transcript`` with offsets.

    Models often omit lead-in words (e.g. ``And``), collapse whitespace, or
    slightly paraphrase one edge. The previous fallback ``(0, len(quote))`` was
    never a valid transcript span.

    Strategy: exact substring (original and apostrophe-normalized), then
    whitespace-tolerant token regex on contiguous sub-phrases; among equal-length
    matches, the **earliest** occurrence in the transcript wins.

    Args:
        transcript: Full episode transcript.
        quote_text: Quote string from the LLM (JSON ``quote_text``).
        min_subphrase_words: Minimum token count when matching a sub-phrase.

    Returns:
        ``(char_start, char_end_exclusive, verbatim)`` or ``None`` if no match.
    """
    if not transcript or not quote_text:
        return None
    qt = quote_text.strip()
    if not qt:
        return None

    pos = transcript.find(qt)
    if pos != -1:
        end = pos + len(qt)
        return (pos, end, transcript[pos:end])

    t_norm = _normalize_apostrophes(transcript)
    q_norm = _normalize_apostrophes(qt)
    if t_norm != transcript or q_norm != qt:
        pos = t_norm.find(q_norm)
        if pos != -1:
            end = pos + len(q_norm)
            return (pos, end, transcript[pos:end])

    span = _subphrase_span(transcript, qt, min_subphrase_words)
    if span is None and (t_norm != transcript or q_norm != qt):
        span = _subphrase_span(t_norm, q_norm, min_subphrase_words)
    if span is None:
        return None
    a, b = span
    return (a, b, transcript[a:b])


def find_grounded_quotes(
    transcript: str,
    insight_text: str,
    qa_model_id: str,
    nli_model_id: str,
    qa_device: Optional[str] = None,
    nli_device: Optional[str] = None,
    qa_score_min: float = QA_SCORE_MIN,
    nli_entailment_min: float = NLI_ENTAILMENT_MIN,
    qa_window_chars: int = 0,
    qa_window_overlap_chars: int = 250,
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
        qa_window_chars: When > 0 and transcript is longer, run QA on overlapping windows.
        qa_window_overlap_chars: Overlap between windows (must be < window when windowing).

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
        spans = extractive_qa.answer_candidates(
            context=transcript,
            question=question,
            model_id=qa_model_id,
            device=qa_device,
            window_chars=qa_window_chars,
            window_overlap_chars=qa_window_overlap_chars,
            top_k=3,
        )
    except Exception as e:
        logger.warning(
            "Extractive QA failed for GIL grounding: %s",
            format_exception_for_log(e),
        )
        return []

    results: List[GroundedQuote] = []
    seen_spans: set[tuple[int, int]] = set()
    for span in spans:
        if span.score < qa_score_min:
            continue
        key = (span.start, span.end)
        if key in seen_spans:
            continue
        seen_spans.add(key)

        verbatim = transcript[span.start : span.end] if span.end <= len(transcript) else span.answer
        try:
            nli_score = nli_loader.entailment_score(
                premise=verbatim,
                hypothesis=insight_text.strip(),
                model_id=nli_model_id,
                device=nli_device,
            )
        except Exception as e:
            logger.warning(
                "NLI failed for GIL grounding: %s",
                format_exception_for_log(e),
            )
            continue

        if nli_score < nli_entailment_min:
            continue

        results.append(
            GroundedQuote(
                char_start=span.start,
                char_end=span.end,
                text=verbatim,
                qa_score=span.score,
                nli_score=nli_score,
            )
        )

    return results


def find_grounded_quotes_via_providers(
    transcript: str,
    insight_text: str,
    quote_extraction_provider: Any,
    entailment_provider: Any,
    qa_score_min: float = QA_SCORE_MIN,
    nli_entailment_min: float = NLI_ENTAILMENT_MIN,
    pipeline_metrics: Optional[Any] = None,
    extract_retries: int = 0,
    prefetched_candidates: Optional[List[Any]] = None,
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
        extract_retries: Extra extract_quotes calls after an empty/non-list result;
            later attempts append ``GIL_EXTRACT_RETRY_INSIGHT_SUFFIX``. NLI always
            uses the original ``insight_text`` (no suffix).
        prefetched_candidates: When non-None (#698 Layer A), skip the per-insight
            ``extract_quotes`` call and use these candidates directly. Empty list is
            valid and means "the bundled call returned nothing for this insight" —
            the caller decides whether to retry staged. ``extract_retries`` is
            ignored on this path because the bundled call already controls retry
            policy at the provider level.

    Returns:
        List of GroundedQuote (char_start, char_end, text, qa_score, nli_score).
    """
    if not (transcript and insight_text):
        return []

    extract_fn = getattr(quote_extraction_provider, "extract_quotes", None)
    score_fn = getattr(entailment_provider, "score_entailment", None)
    if prefetched_candidates is None and not callable(extract_fn):
        logger.debug("Provider missing extract_quotes; skipping provider grounding.")
        return []
    if not callable(score_fn):
        logger.debug("Provider missing score_entailment; skipping provider grounding.")
        return []

    candidates: List[Any]
    if prefetched_candidates is not None:
        candidates = list(prefetched_candidates)
    else:
        candidates = []
        retries = max(0, int(extract_retries))
        for attempt in range(retries + 1):
            insight_for_extract = insight_text.strip()
            if attempt > 0:
                insight_for_extract += GIL_EXTRACT_RETRY_INSIGHT_SUFFIX
            try:
                raw = extract_fn(  # type: ignore[misc]
                    transcript=transcript.strip(),
                    insight_text=insight_for_extract,
                    pipeline_metrics=pipeline_metrics,
                )
            except Exception as e:
                logger.warning(
                    "extract_quotes failed for GIL grounding: %s",
                    format_exception_for_log(e),
                )
                return []

            if pipeline_metrics is not None and hasattr(
                pipeline_metrics, "gi_evidence_extract_quotes_calls"
            ):
                pipeline_metrics.gi_evidence_extract_quotes_calls += 1

            if not isinstance(raw, list):
                logger.debug(
                    "extract_quotes returned non-list (%s); skipping this attempt.",
                    type(raw).__name__,
                )
                continue
            if raw:
                candidates = raw
                break

    if not candidates:
        return []

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
            pipeline_metrics, "gi_evidence_nli_candidates_queued"
        ):
            pipeline_metrics.gi_evidence_nli_candidates_queued += 1
        try:
            nli_score = score_fn(
                premise=text,
                hypothesis=insight_stripped,
                pipeline_metrics=pipeline_metrics,
            )
        except Exception as e:
            logger.debug("score_entailment failed for quote: %s", e)
            continue
        if pipeline_metrics is not None and hasattr(
            pipeline_metrics, "gi_evidence_score_entailment_calls"
        ):
            pipeline_metrics.gi_evidence_score_entailment_calls += 1
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

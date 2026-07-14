"""GIL extraction pipeline: transcript + metadata -> nodes/edges -> artifact dict.

Builds a valid gi.json with Episode, Insight(s), Quote(s), and SUPPORTED_BY edges.
When cfg is provided and gi_require_grounding is True, uses the evidence stack
via ``quote_extraction_provider`` / ``entailment_provider`` (from callers or
``create_gil_evidence_providers(cfg, summary_provider=...)``). Insight texts come
from insight_texts, insight_provider.generate_insights(), or stub.

When transcript_segments are provided (from transcription with word/segment
timestamps), Quote nodes get precise timestamp_start_ms and timestamp_end_ms
(FR2.2).
"""

from __future__ import annotations

import logging
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING
from unittest.mock import Mock

from .. import config_constants
from ..exceptions import GILGroundingUnsatisfiedError
from ..graph_id_utils import (
    episode_node_id,
    gil_insight_node_id,
    gil_quote_node_id,
    person_node_id,
    slugify_label,
    topic_node_id_from_slug,
)
from ..providers.ml.diarization.roster import friendly_voice_label
from ..utils.log_redaction import format_exception_for_log
from .grounding import GroundedQuote
from .invariants import log_artifact_invariants
from .provenance import resolve_gil_artifact_model_version
from .speakers import build_unverified_named_turns, speaker_for_char

if TYPE_CHECKING:
    from podcast_scraper import config

logger = logging.getLogger(__name__)

# Max |len(transcript) - sum(segment text)| before skipping segment-based timestamps (FR2.2).
# Avoids mapping quote char offsets to wrong audio after reformatting or edited transcripts.
SEGMENT_TRANSCRIPT_ALIGNMENT_MAX_DELTA = 50

# Stub insight text used when no real insights (single stub)
_STUB_INSIGHT_TEXT = "Summary insight (stub)."


def _ground_insights_dispatch(
    cfg: Any,
    insight_specs: List[Tuple[str, Any]],
    transcript: str,
    quote_extraction_provider: Any,
    entailment_provider: Any,
    qa_score_min: float,
    nli_entailment_min: float,
    extract_retries: int,
    pipeline_metrics: Optional[Any],
    prefetched_by_idx: Optional[Dict[int, List[Any]]],
) -> List[List[GroundedQuote]]:
    """Pick the grounding flow based on ``cfg.gil_evidence_nli_mode`` (#698 Layer B).

    Bundled NLI mode raises through to the staged fallback when the bundled call
    itself fails (whole-batch fallback) and records a ``..._bundled_fallbacks``
    counter for matrix attribution.
    """
    nli_mode = getattr(cfg, "gil_evidence_nli_mode", "staged")
    if nli_mode == "bundled":
        try:
            return _ground_insights_with_bundled_nli(
                insight_specs=insight_specs,
                transcript=transcript,
                quote_extraction_provider=quote_extraction_provider,
                entailment_provider=entailment_provider,
                qa_score_min=qa_score_min,
                nli_entailment_min=nli_entailment_min,
                extract_retries=extract_retries,
                chunk_size=int(getattr(cfg, "gil_evidence_nli_chunk_size", 15)),
                pipeline_metrics=pipeline_metrics,
                prefetched_by_idx=prefetched_by_idx,
            )
        except Exception:
            if pipeline_metrics is not None and hasattr(
                pipeline_metrics, "gi_evidence_score_entailment_bundled_fallbacks"
            ):
                pipeline_metrics.gi_evidence_score_entailment_bundled_fallbacks += 1
            # Fall through to staged below.
    return _ground_insights_with_optional_prefetch(
        insight_specs=insight_specs,
        transcript=transcript,
        quote_extraction_provider=quote_extraction_provider,
        entailment_provider=entailment_provider,
        qa_score_min=qa_score_min,
        nli_entailment_min=nli_entailment_min,
        extract_retries=extract_retries,
        pipeline_metrics=pipeline_metrics,
        prefetched_by_idx=prefetched_by_idx,
    )


def _gather_qa_passing_candidates(
    insight_specs: List[Tuple[str, Any]],
    transcript: str,
    quote_extraction_provider: Any,
    qa_score_min: float,
    extract_retries: int,
    pipeline_metrics: Optional[Any],
    prefetched_by_idx: Optional[Dict[int, List[Any]]],
) -> List[List[Any]]:
    """Per-insight: collect QA-passing quote candidates (#698 Layer B prep).

    For each insight, use prefetched candidates when available (Layer A); otherwise
    issue the staged ``extract_quotes`` call. Filters by ``qa_score_min`` so the
    bundled NLI call only sees candidates worth scoring.
    """
    from .grounding import GIL_EXTRACT_RETRY_INSIGHT_SUFFIX

    extract_fn = getattr(quote_extraction_provider, "extract_quotes", None)
    per_insight: List[List[Any]] = []
    for idx, (it_text, _) in enumerate(insight_specs):
        cands: List[Any] = []
        if prefetched_by_idx is not None:
            cands = list(prefetched_by_idx.get(idx) or [])
        if not cands and callable(extract_fn):
            retries = max(0, int(extract_retries))
            for attempt in range(retries + 1):
                ins_for_extract = it_text.strip()
                if attempt > 0:
                    ins_for_extract += GIL_EXTRACT_RETRY_INSIGHT_SUFFIX
                try:
                    raw = extract_fn(
                        transcript=transcript,
                        insight_text=ins_for_extract,
                        pipeline_metrics=pipeline_metrics,
                    )
                except Exception as exc:
                    logger.warning(
                        "extract_quotes failed for GIL grounding: %s",
                        format_exception_for_log(exc),
                    )
                    raw = []
                if pipeline_metrics is not None and hasattr(
                    pipeline_metrics, "gi_evidence_extract_quotes_calls"
                ):
                    pipeline_metrics.gi_evidence_extract_quotes_calls += 1
                if isinstance(raw, list) and raw:
                    cands = raw
                    break
        # QA filter
        kept: List[Any] = []
        for c in cands:
            text = getattr(c, "text", None)
            if text is None or getattr(c, "char_start", None) is None:
                continue
            if getattr(c, "qa_score", 0.0) < qa_score_min:
                continue
            kept.append(c)
        if pipeline_metrics is not None and hasattr(
            pipeline_metrics, "gi_evidence_nli_candidates_queued"
        ):
            pipeline_metrics.gi_evidence_nli_candidates_queued += len(kept)
        per_insight.append(kept)
    return per_insight


def _ground_insights_with_bundled_nli(
    insight_specs: List[Tuple[str, Any]],
    transcript: str,
    quote_extraction_provider: Any,
    entailment_provider: Any,
    qa_score_min: float,
    nli_entailment_min: float,
    extract_retries: int,
    chunk_size: int,
    pipeline_metrics: Optional[Any],
    prefetched_by_idx: Optional[Dict[int, List[Any]]],
) -> List[List[GroundedQuote]]:
    """Bundled NLI flow (#698 Layer B): gather all pairs, call once, distribute scores.

    This implements the option (a) win from #698: every (insight, candidate) pair
    after QA filtering is sent in chunked bundled NLI calls instead of one call
    per pair. With ~70 pairs across an episode and ``chunk_size=15``, that's
    ~5 calls instead of 70.

    Per-pair fallback: when the bundled call doesn't return a score for a pair
    (model omitted it, parse error skipped it), the dispatcher issues a staged
    per-pair NLI call for just that pair so we don't drop candidates silently.

    Whole-batch fallback: when the bundled call raises, this function falls back
    to the staged per-insight loop in ``_ground_insights_with_optional_prefetch``
    via the caller's exception handling. Caller increments
    ``gi_evidence_score_entailment_bundled_fallbacks`` for that case.
    """
    bundled_fn = getattr(entailment_provider, "score_entailment_bundled", None)
    if not callable(bundled_fn):
        # Provider doesn't implement bundled NLI — fall back to staged path.
        return _ground_insights_with_optional_prefetch(
            insight_specs=insight_specs,
            transcript=transcript,
            quote_extraction_provider=quote_extraction_provider,
            entailment_provider=entailment_provider,
            qa_score_min=qa_score_min,
            nli_entailment_min=nli_entailment_min,
            extract_retries=extract_retries,
            pipeline_metrics=pipeline_metrics,
            prefetched_by_idx=prefetched_by_idx,
        )

    per_insight_candidates = _gather_qa_passing_candidates(
        insight_specs=insight_specs,
        transcript=transcript,
        quote_extraction_provider=quote_extraction_provider,
        qa_score_min=qa_score_min,
        extract_retries=extract_retries,
        pipeline_metrics=pipeline_metrics,
        prefetched_by_idx=prefetched_by_idx,
    )

    # Build flat (insight_idx, q_idx, premise, hypothesis) list.
    pair_list: List[Tuple[int, int, str, str]] = []
    for i_idx, (it_text, _) in enumerate(insight_specs):
        for q_idx, c in enumerate(per_insight_candidates[i_idx]):
            pair_list.append((i_idx, q_idx, str(c.text), it_text.strip()))

    if not pair_list:
        return [[] for _ in insight_specs]

    bundled_scores: Dict[int, float]
    try:
        bundled_scores = bundled_fn(
            pairs=[(p[2], p[3]) for p in pair_list],
            chunk_size=chunk_size,
            pipeline_metrics=pipeline_metrics,
        )
    except Exception as exc:
        logger.warning(
            "score_entailment_bundled failed; raising for staged fallback: %s",
            format_exception_for_log(exc),
        )
        raise

    if pipeline_metrics is not None and hasattr(
        pipeline_metrics, "gi_evidence_score_entailment_bundled_calls"
    ):
        # Each chunk is one provider call; chunk count == number of bundled calls.
        chunk_count = (len(pair_list) + max(1, chunk_size) - 1) // max(1, chunk_size)
        pipeline_metrics.gi_evidence_score_entailment_bundled_calls += chunk_count

    # Per-pair fallback for any pair the bundled call didn't score.
    score_fn = getattr(entailment_provider, "score_entailment", None)
    final_scores: Dict[int, float] = {}
    for pair_idx, (_, _, premise, hypothesis) in enumerate(pair_list):
        if pair_idx in bundled_scores:
            final_scores[pair_idx] = bundled_scores[pair_idx]
            continue
        if not callable(score_fn):
            final_scores[pair_idx] = 0.0
            continue
        try:
            final_scores[pair_idx] = float(
                score_fn(
                    premise=premise,
                    hypothesis=hypothesis,
                    pipeline_metrics=pipeline_metrics,
                )
            )
            if pipeline_metrics is not None and hasattr(
                pipeline_metrics, "gi_evidence_score_entailment_calls"
            ):
                pipeline_metrics.gi_evidence_score_entailment_calls += 1
        except Exception:
            final_scores[pair_idx] = 0.0

    # Distribute scores to per-insight GroundedQuote lists.
    grounded_by_insight: List[List[GroundedQuote]] = [[] for _ in insight_specs]
    for pair_idx, (i_idx, q_idx, _, _) in enumerate(pair_list):
        score = final_scores.get(pair_idx, 0.0)
        if score < nli_entailment_min:
            continue
        c = per_insight_candidates[i_idx][q_idx]
        grounded_by_insight[i_idx].append(
            GroundedQuote(
                char_start=int(c.char_start),
                char_end=int(c.char_end),
                text=str(c.text),
                qa_score=float(getattr(c, "qa_score", 0.0)),
                nli_score=score,
            )
        )
    return grounded_by_insight


def _ground_insights_with_optional_prefetch(
    insight_specs: List[Tuple[str, Any]],
    transcript: str,
    quote_extraction_provider: Any,
    entailment_provider: Any,
    qa_score_min: float,
    nli_entailment_min: float,
    extract_retries: int,
    pipeline_metrics: Optional[Any],
    prefetched_by_idx: Optional[Dict[int, List[Any]]],
) -> List[List[GroundedQuote]]:
    """Run :func:`find_grounded_quotes_via_providers` per insight, honouring prefetched.

    Extracted from ``build_artifact`` to keep that orchestrator under the project
    cyclomatic-complexity threshold. When ``prefetched_by_idx`` is non-None and
    contains a non-empty list for the insight at index ``idx``, those candidates
    are used in place of a fresh ``extract_quotes`` call. Otherwise the per-insight
    staged extract path runs (Layer A failure mode for that insight).
    """
    from .grounding import find_grounded_quotes_via_providers

    out: List[List[GroundedQuote]] = []
    for idx, (it_text, _) in enumerate(insight_specs):
        candidates_for_insight: Optional[List[Any]] = None
        if prefetched_by_idx is not None:
            rows = prefetched_by_idx.get(idx) or []
            if rows:
                candidates_for_insight = rows
        grounded = find_grounded_quotes_via_providers(
            transcript=transcript,
            insight_text=it_text,
            quote_extraction_provider=quote_extraction_provider,
            entailment_provider=entailment_provider,
            qa_score_min=qa_score_min,
            nli_entailment_min=nli_entailment_min,
            pipeline_metrics=pipeline_metrics,
            extract_retries=extract_retries,
            prefetched_candidates=candidates_for_insight,
        )
        out.append(grounded if isinstance(grounded, list) else [])
    return out


def _maybe_prefetch_bundled_candidates(
    cfg: Any,
    quote_extraction_provider: Any,
    transcript: str,
    insight_texts: List[str],
    pipeline_metrics: Optional[Any],
) -> Optional[Dict[int, List[Any]]]:
    """Run the bundled ``extract_quotes_bundled`` call when configured (#698 Layer A).

    Returns a dict mapping each insight index to its candidate list when the
    bundled call succeeds. Returns ``None`` when bundled mode is off, the
    provider doesn't expose the bundled method, or the bundled call raises —
    callers should fall back to the staged per-insight extract path. Metrics
    counters track adoption (``..._bundled_calls``) and failure
    (``..._bundled_fallbacks``) so the autoresearch matrix can attribute
    savings.
    """
    quote_mode = getattr(cfg, "gil_evidence_quote_mode", "staged")
    extract_bundled_fn = getattr(quote_extraction_provider, "extract_quotes_bundled", None)
    if quote_mode != "bundled" or not callable(extract_bundled_fn):
        return None
    try:
        prefetched = extract_bundled_fn(
            transcript=transcript,
            insight_texts=insight_texts,
            pipeline_metrics=pipeline_metrics,
        )
    except Exception as exc:
        logger.warning(
            "extract_quotes_bundled failed; falling back to staged: %s",
            format_exception_for_log(exc),
        )
        if pipeline_metrics is not None and hasattr(
            pipeline_metrics, "gi_evidence_extract_quotes_bundled_fallbacks"
        ):
            pipeline_metrics.gi_evidence_extract_quotes_bundled_fallbacks += 1
        return None
    if pipeline_metrics is not None and hasattr(
        pipeline_metrics, "gi_evidence_extract_quotes_bundled_calls"
    ):
        pipeline_metrics.gi_evidence_extract_quotes_bundled_calls += 1
    if not isinstance(prefetched, dict):
        return None
    out: Dict[int, List[Any]] = {}
    for k, v in prefetched.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, list):
            out[idx] = v
    return out


def _record_stub_fallback(pipeline_metrics: Optional[Any], exc: Exception) -> None:
    """#701: surface evidence-stack stub fallback at WARNING + metrics.

    Originally a ``logger.debug`` call which masked silent stub-degradation
    in normal logs. cloud_thin produced 1-stub gi.json across 9 real-feed
    episodes for weeks because ``_rank_about_edges_for_insights`` raised
    ImportError on ``sentence-transformers`` (in [ml]/[search] only) and
    the debug log never surfaced the failure. Promote to WARNING + flip a
    metrics counter so ops dashboards see stub fallback in real time.
    """
    if pipeline_metrics is not None and hasattr(
        pipeline_metrics, "gi_artifact_stub_fallback_count"
    ):
        pipeline_metrics.gi_artifact_stub_fallback_count += 1
    logger.warning(
        "GIL evidence stack (provider path) failed, falling back to " "stub/degraded artifact: %s",
        exc,
        exc_info=True,
    )


def _apply_gi_insight_filters(
    insight_specs: List[Tuple[str, str]], pipeline_metrics: Optional[Any]
) -> List[Tuple[str, str]]:
    """#652 Part B — run ad + dialogue filters on (text, type) specs.

    Source-agnostic (prefilled / provider / stub).
    Conservative thresholds live in ``gi.filters``. Extracted helper so
    ``build_artifact`` stays under the cyclomatic-complexity budget.
    """
    from .filters import apply_insight_filters

    insight_dicts = [{"text": t, "insight_type": k} for t, k in insight_specs]
    kept, ads_dropped, dialogue_dropped = apply_insight_filters(insight_dicts)
    if not (ads_dropped or dialogue_dropped):
        return insight_specs
    if pipeline_metrics is not None:
        if ads_dropped and hasattr(pipeline_metrics, "record_ads_filtered"):
            pipeline_metrics.record_ads_filtered(ads_dropped)
        if dialogue_dropped and hasattr(pipeline_metrics, "record_dialogue_insights_dropped"):
            pipeline_metrics.record_dialogue_insights_dropped(dialogue_dropped)
    return [(d["text"], d.get("insight_type") or "claim") for d in kept]


def _rank_about_edges_for_insights(
    insight_texts: List[str],
    topic_specs: List[Tuple[str, str]],
    *,
    top_k: Optional[int],
    floor: Optional[float],
    encoder: Optional[Any],
) -> List[List[Tuple[str, float]]]:
    """Thin wrapper around ``about_edges.rank_about_edges`` with pipeline
    defaults and an empty-input short-circuit (avoids loading the embedding
    model when there are no topics to score against).

    #701: graceful fallback when ``sentence-transformers`` is not installed.
    Pipeline images that ship only ``[llm]`` extras (no ``[search]`` / ``[ml]``)
    don't have the embedding library — without this guard the
    ``ImportError`` bubbles through ``_artifact_from_multi_insight``, gets
    swallowed by ``build_artifact``'s outer ``except Exception`` (line ~783),
    and silently degrades the whole episode to a 1-stub-insight artifact.
    Returning empty edges keeps the rest of the artifact (insights + quotes)
    intact at the cost of no insight→topic ABOUT edges.
    """
    if not insight_texts or not topic_specs:
        return [[] for _ in insight_texts]
    try:
        from .about_edges import (
            ABOUT_EDGE_DEFAULT_FLOOR,
            ABOUT_EDGE_DEFAULT_TOP_K,
            rank_about_edges,
        )

        k = ABOUT_EDGE_DEFAULT_TOP_K if top_k is None else top_k
        f = ABOUT_EDGE_DEFAULT_FLOOR if floor is None else floor
        return rank_about_edges(
            insight_texts,
            topic_specs,
            top_k=k,
            floor=f,
            encoder=encoder,
        )
    except ImportError as exc:
        logger.warning(
            "GI ABOUT-edge ranking skipped (missing sentence-transformers): %s. "
            "Insights persist without insight->topic ABOUT edges.",
            exc,
        )
        return [[] for _ in insight_texts]


def _dedupe_topic_node_specs(
    topic_labels: Optional[List[str]],
) -> List[Tuple[str, str]]:
    """Build ordered (topic_node_id, display_label) list; slug-dedupe like KG."""
    if not topic_labels:
        return []
    seen_slugs: Set[str] = set()
    out: List[Tuple[str, str]] = []
    for lab in topic_labels:
        raw = (lab or "").strip()
        if not raw:
            continue
        slug = slugify_label(raw)
        if slug in seen_slugs:
            # #653 Part C: within-episode dedup — skip second Topic with same slug.
            continue
        seen_slugs.add(slug)
        # #653 Part A: truncate at word boundary rather than mid-word. Short KG
        # canonical topics (2–3 words, typically < 50 chars) pass through
        # unchanged; only legacy long bullet-slugs (the fallback path) exercise
        # this branch, and `textwrap.shorten` uses whitespace as break hints.
        display = raw if len(raw) <= 200 else textwrap.shorten(raw, width=200, placeholder="…")
        out.append((topic_node_id_from_slug(slug), display))
    return out


def _insight_confidence_from_quotes(quotes: List[GroundedQuote]) -> Optional[float]:
    """Aggregate QA (preferred) or NLI scores from grounded quotes for Insight.confidence."""
    qa_vals = [
        float(gq.qa_score)
        for gq in quotes
        if isinstance(gq, GroundedQuote) and gq.qa_score is not None
    ]
    if qa_vals:
        return max(qa_vals)
    nli_vals = [
        float(gq.nli_score)
        for gq in quotes
        if isinstance(gq, GroundedQuote) and gq.nli_score is not None
    ]
    if nli_vals:
        return max(nli_vals)
    return None


def _cfg_float(cfg: Any, name: str, default: float) -> float:
    """Read a float from cfg; tolerate MagicMock / bad types in tests."""
    if cfg is None:
        return default
    raw = getattr(cfg, name, default)
    # MagicMock.__float__ returns 1.0; treat mocks as "unset" and use default.
    if isinstance(raw, Mock):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _cfg_int(cfg: Any, name: str, default: int) -> int:
    """Read an int from cfg; tolerate MagicMock / bad types in tests."""
    if cfg is None:
        return default
    raw = getattr(cfg, name, default)
    if isinstance(raw, Mock):
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _handle_zero_grounded_quotes(
    *,
    episode_id: str,
    insights: List[str],
    total_grounded: int,
    cfg: Optional["config.Config"],
    pipeline_metrics: Optional[Any],
) -> None:
    """Warn, update metrics, and optionally fail when grounding required but empty."""
    if total_grounded > 0 or not insights:
        return
    if pipeline_metrics is not None and hasattr(
        pipeline_metrics, "gi_episodes_zero_grounded_when_required"
    ):
        pipeline_metrics.gi_episodes_zero_grounded_when_required += 1
    if pipeline_metrics is not None and hasattr(pipeline_metrics, "gi_grounding_degraded"):
        pipeline_metrics.gi_grounding_degraded = True
    logger.warning(
        "GIL: episode %s: gi_require_grounding is True but evidence produced 0 grounded "
        "quotes for %d insight(s)",
        episode_id,
        len(insights),
    )
    if cfg is not None and getattr(cfg, "gi_fail_on_missing_grounding", False):
        raise GILGroundingUnsatisfiedError(
            message=(
                f"No grounded quotes for episode {episode_id} " f"({len(insights)} insight(s))."
            ),
            provider="GIL/grounding",
            suggestion=(
                "Install .[ml] for local NLI, tune thresholds, or set "
                "gi_fail_on_missing_grounding: false."
            ),
        )


def _safe_iso_date(s: Optional[str]) -> str:
    """Return ISO date-time string; use placeholder if missing."""
    if s and s.strip():
        return s.strip()
    return "2020-01-01T00:00:00Z"


def _segment_text_cumulative_length(segments: List[Dict[str, Any]]) -> int:
    """Sum of len(segment['text']) over dict segments (same rules as timestamp mapping)."""
    pos = 0
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        text = seg.get("text") or ""
        pos += len(text)
    return pos


def _transcript_segments_alignment_delta(
    transcript: str, segments: List[Dict[str, Any]]
) -> Tuple[int, int, int]:
    """Return ``(len(transcript), reconstructed_segment_len, abs_delta)``."""
    recon = _segment_text_cumulative_length(segments)
    lt = len(transcript)
    return lt, recon, abs(lt - recon)


def _transcript_segments_aligned(
    transcript: str,
    segments: List[Dict[str, Any]],
    max_delta: int = SEGMENT_TRANSCRIPT_ALIGNMENT_MAX_DELTA,
) -> bool:
    """True if concatenated segment text length matches transcript within ``max_delta``."""
    _, _, delta = _transcript_segments_alignment_delta(transcript, segments)
    return delta <= max_delta


def _segment_char_spans(
    transcript: str, segments: List[Dict[str, Any]]
) -> Optional[List[Tuple[int, int, Dict[str, Any]]]]:
    """Return ``[(seg_start, seg_end, seg)]`` char spans for the segments.

    #974: when segments carry explicit ``char_start`` / ``char_end`` (the ad-free
    processing base, where each segment knows its exact position in the screenplay —
    markers and all), use those directly — a quote ``char_start`` maps to its segment
    with no heuristic. Otherwise fall back to cumulative ``len(text)`` positions and
    apply the legacy alignment guard (returning ``None`` when the transcript length and
    the summed segment text diverge by more than ``SEGMENT_TRANSCRIPT_ALIGNMENT_MAX_DELTA``,
    so callers skip an unreliable mapping on a reformatted transcript).
    """
    dict_segs = [s for s in segments if isinstance(s, dict)]
    if dict_segs and all("char_start" in s and "char_end" in s for s in dict_segs):
        return [(int(s["char_start"]), int(s["char_end"]), s) for s in dict_segs]

    spans: List[Tuple[int, int, Dict[str, Any]]] = []
    pos = 0
    for seg in dict_segs:
        text = seg.get("text") or ""
        spans.append((pos, pos + len(text), seg))
        pos += len(text)
    if not spans:
        return None
    if abs(len(transcript) - pos) > SEGMENT_TRANSCRIPT_ALIGNMENT_MAX_DELTA:
        return None
    return spans


def _char_range_to_ms(
    transcript: str,
    char_start: int,
    char_end: int,
    segments: List[Dict[str, Any]],
) -> Tuple[int, int]:
    """Map character span to (start_ms, end_ms) using transcription segments.

    Segments are expected to have "start" (seconds), "end" (seconds), "text" (str).
    Builds cumulative character offsets per segment and finds overlapping segment(s)
    for [char_start, char_end]; returns the time range of those segments in ms.

    Args:
        transcript: Full transcript text (used to validate segment text alignment).
        char_start: Quote start character offset (0-based).
        char_end: Quote end character offset (exclusive).
        segments: List of {"start": float, "end": float, "text": str}.

    Returns:
        (timestamp_start_ms, timestamp_end_ms). (0, 0) if no segments or no overlap.
    """
    if not segments or char_start >= char_end:
        return 0, 0
    spans = _segment_char_spans(transcript, segments)
    if not spans:
        return 0, 0
    # Find segments overlapping [char_start, char_end]; use first overlap for start, last for end
    start_ms = 0
    end_ms = 0
    first_set = False
    for seg_start, seg_end, seg in spans:
        if seg_end <= char_start or seg_start >= char_end:
            continue
        if not first_set:
            start_ms = int(float(seg.get("start", 0.0)) * 1000)
            first_set = True
        end_ms = int(float(seg.get("end", 0.0)) * 1000)
    return start_ms, end_ms


def _segment_speaker_label(seg: Dict[str, Any]) -> Optional[str]:
    """Return normalized speaker label from a segment dict if present.

    Prefer the human-readable ``speaker_label`` (e.g. ``Maya``, set by the
    diarization name-mapping) over the raw pyannote ``speaker`` id
    (``SPEAKER_00``) so GI Person nodes carry real names, not anonymous ids.
    """
    raw = seg.get("speaker_label")
    if raw is None or not str(raw).strip():
        raw = seg.get("speaker")
    if raw is None:
        raw = seg.get("speaker_id")
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


# What the voice behind a quote is, and what may be done with it.
#
# The diarization roster already types every voice, and GI has been ignoring it — which is why ad
# copy could be grounded as an insight and an anonymous voice could mint a Person node called
# "SPEAKER_09". Ad copy is *written* to be quotable ("If you play our games, you probably know
# there's something a bit different about them"), which makes it the perfect false insight.
#
#   commercial     an advertisement. NEVER grounded. There is no insight in an ad read.
#   unknown        a person we FAILED to name. Not surfaceable — and COUNTED, because a defect that
#                  costs nothing gets fixed by nobody.
#   unidentified   a person NOBODY names: the vox-pop in a narrated piece. Not surfaceable — an
#                  unattributed STANCE is not a stance, it is a floating opinion that nobody holds
#                  and nobody can disagree with. But it stays eligible for CONNECT: a fact is still
#                  a fact, and on Planet Money and The Daily the tape IS the story (36-40% of those
#                  episodes), so discarding it outright would gut them.
VOICE_TYPE_NEVER_GROUND = frozenset({"commercial"})
VOICE_TYPE_NOT_SURFACEABLE = frozenset({"commercial", "unknown", "unidentified"})


def _voice_type_for_char_range(
    transcript: str,
    char_start: int,
    char_end: int,
    segments: List[Dict[str, Any]],
) -> Optional[str]:
    """The ``voice_type`` of the voice speaking at ``char_start`` — ``None`` when it is a person."""
    if not segments or char_start >= char_end:
        return None
    spans = _segment_char_spans(transcript, segments)
    if not spans:
        return None
    for seg_start, seg_end, seg in spans:
        if seg_start <= char_start < seg_end:
            vt = seg.get("voice_type")
            return str(vt) if vt else None
    return None


def _resolve_quote_speaker(
    gq: Any,
    speaker_label: Optional[str],
    episode_id: str,
    transcript_text: Optional[str],
    transcript_segments: Optional[List[Dict[str, Any]]],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """``(person_id, friendly_name, voice_type)`` for a quote's speaker.

    A PERSON NODE MUST BE A PERSON, and "SPEAKER_09" is not one. We were minting a Person for every
    unresolved voice and hanging a SPOKEN_BY edge on it — 19% of the Person nodes in the shipped
    corpus were called SPEAKER_NN, and #1167 then filtered them back out of the trending/consensus
    surfaces downstream. That is a mop, not a gate, and it only worked because the id happened to be
    ugly: the moment those voices got a friendly name, it would have broken.

    The roster already knows the voice is not a person. So no Person node, no SPOKEN_BY edge, and
    the quote carries the friendly label instead ("Unidentified speaker") — the surface can name the
    speaker without the graph inventing someone.
    """
    voice_type = (
        _voice_type_for_char_range(
            transcript_text or "", gq.char_start, gq.char_end, transcript_segments
        )
        if transcript_segments
        else None
    )
    if voice_type:
        return None, friendly_voice_label(voice_type), voice_type
    if speaker_label:
        # Episode-scope the id for an unnamed voice (SPEAKER_00) so it can't merge across
        # episodes; a real, resolved name stays a global person id (#1b).
        return person_node_id(speaker_label, episode_id), None, None
    return None, None, None


def _quote_props(
    gq: Any,
    *,
    episode_id: str,
    transcript_ref: str,
    person_id: Optional[str],
    speaker_name: Optional[str],
    voice_type: Optional[str],
    ts_start: int,
    ts_end: int,
) -> Dict[str, Any]:
    """The Quote node's properties. ``speaker_name`` / ``speaker_voice_type`` appear ONLY for a
    voice that is not a person — so a surface can name the speaker ("Unidentified speaker") without
    the graph inventing one."""
    props: Dict[str, Any] = {
        "text": gq.text,
        "episode_id": episode_id,
        "speaker_id": person_id,
        "char_start": gq.char_start,
        "char_end": gq.char_end,
        "timestamp_start_ms": ts_start,
        "timestamp_end_ms": ts_end,
        "transcript_ref": transcript_ref,
    }
    if voice_type:
        props["speaker_voice_type"] = voice_type
    if speaker_name:
        props["speaker_name"] = speaker_name
    return props


def _apply_voice_flags(
    insight_props: Dict[str, Any],
    quotes: List[Any],
    transcript_text: Optional[str],
    transcript_segments: Optional[List[Dict[str, Any]]],
) -> int:
    """Stamp the speaking voice's type on an insight.

    Returns 1 when the insight is unsurfaceable BY OUR OWN FAULT (an ``unknown`` voice).

    The rule this exists to enforce: AN UNATTRIBUTED STANCE IS NOT A STANCE. It is a floating
    opinion that nobody holds and nobody can disagree with, so it cannot be SURFACEd whatever the
    classifier calls it.

    It stays eligible for CONNECT — a fact is still a fact. That distinction is not academic: on
    Planet Money and The Daily the tape IS the story (36-40% of those episodes), and discarding it
    outright would gut the narrated shows to protect them from a problem they do not have.
    """
    vt, is_defect = _voice_flags_for_insight(quotes, transcript_text, transcript_segments)
    if not vt:
        return 0
    insight_props["speaker_voice_type"] = vt
    if vt in VOICE_TYPE_NOT_SURFACEABLE:
        insight_props["surfaceable"] = False
    return 1 if is_defect else 0


def _log_voice_exclusions(episode_id: str, dropped_ads: int, by_defect: int) -> None:
    """A silent exclusion is a cost nobody sees, and a defect that costs nothing is never fixed."""
    if dropped_ads:
        logger.info(
            "  → GI refused %d quote(s) grounded inside an advertisement [%s]",
            dropped_ads,
            episode_id,
        )
    if by_defect:
        logger.warning(
            "  → %d insight(s) cannot be surfaced because a voice went unnamed and a name WAS "
            "available for it [%s] — this is our defect, and it costs us their words",
            by_defect,
            episode_id,
        )


def _is_advertisement_span(
    gq: Any,
    transcript_text: Optional[str],
    transcript_segments: Optional[List[Dict[str, Any]]],
) -> bool:
    """True when this span was spoken inside an ADVERT. Such a span is never evidence.

    Ad copy is written to be quotable — "if you play our games, you probably know there's something
    a bit different about them" — which makes it the most fluent, most confident false insight
    available. Refused here, where a span becomes a Quote, rather than filtered off a surface later
    and left in the corpus.
    """
    if not transcript_segments:
        return False
    vt = _voice_type_for_char_range(
        transcript_text or "", gq.char_start, gq.char_end, transcript_segments
    )
    return vt in VOICE_TYPE_NEVER_GROUND


def _gate_on_evidence(
    insight_specs: List[Tuple[str, str]],
    insight_quotes: List[List[Any]],
    *,
    cfg: Any,
    provider: Any,
    transcript_text: Optional[str],
    transcript_segments: Optional[List[Dict[str, Any]]],
    pipeline_metrics: Optional[Any],
) -> Tuple[List[Tuple[str, str]], List[List[Any]]]:
    """Run the value gate on the insight AND the evidence that grounds it.

    The specs and their quote lists are filtered TOGETHER — they are index-aligned everywhere
    downstream, and dropping one without the other would silently attach the wrong quote to the
    wrong insight.
    """
    from .value_gate import InsightEvidence, value_gate_keep_mask

    evidence: List[Optional[InsightEvidence]] = []
    for quotes in insight_quotes:
        first = next((q for q in quotes if isinstance(q, GroundedQuote)), None)
        if first is None:
            evidence.append(None)
            continue
        speaker: Optional[str] = None
        voice_type: Optional[str] = None
        if transcript_segments:
            voice_type = _voice_type_for_char_range(
                transcript_text or "", first.char_start, first.char_end, transcript_segments
            )
            speaker = _speaker_id_for_char_range(
                transcript_text or "", first.char_start, first.char_end, transcript_segments
            )
        evidence.append(InsightEvidence(quote=first.text, speaker=speaker, voice_type=voice_type))

    # A MASK, not a filtered list. The specs and their quote lists are index-aligned everywhere
    # downstream, so they are dropped TOGETHER, BY POSITION. Re-pairing by identity or by content
    # would mis-align an episode that says the same thing twice — CPython hands out one object for
    # two equal constant tuples — and a quote attached to the wrong insight is a fabricated
    # attribution, which is worse than the filler the gate exists to remove.
    keep = value_gate_keep_mask(
        insight_specs,
        provider=provider,
        cfg=cfg,
        pipeline_metrics=pipeline_metrics,
        evidence=evidence,
    )
    if all(keep):
        return insight_specs, insight_quotes

    specs_out = [spec for spec, k in zip(insight_specs, keep) if k]
    quotes_out = [quotes for quotes, k in zip(insight_quotes, keep) if k]
    return specs_out, quotes_out


def _voice_flags_for_insight(
    quotes: List[Any],
    transcript_text: Optional[str],
    transcript_segments: Optional[List[Dict[str, Any]]],
) -> Tuple[Optional[str], bool]:
    """``(voice_type, is_our_defect)`` for the voice behind an insight's first grounded quote.

    ``voice_type`` is None when a real, named person said it. ``is_our_defect`` is True only for
    ``unknown`` — a voice a name WAS available for and we failed to attach. That one is counted,
    because a defect that costs nothing gets fixed by nobody.
    """
    if not transcript_segments or not quotes:
        return None, False
    first = next((q for q in quotes if isinstance(q, GroundedQuote)), None)
    if first is None:
        return None, False
    vt = _voice_type_for_char_range(
        transcript_text or "", first.char_start, first.char_end, transcript_segments
    )
    return vt, vt == "unknown"


def _speaker_id_for_char_range(
    transcript: str,
    char_start: int,
    char_end: int,
    segments: List[Dict[str, Any]],
) -> Optional[str]:
    """Map quote character span to a speaker using optional diarization on segments.

    Segments follow the same layout as ``_char_range_to_ms`` (cumulative ``text``).
    When ``speaker`` or ``speaker_id`` is set on overlapping segments, prefers the
    segment containing ``char_start``, else the segment with the largest overlap.
    Returns ``None`` when labels are absent or transcript/segments do not align.
    """
    if not segments or char_start >= char_end:
        return None
    spans = _segment_char_spans(transcript, segments)
    if not spans:
        return None
    for seg_start, seg_end, seg in spans:
        if seg_start <= char_start < seg_end:
            sp = _segment_speaker_label(seg)
            if sp:
                return sp
            break
    best_overlap = 0
    best: Optional[str] = None
    for seg_start, seg_end, seg in spans:
        if seg_end <= char_start or seg_start >= char_end:
            continue
        ov = min(seg_end, char_end) - max(seg_start, char_start)
        sp = _segment_speaker_label(seg)
        if sp and ov > best_overlap:
            best_overlap = ov
            best = sp
    return best


def _attach_person_for_quote(
    nodes: list,
    edges: list,
    quote_id: str,
    speaker_label: Optional[str],
    person_id_value: Optional[str],
    persons_added: Set[str],
) -> None:
    """Add Person node and SPOKEN_BY (Quote -> Person) when diarization label exists."""
    if (
        not speaker_label
        or not str(speaker_label).strip()
        or not person_id_value
        or not str(person_id_value).strip()
    ):
        return
    raw = str(speaker_label).strip()
    pid = str(person_id_value).strip()
    if pid not in persons_added:
        nodes.append(
            {
                "id": pid,
                "type": "Person",
                "properties": {"name": raw},
            }
        )
        persons_added.add(pid)
    edges.append({"type": "SPOKEN_BY", "from": quote_id, "to": pid})


def _position_hint_from_timestamp_starts(
    timestamp_starts_ms: List[int],
    episode_duration_ms: Optional[int],
    *,
    duration_fallback_ms: Optional[int] = None,
    transcript_segments: Optional[List[Any]] = None,
) -> Optional[float]:
    """Mean quote start ms / episode duration, via RFC-097 4-step waterfall.

    Delegates to :func:`gi.position_hint.compute_position_hint`. Step ordering:

    1. ``episode_duration_ms`` from RSS ``<itunes:duration>``
    2. Last segment's ``end × 1000`` from ``transcript_segments`` (NEW, RFC-097
       chunk 5)
    3. ``duration_fallback_ms`` (typically ``max(Quote.timestamp_end_ms)``)
    4. Skip emission (returns ``None``)

    ``transcript_segments`` is optional; when omitted the function falls
    through to step 3 (preserves pre-RFC-097 behavior at call sites that
    don't yet thread segments).
    """
    from .position_hint import compute_position_hint

    value, _step = compute_position_hint(
        timestamp_starts_ms,
        episode_duration_ms,
        transcript_segments=transcript_segments,
        quote_end_fallback_ms=duration_fallback_ms,
    )
    return value


def _build_stub_artifact(
    episode_id: str,
    transcript_text: str,
    *,
    model_version: str,
    prompt_version: str,
    podcast_id: str,
    episode_title: str,
    date_str: str,
    transcript_ref: str,
    episode_duration_ms: Optional[int] = None,
    feed_id: Optional[str] = None,
    transcript_segments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build minimal stub artifact (one Episode, one Insight, one Quote, one SUPPORTED_BY)."""
    ep_node_id = episode_node_id(episode_id)
    stub_insight_text = _STUB_INSIGHT_TEXT
    insight_id = gil_insight_node_id(episode_id, 0, stub_insight_text)
    # char offsets must slice the transcript back to ``text`` exactly: char_end is
    # EXCLUSIVE (contracts.py) and char_start accounts for any leading whitespace the
    # ``.strip()`` drops — otherwise the Quote violates the offset invariant (Fault A).
    base_text = transcript_text or "No transcript."
    text_slice = base_text.strip()[:100]
    char_start = len(base_text) - len(base_text.lstrip())
    char_end = char_start + len(text_slice)
    quote_id = gil_quote_node_id(episode_id, 0, text_slice, char_start, char_end)
    # #876/#974 Fault B: when diarized segments are available, the stub Quote still carries
    # the speaker + timing of the segment its char range falls in (parity with the real path).
    quote_speaker_id: Optional[str] = None
    ts_start_ms = 0
    ts_end_ms = 0
    if transcript_segments:
        quote_speaker_id = _speaker_id_for_char_range(
            base_text, char_start, char_end, transcript_segments
        )
        ts_start_ms, ts_end_ms = _char_range_to_ms(
            base_text, char_start, char_end, transcript_segments
        )

    ep_props: Dict[str, Any] = {
        "podcast_id": podcast_id,
        "title": episode_title,
        "publish_date": date_str,
    }
    if episode_duration_ms is not None and episode_duration_ms > 0:
        ep_props["duration_ms"] = int(episode_duration_ms)
    # #658 — Episode nodes carry the stable feed identifier so the
    # graph Feed chip can filter by feed across the merged corpus.
    if feed_id is not None:
        ep_props["feed_id"] = str(feed_id).strip()

    nodes: list = [
        {
            "id": ep_node_id,
            "type": "Episode",
            "properties": ep_props,
        },
        {
            "id": insight_id,
            "type": "Insight",
            "properties": {
                "text": _STUB_INSIGHT_TEXT,
                "episode_id": episode_id,
                "grounded": True,
                "insight_type": "unknown",
                # RFC-097 chunk 9 (ADR-101): required in v3.0 strict.
                "position_hint": 0.5,
            },
        },
        {
            "id": quote_id,
            "type": "Quote",
            "properties": {
                "text": text_slice,
                "episode_id": episode_id,
                "speaker_id": quote_speaker_id,
                "char_start": char_start,
                "char_end": char_end,
                "timestamp_start_ms": ts_start_ms,
                "timestamp_end_ms": ts_end_ms,
                "transcript_ref": transcript_ref,
            },
        },
    ]
    edges: list = [
        {"type": "HAS_INSIGHT", "from": ep_node_id, "to": insight_id},
        {"type": "SUPPORTED_BY", "from": insight_id, "to": quote_id},
    ]
    return {
        "schema_version": "3.0",  # RFC-097 chunk 9: v3.0 GI emit
        "model_version": model_version,
        "prompt_version": prompt_version,
        "episode_id": episode_id,
        "nodes": nodes,
        "edges": edges,
    }


#: RFC-097/gi.schema.json v3.0: legal Insight.insight_type values.
_INSIGHT_TYPE_ALLOWED = frozenset({"claim", "recommendation", "observation", "question", "unknown"})

#: Legacy upstream synonyms (megabundle/extraction-bundled prompts pre-RFC-097
#: emitted ``claim | fact | opinion`` per ``prompting/megabundle.py``). Mapped
#: to the schema-valid vocab so prefilled artifacts stay queryable post-v3.0.
_INSIGHT_TYPE_LEGACY_SYNONYMS = {
    "fact": "claim",
    "opinion": "observation",
}


def _normalize_insight_type(raw: Any) -> str:
    if isinstance(raw, str):
        k = raw.strip().lower()
        if k in _INSIGHT_TYPE_ALLOWED:
            return k
        if k in _INSIGHT_TYPE_LEGACY_SYNONYMS:
            return _INSIGHT_TYPE_LEGACY_SYNONYMS[k]
    return "unknown"


def _classify_when_unknown(text: str, itype: str) -> str:
    """Rule-classify *text* when *itype* is ``"unknown"``; pass through otherwise.

    Provider-supplied :func:`generate_insights` returns of ``List[str]`` flow
    in as ``(text, "unknown")``. The classifier inspects the text and assigns
    one of ``claim | recommendation | observation | question | unknown``
    (RFC-072 §2a / RFC-097 v3.0). When the provider already supplied a
    valid type (dict return), the existing type is preserved.

    Defensive: never returns a value outside :data:`_INSIGHT_TYPE_ALLOWED`.
    """
    if itype != "unknown":
        return itype
    from .insight_type_classifier import classify_insight_type

    return classify_insight_type(text)


def _parse_insight_item(item: Any) -> Optional[Tuple[str, str]]:
    if isinstance(item, str):
        s = item.strip()
        return (s, "unknown") if s else None
    if isinstance(item, dict):
        text = item.get("text") or item.get("insight")
        if isinstance(text, str) and text.strip():
            return (text.strip(), _normalize_insight_type(item.get("insight_type")))
    return None


def _resolve_insight_specs(
    transcript_text: str,
    cfg: Optional["config.Config"],
    insight_texts: Optional[List[str]] = None,
    insight_provider: Optional[Any] = None,
    episode_title: Optional[str] = None,
    pipeline_metrics: Optional[Any] = None,
) -> List[Tuple[str, str]]:
    """Resolve (insight text, insight_type) pairs for GIL.

    Order: use insight_texts if non-empty (type ``unknown``); else provider
    ``generate_insights`` (strings or dicts with ``text`` / ``insight_type``); else stub.
    """
    max_insights = config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX
    if cfg is not None:
        max_insights = getattr(
            cfg, "gi_max_insights", config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX
        )

    if insight_texts:
        resolved = [(s.strip(), "unknown") for s in insight_texts if (s and s.strip())][
            :max_insights
        ]
        if resolved:
            # RFC-097 v3.0 chunk-5: classify the bullet-derived strings — they
            # arrive as ``"unknown"`` (the input layer has no type signal) so
            # the rule-based classifier is the only opportunity to assign a
            # meaningful type before the Insight node is built.
            return [(t, _classify_when_unknown(t, k)) for t, k in resolved]

    source = "stub"
    if cfg is not None:
        source = getattr(cfg, "gi_insight_source", "stub")

    if source == "provider" and insight_provider is not None:
        gen = getattr(insight_provider, "generate_insights", None)
        if callable(gen):
            try:
                from .chunked_extraction import generate_chunked

                out = generate_chunked(
                    gen,
                    transcript_text or "",
                    episode_title=episode_title,
                    max_insights=max_insights,
                    chunk_chars=int(getattr(cfg, "gi_insight_chunk_chars", 0) or 0),
                    dedupe_threshold=float(
                        getattr(cfg, "gi_insight_dedupe_threshold", 0.75) or 0.75
                    ),
                    pipeline_metrics=pipeline_metrics,
                )
                if isinstance(out, list):
                    resolved_specs: List[Tuple[str, str]] = []
                    for item in out:
                        p = _parse_insight_item(item)
                        if p:
                            resolved_specs.append(p)
                    # The cap is per PASS, not per episode. Truncating the merged list to
                    # gi_max_insights would clip a 3-pass extraction (56 insights) straight back to
                    # 50 and silently erase the whole gain of chunking.
                    from .chunked_extraction import plan_chunks

                    passes = plan_chunks(
                        transcript_text or "",
                        int(getattr(cfg, "gi_insight_chunk_chars", 0) or 0),
                    )
                    resolved_specs = resolved_specs[: max_insights * passes]
                    if resolved_specs:
                        # THE VALUE GATE USED TO RUN HERE, and that was the bug. It ran BEFORE
                        # grounding, so the quotes did not exist yet — it graded a bare sentence
                        # while its own rubric asked for "a position a NAMED PERSON took", "a
                        # disagreement BETWEEN SPEAKERS" and "an AD read", none of which it could
                        # see. It now runs in `build_artifact`, after the evidence exists
                        # (ADR-110's lesson, one layer up).
                        #
                        # RFC-097 v3.0 chunk-5: classify any unknown-typed specs (providers
                        # returning ``List[str]`` flow in as ``"unknown"``; structured dict items
                        # keep their provider-supplied type).
                        return [(t, _classify_when_unknown(t, k)) for t, k in resolved_specs]
            except Exception as e:
                logger.debug(
                    "generate_insights failed, falling back to stub: %s",
                    e,
                    exc_info=True,
                )

    # Stub fallback — genuinely no signal; ``"unknown"`` is correct here.
    return [(_STUB_INSIGHT_TEXT, "unknown")]


def build_artifact(
    episode_id: str,
    transcript_text: str,
    *,
    model_version: str = "stub",
    prompt_version: str = "v1",
    podcast_id: Optional[str] = None,
    episode_title: Optional[str] = None,
    publish_date: Optional[str] = None,
    transcript_ref: str = "transcript.txt",
    transcript_segments: Optional[List[Dict[str, Any]]] = None,
    cfg: Optional["config.Config"] = None,
    insight_texts: Optional[List[str]] = None,
    insight_provider: Optional[Any] = None,
    quote_extraction_provider: Optional[Any] = None,
    entailment_provider: Optional[Any] = None,
    summary_provider: Optional[Any] = None,
    pipeline_metrics: Optional[Any] = None,
    gil_created_evidence_providers: Optional[List[Any]] = None,
    topic_labels: Optional[List[str]] = None,
    episode_duration_ms: Optional[int] = None,
    prefilled_insights: Optional[List[Dict[str, Any]]] = None,
    feed_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a GIL artifact for one episode.

    Insight texts come from insight_texts (e.g. summary bullets), from
    insight_provider.generate_insights() when gi_insight_source=provider, or
    a single stub. For each insight, provider-based QA + NLI finds grounded
    quotes; artifact has one Insight node per insight and their
    Quote nodes + SUPPORTED_BY edges.

    When transcript_segments is provided (from transcription with segments),
    Quote nodes get precise timestamp_start_ms and timestamp_end_ms (FR2.2).

    Args:
        episode_id: Stable episode key (RSS GUID family). Episode node id is
            ``episode:{episode_id}``; do not pass a value that already includes the
            ``episode:`` prefix.
        transcript_text: Full transcript text (used for QA context and quote spans).
        model_version: Fallback model id when ``cfg`` is omitted; when ``cfg`` is set,
            ``gi.json`` ``model_version`` is derived from ``gi_insight_source`` and the
            summarization / insight provider (see ``gi.provenance``).
        prompt_version: Prompt version tag.
        podcast_id: Optional podcast node ID for Episode.podcast_id.
        episode_title: Optional title for Episode node.
        publish_date: Optional ISO date-time for Episode node.
        transcript_ref: Reference string for Quote.transcript_ref.
        transcript_segments: Optional list of {"start", "end", "text"} from transcription.
        cfg: Optional config; when set and gi_require_grounding, use evidence stack.
        insight_texts: Optional precomputed list of insight strings (e.g. summary bullets).
        insight_provider: Optional provider with generate_insights() when source=provider.
        quote_extraction_provider: Optional provider with extract_quotes() for GIL QA.
        entailment_provider: Optional provider with score_entailment() for GIL NLI.
        summary_provider: When quote/entail providers are omitted, reused if cfg matches
            summary_provider (see ``create_gil_evidence_providers``).
        pipeline_metrics: Optional metrics; when set, evidence path counters are updated.
        gil_created_evidence_providers: When set, instantiated evidence providers (not equal
            to ``summary_provider``) are appended for caller cleanup (e.g. ``.cleanup()``).
        topic_labels: Optional episode topic labels (e.g. from summary bullets); creates
            Topic nodes and ABOUT edges (Insight → Topic) aligned with KG ``topic:{slug}`` ids.
        episode_duration_ms: Optional episode length in ms for ``Episode.duration_ms`` and
            ``Insight.position_hint``; omit when unknown.

    Returns:
        Dict with schema_version, model_version, prompt_version, episode_id, nodes, edges.
    """
    pid = podcast_id or "podcast:unknown"
    title = (episode_title or "Episode").strip() or "Episode"
    date_str = _safe_iso_date(publish_date)

    # #643: when llm_pipeline_mode=mega_bundled/extraction_bundled has already
    # produced insights, short-circuit provider dispatch entirely.
    insight_specs: List[Tuple[str, str]] = []
    if prefilled_insights:
        max_insights_pref = (
            getattr(cfg, "gi_max_insights", config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX)
            if cfg is not None
            else config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX
        )
        for item in prefilled_insights:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            # RFC-097 v3.0 + gi.schema.json: vocab is
            # {claim, recommendation, observation, question, unknown}.
            # Falls back to "claim" when the prefilled item carries no type
            # (matches the legacy default for prefilled insights).
            raw_itype = item.get("insight_type")
            itype = _normalize_insight_type(raw_itype) if raw_itype else "claim"
            insight_specs.append((text, itype))
            if len(insight_specs) >= max_insights_pref:
                break
    if not insight_specs:
        insight_specs = _resolve_insight_specs(
            transcript_text=transcript_text or "",
            cfg=cfg,
            insight_texts=insight_texts,
            insight_provider=insight_provider,
            episode_title=episode_title or title,
            pipeline_metrics=pipeline_metrics,
        )

    # #652 Part B — ad + dialogue filters applied post-resolution.
    if insight_specs:
        insight_specs = _apply_gi_insight_filters(insight_specs, pipeline_metrics)

    artifact_model_version = model_version
    if cfg is not None:
        raw_gi_src = getattr(cfg, "gi_insight_source", "stub")
        if isinstance(raw_gi_src, str):
            gi_src_norm = raw_gi_src.strip().lower() or "stub"
        else:
            gi_src_norm = "stub"
        lineage_provider = summary_provider or insight_provider
        artifact_model_version = resolve_gil_artifact_model_version(
            cfg,
            lineage_provider,
            gi_insight_source=gi_src_norm,
        )

    use_evidence_stack = (
        cfg is not None
        and getattr(cfg, "generate_gi", False)
        and getattr(cfg, "gi_require_grounding", True)
        and (transcript_text or "").strip()
    )

    if use_evidence_stack and insight_specs:
        assert cfg is not None
        try:
            from .deps import create_gil_evidence_providers
            from .grounding import NLI_ENTAILMENT_MIN, QA_SCORE_MIN

            q_prov = quote_extraction_provider
            e_prov = entailment_provider
            if q_prov is None or e_prov is None:
                q_new, e_new = create_gil_evidence_providers(cfg, summary_provider=summary_provider)
                if quote_extraction_provider is None:
                    q_prov = q_new
                    if (
                        gil_created_evidence_providers is not None
                        and q_prov is not summary_provider
                        and hasattr(q_prov, "cleanup")
                    ):
                        gil_created_evidence_providers.append(q_prov)
                else:
                    q_prov = quote_extraction_provider
                if entailment_provider is None:
                    e_prov = e_new
                    if (
                        gil_created_evidence_providers is not None
                        and e_prov is not summary_provider
                        and e_prov is not q_prov
                        and hasattr(e_prov, "cleanup")
                    ):
                        gil_created_evidence_providers.append(e_prov)
                else:
                    e_prov = entailment_provider
            else:
                q_prov = quote_extraction_provider
                e_prov = entailment_provider

            _er = max(0, _cfg_int(cfg, "gi_evidence_extract_retries", 0))
            qa_min = _cfg_float(cfg, "gi_qa_score_min", QA_SCORE_MIN)
            nli_min = _cfg_float(cfg, "gi_nli_entailment_min", NLI_ENTAILMENT_MIN)

            # #698 Layer A — bundled extract_quotes pre-fetch (helper above).
            prefetched_by_idx = _maybe_prefetch_bundled_candidates(
                cfg=cfg,
                quote_extraction_provider=q_prov,
                transcript=transcript_text.strip(),
                insight_texts=[t for t, _ in insight_specs],
                pipeline_metrics=pipeline_metrics,
            )
            # #698 Layer B — bundled score_entailment dispatch.
            insight_quotes = _ground_insights_dispatch(
                cfg=cfg,
                insight_specs=insight_specs,
                transcript=transcript_text.strip(),
                quote_extraction_provider=q_prov,
                entailment_provider=e_prov,
                qa_score_min=qa_min,
                nli_entailment_min=nli_min,
                extract_retries=_er,
                pipeline_metrics=pipeline_metrics,
                prefetched_by_idx=prefetched_by_idx,
            )
            # NOW the gate can see what it is grading. The extractor cannot be made selective by
            # prompting — across three prompt variants the CORE count barely moved (13.3 / 10.3 /
            # 12.0 per episode) while filler tracked whatever the prompt encouraged — so filler is
            # trimmed by a judge. But the judge was being handed a bare sentence, BEFORE grounding,
            # and asked to spot "a position a NAMED PERSON took" and "an AD read". It ran blind.
            #
            # Here it reads the insight together with the verbatim quote that supports it, who said
            # it, and what kind of voice that is — and, decisively, it can now see when NOTHING
            # supports it, which is the strongest FILLER signal there is.
            insight_specs, insight_quotes = _gate_on_evidence(
                insight_specs,
                insight_quotes,
                cfg=cfg,
                provider=insight_provider,
                transcript_text=transcript_text,
                transcript_segments=transcript_segments,
                pipeline_metrics=pipeline_metrics,
            )

            total_grounded = sum(len(q) for q in insight_quotes)
            _handle_zero_grounded_quotes(
                episode_id=episode_id,
                insights=[t for t, _ in insight_specs],
                total_grounded=total_grounded,
                cfg=cfg,
                pipeline_metrics=pipeline_metrics,
            )
            if pipeline_metrics is not None and hasattr(
                pipeline_metrics, "gi_evidence_stack_completed"
            ):
                pipeline_metrics.gi_evidence_stack_completed += 1
            return _artifact_from_multi_insight(
                episode_id=episode_id,
                insight_specs=insight_specs,
                insight_quotes_list=insight_quotes,
                model_version=artifact_model_version,
                prompt_version=prompt_version,
                podcast_id=pid,
                episode_title=title,
                date_str=date_str,
                transcript_ref=transcript_ref,
                transcript_text=transcript_text or "",
                transcript_segments=transcript_segments,
                topic_labels=topic_labels,
                episode_duration_ms=episode_duration_ms,
                feed_id=feed_id,
            )
        except GILGroundingUnsatisfiedError:
            raise
        except Exception as e:
            _record_stub_fallback(pipeline_metrics, e)

    # Single stub path (no evidence stack or fallback)
    if len(insight_specs) == 1 and insight_specs[0][0] == _STUB_INSIGHT_TEXT:
        return _build_stub_artifact(
            episode_id=episode_id,
            transcript_text=transcript_text,
            model_version=artifact_model_version,
            prompt_version=prompt_version,
            podcast_id=pid,
            episode_title=title,
            date_str=date_str,
            transcript_ref=transcript_ref,
            episode_duration_ms=episode_duration_ms,
            feed_id=feed_id,
            transcript_segments=transcript_segments,
        )

    # Multiple insights but evidence stack failed: still emit multi-insight artifact
    # with no quotes (grounded=False per insight)
    try:
        return _artifact_from_multi_insight(
            episode_id=episode_id,
            insight_specs=insight_specs,
            insight_quotes_list=[[]] * len(insight_specs),
            model_version=artifact_model_version,
            prompt_version=prompt_version,
            podcast_id=pid,
            episode_title=title,
            date_str=date_str,
            transcript_ref=transcript_ref,
            transcript_text=transcript_text or "",
            transcript_segments=transcript_segments,
            topic_labels=topic_labels,
            episode_duration_ms=episode_duration_ms,
            feed_id=feed_id,
        )
    except Exception:
        return _build_stub_artifact(
            episode_id=episode_id,
            transcript_text=transcript_text,
            model_version=artifact_model_version,
            prompt_version=prompt_version,
            podcast_id=pid,
            episode_title=title,
            date_str=date_str,
            transcript_ref=transcript_ref,
            episode_duration_ms=episode_duration_ms,
            feed_id=feed_id,
            transcript_segments=transcript_segments,
        )


def _speaker_for_insight(
    quotes: Sequence[Any],
    transcript_text: str,
    transcript_segments: Optional[List[Dict[str, Any]]],
    named_turns: Sequence[Tuple[int, str]],
) -> Optional[str]:
    """Who said this insight — the speaker of the turn its first grounded quote sits in.

    An insight is a claim a PERSON made, and its worth depends on who made it: a host summarising a
    deal is a headline, while the guest expert taking a position is the reason the episode exists.
    Without the speaker the two are indistinguishable, and a "stance" is unfalsifiable — a position
    needs an owner.

    Prefers diarized segments when they exist; otherwise reads the name straight out of the
    transcript's own speaker markers. An ungrounded insight has no quote and therefore, correctly,
    no speaker.
    """
    for gq in quotes:
        if not isinstance(gq, GroundedQuote):
            continue
        who: Optional[str] = None
        if transcript_segments:
            who = _speaker_id_for_char_range(
                transcript_text, gq.char_start, gq.char_end, transcript_segments
            )
        elif named_turns:
            who = speaker_for_char(gq.char_start, named_turns)
        if who:
            return who
    return None


def _artifact_from_multi_insight(
    episode_id: str,
    insight_specs: List[Tuple[str, str]],
    insight_quotes_list: List[List[GroundedQuote]],
    *,
    model_version: str,
    prompt_version: str,
    podcast_id: str,
    episode_title: str,
    date_str: str,
    transcript_ref: str,
    transcript_text: Optional[str] = None,
    transcript_segments: Optional[List[Dict[str, Any]]] = None,
    topic_labels: Optional[List[str]] = None,
    episode_duration_ms: Optional[int] = None,
    about_edge_top_k: Optional[int] = None,
    about_edge_floor: Optional[float] = None,
    about_edge_encoder: Optional[Any] = None,
    feed_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build artifact from Episode + N Insights + their grounded quote lists.

    When transcript_text and transcript_segments are provided, Quote nodes get
    precise timestamp_start_ms and timestamp_end_ms (FR2.2).
    """
    ep_node_id = episode_node_id(episode_id)
    ep_props: Dict[str, Any] = {
        "podcast_id": podcast_id,
        "title": episode_title,
        "publish_date": date_str,
    }
    if episode_duration_ms is not None and episode_duration_ms > 0:
        ep_props["duration_ms"] = int(episode_duration_ms)
    # #658 — Episode nodes carry the stable feed identifier so the
    # graph Feed chip can filter by feed across the merged corpus.
    if feed_id is not None:
        ep_props["feed_id"] = str(feed_id).strip()

    nodes: list = [
        {
            "id": ep_node_id,
            "type": "Episode",
            "properties": ep_props,
        },
    ]
    edges: list = []
    quote_global_idx = 0
    dropped_ad_quotes = 0  # spans refused because an advertisement is never evidence
    unsurfaceable_by_defect = 0  # insights we cannot surface because WE failed to name a voice
    persons_added: Set[str] = set()
    use_segments_raw = bool(
        transcript_text and transcript_segments and len(transcript_segments) > 0
    )
    if use_segments_raw and transcript_segments is not None:
        # #974: segments carrying explicit char_start/char_end (the ad-free processing
        # base) know their exact position in the screenplay — markers and all — so a
        # quote maps to its segment without any cumulative-length check. Only the legacy
        # path (plain segments, no offsets) needs the #545 alignment guard below.
        has_offsets = all(
            isinstance(s, dict) and "char_start" in s and "char_end" in s
            for s in transcript_segments
        )
        if has_offsets:
            use_segments = True
        else:
            aligned = _transcript_segments_aligned(transcript_text or "", transcript_segments)
            if not aligned:
                lt, recon, delta = _transcript_segments_alignment_delta(
                    transcript_text or "", transcript_segments
                )
                logger.warning(
                    "GIL: episode %s transcript vs segment text length mismatch (%s): "
                    "len(transcript)=%d concatenated_segment_len=%d abs_delta=%d (max_delta=%d); "
                    "skipping segment-based quote timestamps and segment speakers (issue #545). "
                    "Re-run with the #974 ad-free base to map exactly.",
                    episode_id,
                    transcript_ref or "",
                    lt,
                    recon,
                    delta,
                    SEGMENT_TRANSCRIPT_ALIGNMENT_MAX_DELTA,
                )
            use_segments = aligned
    else:
        use_segments = False

    # Fallback speaker source: the diarized transcript names its own speakers ("Kevin Roose: ...").
    # Segments were the pipeline's ONLY speaker path, and no profile produces them
    # (backfill_transcript_segments is false everywhere, and enabling it forces a re-transcription),
    # so every quote has shipped with speaker_id=None and no insight has ever known who said it —
    # while the name sat in the transcript, unread.
    named_turns: List[Tuple[int, str]] = []
    if not use_segments and (transcript_text or "").strip():
        named_turns = build_unverified_named_turns(transcript_text or "")

    topic_node_specs = _dedupe_topic_node_specs(topic_labels)
    for tid, display_label in topic_node_specs:
        nodes.append(
            {
                "id": tid,
                "type": "Topic",
                "properties": {"label": display_label},
            }
        )

    # #664: rank insight→topic ABOUT edges semantically (top-K + floor) instead
    # of emitting the full insights × topics cross-product.
    about_edges_per_insight = _rank_about_edges_for_insights(
        [t for t, _ in insight_specs],
        topic_node_specs,
        top_k=about_edge_top_k,
        floor=about_edge_floor,
        encoder=about_edge_encoder,
    )

    # Pad so we have one quote list per insight
    while len(insight_quotes_list) < len(insight_specs):
        insight_quotes_list.append([])

    for idx, ((it_text, it_type), quotes) in enumerate(zip(insight_specs, insight_quotes_list)):
        insight_id = gil_insight_node_id(episode_id, idx, it_text)
        insight_confidence = _insight_confidence_from_quotes(quotes)
        timestamp_starts_ms: List[int] = []
        timestamp_ends_ms: List[int] = []
        insight_props: Dict[str, Any] = {
            "text": it_text,
            "episode_id": episode_id,
            "grounded": len(quotes) > 0,
            "insight_type": it_type,
        }
        for gq in quotes:
            if not isinstance(gq, GroundedQuote):
                continue
            ts_start, ts_end_q = 0, 0
            if use_segments and transcript_segments:
                ts_start, ts_end_q = _char_range_to_ms(
                    transcript_text or "",
                    gq.char_start,
                    gq.char_end,
                    transcript_segments,
                )
            if ts_start > 0:
                timestamp_starts_ms.append(ts_start)
            if ts_end_q > 0:
                timestamp_ends_ms.append(ts_end_q)
        duration_fb = max(timestamp_ends_ms) if timestamp_ends_ms else None
        ph = _position_hint_from_timestamp_starts(
            timestamp_starts_ms,
            episode_duration_ms,
            duration_fallback_ms=duration_fb,
            transcript_segments=transcript_segments,
        )
        # RFC-097 chunk 9 (ADR-101): position_hint is required in v3.0 strict.
        # When the 4-step waterfall (RFC-072 §3) returns None for degenerate
        # inputs (no quote timestamps, no char positions, single insight),
        # fall back to 0.5 — the midpoint default for "we don't know where
        # in the episode this lives".
        insight_props["position_hint"] = ph if ph is not None else 0.5

        speaker_for_insight = _speaker_for_insight(
            quotes,
            transcript_text or "",
            transcript_segments if use_segments else None,
            named_turns,
        )
        if speaker_for_insight:
            insight_props["speaker"] = speaker_for_insight

        # What KIND of voice said it — so routing can honour the one rule that matters:
        # AN UNATTRIBUTED STANCE IS NOT A STANCE. It is a floating opinion that nobody holds and
        # nobody can disagree with, so it cannot be SURFACEd, whatever the classifier calls it.
        #
        # It stays eligible for CONNECT: a fact is still a fact. That distinction is not academic —
        # on Planet Money and The Daily the tape IS the story (36-40% of those episodes), and
        # discarding it outright would gut the narrated shows to protect against a problem they do
        # not have.
        unsurfaceable_by_defect += _apply_voice_flags(
            insight_props, quotes, transcript_text, transcript_segments if use_segments else None
        )

        insight_node: Dict[str, Any] = {
            "id": insight_id,
            "type": "Insight",
            "properties": insight_props,
        }
        if insight_confidence is not None:
            insight_node["confidence"] = float(insight_confidence)
        nodes.append(insight_node)
        edges.append({"type": "HAS_INSIGHT", "from": ep_node_id, "to": insight_id})
        for tid, confidence in about_edges_per_insight[idx]:
            # Clamp to schema range [0, 1]; cosine is theoretically [-1, 1] but
            # the floor filter already drops low values in practice.
            conf = max(0.0, min(1.0, float(confidence)))
            edges.append(
                {
                    "type": "ABOUT",
                    "from": insight_id,
                    "to": tid,
                    "properties": {"confidence": round(conf, 4)},
                }
            )
        for gq in quotes:
            if not isinstance(gq, GroundedQuote):
                continue

            if _is_advertisement_span(
                gq, transcript_text, transcript_segments if use_segments else None
            ):
                dropped_ad_quotes += 1
                continue

            quote_id = gil_quote_node_id(
                episode_id,
                quote_global_idx,
                gq.text,
                gq.char_start,
                gq.char_end,
            )
            quote_global_idx += 1
            ts_start, ts_end = 0, 0
            speaker_label: Optional[str] = None
            person_id_for_quote: Optional[str] = None
            if use_segments and transcript_segments:
                ts_start, ts_end = _char_range_to_ms(
                    transcript_text or "",
                    gq.char_start,
                    gq.char_end,
                    transcript_segments,
                )
                speaker_label = _speaker_id_for_char_range(
                    transcript_text or "",
                    gq.char_start,
                    gq.char_end,
                    transcript_segments,
                )
            elif named_turns:
                # No segments — but the diarized transcript names its speakers in plain text, and
                # the quote knows its char offset. Segments are the ONLY speaker path the pipeline
                # had, and no profile generates them (backfill_transcript_segments is false
                # everywhere, and turning it on forces a re-transcription), so every insight has
                # shipped without knowing who said it. Read the name out of the transcript instead.
                speaker_label = speaker_for_char(gq.char_start, named_turns)

            person_id_for_quote, quote_speaker_name, quote_voice_type = _resolve_quote_speaker(
                gq,
                speaker_label,
                episode_id,
                transcript_text,
                transcript_segments if use_segments else None,
            )
            if quote_voice_type:
                speaker_label = None  # not a person — nothing to mint
            nodes.append(
                {
                    "id": quote_id,
                    "type": "Quote",
                    "properties": _quote_props(
                        gq,
                        episode_id=episode_id,
                        transcript_ref=transcript_ref,
                        person_id=person_id_for_quote,
                        speaker_name=quote_speaker_name,
                        voice_type=quote_voice_type,
                        ts_start=ts_start,
                        ts_end=ts_end,
                    ),
                }
            )
            edges.append({"type": "SUPPORTED_BY", "from": insight_id, "to": quote_id})
            _attach_person_for_quote(
                nodes,
                edges,
                quote_id,
                speaker_label,
                person_id_for_quote,
                persons_added,
            )

    artifact = {
        "schema_version": "3.0",  # RFC-097 chunk 9: v3.0 GI emit
        "model_version": model_version,
        "prompt_version": prompt_version,
        "episode_id": episode_id,
        "nodes": nodes,
        "edges": edges,
    }

    _log_voice_exclusions(episode_id, dropped_ad_quotes, unsurfaceable_by_defect)

    # The stage checks its own output. Every GI bug in this arc shipped silently — insights with no
    # quotes, quotes with no speaker, a speaker who never holds the mic — and every one of them
    # reported success. Logging, not raising: a disconnected wire must be loud, but an episode that
    # already paid for transcription should still emit what it has.
    log_artifact_invariants(artifact, transcript_text, named_turns)
    return artifact

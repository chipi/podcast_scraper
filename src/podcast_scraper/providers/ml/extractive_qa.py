"""Extractive QA pipeline for GIL evidence stack (Issue #435).

Lazy-loads a transformers question-answering pipeline; returns answer spans
with character offsets and scores. Used for "What evidence supports: {insight}?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

from ...cache import get_transformers_cache_dir
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# One pipeline per (resolved_model_id, device_key) within the process (Issue: worker procs still
# load separately).
_qa_pipelines: Dict[Tuple[str, str], object] = {}


def _safe_score_float(value: Any, fallback: float = 0.0) -> float:
    """Convert pipeline score (tensor or scalar) to float; avoid .item() on meta tensors."""
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except RuntimeError as e:
            if "meta" in str(e).lower() or "item" in str(e).lower():
                logger.debug("QA score on meta device: %s", e)
                return fallback
            raise
    return fallback


@dataclass
class QASpan:
    """Single extractive QA result: answer text and character offsets."""

    answer: str
    start: int
    end: int
    score: float


def _get_device(device: Optional[str]) -> str:
    """Resolve device for the HF QA pipeline.

    MPS is avoided: auto device selection used to pick ``mps``, but the QA stack
    can end up on ``meta`` or hit unsupported ops; CUDA when available, else CPU.
    """
    if device is not None and device.strip():
        dev = device.strip().lower()
        if dev == "mps":
            return "cpu"
        return dev
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def load_qa_pipeline(
    model_id: str,
    device: Optional[str] = None,
) -> object:
    """Load HuggingFace QA pipeline.

    Args:
        model_id: Alias (e.g. roberta-squad2) or full HF ID.
        device: Device (cpu, cuda, mps) or None for auto.

    Returns:
        transformers Pipeline for question-answering.
    """
    from transformers import pipeline

    resolved = ModelRegistry.resolve_evidence_model_id(model_id)
    dev = _get_device(device)
    logger.info("Loading extractive QA model %s on %s", resolved, dev)
    # QA pipeline: device -1 = CPU, 0 = CUDA; MPS often unsupported for QA
    pipe_device = 0 if dev == "cuda" else -1
    cache_dir = str(get_transformers_cache_dir().resolve())
    # Match summarizer / model_loader: no hub access at load time (preload or
    # make preload-ml-models must populate the cache).
    model_kw = {"local_files_only": True, "cache_dir": cache_dir}
    pipe = pipeline(
        "question-answering",
        model=resolved,
        device=pipe_device,
        model_kwargs=model_kw,
    )
    return pipe


def get_qa_pipeline(
    model_id: str,
    device: Optional[str] = None,
) -> object:
    """Return cached QA pipeline or load and cache it (lazy, keyed by model + device)."""
    resolved = ModelRegistry.resolve_evidence_model_id(model_id)
    dev = _get_device(device)
    key = (resolved, dev)
    if key not in _qa_pipelines:
        _qa_pipelines[key] = load_qa_pipeline(model_id, device=device)
    return _qa_pipelines[key]


def _qa_pipeline_call(pipe_fn: Callable[..., Dict[str, Any]], question: str, ctx: str) -> QASpan:
    out = pipe_fn(question=question, context=ctx, max_answer_len=512)
    return QASpan(
        answer=out["answer"],
        start=out["start"],
        end=out["end"],
        score=_safe_score_float(out["score"]),
    )


def _iter_context_windows(
    context: str,
    window_chars: int,
    overlap_chars: int,
) -> List[Tuple[int, int, str]]:
    """Split context into overlapping slices; returns (global_start, global_end, slice)."""
    n = len(context)
    if window_chars <= 0 or n <= window_chars:
        return [(0, n, context)]
    step = max(1, window_chars - overlap_chars)
    out: List[Tuple[int, int, str]] = []
    start = 0
    while start < n:
        end = min(n, start + window_chars)
        out.append((start, end, context[start:end]))
        if end >= n:
            break
        start += step
    return out


def answer(
    context: str,
    question: str,
    model_id: str,
    device: Optional[str] = None,
    *,
    window_chars: int = 0,
    window_overlap_chars: int = 250,
) -> QASpan:
    """Run extractive QA; optionally scan overlapping windows for long transcripts.

    When ``window_chars`` > 0 and ``len(context)`` exceeds it, runs the QA pipeline on
    each window and returns the span with the **highest** score (global char offsets).

    Args:
        context: Full passage (e.g. episode transcript).
        question: SQuAD-style question.
        model_id: HF model id or registry alias.
        device: Device hint (MPS coerced to CPU for QA).
        window_chars: Max characters per QA window; ``0`` = single call on full context.
        window_overlap_chars: Overlap between consecutive windows when windowing.
    """
    pipe = get_qa_pipeline(model_id, device=device)
    pipe_fn: Callable[..., Dict[str, Any]] = cast(Callable[..., Dict[str, Any]], pipe)

    if window_chars <= 0 or len(context) <= window_chars:
        return _qa_pipeline_call(pipe_fn, question, context)

    best: Optional[QASpan] = None
    overlap = max(0, window_overlap_chars)
    windows = _iter_context_windows(context, window_chars, overlap)
    for win_start, win_end, slice_ in windows:
        try:
            span = _qa_pipeline_call(pipe_fn, question, slice_)
        except Exception as e:
            logger.debug("QA window [%s:%s] failed: %s", win_start, win_end, e)
            continue
        g_start = win_start + span.start
        g_end = win_start + span.end
        if g_end > len(context) or g_start < 0 or g_end < g_start:
            logger.debug(
                "QA window [%s:%s] produced out-of-range global span [%s:%s]",
                win_start,
                win_end,
                g_start,
                g_end,
            )
            continue
        cand = QASpan(
            answer=context[g_start:g_end],
            start=g_start,
            end=g_end,
            score=span.score,
        )
        if best is None or cand.score > best.score:
            best = cand

    if best is not None:
        logger.debug(
            "Windowed QA: %d windows, best score=%.4f span=[%s:%s]",
            len(windows),
            best.score,
            best.start,
            best.end,
        )
        return best

    logger.debug("Windowed QA produced no valid span; falling back to full context")
    return _qa_pipeline_call(pipe_fn, question, context)


def answer_multi(
    context: str,
    questions: List[str],
    model_id: str,
    device: Optional[str] = None,
) -> List[QASpan]:
    """Multiple questions over the same context; returns one span per question."""
    pipe = get_qa_pipeline(model_id, device=device)
    pipe_fn: Callable[..., Dict[str, Any]] = cast(Callable[..., Dict[str, Any]], pipe)
    results: List[QASpan] = []
    for q in questions:
        results.append(_qa_pipeline_call(pipe_fn, q, context))
    return results

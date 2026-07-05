"""Extractive QA for the GIL evidence stack (Issue #435).

Lazy-loads an ``AutoModelForQuestionAnswering`` + ``AutoTokenizer`` pair
and returns answer spans (character offsets + score) for
"What evidence supports: {insight}?" style questions.

Historical note (#382, 2026-07): this module used to lean on
``transformers.pipeline("question-answering")``. transformers v5 removed
that pipeline task; we now do the QA forward pass directly. The public
API (:class:`QASpan`, :func:`answer`, :func:`answer_candidates`,
:func:`answer_multi`, :func:`clear_qa_pipeline_cache`) is preserved so
callers and tests do not change. Internal helpers previously named
``_qa_pipeline_*`` are renamed to ``_qa_forward_*`` to reflect the shift.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# One (model, tokenizer) pair per (resolved_model_id, device_key) within the
# process. Worker processes still load their own copy — this cache is per-PID.
# Lock prevents concurrent loads that would exhaust memory on CI (the same
# meta-tensor crash we hit when processing_parallelism > 1).
_qa_models: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
_qa_models_lock = threading.Lock()

# Pipeline-compatible defaults. Match the transformers QuestionAnsweringPipeline
# post-processing so behavior parity with the pre-v5 code is tight.
_QA_MAX_SEQ_LEN = 384
_QA_DOC_STRIDE = 128
_QA_MAX_ANSWER_LEN = 512  # matches the ``max_answer_len=512`` we passed to the pipeline


def clear_qa_pipeline_cache() -> None:
    """Drop cached HF QA (model, tokenizer) pairs — GitHub #539 multi-feed hygiene.

    Safe to call between feeds or after ML teardown; next use lazily rebuilds.
    Name is retained for API stability (service.py / orchestration.py call it).
    """
    _qa_models.clear()


def _safe_score_float(value: Any, fallback: float = 0.0) -> float:
    """Convert a torch scalar (tensor or float) to float; avoid .item() on meta tensors."""
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
    """Resolve device for the HF QA model.

    MPS is avoided: auto device selection used to pick ``mps``, but the QA stack
    can end up on ``meta`` or hit unsupported ops. CUDA when available, else CPU.
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
) -> Tuple[Any, Any]:
    """Load HuggingFace QA (model, tokenizer) pair.

    Name kept for API stability with pre-v5 callers; returns a 2-tuple now
    instead of a ``Pipeline``. Callers should treat the return value as opaque
    and go through :func:`answer` / :func:`answer_candidates` / :func:`answer_multi`.

    Args:
        model_id: Alias (e.g. ``roberta-squad2``) or full HF ID.
        device: Device (cpu, cuda, mps) or None for auto.

    Returns:
        ``(model, tokenizer)`` — model already moved to the chosen device and in ``eval()``.
    """
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    from ...cache import get_transformers_cache_dir

    resolved = ModelRegistry.resolve_evidence_model_id(model_id)
    dev = _get_device(device)
    logger.info("Loading extractive QA model %s on %s", resolved, dev)

    cache_dir = str(get_transformers_cache_dir().resolve())
    load_kwargs: Dict[str, Any] = {
        "cache_dir": cache_dir,
        "local_files_only": True,
        "trust_remote_code": False,  # Security: don't execute remote code
        "low_cpu_mem_usage": False,  # #539: lazy meta init breaks re-loads in-process
    }
    tokenizer = AutoTokenizer.from_pretrained(resolved, **load_kwargs)  # nosec B615
    model = AutoModelForQuestionAnswering.from_pretrained(resolved, **load_kwargs)  # nosec B615
    if dev != "cpu":
        model = model.to(dev)
    model.eval()
    return model, tokenizer


def get_qa_pipeline(
    model_id: str,
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Return cached QA (model, tokenizer) pair or load and cache it.

    Thread-safe: prevents concurrent loads that would exhaust memory on CI
    (meta tensor crash when processing_parallelism > 1). Name kept for API
    stability with pre-v5 callers.
    """
    resolved = ModelRegistry.resolve_evidence_model_id(model_id)
    dev = _get_device(device)
    key = (resolved, dev)
    with _qa_models_lock:
        if key not in _qa_models:
            _qa_models[key] = load_qa_pipeline(model_id, device=device)
        return _qa_models[key]


def _qa_forward_top_k(
    model: Any,
    tokenizer: Any,
    question: str,
    context: str,
    *,
    top_k: int,
    max_answer_len: int = _QA_MAX_ANSWER_LEN,
    max_seq_len: int = _QA_MAX_SEQ_LEN,
    doc_stride: int = _QA_DOC_STRIDE,
) -> List[QASpan]:
    """Return up to ``top_k`` extractive-QA spans by ``start_logit + end_logit`` score.

    Mimics the post-processing of ``transformers.QuestionAnsweringPipeline``:

    1. Tokenize (question, context) with overflow chunking (``stride=doc_stride``,
       ``max_length=max_seq_len``) so long contexts are handled without truncation.
    2. For each chunk, run the model, then walk the top-``top_k`` (start, end) pairs
       filtered to context tokens (``sequence_ids == 1``), ``start <= end``, and
       ``end - start + 1 <= max_answer_len``.
    3. Aggregate across chunks by score, dedup by ``(char_start, char_end)``, return
       the top-``top_k`` by score with character offsets into the original context.
    """
    import torch

    if top_k < 1:
        top_k = 1

    encoding = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=max_seq_len,
        stride=doc_stride,
        padding="max_length",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    offset_mappings = encoding.pop("offset_mapping")
    encoding.pop("overflow_to_sample_mapping", None)
    device = next(model.parameters()).device
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
    start_logits = outputs.start_logits.detach().cpu()
    end_logits = outputs.end_logits.detach().cpu()

    n_chunks = start_logits.shape[0]
    candidates: List[QASpan] = []
    for chunk_idx in range(n_chunks):
        s_logits = start_logits[chunk_idx]
        e_logits = end_logits[chunk_idx]
        offsets = offset_mappings[chunk_idx]
        sequence_ids = encoding.get("sequence_ids")
        # ``sequence_ids`` is not always present in the encoding dict returned by
        # newer tokenizers; use BatchEncoding.sequence_ids() as the fallback.
        if sequence_ids is None:
            batch_seq_ids = encoding.get("sequence_ids", None)
            if batch_seq_ids is None:
                # Recover via a fresh call — tokenizer instance preserves state.
                bat = tokenizer(
                    question,
                    context,
                    return_tensors="pt",
                    truncation="only_second",
                    max_length=max_seq_len,
                    stride=doc_stride,
                    padding="max_length",
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                )
                seq_ids = bat.sequence_ids(chunk_idx)
            else:
                seq_ids = batch_seq_ids[chunk_idx]
        else:
            seq_ids = sequence_ids[chunk_idx]

        # Mask logits outside the context segment (sequence_id == 1). Special
        # tokens have sequence_id == None; question tokens have sequence_id == 0.
        # We want context tokens only.
        context_mask = torch.tensor([sid == 1 for sid in seq_ids], dtype=torch.bool)
        neg_inf = torch.finfo(s_logits.dtype).min
        s_logits = torch.where(context_mask, s_logits, torch.full_like(s_logits, neg_inf))
        e_logits = torch.where(context_mask, e_logits, torch.full_like(e_logits, neg_inf))

        # Grab top-k*2 starts and ends independently, then combine into pairs
        # (matches the pipeline's ``n_best_size = 20`` heuristic scaled to top_k).
        n_best = max(top_k * 4, 20)
        start_idx = torch.topk(s_logits, min(n_best, s_logits.numel())).indices.tolist()
        end_idx = torch.topk(e_logits, min(n_best, e_logits.numel())).indices.tolist()

        for si in start_idx:
            for ei in end_idx:
                if ei < si:
                    continue
                if ei - si + 1 > max_answer_len:
                    continue
                if offsets[si][0] == 0 and offsets[si][1] == 0:
                    continue
                if offsets[ei][0] == 0 and offsets[ei][1] == 0:
                    continue
                char_start = int(offsets[si][0])
                char_end = int(offsets[ei][1])
                if char_end <= char_start:
                    continue
                score = float(s_logits[si].item() + e_logits[ei].item())
                candidates.append(
                    QASpan(
                        answer=context[char_start:char_end],
                        start=char_start,
                        end=char_end,
                        score=score,
                    )
                )

    if not candidates:
        return []

    # Softmax the aggregated top-k by score for a pipeline-parity probability.
    # We apply softmax over the surviving candidate scores to mimic
    # transformers' post-processing (the pipeline reports softmax'd probs).
    candidates.sort(key=lambda s: s.score, reverse=True)
    # Dedup by (char_start, char_end); keep the higher-scored entry (already sorted).
    seen: set[Tuple[int, int]] = set()
    unique: List[QASpan] = []
    for cand in candidates:
        key = (cand.start, cand.end)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
        if len(unique) >= top_k:
            break

    # Convert raw logit-sums to softmax probabilities across the retained set.
    import math

    max_score = max(u.score for u in unique)
    exps = [math.exp(u.score - max_score) for u in unique]
    total = sum(exps)
    if total > 0:
        unique = [
            QASpan(answer=u.answer, start=u.start, end=u.end, score=exps[i] / total)
            for i, u in enumerate(unique)
        ]
    return unique


def _qa_forward_top1(model: Any, tokenizer: Any, question: str, context: str) -> QASpan:
    """Convenience wrapper: return the single best span or a zero-score empty span."""
    spans = _qa_forward_top_k(model, tokenizer, question, context, top_k=1)
    if spans:
        return spans[0]
    return QASpan(answer="", start=0, end=0, score=0.0)


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

    When ``window_chars`` > 0 and ``len(context)`` exceeds it, runs the QA forward on
    each window and returns the span with the **highest** score (global char offsets).

    Args:
        context: Full passage (e.g. episode transcript).
        question: SQuAD-style question.
        model_id: HF model id or registry alias.
        device: Device hint (MPS coerced to CPU for QA).
        window_chars: Max characters per QA window; ``0`` = single call on full context.
        window_overlap_chars: Overlap between consecutive windows when windowing.
    """
    model, tokenizer = get_qa_pipeline(model_id, device=device)

    if window_chars <= 0 or len(context) <= window_chars:
        return _qa_forward_top1(model, tokenizer, question, context)

    best: Optional[QASpan] = None
    overlap = max(0, window_overlap_chars)
    windows = _iter_context_windows(context, window_chars, overlap)
    for win_start, win_end, slice_ in windows:
        try:
            span = _qa_forward_top1(model, tokenizer, question, slice_)
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
    return _qa_forward_top1(model, tokenizer, question, context)


def answer_candidates(
    context: str,
    question: str,
    model_id: str,
    device: Optional[str] = None,
    *,
    window_chars: int = 0,
    window_overlap_chars: int = 250,
    top_k: int = 3,
) -> List[QASpan]:
    """Extractive QA returning up to ``top_k`` candidate spans (Issue #487 / EV-1).

    When the transcript fits one context (no windowing), the QA forward is called
    with ``top_k`` so multiple non-identical spans can be scored with NLI downstream.

    When ``window_chars`` > 0 and the transcript is long, windowing still returns a
    **single** best span (multi-candidate window merge is not implemented yet).
    """
    model, tokenizer = get_qa_pipeline(model_id, device=device)
    top_k_i = max(1, min(int(top_k), 10))

    if window_chars <= 0 or len(context) <= window_chars:
        return _qa_forward_top_k(model, tokenizer, question, context, top_k=top_k_i)

    single = answer(
        context,
        question,
        model_id,
        device=device,
        window_chars=window_chars,
        window_overlap_chars=window_overlap_chars,
    )
    return [single]


def answer_multi(
    context: str,
    questions: List[str],
    model_id: str,
    device: Optional[str] = None,
) -> List[QASpan]:
    """Multiple questions over the same context; returns one span per question."""
    model, tokenizer = get_qa_pipeline(model_id, device=device)
    results: List[QASpan] = []
    for q in questions:
        results.append(_qa_forward_top1(model, tokenizer, q, context))
    return results

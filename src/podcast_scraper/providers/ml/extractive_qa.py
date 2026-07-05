"""Extractive QA for the GIL evidence stack (Issue #435).

Post-#382 Phase E: this module is a thin functional wrapper over
:class:`QAEvidenceBackend` (in :mod:`.hf_evidence_backend`). The public
surface (:class:`QASpan`, :func:`answer`, :func:`answer_candidates`,
:func:`answer_multi`, :func:`clear_qa_model_cache`) is preserved so
callers (``gi.grounding``, ``ml_provider``, tests) do not move.

Historical note: v4 used ``transformers.pipeline("question-answering")``
here (retired in transformers v5 — #382). Phase 3 replaced that with a
direct ``AutoModelForQuestionAnswering`` + tokenizer forward pass. Phase E
now folds that load/cache logic into the shared ``HFEvidenceBackend``
alongside NLI and embedding, so the three loaders share one shape.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional, Tuple

from .hf_evidence_backend import HFEvidenceBackend, standard_hf_load_kwargs

logger = logging.getLogger(__name__)

# Pipeline-compatible defaults. Match the transformers QuestionAnsweringPipeline
# post-processing so behavior parity with the pre-v5 code stays tight.
_QA_MAX_SEQ_LEN = 384
_QA_DOC_STRIDE = 128
_QA_MAX_ANSWER_LEN = 512


@dataclass
class QASpan:
    """Single extractive QA result: answer text and character offsets."""

    answer: str
    start: int
    end: int
    score: float


class QAEvidenceBackend(HFEvidenceBackend):
    """AutoModelForQuestionAnswering + AutoTokenizer under a QA head.

    MPS is unsupported for this task (attention kernels have failed on
    MPS historically; QA post-processing needs stable ``.item()`` on the
    logits). ``mps_supported = False`` causes :func:`resolve_evidence_device`
    to coerce ``mps`` → ``cpu``.
    """

    kind = "qa"
    mps_supported = False

    _instances: ClassVar[dict] = {}
    _instances_lock: ClassVar[threading.Lock] = threading.Lock()

    def _load(self) -> None:
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        kw = standard_hf_load_kwargs()
        self.tokenizer = AutoTokenizer.from_pretrained(self.resolved_id, **kw)  # nosec B615
        self.model = AutoModelForQuestionAnswering.from_pretrained(  # nosec B615
            self.resolved_id, **kw
        )
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()

    # ---- Task methods ------------------------------------------------

    def answer_top_k(
        self,
        question: str,
        context: str,
        *,
        top_k: int,
        max_answer_len: int = _QA_MAX_ANSWER_LEN,
        max_seq_len: int = _QA_MAX_SEQ_LEN,
        doc_stride: int = _QA_DOC_STRIDE,
    ) -> List[QASpan]:
        """Return up to ``top_k`` spans by ``start_logit + end_logit`` score.

        Mimics :class:`transformers.QuestionAnsweringPipeline` post-processing:
        tokenize with overflow chunking, per-chunk logit walk filtered to
        context tokens, aggregate across chunks, dedup by ``(char_start,
        char_end)``, softmax across the retained set for pipeline-parity
        probability output.
        """
        import torch

        if top_k < 1:
            top_k = 1

        encoding = self.tokenizer(
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
        device = next(self.model.parameters()).device
        input_dict = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**input_dict)
        start_logits = outputs.start_logits.detach().cpu()
        end_logits = outputs.end_logits.detach().cpu()

        # Re-tokenize once to recover BatchEncoding.sequence_ids(chunk_idx)
        # (tokenizer instance is stateless; the fresh call is cheap and
        # matches the encoding we already have — same seed data).
        bat = self.tokenizer(
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

        n_chunks = start_logits.shape[0]
        candidates: List[QASpan] = []
        for chunk_idx in range(n_chunks):
            s_logits = start_logits[chunk_idx]
            e_logits = end_logits[chunk_idx]
            offsets = offset_mappings[chunk_idx]
            seq_ids = bat.sequence_ids(chunk_idx)

            context_mask = torch.tensor([sid == 1 for sid in seq_ids], dtype=torch.bool)
            neg_inf = torch.finfo(s_logits.dtype).min
            s_logits = torch.where(context_mask, s_logits, torch.full_like(s_logits, neg_inf))
            e_logits = torch.where(context_mask, e_logits, torch.full_like(e_logits, neg_inf))

            n_best = max(top_k * 4, 20)
            start_idx = torch.topk(s_logits, min(n_best, s_logits.numel())).indices.tolist()
            end_idx = torch.topk(e_logits, min(n_best, e_logits.numel())).indices.tolist()

            for si in start_idx:
                for ei in end_idx:
                    if ei < si or ei - si + 1 > max_answer_len:
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

        candidates.sort(key=lambda s: s.score, reverse=True)
        seen: set = set()
        unique: List[QASpan] = []
        for cand in candidates:
            key = (cand.start, cand.end)
            if key in seen:
                continue
            seen.add(key)
            unique.append(cand)
            if len(unique) >= top_k:
                break

        max_score = max(u.score for u in unique)
        exps = [math.exp(u.score - max_score) for u in unique]
        total = sum(exps)
        if total > 0:
            unique = [
                QASpan(
                    answer=u.answer,
                    start=u.start,
                    end=u.end,
                    score=exps[i] / total,
                )
                for i, u in enumerate(unique)
            ]
        return unique

    def answer_top1(self, question: str, context: str) -> QASpan:
        """Return the single best span or a zero-score empty QASpan on failure."""
        spans = self.answer_top_k(question, context, top_k=1)
        if spans:
            return spans[0]
        return QASpan(answer="", start=0, end=0, score=0.0)


# ---- Module-level thin wrappers (public API preserved) -------------------


def clear_qa_model_cache() -> None:
    """Drop cached QA backends — multi-feed hygiene (GitHub #539).

    Cleared between feeds (GitHub #539) to force fresh weight-tensor state
    and avoid meta-tensor / lazy-init sharing across independent runs.
    """
    QAEvidenceBackend.clear_cache()


def _get_qa_backend(model_id: str, device: Optional[str] = None) -> "QAEvidenceBackend":
    from typing import cast as _cast

    return _cast(QAEvidenceBackend, QAEvidenceBackend.get_or_load(model_id, device=device))


def load_qa_model(
    model_id: str,
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load HuggingFace QA (model, tokenizer) pair (uncached).

    Post-#382 canonical name. Returns a 2-tuple; the previous
    ``load_qa_pipeline`` alias emits :class:`DeprecationWarning`.
    """
    backend = QAEvidenceBackend(model_id, device=device)
    backend._ensure_loaded()
    return backend.model, backend.tokenizer


def get_qa_model(
    model_id: str,
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Return cached QA (model, tokenizer) pair or load and cache it.

    Post-#382 canonical name. Backing cache lives in
    :attr:`QAEvidenceBackend._instances`. The previous ``get_qa_pipeline``
    alias emits :class:`DeprecationWarning`.
    """
    backend = _get_qa_backend(model_id, device=device)
    return backend.model, backend.tokenizer


# ---- Deprecation aliases (removed in v3.0.0) -----------------------------


def load_qa_pipeline(
    model_id: str,
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Deprecated alias for :func:`load_qa_model` (#382).

    transformers v5 removed ``pipeline("question-answering")`` — this
    function has not returned a pipeline object since Phase 3 of #382.
    Migrate callers to :func:`load_qa_model`.
    """
    import warnings

    warnings.warn(
        "load_qa_pipeline is deprecated; use load_qa_model instead. "
        "The QA loader has returned a (model, tokenizer) tuple since #382 "
        "(transformers v5 removed pipeline('question-answering')).",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_qa_model(model_id, device=device)


def get_qa_pipeline(
    model_id: str,
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Deprecated alias for :func:`get_qa_model` (#382)."""
    import warnings

    warnings.warn(
        "get_qa_pipeline is deprecated; use get_qa_model instead. "
        "The QA getter has returned a (model, tokenizer) tuple since #382.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_qa_model(model_id, device=device)


def _iter_context_windows(
    context: str,
    window_chars: int,
    overlap_chars: int,
) -> List[Tuple[int, int, str]]:
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

    See :meth:`QAEvidenceBackend.answer_top1` for the per-window forward pass.
    """
    backend = _get_qa_backend(model_id, device=device)

    if window_chars <= 0 or len(context) <= window_chars:
        return backend.answer_top1(question, context)

    best: Optional[QASpan] = None
    overlap = max(0, window_overlap_chars)
    windows = _iter_context_windows(context, window_chars, overlap)
    for win_start, win_end, slice_ in windows:
        try:
            span = backend.answer_top1(question, slice_)
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
    return backend.answer_top1(question, context)


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
    """Extractive QA returning up to ``top_k`` candidate spans (Issue #487 / EV-1)."""
    backend = _get_qa_backend(model_id, device=device)
    top_k_i = max(1, min(int(top_k), 10))

    if window_chars <= 0 or len(context) <= window_chars:
        return backend.answer_top_k(question, context, top_k=top_k_i)

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
    backend = _get_qa_backend(model_id, device=device)
    return [backend.answer_top1(q, context) for q in questions]

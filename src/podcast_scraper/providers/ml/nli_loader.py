"""NLI cross-encoder loader for GIL evidence stack (Issue #435).

Lazy-loads a cross-encoder for premise/hypothesis → entailment score.
Used to validate that a quote supports an insight.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...cache import get_transformers_cache_dir
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# CrossEncoder per (resolved_model_id, device) within the process.
_nli_models: Dict[Tuple[str, str], object] = {}


def _scalar_to_float(value: object, fallback: float = 0.0) -> float:
    """Convert tensor or scalar to float; avoid .item() on meta tensors (e.g. GIL + API-only)."""
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except RuntimeError as e:
            if "meta" in str(e).lower() or "item" in str(e).lower():
                logger.debug("Score on meta device or non-materialized tensor: %s", e)
                return fallback
            raise
        except ValueError as e:
            # NumPy: "can only convert an array of size 1 to a Python scalar"
            if "scalar" in str(e).lower() or "size 1" in str(e).lower():
                logger.debug("Non-scalar score passed to .item()-style conversion: %s", e)
                return fallback
            raise
    if hasattr(value, "__len__") and len(value):
        return _scalar_to_float(value[0], fallback)  # type: ignore[index]
    return fallback


def _entailment_class_index(model: object) -> int:
    """Return classifier index for the entailment label (CrossEncoder / HF id2label)."""
    inner = getattr(model, "model", model)
    cfg = getattr(inner, "config", None)
    id2label = getattr(cfg, "id2label", None) if cfg is not None else None
    if isinstance(id2label, dict):
        for k, v in id2label.items():
            if str(v).lower() == "entailment":
                try:
                    return int(k)
                except (TypeError, ValueError):
                    continue
    # MNLI-style three-class: contradiction, neutral, entailment
    return 2


def _softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    if s <= 0:
        n = len(logits)
        return [1.0 / n] * n
    return [e / s for e in exps]


def _predict_raw_to_nested_lists(raw: object) -> object:
    """Turn tensor / ndarray outputs into nested Python lists for uniform parsing."""
    if hasattr(raw, "detach"):
        try:
            raw = raw.detach().cpu()
        except RuntimeError as e:
            if "meta" in str(e).lower():
                raise
            raise
    if hasattr(raw, "tolist"):
        try:
            return raw.tolist()
        except RuntimeError as e:
            if "meta" in str(e).lower():
                raise
            raise
    return raw


def _rows_from_predict_output(raw: object) -> List[List[float]]:
    """Normalize CrossEncoder.predict output to one row of floats per premise/hypothesis pair."""
    data = _predict_raw_to_nested_lists(raw)
    if data is None:
        return []
    if isinstance(data, (int, float)):
        return [[float(data)]]
    if not isinstance(data, list) or len(data) == 0:
        return []

    first = data[0]
    if isinstance(first, (list, tuple)):
        return [[float(x) for x in row] for row in data]  # type: ignore[list-item]
    return [[float(x) for x in data]]


def _row_to_entailment_probability(row: Sequence[float], entail_idx: int) -> float:
    """Map one row of logits or a single score to P(entailment) in [0, 1]."""
    if not row:
        return 0.0
    if len(row) == 1:
        return float(max(0.0, min(1.0, row[0])))
    if entail_idx < 0 or entail_idx >= len(row):
        entail_idx = min(len(row) - 1, max(0, entail_idx))
    probs = _softmax(list(row))
    return float(max(0.0, min(1.0, probs[entail_idx])))


def predict_output_to_entailment_scores(raw: object, model: object) -> List[float]:
    """Convert CrossEncoder.predict output to one float per input pair (entailment probability).

    Handles scalar outputs, shape (3,) logits, (1, 3), (n, 3), and legacy single-logit rows.
    """
    try:
        rows = _rows_from_predict_output(raw)
    except RuntimeError as e:
        if "meta" in str(e).lower():
            logger.debug("NLI predict on meta device: %s", e)
            return []
        raise
    if not rows:
        return []
    idx = _entailment_class_index(model)
    return [_row_to_entailment_probability(r, idx) for r in rows]


@contextlib.contextmanager
def _silence_ml_nli_inference_noise():
    """Reduce Hugging Face / hub chatter during CrossEncoder.predict (tqdm, warnings)."""
    env_keys = (
        ("TRANSFORMERS_VERBOSITY", "error"),
        ("HF_HUB_DISABLE_PROGRESS_BARS", "1"),
        ("TQDM_DISABLE", "1"),
    )
    saved: Dict[str, Optional[str]] = {}
    for key, val in env_keys:
        saved[key] = os.environ.get(key)
        os.environ[key] = val

    old_hf_verb: Optional[int] = None
    try:
        from transformers.utils import logging as hf_logging

        old_hf_verb = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
    except ImportError:
        pass

    try:
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        if old_hf_verb is not None:
            try:
                from transformers.utils import logging as hf_logging

                hf_logging.set_verbosity(old_hf_verb)
            except ImportError:
                pass


def _get_device(device: Optional[str]) -> str:
    if device is not None and device.strip():
        return device.strip().lower()
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def load_nli_model(
    model_id: str,
    device: Optional[str] = None,
) -> object:
    """Load NLI cross-encoder model.

    Args:
        model_id: Alias (e.g. nli-deberta-base) or full HF ID.
        device: Device (cpu, cuda, mps) or None for auto.

    Returns:
        CrossEncoder instance (sentence_transformers).
    """
    from sentence_transformers import CrossEncoder

    resolved = ModelRegistry.resolve_evidence_model_id(model_id)
    dev = _get_device(device)
    logger.info("Loading NLI model %s on %s", resolved, dev)
    cache_dir = str(get_transformers_cache_dir().resolve())
    # ST 3+: local_files_only on CrossEncoder only (not inside tokenizer/model kwargs — duplicate
    # key error). ST 5.3+: prefer cache_folder + tokenizer_kwargs/model_kwargs over *_args.
    # ST 2.x: CrossEncoder doesn't accept local_files_only or cache_folder at all.
    import inspect

    ce_params = set(inspect.signature(CrossEncoder.__init__).parameters)
    ce_kwargs: Dict[str, Any] = {"device": dev}
    if "local_files_only" in ce_params:
        ce_kwargs["local_files_only"] = True
    if "cache_folder" in ce_params:
        ce_kwargs["cache_folder"] = cache_dir
    model = CrossEncoder(resolved, **ce_kwargs)
    return model


def get_nli_model(
    model_id: str,
    device: Optional[str] = None,
) -> object:
    """Return cached NLI model or load and cache it (lazy, keyed by model + device)."""
    resolved = ModelRegistry.resolve_evidence_model_id(model_id)
    dev = _get_device(device)
    key = (resolved, dev)
    if key not in _nli_models:
        _nli_models[key] = load_nli_model(model_id, device=device)
    return _nli_models[key]


def entailment_score(
    premise: str,
    hypothesis: str,
    model_id: str,
    device: Optional[str] = None,
) -> float:
    """Score entailment of hypothesis given premise (0–1, higher = more entailment).

    Args:
        premise: Evidence text (e.g. quote).
        hypothesis: Claim (e.g. insight).
        model_id: Model alias or full HF ID.
        device: Device or None for auto.

    Returns:
        Entailment probability for the model's entailment class (multi-class NLI) or raw
        score when the model returns a single logit.
    """
    model = get_nli_model(model_id, device=device)
    try:
        with _silence_ml_nli_inference_noise():
            raw = model.predict([[premise, hypothesis]])
    except RuntimeError as e:
        if "meta" in str(e).lower():
            logger.debug("NLI predict failed (meta device): %s", e)
            return 0.0
        raise
    scores = predict_output_to_entailment_scores(raw, model)
    if not scores:
        return 0.0
    return float(scores[0])


def entailment_scores_batch(
    pairs: List[Tuple[str, str]],
    model_id: str,
    device: Optional[str] = None,
) -> List[float]:
    """Batch premise/hypothesis pairs; returns list of entailment scores."""
    model = get_nli_model(model_id, device=device)
    try:
        with _silence_ml_nli_inference_noise():
            raw = model.predict(pairs)
    except RuntimeError as e:
        if "meta" in str(e).lower():
            logger.debug("NLI batch predict failed (meta device): %s", e)
            return [0.0] * len(pairs)
        raise
    scores = predict_output_to_entailment_scores(raw, model)
    if len(scores) != len(pairs):
        # Extremely defensive: pad or trim to match batch size
        if len(scores) < len(pairs):
            scores = scores + [0.0] * (len(pairs) - len(scores))
        else:
            scores = scores[: len(pairs)]
    return scores

"""NLI cross-encoder loader for GIL evidence stack (Issue #435).

Post-#382 Phase E: thin functional wrapper over :class:`NLIEvidenceBackend`
(in :mod:`.hf_evidence_backend`). The public surface
(:func:`load_nli_model`, :func:`get_nli_model`, :func:`entailment_score`,
:func:`entailment_scores_batch`, :func:`predict_output_to_entailment_scores`)
is preserved so callers and tests do not move.

The load / cache / device-resolution scaffolding used to be duplicated
across three modules (QA, NLI, embedding). It now lives in the shared
:class:`HFEvidenceBackend`. Post-processing helpers stay module-local
because they are NLI-specific (softmax over id2label logits with an
entailment-class-index sniff).
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import threading
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

from .hf_evidence_backend import HFEvidenceBackend

logger = logging.getLogger(__name__)


def _scalar_to_float(value: object, fallback: float = 0.0) -> float:
    """Convert tensor or scalar to float; avoid .item() on meta tensors."""
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
            if "scalar" in str(e).lower() or "size 1" in str(e).lower():
                logger.debug("Non-scalar score passed to .item()-style conversion: %s", e)
                return fallback
            raise
    if hasattr(value, "__len__") and len(value):  # type: ignore[arg-type]
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
    return 2  # MNLI-style: contradiction, neutral, entailment


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
    if hasattr(raw, "detach"):
        try:
            raw = raw.detach().cpu()  # type: ignore[attr-defined]
        except RuntimeError as e:
            if "meta" in str(e).lower():
                raise
            raise
    if hasattr(raw, "tolist"):
        try:
            return raw.tolist()  # type: ignore[attr-defined]
        except RuntimeError as e:
            if "meta" in str(e).lower():
                raise
            raise
    return raw


def _rows_from_predict_output(raw: object) -> List[List[float]]:
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
    if not row:
        return 0.0
    if len(row) == 1:
        return float(max(0.0, min(1.0, row[0])))
    if entail_idx < 0 or entail_idx >= len(row):
        entail_idx = min(len(row) - 1, max(0, entail_idx))
    probs = _softmax(list(row))
    return float(max(0.0, min(1.0, probs[entail_idx])))


def predict_output_to_entailment_scores(raw: object, model: object) -> List[float]:
    """Convert CrossEncoder.predict output to one float per input pair (entailment probability)."""
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
    """Reduce Hugging Face / hub chatter during CrossEncoder.predict."""
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


class NLIEvidenceBackend(HFEvidenceBackend):
    """sentence-transformers CrossEncoder wrapper.

    Uses CrossEncoder rather than raw ``AutoModel*`` because our NLI checkpoints
    are shipped as sentence-transformers cross-encoders and rely on their
    predict-time post-processing. Load discipline goes through
    :class:`HFEvidenceBackend`'s cache; the CrossEncoder call signature is
    sniffed via ``inspect`` (ST 2.x vs 3.x vs 5.x differ on which of
    ``local_files_only`` / ``cache_folder`` they accept).
    """

    kind = "nli"
    mps_supported = True

    _instances: ClassVar[dict] = {}
    _instances_lock: ClassVar[threading.Lock] = threading.Lock()

    def _load(self) -> None:
        import inspect

        from sentence_transformers import CrossEncoder

        from ...cache import get_transformers_cache_dir

        cache_dir = str(get_transformers_cache_dir().resolve())
        ce_params = set(inspect.signature(CrossEncoder.__init__).parameters)
        ce_kwargs: Dict[str, Any] = {"device": self.device}
        if "local_files_only" in ce_params:
            ce_kwargs["local_files_only"] = True
        if "cache_folder" in ce_params:
            ce_kwargs["cache_folder"] = cache_dir
        self.model = CrossEncoder(self.resolved_id, **ce_kwargs)  # nosec B615

    def predict_scores(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Return entailment probabilities (0–1) for each (premise, hypothesis) pair."""
        try:
            with _silence_ml_nli_inference_noise():
                raw = self.model.predict(pairs)
        except RuntimeError as e:
            if "meta" in str(e).lower():
                logger.debug("NLI predict failed (meta device): %s", e)
                return [0.0] * len(pairs)
            raise
        scores = predict_output_to_entailment_scores(raw, self.model)
        if len(scores) != len(pairs):
            if len(scores) < len(pairs):
                scores = scores + [0.0] * (len(pairs) - len(scores))
            else:
                scores = scores[: len(pairs)]
        return scores


# ---- Module-level thin wrappers (public API preserved) -------------------


def _get_nli_backend(model_id: str, device: Optional[str] = None) -> "NLIEvidenceBackend":
    from typing import cast as _cast

    return _cast(NLIEvidenceBackend, NLIEvidenceBackend.get_or_load(model_id, device=device))


def load_nli_model(
    model_id: str,
    device: Optional[str] = None,
) -> object:
    """Load NLI CrossEncoder — returns the underlying CrossEncoder instance."""
    backend = NLIEvidenceBackend(model_id, device=device)
    backend._ensure_loaded()
    return backend.model


def get_nli_model(
    model_id: str,
    device: Optional[str] = None,
) -> object:
    """Return cached NLI CrossEncoder or load and cache it (lazy, keyed by model + device)."""
    backend = _get_nli_backend(model_id, device=device)
    return backend.model


def entailment_score(
    premise: str,
    hypothesis: str,
    model_id: str,
    device: Optional[str] = None,
) -> float:
    """Score entailment of hypothesis given premise (0–1, higher = more entailment)."""
    backend = _get_nli_backend(model_id, device=device)
    scores = backend.predict_scores([(premise, hypothesis)])
    return float(scores[0]) if scores else 0.0


def entailment_scores_batch(
    pairs: List[Tuple[str, str]],
    model_id: str,
    device: Optional[str] = None,
) -> List[float]:
    """Batch premise/hypothesis pairs; returns list of entailment scores."""
    backend = _get_nli_backend(model_id, device=device)
    return backend.predict_scores(pairs)

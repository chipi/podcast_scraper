"""Pluggable query router (RFC-092 / #860).

A thin classification seam over the rules table in ``router.py``. The
``RetrievalLayer`` calls ``QueryRouter.classify`` to pick an intent, which drives
``signal_weights_for`` / ``tier_weights_for``; *how* the intent is decided —
hand-written rules or a learned classifier — is swapped here without touching
fusion, the backend, or the weight tables.

- ``RulesQueryRouter`` wraps the deterministic ``classify_query`` (today's default,
  zero dependencies, zero training data).
- ``MLQueryRouter`` loads a joblib-persisted scikit-learn classifier over MiniLM
  query embeddings (trained by ``scripts/train_query_router.py``). It degrades to
  the rules router when the model file is absent or fails to load, so enabling
  ``router.mode: ml`` can never harden into a failure — worst case is rules
  behaviour. Misclassification only perturbs RRF weights, never drops results
  (RFC-090 §3.6).

The trained model is a LogisticRegression over the same 384-dim all-MiniLM-L6-v2
space the index uses (no ONNX runtime dependency — sklearn + joblib are already
in the stack). Bootstrapping the ≥500 labeled queries it needs is the train
script's job; until that model exists, ``get_query_router`` returns rules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Protocol

from .router import classify_query, QUERY_TYPES

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "./data/query_router.joblib"


class QueryRouter(Protocol):
    """Maps query text → one of ``router.QUERY_TYPES``."""

    def classify(self, text: str) -> str:
        """Return the detected intent for *text*."""
        ...


class RulesQueryRouter:
    """Deterministic router — wraps the hand-written ``classify_query`` rules."""

    mode = "rules"

    def classify(self, text: str) -> str:
        """Return the rules-based intent for *text*."""
        return classify_query(text)


class MLQueryRouter:
    """Learned router — sklearn classifier over MiniLM query embeddings.

    Loads lazily and falls back to ``RulesQueryRouter`` if the model is missing or
    unloadable, so the ML mode is always safe to enable.
    """

    mode = "ml"

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._model = None
        self._fallback = RulesQueryRouter()
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.model_path.is_file():
            logger.warning("query-router model %s absent; using rules", self.model_path)
            return
        try:
            import joblib

            self._model = joblib.load(self.model_path)
        except Exception as exc:  # noqa: BLE001 - any load failure degrades to rules
            logger.warning("query-router model load failed (%s); using rules", exc)
            self._model = None

    def _embed(self, text: str) -> List[float]:
        import numpy as np

        from ..providers.ml.embedding_loader import encode

        arr = np.asarray(encode(text, "minilm-l6", allow_download=True), dtype=float).ravel()
        return [float(x) for x in arr.tolist()]

    def classify(self, text: str) -> str:
        """Predict the intent for *text* (rules fallback if no model)."""
        self._ensure_loaded()
        if self._model is None:
            return self._fallback.classify(text)
        try:
            label = str(self._model.predict([self._embed(text)])[0])
        except Exception as exc:  # noqa: BLE001 - inference failure degrades to rules
            logger.warning("query-router inference failed (%s); using rules", exc)
            return self._fallback.classify(text)
        return label if label in QUERY_TYPES else self._fallback.classify(text)


def get_query_router(mode: str = "rules", *, model_path: Optional[str] = None) -> QueryRouter:
    """Build the router for *mode* (``rules`` | ``ml``); unknown modes → rules."""
    if mode == "ml":
        return MLQueryRouter(model_path or DEFAULT_MODEL_PATH)
    return RulesQueryRouter()

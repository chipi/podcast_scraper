"""Rules-based query router (RFC-090 §3.6).

Classifies a query into one intent and selects per-intent signal + tier weights.
Deterministic and dependency-free; the ML router (RFC-092 / #860) replaces
``classify_query`` behind the same weights without changing this interface.
Misclassification degrades to sub-optimal weights, not wrong results — RRF is
robust to weight perturbation.
"""

from __future__ import annotations

import re
from typing import Dict

TEMPORAL_WORDS = ["evolve", "changed", "over time", "history", "trend", "shift"]
SYNTH_WORDS = ["compare", "contrast", " vs ", "versus", "across shows"]
EVIDENCE_WORDS = ["quote", "said", "exactly", "transcript", "verbatim", "phrase"]
_NAME_RE = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")

QUERY_TYPES = (
    "entity_lookup",
    "raw_evidence",
    "temporal_tracking",
    "cross_show_synthesis",
    "semantic",
)


def classify_query(text: str) -> str:
    """Return the detected intent for *text* (one of ``QUERY_TYPES``)."""
    t = f" {text.lower()} "
    if any(w in t for w in EVIDENCE_WORDS):
        return "raw_evidence"
    if any(w in t for w in TEMPORAL_WORDS):
        return "temporal_tracking"
    if any(w in t for w in SYNTH_WORDS):
        return "cross_show_synthesis"
    if _NAME_RE.search(text):
        return "entity_lookup"
    return "semantic"


# Signal weights per query type (BM25 vs dense vector vs KG proximity, RFC-091).
SIGNAL_WEIGHTS: Dict[str, Dict[str, float]] = {
    "entity_lookup": {"bm25": 1.4, "vector": 0.6, "kg": 1.2},
    "raw_evidence": {"bm25": 1.5, "vector": 0.5, "kg": 0.5},
    "temporal_tracking": {"bm25": 0.8, "vector": 1.2, "kg": 1.0},
    "cross_show_synthesis": {"bm25": 0.7, "vector": 1.3, "kg": 1.1},
    "semantic": {"bm25": 1.0, "vector": 1.0, "kg": 1.0},
}

# Tier weights per query type (override the RRF defaults).
TIER_WEIGHTS_BY_QUERY: Dict[str, Dict[str, float]] = {
    "entity_lookup": {"insight": 1.3, "segment": 0.9},
    "raw_evidence": {"insight": 0.9, "segment": 1.3},
    "temporal_tracking": {"insight": 1.2, "segment": 1.0},
    "cross_show_synthesis": {"insight": 1.2, "segment": 1.0},
    "semantic": {"insight": 1.2, "segment": 1.0},
}


def signal_weights_for(query_type: str) -> Dict[str, float]:
    """Signal weights for *query_type* (falls back to ``semantic``)."""
    return SIGNAL_WEIGHTS.get(query_type, SIGNAL_WEIGHTS["semantic"])


def tier_weights_for(query_type: str) -> Dict[str, float]:
    """Tier weights for *query_type* (falls back to ``semantic``)."""
    return TIER_WEIGHTS_BY_QUERY.get(query_type, TIER_WEIGHTS_BY_QUERY["semantic"])

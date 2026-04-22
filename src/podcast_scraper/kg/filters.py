"""#652 Part B — deterministic post-extraction validators for KG topics + entities.

Two filters that run on the final topic/entity lists regardless of source
(``provider``, ``summary_bullets``, prefilled from mega/extraction bundle):

1. Topic normalizer — lowercases-strips, trims to ≤ 4 tokens, drops leading
   and medial stopwords, dedupes near-matches within an episode via
   normalized-form equality. Keeps first-occurrence order.

2. Entity-kind repair — maintains a curated ``KNOWN_ORGS`` set seeded from
   the 100-ep `my-manual-run4` corpus; forces ``kind=org`` for exact-match
   only. Source-agnostic — fixes both LLM-assigned kind errors and spaCy
   NER label mistakes. False negatives (missing an org) strictly preferred
   over false positives (wrongly overriding). Anything not in the list:
   the model's / NER's answer governs.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Topic normalizer (Finding 2)
# ---------------------------------------------------------------------------

_TOPIC_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "of",
        "for",
        "vs",
        "to",
        "on",
        "at",
        "by",
        "from",
        "with",
    }
)

_TOPIC_MAX_TOKENS = 4

_PUNCTUATION_RE = re.compile(r"[^\w\s-]")
_MULTI_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_topic_label(label: str) -> Optional[str]:
    """Return a lower-cased, stopword-stripped, ≤4-token topic label, or None.

    Deterministic — same input always produces the same output. Preserves
    internal hyphens (``ai-agents`` stays as one token) but strips punctuation.
    """
    if not label:
        return None
    text = _PUNCTUATION_RE.sub(" ", label).lower().strip()
    text = _MULTI_WHITESPACE_RE.sub(" ", text)
    if not text:
        return None
    tokens = text.split(" ")
    # Drop leading stopwords.
    while tokens and tokens[0] in _TOPIC_STOPWORDS:
        tokens = tokens[1:]
    # Drop trailing stopwords too — "markets of the" → "markets".
    while tokens and tokens[-1] in _TOPIC_STOPWORDS:
        tokens = tokens[:-1]
    # Trim to max tokens.
    tokens = tokens[:_TOPIC_MAX_TOKENS]
    # Drop medial stopwords between content words (e.g. "markets in flux"
    # becomes "markets flux"); safe because we already capped at 4 tokens.
    tokens = [t for t in tokens if t not in _TOPIC_STOPWORDS]
    if not tokens:
        return None
    return " ".join(tokens)


def normalize_topic_labels(labels: Sequence[str]) -> Tuple[List[str], int]:
    """Normalize + dedupe a topic list. Returns ``(normalized, change_count)``.

    ``change_count`` counts every label whose normalized form differs from
    the input OR that was dropped as a near-duplicate. Used to populate the
    ``topics_normalized_count`` metric.
    """
    out: List[str] = []
    seen_normalized: set[str] = set()
    changes = 0
    for raw in labels:
        raw_str = str(raw or "").strip()
        normalized = _normalize_topic_label(raw_str)
        if normalized is None:
            changes += 1
            continue
        if normalized in seen_normalized:
            changes += 1
            continue
        seen_normalized.add(normalized)
        if normalized != raw_str:
            changes += 1
        out.append(normalized)
    return out, changes


# ---------------------------------------------------------------------------
# Entity-kind repair (Finding 7)
# ---------------------------------------------------------------------------

# Curated set seeded from 100-ep `my-manual-run4` observations. False negatives
# (missing an org) strictly preferred over false positives.
KNOWN_ORGS: frozenset[str] = frozenset(
    {
        # Podcasts / shows that NER mis-classifies as people.
        "npr",
        "planet money",
        "the daily",
        "the journal",
        "tomorrow's cure",
        "no priors",
        "invest like the best",
        # Media organisations.
        "wsj",
        "the wall street journal",
        # Sponsor-ad companies from the 100-ep top-cluster analysis.
        "ramp",
        "workos",
        "rogo",
        # Common tech / AI orgs referenced in podcast episodes.
        "openai",
        "anthropic",
        "google",
        "meta",
        "amazon",
        "microsoft",
        "tesla",
        "nvidia",
        "apple",
    }
)


def repair_entity_kind(entities: Sequence[dict]) -> Tuple[List[dict], int]:
    """Force ``kind='org'`` on any entity whose name matches ``KNOWN_ORGS``.

    Returns ``(updated_entities, repaired_count)``. All other fields on each
    entity dict are passed through unchanged.
    """
    out: List[dict] = []
    repaired = 0
    for ent in entities:
        if not isinstance(ent, dict):
            out.append(ent)
            continue
        name = str(ent.get("name") or "").strip().lower()
        current_kind = str(ent.get("kind") or "").strip().lower()
        if name and name in KNOWN_ORGS and current_kind != "org":
            new_ent = dict(ent)
            new_ent["kind"] = "org"
            out.append(new_ent)
            repaired += 1
        else:
            out.append(ent)
    return out, repaired


__all__ = [
    "KNOWN_ORGS",
    "normalize_topic_labels",
    "repair_entity_kind",
]

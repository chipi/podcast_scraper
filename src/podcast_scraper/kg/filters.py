"""#652 Part B — deterministic post-extraction validators for KG topics + entities.

Two filters that run on the final topic/entity lists regardless of source
(``provider``, prefilled from mega/extraction bundle):

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

import difflib
import re
from typing import Dict, List, Optional, Sequence, Tuple

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

# Bumped from 4 → 6 after #652 stabilization. Real-corpus audit showed many
# genuine multi-word topics like "AI ethics and public perception" or "global
# oil supply chain" that get mangled at 4 tokens.
_TOPIC_MAX_TOKENS = 6

# Strip punctuation EXCEPT '&' and '-' (preserves "P&I", "AT&T", "ai-agents").
# Apostrophes are handled separately so "China's" → "chinas" not "china s".
_PUNCTUATION_RE = re.compile(r"[^\w\s\-&]")
_APOSTROPHE_RE = re.compile(r"'")
_MULTI_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_topic_label(label: str) -> Optional[str]:
    """Return a lower-cased, stopword-stripped, ≤6-token topic label, or None.

    Design (post-#652-stabilization audit on ``my-manual-run4`` 100-ep corpus):

    * Lowercase + collapse whitespace (always).
    * Strip apostrophes WITHOUT inserting a space ("China's" → "chinas").
    * Strip punctuation EXCEPT ``&`` and ``-`` so "P&I", "AT&T", "ai-agents"
      survive.
    * Cap at ≤6 tokens (was 4 — too aggressive; lost meaning on multi-word
      topics like "AI ethics and public perception").
    * Strip leading + trailing stopwords ONLY. Medial stopwords are KEPT
      because removing them destroyed meaning ("International Group of P&I
      Clubs" → "international group p" was a regression).
    * Dedupe via normalized-form equality at the caller.
    """
    if not label:
        return None
    text = label.lower()
    text = _APOSTROPHE_RE.sub("", text)  # "china's" → "chinas" (no orphan)
    text = _PUNCTUATION_RE.sub(" ", text)
    text = _MULTI_WHITESPACE_RE.sub(" ", text).strip()
    if not text:
        return None
    tokens = text.split(" ")
    # Drop leading stopwords.
    while tokens and tokens[0] in _TOPIC_STOPWORDS:
        tokens = tokens[1:]
    # Drop trailing stopwords ("markets of the" → "markets of" → "markets").
    while tokens and tokens[-1] in _TOPIC_STOPWORDS:
        tokens = tokens[:-1]
    # Cap at max tokens AFTER stopword trimming.
    tokens = tokens[:_TOPIC_MAX_TOKENS]
    # NOTE: medial stopwords are intentionally preserved. The previous
    # implementation stripped them too, which destroyed meaning for topics
    # like "personal journeys of dissent" → "personal journeys dissent".
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


# ---------------------------------------------------------------------------
# Entity name consolidation (#851) — within-episode duplicate-spelling merge
# ---------------------------------------------------------------------------
#
# The KG LLM frequently emits the SAME real entity under two spellings in one
# episode (often the transcript's literal form AND the real-world-correct name),
# e.g. "Burne Hobart" + "Byrne Hobart". Topic normalization (#652) dedupes by
# exact normalized-form equality, which cannot collapse spelling variants. This
# is the deterministic safety net: a conservative, entity-type-aware, within-
# episode merge. The extraction prompt (#851 primary) is responsible for the
# CORRECT surviving spelling; this pass only guarantees one node per entity.
#
# Thresholds are deliberately conservative and centralized for later tuning.
# The within-episode scope is itself a safety feature: two genuinely-different
# similar-named people rarely co-occur in one episode. The acronym guard keeps
# UPS ≠ USPS (the confirmed false-merge landmine).

_PERSON_SURNAME_RATIO = 0.82  # surname similarity to call two person names variants
_PERSON_SURNAME_RATIO_RELAXED = 0.75  # relaxed when first names are identical
_PERSON_FIRST_RATIO = 0.70  # first-name similarity floor
_PERSON_OVERALL_RATIO = 0.80  # whole-string similarity floor
_SINGLE_TOKEN_RATIO = 0.90  # single-token names (non-acronym)
_ORG_RATIO = 0.92  # orgs: strict; only very-high similarity merges
_ACRONYM_MAX_LEN = 5  # single token ≤ this length is treated as an acronym


def _clean_entity_name(name: str) -> str:
    """Lowercase, strip punctuation (keep ``&``/``-``), collapse whitespace."""
    text = _APOSTROPHE_RE.sub("", str(name or "").lower())
    text = _PUNCTUATION_RE.sub(" ", text)
    return _MULTI_WHITESPACE_RE.sub(" ", text).strip()


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _is_acronymish(raw: str, clean: str) -> bool:
    """Single short token, or an all-caps short string — never fuzzy-merged."""
    tokens = clean.split()
    if len(tokens) == 1 and len(clean) <= _ACRONYM_MAX_LEN:
        return True
    raw_compact = re.sub(r"[^A-Za-z]", "", str(raw or ""))
    return bool(raw_compact) and raw_compact.isupper() and len(raw_compact) <= _ACRONYM_MAX_LEN + 1


def _normalize_kind(kind: object) -> str:
    return "org" if str(kind or "").strip().lower() in ("org", "organization") else "person"


def _are_entity_variants(a_raw: str, b_raw: str, kind: str) -> bool:
    """Conservative, type-aware test: are two names the same entity, variant-spelled?"""
    a, b = _clean_entity_name(a_raw), _clean_entity_name(b_raw)
    if not a or not b:
        return False
    if a == b:
        return True
    # Acronym / short-token guard — the UPS vs USPS landmine.
    if _is_acronymish(a_raw, a) or _is_acronymish(b_raw, b):
        return False
    ta, tb = a.split(), b.split()
    if kind == "org":
        return _ratio(a, b) >= _ORG_RATIO
    # Persons: token-structure aware (surname is the discriminator).
    if len(ta) >= 2 and len(tb) >= 2 and len(ta) == len(tb):
        fa, fb, sa, sb = ta[0], tb[0], ta[-1], tb[-1]
        first_ok = (
            fa == fb
            or fa.startswith(fb)
            or fb.startswith(fa)
            or _ratio(fa, fb) >= _PERSON_FIRST_RATIO
        )
        if (
            first_ok
            and _ratio(sa, sb) >= _PERSON_SURNAME_RATIO
            and _ratio(a, b) >= _PERSON_OVERALL_RATIO
        ):
            return True
        # Strong corroboration: identical first name + close surname.
        if fa == fb and _ratio(sa, sb) >= _PERSON_SURNAME_RATIO_RELAXED:
            return True
        return False
    if len(ta) == 1 and len(tb) == 1:
        return _ratio(a, b) >= _SINGLE_TOKEN_RATIO
    return False


def _pick_canonical(members: List[Dict]) -> Dict:
    """Pick the surviving entity: longest name (most complete), tie → lexical.

    Backfills missing/empty fields (e.g. ``description``) from the merged-away
    members so consolidation never drops data. The chosen *name* still wins; only
    absent fields are filled. Note this guarantees *a* spelling, not the correct
    one — the extraction prompt (#851 primary) owns correctness, this pass owns
    deduplication.
    """
    chosen = min(
        members, key=lambda e: (-len(str(e.get("name") or "")), str(e.get("name") or "").lower())
    )
    if len(members) == 1:
        return chosen
    merged = dict(chosen)
    for other in members:
        if other is chosen:
            continue
        for key, value in other.items():
            if key == "name":
                continue
            if merged.get(key) in (None, "") and value not in (None, ""):
                merged[key] = value
    return merged


def consolidate_entity_names(entities: Sequence[dict]) -> Tuple[List[dict], int]:
    """Merge within-episode duplicate-spelling entities. Returns ``(entities, merged_count)``.

    Groups same-kind entities into variant clusters (conservative, type-aware) and
    emits one canonical entity per cluster, preserving the canonical dict's fields
    (``entity_kind``, ``description``, …). First-cluster order preserved.
    """
    clusters: List[Dict] = []  # {"kind": str, "names": [str], "members": [dict]}
    passthrough: List[dict] = []
    for ent in entities:
        name = str(ent.get("name") or "").strip() if isinstance(ent, dict) else ""
        if not name:
            passthrough.append(ent)
            continue
        kind = _normalize_kind(ent.get("entity_kind"))
        for cl in clusters:
            if cl["kind"] == kind and any(_are_entity_variants(name, n, kind) for n in cl["names"]):
                cl["names"].append(name)
                cl["members"].append(ent)
                break
        else:
            clusters.append({"kind": kind, "names": [name], "members": [ent]})

    out: List[dict] = []
    merged = 0
    for cl in clusters:
        out.append(_pick_canonical(cl["members"]))
        merged += len(cl["members"]) - 1
    out.extend(passthrough)
    return out, merged


__all__ = [
    "KNOWN_ORGS",
    "consolidate_entity_names",
    "normalize_topic_labels",
    "repair_entity_kind",
]

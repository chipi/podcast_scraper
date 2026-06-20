"""#1035 — NER pre-pass for KG entity extraction.

Extracts PERSON + ORG candidate spans from the transcript via spaCy NER,
dedupes + caps, and returns a list of hint dicts the KG extraction prompt
can render as a candidate list. Closes the 0% entity-coverage gap surfaced
by the #1033 cohort rerun.

The LLM still owns the final ``entities[]`` decision — these are HINTS,
not a confirmed list. The LLM may reject misclassifications, fix
spellings, and add entities the NER missed.

Decoupled from ``providers/ml/ner_extraction.py`` so it can run without
loading the experiment-eval LLM-NER path.
"""

from __future__ import annotations

import logging
import re
import string
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_KEEP_LABELS = frozenset({"PERSON", "ORG"})

# Reject tokens that are noise: single letters, all-digit spans, all-punct,
# stop-word fragments. Bare initials (e.g. "J.") add no signal and the LLM
# would have to drop them anyway.
_NOISE_TOKEN_RE = re.compile(r"^[A-Z]\.?$|^\d+$")
_PUNCT_TRANSLATE = str.maketrans("", "", string.punctuation)


def _normalize_candidate_text(text: str) -> str:
    """Strip surrounding whitespace + trailing/leading punctuation, collapse spaces."""
    s = (text or "").strip()
    s = s.strip(string.punctuation + string.whitespace)
    # collapse internal whitespace runs
    s = re.sub(r"\s+", " ", s)
    return s


def _is_noise_candidate(text: str) -> bool:
    """True when *text* should be dropped before sending to the LLM."""
    if len(text) < 2:
        return True
    if _NOISE_TOKEN_RE.match(text):
        return True
    # All punctuation after normalization → noise (defensive; normalize should catch).
    stripped = text.translate(_PUNCT_TRANSLATE).strip()
    if not stripped:
        return True
    # Length-bounded conservative noise filter — generic conversational fillers
    # spaCy sometimes mis-tags as PERSON (e.g. "Anyway", "Right"). Whitelist by
    # length: real names are rare under 3 chars.
    if len(stripped) < 3 and stripped.isalpha():
        return True
    return False


def extract_kg_ner_hints(
    transcript: str,
    nlp: Any,
    max_candidates: int,
    *,
    known_org: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Run spaCy NER on *transcript*, return deduped + capped PERSON+ORG hints.

    Args:
        transcript: Cleaned transcript text (same text the LLM will see).
        nlp: spaCy NLP model with NER component (caller resolves via
            ``cfg.ner_model``). When ``None``, returns ``[]`` immediately —
            caller falls back to v4 prompt.
        max_candidates: Hard cap on returned hint count (recommended:
            ``max_entities * 3``, see SPEC_1035_NER_PREPASS_DESIGN.md).
        known_org: Optional canonical ORG name (e.g. podcast/show title from
            RSS metadata). When provided and not already found by spaCy,
            inserted into the candidate list once. Avoids missing the show
            entity when NER doesn't tag it.

    Returns:
        List of ``{"text": str, "label": "PERSON" | "ORG"}`` dicts. Dedup
        applied on case-folded text. Empty list when ``nlp is None``, the
        transcript is empty, or NER raises.
    """
    if nlp is None or not transcript or not transcript.strip():
        return []
    if max_candidates < 1:
        return []

    try:
        doc = nlp(transcript)
    except Exception as exc:
        logger.warning("NER pre-pass failed (falling back to no-hints): %s", exc)
        return []

    seen_keys: set[str] = set()
    out: List[Dict[str, str]] = []

    for ent in doc.ents:
        if ent.label_ not in _KEEP_LABELS:
            continue
        text = _normalize_candidate_text(ent.text)
        if not text or _is_noise_candidate(text):
            continue
        key = text.casefold()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append({"text": text, "label": ent.label_})
        if len(out) >= max_candidates:
            break

    # Inject known ORG (e.g. show title) when not already present.
    if known_org:
        ko = _normalize_candidate_text(known_org)
        if ko and not _is_noise_candidate(ko):
            ko_key = ko.casefold()
            if ko_key not in seen_keys and len(out) < max_candidates:
                seen_keys.add(ko_key)
                out.append({"text": ko, "label": "ORG"})

    return out

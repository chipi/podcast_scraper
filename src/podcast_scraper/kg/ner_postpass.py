"""#1058 chunk 1 — KG Organization post-pass for LLM-free profiles.

Background: under airgapped / airgapped_thin the summary provider is a
local transformer (BART / SummLlama) that cannot extract structured
JSON entities. KG ``Organization`` nodes therefore never land, and the
downstream ``MENTIONS_ORG`` / connectivity surfaces stay empty even
after a real pipeline run on a real corpus. That breaks #1058's CI
testability target: we want a real pipeline → real connectivity →
live server walk that does NOT call a cloud LLM.

This module fixes the Organization-side gap deterministically. It runs
spaCy NER on the GI Insight texts, extracts ``ORG`` spans, sanitises
them, and writes them as first-class ``Organization`` nodes into the
KG artifact. Idempotent — re-running over a KG that already carries
the same org adds nothing.

Wired into ``workflow/metadata_generation.py`` next to the existing
typed-MENTIONS GI post-pass (#1076), gated on
``cfg.kg_organizations_use_ner``. Default off; YAML overlays for
airgapped + airgapped_thin flip it on (the only profiles where this
matters — every other profile gets Org nodes from the LLM directly).

Shape conformance: produces ``Organization`` nodes that satisfy
``docs/architecture/kg/kg.schema.json`` strict (``id: org:{slug}``,
``properties.name`` required, ``properties.role: "mentioned"``).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


_SLUG_NORMALISE = re.compile(r"[^a-z0-9]+")

# Single-token spans this short are almost always false positives ("CO",
# "AI", "TV" etc.). Multi-token spans pass.
_MIN_SINGLE_TOKEN_LEN = 3

# Confidence floor copied from PERSON extraction. spaCy doesn't expose a
# per-entity probability on most pipelines so we use a placeholder.
_DEFAULT_CONFIDENCE = 0.5


def slug_for_org(name: str) -> str:
    """Produce a stable kebab-slug for an Organization name.

    Lowercased, non-alphanumeric collapsed to ``-``, trimmed. Empty
    input returns empty string (caller filters)."""
    if not name:
        return ""
    return _SLUG_NORMALISE.sub("-", name.strip().lower()).strip("-")


def _looks_like_org(span_text: str) -> bool:
    """Reject spaCy ORG spans that are likely false positives.

    The conservative filter mirrors the PERSON-side disambiguation
    guard from #1076 — better to drop a real org than to populate the
    KG with junk that downstream queries treat as a real entity."""
    text = (span_text or "").strip()
    if not text:
        return False
    if len(text) < _MIN_SINGLE_TOKEN_LEN and " " not in text:
        return False
    # All-numeric tokens (years, room numbers) tagged ORG by spaCy
    # occasionally — reject.
    if text.replace(" ", "").isdigit():
        return False
    return True


def extract_org_entities(text: str, nlp: Any) -> List[Tuple[str, float]]:
    """Return a list of unique ``(org_name, confidence)`` pairs from
    spaCy NER on ``text``. Empty list if ``nlp`` is None or extraction
    fails."""
    if not text or nlp is None:
        return []
    try:
        doc = nlp(text)
    except Exception as exc:
        logger.warning("kg.ner_postpass: spaCy parse failed (%s); returning []", exc)
        return []

    seen: Set[str] = set()
    out: List[Tuple[str, float]] = []
    for ent in getattr(doc, "ents", []) or []:
        if getattr(ent, "label_", None) != "ORG":
            continue
        name = (ent.text or "").strip()
        if not _looks_like_org(name):
            continue
        slug = slug_for_org(name)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        out.append((name, _DEFAULT_CONFIDENCE))
    return out


def _existing_org_slugs(kg_artifact: Dict[str, Any]) -> Set[str]:
    """Set of slugs already on the KG artifact (idempotency check)."""
    slugs: Set[str] = set()
    for n in kg_artifact.get("nodes") or []:
        if not isinstance(n, dict):
            continue
        if n.get("type") != "Organization":
            continue
        node_id = n.get("id") or ""
        if isinstance(node_id, str) and node_id.startswith("org:"):
            slugs.add(node_id[len("org:") :])
    return slugs


def _insight_texts(gi_artifact: Dict[str, Any]) -> List[str]:
    """Yield non-empty Insight texts from the GI artifact."""
    out: List[str] = []
    for n in gi_artifact.get("nodes") or []:
        if not isinstance(n, dict) or n.get("type") != "Insight":
            continue
        props = n.get("properties") or {}
        text = (props.get("text") or "").strip()
        if text:
            out.append(text)
    return out


def add_org_entities_from_ner(
    kg_artifact: Dict[str, Any],
    gi_artifact: Dict[str, Any],
    nlp: Any,
) -> int:
    """Run spaCy ORG extraction over every Insight text in ``gi_artifact``
    and add unique ``Organization`` nodes to ``kg_artifact`` in place.

    Returns the count of nodes added. Idempotent — slugs already on the
    KG are skipped. Safe no-op when ``nlp`` is None or no Insights
    exist.
    """
    if nlp is None:
        return 0

    nodes = kg_artifact.setdefault("nodes", [])
    if not isinstance(nodes, list):
        return 0

    existing = _existing_org_slugs(kg_artifact)
    insight_texts = _insight_texts(gi_artifact)
    if not insight_texts:
        return 0

    new_slugs: Set[str] = set()
    added = 0
    for text in insight_texts:
        for name, _confidence in extract_org_entities(text, nlp):
            slug = slug_for_org(name)
            if not slug or slug in existing or slug in new_slugs:
                continue
            new_slugs.add(slug)
            nodes.append(
                {
                    "id": f"org:{slug}",
                    "type": "Organization",
                    "properties": {
                        "name": name,
                        "role": "mentioned",
                    },
                }
            )
            added += 1
    return added


def apply_org_postpass_to_kg_artifact(
    kg_artifact: Dict[str, Any],
    gi_artifact: Dict[str, Any],
    nlp: Any,
) -> int:
    """Public entry point — convenience wrapper around
    ``add_org_entities_from_ner`` with a stable name parallel to
    ``apply_typed_mentions_to_gi_artifact``.

    Returns the count of Organization nodes added.
    """
    return add_org_entities_from_ner(kg_artifact, gi_artifact, nlp)

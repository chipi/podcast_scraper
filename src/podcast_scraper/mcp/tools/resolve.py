"""``resolve_entity`` tool (RFC-095 slice 1) — name → canonical corpus id.

The keystone for the relational/CIL tools: agents start from names ("Sam Altman",
"inflation"); those tools need canonical ids (``person:sam-altman``, ``topic:inflation``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...enrichment.enrichers._loaders import is_unresolved_speaker_placeholder
from ..context import CorpusContext


def _kind_from_id(node_id: str) -> str:
    return node_id.split(":", 1)[0] if ":" in node_id else ""


def resolve_entity(ctx: CorpusContext, name: str, kind: Optional[str] = None) -> Dict[str, Any]:
    """Resolve a freeform *name* to its canonical corpus entity id.

    Returns ``{"query": name, "candidates": [{id, kind, display_name, score, method}]}``
    (0 or 1 candidate — the resolver's best match, with provenance). ``kind`` is advisory
    in this slice: resolution runs across all entity types and the candidate carries its
    actual kind. Empty/whitespace name or no match → empty ``candidates``.
    """
    cleaned = (name or "").strip()
    if not cleaned:
        return {"query": cleaned, "candidates": []}
    # Imported lazily so the package imports without the search/identity deps loaded.
    from ...identity.resolver import get_entity_resolver

    resolver = get_entity_resolver(ctx.corpus_dir)
    detail = resolver.resolve_detail(cleaned)
    candidates: List[Dict[str, Any]] = []
    if detail is not None:
        record = resolver.registry.records.get(detail.id, {})
        # #1193: never resolve a name to an unresolved diarization placeholder
        # (``person:speaker-NN``) — the MCP read surface had no such guard.
        if is_unresolved_speaker_placeholder(str(detail.id), str(record.get("display_name") or "")):
            return {"query": cleaned, "candidates": []}
        candidates.append(
            {
                "id": detail.id,
                "kind": str(record.get("type") or _kind_from_id(detail.id)),
                "display_name": str(record.get("display_name") or ""),
                "score": float(detail.score),
                "method": detail.method,
            }
        )
    return {"query": cleaned, "candidates": candidates}

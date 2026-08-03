"""``corpus_trending`` tool — momentum "what's hot corpus-wide" (RFC-103).

Wraps the same momentum capability as ``GET /api/corpus/trending`` and the consumer
``/api/app/trending`` / discover ranker, so "hot" means one thing across every surface.
Corpus scope only (no per-user engagement): content-velocity from the corpus itself.

Pivot-friendly by design: every returned entity carries a namespaced ``entity_id`` +
``kind`` that seeds the graph/relational tools (``entity_neighborhood``,
``insights_about_entity``, ``topic_entities``, ``show_episodes`` …) — so an agent can go
"what's hot → expand that entity" in two calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..context import CorpusContext

# The momentum kinds the operator global view exposes (parity with corpus_trending route).
_KINDS = ("topic", "cluster", "storyline", "person", "episode", "show", "insight")


def corpus_trending(
    ctx: CorpusContext,
    kind: Optional[str] = None,
    limit: int = 8,
) -> Dict[str, Any]:
    """Top momentum entities corpus-wide, per kind.

    ``kind``: one of topic|cluster|storyline|person|episode|show|insight, or ``None`` for
    all kinds. ``limit`` is per-kind (1-50). Returns ``{as_of_week, kinds: {kind:
    [{entity_id, kind, label, velocity, volume, heating_up, total, series}]}}`` — velocity
    >1 = rising, ``series`` is the weekly sparkline. ``entity_id`` pivots into the graph
    tools.
    """
    from ...server.app_momentum import MomentumConfig, resolve_as_of_week, trending

    lim = max(1, min(50, int(limit)))
    cfg = MomentumConfig.from_dict(None)
    wanted = (kind,) if kind in _KINDS else _KINDS
    if kind is not None and kind not in _KINDS:
        return {
            "as_of_week": resolve_as_of_week(),
            "kinds": {},
            "error": "unknown_kind",
            "detail": f"kind must be one of {list(_KINDS)}",
        }
    out: Dict[str, List[Dict[str, Any]]] = {}
    for k in wanted:
        rows = trending(ctx.corpus_dir, None, kind=k, scope="corpus", limit=lim, config=cfg)
        out[k] = [dict(vars(r)) for r in rows]
    return {"as_of_week": resolve_as_of_week(), "kinds": out, "error": None, "detail": None}

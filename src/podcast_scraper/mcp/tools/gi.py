"""GI / grounded-insight MCP tools — faceted discovery, per-episode insights, compare.

Wrap the shipped GI capabilities behind the same read surface the viewer/player use:
``run_uc5_insight_explorer`` (faceted cross-episode discovery), ``insights_from_gi``
(salience-ranked per-episode insights, ADR-135), and ``compare_subjects`` (RFC-093
two-subject compare with the RFC-072 insight_type filter). All read-only; ids in / ids out
so results keep pivoting into the graph tools.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

from ..context import CorpusContext

if TYPE_CHECKING:
    from ...search.compare import SubjectKind

# id-prefix -> compare SubjectKind (Literal person|topic|episode|feed|show).
_SUBJECT_KIND: Dict[str, "SubjectKind"] = {
    "person": "person",
    "topic": "topic",
    "episode": "episode",
    "ep": "episode",
    "feed": "feed",
    "show": "show",
    "podcast": "show",
}


def explore_insights(
    ctx: CorpusContext,
    topic: Optional[str] = None,
    speaker: Optional[str] = None,
    grounded_only: bool = False,
    min_confidence: Optional[float] = None,
    sort_by: str = "confidence",
    limit: int = 50,
) -> Dict[str, Any]:
    """Faceted cross-episode insight discovery (UC5) — "all insights matching these facets".

    Filter the corpus's grounded insights by ``topic`` and/or ``speaker`` (canonical ids),
    ``grounded_only``, and ``min_confidence``; ``sort_by`` is ``confidence`` or ``time``.
    Returns insights with quotes + provenance ids (pivot each on via ``insight_detail``).
    The discovery complement to the entity/topic-scoped relational tools.
    """
    from ...gi.explore import explore_output_to_rfc_dict, run_uc5_insight_explorer

    order: Literal["confidence", "time"] = "time" if sort_by == "time" else "confidence"
    out = run_uc5_insight_explorer(
        ctx.corpus_dir,
        topic=topic,
        speaker=speaker,
        grounded_only=grounded_only,
        min_confidence=min_confidence,
        sort_by=order,
        limit=max(1, min(500, int(limit))),
    )
    return explore_output_to_rfc_dict(out)


def episode_insights(
    ctx: CorpusContext, metadata_path: str, limit: Optional[int] = None
) -> Dict[str, Any]:
    """Salience-ranked grounded insights for one episode (ADR-135), with supporting quotes.

    ``metadata_path`` comes from a ``list_episodes`` / ``search_corpus`` hit. ``limit`` caps
    to the top-N by salience (drop-tagged excluded). Fills the "insights for THIS episode"
    gap — the relational tools are entity/topic-scoped, not episode-scoped.
    """
    from ...server.app_corpus_access import load_json_artifact
    from ...server.app_gi_view import insights_from_gi
    from ...server.corpus_catalog import catalog_row_for_metadata_path

    row = catalog_row_for_metadata_path(ctx.corpus_dir, metadata_path)
    if row is None or not getattr(row, "has_gi", False):
        return {"episode": metadata_path, "insights": [], "note": "no GI artifact for this episode"}
    artifact = load_json_artifact(ctx.corpus_dir, row.gi_relative_path)
    insights = insights_from_gi(artifact, limit=limit)
    return {"episode": metadata_path, "insights": [i.model_dump() for i in insights]}


def _subject_ref(subject_id: str) -> Any:
    from ...search.compare import SubjectRef

    prefix = subject_id.split(":", 1)[0] if ":" in subject_id else ""
    kind = _SUBJECT_KIND.get(prefix, "topic")
    return SubjectRef(kind=kind, id=subject_id)


def _pack_dict(pack: Any) -> Dict[str, Any]:
    # compare.BriefingPack carries `rendered` + `top_insight_id` as fields (distinct from the
    # search context_pack, which exposes render()/top_insight).
    return {
        "rendered": pack.rendered,
        "token_count": pack.token_count,
        "top_insight_id": pack.top_insight_id,
        "coverage_summary": pack.coverage_summary,
        "confidence_p50": pack.confidence_p50,
    }


def compare_subjects(
    ctx: CorpusContext,
    subject_a: str,
    subject_b: str,
    q: str = "",
    top_k: int = 10,
    max_tokens: int = 2000,
    insight_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compare two subjects across the corpus — briefing pack per side + a judge summary.

    ``subject_a``/``subject_b`` are canonical ids (``person:`` / ``topic:`` / ``show:`` …,
    resolve names first). ``q`` optionally focuses the comparison; ``insight_types`` (e.g.
    ``["claim"]``) narrows both sides symmetrically to those RFC-072 types. Returns
    ``{pack_a, pack_b, judge_summary}`` — the deterministic compare the web Search-v3 view
    uses, so an agent doesn't run two packs and diff them badly.
    """
    from ...search.compare import compare_subjects as _compare

    outcome = _compare(
        ctx.corpus_dir,
        _subject_ref(subject_a),
        _subject_ref(subject_b),
        q=q,
        top_k=max(1, min(50, int(top_k))),
        max_tokens=int(max_tokens),
        insight_types=insight_types or None,
    )
    if outcome.error:
        return {"error": outcome.error, "detail": outcome.detail}
    return {
        "subject_a": subject_a,
        "subject_b": subject_b,
        "pack_a": _pack_dict(outcome.pack_a),
        "pack_b": _pack_dict(outcome.pack_b),
        "judge_summary": outcome.judge_summary,
    }

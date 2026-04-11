"""Episode-level similar results from the vector index (RFC-067 Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from podcast_scraper.search.corpus_scope import normalize_feed_id
from podcast_scraper.search.corpus_search import run_corpus_search

MIN_SIMILARITY_QUERY_LEN = 12
DEFAULT_MAX_QUERY_CHARS = 6000


@dataclass
class SimilarEpisodesOutcome:
    """Result of ``run_similar_episodes`` for HTTP mapping."""

    query_used: str = ""
    items: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    detail: Optional[str] = None


def build_similarity_query(
    summary_title: Optional[str],
    summary_bullets: Sequence[str],
    episode_title: str,
    *,
    max_chars: int = DEFAULT_MAX_QUERY_CHARS,
) -> str:
    """Concatenate summary fields for embedding; fall back to episode title."""
    parts: List[str] = []
    if isinstance(summary_title, str) and summary_title.strip():
        parts.append(summary_title.strip())
    for b in summary_bullets:
        if isinstance(b, str) and b.strip():
            parts.append(b.strip())
    if not parts and isinstance(episode_title, str) and episode_title.strip():
        parts.append(episode_title.strip())
    q = " ".join(parts).strip()
    if len(q) <= max_chars:
        return q
    cut = q[:max_chars]
    if " " in cut:
        return cut.rsplit(" ", 1)[0].strip() or cut.strip()
    return cut.strip()


def episode_scope_key(meta: Dict[str, Any]) -> Optional[tuple[str, str]]:
    """Return ``(normalized_feed_id, episode_id)`` for hit de-duplication, or None."""
    ep = meta.get("episode_id")
    if not isinstance(ep, str) or not ep.strip():
        return None
    fn = normalize_feed_id(meta.get("feed_id"))
    return (fn or "", ep.strip())


def _is_source_scope(
    key: tuple[str, str],
    source_feed_id: str,
    source_episode_id: Optional[str],
) -> bool:
    if not source_episode_id or not str(source_episode_id).strip():
        return False
    return key == (source_feed_id, str(source_episode_id).strip())


def merge_similar_episode_hits(
    enriched_rows: Sequence[Dict[str, Any]],
    *,
    source_feed_id: str,
    source_episode_id: Optional[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Keep the best-scoring hit per (feed_id, episode_id); drop the source episode."""
    best: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in enriched_rows:
        meta = dict(row.get("metadata") or {})
        key = episode_scope_key(meta)
        if key is None:
            continue
        if _is_source_scope(key, source_feed_id, source_episode_id):
            continue
        score = float(row.get("score", 0.0))
        prev = best.get(key)
        if prev is None or score > float(prev.get("score", 0.0)):
            best[key] = {
                "score": score,
                "metadata": meta,
                "text": str(row.get("text") or ""),
                "doc_type": meta.get("doc_type"),
            }
    ranked = sorted(best.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)
    k = max(1, min(int(top_k), 50))
    return ranked[:k]


def run_similar_episodes(
    output_dir: Path,
    *,
    summary_title: Optional[str],
    summary_bullets: Sequence[str],
    episode_title: str,
    source_feed_id: str,
    source_episode_id: Optional[str],
    top_k: int = 8,
    index_path: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> SimilarEpisodesOutcome:
    """Embed summary-derived text, search FAISS, return deduped peer episodes."""
    q = build_similarity_query(
        summary_title,
        summary_bullets,
        episode_title,
    )
    if len(q) < MIN_SIMILARITY_QUERY_LEN:
        return SimilarEpisodesOutcome(
            error="insufficient_text",
            detail="Need longer summary or title for similarity search.",
        )

    fetch_cap = min(100, max(top_k * 10, top_k + 10))
    outcome = run_corpus_search(
        output_dir,
        q,
        doc_types=None,
        feed=None,
        since=None,
        speaker=None,
        grounded_only=False,
        top_k=fetch_cap,
        index_path=index_path,
        embedding_model=embedding_model,
        dedupe_kg_surfaces=False,
    )
    if outcome.error:
        return SimilarEpisodesOutcome(
            error=outcome.error,
            detail=outcome.detail,
            query_used=q,
        )

    merged = merge_similar_episode_hits(
        outcome.results,
        source_feed_id=source_feed_id,
        source_episode_id=source_episode_id,
        top_k=top_k,
    )
    return SimilarEpisodesOutcome(query_used=q, items=merged)

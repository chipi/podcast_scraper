"""Shared semantic corpus search (CLI + HTTP viewer)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

from podcast_scraper import perf_cache
from podcast_scraper.search.cil_lift_overrides import load_cil_lift_overrides
from podcast_scraper.search.cli_handlers import (
    _enrich_hit,
    _hit_passes_cli_filters,
    _metadata_relpath_by_scope_from_corpus,
    _parse_since,
    merged_episode_gi_paths,
)
from podcast_scraper.search.groupings import (
    load_theme_enrichment_map,
    STORYLINE_DOC_TYPE,
    storyline_episode_ids,
    top_storylines_by_member_count,
)
from podcast_scraper.search.hybrid_search import (
    _AUX_DOC_TYPES,
    hybrid_candidates,
    QueryEmbeddingError,
)
from podcast_scraper.search.protocol import SearchResult
from podcast_scraper.search.transcript_chunk_lift import (
    lift_row_if_transcript,
    TranscriptLiftGiCache,
)

logger = logging.getLogger(__name__)

_DEDUPE_KG_DOC_TYPES = frozenset({"kg_entity", "kg_topic"})
_KG_SURFACE_MAX_EPISODE_IDS = 48


def _normalize_kg_surface_text(text: str) -> str:
    """Lowercase, trim, collapse whitespace (aligns with graph Entity/Topic name dedupe idea)."""
    raw = (text or "").lower().strip()
    return re.sub(r"\s+", " ", raw)


def dedupe_kg_surface_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge duplicate ``kg_entity`` / ``kg_topic`` hits that share the same embedded surface text.

    Keeps the highest-scoring row first in list order; adds ``kg_surface_match_count`` and
    ``kg_surface_episode_ids`` when more than one episode contributed.
    """
    out: List[Dict[str, Any]] = []
    kg_winners: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        meta = row.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
        dt = meta.get("doc_type")
        if dt not in _DEDUPE_KG_DOC_TYPES:
            out.append(row)
            continue
        text = str(row.get("text") or "")
        sk = f"{dt}\0{_normalize_kg_surface_text(text)}"
        if sk not in kg_winners:
            kg_winners[sk] = row
            out.append(row)
            continue
        winner = kg_winners[sk]
        wmeta = winner.get("metadata")
        if not isinstance(wmeta, dict):
            wmeta = {}
            winner["metadata"] = wmeta
        ep = meta.get("episode_id")
        new_id = ep.strip() if isinstance(ep, str) and ep.strip() else None

        collected: List[str] = []
        seen: Set[str] = set()
        raw_prev = wmeta.get("kg_surface_episode_ids")
        if isinstance(raw_prev, list):
            for x in raw_prev:
                if isinstance(x, str) and x.strip() and x.strip() not in seen:
                    seen.add(x.strip())
                    collected.append(x.strip())
        else:
            w_ep = wmeta.get("episode_id")
            if isinstance(w_ep, str) and w_ep.strip() and w_ep.strip() not in seen:
                seen.add(w_ep.strip())
                collected.append(w_ep.strip())
        if new_id and new_id not in seen:
            seen.add(new_id)
            collected.append(new_id)
        wmeta["kg_surface_episode_ids"] = collected[:_KG_SURFACE_MAX_EPISODE_IDS]
        wmeta["kg_surface_match_count"] = len(collected)
    return out


@dataclass
class CorpusSearchOutcome:
    """Result of ``run_corpus_search`` (HTTP maps this to JSON; CLI maps to exit codes)."""

    results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    """Machine-readable: ``empty_query``, ``no_index``, ``load_failed``, ``embed_failed``."""
    detail: Optional[str] = None
    """Optional human/debug message (logged server-side; may be omitted in API)."""
    lift_stats: Optional[Dict[str, int]] = None
    """Per-response lift counters; set on success after ``top_k`` slice."""


def _attach_topic_cluster_metadata(rows: List[Dict[str, Any]], corpus_root: Path) -> None:
    """Join ``topic_clusters.json`` into ``kg_topic`` metadata (query-time join)."""
    m = load_theme_enrichment_map(corpus_root)
    if not m:
        return
    for row in rows:
        meta = row.get("metadata")
        if not isinstance(meta, dict):
            continue
        if meta.get("doc_type") != "kg_topic":
            continue
        sid = meta.get("source_id")
        if not isinstance(sid, str) or not sid.strip():
            continue
        info = m.get(sid.strip())
        if info:
            meta["topic_cluster"] = dict(info)


def _attach_storyline_metadata(
    rows: List[Dict[str, Any]], corpus_root: Path
) -> List[Dict[str, Any]]:
    """Join the theme-cluster artifact into ``storyline`` hits, and DROP hits whose cluster is gone.

    A query-time join, the sibling of :func:`_attach_topic_cluster_metadata`, and it is required
    rather than decorative for two reasons found in review:

    * **The indexed row cannot carry these fields.** The aux schema has no label/size/anchor
      columns, and the read path rebuilds hit metadata from a fixed field list
      (``hybrid_search._to_search_result``), so anything the indexer put in the row's metadata dict
      never reaches a caller. Without this join a client sees ``source_id`` only and renders the raw
      ``thc:`` slug as the title.
    * **``anchor_topic_id`` is what a storyline is OPENED by.** There is no storyline endpoint — the
      anchor topic's card IS the storyline — so a link built from the ``thc:`` id 404s. The anchor
      lives in the artifact, not the index.

    Dropping hits for clusters that no longer exist is the read-side half of orphan handling:
    ``thc:`` ids are label-derived, so a relabelled or re-anchored cluster mints a NEW id and the
    old row lingers in the index until a build prunes it. A user must never be offered a storyline
    that is gone.
    """
    summaries = {
        str(s["id"]): s for s in top_storylines_by_member_count(corpus_root, 10_000, min_members=1)
    }
    # The episodes a storyline draws on, so a listening-scoped caller can decide whether it is
    # "mine". A storyline has no single episode, so this union IS its only membership.
    episodes_by_cluster = storyline_episode_ids(corpus_root)
    out: List[Dict[str, Any]] = []
    for row in rows:
        meta = row.get("metadata")
        if not isinstance(meta, dict) or meta.get("doc_type") != STORYLINE_DOC_TYPE:
            out.append(row)
            continue
        sid = meta.get("source_id")
        info = summaries.get(str(sid).strip()) if isinstance(sid, str) else None
        if info is None:
            continue  # orphaned row — the cluster it names no longer exists
        meta["storyline_label"] = info["label"]
        meta["storyline_size"] = info["size"]
        meta["anchor_topic_id"] = info["anchor_topic_id"]
        meta["storyline_episode_ids"] = sorted(episodes_by_cluster.get(str(sid).strip(), ()))
        out.append(row)
    return out


def _lift_stats_for_page(enriched: List[Dict[str, Any]]) -> Dict[str, int]:
    transcript_returned = 0
    lift_applied = 0
    for r in enriched:
        meta_r = r.get("metadata")
        if isinstance(meta_r, dict) and meta_r.get("doc_type") == "transcript":
            transcript_returned += 1
        if isinstance(r.get("lifted"), dict):
            lift_applied += 1
    return {
        "transcript_hits_returned": transcript_returned,
        "lift_applied": lift_applied,
    }


def _enrich_lift_and_slice(
    filtered: List[SearchResult],
    output_dir: Path,
    gi_cache: Dict[str, Path],
    rel_by_scope: Dict[str, str],
    *,
    top_k: int,
    dedupe_kg_surfaces: bool,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    title_cache: Dict[str, tuple[str, str]] = {}
    enriched = [
        _enrich_hit(
            h,
            gi_cache,
            metadata_relpath_by_scope=rel_by_scope,
            corpus_root=output_dir,
            title_cache=title_cache,
        )
        for h in filtered
    ]
    lift_overrides = load_cil_lift_overrides(output_dir)
    lift_cache = TranscriptLiftGiCache()
    for row in enriched:
        meta = row.get("metadata")
        if not isinstance(meta, dict) or meta.get("doc_type") != "transcript":
            continue
        ep = meta.get("episode_id")
        if not isinstance(ep, str) or not ep.strip():
            continue
        gpath = gi_cache.get(ep.strip())
        if gpath is not None and gpath.is_file():
            lift_row_if_transcript(
                row,
                output_dir,
                gpath,
                lift_cache,
                lift_overrides,
            )
    if dedupe_kg_surfaces:
        enriched = dedupe_kg_surface_rows(enriched)
    _attach_topic_cluster_metadata(enriched, output_dir)
    # AFTER the topic join and BEFORE the page slice: dropping an orphaned storyline must
    # not leave a hole in the returned page.
    enriched = _attach_storyline_metadata(enriched, output_dir)
    page = enriched[:top_k]
    return page, _lift_stats_for_page(page)


def _query_scope(
    *,
    doc_types: Optional[Sequence[str]] = None,
    feed: Optional[str],
    since: Optional[str],
    episode_id: Optional[str],
    episode_ids: Optional[Sequence[str]],
) -> Optional[Dict[str, Any]]:
    """The scopes that go INTO the query, as LanceDB prefilters (operator 2026-10-10).

    Filtered afterwards, a scoped search ranked the whole corpus first, kept the top few hundred,
    and lost the scope's own matches on a large corpus — the Brief's search within one episode
    found nothing that way. Each prefilter here is never STRICTER than the exact check that still
    runs on the results (``_hit_passes_cli_filters``): the date is cut a day early, and the show
    match is a looser LIKE of the same substring.

    Speaker and topic name columns only some tables have, so they go in per table
    (:func:`_tier_scope`). A type filter goes in only when every requested type lives in the aux
    table, the one table with a ``doc_type`` column (a "storylines only" search otherwise ranked
    every summary and topic first and kept few storylines).
    """
    scope: Dict[str, Any] = {}
    wanted = sorted(
        {t.strip().lower() for t in doc_types or [] if isinstance(t, str) and t.strip()}
    )
    if wanted and set(wanted) <= _AUX_DOC_TYPES:
        scope["doc_type__in"] = wanted
    if episode_id:
        scope["episode_id"] = episode_id
    if episode_ids is not None:
        scope["episode_id__in"] = [e for e in episode_ids if isinstance(e, str)]
    if isinstance(feed, str) and feed.strip():
        scope["show_id__contains"] = feed.strip()
    since_dt = _parse_since(since) if isinstance(since, str) and since.strip() else None
    if since_dt is not None:
        scope["publish_date__gte"] = (since_dt - timedelta(days=1)).date().isoformat()
    return scope or None


class _InsightScope(NamedTuple):
    """One insight's scope keys, as the result filters read them from its GI file."""

    episode_id: str
    insight_id: str
    #: ``speaker_id`` of every Quote it is SUPPORTED_BY, lowercased
    #: (``_insight_passes_speaker_filter``).
    speakers: tuple[str, ...]
    #: Every ``topic:`` id it is ABOUT, and those topics' label / name / display_name, lowercased
    #: (``_insight_passes_topic_filter``).
    topics: tuple[str, ...]


def _insight_scopes_from_gi(episode_id: str, artifact: Dict[str, Any]) -> List[_InsightScope]:
    nodes = {n.get("id"): n for n in artifact.get("nodes") or [] if isinstance(n, dict)}
    quotes_of: Dict[Any, List[Any]] = {}
    topics_of: Dict[Any, List[str]] = {}
    for e in artifact.get("edges") or []:
        if not isinstance(e, dict):
            continue
        frm, to = e.get("from"), e.get("to")
        if e.get("type") == "SUPPORTED_BY":
            quotes_of.setdefault(frm, []).append(to)
        elif e.get("type") == "ABOUT" and isinstance(to, str) and to.startswith("topic:"):
            topics_of.setdefault(frm, []).append(to)
    out: List[_InsightScope] = []
    for nid, node in nodes.items():
        if node.get("type") != "Insight" or not isinstance(nid, str):
            continue
        speakers = []
        for qid in quotes_of.get(nid, []):
            q = nodes.get(qid)
            sid = ((q or {}).get("properties") or {}).get("speaker_id")
            if q and q.get("type") == "Quote" and isinstance(sid, str):
                speakers.append(sid.lower())
        topics = []
        for tid in topics_of.get(nid, []):
            topics.append(tid.lower())
            props = (nodes.get(tid) or {}).get("properties")
            if isinstance(props, dict):
                for key in ("label", "name", "display_name"):
                    v = props.get(key)
                    if isinstance(v, str):
                        topics.append(v.lower())
        out.append(_InsightScope(episode_id, nid, tuple(speakers), tuple(topics)))
    return out


#: ``id(GI path map) -> (that map, its scopes)``; see :func:`_cached_insight_scopes`.
_SCOPES_BY_PATH_MAP: Dict[int, Tuple[Dict[str, Path], List["_InsightScope"]]] = {}


def _cached_insight_scopes(output_dir: Path) -> List[_InsightScope]:
    """Every insight's speaker and topic keys across the corpus, from the same GI files the result
    filters read (~1 ms per episode), built once per corpus generation.

    Memoized on the GI path map itself: :func:`cached_episode_gi_paths` hands back the same map
    object until the corpus changes, so a different map is a new generation. Keying on the corpus
    path instead would turn the caller's path into a filesystem path here (CodeQL
    py/path-injection) for no gain — the map already carries the generation.
    """
    paths = cached_episode_gi_paths(output_dir)
    hit = _SCOPES_BY_PATH_MAP.get(id(paths))
    if hit is not None and hit[0] is paths:
        return hit[1]
    rows: List[_InsightScope] = []
    for episode_id, path in paths.items():
        try:
            artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(artifact, dict):
            rows.extend(_insight_scopes_from_gi(episode_id, artifact))
    if len(_SCOPES_BY_PATH_MAP) >= 4:  # old generations: their maps are gone from perf_cache
        _SCOPES_BY_PATH_MAP.clear()
    _SCOPES_BY_PATH_MAP[id(paths)] = (paths, rows)
    return rows


def _tier_scope(
    output_dir: Path, *, speaker: Optional[str], topic: Optional[str]
) -> Optional[Dict[str, Optional[Dict[str, Any]]]]:
    """A speaker or topic scope as per-table prefilters (``SearchQuery.tier_filters``).

    Filtered afterwards, a speaker search ranked the whole corpus, kept a few hundred rows, and
    lost that speaker's own insights whenever others outranked them. Each table gets exactly what
    the result filter (``_hit_passes_cli_filters``) would let through from it, never less — the
    exact check still runs on the results:

    * insights: the ids whose GI quotes / ABOUT topics match, resolved here from the GI files;
    * aux: a topic scope keeps ``kg_topic`` rows whose id contains it (the only aux rows a topic
      filter passes); a speaker scope leaves aux to the result filter;
    * transcript tables: a topic scope can match nothing there, so they are not read; a speaker
      scope passes them unchanged, as the result filter does.
    """
    if not speaker and not topic:
        return None
    s_key, t_key = (speaker or "").lower(), (topic or "").lower()
    matched = [
        r
        for r in _cached_insight_scopes(output_dir)
        if (not speaker or any(s_key in s for s in r.speakers))
        and (not topic or any(t_key in t for t in r.topics))
    ]
    scope: Dict[str, Optional[Dict[str, Any]]] = {
        "insight": (
            {
                "episode_id__in": sorted({r.episode_id for r in matched}),
                "source_id__in": sorted({r.insight_id for r in matched}),
            }
            if matched
            else None
        )
    }
    if topic:
        scope["aux"] = {"doc_type": "kg_topic", "source_id__contains": topic}
        scope["segment"] = None
        scope["segment_nonen"] = None
    return scope


def run_corpus_search(
    output_dir: Path,
    query: str,
    *,
    doc_types: Optional[Sequence[str]] = None,
    feed: Optional[str] = None,
    since: Optional[str] = None,
    speaker: Optional[str] = None,
    topic: Optional[str] = None,
    episode_id: Optional[str] = None,
    episode_ids: Optional[Sequence[str]] = None,
    grounded_only: bool = False,
    top_k: int = 10,
    index_path: Optional[str] = None,
    embedding_model: Optional[str] = None,
    dedupe_kg_surfaces: bool = True,
) -> CorpusSearchOutcome:
    """Embed ``query``, search the LanceDB index, apply metadata filters, return enriched rows."""
    q = query.strip()
    if not q:
        return CorpusSearchOutcome(error="empty_query")

    top_k = max(1, min(int(top_k), 100))
    types_norm: Optional[List[str]] = None
    if doc_types:
        types_norm = [x.strip().lower() for x in doc_types if isinstance(x, str) and x.strip()]
        if not types_norm:
            types_norm = None

    # ADR-099 / #995: the LanceDB two-tier index is the single search path — no FAISS
    # fallback. ``hybrid_candidates`` returns None when there is no usable index, and
    # raises ``QueryEmbeddingError`` when the index is fine but the query couldn't be
    # embedded (model missing/offline). These are distinct: re-indexing fixes the first,
    # not the second — so surface them as different error codes.
    try:
        candidates = hybrid_candidates(
            output_dir,
            q,
            top_k=top_k,
            doc_types=doc_types,
            embedding_model=embedding_model,
            filters=_query_scope(
                doc_types=doc_types,
                feed=feed,
                since=since,
                episode_id=episode_id,
                episode_ids=episode_ids,
            ),
            tier_filters=_tier_scope(output_dir, speaker=speaker, topic=topic),
        )
    except QueryEmbeddingError as exc:
        return CorpusSearchOutcome(
            error="embed_failed",
            detail=f"query embedding failed (model missing or offline): {exc}",
        )
    if candidates is None:
        return CorpusSearchOutcome(error="no_index", detail="no LanceDB index (run `cli index`)")
    if episode_ids is not None:
        wanted = {e for e in episode_ids if isinstance(e, str)}
        candidates = [c for c in candidates if c.metadata.get("episode_id") in wanted]
    return _filter_and_enrich(
        candidates,
        output_dir,
        types_norm=types_norm,
        feed=feed,
        since=since,
        speaker=speaker,
        topic=topic,
        episode_id=episode_id,
        grounded_only=grounded_only,
        top_k=top_k,
        dedupe_kg_surfaces=dedupe_kg_surfaces,
        collect_cap=len(candidates),
    )


def cached_episode_gi_paths(output_dir: Path) -> Dict[str, Path]:
    """:func:`merged_episode_gi_paths`, cached per corpus generation (``corpus_mtime``).

    Every search read these two maps by walking the WHOLE corpus (``discover_metadata_files``
    plus one JSON read per episode) -- twice per request, and most of a search's time: on prod
    2026-10-06 a search took ~4 s with them and ~0.4 s once they were cached. Worse, a directory
    walk hands the GIL back and forth on every filesystem call, so beside a CPU-bound thread (the
    cache warmer's entity-id map after a restart) each handoff waits out the switch interval:
    the same search took 186-195 s until the warmer finished, and the player post-deploy smoke
    failed on 504s. Read-only maps; callers only ``.get`` from them.
    """
    root = Path(output_dir)
    paths: Dict[str, Path] = perf_cache.get_or_compute(
        "search_episode_gi_paths",
        str(root.resolve()),
        perf_cache.corpus_mtime(root),
        lambda: merged_episode_gi_paths(output_dir),
    )
    return paths


def cached_metadata_relpath_by_scope(output_dir: Path) -> Dict[str, str]:
    """:func:`_metadata_relpath_by_scope_from_corpus`, cached the same way, for the same reason
    (see :func:`cached_episode_gi_paths`)."""
    root = Path(output_dir)
    relpaths: Dict[str, str] = perf_cache.get_or_compute(
        "search_metadata_relpath_by_scope",
        str(root.resolve()),
        perf_cache.corpus_mtime(root),
        lambda: _metadata_relpath_by_scope_from_corpus(output_dir),
    )
    return relpaths


def _filter_and_enrich(
    hits: Sequence[SearchResult],
    output_dir: Path,
    *,
    types_norm: Optional[List[str]],
    feed: Optional[str],
    since: Optional[str],
    speaker: Optional[str],
    topic: Optional[str],
    episode_id: Optional[str],
    grounded_only: bool,
    top_k: int,
    dedupe_kg_surfaces: bool,
    collect_cap: int,
) -> CorpusSearchOutcome:
    """Apply metadata filters + enrich/lift/dedupe a candidate list (backend-agnostic).

    Drives the LanceDB two-tier hybrid retrieval path (RFC-090 Phase 2 / ADR-099) — the
    single search path since FAISS was retired (#995) — producing the enriched response shape.
    """
    since_dt = _parse_since(since) if isinstance(since, str) and since.strip() else None
    gi_cache = cached_episode_gi_paths(output_dir)
    rel_by_scope = cached_metadata_relpath_by_scope(output_dir)
    filtered: List[SearchResult] = []
    for h in hits:
        dt = h.metadata.get("doc_type")
        if types_norm and len(types_norm) > 1:
            if dt not in types_norm:
                continue
        elif types_norm and len(types_norm) == 1:
            if dt != types_norm[0]:
                continue

        if _hit_passes_cli_filters(
            h,
            feed_substr=feed,
            since_dt=since_dt,
            speaker_substr=speaker,
            topic_substr=topic,
            episode_id=episode_id,
            grounded_only=grounded_only,
            gi_by_episode=gi_cache,
        ):
            filtered.append(h)
        if len(filtered) >= collect_cap:
            break

    enriched, lift_stats = _enrich_lift_and_slice(
        filtered,
        output_dir,
        gi_cache,
        rel_by_scope,
        top_k=top_k,
        dedupe_kg_surfaces=dedupe_kg_surfaces,
    )
    return CorpusSearchOutcome(results=enriched, lift_stats=lift_stats)

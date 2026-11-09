"""CLI handlers for semantic corpus search (`search` and `index` commands, #484 Step 4)."""

from __future__ import annotations

import argparse
import json
import logging
import os
from argparse import Namespace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple

from podcast_scraper import config
from podcast_scraper.search.corpus_scope import (
    discover_metadata_files,
    gi_map_lookup_key_from_vector_meta,
    index_fingerprint_scope_key,
    normalize_feed_id,
)
from podcast_scraper.search.indexer import _scope_display_titles, index_corpus
from podcast_scraper.search.protocol import SearchResult
from podcast_scraper.utils.log_redaction import format_exception_for_log
from podcast_scraper.utils.path_validation import (
    safe_relpath_under_corpus_root,
    safe_resolve_directory,
)
from podcast_scraper.workflow.metadata_generation import _determine_gi_path

# Align with gi.explore exit codes
EXIT_SUCCESS = 0
EXIT_INVALID_ARGS = 2
EXIT_NO_ARTIFACTS = 3
EXIT_LEDGER_FAILED = 4

_PLACEHOLDER_RSS = "https://example.com/.podcast-scraper/vector-cli-placeholder"


def _minimal_vector_config(
    output_dir: str,
    *,
    vector_index_path: Optional[str] = None,
    vector_embedding_model: Optional[str] = None,
    vector_embedding_endpoint: Optional[str] = None,
    vector_embedding_provider: Optional[str] = None,
    vector_index_types: Optional[List[str]] = None,
) -> config.Config:
    """Build a Config sufficient for embedding + index paths (RSS is unused)."""
    payload: Dict[str, Any] = {
        "rss_url": _PLACEHOLDER_RSS,
        "output_dir": output_dir,
        "vector_search": True,
    }
    if vector_index_path is not None:
        payload["vector_index_path"] = vector_index_path
    if vector_embedding_model is not None:
        payload["vector_embedding_model"] = vector_embedding_model
    if vector_embedding_endpoint is not None:
        payload["vector_embedding_endpoint"] = vector_embedding_endpoint
    if vector_embedding_provider is not None:
        payload["vector_embedding_provider"] = vector_embedding_provider
    if vector_index_types is not None:
        payload["vector_index_types"] = vector_index_types
    return config.Config(**payload)


def _resolve_index_dir(output_dir: Path, index_path: Optional[str]) -> Path:
    base = safe_resolve_directory(output_dir)
    if base is None:
        return Path(os.path.normpath(str(output_dir))) / "search"
    base_s = os.path.normpath(str(base))
    if not index_path or not str(index_path).strip():
        default = os.path.normpath(os.path.join(base_s, "search"))
        # codeql[py/path-injection] -- normpath on hardcoded child.
        return Path(default)
    safe = safe_relpath_under_corpus_root(base, str(index_path).strip())
    if safe is None:
        default = os.path.normpath(os.path.join(base_s, "search"))
        return Path(default)
    # codeql[py/path-injection] -- safe from normpath+startswith in safe_relpath.
    return Path(safe)


def _episode_to_gi_path(output_dir: Path) -> Dict[str, Path]:
    """Map episode_id -> resolved gi.json path."""
    root = safe_resolve_directory(output_dir)
    if root is None:
        return {}
    root_s = os.path.normpath(str(root))
    safe_prefix = root_s + os.sep
    meta_dir_s = os.path.normpath(os.path.join(root_s, "metadata"))
    # codeql[py/path-injection] -- meta_dir_s from normpath on hardcoded child.
    if not meta_dir_s.startswith(safe_prefix):
        return {}
    if not os.path.isdir(meta_dir_s):
        return {}
    out: Dict[str, Path] = {}
    meta_dir = Path(meta_dir_s)
    for meta_path in sorted(meta_dir.glob("*.metadata.json")):
        mp_s = os.path.normpath(str(meta_path))
        if not mp_s.startswith(safe_prefix):
            continue
        try:
            with open(mp_s, encoding="utf-8") as _fh:
                doc = json.loads(_fh.read())
        except (OSError, json.JSONDecodeError):
            continue
        ep = doc.get("episode") or {}
        eid = ep.get("episode_id")
        if not isinstance(eid, str) or not eid:
            continue
        gi = doc.get("grounded_insights")
        if isinstance(gi, dict):
            rel = gi.get("artifact_path")
            if isinstance(rel, str) and rel.strip():
                safe = safe_relpath_under_corpus_root(root, rel.strip())
                if safe is not None:
                    safe = os.path.normpath(safe)
                    if safe.startswith(safe_prefix) and os.path.isfile(safe):
                        out[eid] = Path(safe)
                        continue
        gi_path_s = os.path.normpath(_determine_gi_path(str(meta_path)))
        # codeql[py/path-injection] -- gi_path_s from normpath; guard below.
        if gi_path_s.startswith(safe_prefix) and os.path.isfile(gi_path_s):
            out[eid] = Path(gi_path_s)
    return out


def _episode_to_gi_path_from_discovered(output_dir: Path) -> Dict[str, Path]:
    """Map episode_id -> ``gi.json`` for every discovered ``*.metadata.json`` under the corpus.

    Covers feed-nested layouts (``feeds/.../metadata/``) where top-level ``metadata/`` is empty
    or absent (#528 offset verification on acceptance / multi-feed outputs).
    """
    root = safe_resolve_directory(output_dir)
    if root is None:
        return {}
    root_s = os.path.normpath(str(root))
    safe_prefix = root_s + os.sep
    out: Dict[str, Path] = {}
    for meta_path in discover_metadata_files(output_dir):
        if meta_path.suffix.lower() != ".json":
            continue
        mp_s = os.path.normpath(str(meta_path))
        if not mp_s.startswith(safe_prefix):
            continue
        try:
            with open(meta_path, encoding="utf-8") as _fh:
                doc = json.loads(_fh.read())
        except (OSError, json.JSONDecodeError):
            continue
        ep = doc.get("episode") or {}
        eid = ep.get("episode_id")
        if not isinstance(eid, str) or not eid:
            continue
        gi = doc.get("grounded_insights")
        if isinstance(gi, dict):
            rel = gi.get("artifact_path")
            if isinstance(rel, str) and rel.strip():
                safe = safe_relpath_under_corpus_root(root, rel.strip())
                if safe is not None:
                    safe = os.path.normpath(safe)
                    if safe.startswith(safe_prefix) and os.path.isfile(safe):
                        out[eid] = Path(safe)
                        continue
        gi_path_s = os.path.normpath(_determine_gi_path(str(meta_path)))
        if gi_path_s.startswith(safe_prefix) and os.path.isfile(gi_path_s):
            out[eid] = Path(gi_path_s)
    return out


def merged_episode_gi_paths(output_dir: Path) -> Dict[str, Path]:
    """Map episode_id -> ``gi.json`` for search, filters, and offset verification.

    Starts from all discovered ``*.metadata.json`` under the corpus (feed-nested layouts),
    then applies the legacy top-level ``metadata/*.metadata.json`` scan. Flat entries
    override discovered keys when both exist (same merge as ``verify-gil-chunk-offsets``).
    """
    out = _episode_to_gi_path_from_discovered(output_dir)
    for eid, pth in _episode_to_gi_path(output_dir).items():
        out[eid] = pth
    return out


def _metadata_relpath_by_scope_from_corpus(output_dir: Path) -> Dict[str, str]:
    """Map fingerprint scope key -> corpus-relative ``*.metadata.json`` path (POSIX).

    Backfills ``source_metadata_relative_path`` on search hits when index rows predate
    that field (incremental index without re-embed, or older indexer builds).
    """
    root = safe_resolve_directory(output_dir)
    if root is None:
        return {}
    root_s = os.path.normpath(str(root))
    safe_prefix = root_s + os.sep
    out: Dict[str, str] = {}
    for meta_path in discover_metadata_files(root):
        mp_s = os.path.normpath(str(meta_path))
        # codeql[py/path-injection] -- normpath+startswith guard.
        if not mp_s.startswith(safe_prefix) and mp_s != root_s:
            continue
        try:
            with open(mp_s, encoding="utf-8") as _fh:
                doc = json.loads(_fh.read())
        except (OSError, json.JSONDecodeError):
            continue
        ep = doc.get("episode") or {}
        eid = ep.get("episode_id")
        if not isinstance(eid, str) or not eid:
            continue
        feed = doc.get("feed") or {}
        fid = feed.get("feed_id")
        key = index_fingerprint_scope_key(normalize_feed_id(fid), eid)
        rel = os.path.relpath(mp_s, root_s).replace("\\", "/")
        if rel.startswith(".."):
            continue
        out[key] = rel
        ep_only = index_fingerprint_scope_key(None, eid)
        if ep_only not in out:
            out[ep_only] = rel
    return out


def _quotes_for_insight(
    artifact: Dict[str, Any], insight_id: str, people: Any = None
) -> List[Dict[str, Any]]:
    """The insight's supporting quotes, each with the name the episode view shows for its speaker
    (``speaker_name``, spelled through ``people``; RFC-131 S6c). ``speaker_id`` stays the episode's.
    """
    from podcast_scraper.read_hooks import quote_speaker_name

    quotes: List[Dict[str, Any]] = []
    nodes = {n.get("id"): n for n in artifact.get("nodes") or [] if isinstance(n, dict)}
    spoken_by = {
        e.get("from"): e.get("to")
        for e in artifact.get("edges") or []
        if isinstance(e, dict) and e.get("type") == "SPOKEN_BY"
    }
    for e in artifact.get("edges") or []:
        if not isinstance(e, dict):
            continue
        if e.get("type") != "SUPPORTED_BY":
            continue
        if e.get("from") != insight_id:
            continue
        qid = e.get("to")
        if not isinstance(qid, str):
            continue
        qnode = nodes.get(qid)
        if not qnode or qnode.get("type") != "Quote":
            continue
        props = qnode.get("properties") or {}
        quotes.append(
            {
                "quote_id": qid,
                "text": props.get("text"),
                "speaker_id": props.get("speaker_id"),
                "speaker_name": quote_speaker_name(artifact, props, spoken_by.get(qid), people),
                "char_start": props.get("char_start"),
                "char_end": props.get("char_end"),
                "timestamp_start_ms": props.get("timestamp_start_ms"),
                "timestamp_end_ms": props.get("timestamp_end_ms"),
            }
        )
    return quotes


def _speaker_matches(sid: Any, key: str, id_map: Optional[Dict[str, str]]) -> bool:
    """A speaker scope matches the id, or its group's canonical id (RFC-131 S6c)."""
    if not isinstance(sid, str):
        return False
    canonical = (id_map or {}).get(sid) or ""
    return key in sid.lower() or (bool(canonical) and key in canonical.lower())


def _insight_passes_speaker_filter(
    artifact: Dict[str, Any],
    insight_id: str,
    speaker_key: str,
    id_map: Optional[Dict[str, str]] = None,
) -> bool:
    key = speaker_key.lower()
    for q in _quotes_for_insight(artifact, insight_id):
        if _speaker_matches(q.get("speaker_id"), key, id_map):
            return True
    return False


def _insight_passes_topic_filter(
    artifact: Dict[str, Any],
    insight_id: str,
    topic_key: str,
) -> bool:
    """True when the insight ABOUTs a topic whose id or label contains ``topic_key``.

    Server-side equivalent of the client-side filter that shipped in the Search v3
    §S1 merge; folds SearchTopicChip out of client-side-only territory.
    """
    key = topic_key.lower()
    topic_ids: set[str] = set()
    for e in artifact.get("edges", []) or []:
        if not isinstance(e, dict):
            continue
        if e.get("type") == "ABOUT" and e.get("from") == insight_id:
            to = e.get("to")
            if isinstance(to, str) and to.startswith("topic:"):
                topic_ids.add(to)
    if not topic_ids:
        return False
    for nid in topic_ids:
        if key in nid.lower():
            return True
    for n in artifact.get("nodes", []) or []:
        if not isinstance(n, dict):
            continue
        if n.get("id") not in topic_ids:
            continue
        props = n.get("properties")
        if isinstance(props, dict):
            for field in ("label", "name", "display_name"):
                v = props.get(field)
                if isinstance(v, str) and key in v.lower():
                    return True
    return False


def _parse_since(s: str) -> Optional[datetime]:
    try:
        d = date.fromisoformat(s.strip())
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    except ValueError:
        return None


def _hit_passes_cli_filters(
    hit: SearchResult,
    *,
    feed_substr: Optional[str],
    since_dt: Optional[datetime],
    speaker_substr: Optional[str],
    topic_substr: Optional[str] = None,
    episode_id: Optional[str] = None,
    grounded_only: bool,
    gi_by_episode: Dict[str, Path],
    id_map: Optional[Dict[str, str]] = None,
) -> bool:
    meta = hit.metadata
    if grounded_only and meta.get("doc_type") == "insight":
        if not meta.get("grounded"):
            return False

    if feed_substr:
        fid = meta.get("feed_id")
        if not isinstance(fid, str) or feed_substr.lower() not in fid.lower():
            return False

    # Search v3 §S6 — exact ``episode_id`` scope. Enables "Search within this
    # episode" rail launcher from EpisodeDetailPanel. Corpus-stable id match
    # (not substring): the rail always knows the exact target.
    if episode_id:
        ep = meta.get("episode_id")
        if not isinstance(ep, str) or ep != episode_id:
            return False

    if since_dt:
        pub = meta.get("publish_date")
        if isinstance(pub, str):
            try:
                pdt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                if pdt.tzinfo is None:
                    pdt = pdt.replace(tzinfo=timezone.utc)
                if pdt < since_dt:
                    return False
            except ValueError:
                return False
        else:
            return False

    if speaker_substr and meta.get("doc_type") == "quote":
        if not _speaker_matches(meta.get("speaker_id"), speaker_substr.lower(), id_map):
            return False

    if speaker_substr and meta.get("doc_type") == "insight":
        ep = meta.get("episode_id")
        if not isinstance(ep, str):
            return False
        gpath = gi_by_episode.get(ep)
        if not gpath or not gpath.is_file():
            return False
        try:
            art = cast(Dict[str, Any], json.loads(gpath.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            return False
        src = meta.get("source_id")
        if not isinstance(src, str):
            return False
        if not _insight_passes_speaker_filter(art, src, speaker_substr, id_map):
            return False

    # Topic substring filter (extracted to keep complexity below flake8's C901 ceiling).
    if topic_substr and not _hit_passes_topic_filter(meta, topic_substr, gi_by_episode):
        return False

    return True


def _hit_passes_topic_filter(
    meta: Dict[str, Any],
    topic_substr: str,
    gi_by_episode: Dict[str, Path],
) -> bool:
    """Topic filter branch for _hit_passes_cli_filters (Search v3 §S1 follow-up).

    - ``kg_topic`` — match on ``source_id`` (slug) + ``topic_label``.
    - ``insight`` — read GI artifact, match ABOUT-edge topic id/label.
    - other doc_types — drop (no topic linkage on the row).
    """
    dt = meta.get("doc_type")
    if dt == "kg_topic":
        src = meta.get("source_id")
        label = meta.get("topic_label") or meta.get("label") or ""
        key = topic_substr.lower()
        return bool(
            (isinstance(src, str) and key in src.lower())
            or (isinstance(label, str) and key in label.lower())
        )
    if dt == "insight":
        ep = meta.get("episode_id")
        src = meta.get("source_id")
        if not isinstance(ep, str) or not isinstance(src, str):
            return False
        gpath = gi_by_episode.get(ep)
        if not gpath or not gpath.is_file():
            return False
        try:
            art = cast(Dict[str, Any], json.loads(gpath.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            return False
        return _insight_passes_topic_filter(art, src, topic_substr)
    # Segments / quotes / aux rows don't carry topic linkage; drop when set (same shape as speaker).
    return False


def _backfill_display_titles_from_corpus(
    meta: Dict[str, Any],
    corpus_root: Path,
    cache: Dict[str, Tuple[str, str]],
) -> None:
    """Set ``episode_title`` / ``feed_title`` from ``*.metadata.json`` when missing on the hit.

    Uses the same title resolution as the vector indexer so older index rows (indexed before
    titles were stamped) still get human labels in the viewer without a full rebuild.
    """
    if meta.get("episode_title") and meta.get("feed_title"):
        return
    rel = meta.get("source_metadata_relative_path")
    if not isinstance(rel, str) or not rel.strip():
        return
    rel_key = rel.strip().replace("\\", "/")
    if rel_key not in cache:
        safe_path = safe_relpath_under_corpus_root(corpus_root, rel_key)
        if safe_path is None:
            cache[rel_key] = ("", "")
        else:
            safe_path = os.path.normpath(safe_path)
            root_s = os.path.normpath(str(corpus_root))
            if not safe_path.startswith(root_s + os.sep) or not os.path.isfile(safe_path):
                cache[rel_key] = ("", "")
            else:
                try:
                    with open(safe_path, encoding="utf-8") as fh:
                        doc = json.loads(fh.read())
                    ep = doc.get("episode") if isinstance(doc.get("episode"), dict) else {}
                    feed = doc.get("feed") if isinstance(doc.get("feed"), dict) else {}
                    et, ft = _scope_display_titles(doc, ep, feed)
                    cache[rel_key] = (et, ft)
                except (OSError, json.JSONDecodeError, TypeError):
                    cache[rel_key] = ("", "")
    et, ft = cache[rel_key]
    if not meta.get("episode_title") and et:
        meta["episode_title"] = et
    if not meta.get("feed_title") and ft:
        meta["feed_title"] = ft


def _enrich_hit(
    hit: SearchResult,
    gi_by_episode: Dict[str, Path],
    *,
    metadata_relpath_by_scope: Optional[Dict[str, str]] = None,
    corpus_root: Optional[Path] = None,
    title_cache: Optional[Dict[str, Tuple[str, str]]] = None,
    people: Any = None,
    docs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One search row. ``people`` spells supporting quotes' speakers (RFC-131 S6c); ``docs`` is a
    per-query cache of parsed GI artifacts, so several hits from one episode parse it once."""
    meta = dict(hit.metadata)
    if metadata_relpath_by_scope:
        smp = meta.get("source_metadata_relative_path")
        if not (isinstance(smp, str) and smp.strip()):
            scope_k = gi_map_lookup_key_from_vector_meta(meta)
            rel: Optional[str] = None
            if scope_k:
                rel = metadata_relpath_by_scope.get(scope_k)
            if not rel:
                ep = meta.get("episode_id")
                if isinstance(ep, str) and ep.strip():
                    rel = metadata_relpath_by_scope.get(
                        index_fingerprint_scope_key(None, ep.strip())
                    )
                elif isinstance(ep, (int, float)):
                    rel = metadata_relpath_by_scope.get(
                        index_fingerprint_scope_key(None, str(ep).strip())
                    )
            if isinstance(rel, str) and rel.strip():
                meta["source_metadata_relative_path"] = rel.strip()
    if corpus_root is not None and title_cache is not None:
        _backfill_display_titles_from_corpus(meta, corpus_root, title_cache)
    text = str(meta.pop("text", "") or "")
    row: Dict[str, Any] = {
        "doc_id": hit.doc_id,
        "score": hit.score,
        "metadata": meta,
        "text": text,
    }
    if meta.get("doc_type") != "insight":
        return row
    ep = meta.get("episode_id")
    src = meta.get("source_id")
    if not isinstance(ep, str) or not isinstance(src, str):
        return row
    gpath = gi_by_episode.get(ep)
    if not gpath or not gpath.is_file():
        return row
    key = str(gpath)
    if docs is not None and key in docs:
        art = docs[key]
    else:
        try:
            art = cast(Dict[str, Any], json.loads(gpath.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            art = None
        if docs is not None:
            docs[key] = art
    if not isinstance(art, dict):
        return row
    row["supporting_quotes"] = _quotes_for_insight(art, src, people)
    return row


def run_search_cli(args: Namespace, logger: logging.Logger) -> int:
    """Run semantic search over the LanceDB corpus index."""
    qparts: Sequence[str] = getattr(args, "query", None) or []
    query = " ".join(qparts).strip()
    if not query:
        logger.error("search: provide a non-empty query")
        return EXIT_INVALID_ARGS
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        logger.error("search: --output-dir is required")
        return EXIT_INVALID_ARGS

    out = Path(output_dir)
    from podcast_scraper.search.corpus_search import run_corpus_search

    doc_type_raw = getattr(args, "doc_type", None)
    doc_types: Optional[List[str]] = None
    if isinstance(doc_type_raw, str) and doc_type_raw.strip():
        doc_types = [doc_type_raw.strip()]

    outcome = run_corpus_search(
        out,
        query,
        doc_types=doc_types,
        feed=getattr(args, "feed", None),
        since=getattr(args, "since", None),
        speaker=getattr(args, "speaker", None),
        grounded_only=bool(getattr(args, "grounded_only", False)),
        top_k=max(1, int(getattr(args, "top_k", 10) or 10)),
        index_path=getattr(args, "index_path", None),
        embedding_model=getattr(args, "embedding_model", None),
        dedupe_kg_surfaces=not bool(getattr(args, "no_dedupe_kg_surfaces", False)),
    )

    if outcome.error == "empty_query":
        logger.error("search: provide a non-empty query")
        return EXIT_INVALID_ARGS
    if outcome.error == "no_index":
        logger.error(
            "No vector index at %s (run `index` or pipeline with vector_search)",
            outcome.detail or "",
        )
        return EXIT_NO_ARTIFACTS
    if outcome.error == "load_failed":
        logger.error("Failed to load index: %s", outcome.detail or "")
        return EXIT_NO_ARTIFACTS
    if outcome.error == "embed_failed":
        logger.error("Embedding failed: %s", outcome.detail or "")
        return EXIT_INVALID_ARGS

    enriched = outcome.results

    fmt = getattr(args, "format", "pretty") or "pretty"
    if fmt == "json":
        payload: Dict[str, Any] = {"query": query, "results": enriched}
        if outcome.lift_stats is not None:
            payload["lift_stats"] = outcome.lift_stats
        print(json.dumps(payload, indent=2))
    else:
        print(f"Query: {query}\n")
        for row in enriched:
            meta = row["metadata"]
            print(f"score={row['score']:.4f}  {meta.get('doc_type')}  ep={meta.get('episode_id')}")
            txt = row.get("text") or ""
            if len(txt) > 200:
                txt = txt[:197] + "..."
            print(f"  {txt}")
            sq = row.get("supporting_quotes") or []
            if sq:
                print(f"  quotes: {len(sq)}")
            print()
    return EXIT_SUCCESS


def run_index_cli(args: Namespace, logger: logging.Logger) -> int:
    """Update or inspect the LanceDB corpus search index."""
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        logger.error("index: --output-dir is required")
        return EXIT_INVALID_ARGS

    _vit_raw = getattr(args, "vector_index_types", None)
    _vit_list: Optional[List[str]] = None
    if isinstance(_vit_raw, str) and _vit_raw.strip():
        _vit_list = [x.strip() for x in _vit_raw.split(",") if x.strip()]

    cfg = _minimal_vector_config(
        output_dir,
        vector_index_path=getattr(args, "vector_index_path", None),
        vector_embedding_model=getattr(args, "embedding_model", None),
        vector_embedding_endpoint=getattr(args, "embedding_endpoint", None),
        vector_embedding_provider=getattr(args, "embedding_provider", None),
        vector_index_types=_vit_list,
    )

    index_dir = _resolve_index_dir(Path(output_dir), getattr(args, "vector_index_path", None))

    if getattr(args, "stats", False):
        from podcast_scraper.search.lance_index_stats import read_lance_index_stats

        st = read_lance_index_stats(index_dir / "lance_index")
        if st is None:
            # #1864: a missing index is an EXPECTED state (not built yet, or the pre-first-index
            # window at startup), not a fault — `stats` already degrades gracefully by returning
            # EXIT_NO_ARTIFACTS. Log at WARNING so it stays visible without being auto-filed as an
            # error signal; the search query path separately serves empty results while unbuilt.
            logger.warning(
                "No search index at %s yet (not built, or pre-first-index window); "
                "nothing to report.",
                index_dir / "lance_index",
            )
            return EXIT_NO_ARTIFACTS
        blob = {
            "total_vectors": st.total_vectors,
            "doc_type_counts": st.doc_type_counts,
            "feeds_indexed": st.feeds_indexed,
            "embedding_model": st.embedding_model,
            "embedding_provider": st.embedding_provider,
            "embedding_dim": st.embedding_dim,
            "last_updated": st.last_updated,
            "index_size_bytes": st.index_size_bytes,
        }
        print(json.dumps(blob, indent=2))
        return EXIT_SUCCESS

    try:
        stats = index_corpus(
            output_dir,
            cfg,
            rebuild=bool(getattr(args, "rebuild", False)),
        )
    except Exception as exc:
        logger.error("index: %s", format_exception_for_log(exc))
        return EXIT_INVALID_ARGS

    if stats.errors:
        for err in stats.errors:
            logger.warning("%s", err)
    logger.info(
        "index: scanned=%s skipped=%s reindexed=%s vectors=%s",
        stats.episodes_scanned,
        stats.episodes_skipped_unchanged,
        stats.episodes_reindexed,
        stats.vectors_upserted,
    )
    # RFC-118: parity with POST /api/index/rebuild, whose thread re-derives topic
    # clusters after the index. The queued corpus_reindex job runs THIS verb, so
    # without the flag a queued reindex would leave clusters stale where the HTTP
    # lever would not. Non-fatal, like the route's cluster step.
    if getattr(args, "with_clusters", False):
        try:
            from podcast_scraper.search.groupings import build_topic_clusters_for_corpus

            payload = build_topic_clusters_for_corpus(output_dir) or {}
            logger.info(
                "index: topic clusters re-derived (topics=%s clusters=%s)",
                payload.get("topic_count"),
                payload.get("cluster_count"),
            )
        except Exception as exc:  # noqa: BLE001 — clusters are non-fatal like the index build
            logger.warning("index: topic-clusters rebuild failed (non-fatal): %s", exc)
    return EXIT_SUCCESS


def parse_search_argv(argv: Sequence[str]) -> Namespace:
    """Parse argv after ``search`` command name."""
    parser = argparse.ArgumentParser(
        prog="podcast_scraper search",
        description="Semantic search over the corpus vector index.",
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Natural-language search query",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Pipeline output directory (contains metadata/ and search/)",
    )
    parser.add_argument(
        "--index-path",
        default=None,
        help="Vector index directory (default: <output-dir>/search)",
    )
    parser.add_argument(
        "--type",
        dest="doc_type",
        choices=[
            "insight",
            "quote",
            "summary",
            "transcript",
            "kg_topic",
            "kg_entity",
        ],
        default=None,
        help="Restrict to document type",
    )
    parser.add_argument(
        "--feed",
        default=None,
        help="Substring match on feed_id metadata",
    )
    parser.add_argument(
        "--since",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only results on or after this publish date (episode metadata)",
    )
    parser.add_argument(
        "--speaker",
        default=None,
        help="Substring match on quote speaker_id (quotes; insights via supporting quotes)",
    )
    parser.add_argument(
        "--grounded-only",
        action="store_true",
        help="Only grounded insights",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Max results after filters",
    )
    parser.add_argument(
        "--format",
        choices=["json", "pretty"],
        default="pretty",
        help="Output format",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override embedding model id (default: model stored in index)",
    )
    parser.add_argument(
        "--no-dedupe-kg-surfaces",
        action="store_true",
        help=(
            "Keep separate kg_entity/kg_topic rows when embedded text matches "
            "(default: merge duplicates)"
        ),
    )
    ns = cast(Namespace, parser.parse_args(list(argv)))
    ns.command = "search"
    return ns


def parse_index_argv(argv: Sequence[str]) -> Namespace:
    """Parse argv after ``index`` command name."""
    parser = argparse.ArgumentParser(
        prog="podcast_scraper index",
        description="Build or inspect the semantic corpus vector index.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Pipeline output directory",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing index and rebuild from scratch",
    )
    parser.add_argument(
        "--with-clusters",
        action="store_true",
        help="Re-derive search/topic_clusters.json after the index (parity with "
        "POST /api/index/rebuild; used by the queued corpus_reindex job).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print index statistics as JSON and exit",
    )
    parser.add_argument(
        "--index-path",
        dest="vector_index_path",
        default=None,
        help="Vector index directory (default: <output-dir>/search)",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "Embedding model id. Sentence-transformers HF id (e.g. "
            "sentence-transformers/all-MiniLM-L6-v2) when --embedding-provider is "
            "sentence_transformers; Ollama tag (e.g. nomic-embed-text) when ollama."
        ),
    )
    parser.add_argument(
        "--embedding-provider",
        default=None,
        choices=["sentence_transformers", "ollama"],
        help=(
            "Override vector_embedding_provider for this run. Default: read from "
            "profile / Config. ADR-098 supersedes the RFC-089 (not in this repository) §D4 shim "
            "path; "
            "use 'ollama' for DGX-served embeddings."
        ),
    )
    parser.add_argument(
        "--embedding-endpoint",
        default=None,
        help=(
            "Base URL for remote embedding provider. With --embedding-provider=ollama, "
            "this is the Ollama base URL (e.g. http://dgx:11434) and /api/embed is "
            "appended internally. With sentence_transformers, an endpoint is treated "
            "as the legacy shim /embed URL (RFC-089 (not in this repository) §D4, deprecated; "
            "superseded by ADR-098)."
        ),
    )
    parser.add_argument(
        "--vector-index-types",
        default=None,
        dest="vector_index_types",
        metavar="TYPES",
        help="Comma-separated doc types to embed (default: all).",
    )
    ns = cast(Namespace, parser.parse_args(list(argv)))
    ns.command = "index"
    return ns


def parse_verify_gil_chunk_offsets_argv(argv: Sequence[str]) -> Namespace:
    """Parse argv after ``verify-gil-chunk-offsets`` (GitHub #528)."""
    parser = argparse.ArgumentParser(
        prog="podcast_scraper verify-gil-chunk-offsets",
        description=(
            "Compare GIL Quote char ranges to indexed transcript chunk metadata per episode "
            "(offset alignment gate before search lift)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Pipeline output directory (metadata/ + search/)",
    )
    parser.add_argument(
        "--index-path",
        default=None,
        help="Vector index directory (default: <output-dir>/search)",
    )
    parser.add_argument(
        "--min-overlap-rate",
        type=float,
        default=0.95,
        metavar="R",
        help="With --strict, fail if overlap rate is below R (default: 0.95)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit 1 when verdict is divergent, no quotes, or overlap rate below "
            "--min-overlap-rate"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=8,
        metavar="N",
        help="Max sample quote ids per episode without overlap (default: 8)",
    )
    ns = cast(Namespace, parser.parse_args(list(argv)))
    ns.command = "verify-gil-chunk-offsets"
    return ns


def run_verify_gil_chunk_offsets_cli(args: Namespace, logger: logging.Logger) -> int:
    """Run Quote vs transcript chunk offset report (#528)."""
    from podcast_scraper.search.gil_chunk_offset_verify import (
        build_offset_alignment_report,
        load_index_metadata_map,
        merge_report_dict,
    )

    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        logger.error("verify-gil-chunk-offsets: --output-dir is required")
        return EXIT_INVALID_ARGS

    root = Path(output_dir)
    index_dir = _resolve_index_dir(root, getattr(args, "index_path", None))
    meta_path = index_dir / "metadata.json"
    if not meta_path.is_file():
        logger.error("No metadata.json at %s (index missing?)", index_dir)
        return EXIT_NO_ARTIFACTS

    try:
        metadata_map = load_index_metadata_map(index_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Failed to read index metadata: %s", format_exception_for_log(exc))
        return EXIT_NO_ARTIFACTS

    gi_cache = merged_episode_gi_paths(root)
    max_samples = int(getattr(args, "max_samples", 8) or 8)
    report = build_offset_alignment_report(
        gi_by_episode=gi_cache,
        metadata_by_doc=metadata_map,
        max_samples_per_episode=max(1, max_samples),
    )
    merge_report_dict(
        report,
        {
            "corpus_root": str(root.resolve()),
            "index_dir": str(index_dir.resolve()),
        },
    )
    print(json.dumps(report, indent=2))

    if not getattr(args, "strict", False):
        return EXIT_SUCCESS

    verdict = str(report.get("verdict") or "")
    rate = report.get("overlap_rate")
    min_r = float(getattr(args, "min_overlap_rate", 0.95) or 0.95)
    if verdict == "no_quotes":
        logger.warning(
            "strict: no Quote nodes found in GI files (pass — nothing to misalign; "
            "see TD-002 in docs/architecture/TECH_DEBT.md)"
        )
        return EXIT_SUCCESS
    if verdict == "no_indexed_transcript_for_quotes":
        logger.warning(
            "strict: GI has Quote nodes but no transcript chunks in the index for "
            "those episodes (pass — overlap_rate is undefined; acceptance may omit "
            "vector indexing for some fixtures)"
        )
        return EXIT_SUCCESS
    if verdict == "divergent":
        logger.error("strict: verdict divergent (overlap rate too low)")
        return EXIT_INVALID_ARGS
    if rate is None or float(rate) < min_r:
        logger.error("strict: overlap_rate %s below %s", rate, min_r)
        return EXIT_INVALID_ARGS
    return EXIT_SUCCESS


# ── cluster-browse (#601 3b) ─────────────────────────────────────────


# ── quote-level search (#601 3c) ─────────────────────────────────────


# ── topic × insight matrix (#601 3d) ─────────────────────────────────


def parse_index_two_tier_argv(argv: Sequence[str]) -> Namespace:
    """Parse ``index-two-tier`` args (RFC-090 from-corpus two-tier indexer, follow-up B)."""
    parser = argparse.ArgumentParser(prog="podcast_scraper index-two-tier")
    parser.add_argument("--output-dir", required=True, help="Corpus root (parent of feeds/).")
    parser.add_argument(
        "--lance-path",
        default=None,
        help="LanceDB index dir (default: <output-dir>/search/lance_index).",
    )
    parser.add_argument("--embedding-model", default=None, help="Override embedding model id.")
    parser.add_argument("--limit-episodes", type=int, default=None, help="Cap episodes (smoke).")
    parser.add_argument(
        "--allow-download", action="store_true", help="Allow embedding-model download."
    )
    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=None,
        help="Rows buffered per tier before one merge_insert flush (default: indexer's).",
    )
    args = parser.parse_args(list(argv))
    args.command = "index-two-tier"
    return args


def run_index_two_tier_cli(args: Namespace, logger: logging.Logger) -> int:
    """Build the two-tier LanceDB index from corpus artifacts (RFC-090 Phase 2 / B)."""
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        logger.error("index-two-tier: --output-dir is required")
        return EXIT_INVALID_ARGS

    from podcast_scraper.search.two_tier_indexer import (
        build_two_tier_index,
        DEFAULT_MODEL,
        DEFAULT_UPSERT_BATCH_SIZE,
    )

    lance_path = getattr(args, "lance_path", None) or str(
        Path(output_dir) / "search" / "lance_index"
    )
    model_id = getattr(args, "embedding_model", None) or DEFAULT_MODEL
    stats = build_two_tier_index(
        output_dir,
        lance_path,
        model_id=model_id,
        limit_episodes=getattr(args, "limit_episodes", None),
        allow_download=bool(getattr(args, "allow_download", False)),
        # ``index-two-tier`` is a full reindex: start from a clean slate so per-document
        # upsert fragments can't accumulate across runs (the build compacts at the end).
        drop_existing=True,
        upsert_batch_size=getattr(args, "upsert_batch_size", None) or DEFAULT_UPSERT_BATCH_SIZE,
    )
    print(
        f"Two-tier index built at {lance_path}: episodes={stats.episodes} "
        f"segments={stats.segments} insights={stats.insights} aux={stats.aux}"
    )
    return EXIT_SUCCESS


# ─────────────────────────────────────────────────────────────────────
# #1058 chunk 3 — cluster-corpus-topics CLI
# ─────────────────────────────────────────────────────────────────────

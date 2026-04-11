"""CLI handlers for semantic corpus search (`search` and `index` commands, #484 Step 4)."""

from __future__ import annotations

import argparse
import json
import logging
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
from podcast_scraper.search.faiss_store import FaissVectorStore
from podcast_scraper.search.indexer import _scope_display_titles, index_corpus
from podcast_scraper.search.protocol import SearchResult
from podcast_scraper.utils.log_redaction import format_exception_for_log
from podcast_scraper.workflow.metadata_generation import _determine_gi_path

# Align with gi.explore exit codes
EXIT_SUCCESS = 0
EXIT_INVALID_ARGS = 2
EXIT_NO_ARTIFACTS = 3

_PLACEHOLDER_RSS = "https://example.com/.podcast-scraper/vector-cli-placeholder"


def _minimal_vector_config(
    output_dir: str,
    *,
    vector_index_path: Optional[str] = None,
    vector_embedding_model: Optional[str] = None,
    vector_faiss_index_mode: Optional[str] = None,
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
    if vector_faiss_index_mode is not None:
        payload["vector_faiss_index_mode"] = vector_faiss_index_mode
    if vector_index_types is not None:
        payload["vector_index_types"] = vector_index_types
    return config.Config(**payload)


def _resolve_index_dir(output_dir: Path, index_path: Optional[str]) -> Path:
    if index_path:
        p = Path(index_path)
        if not p.is_absolute():
            p = output_dir / p
        return p.resolve()
    return (output_dir / "search").resolve()


def _episode_to_gi_path(output_dir: Path) -> Dict[str, Path]:
    """Map episode_id -> resolved gi.json path."""
    meta_dir = output_dir / "metadata"
    out: Dict[str, Path] = {}
    if not meta_dir.is_dir():
        return out
    for meta_path in sorted(meta_dir.glob("*.metadata.json")):
        try:
            doc = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        ep = doc.get("episode") or {}
        eid = ep.get("episode_id")
        if not isinstance(eid, str) or not eid:
            continue
        gi = doc.get("grounded_insights")
        if isinstance(gi, dict):
            rel = gi.get("artifact_path")
            if isinstance(rel, str) and rel.strip():
                gp = (output_dir / rel.strip()).resolve()
                if gp.is_file():
                    out[eid] = gp
                    continue
        gp = Path(_determine_gi_path(str(meta_path))).resolve()
        if gp.is_file():
            out[eid] = gp
    return out


def _metadata_relpath_by_scope_from_corpus(output_dir: Path) -> Dict[str, str]:
    """Map fingerprint scope key -> corpus-relative ``*.metadata.json`` path (POSIX).

    Backfills ``source_metadata_relative_path`` on search hits when FAISS rows predate
    that field (incremental index without re-embed, or older indexer builds).
    """
    root = output_dir.resolve()
    out: Dict[str, str] = {}
    for meta_path in discover_metadata_files(root):
        try:
            doc = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        ep = doc.get("episode") or {}
        eid = ep.get("episode_id")
        if not isinstance(eid, str) or not eid:
            continue
        feed = doc.get("feed") or {}
        fid = feed.get("feed_id")
        key = index_fingerprint_scope_key(normalize_feed_id(fid), eid)
        try:
            rel = meta_path.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        out[key] = rel
        # Hits with only episode_id in vector metadata (missing feed_id) cannot use compound key.
        ep_only = index_fingerprint_scope_key(None, eid)
        if ep_only not in out:
            out[ep_only] = rel
    return out


def _quotes_for_insight(artifact: Dict[str, Any], insight_id: str) -> List[Dict[str, Any]]:
    quotes: List[Dict[str, Any]] = []
    nodes = {n.get("id"): n for n in artifact.get("nodes") or [] if isinstance(n, dict)}
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
                "char_start": props.get("char_start"),
                "char_end": props.get("char_end"),
                "timestamp_start_ms": props.get("timestamp_start_ms"),
                "timestamp_end_ms": props.get("timestamp_end_ms"),
            }
        )
    return quotes


def _insight_passes_speaker_filter(
    artifact: Dict[str, Any],
    insight_id: str,
    speaker_key: str,
) -> bool:
    key = speaker_key.lower()
    for q in _quotes_for_insight(artifact, insight_id):
        sid = q.get("speaker_id")
        if isinstance(sid, str) and key in sid.lower():
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
    grounded_only: bool,
    gi_by_episode: Dict[str, Path],
) -> bool:
    meta = hit.metadata
    if grounded_only and meta.get("doc_type") == "insight":
        if not meta.get("grounded"):
            return False

    if feed_substr:
        fid = meta.get("feed_id")
        if not isinstance(fid, str) or feed_substr.lower() not in fid.lower():
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
        sid = meta.get("speaker_id")
        if not isinstance(sid, str) or speaker_substr.lower() not in sid.lower():
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
        if not _insight_passes_speaker_filter(art, src, speaker_substr):
            return False

    return True


def _backfill_display_titles_from_corpus(
    meta: Dict[str, Any],
    corpus_root: Path,
    cache: Dict[str, Tuple[str, str]],
) -> None:
    """Set ``episode_title`` / ``feed_title`` from ``*.metadata.json`` when missing on the hit.

    Uses the same title resolution as the vector indexer so older FAISS rows (indexed before
    titles were stamped) still get human labels in the viewer without a full rebuild.
    """
    if meta.get("episode_title") and meta.get("feed_title"):
        return
    rel = meta.get("source_metadata_relative_path")
    if not isinstance(rel, str) or not rel.strip():
        return
    rel_key = rel.strip().replace("\\", "/")
    if rel_key not in cache:
        path = (corpus_root / rel_key).resolve()
        try:
            corpus_root_res = corpus_root.resolve()
            path.relative_to(corpus_root_res)
        except ValueError:
            cache[rel_key] = ("", "")
        else:
            try:
                if not path.is_file():
                    cache[rel_key] = ("", "")
                else:
                    doc = json.loads(path.read_text(encoding="utf-8"))
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
) -> Dict[str, Any]:
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
    try:
        art = cast(Dict[str, Any], json.loads(gpath.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return row
    row["supporting_quotes"] = _quotes_for_insight(art, src)
    return row


def run_search_cli(args: Namespace, logger: logging.Logger) -> int:
    """Run semantic search over a FAISS corpus index."""
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
        print(json.dumps({"query": query, "results": enriched}, indent=2))
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
    """Update or inspect the FAISS corpus index."""
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
        vector_faiss_index_mode=getattr(args, "vector_faiss_index_mode", None),
        vector_index_types=_vit_list,
    )

    index_dir = _resolve_index_dir(Path(output_dir), getattr(args, "vector_index_path", None))

    if getattr(args, "stats", False):
        if not (index_dir / "vectors.faiss").is_file():
            logger.error("No index at %s", index_dir)
            return EXIT_NO_ARTIFACTS
        try:
            store = FaissVectorStore.load(index_dir)
            st = store.stats()
        except Exception as exc:
            logger.error("Failed to read index: %s", format_exception_for_log(exc))
            return EXIT_NO_ARTIFACTS
        blob = {
            "total_vectors": st.total_vectors,
            "doc_type_counts": st.doc_type_counts,
            "feeds_indexed": st.feeds_indexed,
            "embedding_model": st.embedding_model,
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
    return EXIT_SUCCESS


def parse_search_argv(argv: Sequence[str]) -> Namespace:
    """Parse argv after ``search`` command name."""
    parser = argparse.ArgumentParser(
        prog="podcast_scraper search",
        description="Semantic search over the corpus vector index (RFC-061).",
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
        description="Build or inspect the semantic corpus vector index (RFC-061).",
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
        help="Sentence-transformers model id for embeddings",
    )
    parser.add_argument(
        "--vector-faiss-index-mode",
        choices=["auto", "flat", "ivf_flat", "ivfpq"],
        default=None,
        dest="vector_faiss_index_mode",
        help="FAISS index structure after indexing (default: auto).",
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

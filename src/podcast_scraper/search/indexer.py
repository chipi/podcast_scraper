"""Embed-and-index corpus pipeline."""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast, Dict, List, Optional, Set, Tuple

from podcast_scraper import config
from podcast_scraper.providers.ml.model_registry import ModelRegistry
from podcast_scraper.search.chunker import chunk_transcript
from podcast_scraper.search.corpus_scope import (
    normalize_feed_id,
    vector_doc_scope_tag,
)
from podcast_scraper.workflow.metadata_generation import _determine_gi_path, _determine_kg_path

logger = logging.getLogger(__name__)

EPISODE_FINGERPRINTS_FILE = "episode_fingerprints.json"


@dataclass
class IndexRunStats:
    """Counters from one ``index_corpus`` run."""

    episodes_scanned: int = 0
    episodes_skipped_unchanged: int = 0
    episodes_reindexed: int = 0
    vectors_upserted: int = 0
    errors: List[str] = field(default_factory=list)


def _embedding_dim(model_id: str) -> int:
    resolved = ModelRegistry.resolve_evidence_model_id(model_id)
    caps = ModelRegistry.get_capabilities(resolved)
    if caps.embedding_dim is not None:
        return int(caps.embedding_dim)
    return 384


def _resolve_index_dir(output_dir: str, cfg: config.Config) -> Path:
    out = Path(output_dir)
    raw = cfg.vector_index_path
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = out / p
        return p.resolve()
    return (out / "search").resolve()


def _load_metadata_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            return cast(Dict[str, Any], json.loads(text))
        import yaml

        data = yaml.safe_load(text)
        return cast(Dict[str, Any], data) if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("Skip unreadable metadata %s: %s", path, exc)
        return None


def _gi_path(episode_root: Path, metadata_path: Path, doc: Dict[str, Any]) -> Path:
    gi = doc.get("grounded_insights")
    if isinstance(gi, dict):
        rel = gi.get("artifact_path")
        if isinstance(rel, str) and rel.strip():
            p = (episode_root / rel.strip()).resolve()
            if p.is_file():
                return p
    return Path(_determine_gi_path(str(metadata_path))).resolve()


def _kg_path(episode_root: Path, metadata_path: Path, doc: Dict[str, Any]) -> Path:
    kg = doc.get("knowledge_graph")
    if isinstance(kg, dict):
        rel = kg.get("artifact_path")
        if isinstance(rel, str) and rel.strip():
            p = (episode_root / rel.strip()).resolve()
            if p.is_file():
                return p
    return Path(_determine_kg_path(str(metadata_path))).resolve()


def _transcript_path(episode_root: Path, doc: Dict[str, Any]) -> Optional[Path]:
    content = doc.get("content") or {}
    rel = content.get("transcript_file_path")
    if not isinstance(rel, str) or not rel.strip():
        return None
    rel = rel.strip()
    # #974: prefer the ad-free processing base (<base>.adfree.txt) when present, so
    # search chunks and enrich-edges SPOKEN_BY share the coordinate space GI computed
    # quote char_start in. Fall back to the raw transcript for pre-#974 corpora.
    base, ext = os.path.splitext(rel)
    adfree = (episode_root / f"{base}.adfree{ext}").resolve()
    if adfree.is_file():
        return adfree
    p = (episode_root / rel).resolve()
    return p if p.is_file() else None


def _filter_rows_by_doc_types(
    rows: List[Tuple[str, str, Dict[str, Any]]],
    allowed: Optional[Set[str]],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    if not allowed:
        return rows
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    for doc_id, text, meta in rows:
        dt = meta.get("doc_type")
        if isinstance(dt, str) and dt in allowed:
            out.append((doc_id, text, meta))
    return out


def _kg_embed_text_topic(props: Dict[str, Any]) -> Optional[str]:
    """Concatenate Topic label and optional description for embedding."""
    parts: List[str] = []
    label = props.get("label")
    if isinstance(label, str) and label.strip():
        parts.append(label.strip())
    desc = props.get("description")
    if isinstance(desc, str) and desc.strip():
        parts.append(desc.strip())
    return " ".join(parts) if parts else None


def _kg_entity_kind_for_meta(props: Dict[str, Any]) -> Optional[str]:
    """person|organization for index metadata."""
    k = props.get("kind")
    if k == "org":
        return "organization"
    if k == "person":
        return "person"
    ek = props.get("entity_kind")
    return ek if isinstance(ek, str) else None


def _kg_embed_text_entity(props: Dict[str, Any]) -> Optional[str]:
    """Concatenate Entity name, optional distinct label, and description for embedding."""
    parts: List[str] = []
    name = props.get("name")
    if isinstance(name, str) and name.strip():
        parts.append(name.strip())
    lab = props.get("label")
    if isinstance(lab, str) and lab.strip():
        nm = (name or "").strip()
        if not nm or lab.strip().casefold() != nm.casefold():
            parts.append(lab.strip())
    desc = props.get("description")
    if isinstance(desc, str) and desc.strip():
        parts.append(desc.strip())
    return " ".join(parts) if parts else None


def _kg_vector_rows_from_path(
    kg_disk: Path,
    scope_tag: str,
    episode_id: str,
    feed_id: Any,
    published: Any,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Parse ``kg.json`` and return vector rows for Topic and Entity nodes only."""
    rows: List[Tuple[str, str, Dict[str, Any]]] = []
    try:
        kg_artifact = cast(Dict[str, Any], json.loads(kg_disk.read_text(encoding="utf-8")))
    except Exception as exc:
        logger.warning("Skip kg.json %s: %s", kg_disk, exc)
        return rows
    for n in kg_artifact.get("nodes") or []:
        if not isinstance(n, dict):
            continue
        nt = n.get("type")
        nid = n.get("id")
        if not isinstance(nid, str):
            continue
        raw_props = n.get("properties")
        props = raw_props if isinstance(raw_props, dict) else {}
        if nt == "Topic":
            ktext = _kg_embed_text_topic(props)
            if ktext:
                rows.append(
                    (
                        f"kg_topic:{scope_tag}:{nid}",
                        ktext,
                        {
                            "doc_type": "kg_topic",
                            "episode_id": episode_id,
                            "feed_id": feed_id,
                            "publish_date": published,
                            "source_id": nid,
                        },
                    )
                )
        elif nt == "Entity":
            ktext = _kg_embed_text_entity(props)
            if ktext:
                ek = _kg_entity_kind_for_meta(props)
                rows.append(
                    (
                        f"kg_entity:{scope_tag}:{nid}",
                        ktext,
                        {
                            "doc_type": "kg_entity",
                            "episode_id": episode_id,
                            "feed_id": feed_id,
                            "publish_date": published,
                            "source_id": nid,
                            "entity_kind": ek if isinstance(ek, str) else None,
                        },
                    )
                )
    return rows


def _scope_display_titles(
    doc: Dict[str, Any], ep: Dict[str, Any], feed: Dict[str, Any]
) -> tuple[str, str]:
    """Episode and feed titles copied onto every vector row for search UI labels.

    Prefers nested ``episode.title`` / ``feed.title`` (pipeline metadata). Falls back to
    flat ``episode_title`` / ``feed_name`` (eval or legacy shapes).
    """
    et_raw = ep.get("title")
    if isinstance(et_raw, str) and et_raw.strip():
        episode_title = et_raw.strip()
    else:
        flat = doc.get("episode_title")
        episode_title = flat.strip() if isinstance(flat, str) and flat.strip() else ""

    ft_raw = feed.get("title")
    if isinstance(ft_raw, str) and ft_raw.strip():
        feed_title = ft_raw.strip()
    else:
        flat_f = doc.get("feed_name")
        feed_title = flat_f.strip() if isinstance(flat_f, str) and flat_f.strip() else ""

    return episode_title, feed_title


def _collect_docs_for_episode(  # noqa: C901
    episode_root: Path,
    metadata_path: Path,
    doc: Dict[str, Any],
    *,
    target_tokens: int,
    overlap_tokens: int,
    metadata_relative_path: str,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    rows: List[Tuple[str, str, Dict[str, Any]]] = []
    ep = doc.get("episode") or {}
    feed = doc.get("feed") or {}
    episode_id = ep.get("episode_id")
    raw_feed_id = feed.get("feed_id")
    feed_norm = normalize_feed_id(raw_feed_id)
    published = ep.get("published_date")
    if not isinstance(episode_id, str) or not episode_id:
        return _rows_with_text_metadata(rows)

    scope_tag = vector_doc_scope_tag(feed_norm, episode_id)
    gi_path = _gi_path(episode_root, metadata_path, doc)
    artifact: Optional[Dict[str, Any]] = None
    if gi_path.is_file():
        try:
            artifact = cast(Dict[str, Any], json.loads(gi_path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Skip gi.json %s: %s", gi_path, exc)

    if artifact:
        for n in artifact.get("nodes") or []:
            nt = n.get("type")
            props = n.get("properties") or {}
            nid = n.get("id")
            if not isinstance(nid, str):
                continue
            if nt == "Insight":
                text = props.get("text")
                if isinstance(text, str) and text.strip():
                    rows.append(
                        (
                            f"insight:{scope_tag}:{nid}",
                            text.strip(),
                            {
                                "doc_type": "insight",
                                "episode_id": episode_id,
                                "feed_id": raw_feed_id,
                                "publish_date": published,
                                "source_id": nid,
                                "grounded": bool(props.get("grounded")),
                            },
                        )
                    )
            elif nt == "Quote":
                text = props.get("text")
                if isinstance(text, str) and text.strip():
                    rows.append(
                        (
                            f"quote:{scope_tag}:{nid}",
                            text.strip(),
                            {
                                "doc_type": "quote",
                                "episode_id": episode_id,
                                "feed_id": raw_feed_id,
                                "publish_date": published,
                                "source_id": nid,
                                "speaker_id": props.get("speaker_id"),
                                "char_start": props.get("char_start"),
                                "char_end": props.get("char_end"),
                                "timestamp_start_ms": props.get("timestamp_start_ms"),
                                "timestamp_end_ms": props.get("timestamp_end_ms"),
                            },
                        )
                    )

    kg_disk = _kg_path(episode_root, metadata_path, doc)
    if kg_disk.is_file():
        rows.extend(
            _kg_vector_rows_from_path(kg_disk, scope_tag, episode_id, raw_feed_id, published)
        )

    summary = doc.get("summary")
    chunk_ts: Optional[List[Dict[str, Any]]] = None
    if isinstance(summary, dict):
        ts_raw = summary.get("timestamps")
        if isinstance(ts_raw, list):
            chunk_ts = [cast(Dict[str, Any], x) for x in ts_raw if isinstance(x, dict)]
        bullets = summary.get("bullets") or []
        if isinstance(bullets, list):
            for i, b in enumerate(bullets):
                if isinstance(b, str) and b.strip():
                    rows.append(
                        (
                            f"bullet:{scope_tag}:{i}",
                            b.strip(),
                            {
                                "doc_type": "summary",
                                "episode_id": episode_id,
                                "feed_id": raw_feed_id,
                                "publish_date": published,
                                "source_id": str(i),
                            },
                        )
                    )

    tpath = _transcript_path(episode_root, doc)
    if tpath is not None:
        try:
            ttext = tpath.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Transcript read failed %s: %s", tpath, exc)
            ttext = ""
        if ttext.strip():
            for ch in chunk_transcript(
                ttext,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
                timestamps=chunk_ts,
            ):
                rows.append(
                    (
                        f"chunk:{scope_tag}:{ch.chunk_index}",
                        ch.text,
                        {
                            "doc_type": "transcript",
                            "episode_id": episode_id,
                            "feed_id": raw_feed_id,
                            "publish_date": published,
                            "source_id": str(ch.chunk_index),
                            "char_start": ch.char_start,
                            "char_end": ch.char_end,
                            "timestamp_start_ms": ch.timestamp_start_ms,
                            "timestamp_end_ms": ch.timestamp_end_ms,
                        },
                    )
                )

    episode_title, feed_title = _scope_display_titles(doc, ep, feed)
    for _, _, meta in rows:
        meta["source_metadata_relative_path"] = metadata_relative_path
        if episode_title:
            meta["episode_title"] = episode_title
        if feed_title:
            meta["feed_title"] = feed_title
    return _rows_with_text_metadata(rows)


def _meta_with_text(base: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Sidecar metadata including embedded text for CLI display (#484)."""
    out = dict(base)
    out["text"] = text
    return out


def _rows_with_text_metadata(
    rows: List[Tuple[str, str, Dict[str, Any]]],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    return [(doc_id, text, _meta_with_text(meta, text)) for doc_id, text, meta in rows]


def _warn_if_zero_vectors_built(stats: IndexRunStats) -> None:
    """Log clearly when episodes were processed but no vectors were upserted."""
    if stats.episodes_scanned == 0 or stats.vectors_upserted > 0:
        return
    if stats.episodes_skipped_unchanged == stats.episodes_scanned:
        return
    hint = (
        "Indexing uses allow_download=False for embeddings; weights must be cached already. "
        "Run `make preload-ml-models` without SKIP_GIL=1 (or ensure vector_embedding_model "
        "matches a cached sentence-transformers checkpoint). Align HF_HUB_CACHE with preload "
        "if you set it explicitly."
    )
    if stats.errors:
        logger.error(
            "Vector index built 0 new vectors (scanned %d episode(s), %d skipped unchanged). "
            "%s First error: %s",
            stats.episodes_scanned,
            stats.episodes_skipped_unchanged,
            hint,
            stats.errors[0],
        )
    else:
        logger.warning(
            "Vector index built 0 new vectors (scanned %d episode(s)) with no error list entries; "
            "check vector_index_types or empty extractable docs. %s",
            stats.episodes_scanned,
            hint,
        )


def index_corpus(
    output_dir: str,
    cfg: config.Config,
    *,
    rebuild: bool = False,
) -> IndexRunStats:
    """Scan episode metadata under *output_dir*, embed documents, build the LanceDB index.

    When *output_dir* contains a ``feeds/`` directory (multi-feed corpus parent),
    discovers ``**/metadata/*.metadata.{json,yaml}`` recursively. Otherwise scans
    ``<output_dir>/metadata/`` only. Artifact paths in metadata are resolved relative
    to each file's episode root (parent of ``metadata/``). Fingerprint keys and
    vector row ids are scoped by ``(feed_id, episode_id)`` when ``feed_id`` is set
    (GitHub #505).

    Args:
        output_dir: Single-feed output root or multi-feed corpus parent.
        cfg: Config with ``vector_*`` fields and embedding model id.
        rebuild: If True, delete the index directory and rebuild from scratch.

    Returns:
        Aggregate run statistics (non-fatal errors are listed in ``errors``).
    """
    stats = IndexRunStats()
    out = Path(output_dir)
    # ADR-099 / #995: the single search index is the LanceDB two-tier index, built from
    # corpus artifacts (segment + insight + aux tiers). FAISS is retired. The builder upserts
    # idempotently (merge on id), so re-running refreshes rows in place.
    from .two_tier_indexer import build_two_tier_index  # lazy: avoids indexer<->two_tier cycle

    lance_path = out / "search" / "lance_index"
    if rebuild and lance_path.exists():
        shutil.rmtree(lance_path, ignore_errors=True)
    try:
        tt = build_two_tier_index(
            out,
            lance_path,
            model_id=cfg.vector_embedding_model,
            target_tokens=cfg.vector_chunk_size_tokens,
            overlap_tokens=cfg.vector_chunk_overlap_tokens,
            allow_download=False,
            upsert_batch_size=cfg.vector_upsert_batch_size,
        )
    except Exception as exc:  # non-fatal: callers (maybe_index_corpus) log + continue
        stats.errors.append(f"two-tier index build: {exc}")
        return stats
    stats.episodes_scanned = tt.episodes
    stats.episodes_reindexed = tt.episodes
    stats.vectors_upserted = tt.segments + tt.insights + tt.aux
    _warn_if_zero_vectors_built(stats)
    return stats


def maybe_index_corpus(output_dir: str, cfg: config.Config) -> None:
    """Run ``index_corpus`` when ``vector_search`` is enabled; log failures only."""
    if getattr(cfg, "skip_auto_vector_index", False) is True:
        return
    if getattr(cfg, "vector_search", False) is not True:
        return
    try:
        index_corpus(output_dir, cfg)
    except Exception as exc:
        logger.warning("Vector index update failed (non-fatal): %s", exc)

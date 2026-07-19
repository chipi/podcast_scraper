"""From-corpus two-tier indexer (RFC-090 / wire-live follow-up B).

Builds the two-tier LanceDB index (#855) directly from corpus artifacts, so a
**fresh** corpus becomes hybrid-searchable without first having a legacy index to
migrate (#858). It reuses the proven indexer extraction —
``discover_metadata_files`` → ``_collect_docs_for_episode`` — which already yields
insight rows and timestamped transcript chunks, then re-embeds the text and upserts
into the ``insight`` (Tier 2) and ``segment`` (Tier 1) tables.

Relationship to the (now-retired) migration (#858): the migration was the fast path
for a corpus that already had a legacy index (it reused those embeddings verbatim);
this indexer is the native path for corpora that don't. Both produce the same
two-tier layout the live hybrid search (RFC-090 Phase 2) reads. Unlike the migration,
this path **populates insight↔segment links** (``linked_insight_ids`` /
``source_segment_id``) from the gi.json ``SUPPORTED_BY`` edges + quote timestamps, so
the compound-result path (``dedup``) actually fires on a natively-built index. The
migration left them empty (it had no edges to supply), so compounds need a native
(re)index.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast, Dict, List, Optional, Tuple

from .. import config as _config, config_constants as _config_constants
from ..providers.ml import embedding_loader
from .backend import AuxDocument, InsightDocument, SegmentDocument
from .backends.lancedb_backend import lance_index_is_stale, LanceDBBackend
from .corpus_scope import discover_metadata_files, episode_root_from_metadata_path
from .indexer import _collect_docs_for_episode, _gi_path, _load_metadata_file
from .segments import link_insights_to_segments, link_insights_to_segments_by_text

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TARGET_TOKENS = 256
DEFAULT_OVERLAP_TOKENS = 32
# Rows accumulated (across episodes) before a tier is flushed in one merge_insert.
# Each flush = one LanceDB transaction (data file + version), so this trades peak
# memory for fewer fragments: ~total_rows/batch transactions per tier instead of
# one-per-document. 512 keeps buffers small (<~1MB/tier at 384-dim) while collapsing
# a 99-episode corpus's thousands of docs into tens of transactions. Single source of
# truth in config_constants so the Config field + profiles share the same default.
DEFAULT_UPSERT_BATCH_SIZE = _config_constants.DEFAULT_VECTOR_UPSERT_BATCH_SIZE

# Non-tiered corpus surfaces indexed into the aux tier for full coverage.
_AUX_DOC_TYPES = frozenset({"kg_entity", "kg_topic", "quote", "summary"})


@dataclass
class TwoTierIndexStats:
    """Counts from a from-corpus two-tier index build."""

    episodes: int = 0
    segments: int = 0
    insights: int = 0
    aux: int = 0
    linked: int = 0


def _embed(text: str, model_id: str, *, allow_download: bool) -> List[float]:
    cfg = _config.Config()
    vec = embedding_loader.encode(
        text,
        model_id,
        return_numpy=False,
        allow_download=allow_download,
        remote_endpoint=cfg.vector_embedding_endpoint,
        provider=cfg.vector_embedding_provider,
    )
    return [float(x) for x in cast(List[float], vec)]


def _insight_grounding_quotes(gi_path: Path) -> Dict[str, Tuple[float, Optional[float]]]:
    """Map each insight node id → its first grounding quote's (start_s, end_s).

    Reads the episode's ``*.gi.json`` (Insight ``SUPPORTED_BY`` Quote; quotes carry
    ``timestamp_*_ms``). This is what lets ``link_insights_to_segments`` connect an
    insight to the transcript segment it was spoken in — the input the compound-result
    path (``dedup``) needs but the retired migration couldn't supply.
    """
    if not gi_path.is_file():
        return {}
    try:
        art = json.loads(gi_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    quote_ts: Dict[str, Tuple[float, Optional[float]]] = {}
    for node in art.get("nodes") or []:
        if node.get("type") == "Quote":
            props = node.get("properties") or {}
            start = props.get("timestamp_start_ms")
            if start is not None:
                end = props.get("timestamp_end_ms")
                quote_ts[node.get("id")] = (
                    float(start) / 1000.0,
                    float(end) / 1000.0 if end is not None else None,
                )
    out: Dict[str, Tuple[float, Optional[float]]] = {}
    for edge in art.get("edges") or []:
        if edge.get("type") == "SUPPORTED_BY":
            insight_id, quote_id = edge.get("from"), edge.get("to")
            if insight_id not in out and quote_id in quote_ts:
                out[insight_id] = quote_ts[quote_id]
    return out


def _insight_grounding_quote_texts(gi_path: Path) -> Dict[str, str]:
    """Map each insight node id → its first grounding quote's verbatim text.

    The text-based counterpart to :func:`_insight_grounding_quotes`. Used to link
    insights to the transcript segment that contains the quote when segments carry
    no usable timestamps (``summary.timestamps`` unpopulated → segment spans at 0.0).
    """
    if not gi_path.is_file():
        return {}
    try:
        art = json.loads(gi_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    quote_text: Dict[str, str] = {}
    for node in art.get("nodes") or []:
        if node.get("type") == "Quote":
            txt = (node.get("properties") or {}).get("text")
            if isinstance(txt, str) and txt.strip():
                quote_text[node.get("id")] = txt.strip()
    out: Dict[str, str] = {}
    for edge in art.get("edges") or []:
        if edge.get("type") == "SUPPORTED_BY":
            insight_id, quote_id = edge.get("from"), edge.get("to")
            if insight_id not in out and quote_id in quote_text:
                out[insight_id] = quote_text[quote_id]
    return out


def _plan_reindex_clear(lance_path: Path, drop_existing: bool) -> Tuple[str, set]:
    """Decide whether a full/stale reindex must clear prior data, and snapshot which tier
    tables already exist so untouched ones can be MVCC-emptied at the end (#1206).

    Returns ``(reason, pre_existing_tiers)``; ``reason`` is empty when no clear is needed.
    A *full* reindex (``drop_existing``) or a schema bump (a stale index) must not inherit
    prior rows — but this replaces the old ``rmtree`` (which stranded in-flight api reads
    with ``Not found``) with an MVCC clear driven by the caller.
    """
    if drop_existing and lance_path.exists():
        reason = "Full reindex"
    elif lance_index_is_stale(lance_path):
        reason = "Stale-schema reindex"
    else:
        return "", set()
    try:
        pre_existing = set(LanceDBBackend(str(lance_path)).existing_tier_tables())
    except Exception:  # noqa: BLE001 - unreadable/absent index just means nothing to clear
        pre_existing = set()
    return reason, pre_existing


def _finalize_reindex_clear(
    lance_path: Path,
    backend: Optional[LanceDBBackend],
    pre_existing_tiers: set,
    overwritten_tiers: set,
) -> None:
    """MVCC-empty any pre-existing tier that received no rows this build, so its stale rows
    don't survive the reindex "clear" (#1206 — rmtree parity without stranding readers).

    Uses the build's own backend when one was created; otherwise (empty new corpus, nothing
    flushed) opens a bare backend just to clear, and drops the now-stale doc_id sidecar.
    """
    stale_tiers = pre_existing_tiers - overwritten_tiers
    if stale_tiers:
        cleanup_backend = backend or LanceDBBackend(str(lance_path))
        for tier in sorted(stale_tiers):
            cleanup_backend.clear_tier_mvcc(tier)
    if backend is None:
        lance_path.parent.joinpath("metadata.json").unlink(missing_ok=True)


def build_two_tier_index(
    corpus_dir: str | Path,
    lance_path: str | Path,
    *,
    model_id: str = DEFAULT_MODEL,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    embed_dim: Optional[int] = None,
    limit_episodes: Optional[int] = None,
    allow_download: bool = False,
    drop_existing: bool = False,
    upsert_batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
) -> TwoTierIndexStats:
    """Build the two-tier LanceDB index at *lance_path* from the corpus at *corpus_dir*.

    Walks episode metadata, extracts insight + transcript-chunk rows, embeds each with
    *model_id*, and upserts into the segment/insight tables (idempotent merge on id).
    ``embed_dim`` is derived from *model_id* when None (so a non-MiniLM model can't
    silently mismatch the schema). ``limit_episodes`` caps the walk. Returns counts.

    ``drop_existing=True`` removes any existing index first — a **full** reindex starts
    from a clean slate so per-document-upsert fragments can't accumulate across runs.
    Either way the index is compacted at the end (see ``LanceDBBackend.compact``), which
    bounds the **incremental** post-pipeline reindex too (it upserts new episodes into
    the existing index, then reclaims the fragments/versions that creates).
    """
    out = Path(corpus_dir)

    # A full/stale reindex clears prior data via LanceDB MVCC (never ``rmtree`` — #1206):
    # the first flush of each tier ``overwrite``-replaces its table in place, and any tier
    # that existed but gets no rows this build is MVCC-emptied at the end (see the helpers).
    clear_reason, pre_existing_tiers = _plan_reindex_clear(Path(lance_path), drop_existing)
    clear_requested = bool(clear_reason)
    overwritten_tiers: set[str] = set()
    if clear_requested:
        logger.info("%s: MVCC-clear (no rmtree) of LanceDB index at %s", clear_reason, lance_path)

    stats = TwoTierIndexStats()
    backend: Optional[LanceDBBackend] = None

    def _ensure_backend(vec_len: int) -> LanceDBBackend:
        # Lazy: size the schema from the model's real dim (or an explicit override) on
        # the first embedded doc, so a non-MiniLM model can't silently mismatch — and an
        # empty corpus creates no index at all.
        nonlocal backend
        if backend is None:
            backend = LanceDBBackend(str(lance_path), embed_dim=embed_dim or vec_len)
        return backend

    # Cross-episode upsert buffers: docs accumulate here and flush in one transaction
    # per tier once a buffer reaches ``upsert_batch_size`` (and once more at the end),
    # so transaction count scales with total_rows/batch, not with document count.
    seg_buf: List[SegmentDocument] = []
    ins_buf: List[InsightDocument] = []
    aux_buf: List[AuxDocument] = []
    batch = max(1, int(upsert_batch_size))
    # FAISS retirement (#1010) dropped the ``search/metadata.json`` sidecar (doc_id -> chunk
    # meta incl. char_start/char_end) that the GIL chunk-offset verifier and ``make
    # verify-gil-offsets-after-acceptance`` still read. Re-emit it from the same row meta we
    # already build, written next to the lance index (see search/gil_chunk_offset_verify.py).
    index_metadata: Dict[str, dict] = {}

    def _flush_tier(tier: str, buf: List, replace: Callable, upsert: Callable) -> None:
        # On a full/stale reindex the first flush of a tier ``overwrite``-replaces its table
        # (MVCC clear-in-place, no rmtree — #1206); later flushes upsert into it as usual.
        if not buf:
            return
        be = _ensure_backend(len(buf[0].embedding))
        if clear_requested and tier not in overwritten_tiers:
            replace(be, buf)
            overwritten_tiers.add(tier)
        else:
            upsert(be, buf)
        buf.clear()

    def _flush_segments() -> None:
        _flush_tier(
            "segment",
            seg_buf,
            lambda be, r: be.replace_segments(r),
            lambda be, r: be.upsert_segments(r),
        )

    def _flush_insights() -> None:
        _flush_tier(
            "insight",
            ins_buf,
            lambda be, r: be.replace_insights(r),
            lambda be, r: be.upsert_insights(r),
        )

    def _flush_auxes() -> None:
        _flush_tier(
            "aux",
            aux_buf,
            lambda be, r: be.replace_auxes(r),
            lambda be, r: be.upsert_auxes(r),
        )

    for meta_path in discover_metadata_files(out):
        if limit_episodes is not None and stats.episodes >= limit_episodes:
            break
        doc = _load_metadata_file(meta_path)
        if not doc:
            continue
        stats.episodes += 1
        episode_root = episode_root_from_metadata_path(meta_path)
        meta_rel = meta_path.resolve().relative_to(out.resolve()).as_posix()
        rows = _collect_docs_for_episode(
            episode_root,
            meta_path,
            doc,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            metadata_relative_path=meta_rel,
        )
        # Collect this episode's docs first, then link insights to the segment that
        # contains their grounding quote, then upsert — so linked_insight_ids /
        # source_segment_id are populated (the compound-result path needs them).
        seg_docs: List[SegmentDocument] = []
        ins_docs: List[InsightDocument] = []
        aux_docs: List[AuxDocument] = []
        node_id_by_insight: Dict[str, Optional[str]] = {}
        for doc_id, text, meta in rows:
            # Strip the embedded chunk ``text``: the offset verifier reads only char_start/
            # char_end/doc_type/episode_id, and keeping ``text`` would re-duplicate the whole
            # transcript corpus into the sidecar JSON (the index-bloat class #1010 just fixed).
            index_metadata[doc_id] = {k: v for k, v in meta.items() if k != "text"}
            doc_type = meta.get("doc_type")
            if doc_type == "insight":
                ins_docs.append(
                    InsightDocument(
                        id=doc_id,
                        text=text,
                        show_id=meta.get("feed_id") or "",
                        episode_id=meta.get("episode_id") or "",
                        entity_type="insight",
                        confidence=0.0,
                        derived=bool(meta.get("grounded")),
                        publish_date=meta.get("publish_date"),
                        source_id=meta.get("source_id"),
                        embedding=_embed(text, model_id, allow_download=allow_download),
                    )
                )
                node_id_by_insight[doc_id] = meta.get("source_id")
            elif doc_type == "transcript":
                seg_docs.append(
                    SegmentDocument(
                        id=doc_id,
                        text=text,
                        show_id=meta.get("feed_id") or "",
                        episode_id=meta.get("episode_id") or "",
                        start_time=float(meta.get("timestamp_start_ms") or 0.0) / 1000.0,
                        end_time=float(meta.get("timestamp_end_ms") or 0.0) / 1000.0,
                        publish_date=meta.get("publish_date"),
                        embedding=_embed(text, model_id, allow_download=allow_download),
                    )
                )
            elif doc_type in _AUX_DOC_TYPES:
                aux_docs.append(
                    AuxDocument(
                        id=doc_id,
                        text=text,
                        show_id=meta.get("feed_id") or "",
                        episode_id=meta.get("episode_id") or "",
                        doc_type=doc_type,
                        publish_date=meta.get("publish_date"),
                        source_id=meta.get("source_id"),
                        embedding=_embed(text, model_id, allow_download=allow_download),
                    )
                )

        gi_path = _gi_path(episode_root, meta_path, doc)
        # Text-containment linking (verbatim grounding quotes) is the primary path —
        # it needs no segment timestamps, which the corpus often lacks. Fall back to
        # timestamp overlap for any insight whose quote text isn't found verbatim.
        grounding_text = _insight_grounding_quote_texts(gi_path)
        text_quotes = [
            (ins.id, grounding_text[node_id])
            for ins in ins_docs
            if (node_id := node_id_by_insight.get(ins.id)) in grounding_text
        ]
        mapping = link_insights_to_segments_by_text(seg_docs, text_quotes)
        grounding = _insight_grounding_quotes(gi_path)
        time_quotes = [
            (ins.id, *grounding[node_id])
            for ins in ins_docs
            if ins.id not in mapping and (node_id := node_id_by_insight.get(ins.id)) in grounding
        ]
        mapping.update(link_insights_to_segments(seg_docs, time_quotes))
        for ins in ins_docs:
            if ins.id in mapping:
                ins.source_segment_id = mapping[ins.id]
        stats.linked += len(mapping)

        # Accumulate into cross-episode buffers; flush a tier in one transaction once it
        # reaches the batch size. Counts reflect rows buffered (all get flushed below).
        seg_buf.extend(seg_docs)
        stats.segments += len(seg_docs)
        ins_buf.extend(ins_docs)
        stats.insights += len(ins_docs)
        aux_buf.extend(aux_docs)
        stats.aux += len(aux_docs)
        if len(seg_buf) >= batch:
            _flush_segments()
        if len(ins_buf) >= batch:
            _flush_insights()
        if len(aux_buf) >= batch:
            _flush_auxes()

    # Final flush of any partial buffers.
    _flush_segments()
    _flush_insights()
    _flush_auxes()

    # Full/stale reindex: MVCC-empty any tier that existed before but got no rows this
    # build, so its stale rows don't survive the "clear" (rmtree parity, no reader stranded).
    if clear_requested:
        _finalize_reindex_clear(Path(lance_path), backend, pre_existing_tiers, overwritten_tiers)

    if backend is not None:
        backend.write_index_meta(model_id)  # query path must embed in the same space
        backend.create_indices()
        # Reclaim the fragments + superseded versions this build created, so the index
        # stays bounded across full AND incremental reindexes.
        backend.compact()
        # Re-emit the FAISS-era ``metadata.json`` sidecar (doc_id -> meta) next to the index;
        # the GIL chunk-offset verifier reads char_start/char_end from it (see note above).
        (Path(lance_path).parent / "metadata.json").write_text(
            json.dumps(index_metadata, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    return stats

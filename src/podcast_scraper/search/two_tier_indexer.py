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

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast, Dict, List, Optional, Tuple

from .. import config as _config, config_constants as _config_constants
from ..providers.ml import embedding_loader
from .backend import AuxDocument, InsightDocument, SegmentDocument
from .backends.lancedb_backend import lance_index_is_stale, LanceDBBackend
from .corpus_scope import (
    discover_metadata_files,
    episode_root_from_metadata_path,
    index_fingerprint_scope_key,
)
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
# 2026-07-22: added ``episode_title`` / ``episode_description`` /
# ``summary_short`` (episode-level metadata surfaces from indexer.py) so the
# rows land in the aux table instead of being dropped by the else branch.
_AUX_DOC_TYPES = frozenset(
    {
        "kg_entity",
        "kg_topic",
        "quote",
        "summary",
        "episode_title",
        "episode_description",
        "summary_short",
    }
)


# D8: per-episode content fingerprints let an incremental reindex skip re-embedding UNCHANGED
# episodes (embedding is the O(N) cost; upsert-merge-on-id is idempotent for storage but was not
# for compute). Persisted next to the lance index. Bump ``_FP_SCHEMA_VERSION`` whenever the row
# construction / chunking changes so every fingerprint invalidates and the next build re-embeds all.
_EPISODE_FINGERPRINTS_FILENAME = "episode_fingerprints.json"
_FP_SCHEMA_VERSION = 1


@dataclass
class TwoTierIndexStats:
    """Counts from a from-corpus two-tier index build."""

    episodes: int = 0
    segments: int = 0
    insights: int = 0
    aux: int = 0
    linked: int = 0
    # D8: episodes whose content fingerprint matched the last build → embed+upsert skipped.
    episodes_skipped_unchanged: int = 0
    # RFC-118: episodes the backbone delta called UNCHANGED but the index fingerprint
    # re-embedded anyway. Nonzero = the two "what changed" definitions drifted (or the
    # embedding model changed, which only the index fingerprint tracks). Observational —
    # the index-level skip stays authoritative.
    backbone_disagreements: int = 0


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


def _episode_fingerprint(rows: List[Tuple[str, str, dict]], model_id: str) -> str:
    """Content hash of an episode's embeddable rows (D8).

    Hashes the exact ``(doc_id, text)`` pairs that would be embedded, plus ``model_id`` +
    ``_FP_SCHEMA_VERSION`` — so any change to embedded content, the embedding model, or the row
    construction changes the hash and forces a re-embed. Order-independent (rows are sorted).
    """
    h = hashlib.sha256()
    h.update(f"v{_FP_SCHEMA_VERSION}\x1f{model_id}".encode("utf-8"))
    for doc_id, text, _meta in sorted(rows, key=lambda r: str(r[0])):
        h.update(b"\x1e")
        h.update(str(doc_id).encode("utf-8"))
        h.update(b"\x1f")
        h.update(str(text).encode("utf-8"))
    return h.hexdigest()


def _episode_scope_key(rows: List[Tuple[str, str, dict]], doc: dict) -> Optional[str]:
    """Stable per-episode fingerprint key: ``index_fingerprint_scope_key(feed_id, episode_id)``.

    Resolves ``(feed_id, episode_id)`` from the embeddable rows' meta (each carries them), falling
    back to the loaded episode metadata. ``None`` when no episode_id is resolvable (never skip).
    """
    for _doc_id, _text, meta in rows:
        eid = meta.get("episode_id")
        if isinstance(eid, str) and eid:
            return index_fingerprint_scope_key(meta.get("feed_id"), eid)
    ep = doc.get("episode") if isinstance(doc, dict) else None
    if isinstance(ep, dict):
        eid = ep.get("episode_id")
        if isinstance(eid, str) and eid:
            return index_fingerprint_scope_key(ep.get("feed_id"), eid)
    return None


def _fingerprints_path(lance_path: Path | str) -> Path:
    return Path(lance_path).parent / _EPISODE_FINGERPRINTS_FILENAME


def _load_episode_fingerprints(lance_path: Path | str) -> Dict[str, str]:
    """Load ``{scope_key: fingerprint}`` from the sidecar next to the index; ``{}`` if absent."""
    try:
        data = json.loads(_fingerprints_path(lance_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    fps = data.get("fingerprints") if isinstance(data, dict) else None
    return {str(k): str(v) for k, v in fps.items()} if isinstance(fps, dict) else {}


def _write_episode_fingerprints(lance_path: Path | str, fps: Dict[str, str]) -> None:
    """Atomically persist the fingerprints sidecar (temp + rename). Non-fatal on failure."""
    p = _fingerprints_path(lance_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(
            json.dumps({"schema": _FP_SCHEMA_VERSION, "fingerprints": fps}, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(p)
    except OSError as exc:  # a fingerprint-write hiccup must never fail the index build
        logger.warning("could not write %s: %s", p, exc)


def _fingerprint_skip_unchanged(
    rows: List[Tuple[str, str, dict]],
    doc: dict,
    model_id: str,
    stored_fps: Dict[str, str],
    result_fps: Dict[str, str],
    clear_requested: bool,
    index_metadata: Dict[str, dict],
) -> bool:
    """D8: decide whether this episode is UNCHANGED and can skip embed+upsert.

    Records the episode's current fingerprint in ``result_fps`` (so it carries forward), and when
    skipping, populates ``index_metadata`` (cheap, no embed) so the rewritten metadata.json sidecar
    stays complete. Returns True only when a prior fingerprint matches and it is not a full rebuild.
    """
    fp = _episode_fingerprint(rows, model_id)
    scope_key = _episode_scope_key(rows, doc)
    if not scope_key:
        return False  # no stable key → always (re)embed
    result_fps[scope_key] = fp
    if clear_requested or stored_fps.get(scope_key) != fp:
        return False
    for doc_id, _text, meta in rows:
        index_metadata[doc_id] = {k: v for k, v in meta.items() if k != "text"}
    return True


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
    backbone_changed_relpaths: Optional[set] = None,
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

    # D8: load prior per-episode fingerprints so an incremental build can skip re-embedding
    # UNCHANGED episodes. A full/stale reindex (clear_requested) ignores them, rebuilds every
    # episode, then rewrites a fresh sidecar. ``result_fps`` accumulates every walked episode's
    # current fingerprint (skipped + embedded) and replaces the file on success.
    stored_fps: Dict[str, str] = {} if clear_requested else _load_episode_fingerprints(lance_path)
    result_fps: Dict[str, str] = {}

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
        # D8: skip re-embedding an UNCHANGED episode. ``_collect_docs_for_episode`` above is cheap;
        # ``_embed`` below is the O(N) cost. If the episode's content fingerprint matches the last
        # build, its rows are already embedded+upserted (merge on id) — skip embed+upsert.
        if _fingerprint_skip_unchanged(
            rows, doc, model_id, stored_fps, result_fps, clear_requested, index_metadata
        ):
            stats.episodes_skipped_unchanged += 1
            continue
        # RFC-118 (observational): the backbone delta called this episode's derivation inputs
        # unchanged, yet the index-level fingerprint decided to re-embed. Count the drift —
        # the two definitions differ legitimately only when the embedding model changed,
        # which the backbone deliberately doesn't track.
        if backbone_changed_relpaths is not None and meta_rel not in backbone_changed_relpaths:
            stats.backbone_disagreements += 1
            logger.debug(
                "backbone delta said unchanged but index fingerprint re-embeds: %s", meta_rel
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
                        insight_type=meta.get("insight_type"),
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

    _finalize_index_build(
        backend,
        lance_path,
        clear_requested=clear_requested,
        pre_existing_tiers=pre_existing_tiers,
        overwritten_tiers=overwritten_tiers,
        index_metadata=index_metadata,
        model_id=model_id,
    )
    # D8: persist fingerprints on success so the NEXT incremental build can skip unchanged episodes.
    # A mid-build failure returns/raises before here, leaving the old sidecar → next build re-embeds
    # (safe). Skipped on a limited walk (a partial episode set would drop fingerprints for un-walked
    # episodes, forcing needless re-embeds — but never staleness).
    if limit_episodes is None:
        _write_episode_fingerprints(lance_path, result_fps)
    return stats


def _finalize_index_build(
    backend: Optional[LanceDBBackend],
    lance_path: str | Path,
    *,
    clear_requested: bool,
    pre_existing_tiers: set,
    overwritten_tiers: set,
    index_metadata: Dict[str, dict],
    model_id: str,
) -> None:
    """Post-walk finalize: clear stale tiers, write index meta, compact, re-emit the sidecar, and
    bump the index dir mtime (D2). Extracted from ``build_two_tier_index`` to keep it under the
    complexity gate."""
    # Full/stale reindex: MVCC-empty any tier that existed before but got no rows this build, so its
    # stale rows don't survive the "clear" (rmtree parity, no reader stranded).
    if clear_requested:
        _finalize_reindex_clear(Path(lance_path), backend, pre_existing_tiers, overwritten_tiers)
    if backend is None:
        return
    backend.write_index_meta(model_id)  # query path must embed in the same space
    backend.create_indices()
    # Reclaim the fragments + superseded versions this build created, so the index stays bounded
    # across full AND incremental reindexes.
    backend.compact()
    # Re-emit the FAISS-era ``metadata.json`` sidecar (doc_id -> meta) next to the index; the GIL
    # chunk-offset verifier reads char_start/char_end from it.
    (Path(lance_path).parent / "metadata.json").write_text(
        json.dumps(index_metadata, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    # D2: LanceDB upserts write into per-table SUBDIRS and don't bump the top-level index dir
    # mtime — but read_lance_index_stats() caches on that mtime (perf_cache.lance_mtime) and derives
    # last_updated from it. Bump it after any build that changed the index so the stats endpoint +
    # viewer index card refresh — INCLUDING a pipeline-subprocess reindex (an in-process cache clear
    # can't cross that boundary; the on-disk mtime can).
    try:
        os.utime(lance_path, None)
    except OSError:  # non-fatal: a stale stats card must never fail the index build
        pass

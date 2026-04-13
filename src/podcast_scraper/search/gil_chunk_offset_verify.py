"""Verify GIL Quote char offsets vs FAISS transcript chunk offsets (#528 / RFC-072 Phase 5)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple


def half_open_ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    """Return True if ``[a0, a1)`` and ``[b0, b1)`` intersect with positive length."""
    return a0 < b1 and b0 < a1


def overlap_width(a0: int, a1: int, b0: int, b1: int) -> int:
    """Length of intersection of half-open ranges (0 if none)."""
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0, hi - lo)


def load_index_metadata_map(index_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load FAISS ``metadata.json`` (doc_id → metadata dict)."""
    path = index_dir / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"metadata.json not found under {index_dir}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("metadata.json must contain a JSON object")
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, dict):
            out[k] = v
    return out


def transcript_chunk_spans_by_episode(
    metadata: Mapping[str, Mapping[str, Any]],
) -> Dict[str, List[Tuple[int, int]]]:
    """Group transcript chunk ``(char_start, char_end)`` by ``episode_id``."""
    out: Dict[str, List[Tuple[int, int]]] = {}
    for _doc_id, meta in metadata.items():
        if meta.get("doc_type") != "transcript":
            continue
        ep = meta.get("episode_id")
        if not isinstance(ep, str) or not ep.strip():
            continue
        try:
            cs = int(meta["char_start"])
            ce = int(meta["char_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if ce <= cs:
            continue
        out.setdefault(ep.strip(), []).append((cs, ce))
    return out


def quote_spans_from_gi(gi: Mapping[str, Any]) -> List[Tuple[str, int, int]]:
    """Return ``(quote_id, char_start, char_end)`` for each Quote node."""
    rows: List[Tuple[str, int, int]] = []
    for node in gi.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        if node.get("type") != "Quote":
            continue
        qid = node.get("id")
        props = node.get("properties")
        if not isinstance(qid, str) or not isinstance(props, dict):
            continue
        try:
            cs = int(props["char_start"])
            ce = int(props["char_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if ce <= cs:
            continue
        rows.append((qid, cs, ce))
    return rows


@dataclass
class EpisodeQuoteOffsetStats:
    """Per-episode quote vs chunk overlap stats."""

    episode_id: str
    transcript_chunks: int = 0
    quotes: int = 0
    quotes_with_overlap: int = 0
    quotes_without_overlap: int = 0
    sample_missing: List[str] = field(default_factory=list)


def _episode_report(
    episode_id: str,
    quote_spans: Sequence[Tuple[str, int, int]],
    chunk_spans: Sequence[Tuple[int, int]],
    *,
    max_samples: int,
) -> EpisodeQuoteOffsetStats:
    st = EpisodeQuoteOffsetStats(episode_id=episode_id, transcript_chunks=len(chunk_spans))
    st.quotes = len(quote_spans)
    if not quote_spans:
        return st
    if not chunk_spans:
        st.quotes_without_overlap = st.quotes
        for qid, _cs, _ce in quote_spans[:max_samples]:
            st.sample_missing.append(qid)
        return st
    for qid, q0, q1 in quote_spans:
        if any(half_open_ranges_overlap(q0, q1, c0, c1) for c0, c1 in chunk_spans):
            st.quotes_with_overlap += 1
        else:
            st.quotes_without_overlap += 1
            if len(st.sample_missing) < max_samples:
                st.sample_missing.append(qid)
    return st


def build_offset_alignment_report(
    *,
    gi_by_episode: Mapping[str, Path],
    metadata_by_doc: Mapping[str, Mapping[str, Any]],
    max_samples_per_episode: int = 8,
) -> dict[str, Any]:
    """Compare Quote spans from GI files to FAISS transcript chunk spans per episode.

    Only episodes present in ``gi_by_episode`` are scanned for quotes. Chunk spans
    come from all transcript rows in the index metadata.
    """
    chunks_by_ep = transcript_chunk_spans_by_episode(metadata_by_doc)
    episodes: List[dict[str, Any]] = []
    total_quotes = 0
    total_with_overlap = 0
    episodes_no_chunks = 0

    for eid in sorted(gi_by_episode.keys()):
        gpath = gi_by_episode[eid]
        chunk_spans = chunks_by_ep.get(eid, [])
        if not chunk_spans:
            episodes_no_chunks += 1
        try:
            gi = json.loads(gpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            episodes.append(
                {
                    "episode_id": eid,
                    "error": "gi_load_failed",
                    "path": str(gpath),
                }
            )
            continue
        if not isinstance(gi, dict):
            episodes.append({"episode_id": eid, "error": "gi_not_object"})
            continue
        qspans = quote_spans_from_gi(gi)
        st = _episode_report(eid, qspans, chunk_spans, max_samples=max_samples_per_episode)
        total_quotes += st.quotes
        total_with_overlap += st.quotes_with_overlap
        episodes.append(
            {
                "episode_id": eid,
                "transcript_chunks": st.transcript_chunks,
                "quotes": st.quotes,
                "quotes_with_chunk_overlap": st.quotes_with_overlap,
                "quotes_without_chunk_overlap": st.quotes_without_overlap,
                "sample_quote_ids_without_overlap": st.sample_missing,
            }
        )

    overlap_rate: float | None
    if total_quotes > 0:
        overlap_rate = total_with_overlap / total_quotes
    else:
        overlap_rate = None

    warnings: List[str] = []
    if episodes_no_chunks:
        warnings.append(
            f"{episodes_no_chunks} episode(s) have GI on disk but no transcript vectors "
            "in the index (skipped chunk overlap for those quotes)."
        )
    if total_quotes == 0:
        warnings.append("No Quote nodes found in scanned GI files.")

    if overlap_rate is None:
        verdict = "no_quotes"
    elif overlap_rate >= 0.99 and episodes_no_chunks == 0:
        verdict = "aligned"
    elif overlap_rate >= 0.85:
        verdict = "mostly_aligned"
    else:
        verdict = "divergent"

    return {
        "verdict": verdict,
        "overlap_rate": overlap_rate,
        "quotes_total": total_quotes,
        "quotes_with_chunk_overlap": total_with_overlap,
        "episodes_scanned": len(gi_by_episode),
        "episodes_without_transcript_chunks": episodes_no_chunks,
        "episodes": episodes,
        "warnings": warnings,
    }


def merge_report_dict(target: MutableMapping[str, Any], extra: Mapping[str, Any]) -> None:
    """Shallow merge string keys from ``extra`` into ``target`` (for CLI metadata)."""
    for k, v in extra.items():
        target[str(k)] = v

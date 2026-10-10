"""0028 — give every transcript passage in the search index the time it was said.

The indexer timed transcript chunks only from ``summary.timestamps``, which the pipeline does not
write, so on prod every passage sat at 0 ms (operator 2026-10-10) and the app had to find the time
again at read time. The indexer now times chunks from the segments of the body it chunks
(``indexer._chunk_timestamps_from_segments``); this brings the rows already in the index to the
same state without re-indexing: no embedding, no model, text on disk only.

For each transcript row at 0 ms in ``segments`` / ``segments_nonen`` whose sidecar entry
(``search/metadata.json``) has a char span: map the span to time through that episode's segments,
exactly as the indexer now does, and write ``start_time`` / ``end_time`` in place. A row is only
written when its stored text is the slice of the current body it maps to — a transcript re-rendered
since indexing is left alone rather than mis-timed. A source-language chunk is timed from the
source text and its own segments, as its offsets index that text. The sidecar's
``timestamp_*_ms`` are updated to match.

``--dry-run`` counts and writes nothing. Idempotent: a timed row is never selected again.
``undo(root)`` puts every written row and sidecar entry back to what it was.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..migration import Migration, MigrationContext, MigrationResult
from ..ownership import match_corpus_owner

MIGRATION_ID = "0028_transcript_chunk_times"
UNDO_FILE = "transcript_chunk_times_undo.json"
_TIERS = ("segment", "segment_nonen")
_BATCH = 1000


def _lance_path(ctx: MigrationContext) -> Path:
    return Path(ctx.options.get("lance_path") or ctx.corpus_root / "search" / "lance_index")


def _load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _quote(v: str) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def _untimed_ids(table: Any) -> List[str]:
    n = int(table.count_rows())
    if not n:
        return []
    rows = table.search().where("start_time = 0 AND end_time = 0").select(["id"]).limit(n).to_list()
    return [str(r["id"]) for r in rows]


def _texts(table: Any, ids: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i in range(0, len(ids), _BATCH):
        chunk = ids[i : i + _BATCH]
        rows = (
            table.search()
            .where(f"id IN ({', '.join(_quote(x) for x in chunk)})")
            .select(["id", "text"])
            .limit(len(chunk))
            .to_list()
        )
        out.update({str(r["id"]): str(r.get("text") or "") for r in rows})
    return out


class _Plan:
    """Which rows get which time, and why the others do not."""

    def __init__(self) -> None:
        self.times: Dict[str, Dict[str, Tuple[int, int]]] = {t: {} for t in _TIERS}
        self.counts: Dict[str, int] = {
            "untimed_rows": 0,
            "timed": 0,
            "no_sidecar_span": 0,
            "source_layer": 0,
            "no_segments": 0,
            "text_mismatch": 0,
            "no_overlap": 0,
        }


def _plan(ctx: MigrationContext) -> Tuple[Optional[_Plan], str]:
    from ...search.backends.lancedb_backend import LanceDBBackend
    from ...search.chunker import _merge_time_for_span
    from ...search.indexer import (
        _indexed_text_language,
        _source_layer_body,
        body_timestamps,
        is_source_layer_chunk,
        source_body_timestamps,
    )

    root = ctx.corpus_root
    lance = _lance_path(ctx)
    sidecar = _load(lance.parent / "metadata.json")
    if not lance.exists() or not isinstance(sidecar, dict):
        return None, "no search index or no index sidecar: nothing to time"
    backend = LanceDBBackend(str(lance))
    plan = _Plan()
    body_cache: Dict[Tuple[str, bool], Optional[Tuple[str, List[Dict[str, Any]]]]] = {}

    def body_for(meta_rel: str, source: bool) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
        """The text a row's offsets index, with its times: the analysis body, or the source one."""
        key = (meta_rel, source)
        if key in body_cache:
            return body_cache[key]
        from ...workflow.transcript_resolution import load_transcript, TranscriptPurpose

        body_cache[key] = None
        meta_path = root / meta_rel
        doc = _load(meta_path)
        if not isinstance(doc, dict):
            return None
        run_dir = meta_path.parent.parent
        if source:
            layer = _source_layer_body(
                run_dir, doc, _indexed_text_language(run_dir, meta_path, doc)
            )
            if layer is None:
                return None
            text = layer[0].read_text(encoding="utf-8", errors="replace")
            ts = source_body_timestamps(layer[0], text)
        else:
            rel = (doc.get("content") or {}).get("transcript_file_path")
            if not isinstance(rel, str) or not rel.strip():
                return None
            body = load_transcript(run_dir, rel.strip(), purpose=TranscriptPurpose.ANALYSIS)
            text, ts = body.text, body_timestamps(body.text, body.segments)
        if ts:
            body_cache[key] = (text, ts)
        return body_cache[key]

    for tier in _TIERS:
        table = backend._open_if_exists(tier)
        if table is None:
            continue
        ids = _untimed_ids(table)
        plan.counts["untimed_rows"] += len(ids)
        candidates: Dict[str, Tuple[str, int, int, bool]] = {}
        for row_id in ids:
            meta = sidecar.get(row_id)
            if not isinstance(meta, dict):
                plan.counts["no_sidecar_span"] += 1
                continue
            source = is_source_layer_chunk(meta)
            cs, ce, rel = (
                meta.get("char_start"),
                meta.get("char_end"),
                meta.get("source_metadata_relative_path"),
            )
            if not isinstance(cs, int) or not isinstance(ce, int) or not isinstance(rel, str):
                plan.counts["no_sidecar_span"] += 1
                continue
            candidates[row_id] = (rel, cs, ce, source)
            plan.counts["source_layer"] += int(source)
        texts = _texts(table, sorted(candidates))
        for row_id, (rel, cs, ce, source) in sorted(candidates.items()):
            body = body_for(rel, source)
            if body is None:
                plan.counts["no_segments"] += 1
                continue
            text, ts = body
            if text[cs:ce].strip() != texts.get(row_id, "").strip():
                plan.counts["text_mismatch"] += 1
                continue
            start, end = _merge_time_for_span(cs, ce, ts)
            if start is None or end is None:
                plan.counts["no_overlap"] += 1
                continue
            plan.times[tier][row_id] = (start, end)
            plan.counts["timed"] += 1
    return plan, ""


def _summary(plan: _Plan) -> str:
    c = plan.counts
    return (
        f"{c['untimed_rows']} transcript row(s) at 0 ms; {c['timed']} timeable from segments on "
        f"disk; left alone: {c['no_segments']} no segments, "
        f"{c['text_mismatch']} transcript changed "
        f"since indexing, {c['no_overlap']} no overlapping segment, {c['no_sidecar_span']} no "
        f"char span ({c['source_layer']} of the rows are source-language text)"
    )


def _set_times(table: Any, times: Dict[str, Tuple[float, float]]) -> None:
    """``{id: (start_s, end_s)}`` written in batches, one transaction per batch.

    A partial-column ``merge_insert`` on ``id``: only the two time columns are sent, so the text,
    the embedding and every other column stay exactly as indexed. (``update`` cannot set a
    different value per row: Lance rejects a ``CASE`` expression.)
    """
    import pyarrow as pa

    items = sorted(times.items())
    for i in range(0, len(items), _BATCH):
        chunk = items[i : i + _BATCH]
        source = pa.table(
            {
                "id": [k for k, _ in chunk],
                "start_time": [float(s) for _, (s, _e) in chunk],
                "end_time": [float(e) for _, (_s, e) in chunk],
            }
        )
        table.merge_insert("id").when_matched_update_all().execute(source)


def undo(root: Path, lance_path: Optional[Path] = None) -> Tuple[int, List[str]]:
    """Put every row and sidecar entry 0028 wrote back to its recorded value."""
    from ...search.backends.lancedb_backend import LanceDBBackend

    root = Path(root)
    lance = Path(lance_path or root / "search" / "lance_index")
    record = _load(lance.parent / UNDO_FILE)
    if not isinstance(record, dict):
        return 0, [f"no {UNDO_FILE}: nothing to undo"]
    backend = LanceDBBackend(str(lance))
    restored = 0
    for tier, rows in (record.get("rows") or {}).items():
        table = backend._open_if_exists(tier)
        if table is None or not rows:
            continue
        _set_times(table, {k: (float(v[0]), float(v[1])) for k, v in rows.items()})
        restored += len(rows)
    sidecar_path = lance.parent / "metadata.json"
    sidecar = _load(sidecar_path)
    if isinstance(sidecar, dict):
        for row_id, (s, e) in (record.get("sidecar") or {}).items():
            if isinstance(sidecar.get(row_id), dict):
                sidecar[row_id]["timestamp_start_ms"] = s
                sidecar[row_id]["timestamp_end_ms"] = e
        _write_atomic(sidecar_path, sidecar)
    (lance.parent / UNDO_FILE).rename(lance.parent / (UNDO_FILE + ".undone"))
    return restored, []


class TranscriptChunkTimesMigration(Migration):
    """Time the transcript rows already in the index from the segments on disk."""

    id = MIGRATION_ID
    to_version = "2.7.22"
    description = (
        "Give every transcript passage in the search index the time it was said (all sat at 0 ms), "
        "from the timed transcript on disk, in place — no re-index"
    )

    def plan(self, ctx: MigrationContext) -> str:
        """What apply() would write — reads the index and the corpus only."""
        plan, why = _plan(ctx)
        return why if plan is None else _summary(plan)

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """No transcript row at 0 ms is timeable from what is on disk."""
        plan, why = _plan(ctx)
        if plan is None:
            return True, why
        left = plan.counts["timed"]
        if left:
            return False, f"{left} transcript row(s) still at 0 ms though their segments time them"
        return True, _summary(plan)

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Write the planned times in place, record the old values first, update the sidecar."""
        from ...search.backends.lancedb_backend import LanceDBBackend

        plan, why = _plan(ctx)
        if plan is None:
            ctx.log(why)
            return MigrationResult(
                self.id, applied=not ctx.dry_run, dry_run=ctx.dry_run, message=why
            )
        message = _summary(plan)
        ctx.log(message)
        if ctx.dry_run or not plan.counts["timed"]:
            return MigrationResult(
                self.id,
                applied=not ctx.dry_run,
                dry_run=ctx.dry_run,
                message=message,
                details=dict(plan.counts),
            )
        lance = _lance_path(ctx)
        sidecar_path = lance.parent / "metadata.json"
        sidecar = _load(sidecar_path) or {}
        undo_path = lance.parent / UNDO_FILE
        previous = _load(undo_path) if undo_path.exists() else None
        record: Dict[str, Any] = (
            previous
            if isinstance(previous, dict)
            else {"migration": MIGRATION_ID, "rows": {}, "sidecar": {}}
        )
        for tier, times in plan.times.items():
            rows = record["rows"].setdefault(tier, {})
            for row_id in times:
                rows.setdefault(row_id, [0.0, 0.0])
                meta = sidecar.get(row_id) or {}
                record["sidecar"].setdefault(
                    row_id, [meta.get("timestamp_start_ms"), meta.get("timestamp_end_ms")]
                )
        # The undo record is on disk before the first row changes.
        _write_atomic(undo_path, record)
        backend = LanceDBBackend(str(lance))
        for tier, times in plan.times.items():
            table = backend._open_if_exists(tier)
            if table is None or not times:
                continue
            _set_times(table, {k: (s / 1000.0, e / 1000.0) for k, (s, e) in times.items()})
            for row_id, (s, e) in times.items():
                if isinstance(sidecar.get(row_id), dict):
                    sidecar[row_id]["timestamp_start_ms"] = s
                    sidecar[row_id]["timestamp_end_ms"] = e
        _write_atomic(sidecar_path, sidecar)
        backend.compact()
        match_corpus_owner(ctx.corpus_root, [undo_path, sidecar_path])
        return MigrationResult(
            self.id, applied=True, dry_run=False, message=message, details=dict(plan.counts)
        )

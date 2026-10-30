"""0038 — served episodes stop crediting raw diarization labels as speakers (RFC-131 N11).

Before RFC-131 S3, a segment still carrying a raw diarization label (``SPEAKER_07``, or the
speaker detector's default ``unknown_guest_1``) and no voice type fell through
``gi.pipeline._voice_type_for_char_range`` to a Person node named after the label, with SPOKEN_BY
edges, and its insights stayed surfaceable. S3 fixed the pipeline (such a voice is ``unknown``) and
the read time. This migration brings the files of already-served episodes to what the pipeline now
writes, from text on disk only (no LLM, no DGX, no reprocess):

- a Quote credited to a raw-label Person loses ``speaker_id`` (and a raw ``speaker_name``) and its
  SPOKEN_BY edge, and takes ``speaker_voice_type`` from the segment holding its ``char_start``
  (``unknown`` when that segment carries a raw label and no type, or when no segment can be found:
  its own record says we failed to name the voice);
- an Insight whose FIRST grounded quote (SUPPORTED_BY order, the pipeline's rule) is such a quote
  takes that quote's type, ``surfaceable: false`` when the pipeline's
  ``VOICE_TYPE_NOT_SURFACEABLE`` says so, and its routing tag and salience from the pipeline's
  ``_apply_route_and_tag`` (when it carries a ``tier``);
- the raw-label Person node is removed (nothing references it after the above), and so is its
  bridge identity.

An Insight's ``speaker`` field is NOT touched: the pipeline writes the raw voice id of the segment
under its first quote there (``_speaker_for_insight``) for every insight, named or not; 28,938
insights in 1,340 served episodes carry one (dry run on snapshot-prod-20261030, 2026-10-31).

Nothing else is written. Search-index Quote rows carry ``speaker_id`` (``search/indexer.py``); they
are edited in place by the run steps of change record 0038, not here. Measured on
snapshot-prod-20261030: 28 Person nodes in 24 episodes, 107 quotes, 53 insights (record 0038).
UNDO restores from ``.podcast_scraper/upgrade-backups/0038/`` through
``raw_label_speakers_removed.jsonl``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ...gi.pipeline import _apply_route_and_tag, _segment_char_spans, VOICE_TYPE_NOT_SURFACEABLE
from ...graph_id_utils import is_raw_speaker_label
from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult

MIGRATION_ID = "0038_raw_label_speakers_removed"
RECEIPTS_FILE = "raw_label_speakers_removed.jsonl"
BACKUP_TAG = "0038"

#: The per-episode counters, in the apply message and the dry-run report.
COUNTERS = (
    "persons_removed",
    "spoken_by_removed",
    "quotes_unattributed",
    "quotes_unplaced",
    "insights_retyped",
    "insights_rerouted",
    "bridge_identities_removed",
)


def _json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _is_raw(name: Any) -> bool:
    return isinstance(name, str) and is_raw_speaker_label(name.strip())


def _segments_for(
    meta_path: Path, ref: str, cache: Dict[str, Any], root: Path
) -> Optional[List[tuple]]:
    """Segment char spans of the transcript a quote names: its ``transcript_ref`` resolved as the
    gold join resolves it (``quality.gold_insights.resolve_ref``: relative to the run directory,
    ``/app/output/`` = the corpus root), else the same file name in the run's ``transcripts/``."""
    from ...quality.gold_insights import resolve_ref

    if ref in cache:
        cached: Optional[List[tuple]] = cache[ref]
        return cached
    spans: Optional[List[tuple]] = None
    path = resolve_ref(ref, root, meta_path.parent.parent) if ref else None
    if path is None or not path.is_file():
        alt = meta_path.parent.parent / "transcripts" / Path(ref).name if ref else None
        path = alt if alt is not None and alt.is_file() else None
    if path is not None and path.name.endswith(".txt"):
        rows = _json(Path(str(path)[: -len(".txt")] + ".segments.json"))
        rows = rows if isinstance(rows, list) else (rows or {}).get("segments")
        rows = [r for r in rows or [] if isinstance(r, dict)]
        text = path.read_text(encoding="utf-8")
        spans = _segment_char_spans(text, rows) if text and rows else None
    cache[ref] = spans
    return spans


class Refused(Exception):
    """The episode is left untouched; the reason is counted."""


def _voice_type(
    props: Dict[str, Any], meta_path: Path, cache: Dict[str, Any], root: Path
) -> Tuple[str, bool]:
    """``(voice type, placed)`` for a raw-credited quote, from the segment holding its start.

    A segment that now carries a NAME and no type refuses the episode: the GI is older than the
    naming, and only a naming migration may credit that name (none on snapshot-prod-20261030)."""
    start = props.get("char_start")
    spans = _segments_for(meta_path, str(props.get("transcript_ref") or ""), cache, root)
    if isinstance(start, int):
        for a, b, seg in spans or ():
            if a <= start < b:
                vt = seg.get("voice_type")
                if vt:
                    return str(vt), True
                label = seg.get("speaker_label")
                if label and not _is_raw(label):
                    raise Refused("a raw-credited quote is on a named segment")
                return "unknown", True
    return "unknown", False


def episode_changes(
    meta_path: Path, root: Path
) -> Optional[Tuple[Dict[Path, Any], Dict[str, int]]]:
    """``({path: new payload}, counters)`` for one episode, or None when it has nothing to change.

    Pure: reads the episode's GI, bridge and transcripts, writes nothing."""
    stem = meta_path.name[: -len(".metadata.json")]
    gi_path = meta_path.with_name(f"{stem}.gi.json")
    gi = _json(gi_path)
    if not isinstance(gi, dict):
        return None
    nodes = [n for n in gi.get("nodes") or [] if isinstance(n, dict)]
    raw: Set[str] = {
        str(n.get("id"))
        for n in nodes
        if n.get("type") == "Person" and _is_raw((n.get("properties") or {}).get("name"))
    }
    bridge_path = meta_path.with_name(f"{stem}.bridge.json")
    bridge = _json(bridge_path)
    idents = bridge.get("identities") if isinstance(bridge, dict) else None
    raw_idents = [
        i
        for i in idents or []
        if isinstance(i, dict) and (i.get("id") in raw or _is_raw(i.get("display_name")))
    ]
    if not raw and not raw_idents:
        return None

    c = {k: 0 for k in COUNTERS}
    new_gi = copy.deepcopy(gi)
    by_id = {str(n.get("id")): n for n in new_gi.get("nodes") or [] if isinstance(n, dict)}
    cache: Dict[str, Any] = {}
    quote_type: Dict[str, str] = {}
    for nid, n in by_id.items():
        props = n.get("properties") or {}
        if n.get("type") != "Quote" or props.get("speaker_id") not in raw:
            continue
        vt, placed = _voice_type(props, meta_path, cache, root)
        props.pop("speaker_id", None)
        if _is_raw(props.get("speaker_name")):
            props.pop("speaker_name", None)
        props["speaker_voice_type"] = vt
        quote_type[nid] = vt
        c["quotes_unattributed"] += 1
        c["quotes_unplaced"] += 0 if placed else 1
    first: Dict[str, str] = {}
    for e in new_gi.get("edges") or []:
        if isinstance(e, dict) and e.get("type") == "SUPPORTED_BY":
            first.setdefault(str(e.get("from")), str(e.get("to")))
    for iid, qid in first.items():
        node = by_id.get(iid)
        if node is None or node.get("type") != "Insight" or qid not in quote_type:
            continue
        props = node.setdefault("properties", {})
        vt = quote_type[qid]
        props["speaker_voice_type"] = vt
        if vt in VOICE_TYPE_NOT_SURFACEABLE:
            props["surfaceable"] = False
        c["insights_retyped"] += 1
        if isinstance(props.get("tier"), int):
            _apply_route_and_tag(props, props["tier"])
            c["insights_rerouted"] += 1
    edges = new_gi.get("edges") or []
    kept = [
        e
        for e in edges
        if not (isinstance(e, dict) and e.get("type") == "SPOKEN_BY" and e.get("to") in raw)
    ]
    c["spoken_by_removed"] = len(edges) - len(kept)
    new_gi["edges"] = kept
    new_gi["nodes"] = [n for n in new_gi.get("nodes") or [] if n.get("id") not in raw]
    c["persons_removed"] = len(raw)

    writes: Dict[Path, Any] = {gi_path: new_gi}
    if raw_idents and isinstance(bridge, dict):
        new_bridge = copy.deepcopy(bridge)
        new_bridge["identities"] = [i for i in idents or [] if i not in raw_idents]
        writes[bridge_path] = new_bridge
        c["bridge_identities_removed"] = len(raw_idents)
    return writes, c


def find_changes(
    root: Path, refused: Optional[Dict[str, List[str]]] = None
) -> Iterable[Tuple[Path, Dict[Path, Any], Dict[str, int]]]:
    """``(metadata, writes, counters)`` per SERVED episode with something to change; a refused
    episode is listed in ``refused`` by reason and left untouched."""
    for meta_path in select_served_artifacts(Path(root), ".metadata.json")[0]:
        try:
            found = episode_changes(meta_path, Path(root))
        except Refused as exc:
            if refused is not None:
                refused.setdefault(str(exc), []).append(str(meta_path.relative_to(root)))
            continue
        if found is not None:
            yield meta_path, found[0], found[1]


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class RawLabelSpeakersRemovedMigration(Migration):
    """No served GI file credits a raw diarization label as a speaker; no bridge shows one."""

    id = MIGRATION_ID
    to_version = "2.7.32"
    description = (
        "quotes and insights credited to a raw diarization label take the voice type the pipeline "
        "now gives such a voice; the raw-label Person nodes and bridge identities go (text only)"
    )

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would change: pure read, no writes."""
        totals = {k: 0 for k in COUNTERS}
        eps = 0
        for _m, _w, c in find_changes(ctx.corpus_root):
            eps += 1
            for k in COUNTERS:
                totals[k] += c[k]
        return f"{self.id}: {eps} episode(s); " + ", ".join(f"{k}={v}" for k, v in totals.items())

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Write each changed GI and bridge, backed up and receipted."""
        root = Path(ctx.corpus_root)
        totals = {k: 0 for k in COUNTERS}
        episodes: List[str] = []
        receipts: List[Dict[str, str]] = []
        refused: Dict[str, List[str]] = {}
        for meta_path, writes, c in find_changes(root, refused):
            episodes.append(str(meta_path.relative_to(root)))
            for k in COUNTERS:
                totals[k] += c[k]
            if ctx.dry_run:
                continue
            for path, payload in writes.items():
                receipts.append(write_with_backup(root, BACKUP_TAG, path, payload))
        if not ctx.dry_run:
            append_receipts(root, RECEIPTS_FILE, {"migration": self.id}, receipts)
        message = (
            f"{len(episodes)} episode(s); "
            + ", ".join(f"{k}={v}" for k, v in totals.items())
            + "".join(f", refused[{k}]={len(v)}" for k, v in sorted(refused.items()))
        )
        return MigrationResult(
            migration_id=self.id,
            applied=not ctx.dry_run,
            dry_run=ctx.dry_run,
            message=message,
            details={
                "episodes": episodes,
                "episodes_written": len(episodes),
                "refused": refused,
                **totals,
            },
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """No served GI credits a raw label as a speaker; no bridge shows one."""
        refused: Dict[str, List[str]] = {}
        left = list(find_changes(ctx.corpus_root, refused))
        held = sum(len(v) for v in refused.values())
        if left or held:
            return False, (
                f"{len(left)} served episode(s) still credit a raw diarization label; "
                f"{held} refused"
            )
        return True, "no served episode credits a raw diarization label as a speaker"


def dry_run_report(root: Path) -> Iterable[Dict[str, Any]]:
    """Read-only: one row per episode the migration would write, with its counters and input
    sha256s (the outcome hook's plan diff)."""
    from ..read_recorder import recording

    for meta_path in select_served_artifacts(Path(root), ".metadata.json")[0]:
        with recording(Path(root)) as rec:
            try:
                found = episode_changes(meta_path, Path(root))
            except Refused as exc:
                yield {"meta": str(meta_path.relative_to(root)), "refused": str(exc)}
                continue
        if found is None:
            continue
        yield {
            "meta": str(meta_path.relative_to(root)),
            "counts": found[1],
            "inputs": rec.inputs(),
        }

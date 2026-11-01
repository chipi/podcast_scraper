"""0039 — every ad-free segment's offsets point at its own words again (text only).

Found by 0041's offset guard (RFC-131 D2, 2026-11-01). Record:
``docs/wip/changes/0039_adfree_segment_offsets_realigned.md``.

A rewrite of an episode's ``.adfree.txt`` that changes the text's length must move every
``.adfree.segments.json`` offset after the change. A one-off hand repair (2026-10-06) removed a
speaker prefix and moved none: one served episode's rows from the change on point 1 to 8
characters early, and m0023's in-place carry, which every speaker rewrite since relies on, then
refuses the episode ("a segment would not slice to its text").

**The rule, all or none per episode.** Each row whose text the stored span does not slice takes
the span of that text found exactly once within ``MAX_DRIFT`` characters of its stored start. The
drift may only grow, in one direction, and spans may not overlap. Anything else refuses the
episode, untouched and listed. Only ``char_start`` / ``char_end`` change; the ad-free text, the ad
map (it indexes the raw transcript) and every other field stay. UNDO restores from
``.podcast_scraper/upgrade-backups/0039/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult

MIGRATION_ID = "0039_adfree_segment_offsets_realigned"
RECEIPTS_FILE = "adfree_segment_offsets_realigned.jsonl"
BACKUP_TAG = "0039"
#: Twice the largest drift measured (8 characters on the one episode, snapshot-prod-20261030: one
#: more per shortened prefix). A repeated short line ("Right.") inside the window is refused by the
#: found-exactly-once rule, not by the window's width.
MAX_DRIFT = 16


class Refused(ValueError):
    """The episode's rows cannot be realigned under the rule."""


def _load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _rows(payload: Any) -> List[Any]:
    rows = payload.get("segments") if isinstance(payload, dict) else payload
    return rows if isinstance(rows, list) else []


def _slices(text: str, row: Any) -> bool:
    cs, ce = row.get("char_start"), row.get("char_end")
    return isinstance(cs, int) and isinstance(ce, int) and text[cs:ce] == row.get("text")


def realigned(text: str, payload: Any) -> Tuple[Any, int]:
    """``(payload with realigned offsets, rows moved)``; raises :class:`Refused`. ``payload`` is
    not modified."""
    out = json.loads(json.dumps(payload))
    moved = 0
    direction = 0
    last_drift = 0
    prev_end = 0
    for row in _rows(out):
        if not isinstance(row, dict) or not isinstance(row.get("char_start"), int):
            continue
        words = row.get("text")
        if not isinstance(words, str) or not words:
            continue
        cs = row["char_start"]
        if not _slices(text, row):
            lo, hi = max(0, cs - MAX_DRIFT), cs + MAX_DRIFT + len(words)
            found = [i for i in range(lo, min(hi, len(text)) + 1) if text.startswith(words, i)]
            found = [i for i in found if abs(i - cs) <= MAX_DRIFT]
            if len(found) != 1:
                raise Refused(f"row at {cs}: its text is found {len(found)} times near it")
            new = found[0]
            drift = new - cs
            sign = 1 if drift > 0 else -1
            if direction and sign != direction:
                raise Refused(f"row at {cs}: the drift changes direction")
            if abs(drift) < abs(last_drift):
                raise Refused(f"row at {cs}: the drift shrinks ({last_drift} then {drift})")
            direction, last_drift = sign, drift
            row["char_start"], row["char_end"] = new, new + len(words)
            moved += 1
        elif last_drift:
            raise Refused(f"row at {cs}: slices again after the drift began")
        if row["char_start"] < prev_end:
            raise Refused(f"row at {row['char_start']}: overlaps the row before")
        prev_end = row["char_end"]
    return out, moved


class _Episode:
    """One episode's ad-free segments, realigned in memory."""

    def __init__(self, meta: Path) -> None:
        self.meta = meta
        self.payload: Optional[Any] = None
        self.seg_path: Optional[Path] = None
        self.moved = 0
        self.refused: Optional[str] = None

    def plan(self) -> bool:
        doc = _load(self.meta)
        rel = str(((doc or {}).get("content") or {}).get("transcript_file_path") or "")
        if not rel.endswith(".txt") or Path(rel).is_absolute():
            return False
        run = self.meta.parent.parent
        text_path = run / rel.replace(".txt", ".adfree.txt")
        seg_path = run / rel.replace(".txt", ".adfree.segments.json")
        payload = _load(seg_path)
        if payload is None or not text_path.is_file():
            return False
        text = text_path.read_text(encoding="utf-8")
        if all(
            _slices(text, r) for r in _rows(payload) if isinstance(r, dict) and "char_start" in r
        ):
            return False
        try:
            self.payload, self.moved = realigned(text, payload)
        except Refused as exc:
            self.refused = str(exc)
            return False
        self.seg_path = seg_path
        return self.moved > 0


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


def _scan(root: Path) -> Tuple[List[_Episode], List[_Episode]]:
    write: List[_Episode] = []
    refused: List[_Episode] = []
    for meta in select_served_artifacts(root, ".metadata.json")[0]:
        ep = _Episode(meta)
        if ep.plan():
            write.append(ep)
        if ep.refused:
            refused.append(ep)
    return write, refused


class AdfreeSegmentOffsetsRealignedMigration(Migration):
    """Ad-free segment offsets that no longer slice to their text, realigned from the text."""

    id = MIGRATION_ID
    to_version = "2.7.33"
    description = (
        "ad-free segment offsets that no longer slice to their own text (a 2026-10-06 hand repair) "
        "realigned from the text: found once, near, drift monotone, all or none per episode"
    )

    def plan(self, ctx: MigrationContext) -> str:
        """Count the episodes and rows apply() would realign, and those it refuses; pure read."""
        write, refused = _scan(ctx.corpus_root)
        rows = sum(ep.moved for ep in write)
        return (
            f"ad-free offsets plan: {len(write)} episode(s), {rows} row(s), {len(refused)} refused"
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Pass when every served ad-free segment slices to its own text and nothing is refused."""
        write, refused = _scan(ctx.corpus_root)
        if write or refused:
            names = [ep.meta.name for ep in write + refused]
            return False, f"{len(write)} episode(s) to realign, {len(refused)} refused: {names[:5]}"
        return True, "every served ad-free segment slices to its own text"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Realign every episode the rule allows, with a backup and a receipt per file."""
        root = ctx.corpus_root
        write, refused = _scan(root)
        receipts: List[Dict[str, str]] = []
        if not ctx.dry_run:
            for ep in write:
                assert ep.seg_path is not None
                receipts.append(write_with_backup(root, BACKUP_TAG, ep.seg_path, ep.payload))
            append_receipts(root, RECEIPTS_FILE, {"migration": self.id}, receipts)
        verb = "would realign" if ctx.dry_run else "realigned"
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=f"{verb} {sum(ep.moved for ep in write)} row(s) in {len(write)} episode(s); "
            f"{len(refused)} refused",
            details={
                "episodes": [
                    {"meta": str(ep.meta.relative_to(root)), "rows": ep.moved} for ep in write
                ],
                "refused": [
                    {"meta": str(ep.meta.relative_to(root)), "why": ep.refused} for ep in refused
                ],
                "files_written": len(receipts),
            },
        )


def dry_run_report(root: Path) -> Iterable[Dict[str, Any]]:
    """Read-only: one row per episode written or refused, with its counts and the sha256 of every
    file its plan read (the outcome hook's comparable plan)."""
    from ..read_recorder import recording

    root = Path(root)
    for meta in select_served_artifacts(root, ".metadata.json")[0]:
        ep = _Episode(meta)
        with recording(root) as rec:
            planned = ep.plan()
        if not planned and not ep.refused:
            continue
        yield {
            "meta": str(meta.relative_to(root)),
            "refused": ep.refused,
            "counts": {"rows_realigned": ep.moved, "refused": int(bool(ep.refused))},
            "inputs": rec.inputs(),
        }

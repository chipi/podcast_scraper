"""0040 — a GI quote's offsets point at its own words again, located by its time (text only).

Found by 0041's after-checks; operator decision 4A (2026-11-01). Record:
``docs/wip/changes/0040_gi_quote_offsets_realigned.md``.

24 served GI quotes did not slice to their own text in the transcript they reference: their offsets
point into an older text. Their text is too short to place by itself (0 to 9 occurrences), but each
quote carries its time, and the transcript's segments carry both times and character spans on the
same timeline. So a quote that does not slice takes the one occurrence of its text that starts
inside a segment overlapping its time window (widened by ``WINDOW_S``). No occurrence, or more than
one, leaves the quote as it is, listed: there is no evidence to fix it with. Only ``char_start`` /
``char_end`` change. UNDO restores from ``.podcast_scraper/upgrade-backups/0040/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult

MIGRATION_ID = "0040_gi_quote_offsets_realigned"
RECEIPTS_FILE = "gi_quote_offsets_realigned.jsonl"
BACKUP_TAG = "0040"
#: How far outside a quote's own time window a segment may start or end and still count.
WINDOW_S = 1.0


def _load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _rows(payload: Any) -> List[Any]:
    rows = payload.get("segments") if isinstance(payload, dict) else payload
    return rows if isinstance(rows, list) else []


def placed(text: str, rows: List[Any], words: str, t0_ms: Any, t1_ms: Any) -> List[int]:
    """Every start of ``words`` in ``text`` inside a segment whose time overlaps ``[t0, t1]``."""
    if not words or not isinstance(t0_ms, (int, float)) or not isinstance(t1_ms, (int, float)):
        return []
    t0, t1 = float(t0_ms) / 1000.0 - WINDOW_S, float(t1_ms) / 1000.0 + WINDOW_S
    out = set()
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get("char_start"), int):
            continue
        try:
            start, end = float(r["start"]), float(r["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end < t0 or start > t1:
            continue
        cs, ce = r["char_start"], r["char_end"]
        i = text.find(words, cs)
        while 0 <= i < ce:
            out.add(i)
            i = text.find(words, i + 1)
    return sorted(out)


class _Episode:
    """One episode's GI quotes, realigned in memory."""

    def __init__(self, meta: Path) -> None:
        self.meta = meta
        self.gi_path = meta.with_name(meta.name[: -len(".metadata.json")] + ".gi.json")
        self.payload: Optional[Any] = None
        self.moved = 0
        self.left: List[Dict[str, Any]] = []
        self.refused: Optional[str] = None

    def plan(self) -> bool:
        doc = _load(self.meta)
        rel = str(((doc or {}).get("content") or {}).get("transcript_file_path") or "")
        gi = _load(self.gi_path)
        if not rel or Path(rel).is_absolute() or not isinstance(gi, dict):
            return False
        tdir = self.meta.parent.parent / Path(rel).parent
        texts: Dict[str, Optional[Tuple[str, List[Any]]]] = {}
        out = json.loads(json.dumps(gi))
        for node in out.get("nodes") or []:
            q = node.get("properties") if isinstance(node, dict) else None
            if node.get("type") != "Quote" or not isinstance(q, dict):
                continue
            cs, ce, words = q.get("char_start"), q.get("char_end"), q.get("text")
            ref = Path(str(q.get("transcript_ref") or "")).name
            if not isinstance(cs, int) or not isinstance(ce, int) or not ref:
                continue
            if ref not in texts:
                texts[ref] = self._transcript(tdir, ref)
            got = texts[ref]
            if got is None:
                continue
            text, rows = got
            if text[cs:ce] == words:
                continue
            hits = placed(
                text, rows, str(words or ""), q.get("timestamp_start_ms"), q.get("timestamp_end_ms")
            )
            if len(hits) != 1:
                self.left.append({"quote": node.get("id"), "found": len(hits)})
                continue
            q["char_start"], q["char_end"] = hits[0], hits[0] + len(str(words))
            self.moved += 1
        if self.moved:
            self.payload = out
        return self.moved > 0

    def _transcript(self, tdir: Path, ref: str) -> Optional[Tuple[str, List[Any]]]:
        path = tdir / ref
        if not path.is_file() or not ref.endswith(".txt"):
            return None
        text = path.read_text(encoding="utf-8")
        rows = _rows(_load(tdir / (ref[: -len(".txt")] + ".segments.json")))
        spans = [r for r in rows if isinstance(r, dict) and isinstance(r.get("char_start"), int)]
        if any(text[r["char_start"] : r["char_end"]] != r.get("text") for r in spans):
            self.refused = f"{ref}: its segments do not slice to their text"
            return None
        return text, spans


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


def _scan(root: Path) -> Tuple[List[_Episode], List[_Episode]]:
    write: List[_Episode] = []
    seen: List[_Episode] = []
    for meta in select_served_artifacts(root, ".metadata.json")[0]:
        ep = _Episode(meta)
        if ep.plan():
            write.append(ep)
        if ep.left or ep.refused:
            seen.append(ep)
    return write, seen


class GiQuoteOffsetsRealignedMigration(Migration):
    """GI quotes that do not slice to their text, realigned by their time."""

    id = MIGRATION_ID
    to_version = "2.7.34"
    description = (
        "GI quotes whose offsets no longer slice to their text realigned by their own time window "
        "against the transcript's segments; a quote time cannot place is left and listed"
    )

    def plan(self, ctx: MigrationContext) -> str:
        """Count the quotes apply() would realign and the ones it leaves; pure read."""
        write, seen = _scan(ctx.corpus_root)
        left = sum(len(ep.left) for ep in seen)
        return f"GI quote offsets plan: {sum(ep.moved for ep in write)} realigned, {left} left"

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Pass when no quote that time can place is left misaligned and no episode is refused."""
        write, seen = _scan(ctx.corpus_root)
        refused = [ep for ep in seen if ep.refused]
        if write or refused:
            return False, f"{len(write)} episode(s) to realign, {len(refused)} refused"
        left = sum(len(ep.left) for ep in seen)
        return True, f"every GI quote time can place slices to its text ({left} left, listed)"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Realign every quote time places, with a backup and a receipt per GI file."""
        root = ctx.corpus_root
        write, seen = _scan(root)
        receipts: List[Dict[str, str]] = []
        if not ctx.dry_run:
            for ep in write:
                receipts.append(write_with_backup(root, BACKUP_TAG, ep.gi_path, ep.payload))
            append_receipts(root, RECEIPTS_FILE, {"migration": self.id}, receipts)
        verb = "would realign" if ctx.dry_run else "realigned"
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=f"{verb} {sum(ep.moved for ep in write)} quote(s) in {len(write)} episode(s); "
            f"{sum(len(ep.left) for ep in seen)} left",
            details={
                "episodes": [
                    {"meta": str(ep.meta.relative_to(root)), "quotes": ep.moved} for ep in write
                ],
                "left": [
                    {"meta": str(ep.meta.relative_to(root)), "quotes": ep.left, "why": ep.refused}
                    for ep in seen
                ],
                "files_written": len(receipts),
            },
        )


def dry_run_report(root: Path) -> Iterable[Dict[str, Any]]:
    """Read-only: one row per episode written or left, its counts and every file its plan read."""
    from ..read_recorder import recording

    root = Path(root)
    for meta in select_served_artifacts(root, ".metadata.json")[0]:
        ep = _Episode(meta)
        with recording(root) as rec:
            planned = ep.plan()
        if not planned and not ep.left and not ep.refused:
            continue
        yield {
            "meta": str(meta.relative_to(root)),
            "refused": ep.refused,
            "counts": {"quotes_realigned": ep.moved, "quotes_left": len(ep.left)},
            "inputs": rec.inputs(),
        }

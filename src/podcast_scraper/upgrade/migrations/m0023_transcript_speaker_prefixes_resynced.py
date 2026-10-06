"""0023 — the text transcripts say who spoke the way the segments now do, offsets and all.

The name repairs (m0012/m0015/m0016/m0018/m0019) removed names from the segment labels and left the
``<name>: `` prefixes of ``.txt`` / ``.adfree.txt`` / ``.cleaned.txt`` alone, so the transcripts
still credited "Andreessen Horowitz", "Machine Learning Street", "Host"… (prod 2026-10-06: 436
served episodes). m0022 fixed the two derived files that hold no text offsets; this one cannot be
a find-and-replace, because two things locate text in these files BY CHARACTER OFFSET:

* GI ``Quote`` nodes — ``char_start`` / ``char_end`` into the file their ``transcript_ref`` names;
* ``.adfree.segments.json`` rows — ``char_start`` / ``char_end`` into ``.adfree.txt``.

A shorter prefix moves every offset after it. So ``.txt`` and ``.adfree.txt`` are re-rendered from
their CURRENT segments with the pipeline's own ``format_diarized_screenplay_with_offsets`` — exactly
what a run writes today, and it splits the line two voices shared under one removed name — and every
offset is carried across by aligning each segment's text in the old file to its place in the new
one. ``.cleaned.txt`` has no offset readers: only its prefixes change, by the old->new label map the
alignment yields, and only where that map is unambiguous.

Refuses an episode rather than guess: the old file must be nothing but the segment texts, speaker
prefixes and separators; every quote must slice to the identical text before and after. A refused
episode is counted and left untouched. Undo restores from ``.podcast_scraper/upgrade-backups/0023/``
while each file is still exactly what this migration wrote.
"""

from __future__ import annotations

import bisect
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ...providers.ml.diarization.formatting import format_diarized_screenplay_with_offsets
from ..corpus_selection import select_served_artifacts
from ..file_rewrite import (
    append_receipts,
    backup_dir,
    file_sha,
    match_corpus_owner,
    undo_from_receipts,
    write_with_backup,
)
from ..migration import Migration, MigrationContext, MigrationResult

MIGRATION_ID = "0023_transcript_speaker_prefixes_resynced"
RECEIPTS_FILE = "transcript_speaker_prefixes_resynced.jsonl"
BACKUP_TAG = "0023"
#: What may sit between two segment texts in a rendered transcript: a space inside a turn, or a
#: newline and the next turn's ``<label>: `` prefix. Anything else means the file is not a render
#: of these segments, and offsets cannot be carried across it.
_GAP = re.compile(r"\A(?: |\n(?P<label>[^\n]{1,200}?): )\Z")
_PREFIX = re.compile(r"(?m)^(?P<label>[^\n:]{1,200}): ")


class Refused(Exception):
    """This episode's transcripts cannot be re-rendered without guessing."""


def _load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _rows(payload: Any) -> List[dict]:
    rows = payload if isinstance(payload, list) else (payload or {}).get("segments")
    return [r for r in rows or [] if isinstance(r, dict)]


def _rendered(rows: List[dict]) -> List[dict]:
    """The rows the formatter renders, in its order (start-time sort, blank text dropped)."""
    ordered = sorted(rows, key=lambda r: float(r.get("start") or 0.0))
    return [r for r in ordered if (r.get("text") or "").strip()]


def _align(old: str, rows: List[dict]) -> Tuple[List[Tuple[int, int]], List[Optional[str]]]:
    """Each rendered row's ``(start, end)`` in *old*, and the label *old* gave its turn.

    Raises :class:`Refused` unless *old* is exactly these texts joined by render separators.
    """
    spans: List[Tuple[int, int]] = []
    labels: List[Optional[str]] = []
    cursor = 0
    label: Optional[str] = None
    for i, row in enumerate(rows):
        text = (row.get("text") or "").strip()
        if i == 0:
            m = re.match(r"(?P<label>[^\n]{1,200}?): ", old)
            if not m:
                raise Refused("no leading speaker prefix")
            label, cursor = m.group("label"), m.end()
            start = cursor
        else:
            start = old.find(text, cursor)
            if start < 0:
                raise Refused("segment text not found")
            gap = _GAP.match(old[cursor:start])
            if not gap:
                raise Refused("unexplained text between segments")
            if gap.group("label") is not None:
                label = gap.group("label")
        if old[start : start + len(text)] != text:
            raise Refused("segment text not at its place")
        spans.append((start, start + len(text)))
        labels.append(label)
        cursor = start + len(text)
    if old[cursor:] not in ("", "\n"):
        raise Refused("unexplained trailing text")
    return spans, labels


class _OffsetMap:
    """Old character offset -> new, through the aligned segment spans."""

    def __init__(self, old: List[Tuple[int, int]], new: List[Tuple[int, int]]) -> None:
        self.old, self.new = old, new
        self.starts = [s for s, _e in old]

    def __call__(self, pos: int, *, is_end: bool) -> int:
        i = bisect.bisect_right(self.starts, pos) - 1
        if i >= 0 and pos <= self.old[i][1]:
            return self.new[i][0] + (pos - self.old[i][0])
        # Between segments (a separator): a start moves to the next text, an end to the last.
        if is_end:
            return self.new[i][1] if i >= 0 else 0
        return self.new[i + 1][0] if i + 1 < len(self.new) else self.new[-1][1]


class _Episode:
    """One episode's transcripts, GI quotes and ad-free segment offsets, rewritten in memory."""

    def __init__(self, meta: Path) -> None:
        self.meta = meta
        self.json_files: Dict[Path, Any] = {}
        self.text_files: Dict[Path, str] = {}
        self.counts: Dict[str, int] = {}

    def _bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def plan(self) -> bool:
        meta = _load(self.meta)
        rel = str(((meta or {}).get("content") or {}).get("transcript_file_path") or "")
        if not rel.endswith(".txt"):
            return False
        try:
            self._plan(self.meta.parent.parent, rel)
        except Refused as exc:
            self.json_files, self.text_files = {}, {}
            self.counts = {"refused": 1, f"refused: {exc}": 1}
            return False
        return bool(self.json_files or self.text_files)

    def _plan(self, run: Path, rel: str) -> None:
        gi_path = Path(str(self.meta)[: -len(".metadata.json")] + ".gi.json")
        gi = _load(gi_path)
        quotes = [
            n
            for n in ((gi or {}).get("nodes") or [])
            if isinstance(n, dict) and n.get("type") == "Quote"
        ]
        renames: Dict[str, Set[str]] = {}
        quotes_moved = 0
        for suffix, seg_suffix in (
            (".txt", ".segments.json"),
            (".adfree.txt", ".adfree.segments.json"),
        ):
            txt_path = run / rel.replace(".txt", suffix)
            seg_path = run / rel.replace(".txt", seg_suffix)
            seg_payload = _load(seg_path)
            if not txt_path.is_file() or seg_payload is None:
                continue
            old = txt_path.read_text(encoding="utf-8")
            rows = _rendered(_rows(seg_payload))
            if not rows:
                continue
            new, emitted = format_diarized_screenplay_with_offsets(rows)
            old_spans, old_labels = _align(old, rows)
            new_spans = [(int(e["char_start"]), int(e["char_end"])) for e in emitted]
            if len(new_spans) != len(old_spans):
                raise Refused("render dropped a row")
            for row, old_label, e in zip(rows, old_labels, emitted):
                if old_label is not None and old_label != e["speaker_label"]:
                    renames.setdefault(old_label, set()).add(e["speaker_label"])
            if new == old:
                continue
            fmap = _OffsetMap(old_spans, new_spans)
            for q in quotes:
                props = q.get("properties") or {}
                # By FILE NAME, not path: refs are run-relative on some episodes and absolute
                # (`/app/output/feeds/…`) on others. Compared as run-relative strings, every
                # absolute-ref quote went unmoved and unchecked — 2,038 broken on prod (reverted).
                ref = props.get("transcript_ref")
                if not isinstance(ref, str) or Path(ref).name != txt_path.name:
                    continue
                cs, ce = props.get("char_start"), props.get("char_end")
                if not isinstance(cs, int) or not isinstance(ce, int):
                    continue
                ncs, nce = fmap(cs, is_end=False), fmap(ce, is_end=True)
                if new[ncs:nce] != old[cs:ce]:
                    raise Refused("a quote would not slice to its text")
                if (ncs, nce) != (cs, ce):
                    props["char_start"], props["char_end"] = ncs, nce
                    quotes_moved += 1
            if seg_suffix == ".adfree.segments.json":
                for row, (span_start, span_end) in zip(rows, new_spans):
                    if "char_start" in row or "char_end" in row:
                        row["char_start"], row["char_end"] = span_start, span_end
                self.json_files[seg_path] = seg_payload
            self.text_files[txt_path] = new
            self._bump("transcripts_rerendered")
        if quotes_moved:
            self.json_files[gi_path] = gi
            self._bump("quotes_moved", quotes_moved)
        self._cleaned(run, rel, renames)

    def _cleaned(self, run: Path, rel: str, renames: Dict[str, Set[str]]) -> None:
        path = run / rel.replace(".txt", ".cleaned.txt")
        safe = {old: next(iter(new)) for old, new in renames.items() if len(new) == 1}
        if not renames or not path.is_file():
            return
        text = path.read_text(encoding="utf-8")
        stale = {m.group("label") for m in _PREFIX.finditer(text)} & set(renames)
        if stale - set(safe):
            self._bump("cleaned_ambiguous")
            return
        if not stale:
            return
        rx = re.compile(
            r"(?m)^("
            + "|".join(re.escape(n) for n in sorted(stale, key=len, reverse=True))
            + r"): "
        )
        new, n = rx.subn(lambda m: safe[m.group(1)] + ": ", text)
        if n:
            self.text_files[path] = new
            self._bump("cleaned_lines", n)


def _write_text_with_backup(root: Path, path: Path, text: str) -> Dict[str, str]:
    """``write_with_backup`` for a plain-text file (that one serialises JSON)."""
    rel = str(path.relative_to(root))
    backup = backup_dir(root, BACKUP_TAG) / rel
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    sha_before = file_sha(path)
    mode = os.stat(path).st_mode & 0o777
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.chmod(tmp, mode)
    os.replace(tmp, path)
    match_corpus_owner(root, [path, backup])
    return {"relpath": rel, "sha_before": sha_before, "sha_after": file_sha(path)}


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class TranscriptSpeakerPrefixesResyncedMigration(Migration):
    """Re-render the text transcripts from the repaired segments, carrying every offset across."""

    id = MIGRATION_ID
    to_version = "2.7.17"
    description = (
        "re-render .txt/.adfree.txt from the repaired segments (pipeline formatter) and shift GI "
        "quote and ad-free segment offsets with them; rename stale .cleaned.txt prefixes"
    )

    def _scan(self, root: Path) -> Tuple[List[_Episode], Dict[str, int]]:
        eps: List[_Episode] = []
        totals: Dict[str, int] = {}
        for meta in select_served_artifacts(root, ".metadata.json")[0]:
            ep = _Episode(meta)
            changed = ep.plan()
            for k, v in ep.counts.items():
                totals[k] = totals.get(k, 0) + v
            if changed:
                eps.append(ep)
        return eps, totals

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would rewrite — pure read, no writes."""
        eps, totals = self._scan(ctx.corpus_root)
        return f"transcript prefixes plan: {len(eps)} episode(s); " + ", ".join(
            f"{k}={v}" for k, v in sorted(totals.items())
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """No served transcript still differs from its segments' render. ``(ok, message)``."""
        eps, _totals = self._scan(ctx.corpus_root)
        left = [ep.meta.name for ep in eps if ep.counts.get("transcripts_rerendered")]
        if left:
            return False, f"{len(left)} episode(s) still render differently: {left[:5]}"
        return True, "every served transcript is its segments' render"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Write every re-rendered transcript and moved offset; back up each file, receipt it."""
        root = ctx.corpus_root
        eps, totals = self._scan(root)
        receipts: List[dict] = []
        if not ctx.dry_run:
            for ep in eps:
                for path, payload in ep.json_files.items():
                    receipts.append(write_with_backup(root, BACKUP_TAG, path, payload))
                for path, text in ep.text_files.items():
                    receipts.append(_write_text_with_backup(root, path, text))
            append_receipts(root, RECEIPTS_FILE, {"migration": MIGRATION_ID}, receipts)
        verb = "would rewrite" if ctx.dry_run else "rewrote"
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=f"{verb} {len(eps)} episode(s): "
            + ", ".join(f"{k}={v}" for k, v in sorted(totals.items())),
            details={
                "episodes": [str(ep.meta.relative_to(root)) for ep in eps[:50]],
                "totals": totals,
                "files_written": len(receipts),
            },
        )

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


#: A speaker prefix at the start of the text or of a line — what a quote must not carry as speech.
_LABEL_IN_TEXT = re.compile(r"(?:\A|\n)[^\n:]{1,200}: ")


def _speech(text: str) -> str:
    """The spoken words only: speaker prefixes out, whitespace normalised."""
    return " ".join(_LABEL_IN_TEXT.sub(" ", text).split())


def _move_quote(
    props: Dict[str, Any], old: str, new: str, new_start: int, new_end: int
) -> Tuple[bool, bool]:
    """Point *props* at the same speech in *new*. ``(moved, retexted)``; raises :class:`Refused`.

    Exact when the moved span slices to the same characters. Otherwise the speech must be the same
    words and only labels/separators differ — the quote ended on the space that joined two voices'
    turns on one shared line (now a newline and a prefix), or STARTED with the removed name's
    prefix ("Americas Online: just imagine…", the name published inside the quote). The span is
    then settled on the speech itself — leading prefix skipped, edge whitespace trimmed — and the
    quote's ``text`` becomes exactly that slice.
    """
    cs, ce = int(props["char_start"]), int(props["char_end"])
    if new[new_start:new_end] == old[cs:ce]:
        moved = (new_start, new_end) != (cs, ce)
        if moved:
            props["char_start"], props["char_end"] = new_start, new_end
        return moved, False
    if props.get("text") != old[cs:ce]:
        raise Refused("a quote's text is not its slice")
    lead = (
        _LABEL_IN_TEXT.match(new, new_start)
        if (new_start == 0 or new[new_start - 1] == "\n")
        else None
    )
    if lead is not None and lead.start() == new_start:
        new_start = lead.end()
    while new_start < new_end and new[new_start].isspace():
        new_start += 1
    while new_end > new_start and new[new_end - 1].isspace():
        new_end -= 1
    if not _speech(new[new_start:new_end]) or _speech(new[new_start:new_end]) != _speech(
        old[cs:ce]
    ):
        raise Refused("a quote would not slice to its text")
    # Never GAIN a speaker label inside a quote: a quote over two voices that shared one line would
    # read "…gateway.\nSPEAKER_02: Welcome…" — worse than what it showed. Swapping a label it
    # already carried (its span already crossed a line) is no worse than before.
    if new[new_start:new_end].count("\n") > old[cs:ce].count("\n"):
        raise Refused("a quote would gain a speaker label")
    props["char_start"], props["char_end"], props["text"] = (
        new_start,
        new_end,
        new[new_start:new_end],
    )
    return True, True


class _OffsetMap:
    """Old character offset -> new, through the aligned segment spans."""

    def __init__(self, old: List[Tuple[int, int]], new: List[Tuple[int, int]]) -> None:
        self.old, self.new = old, new
        self.starts = [s for s, _e in old]

    def __call__(self, pos: int, *, is_end: bool) -> int:
        # An END at exactly a segment's start belongs to the segment BEFORE it (the span stops
        # there); mapped into the next one it would jump past the new line's speaker prefix.
        i = (bisect.bisect_left if is_end else bisect.bisect_right)(self.starts, pos) - 1
        if i >= 0 and pos <= self.old[i][1]:
            return self.new[i][0] + (pos - self.old[i][0])
        # Between segments (a separator): a start moves to the next text, an end to the last.
        if is_end:
            return self.new[i][1] if i >= 0 else 0
        return self.new[i + 1][0] if i + 1 < len(self.new) else self.new[-1][1]


class _Episode:
    """One episode's transcripts, GI quotes and ad-free segment offsets, rewritten in memory."""

    def __init__(self, meta: Path, root: Optional[Path] = None) -> None:
        self.meta = meta
        self.root = root
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
        run = self.meta.parent.parent
        try:
            self._plan(run, rel)
        except Refused as exc:
            self.json_files, self.text_files = {}, {}
            self.counts = {"refused": 1, f"refused: {exc}": 1}
            # Not a render of its segments, so it cannot be re-rendered — but a removed name that
            # belonged to ONE voice can still be renamed where it stands, every offset shifted by
            # the exact length change. Same guard: every quote/segment slices to the same text.
            try:
                self._in_place(run, rel)
            except Refused as exc2:
                self.json_files, self.text_files = {}, {}
                self._bump(f"in_place refused: {exc2}")
        return bool(self.json_files or self.text_files)

    def _old_names(self, run: Path, rel: str) -> Dict[str, str]:
        """``{removed name: what its ONE voice is now called}``, from the diagnostics as the roster
        wrote them (m0022's backup when m0022 has since rewritten them)."""
        dpath = run / rel.replace(".txt", ".speakers.diagnostics.json")
        diag = _load(dpath)
        if self.root is not None:
            try:
                rel_d = Path(os.path.relpath(dpath, self.root))
                backup = _load(backup_dir(self.root, "0022") / rel_d)
                diag = backup if isinstance(backup, dict) else diag
            except ValueError:
                pass
        labels: Dict[str, Set[str]] = {}
        for row in _rows(_load(run / rel.replace(".txt", ".segments.json"))):
            voice = row.get("speaker")
            if isinstance(voice, str) and voice:
                labels.setdefault(voice, set()).add(str(row.get("speaker_label") or voice))
        current = {lab for labs in labels.values() for lab in labs}
        voices_by_name: Dict[str, Set[str]] = {}
        for v in (diag or {}).get("voices") or []:
            if isinstance(v, dict) and v.get("named") and v.get("resolved_name") and v.get("voice"):
                voices_by_name.setdefault(str(v["resolved_name"]), set()).add(str(v["voice"]))
        out: Dict[str, str] = {}
        for name, voices in voices_by_name.items():
            if name in current or len(voices) != 1:
                continue
            labs = labels.get(next(iter(voices)), set())
            if len(labs) == 1:
                out[name] = next(iter(labs))
        return out

    def _in_place(self, run: Path, rel: str, gi: Optional[Any] = None) -> None:
        renames = self._old_names(run, rel)
        if not renames:
            return
        rx = re.compile(
            r"(?m)^("
            + "|".join(re.escape(n) for n in sorted(renames, key=len, reverse=True))
            + r"): "
        )
        gi_path = Path(str(self.meta)[: -len(".metadata.json")] + ".gi.json")
        gi = gi if gi is not None else _load(gi_path)
        quotes_moved = 0
        for suffix in (".txt", ".adfree.txt", ".cleaned.txt"):
            path = run / rel.replace(".txt", suffix)
            if not path.is_file():
                continue
            old = path.read_text(encoding="utf-8")
            shifts: List[Tuple[int, int]] = []
            parts: List[str] = []
            last = 0
            for m in rx.finditer(old):
                parts += [old[last : m.start()], renames[m.group(1)] + ": "]
                shifts.append((m.start(), len(renames[m.group(1)]) - len(m.group(1))))
                last = m.end()
            if not shifts:
                continue
            new = "".join(parts) + old[last:]
            starts = [q for q, _d in shifts]
            cum = [0]
            for _q, d in shifts:
                cum.append(cum[-1] + d)

            def moved(pos: int) -> int:
                return pos + cum[bisect.bisect_right(starts, pos - 1)]

            for q in (gi or {}).get("nodes") or []:
                props = (q.get("properties") or {}) if isinstance(q, dict) else {}
                ref = props.get("transcript_ref")
                if q.get("type") != "Quote" or not isinstance(ref, str):
                    continue
                if Path(ref).name != path.name:
                    continue
                cs, ce = props.get("char_start"), props.get("char_end")
                if not isinstance(cs, int) or not isinstance(ce, int):
                    continue
                did_move, retexted = _move_quote(props, old, new, moved(cs), moved(ce))
                quotes_moved += did_move
                if retexted:
                    self._bump("quotes_retexted")
            if suffix == ".adfree.txt":
                seg_path = run / rel.replace(".txt", ".adfree.segments.json")
                seg_payload = _load(seg_path)
                touched = False
                for row in _rows(seg_payload):
                    cs, ce = row.get("char_start"), row.get("char_end")
                    if not isinstance(cs, int) or not isinstance(ce, int):
                        continue
                    new_start, new_end = moved(cs), moved(ce)
                    if new[new_start:new_end] != old[cs:ce]:
                        raise Refused("a segment would not slice to its text")
                    if (new_start, new_end) != (cs, ce):
                        row["char_start"], row["char_end"] = new_start, new_end
                        touched = True
                if touched:
                    self.json_files[seg_path] = seg_payload
            self.text_files[path] = new
            self._bump("in_place_lines", len(shifts))
        if quotes_moved:
            self.json_files[gi_path] = gi
            self._bump("quotes_moved", quotes_moved)
        if self.text_files:
            self._bump("in_place_episodes")

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
                did_move, retexted = _move_quote(
                    props, old, new, fmap(cs, is_end=False), fmap(ce, is_end=True)
                )
                quotes_moved += did_move
                if retexted:
                    self._bump("quotes_retexted")
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


def _write_text_with_backup(
    root: Path, path: Path, text: str, tag: str = BACKUP_TAG
) -> Dict[str, str]:
    """``write_with_backup`` for a plain-text file (that one serialises JSON)."""
    rel = str(path.relative_to(root))
    backup = backup_dir(root, tag) / rel
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
            ep = _Episode(meta, root)
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

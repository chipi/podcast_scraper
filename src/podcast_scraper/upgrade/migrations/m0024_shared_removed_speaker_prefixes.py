"""0024 — no transcript still names a removed speaker, including the ones 0023 refused (#2294).

0023 renamed a removed name's ``<name>: `` prefixes to the ONE voice it had belonged to, and refused
every episode where that could not be done exactly. Prod 2026-10-07: 17 episodes, three names —
"Machine Learning Street" (the show), "Norman Conquest" (a topic), "Andreessen Horowitz" (the
firm) — almost always on BOTH host voices. There the rendered line holds both hosts' turns,
``.cleaned.txt`` holds no voice ids at all, and splitting the line would put a speaker label inside
a GI quote. No line can be credited to either host without guessing.

So a removed name that sat on two or more voices becomes ``SPEAKER`` — the pipeline formatter's own
label for a turn it cannot attribute — in ``.txt``, ``.adfree.txt`` and ``.cleaned.txt``, in place.
Nothing is split, so no quote gains a label; a name on one voice still becomes that voice's label.
Every offset after a rename shifts by the exact length change, through 0023's guarded in-place path:
each GI quote and ad-free segment must slice to the same speech, or the episode is left untouched.

One more prod shape: a quote extracted up to a removed name's prefix and cut inside it — text
ending ``"…get this deployed\\nAnd"`` where the file reads ``\\nAndreessen Horowitz: ``. The cut-off
label is trimmed from the quote first, so its ``text`` is the speech alone.

``verify`` counts every served episode still carrying a removed name's prefix — including any this
migration refuses — so a refusal can no longer read as done (0023's verify only saw what it wrote).
Undo restores from ``.podcast_scraper/upgrade-backups/0024/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, backup_dir, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult
from .m0023_transcript_speaker_prefixes_resynced import (
    _Episode as _ResyncEpisode,
    _load,
    _PREFIX,
    _rows,
    _write_text_with_backup,
    Refused,
)

MIGRATION_ID = "0024_shared_removed_speaker_prefixes"
RECEIPTS_FILE = "shared_removed_speaker_prefixes.jsonl"
BACKUP_TAG = "0024"
#: What the pipeline's formatter writes for a turn with no label (``formatting.py``).
UNATTRIBUTED = "SPEAKER"
_TEXT_SUFFIXES = (".txt", ".adfree.txt", ".cleaned.txt")


class _Episode(_ResyncEpisode):
    """0023's in-place rename, extended to names more than one voice carried."""

    def _old_names(self, run: Path, rel: str) -> Dict[str, str]:
        """``{removed name: its replacement}``: the voice's label for one voice, else SPEAKER."""
        out = dict(super()._old_names(run, rel))
        for name, voices in self._removed_voices(run, rel).items():
            if name not in out and len(voices) > 1:
                out[name] = UNATTRIBUTED
        return out

    def _removed_voices(self, run: Path, rel: str) -> Dict[str, Set[str]]:
        """``{removed name: the voices the roster had given it}`` — names no segment carries now."""
        dpath = run / rel.replace(".txt", ".speakers.diagnostics.json")
        diag = _load(dpath)
        if self.root is not None:
            try:
                backup = _load(backup_dir(self.root, "0022") / dpath.relative_to(self.root))
                diag = backup if isinstance(backup, dict) else diag
            except ValueError:
                pass
        current: Set[str] = set()
        for row in _rows(_load(run / rel.replace(".txt", ".segments.json"))):
            for key in ("speaker_label", "speaker"):
                if row.get(key):
                    current.add(str(row[key]))
        out: Dict[str, Set[str]] = {}
        for v in (diag or {}).get("voices") or []:
            if isinstance(v, dict) and v.get("named") and v.get("resolved_name") and v.get("voice"):
                name = str(v["resolved_name"])
                if name not in current and name != UNATTRIBUTED:
                    out.setdefault(name, set()).add(str(v["voice"]))
        return out

    def stale_names(self) -> Set[str]:
        """Removed names still standing as a line prefix in any of this episode's transcripts."""
        meta = _load(self.meta)
        rel = str(((meta or {}).get("content") or {}).get("transcript_file_path") or "")
        if not rel.endswith(".txt"):
            return set()
        run = self.meta.parent.parent
        removed = set(self._removed_voices(run, rel))
        found: Set[str] = set()
        for suffix in _TEXT_SUFFIXES:
            path = run / rel.replace(".txt", suffix)
            if removed and path.is_file():
                text = path.read_text(encoding="utf-8")
                found |= {m.group("label") for m in _PREFIX.finditer(text)} & removed
        return found

    def plan(self) -> bool:
        meta = _load(self.meta)
        rel = str(((meta or {}).get("content") or {}).get("transcript_file_path") or "")
        if not rel.endswith(".txt") or not self.stale_names():
            return False
        run = self.meta.parent.parent
        gi_path = Path(str(self.meta)[: -len(".metadata.json")] + ".gi.json")
        gi = _load(gi_path)
        try:
            trimmed = self._trim_cut_labels(run, rel, gi)
            self._in_place(run, rel, gi)
        except Refused as exc:
            self.json_files, self.text_files = {}, {}
            self.counts = {"refused": 1, f"refused: {exc}": 1}
            return False
        if trimmed:
            self.json_files[gi_path] = gi
            self._bump("quotes_trimmed", trimmed)
        return bool(self.json_files or self.text_files)

    def _trim_cut_labels(self, run: Path, rel: str, gi: Optional[Any]) -> int:
        """Drop a removed name's cut-off prefix from the END of each quote; how many were trimmed.

        Only when the quote's slice ends ``\\n<p>``, ``<p>`` a proper start of a removed name, AND
        the file continues into that full ``<name>: `` there — the quote stopped inside the label.
        """
        names = sorted(self._removed_voices(run, rel), key=len, reverse=True)
        if not names:
            return 0
        trimmed = 0
        for node in (gi or {}).get("nodes") or []:
            props = (node.get("properties") or {}) if isinstance(node, dict) else {}
            ref = props.get("transcript_ref")
            cs, ce = props.get("char_start"), props.get("char_end")
            if node.get("type") != "Quote" or not isinstance(ref, str):
                continue
            if not isinstance(cs, int) or not isinstance(ce, int):
                continue
            # By FILE NAME: refs are run-relative on some episodes and absolute on others (0023).
            path = (run / rel).parent / Path(ref).name
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            cut = self._cut_label_at(text, cs, ce, names)
            if cut is None:
                continue
            if props.get("text") != text[cs:ce]:
                raise Refused("a quote's text is not its slice")
            end = cut
            while end > cs and text[end - 1].isspace():
                end -= 1
            if end <= cs:
                raise Refused("a quote is nothing but a cut-off label")
            props["char_end"], props["text"] = end, text[cs:end]
            trimmed += 1
        return trimmed

    @staticmethod
    def _cut_label_at(text: str, cs: int, ce: int, names: List[str]) -> Optional[int]:
        """Where the quote's trailing cut-off label starts (the newline), or ``None``."""
        line_start = text.rfind("\n", cs, ce)
        if line_start < 0:
            return None
        tail = text[line_start + 1 : ce]
        if not tail or ":" in tail:
            return None
        for name in names:
            if len(tail) < len(name) and name.startswith(tail):
                full = text[line_start + 1 : line_start + 1 + len(name) + 2]
                if full == name + ": ":
                    return line_start
        return None


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class SharedRemovedSpeakerPrefixesMigration(Migration):
    """Rename every removed name's transcript prefix; SPEAKER where more than one voice had it."""

    id = MIGRATION_ID
    to_version = "2.7.18"
    description = (
        "rename removed speaker names left in .txt/.adfree.txt/.cleaned.txt prefixes — SPEAKER "
        "where the name was on 2+ voices — shifting GI quote and ad-free segment offsets"
    )

    def _scan(self, root: Path) -> Tuple[List[_Episode], List[_Episode], Dict[str, int]]:
        """``(episodes to write, episodes still stale, totals)``."""
        to_write: List[_Episode] = []
        stale: List[_Episode] = []
        totals: Dict[str, int] = {}
        for meta in select_served_artifacts(root, ".metadata.json")[0]:
            ep = _Episode(meta, root)
            if not ep.stale_names():
                continue
            stale.append(ep)
            if ep.plan():
                to_write.append(ep)
            for k, v in ep.counts.items():
                totals[k] = totals.get(k, 0) + v
        return to_write, stale, totals

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would rewrite — pure read, no writes."""
        eps, stale, totals = self._scan(ctx.corpus_root)
        return (
            f"removed speaker prefixes plan: {len(stale)} stale episode(s), "
            f"{len(eps)} writable; " + ", ".join(f"{k}={v}" for k, v in sorted(totals.items()))
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """No served transcript names a removed speaker — refused episodes count. ``(ok, msg)``."""
        _eps, stale, _totals = self._scan(ctx.corpus_root)
        if stale:
            names = sorted({n for ep in stale for n in ep.stale_names()})
            return False, f"{len(stale)} episode(s) still name a removed speaker: {names[:5]}"
        return True, "no served transcript names a removed speaker"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Write every renamed transcript and moved offset; back up each file, receipt it."""
        root = ctx.corpus_root
        eps, stale, totals = self._scan(root)
        receipts: List[dict] = []
        if not ctx.dry_run:
            for ep in eps:
                for path, payload in ep.json_files.items():
                    receipts.append(write_with_backup(root, BACKUP_TAG, path, payload))
                for path, text in ep.text_files.items():
                    receipts.append(_write_text_with_backup(root, path, text, BACKUP_TAG))
            append_receipts(root, RECEIPTS_FILE, {"migration": MIGRATION_ID}, receipts)
        verb = "would rewrite" if ctx.dry_run else "rewrote"
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=f"{verb} {len(eps)} of {len(stale)} stale episode(s): "
            + ", ".join(f"{k}={v}" for k, v in sorted(totals.items())),
            details={
                "episodes": [str(ep.meta.relative_to(root)) for ep in eps[:50]],
                "refused": [str(ep.meta.relative_to(root)) for ep in stale if ep not in eps],
                "totals": totals,
                "files_written": len(receipts),
            },
        )

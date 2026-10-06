"""0022 — context.json and the speaker diagnostics follow the surfaces the name repairs rewrote.

m0012, m0015, m0016, m0018 and m0019 removed names from metadata, segments, KG, GI and bridge, and
left two files derived from them alone, so every removed name was still published there. Measured on
prod 2026-10-06: 436 served episodes ("Andreessen Horowitz", "Machine Learning Street", "World
Bank", "Host", "Norman Conquest", …).

* ``.context.json`` — ``basic.hosts`` / ``basic.guests`` / ``people`` are a digest of metadata, GI
  and KG (``build_context_digest``). Those three fields are rebuilt; nothing else in it is touched.
* ``.speakers.diagnostics.json`` — a voice the roster once named keeps that name here after its
  segment label is gone. Its ``resolved_name`` / ``named`` / ``source`` now follow the label (the
  raw ``SPEAKER_NN``, unnamed, when there is none), and the summary's named counts with it.

The segment labels are the truth this reads: they are what every repair rewrote and what every
re-derive reads back. A voice whose segments carry more than one label is left and counted.

NOT the text transcripts. Their ``<name>: `` prefixes are stale too, but GI quotes and the ad-free
segments locate text in them by CHARACTER OFFSET, and a prefix of a different length moves every
offset after it. Rewriting them is a separate migration that shifts those offsets with it.

Undo restores each file from ``.podcast_scraper/upgrade-backups/0022/`` while it is still exactly
what this migration wrote.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ...builders.context_digest_builder import build_context_digest
from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult

MIGRATION_ID = "0022_derived_speaker_surfaces_resynced"
RECEIPTS_FILE = "derived_speaker_surfaces_resynced.jsonl"
BACKUP_TAG = "0022"
#: (parent key or None for top level, field) — the speaker fields of the context digest.
_CONTEXT_FIELDS = (("basic", "hosts"), ("basic", "guests"), (None, "people"))


def _load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _labels_by_voice(payload: Any) -> Dict[str, Set[Optional[str]]]:
    rows = payload if isinstance(payload, list) else (payload or {}).get("segments")
    out: Dict[str, Set[Optional[str]]] = {}
    for row in rows or []:
        voice = row.get("speaker") if isinstance(row, dict) else None
        if isinstance(voice, str) and voice:
            out.setdefault(voice, set()).add(row.get("speaker_label") or None)
    return out


class _Episode:
    """One episode's derived speaker surfaces, rewritten in memory; nothing touches disk here."""

    def __init__(self, meta: Path) -> None:
        self.meta = meta
        self.files: Dict[Path, Any] = {}
        self.counts: Dict[str, int] = {}

    def _bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def plan(self) -> bool:
        meta = _load(self.meta)
        if not isinstance(meta, dict):
            return False
        self._context(meta)
        rel = str((meta.get("content") or {}).get("transcript_file_path") or "")
        if rel.endswith(".txt"):
            self._diagnostics(self.meta.parent.parent, rel)
        return bool(self.files)

    def _context(self, meta: dict) -> None:
        base = str(self.meta)[: -len(".metadata.json")]
        path = Path(base + ".context.json")
        ctx = _load(path)
        if not isinstance(ctx, dict):
            return
        rebuilt = build_context_digest(
            str(ctx.get("episode_id") or ""),
            gi_artifact=_load(Path(base + ".gi.json")),
            kg_artifact=_load(Path(base + ".kg.json")),
            metadata=meta,
        )
        hit = False
        for parent, key in _CONTEXT_FIELDS:
            old_holder = ctx.get(parent) if parent else ctx
            new_holder = rebuilt.get(parent) if parent else rebuilt
            if not isinstance(old_holder, dict) or not isinstance(new_holder, dict):
                continue
            if key in new_holder and old_holder.get(key) != new_holder[key]:
                old_holder[key] = new_holder[key]
                hit = True
        if hit:
            self.files[path] = ctx
            self._bump("context_rebuilt")

    def _diagnostics(self, run: Path, rel: str) -> None:
        labels = _labels_by_voice(_load(run / rel.replace(".txt", ".segments.json")))
        path = run / rel.replace(".txt", ".speakers.diagnostics.json")
        diag = _load(path)
        if not labels or not isinstance(diag, dict):
            return
        unnamed = renamed = 0
        for v in diag.get("voices") or []:
            if not isinstance(v, dict) or not v.get("named"):
                continue
            voice, name = v.get("voice"), v.get("resolved_name")
            if voice not in labels or not name:
                continue
            if len(labels[voice]) != 1:
                self._bump("ambiguous_voice_labels")
                continue
            now = next(iter(labels[voice]))
            if now == name:
                continue
            if now is None:
                v.update(resolved_name=voice, named=False, source="raw")
                if v.get("voice_type") == "person":
                    v["voice_type"] = "unknown"
                unnamed += 1
            else:
                v["resolved_name"] = now
                renamed += 1
        if not (unnamed or renamed):
            return
        summary = diag.get("summary")
        if unnamed and isinstance(summary, dict):
            for holder in (summary, summary.get("exposed")):
                if isinstance(holder, dict) and isinstance(holder.get("named"), int):
                    holder["named"] = max(0, holder["named"] - unnamed)
            if isinstance(summary.get("unresolved"), int):
                summary["unresolved"] += unnamed
        self.files[path] = diag
        self._bump("diagnostics_voices_unnamed", unnamed)
        self._bump("diagnostics_voices_renamed", renamed)


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class DerivedSpeakerSurfacesResyncedMigration(Migration):
    """Bring context.json's speaker fields and the speaker diagnostics in line with the segments."""

    id = MIGRATION_ID
    to_version = "2.7.16"
    description = (
        "the name repairs rewrote metadata/segments/KG/GI/bridge but not the files derived from "
        "them: rebuild context.json's hosts/guests/people and make each diagnostics voice name "
        "follow its segment label"
    )

    def _episodes(self, root: Path) -> Iterable[_Episode]:
        for meta in select_served_artifacts(root, ".metadata.json")[0]:
            ep = _Episode(meta)
            if ep.plan():
                yield ep

    def _totals(self, root: Path) -> Tuple[List[_Episode], Dict[str, int]]:
        eps = list(self._episodes(root))
        totals: Dict[str, int] = {}
        for ep in eps:
            for k, v in ep.counts.items():
                totals[k] = totals.get(k, 0) + v
        return eps, totals

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would rewrite — pure read, no writes."""
        eps, totals = self._totals(ctx.corpus_root)
        return f"derived surfaces plan: {len(eps)} episode(s); " + ", ".join(
            f"{k}={v}" for k, v in sorted(totals.items())
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """No served episode's context digest or diagnostics disagree with it. ``(ok, message)``."""
        left = [ep.meta.name for ep in self._episodes(ctx.corpus_root)]
        if left:
            return False, f"{len(left)} episode(s) still carry a stale speaker name: {left[:5]}"
        return True, "every served episode's context digest and diagnostics follow its segments"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Rewrite every stale context digest and diagnostics file; back up each, receipt it."""
        root = ctx.corpus_root
        eps, totals = self._totals(root)
        receipts: List[dict] = []
        if not ctx.dry_run:
            for ep in eps:
                for path in sorted(ep.files):
                    receipts.append(write_with_backup(root, BACKUP_TAG, path, ep.files[path]))
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

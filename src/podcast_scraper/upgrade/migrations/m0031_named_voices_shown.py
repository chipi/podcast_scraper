"""0031 — the voices 0029 named, shown the way a fresh run shows a named person (#2380).

0029 put its names on with m0025's voice-by-voice rewrite, which until 1bc9d18d0 kept the hiding
an unnamed voice carries: the voice's segments stayed typed ``unknown``, its quotes kept
``speaker_voice_type``, and every insight whose first quote it speaks stayed ``surfaceable: false``
and routed to connect. Measured on the 2026-10-16 snapshot: 1,779 insights named but hidden in the
73 episodes 0029 wrote; their pre-0029 backups have none. Two more of 0029's episodes were
reprocessed after it ran (2026-10-13, 2026-10-14) and came back with the voice unnamed again, so
0029's own verify lists them.

0029 is in every applied ledger and never runs again, so this migration carries its list
(``NAMING_ARC_VOICES``) once more, through the fixed rewrite. Per listed voice that 0029 names:

- stored as the replay read it: named now, exactly as 0029 would name it today;
- stored with the new name, its segments still typed: the name re-given through the same
  rewrite, which changes no label and lifts the hiding;
- stored with the new name and untyped: done (a second run writes nothing);
- anything else: the episode is refused, untouched and listed.

ORDER. This is the part of 0029 that went missing, so it takes the number right after 0030. The
naming re-derivation that held this number until 2026-10-18 is now 0032 (held, #2380); it never
ran on prod, and the hold was moved with it.

``verify`` fails while anything is left to write or refused. UNDO restores from
``.podcast_scraper/upgrade-backups/0031/`` while each file is still what this migration wrote.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from ..file_rewrite import undo_from_receipts
from ..migration import MigrationContext
from . import m0029_naming_arc_voices_repaired as m29
from .m0025_one_person_one_entry import _load, _run_relative

MIGRATION_ID = "0031_named_voices_shown"
RECEIPTS_FILE = "named_voices_shown.jsonl"
BACKUP_TAG = "0031"


def _typed_voices(meta: Path, rel: str) -> Set[str]:
    """Voices with any segment carrying a ``voice_type`` (a named person carries none)."""
    payload = _load(meta.parent.parent / rel.replace(".txt", ".segments.json"))
    rows = payload if isinstance(payload, list) else (payload or {}).get("segments")
    return {
        str(r["speaker"])
        for r in rows or []
        if isinstance(r, dict) and r.get("speaker") and r.get("voice_type")
    }


class _Episode(m29._Episode):
    """0029's decision, plus: a voice already carrying its new name but still typed is re-given
    it, so the fixed rewrite lifts the hiding."""

    def decide(self) -> bool:
        meta = _load(self.meta)
        if not isinstance(meta, dict):
            return False
        episode_id = str((meta.get("episode") or {}).get("episode_id") or "")
        mine = {v: e for (eid, v), e in self._map.items() if eid == episode_id and e[2]}
        if not mine:
            return False
        rel = _run_relative(
            (meta.get("content") or {}).get("transcript_file_path"), self.meta.parent.parent
        )
        if not rel.endswith(".txt"):
            self.refused = "no transcript"
            return True
        self._load_seg_voices(rel)
        stored = m29._stored_by_voice(self.meta, rel)
        typed = _typed_voices(self.meta, rel)
        for voice, (old_name, old_role, new_name, new_role, _why) in sorted(mine.items()):
            name, role = stored.get(voice, (None, None))
            name = None if not name or m29._DIARIZATION_LABEL.match(name) else name
            if (name, role) == (new_name, new_role):
                if voice not in typed:
                    continue  # done
                self._bump("voices_shown")
            elif (name, role) != (old_name, old_role):
                self.refused = (
                    f"{voice}: stored {name!r}/{role!r}, the replay read {old_name!r}/{old_role!r}"
                )
                return True
            self.voice_names[voice] = new_name
            if new_role:
                self.voice_roles[voice] = new_role
        return bool(self.voice_names)


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class NamedVoicesShownMigration(m29.NamingArcVoicesRepairedMigration):
    """0029's voices through the fixed rewrite: named where still unnamed, shown where hidden."""

    id = MIGRATION_ID
    to_version = "2.7.25"
    description = (
        "0029's voices through the fixed voice-by-voice rewrite (#2380): the ones a reprocess "
        "unnamed again are named, and the ones still typed and hidden under their name are shown"
    )
    _episode_class = _Episode
    _receipts_file = RECEIPTS_FILE
    _backup_tag = BACKUP_TAG

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Pass when every voice 0029 names carries its name and no hiding, and none refused."""
        ok, message = super().verify(ctx)
        return ok, ("every voice 0029 names carries its name and no hiding" if ok else message)


def dry_run_report(root: Path) -> Iterable[Dict[str, Any]]:
    """Read-only: one row per episode the migration would write or refuses."""
    write, refused, _totals = NamedVoicesShownMigration()._scan(Path(root))
    for ep in write + refused:
        yield {
            "meta": str(ep.meta.relative_to(root)),
            "voices": dict(ep.voice_names),
            "refused": ep.refused,
        }

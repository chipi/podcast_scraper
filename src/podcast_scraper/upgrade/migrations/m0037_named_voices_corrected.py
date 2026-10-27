"""0037 — served voices that carry another person's name get their own (text only).

The list is not computed: each entry is a voice whose served name two blind readers (night 1,
naming plan item 13) or the episode's own diagnostics and text (#2094 D5) show to be another
person's, with the name the text shows it is.
Record: ``docs/wip/changes/0037_named_voices_corrected.md``.

Entries are 0035's shape (``old name, old role, new name, new role, evidence``) in
``upgrade/lists/0037_named_voices_corrected.json``; the evidence names the readers' basis and the
deciding segment, and an entry without it is refused when the list is loaded. Applied through
0035's reviewed-list machinery (verdicts with basis columns bound by the list's sha256, a listed
voice without a verdict fails ``verify``), 0029's voice-by-voice rewrite and 0031's derived
surfaces, with one rule of its own: voices that exchange names inside one episode (a swap) are
applied together or not at all, since half a swap publishes one name on two voices. A renamed
voice's diagnostics ``source`` is the rewrite's own (``roster``), not the voice identity rung's.
Not in the registry until the verdicts exist. UNDO restores from
``.podcast_scraper/upgrade-backups/0037/``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from .. import reviewer_verdicts as rv
from ..file_rewrite import undo_from_receipts
from . import (
    m0029_naming_arc_voices_repaired as m29,
    m0034_talk_floor_names_removed as m34,
    m0035_voice_identity_names as m35,
)

MIGRATION_ID = "0037_named_voices_corrected"
RECEIPTS_FILE = "named_voices_corrected.jsonl"
BACKUP_TAG = "0037"
LIST_FILE = Path(__file__).resolve().parents[1] / "lists" / f"{MIGRATION_ID}.json"
#: What an entry's evidence must name: the deciding segment of the episode's ``.segments.json``.
_EVIDENCE = re.compile(r"\bsegment \d+\b")


class ListError(ValueError):
    """The frozen list holds an entry this migration refuses to apply."""


def load_list(path: Path = LIST_FILE) -> Dict[rv.Key, m29.Entry]:
    """The frozen list; empty when it has not been made yet. Every entry is a rename (a name
    replaced by its own person's) and names its deciding segment."""
    out = m34.load_list(path)
    for key, (old, _old_role, new, _new_role, why) in out.items():
        if not old or not new:
            raise ListError(f"{key}: 0037 only renames; old and new names are both required")
        if not _EVIDENCE.search(why or ""):
            raise ListError(f"{key}: the entry names no deciding segment ({why!r})")
    return out


def swap_groups(listed: Dict[rv.Key, m29.Entry]) -> List[Set[rv.Key]]:
    """Voices of one episode linked by a name one holds and another takes: each group is applied
    whole or not at all."""
    groups: List[Set[rv.Key]] = []
    by_episode: Dict[str, List[rv.Key]] = {}
    for key in listed:
        by_episode.setdefault(key[0], []).append(key)
    for keys in by_episode.values():
        for a in keys:
            for b in keys:
                if a < b and (
                    str(listed[a][0]).lower() == str(listed[b][2]).lower()
                    or str(listed[b][0]).lower() == str(listed[a][2]).lower()
                ):
                    merged = {a, b}
                    for g in [g for g in groups if g & merged]:
                        merged |= g
                        groups.remove(g)
                    groups.append(merged)
    return groups


class _Episode(m35._Episode):
    """0035's rewrite with the rewrite's own source (``roster``) on a renamed voice: a reviewed
    correction is not the voice identity rung's name and must not vote in the feed bank as one."""

    RENAMED_SOURCE = m34._Episode.RENAMED_SOURCE
    RESPELLING_TAKES_RENAMED_SOURCE = m34._Episode.RESPELLING_TAKES_RENAMED_SOURCE


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class NamedVoicesCorrectedMigration(m35.VoiceIdentityNamesMigration):
    """Wrong-person names on served voices replaced by the voices' own, as listed and reviewed."""

    id = MIGRATION_ID
    to_version = "2.7.31"
    description = (
        "served voices holding another person's name take their own (night-1 blind reads, #2094 "
        "D5): frozen, reviewed list, swaps applied whole, through 0029's rewrite with 0031's "
        "derived surfaces"
    )
    _episode_class = _Episode
    _receipts_file = RECEIPTS_FILE
    _backup_tag = BACKUP_TAG
    _plan_label = "named voices corrected"
    _verified = "every rename the review confirmed carries the voice's own person's name"

    def listed(self) -> Dict[rv.Key, m29.Entry]:
        """This migration's frozen list (``upgrade/lists/0037_named_voices_corrected.json``)."""
        return load_list()

    def _verdicts(self) -> rv.Verdicts:
        """The frozen list's verdicts, with their basis columns and 0035's basis set."""
        return rv.load(
            self.id,
            self.review_names(),
            self._verdicts_dir,
            listed=self.listed(),
            require_basis=True,
            bases=m35.BASES,
        )

    def _reviewed(self) -> Tuple[Dict[rv.Key, m29.Entry], List[rv.Key]]:
        """0035's rule (a rename applies on ``CORRECT`` / ``CORRECT_MISSPELLED``), then a swap
        group with any member not applied is dropped whole."""
        apply, missing = super()._reviewed()
        for group in swap_groups(self.listed()):
            if not group <= set(apply):
                for key in group:
                    apply.pop(key, None)
        return apply, missing


def verdicts_file() -> Path:
    """The verdicts file this migration reads (a plan-level input, hashed into the code digest)."""
    return rv.verdicts_path(MIGRATION_ID)


def list_file() -> Path:
    """The frozen list this migration applies (a plan-level input, hashed into the code digest)."""
    return LIST_FILE


def dry_run_report(root: Path) -> Iterable[Dict[str, Any]]:
    """Read-only: one row per episode the migration would write or refuses, with its counters
    and input sha256s (0034's ``plan_rows``)."""
    return m34.plan_rows(root, NamedVoicesCorrectedMigration())

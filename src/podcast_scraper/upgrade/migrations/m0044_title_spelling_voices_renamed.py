"""0044 — stored voices take the spelling their own episode title states (B5's stored episodes).

B5 (``fe231664``, #2363) made the people an episode title names spelling references, so a new
episode's voice carries the title's spelling. Episodes stored before it keep the heard (ASR)
spelling. The list is the roster replay of ``fe231664^`` against ``fe231664`` over the snapshot,
each entry a voice whose stored name is not its title's spelling of its person, with the deciding
segment; a fresh blind reader's verdicts decide which apply.
Record: ``docs/wip/changes/0044_title_spelling_voices_renamed.md``.

0037's machinery as it stands: entries are renames naming a deciding segment, verdicts with basis
columns bound by the list's sha256, swaps applied whole, 0029's voice-by-voice rewrite with 0031's
derived surfaces. UNDO restores from ``.podcast_scraper/upgrade-backups/0044/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .. import reviewer_verdicts as rv
from ..file_rewrite import undo_from_receipts
from . import (
    m0029_naming_arc_voices_repaired as m29,
    m0034_talk_floor_names_removed as m34,
    m0037_named_voices_corrected as m37,
)

MIGRATION_ID = "0044_title_spelling_voices_renamed"
RECEIPTS_FILE = "title_spelling_voices_renamed.jsonl"
BACKUP_TAG = "0044"
LIST_FILE = Path(__file__).resolve().parents[1] / "lists" / f"{MIGRATION_ID}.json"


def load_list(path: Path = LIST_FILE) -> Dict[rv.Key, m29.Entry]:
    """The frozen list, held to 0037's rule: renames only, each naming its deciding segment."""
    return m37.load_list(path)


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class TitleSpellingVoicesRenamedMigration(m37.NamedVoicesCorrectedMigration):
    """Stored voices renamed to their episode title's spelling of their person, as reviewed."""

    id = MIGRATION_ID
    to_version = "2.7.38"
    description = (
        "stored voices take their episode title's spelling of their person (B5's stored "
        "episodes, #2363): frozen, reviewed list, through 0037's rename machinery"
    )
    _receipts_file = RECEIPTS_FILE
    _backup_tag = BACKUP_TAG
    _plan_label = "title spelling voices renamed"
    _verified = "every rename the review confirmed carries the title's spelling of its person"

    def listed(self) -> Dict[rv.Key, m29.Entry]:
        """This migration's frozen list (``upgrade/lists/<id>.json``)."""
        return load_list()


def verdicts_file() -> Path:
    """The verdicts file this migration reads (a plan-level input, hashed into the code digest)."""
    return rv.verdicts_path(MIGRATION_ID)


def list_file() -> Path:
    """The frozen list this migration applies (a plan-level input, hashed into the code digest)."""
    return LIST_FILE


def dry_run_report(root: Path) -> Iterable[Dict[str, Any]]:
    """Read-only: one row per episode it would write or refuses (0034's ``plan_rows``)."""
    return m34.plan_rows(root, TitleSpellingVoicesRenamedMigration())

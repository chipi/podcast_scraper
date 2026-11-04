"""0045 — one episode's guest-named voices: the show's host's voice and the guest's own spelling.

Found by 0044's blind review: in one episode the second host's voice carries a heard spelling of
the titled guest, and the guest's own voice carries another heard spelling. The list is those two
voices, each with its deciding segment; a fresh blind reader's verdicts decide which apply.
Record: ``docs/wip/changes/0045_episode_voices_corrected.md``.

0037's machinery as it stands: entries are renames naming a deciding segment, verdicts with basis
columns bound by the list's sha256, swaps applied whole, 0029's voice-by-voice rewrite with 0031's
derived surfaces. UNDO restores from ``.podcast_scraper/upgrade-backups/0045/``.
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

MIGRATION_ID = "0045_episode_voices_corrected"
RECEIPTS_FILE = "episode_voices_corrected.jsonl"
BACKUP_TAG = "0045"
LIST_FILE = Path(__file__).resolve().parents[1] / "lists" / f"{MIGRATION_ID}.json"


def load_list(path: Path = LIST_FILE) -> Dict[rv.Key, m29.Entry]:
    """The frozen list, held to 0037's rule: renames only, each naming its deciding segment."""
    return m37.load_list(path)


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class EpisodeVoicesCorrectedMigration(m37.NamedVoicesCorrectedMigration):
    """One episode's guest-named voices given their own persons' names, as reviewed."""

    id = MIGRATION_ID
    to_version = "2.7.39"
    description = (
        "one episode's two guest-named voices take their own names: the show's host's voice and "
        "the guest's title spelling (found by 0044's review): frozen, reviewed list, 0037's "
        "machinery"
    )
    _receipts_file = RECEIPTS_FILE
    _backup_tag = BACKUP_TAG
    _plan_label = "episode voices corrected"
    _verified = "every rename the review confirmed carries the voice's own person's name"

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
    return m34.plan_rows(root, EpisodeVoicesCorrectedMigration())

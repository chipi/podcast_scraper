"""0033 — a name 0015 removed by hand, back on a voice after a reprocess, put right.

0015 removed six names no gate rule catches safely ("Generation", "Moral", ...) on 2026-10-03. A
migration runs once, so a reprocess after it could publish one again, and one did: The Long Run's
"Ep176: David Roblin on Building a TechBio Company" was reprocessed on 2026-10-14, and the
self-introduction reader took the guest's "I'm Generation X, I suppose" for "I'm Generation". The
guest's voice (SPEAKER_00) was published as "Generation" on 58 insights and 109 quotes, and the
stated guest, David Roblin, was left on no voice. 0015's verify has failed on it since.

Two rules stop it coming back (``speaker_detectors.hosts``): the publish gate refuses 0015's
hand-removed names, and a one-word self-introduction followed by a lone capital letter is a label,
not a name. This migration repairs the episode already published.

THE REPAIR. A fresh reviewer read the episode (2026-10-19) and judged SPEAKER_00 to be David Roblin
(``upgrade/verdicts/0033_hand_removed_names_refused.tsv``, the evidence segments in its row). The
voice is renamed through 0031's voice-by-voice rewrite, which carries every derived surface 0031
measured (transcript prefixes and offsets, turns, diagnostics, context, ad maps, metadata counts):
the "Generation" person merges into the ``person:david-roblin`` the episode's KG already holds, and
its 58 insights stay shown. With no verdict the episode is refused, untouched. A rename, not an
unnaming: a wrong name is worse than a hidden insight, and this name is the right one.

UNDO restores from ``.podcast_scraper/upgrade-backups/0033/`` while each file is still as written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .. import reviewer_verdicts as rv
from ..file_rewrite import undo_from_receipts
from . import m0029_naming_arc_voices_repaired as m29, m0031_named_voices_shown as m31

MIGRATION_ID = "0033_hand_removed_names_refused"
RECEIPTS_FILE = "hand_removed_names_refused.jsonl"
BACKUP_TAG = "0033"

#: ``{(episode id, voice): (old name, old role, new name, new role, why)}``, as 0029's list.
VOICES: Dict[rv.Key, m29.Entry] = {
    ("tag:soundcloud,2010:tracks/2067916744", "SPEAKER_00"): (
        "Generation",
        "guest",
        "David Roblin",
        "guest",
        "The Long Run 'Ep176: David Roblin on Building a TechBio Company': the "
        "self-introduction reader took 'I'm Generation X, I suppose' for a name; the voice "
        "speaks as the stated guest ('Thanks, Luke. It's an absolute pleasure to be on.')",
    ),
}


class _Episode(m31._Episode):
    """0031's episode, deciding by this migration's list."""

    def __init__(
        self,
        meta: Path,
        root: Path,
        voices: Optional[Dict] = None,
        verdicts: Optional[rv.Verdicts] = None,
    ) -> None:
        super().__init__(meta, root, VOICES if voices is None else voices, verdicts)


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class HandRemovedNamesRefusedMigration(m31.NamedVoicesShownMigration):
    """The voice a reprocess published under a hand-removed name, named as the review read it."""

    id = MIGRATION_ID
    to_version = "2.7.27"
    description = (
        "the voice a reprocess published as 'Generation' (a name 0015 removed by hand), named "
        "David Roblin as a fresh review read it, through 0031's voice-by-voice rewrite"
    )
    _episode_class = _Episode
    _receipts_file = RECEIPTS_FILE
    _backup_tag = BACKUP_TAG
    _plan_label = "hand-removed names"
    _names_by = "this migration names"

    def listed(self) -> Dict[rv.Key, m29.Entry]:
        """This migration's own list (``VOICES``)."""
        return VOICES


def verdicts_file() -> Path:
    """The verdicts file this migration reads (a plan-level input, hashed into the code digest)."""
    return rv.verdicts_path(MIGRATION_ID)


def dry_run_report(root: Path) -> Iterable[Dict[str, Any]]:
    """Read-only: one row per episode the migration would write or refuses (the outcome hook's
    plan contract), as 0031's."""
    return m31.dry_run_rows(Path(root), HandRemovedNamesRefusedMigration())

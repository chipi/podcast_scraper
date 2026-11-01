"""The ledger step's respelling: after each build, the episode files take the new group spellings.

Operator decision, 2026-10-31 (RFC-131 D2 item 9): a migration (0041) writes each joined person's
group spelling into the episode files; a later build can change a group's spelling (new episodes
add publisher text), so the ledger step respells the files after every build, text only, with
receipts. This is 0041's own decision and rewrite (``m0041.episode_renames``, ``m0041._Episode``),
so the two never disagree. Record: ``docs/wip/changes/0041_joined_names_respelled.md``.

**Against the files, not the previous ledger.** Every episode holding a joined id is decided from
what its files say now, so a respelling deferred once (the corpus lock held by another writer) is
caught up by the next build, and a second run writes nothing.

**Each run is its own undo:** backups under ``.podcast_scraper/upgrade-backups/ledger-respell-
<stamp>/`` and receipts in ``ledger_respell_<stamp>.jsonl`` at the corpus root, the stamp
filesystem-safe; :func:`undo_respell` restores one run.

**Under the corpus write lock**, taken here without waiting: held by another writer, the
respelling is deferred and logged, and read time keeps every surface but the transcript right.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RECEIPTS_PREFIX = "ledger_respell_"
BACKUP_PREFIX = "ledger-respell-"
_ID = "ledger_respell"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def respell_files(
    root: Path | str, ledger: Dict[str, Any], *, log: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Respell every episode whose files differ from ``ledger``'s group spellings. Returns the
    counts (``written``, ``refused``, ``files``, ``deferred``, ``stamp``)."""
    log = log or logger
    root = Path(root)
    out: Dict[str, Any] = {"written": 0, "refused": 0, "files": 0, "deferred": False, "stamp": None}
    try:
        from podcast_scraper.utils.corpus_lock import corpus_parent_lock

        lock = corpus_parent_lock(root, logger=log)
        lock.__enter__()
    except RuntimeError as exc:
        log.warning("ledger respell deferred: the corpus lock is held (%s)", exc)
        out["deferred"] = True
        return out
    try:
        out.update(_respell_locked(root, ledger))
    finally:
        lock.__exit__(None, None, None)
    if out["written"] or out["refused"]:
        log.info(
            "ledger respell %s: %d episode(s) written (%d files), %d refused",
            out["stamp"],
            out["written"],
            out["files"],
            out["refused"],
        )
    return out


def _respell_locked(root: Path, ledger: Dict[str, Any]) -> Dict[str, Any]:
    from podcast_scraper.upgrade.corpus_selection import select_served_artifacts
    from podcast_scraper.upgrade.file_rewrite import append_receipts, write_with_backup
    from podcast_scraper.upgrade.migrations import m0041_joined_names_respelled as m39
    from podcast_scraper.upgrade.migrations.m0023_transcript_speaker_prefixes_resynced import (
        _write_text_with_backup,
    )

    view = m39.view_of(ledger)
    if not len(view.display):
        return {}
    stamp = _stamp()
    tag = BACKUP_PREFIX + stamp
    receipts: List[dict] = []
    written = refused = 0
    for meta in select_served_artifacts(root, ".metadata.json")[0]:
        ep = m39._Episode(meta, root, view=view)
        if ep.plan():
            written += 1
            for path, payload in sorted(ep.json_files.items()):
                receipts.append(write_with_backup(root, tag, path, payload))
            for path, text in sorted(ep.text_files.items()):
                receipts.append(_write_text_with_backup(root, path, text, tag))
        if ep.refused:
            refused += 1
    append_receipts(
        root,
        RECEIPTS_PREFIX + stamp + ".jsonl",
        {"migration": _ID, "ledger_built_at": ledger.get("built_at"), "stamp": stamp},
        receipts,
    )
    return {"written": written, "refused": refused, "files": len(receipts), "stamp": stamp}


def undo_respell(root: Path | str, stamp: str) -> Tuple[int, List[str]]:
    """Restore the files one respelling run wrote, if still as it left them."""
    from podcast_scraper.upgrade.file_rewrite import undo_from_receipts

    return undo_from_receipts(
        Path(root), RECEIPTS_PREFIX + stamp + ".jsonl", BACKUP_PREFIX + stamp, _ID
    )

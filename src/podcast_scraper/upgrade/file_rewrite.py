"""Backed-up, receipted, undoable artifact rewrites for migrations that change whole files.

One pattern, shared so each migration does not grow its own copy: before a file is replaced it is
copied, as it was, under ``.podcast_scraper/upgrade-backups/<tag>/<relpath>``; every write appends
a receipt ``{relpath, sha_before, sha_after}`` to a corpus-root JSONL whose first line for a run is
a ``header`` (anything the migration must remember, e.g. a frozen set); ``undo`` restores a file
only while it is still exactly what the migration wrote. Ownership follows the corpus root
(``ownership.match_corpus_owner``).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .ownership import created_dirs, match_corpus_owner
from .role_ledger import file_sha


def dump_json(payload: Any) -> str:
    """Serialise a corpus artifact the way the pipeline writes it.

    `ensure_ascii=False` and the trailing newline are not cosmetic: a migration rewrites
    files the pipeline also writes, and a different encoding or a missing final newline
    would make every migrated file differ from its own regenerated form.
    """
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def backup_dir(root: Path, tag: str) -> Path:
    """Where one migration's backups live, keyed by its tag.

    Inside the corpus rather than beside it, so a restored or copied corpus carries its
    own undo history with it; per-tag, so two migrations cannot overwrite each other's
    backup of the same file.
    """
    return Path(root) / ".podcast_scraper" / "upgrade-backups" / tag


def write_with_backup(root: Path, tag: str, path: Path, payload: Any) -> Dict[str, str]:
    """Back up *path*, replace it atomically with *payload*, return its receipt."""
    root = Path(root)
    rel = str(path.relative_to(root))
    backup = backup_dir(root, tag) / rel
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copyfile(path, backup)
    sha_before = file_sha(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(dump_json(payload), encoding="utf-8")
    os.replace(tmp, path)
    match_corpus_owner(root, [path, backup, *created_dirs(backup.parent, root)])
    return {"relpath": rel, "sha_before": sha_before, "sha_after": file_sha(path)}


def append_receipts(
    root: Path, receipts_file: str, header: Dict[str, Any], receipts: List[Dict[str, str]]
) -> None:
    """Append one run's header + receipts (no-op when nothing was written)."""
    if not receipts:
        return
    path = Path(root) / receipts_file
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "header", **header}, ensure_ascii=False) + "\n")
        for row in receipts:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    match_corpus_owner(Path(root), [path])


def read_receipts(root: Path, receipts_file: str) -> Tuple[Dict[str, Any], List[dict]]:
    """``(last header, every receipt row)``."""
    header: Dict[str, Any] = {}
    rows: List[dict] = []
    try:
        lines = (Path(root) / receipts_file).read_text(encoding="utf-8").splitlines()
    except OSError:
        return header, rows
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") == "header":
            header = row
        else:
            rows.append(row)
    return header, rows


def undo_from_receipts(
    root: Path, receipts_file: str, tag: str, migration_id: str
) -> Tuple[int, List[str]]:
    """Restore each written file still as the migration left it. ``(restored, refused)``."""
    root = Path(root)
    _header, rows = read_receipts(root, receipts_file)
    restored, refused = 0, []
    for row in rows:
        target = root / row["relpath"]
        backup = backup_dir(root, tag) / row["relpath"]
        if file_sha(target) != row["sha_after"]:
            refused.append(f"{row['relpath']}: changed since the migration wrote it")
            continue
        if not backup.is_file():
            refused.append(f"{row['relpath']}: no backup")
            continue
        tmp = target.with_name(target.name + ".tmp")
        shutil.copyfile(backup, tmp)
        os.replace(tmp, target)
        match_corpus_owner(root, [target])
        restored += 1
    if restored:
        try:
            from .state import FilesystemStateStore

            FilesystemStateStore(root).record_reverted(migration_id)
        except Exception:  # noqa: BLE001 — the files are restored; never undo the undo
            pass
    return restored, refused

"""0027 — store the covers the corpus only points at, so phones are never sent an original.

The app serves artwork from our own store (``/api/app/artwork``), downscaled for every slot. A
cover the pipeline did not store has no store path, so the app falls back to the feed host's URL —
and nothing can downscale that: on prod (2026-10-08) "The China-Global South Podcast" cover, a
3000x3000 PNG of 11.9 MB, went to every phone in full for a 116 px tile. It was never stored
because the writer's size cap was 8 MB (now 32 MB).

For every served episode whose ``feed`` or ``episode`` block has an ``image_url`` but no
``image_local_relpath``: download the image once per URL into the content-addressed store (which
also writes its thumb and medium copies), then record the store path on those episodes.
Metadata rewrites are backed up and receipted (``undo``). An image that cannot be fetched or is not
an image is recorded in ``missing_covers_failed.jsonl`` and left to the remote URL, as today.

FETCHES — like 0021. ``--dry-run`` lists what it would download and writes nothing.
Idempotent: a block that already has a store path is never touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...config_constants import DEFAULT_USER_AGENT
from ...utils.corpus_artwork import (
    download_podcast_artwork,
    medium_path,
    thumbnail_path,
)
from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult
from ..ownership import created_dirs, match_corpus_owner

MIGRATION_ID = "0027_missing_covers_stored"
RECEIPTS_FILE = "missing_covers_stored.jsonl"
FAILED_FILE = "missing_covers_failed.jsonl"
BACKUP_TAG = "0027"
DEFAULT_FETCH_TIMEOUT = 30.0
_BLOCKS = ("feed", "episode")


def _load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _missing(payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    """``[(block, url)]`` for each block with an image URL and no store path."""
    out: List[Tuple[str, str]] = []
    for block in _BLOCKS:
        b = payload.get(block)
        if not isinstance(b, dict):
            continue
        url = str(b.get("image_url") or "").strip()
        if url.startswith(("http://", "https://")) and not str(b.get("image_local_relpath") or ""):
            out.append((block, url))
    return out


def _failed(root: Path) -> set:
    try:
        lines = (root / FAILED_FILE).read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    return {json.loads(line).get("url") for line in lines if line.strip()}


def _wanted(root: Path) -> Tuple[Dict[str, List[Tuple[Path, str]]], int]:
    """``({url: [(metadata path, block)]}, served count)``."""
    served, _superseded = select_served_artifacts(root, ".metadata.json")
    by_url: Dict[str, List[Tuple[Path, str]]] = {}
    for path in served:
        payload = _load(path)
        if payload is None:
            continue
        for block, url in _missing(payload):
            by_url.setdefault(url, []).append((path, block))
    return by_url, len(served)


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore every metadata file this migration rewrote (the stored images stay: derived data)."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class MissingCoversStoredMigration(Migration):
    """Download and record the covers the corpus only links to."""

    id = MIGRATION_ID
    to_version = "2.7.21"
    description = (
        "Store covers that only exist as a feed-host URL (one was an 11.9 MB PNG sent to phones "
        "in full) and record their store path, so every slot gets a downscale"
    )

    def plan(self, ctx: MigrationContext) -> str:
        """What apply() would fetch — pure read, no network."""
        by_url, served = _wanted(ctx.corpus_root)
        known_bad = _failed(ctx.corpus_root)
        todo = [u for u in by_url if u not in known_bad]
        refs = sum(len(by_url[u]) for u in todo)
        return (
            f"missing covers plan: {served} episode(s); {len(todo)} image(s) to fetch "
            f"for {refs} reference(s)"
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Every image URL has a store path, or is recorded unfetchable. Reads the corpus only."""
        by_url, _served = _wanted(ctx.corpus_root)
        left = [u for u in by_url if u not in _failed(ctx.corpus_root)]
        if left:
            return False, f"{len(left)} cover(s) still only a remote URL: {left[:3]}"
        return True, "every cover is stored, or recorded as unfetchable"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Fetch each missing cover once, store it with its thumb and medium, record the path."""
        root = ctx.corpus_root
        timeout = float(ctx.options.get("fetch_timeout") or DEFAULT_FETCH_TIMEOUT)
        by_url, served = _wanted(root)
        known_bad = _failed(root)
        todo = sorted(u for u in by_url if u not in known_bad)
        stored: Dict[str, str] = {}
        failed: List[str] = []
        files_written = 0
        for url in todo:
            refs = by_url[url]
            ctx.log(f"  {url[:90]} — {len(refs)} reference(s)")
            if ctx.dry_run:
                continue
            rel = download_podcast_artwork(
                url, root, user_agent=DEFAULT_USER_AGENT, timeout=int(timeout)
            )
            if not rel:
                failed.append(url)
                ctx.log("    FAILED — not fetched or not an image; left to the remote URL")
                continue
            original = root / rel
            derived = [thumbnail_path(root, str(original)), medium_path(root, str(original))]
            owned = [original, *created_dirs(original.parent, root)]
            for d in derived:
                if d.exists():
                    owned += [d, *created_dirs(d.parent, root)]
            match_corpus_owner(root, owned)
            stored[url] = rel
            receipts: List[Dict[str, str]] = []
            for path, block in refs:
                payload = _load(path)
                if payload is None or not isinstance(payload.get(block), dict):
                    continue
                if str(payload[block].get("image_local_relpath") or ""):
                    continue
                payload[block]["image_local_relpath"] = rel
                receipts.append(write_with_backup(root, BACKUP_TAG, path, payload))
            append_receipts(
                root,
                RECEIPTS_FILE,
                {"migration": MIGRATION_ID, "url": url, "relpath": rel},
                receipts,
            )
            files_written += len(receipts)
        if failed and not ctx.dry_run:
            with (root / FAILED_FILE).open("a", encoding="utf-8") as fh:
                for url in failed:
                    fh.write(json.dumps({"url": url}) + "\n")
            match_corpus_owner(root, [root / FAILED_FILE])
        verb = "would fetch" if ctx.dry_run else "stored"
        count = len(todo) if ctx.dry_run else len(stored)
        message = (
            f"{served} episode(s): {verb} {count} cover(s); {files_written} metadata file(s) "
            f"updated; {len(failed)} unfetchable (recorded)"
        )
        ctx.log(message)
        return MigrationResult(
            migration_id=self.id,
            applied=not ctx.dry_run,
            dry_run=ctx.dry_run,
            message=message,
            details={
                "to_fetch": len(todo),
                "stored": stored,
                "files_written": files_written,
                "failed": failed,
            },
        )

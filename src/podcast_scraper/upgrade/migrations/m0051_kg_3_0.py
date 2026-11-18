"""0051 — every served podcast KG file rewritten from KG 2.x to KG 3.0 (R3b, the KG half).

Record ``docs/wip/changes/r3b_kg_3_0.md``; ontology ``docs/architecture/corpus/ontology-v3.md``.
Each served KG 2.x file becomes ``kg.v3_file.to_v3_file`` of itself, from stored evidence only:
``Episode``/``Podcast``/``HAS_EPISODE`` become ``ContentItem``/``Source``/``HAS_ITEM``; ``HOSTS``
and ``GUESTS_ON`` (every one points at the show) become ``CONTRIBUTES_TO`` with role ``host`` or
``guest``; the extraction's ``transcript_ref`` becomes ``document_ref`` with its ``variant``; the
root names its item; ids never change (T-8); ``respelled_from`` stays. When the publisher registry
lists the show's ``Source`` id, ``PUBLISHED_BY`` is added.

A KG 3.x file is left as it is, so a second apply writes nothing. Every written file validates
strictly against ``kg.v3.schema.json`` first; a file that does not is refused, not written. Backups
and receipts: ``.podcast_scraper/upgrade-backups/0051/`` and ``kg_3_0.jsonl``. An older migration's
undo is refused on a file this one rewrote (``undo_from_receipts`` compares the file with what that
migration left).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from podcast_scraper.kg.as_v3 import is_v3
from podcast_scraper.kg.schema import validate_artifact
from podcast_scraper.kg.v3_file import to_v3_file
from podcast_scraper.publishers import load_registry, Registry
from podcast_scraper.upgrade.corpus_selection import select_served_artifacts
from podcast_scraper.upgrade.file_rewrite import (
    append_receipts,
    undo_from_receipts,
    write_with_backup,
)
from podcast_scraper.upgrade.migration import Migration, MigrationContext, MigrationResult

MIGRATION_ID = "0051_kg_3_0"
RECEIPTS_FILE = "kg_3_0.jsonl"
BACKUP_TAG = "0051"

#: The per-episode counters, in the apply message and the dry-run report.
COUNTERS = ("files_written", "contributes_to", "published_by", "refused_invalid")


def _json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _language(kg_path: Path) -> Optional[str]:
    meta = _json(kg_path.with_name(kg_path.name[: -len(".kg.json")] + ".metadata.json"))
    if not isinstance(meta, dict):
        return None
    lang = (meta.get("episode") or {}).get("language")
    return str(lang) if lang else None


def episode_changes(
    kg_path: Path, registry: Registry
) -> Optional[Tuple[Dict[str, Any], Dict[str, int]]]:
    """``(kg_3_0, counters)`` for a KG 2.x file, ``None`` for a 3.x or unreadable one. Pure."""
    kg = _json(kg_path)
    if not isinstance(kg, dict) or is_v3(kg):
        return None
    v3 = to_v3_file(kg, language=_language(kg_path), registry=registry)
    counters = {k: 0 for k in COUNTERS}
    try:
        validate_artifact(v3, strict=True)
    except ValueError:
        counters["refused_invalid"] = 1
        return v3, counters
    counters["files_written"] = 1
    for e in v3.get("edges") or []:
        if e.get("type") == "CONTRIBUTES_TO":
            counters["contributes_to"] += 1
        elif e.get("type") == "PUBLISHED_BY":
            counters["published_by"] += 1
    return v3, counters


def find_changes(
    root: Path, registry: Optional[Registry] = None
) -> Iterable[Tuple[Path, Dict[Path, Any], Dict[str, int]]]:
    """``(kg_path, writes, counters)`` per SERVED KG 2.x file; a refused file yields no write."""
    registry = load_registry() if registry is None else registry
    for kg_path in select_served_artifacts(Path(root), ".kg.json")[0]:
        found = episode_changes(kg_path, registry)
        if found is None:
            continue
        v3, counters = found
        writes = {} if counters["refused_invalid"] else {kg_path: v3}
        yield kg_path, writes, counters


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class KgThreeMigration(Migration):
    """Every served podcast KG file is KG 3.0."""

    id = MIGRATION_ID
    to_version = "2.8.1"
    description = (
        "every served KG file rewritten from KG 2.x to KG 3.0 (R3b): ontology v3 names, hosts and "
        "guests as CONTRIBUTES_TO with their role, the evidence's document reference, PUBLISHED_BY "
        "from the publisher registry"
    )

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would change: pure read, no writes."""
        totals = {k: 0 for k in COUNTERS}
        for _p, _w, c in find_changes(ctx.corpus_root):
            for k in COUNTERS:
                totals[k] += c[k]
        return f"{self.id}: " + ", ".join(f"{k}={v}" for k, v in totals.items())

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Write each KG 2.x file as KG 3.0, backed up and receipted under 0051."""
        root = Path(ctx.corpus_root)
        totals = {k: 0 for k in COUNTERS}
        episodes: List[str] = []
        refused: List[str] = []
        receipts: List[Dict[str, str]] = []
        for kg_path, writes, c in find_changes(root):
            rel = str(kg_path.relative_to(root))
            for k in COUNTERS:
                totals[k] += c[k]
            if c["refused_invalid"]:
                refused.append(rel)
                continue
            episodes.append(rel)
            if ctx.dry_run:
                continue
            for path, payload in writes.items():
                receipts.append(write_with_backup(root, BACKUP_TAG, path, payload))
        if not ctx.dry_run and receipts:
            append_receipts(root, RECEIPTS_FILE, {"migration": self.id}, receipts)
        message = f"{len(episodes)} file(s); " + ", ".join(f"{k}={v}" for k, v in totals.items())
        return MigrationResult(
            migration_id=self.id,
            applied=not ctx.dry_run,
            dry_run=ctx.dry_run,
            message=message,
            details={"episodes": episodes, "refused": refused},
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Every served KG file is KG 3.0 and valid, and no file is a view."""
        root = Path(ctx.corpus_root)
        left = invalid = 0
        for kg_path in select_served_artifacts(root, ".kg.json")[0]:
            kg = _json(kg_path)
            if not isinstance(kg, dict) or not is_v3(kg) or "adapted_from" in kg:
                left += 1
                continue
            try:
                validate_artifact(kg, strict=True)
            except ValueError:
                invalid += 1
        if left or invalid:
            return False, f"{left} served KG file(s) not KG 3.0, {invalid} invalid against KG 3.0"
        return True, "every served KG file is KG 3.0 and valid"


def dry_run_report(root: Path) -> Iterable[Dict[str, Any]]:
    """Read-only: one row per episode the migration would write, keyed by its metadata path as the
    outcome hook reads it (``meta``), with its counters and input sha256s (the plan diff)."""
    from podcast_scraper.upgrade.read_recorder import recording

    registry = load_registry()
    for kg_path in select_served_artifacts(Path(root), ".kg.json")[0]:
        with recording(Path(root)) as rec:
            found = episode_changes(kg_path, registry)
        if found is None:
            continue
        meta = kg_path.with_name(kg_path.name[: -len(".kg.json")] + ".metadata.json")
        yield {
            "meta": str(meta.relative_to(root)),
            "counts": found[1],
            "inputs": rec.inputs(),
        }

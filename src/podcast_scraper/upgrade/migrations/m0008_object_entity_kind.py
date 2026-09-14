"""0008 — stamp ``.kg.json`` envelopes as schema 2.1 (the Object entity kind, #2057).

v2.1 adds a third node type, ``Object``: a NAMED THING that is neither a person nor a body of
people — an event, place, creative work, product, podcast, book or standard — and the catch-all
for an entity whose kind the extractor omitted or gave outside the vocabulary.

WHAT THIS MIGRATION CANNOT DO, stated plainly because it is the important part:

    It cannot retroactively reclassify existing Person nodes that should be Objects.

The misclassification happened at EXTRACTION time. ``_normalize_entity_kind`` was two branches —
five organisation synonyms, then ``return "person"`` — so when the model said ``event`` or
``podcast``, that answer was coerced to ``person`` and the original value was never written to
disk. A corpus therefore records ``Person(Norman Conquest)`` with no trace of the ``event`` the
extractor actually reported. The evidence is gone; guessing it back from the name would be
inventing data, which is worse than the pollution.

So existing misclassifications are cleaned by RE-ENRICHMENT, not by this migration. Re-running KG
extraction on an episode produces Object nodes under the new prompt and classifier. This migration
does the part that is safe and mechanical: it makes a corpus declare 2.1 so readers know Object
nodes are permitted, without touching any node.

Idempotent: an artifact already at 2.1 is skipped without a write. Unparsable files are recorded
and skipped rather than failing the run (mirrors 0003/0005/0006/0007).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

from ..migration import Migration, MigrationContext, MigrationResult

#: The version this migration stamps.
_TARGET_SCHEMA_VERSION = "2.1"

#: Versions we upgrade FROM. Anything older must run 0006 first, which the registry order
#: guarantees; anything newer is left alone rather than downgraded.
_UPGRADABLE_FROM = frozenset({"2.0"})


def _iter_kg_files(root: Path) -> Iterable[Path]:
    """All ``*.kg.json`` files under *root* (recursive). Stable order."""
    return sorted(root.rglob("*.kg.json"))


def _load(path: Path) -> Tuple[dict | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, exc.__class__.__name__


def _write_atomic(path: Path, payload: dict) -> None:
    """tmp + os.replace — a kill mid-write must not leave a truncated, unparsable artifact."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class ObjectEntityKindMigration(Migration):
    """Declare schema 2.1 so Object nodes are readable; reclassification needs re-enrichment."""

    id = "0008_object_entity_kind"
    to_version = "2.7.2"
    description = (
        "#2057: add the Object entity kind — a named thing that is neither a person nor an "
        "organization. Stamps .kg.json as schema 2.1 so readers accept Object nodes. Existing "
        "Person nodes that should be Objects are NOT reclassified: the extractor's original "
        "entity_kind was coerced away at write time and is not recoverable from disk, so that "
        "cleanup is a re-enrichment, not a migration"
    )

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Stamp every readable ``.kg.json`` as schema 2.1; report what could not be read.

        Only the schema stamp changes. Person nodes that SHOULD have been Objects stay Person —
        deciding that needs the extractor, so those need a re-enrich, not this migration.
        """
        files = list(_iter_kg_files(ctx.corpus_root))
        changed: List[str] = []
        unchanged = 0
        skipped_newer = 0
        unparsable: List[str] = []

        for path in files:
            payload, err = _load(path)
            if payload is None:
                unparsable.append(f"{path.name}: {err}")
                continue
            version = str(payload.get("schema_version") or "")
            if version == _TARGET_SCHEMA_VERSION:
                unchanged += 1
                continue
            if version not in _UPGRADABLE_FROM:
                # Older than 2.0 (0006 handles it, and runs first) or newer than we know about.
                skipped_newer += 1
                continue
            payload["schema_version"] = _TARGET_SCHEMA_VERSION
            changed.append(str(path.relative_to(ctx.corpus_root)))
            if not ctx.dry_run:
                _write_atomic(path, payload)

        message = (
            f"{'would stamp' if ctx.dry_run else 'stamped'} {len(changed)} artifact(s) as "
            f"schema {_TARGET_SCHEMA_VERSION}; {unchanged} already-current, "
            f"{skipped_newer} not upgradable from here, {len(unparsable)} unparsable. "
            "Existing Person nodes that should be Objects need a re-enrich, not this migration"
        )
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=message,
            details={
                "artifacts_scanned": len(files),
                "changed": len(changed),
                "already_current": unchanged,
                "not_upgradable": skipped_newer,
                "unparsable": unparsable[:20],
            },
        )

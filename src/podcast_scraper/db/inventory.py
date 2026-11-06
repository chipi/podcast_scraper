"""The file inventory of an app data dir (ADR-169): who owns every file under ``APP_DATA_DIR``.

``db import --from-files`` walks the whole directory and classifies each file as

- ``imported``: an importer in ``db.importers`` reads it into the database;
- ``file``: still a file, its store not yet migrated (declared in ``_file_claims`` below for the
  platform's own stores, and by each installed extension's ``app_data_files`` for its own);
- ``ignored``: a lock file;
- ``retired``: data a removed store left behind; no code reads it, it stays with the frozen files
  and is not imported;
- ``ops``: a backup or receipt an ops script in ``scripts/ops/`` writes for its own undo;
- ``unclaimed``: matches nothing declared. That is a store nobody told the importer about, which
  would be lost by a cut-over, so it fails the command.

Names come from the stores' own constants (never a catch-all glob). When a store migrates, its
entries move from ``_file_claims`` (or its extension's list) to its importer's ``claims()``.
A file only an extension that is not installed would claim is ``unclaimed``: the inventory covers
what this process can see.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Kind = Literal["imported", "file", "ignored", "retired", "ops"]

#: Per-user files of stores that were removed; the commit that removed each store is the owner.
RETIRED_USER_FILES = {
    "ranking_events.jsonl": "ranking_telemetry (removed in 6c6c08c6, #2327)",
}


@dataclass(frozen=True)
class Claim:
    """One declared file shape: a regex over the path relative to the data dir (posix)."""

    pattern: re.Pattern[str]
    kind: Kind
    owner: str


def user_file(name: str, kind: Kind, owner: str) -> Claim:
    """A claim for ``users/<id>/<name>``."""
    return Claim(re.compile(rf"users/[^/]+/{re.escape(name)}"), kind, owner)


def root_file(name: str, kind: Kind, owner: str) -> Claim:
    """A claim for a file directly in the data dir."""
    return Claim(re.compile(re.escape(name)), kind, owner)


def dir_files(dirname: str, file_regex: str, kind: Kind, owner: str) -> Claim:
    """A claim for files of one shape directly under ``<dirname>/``."""
    return Claim(re.compile(rf"{re.escape(dirname)}/{file_regex}"), kind, owner)


def _file_claims() -> list[Claim]:
    """The platform's own files that are still files, by owning store."""
    # Imported here: the server modules import the database layer.
    from podcast_scraper.server import app_push_store
    from podcast_scraper.server.routes import app_profile

    claims: list[Claim] = []

    # Avatars stay files until object storage (E12; the operator, 2026-11-05).
    claims.extend(
        Claim(re.compile(rf"users/[^/]+/avatar\.{re.escape(ext)}"), "file", "avatar")
        for ext, _media in app_profile._ALLOWED.values()
    )

    # An installed extension's store. When the extension is installed its importer claims the
    # file as imported, and importer claims are matched first (declared_claims); in an image
    # without it the file, if present, stays a file instead of failing as UNCLAIMED.
    claims.append(root_file("ranking_config.json", "file", "ranking_config (player)"))

    claims.extend(user_file(n, "retired", owner) for n, owner in RETIRED_USER_FILES.items())
    # A one-off manual prune of dead push tokens kept the old file beside the live one (one prod
    # account, 2026-10-07). No code writes it; only this exact shape is accepted.
    claims.append(
        Claim(
            re.compile(
                rf"users/[^/]+/{re.escape(app_push_store._FILE_NAME)}\.bak-dead-tokens-\d{{14}}"
            ),
            "ops",
            "manual push-token prune",
        )
    )
    return claims


def _extension_claims() -> list[Claim]:
    """What every installed extension declares for the files it keeps."""
    from podcast_scraper.extensions import load_extensions

    return [claim for ext in load_extensions() for claim in ext.app_data_files()]


_LOCK = Claim(re.compile(r"(?:.*/)?[^/]*\.lock"), "ignored", "lock")


def declared_claims() -> list[Claim]:
    """Every declared file shape: imported, still a file (the platform's and the installed
    extensions'), ignored."""
    from podcast_scraper.db.importers import import_claims

    return [*import_claims(), *_file_claims(), *_extension_claims(), _LOCK]


@dataclass
class Inventory:
    """The result of classifying every file under a data dir."""

    #: kind -> count; ``unclaimed`` is its own key.
    by_class: Counter[str] = field(default_factory=Counter)
    #: (kind, owner) -> count
    by_owner: Counter[tuple[str, str]] = field(default_factory=Counter)
    unclaimed: list[str] = field(default_factory=list)


def scan(data_dir: Path, claims: list[Claim] | None = None) -> Inventory:
    """Classify every file under ``data_dir``."""
    declared = claims if claims is not None else declared_claims()
    out = Inventory()
    for dirpath, _dirs, files in os.walk(data_dir):
        for fname in files:
            rel = (Path(dirpath) / fname).relative_to(data_dir).as_posix()
            hit = next((c for c in declared if c.pattern.fullmatch(rel)), None)
            if hit is None:
                out.by_class["unclaimed"] += 1
                out.unclaimed.append(rel)
            else:
                out.by_class[hit.kind] += 1
                out.by_owner[(hit.kind, hit.owner)] += 1
    out.unclaimed.sort()
    return out

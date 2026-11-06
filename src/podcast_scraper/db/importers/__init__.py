"""One importer per store (ADR-169): files in, rows out, verified record for record.

``db import --from-files`` runs the registered importers and then walks the whole data dir
(``db.inventory``) so a file no importer and no "still a file" entry claims is reported. A store's
PR registers its importer here and moves its files from the inventory's "still a file" table to
its importer's ``claims()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from sqlalchemy import Engine

from podcast_scraper.db.inventory import Claim


@dataclass
class Mismatch:
    """One record that did not import or does not verify: what the file says vs the database."""

    record: str
    detail: str


@dataclass
class StoreReport:
    store: str
    read: int = 0
    inserted: int = 0
    already_present: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)


class Importer(Protocol):
    name: str

    def claims(self) -> list[Claim]:
        """The files this importer reads, for the inventory."""
        ...

    def run(self, data_dir: Path, engine: Engine, *, dry_run: bool) -> StoreReport:
        """Read the store's files exactly as its store module does, insert idempotently, verify.

        ``engine`` is bound to the target plane's schema. With ``dry_run`` nothing is written; the
        report says what would be inserted and verifies what is already there.
        """
        ...


def registry(plane: str | None = None) -> dict[str, Importer]:
    """Every importer, in run order: users first, since the per-user stores reference them, then
    the installed modules' (ADR-172), only those whose module runs on ``plane`` when it is given.
    """
    from podcast_scraper.db.importers.kernel import per_user_importers, table_importers
    from podcast_scraper.db.importers.users import UsersImporter
    from podcast_scraper.extensions import load_extensions

    ordered: list[Importer] = [UsersImporter(), *per_user_importers(), *table_importers()]
    for ext in load_extensions():
        module = ext.database
        if module is not None and (plane is None or plane in module.planes):
            ordered.extend(module.importers())
    found: dict[str, Importer] = {}
    for imp in ordered:
        if imp.name in found:
            raise ValueError(f"two importers are named {imp.name!r}")
        found[imp.name] = imp
    return found


def import_claims() -> list[Claim]:
    return [claim for imp in registry().values() for claim in imp.claims()]

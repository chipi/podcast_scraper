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


def registry() -> dict[str, Importer]:
    from podcast_scraper.db.importers.users import UsersImporter

    return {imp.name: imp for imp in (UsersImporter(),)}


def import_claims() -> list[Claim]:
    return [claim for imp in registry().values() for claim in imp.claims()]

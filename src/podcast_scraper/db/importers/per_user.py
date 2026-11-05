"""Importers for the per-user JSON stores (ADR-169): ``users/<id>/<file>`` into one table.

Each store differs only in how its file is read (exactly as the file-based store read it), which
rows it becomes, and how a row reads back for the comparison. The run is the same for all: read
every file, refuse ids that are not user ids and files whose account has no ``users`` row (the
foreign key would refuse them anyway), insert what is missing, then compare every record with what
the database now holds.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, select, Table
from sqlalchemy.dialects.postgresql import insert as pg_insert

from podcast_scraper.db.importers import Mismatch, StoreReport
from podcast_scraper.db.inventory import Claim, user_file
from podcast_scraper.db.schema import users
from podcast_scraper.server.app_user_store import is_safe_user_id

#: What a file holds once read: the value the store's reads would return, or ``None`` when the
#: file is unreadable (reported, never imported as empty).
Reader = Callable[[Path], Any]


class Unreadable(Exception):
    """The file exists but cannot be read the way its store reads it."""


def known_accounts(data_dir: Path, engine: Engine, *, dry_run: bool) -> set[str]:
    """The account ids a per-user row may name: the ``users`` rows, plus, in a dry run (where the
    users importer inserted nothing), every user dir with a ``profile.json`` it would insert."""
    with engine.connect() as conn:
        accounts = set(conn.execute(select(users.c.user_id)).scalars())
    users_dir = data_dir / "users"
    if dry_run and users_dir.is_dir():
        accounts |= {
            child.name
            for child in users_dir.iterdir()
            if is_safe_user_id(child.name) and (child / "profile.json").is_file()
        }
    return accounts


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Unreadable(str(exc)) from exc


@dataclass
class PerUserImporter:
    """``users/<id>/<file_name>`` into ``table``, one store per instance."""

    name: str
    file_name: str
    table: Table
    #: The file's content as the store reads it; raises ``Unreadable``.
    read: Reader
    #: The rows one user's value becomes (``user_id`` is added).
    to_rows: Callable[[Any], list[dict[str, Any]]]
    #: The database's rows for one user, back as the value the store's reads return.
    from_rows: Callable[[list[dict[str, Any]]], Any]
    #: How a user's rows are ordered when read back.
    order_by: str | None = None

    def claims(self) -> list[Claim]:
        """The files this importer reads."""
        return [user_file(self.file_name, "imported", self.name)]

    def _read_files(self, data_dir: Path, report: StoreReport) -> dict[str, Any]:
        found: dict[str, Any] = {}
        users_dir = data_dir / "users"
        if not users_dir.is_dir():
            return found
        for child in sorted(users_dir.iterdir()):
            path = child / self.file_name
            if not child.is_dir() or not path.is_file():
                continue
            report.read += 1
            if not is_safe_user_id(child.name):
                report.mismatches.append(Mismatch(child.name, "not a user id; not importable"))
                continue
            try:
                found[child.name] = self.read(path)
            except Unreadable as exc:
                report.mismatches.append(Mismatch(child.name, f"{self.file_name}: {exc}"))
        return found

    def _stored(self, engine: Engine) -> dict[str, Any]:
        query = select(self.table)
        if self.order_by is not None:
            query = query.order_by(self.table.c[self.order_by])
        by_user: dict[str, list[dict[str, Any]]] = {}
        with engine.connect() as conn:
            for row in conn.execute(query).mappings():
                by_user.setdefault(row["user_id"], []).append(dict(row))
        return {uid: self.from_rows(rows) for uid, rows in by_user.items()}

    def run(self, data_dir: Path, engine: Engine, *, dry_run: bool) -> StoreReport:
        """Import every user's file, then verify each against its rows."""
        report = StoreReport(store=self.name)
        found = self._read_files(data_dir, report)
        accounts = known_accounts(data_dir, engine, dry_run=dry_run)
        stored = self._stored(engine)

        to_insert: list[str] = []
        for uid in found:
            if uid not in accounts:
                report.mismatches.append(Mismatch(uid, "no account row; import users first"))
            elif uid in stored:
                report.already_present += 1
            else:
                to_insert.append(uid)

        if dry_run:
            report.inserted = len(to_insert)
        else:
            with engine.begin() as conn:
                for uid in to_insert:
                    rows = [{**row, "user_id": uid} for row in self.to_rows(found[uid])]
                    if rows:
                        conn.execute(pg_insert(self.table).on_conflict_do_nothing(), rows)
                    report.inserted += 1
            stored = self._stored(engine)

        refused = {m.record for m in report.mismatches}
        for uid, value in found.items():
            if uid in refused:
                continue
            if uid not in stored:
                if dry_run and uid in to_insert:
                    continue
                if self.to_rows(value):
                    report.mismatches.append(Mismatch(uid, "missing from the database"))
                continue
            if stored[uid] != value:
                report.mismatches.append(Mismatch(uid, f"file={value!r} database={stored[uid]!r}"))
        return report

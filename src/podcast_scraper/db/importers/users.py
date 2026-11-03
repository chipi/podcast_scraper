"""The ``users`` importer: ``users/<id>/profile.json`` into the ``users`` table (ADR-169)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from podcast_scraper.db.importers import Mismatch, StoreReport
from podcast_scraper.db.inventory import Claim, user_file
from podcast_scraper.db.schema import users
from podcast_scraper.server import app_roles
from podcast_scraper.server.app_user_store import (
    is_safe_user_id,
    row_to_user,
    User,
    user_to_row,
)

PROFILE_FILE = "profile.json"


def read_profile(path: Path, user_id: str) -> User | None:
    """Read one profile exactly as the file-based store's ``get_user`` did; None when absent or
    unreadable. Ids are taken verbatim from the directory name, never re-derived."""
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    return User(
        user_id=user_id,
        email=str(doc.get("email", "")),
        name=str(doc.get("name", "")),
        # Profiles written before Area E have no ``username``.
        username=str(doc.get("username", "")),
        image=(str(doc["image"]) if doc.get("image") else None),
        provider=str(doc.get("provider", "")),
        subject=str(doc.get("subject", "")),
        disabled=bool(doc.get("disabled", False)),
        # Profiles written before #1128 have no ``role`` → default to listener.
        role=app_roles.normalize_role(doc.get("role")),
        # Profiles written before #2265 have no ``analytics_id``.
        analytics_id=str(doc.get("analytics_id", "")),
        # Profiles written before #1471 have no ``mcp_access`` → the default.
        mcp_access=bool(doc.get("mcp_access", True)),
    )


def _diff(file_user: User, db_user: User) -> list[str]:
    return [
        f"{name}: file={getattr(file_user, name)!r} database={getattr(db_user, name)!r}"
        for name in User.__dataclass_fields__
        if getattr(file_user, name) != getattr(db_user, name)
    ]


class UsersImporter:
    """Imports ``users/<id>/profile.json`` into the ``users`` table."""

    name = "users"

    def claims(self) -> list[Claim]:
        """The files this importer reads: ``users/<id>/profile.json``."""
        return [user_file(PROFILE_FILE, "imported", self.name)]

    def run(self, data_dir: Path, engine: Engine, *, dry_run: bool) -> StoreReport:
        """Import every profile, then verify each record field by field against its row."""
        report = StoreReport(store=self.name)
        users_dir = data_dir / "users"
        found: dict[str, tuple[User, datetime]] = {}
        if users_dir.is_dir():
            for child in sorted(users_dir.iterdir()):
                profile = child / PROFILE_FILE
                if not child.is_dir() or not profile.is_file():
                    continue
                report.read += 1
                if not is_safe_user_id(child.name):
                    report.mismatches.append(
                        Mismatch(child.name, "directory name is not a user id; not importable")
                    )
                    continue
                user = read_profile(profile, child.name)
                if user is None:
                    report.mismatches.append(Mismatch(child.name, "profile.json is unreadable"))
                    continue
                stamp = datetime.fromtimestamp(profile.stat().st_mtime, tz=timezone.utc)
                found[child.name] = (user, stamp)

        with engine.connect() as conn:
            present = {r["user_id"]: row_to_user(r) for r in conn.execute(select(users)).mappings()}
        self._collisions(found, present, report)

        to_insert = [
            uid
            for uid in found
            if uid not in present and uid not in {m.record for m in report.mismatches}
        ]
        report.already_present = sum(1 for uid in found if uid in present)
        if dry_run:
            report.inserted = len(to_insert)
        else:
            with engine.begin() as conn:
                for uid in to_insert:
                    user, stamp = found[uid]
                    values: dict[str, Any] = {
                        **user_to_row(user),
                        "created_at": stamp,
                        "updated_at": stamp,
                    }
                    inserted = conn.execute(
                        pg_insert(users)
                        .values(**values)
                        .on_conflict_do_nothing(index_elements=[users.c.user_id])
                        .returning(users.c.user_id)
                    )
                    report.inserted += 1 if inserted.first() is not None else 0

        self._verify(found, to_insert, engine, report, dry_run=dry_run)
        return report

    @staticmethod
    def _collisions(
        found: dict[str, tuple[User, datetime]], present: dict[str, User], report: StoreReport
    ) -> None:
        """Two different ids that claim one handle or one identity would fail the UNIQUE
        constraints; say so by id instead of failing half-way."""
        owner: dict[tuple[str, str], str] = {}
        for uid, user in present.items():
            if user.username:
                owner[("username", user.username)] = uid
            owner[("identity", f"{user.provider}\x00{user.subject}")] = uid
        for uid, (user, _stamp) in found.items():
            if uid in present:
                continue
            keys = [("identity", f"{user.provider}\x00{user.subject}")]
            if user.username:
                keys.append(("username", user.username))
            for key in keys:
                other = owner.get(key)
                if other is not None and other != uid:
                    report.mismatches.append(
                        Mismatch(uid, f"{key[0]} already belongs to {other}; not inserted")
                    )
                    break
            else:
                for key in keys:
                    owner[key] = uid

    @staticmethod
    def _verify(
        found: dict[str, tuple[User, datetime]],
        to_insert: list[str],
        engine: Engine,
        report: StoreReport,
        *,
        dry_run: bool,
    ) -> None:
        """Record for record: every file user that should be in the database equals its row."""
        already_reported = {m.record for m in report.mismatches}
        with engine.connect() as conn:
            rows = {r["user_id"]: row_to_user(r) for r in conn.execute(select(users)).mappings()}
        for uid, (user, _stamp) in found.items():
            if uid in already_reported:
                continue
            stored = rows.get(uid)
            if stored is None:
                if dry_run and uid in to_insert:
                    continue
                report.mismatches.append(Mismatch(uid, "missing from the database after import"))
                continue
            diffs = _diff(user, stored)
            if diffs:
                report.mismatches.append(Mismatch(uid, "; ".join(diffs)))

"""The second moment's importers (ADR-169, E02 PR 2): every remaining kernel store.

Each reads its files exactly as the file-based store read them (the reading code is kept here,
since the stores no longer read files), inserts idempotently, and verifies record for record.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, select, Table
from sqlalchemy.dialects.postgresql import insert as pg_insert

from podcast_scraper.db.importers import Mismatch, StoreReport
from podcast_scraper.db.importers.per_user import (
    known_accounts,
    PerUserImporter,
    read_json,
    Unreadable,
)
from podcast_scraper.db.inventory import Claim, dir_files, root_file, user_file
from podcast_scraper.db.schema import (
    access_policy,
    app_releases,
    apple_tokens,
    audit_log,
    comms,
    graph_events,
    magic_link_requests,
    magic_link_used,
    outbox,
    push_subscriptions,
    user_preferences,
)
from podcast_scraper.server import (
    app_access_store,
    app_comms_store,
    app_graph_telemetry,
    app_magic_link,
    app_outbox_store,
    app_push_store,
    app_release_store,
    app_user_preferences,
)
from podcast_scraper.server.app_account_deletion import _APPLE_TOKEN_FILE
from podcast_scraper.server.app_user_store import is_safe_user_id


def _epoch(value: Any) -> datetime:
    return datetime.fromtimestamp(int(value), tz=timezone.utc)


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


# --- per-user JSON documents ---------------------------------------------------------------------


def _read_object(path: Path) -> dict[str, Any]:
    doc = read_json(path)
    if not isinstance(doc, dict):
        raise Unreadable(f"expected an object, found {type(doc).__name__}")
    return doc


def _read_subscriptions(path: Path) -> list[dict[str, Any]]:
    doc = read_json(path)
    if not isinstance(doc, list):
        raise Unreadable(f"expected a list, found {type(doc).__name__}")
    # As ``list_subscriptions`` read the file: entries without an endpoint were invisible.
    return [s for s in doc if isinstance(s, dict) and s.get("endpoint")]


def _read_apple_token(path: Path) -> str:
    token = _read_object(path).get("refresh_token")
    if not token:
        raise Unreadable("no refresh_token")
    return str(token)


def _comms_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    # The store reads every file through ``_merged``; that document is what is stored.
    return [app_comms_store.row_values(app_comms_store._merged(raw))]


def _comms_back(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = rows[0]
    return app_comms_store._merged(
        app_comms_store.stored_document(row["settings"], row["unsubscribe_ref"])
    )


def per_user_importers() -> list[PerUserImporter]:
    return [
        PerUserImporter(
            name="user_preferences",
            file_name=app_user_preferences._FILE_NAME,
            table=user_preferences,
            read=_read_object,
            to_rows=lambda payload: [{"payload": payload}],
            from_rows=lambda rows: rows[0]["payload"],
        ),
        PerUserImporter(
            name="push",
            file_name=app_push_store._FILE_NAME,
            table=push_subscriptions,
            read=_read_subscriptions,
            to_rows=lambda subs: [{"endpoint": s["endpoint"], "subscription": s} for s in subs],
            from_rows=lambda rows: [r["subscription"] for r in rows],
            order_by="seq",
        ),
        PerUserImporter(
            name="apple_tokens",
            file_name=_APPLE_TOKEN_FILE,
            table=apple_tokens,
            read=_read_apple_token,
            to_rows=lambda token: [{"refresh_token": token}],
            from_rows=lambda rows: rows[0]["refresh_token"],
        ),
        PerUserImporter(
            name="comms",
            file_name=app_comms_store._FILE_NAME,
            table=comms,
            read=lambda path: app_comms_store._merged(_read_object(path)),
            to_rows=_comms_rows,
            from_rows=_comms_back,
        ),
    ]


# --- importers over a whole table ----------------------------------------------------------------


class _TableImporter:
    """Shared run: ``_read`` the files into keyed records, insert the missing keys, compare all.

    ``_read`` returns ``{key: (values_to_insert, comparable)}``; ``_stored`` the database's
    ``{key: comparable}``. A table that already holds rows is treated key by key, so a re-run
    inserts nothing and verifies everything.
    """

    name = ""
    table: Table

    def _read(self, data_dir: Path, report: StoreReport) -> dict[str, tuple[dict[str, Any], Any]]:
        raise NotImplementedError

    def _stored(self, engine: Engine) -> dict[str, Any]:
        raise NotImplementedError

    def _conflict_target(self) -> list[Any] | None:
        return None

    def run(self, data_dir: Path, engine: Engine, *, dry_run: bool) -> StoreReport:
        report = StoreReport(store=self.name)
        found = self._read(data_dir, report)
        present = self._stored(engine)
        refused = {m.record for m in report.mismatches}
        to_insert = [k for k in found if k not in present and k not in refused]
        report.already_present = sum(1 for k in found if k in present)
        if dry_run:
            report.inserted = len(to_insert)
        else:
            with engine.begin() as conn:
                for key in to_insert:
                    stmt = pg_insert(self.table).values(**found[key][0])
                    target = self._conflict_target()
                    stmt = (
                        stmt.on_conflict_do_nothing(index_elements=target)
                        if target
                        else stmt.on_conflict_do_nothing()
                    )
                    conn.execute(stmt)
                    report.inserted += 1
            present = self._stored(engine)
        for key, (_values, comparable) in found.items():
            if key in refused:
                continue
            if key not in present:
                if not (dry_run and key in to_insert):
                    report.mismatches.append(Mismatch(key, "missing from the database"))
                continue
            if present[key] != comparable:
                report.mismatches.append(
                    Mismatch(key, f"file={comparable!r} database={present[key]!r}")
                )
        return report


class AccessPolicyImporter(_TableImporter):
    """``access_policy.json`` into the one ``access_policy`` row."""

    name = "access_policy"
    table = access_policy

    def claims(self) -> list[Claim]:
        return [root_file(app_access_store._FILE_NAME, "imported", self.name)]

    def _read(self, data_dir: Path, report: StoreReport) -> dict[str, tuple[dict[str, Any], Any]]:
        path = data_dir / app_access_store._FILE_NAME
        if not path.is_file():
            return {}
        report.read += 1
        try:
            policy = app_access_store.policy_from_dict(_read_object(path))
        except Unreadable as exc:
            # The store logged this and fell back to env; importing nothing would keep that
            # fallback silently, so it stops the import instead.
            report.mismatches.append(Mismatch("access_policy", f"unreadable: {exc}"))
            return {}
        values = app_access_store.policy_to_dict(policy)
        return {"access_policy": ({"id": 1, **values, "updated_at": _mtime(path)}, values)}

    def _stored(self, engine: Engine) -> dict[str, Any]:
        with engine.connect() as conn:
            row = conn.execute(select(access_policy)).mappings().first()
        if row is None:
            return {}
        return {
            "access_policy": {
                "mode": row["mode"],
                "allowed_emails": sorted(row["allowed_emails"]),
                "allowed_domains": sorted(row["allowed_domains"]),
            }
        }


class ReleasesImporter(_TableImporter):
    """``app_releases.json`` (and the older ``player_release.json``) into ``shared.app_releases``,
    read as ``load_released_versions`` read them: the newer file first, the older one only for a
    player entry the newer lacks; malformed entries were invisible and stay out."""

    name = "releases"
    table = app_releases

    def claims(self) -> list[Claim]:
        return [
            root_file(app_release_store.RELEASES_FILE, "imported", self.name),
            root_file(app_release_store.LEGACY_PLAYER_FILE, "imported", self.name),
        ]

    def _conflict_target(self) -> list[Any] | None:
        return [app_releases.c.app]

    def _read(self, data_dir: Path, report: StoreReport) -> dict[str, tuple[dict[str, Any], Any]]:
        versions: dict[str, tuple[str, datetime]] = {}
        current = data_dir / app_release_store.RELEASES_FILE
        legacy = data_dir / app_release_store.LEGACY_PLAYER_FILE
        for path in (current, legacy):
            if path.is_file():
                report.read += 1
        raw = self._doc(current, report).get("versions")
        for app, value in raw.items() if isinstance(raw, dict) else ():
            if (
                isinstance(app, str)
                and app_release_store.valid_app_id(app)
                and isinstance(value, str)
                and app_release_store.valid_player_version(value)
            ):
                versions[app] = (value, _mtime(current))
        if app_release_store.PLAYER not in versions:
            old = self._doc(legacy, report).get("player_version")
            if isinstance(old, str) and app_release_store.valid_player_version(old):
                versions[app_release_store.PLAYER] = (old, _mtime(legacy))
        return {
            app: ({"app": app, "version": version, "updated_at": stamp}, version)
            for app, (version, stamp) in versions.items()
        }

    @staticmethod
    def _doc(path: Path, report: StoreReport) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            return _read_object(path)
        except Unreadable as exc:
            report.mismatches.append(Mismatch(path.name, f"unreadable: {exc}"))
            return {}

    def _stored(self, engine: Engine) -> dict[str, Any]:
        with engine.connect() as conn:
            return {
                app: version
                for app, version in conn.execute(
                    select(app_releases.c.app, app_releases.c.version)
                ).all()
            }


class OutboxImporter(_TableImporter):
    """``outbox/<sha256(id)>.json`` into ``outbox``, keyed by the envelope id inside each file."""

    name = "outbox"
    table = outbox

    def claims(self) -> list[Claim]:
        return [dir_files("outbox", r"[0-9a-f]{64}\.json", "imported", self.name)]

    def _conflict_target(self) -> list[Any] | None:
        return [outbox.c.envelope_id]

    def _read(self, data_dir: Path, report: StoreReport) -> dict[str, tuple[dict[str, Any], Any]]:
        found: dict[str, tuple[dict[str, Any], Any]] = {}
        directory = data_dir / "outbox"
        if not directory.is_dir():
            return found
        for path in sorted(directory.glob("*.json")):
            report.read += 1
            try:
                record = _read_object(path)
            except Unreadable as exc:
                report.mismatches.append(Mismatch(path.name, f"unreadable: {exc}"))
                continue
            envelope = record.get("envelope")
            status = record.get("status")
            envelope_id = str(envelope.get("id") or "") if isinstance(envelope, dict) else ""
            if not envelope_id or status not in ({"pending"} | app_outbox_store._TERMINAL):
                report.mismatches.append(Mismatch(path.name, "no envelope id or no known status"))
                continue
            # The store named each file by the id's hash; a file under another name was never one
            # the store could find.
            if hashlib.sha256(envelope_id.encode("utf-8")).hexdigest() != path.stem:
                report.mismatches.append(Mismatch(path.name, "file name is not the id's hash"))
                continue
            assert isinstance(envelope, dict)
            values = {
                "envelope_id": envelope_id,
                "envelope": envelope,
                "status": status,
                "detail": record.get("detail"),
                "updated_at": _epoch(record.get("updated_at") or 0),
                **app_outbox_store.envelope_columns(envelope),
            }
            found[envelope_id] = (values, (envelope, status, record.get("detail")))
        return found

    def _stored(self, engine: Engine) -> dict[str, Any]:
        with engine.connect() as conn:
            rows = conn.execute(
                select(outbox.c.envelope_id, outbox.c.envelope, outbox.c.status, outbox.c.detail)
            ).all()
        return {r.envelope_id: (r.envelope, r.status, r.detail) for r in rows}


class MagicLinkImporter(_TableImporter):
    """``magic_link_used/<token_id>.json`` (a link cannot be used twice across the switch) and
    ``magic_link_recent/<hash>.txt`` (the send throttle), each kept with the time it records."""

    name = "magic_link"
    table = magic_link_used

    def claims(self) -> list[Claim]:
        return [
            dir_files(app_magic_link._USED_DIR, r"[A-Za-z0-9_-]+\.json", "imported", self.name),
            dir_files(app_magic_link._RECENT_DIR, r"[0-9a-f]{32}\.txt", "imported", self.name),
        ]

    def run(self, data_dir: Path, engine: Engine, *, dry_run: bool) -> StoreReport:
        report = StoreReport(store=self.name)
        used: dict[str, datetime] = {}
        for path in sorted((data_dir / app_magic_link._USED_DIR).glob("*.json")):
            report.read += 1
            try:
                used[path.stem] = _epoch(_read_object(path)["used_at"])
            except (Unreadable, KeyError, TypeError, ValueError):
                # The store only ever checked that the file existed; its time is the mtime.
                used[path.stem] = _mtime(path)
        recent: dict[str, datetime] = {}
        for path in sorted((data_dir / app_magic_link._RECENT_DIR).glob("*.txt")):
            report.read += 1
            # The route compared the file's mtime with the throttle window.
            recent[path.stem] = _mtime(path)
        pairs = (
            (magic_link_used, "token_id", "used_at", used),
            (magic_link_requests, "email_hash", "requested_at", recent),
        )
        with engine.connect() as conn:
            stored = {
                table.name: dict(conn.execute(select(table.c[key], table.c[at])).all())
                for table, key, at, _ in pairs
            }
        for table, key, at, rows in pairs:
            missing = {k: v for k, v in rows.items() if k not in stored[table.name]}
            report.already_present += len(rows) - len(missing)
            if dry_run:
                report.inserted += len(missing)
                continue
            if missing:
                with engine.begin() as conn:
                    conn.execute(
                        pg_insert(table).on_conflict_do_nothing(),
                        [{key: k, at: v} for k, v in missing.items()],
                    )
                report.inserted += len(missing)
        if not dry_run:
            with engine.connect() as conn:
                stored = {
                    table.name: dict(conn.execute(select(table.c[key], table.c[at])).all())
                    for table, key, at, _ in pairs
                }
        for table, _key, _at, rows in pairs:
            for k, v in rows.items():
                got = stored[table.name].get(k)
                if got is None and dry_run:
                    continue
                if got != v:
                    report.mismatches.append(Mismatch(f"{table.name}/{k}", f"{v} vs {got}"))
        return report


class _LogImporter:
    """An append-only JSONL log into a table. A log has no key per line, so the import is whole:
    an empty table gets every line, a table that already holds exactly the file's lines is
    already present, anything else is a mismatch to look at (never a partial append)."""

    name = ""
    table: Table

    def _files(self, data_dir: Path) -> list[tuple[str | None, Path]]:
        raise NotImplementedError

    def _row(self, owner: str | None, rec: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _parse(self, line: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def _stored(self, engine: Engine) -> list[tuple[str | None, Any]]:
        raise NotImplementedError

    def run(self, data_dir: Path, engine: Engine, *, dry_run: bool) -> StoreReport:
        report = StoreReport(store=self.name)
        lines: list[tuple[str | None, dict[str, Any]]] = []
        for owner, path in self._files(data_dir):
            report.read += 1
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                report.mismatches.append(Mismatch(str(path.name), f"unreadable: {exc}"))
                continue
            lines.extend((owner, rec) for rec in map(self._parse, text.splitlines()) if rec)
        expected = [(owner, rec) for owner, rec in lines]
        present = self._stored(engine)
        if present == expected:
            report.already_present = len(expected)
            return report
        if present:
            report.mismatches.append(
                Mismatch(self.name, f"table holds {len(present)} rows, files {len(expected)}")
            )
            return report
        if dry_run:
            report.inserted = len(expected)
            return report
        if expected:
            with engine.begin() as conn:
                conn.execute(self.table.insert(), [self._row(o, r) for o, r in expected])
        report.inserted = len(expected)
        if self._stored(engine) != expected:
            report.mismatches.append(Mismatch(self.name, "rows differ from the files"))
        return report


class AuditImporter(_LogImporter):
    """``audit.jsonl`` into ``audit_log``, in file order (each line already carries its ``ts``)."""

    name = "audit"
    table = audit_log

    def claims(self) -> list[Claim]:
        return [root_file("audit.jsonl", "imported", self.name)]

    def _files(self, data_dir: Path) -> list[tuple[str | None, Path]]:
        path = data_dir / "audit.jsonl"
        return [(None, path)] if path.is_file() else []

    def _parse(self, line: str) -> dict[str, Any] | None:
        try:
            rec = json.loads(line)
        except ValueError:
            return None
        return rec if isinstance(rec, dict) else None

    def _row(self, owner: str | None, rec: dict[str, Any]) -> dict[str, Any]:
        ts = rec.get("ts")
        row: dict[str, Any] = {"record": rec}
        if isinstance(ts, (int, float)) and not isinstance(ts, bool):
            row["ts"] = _epoch(ts)
        return row

    def _stored(self, engine: Engine) -> list[tuple[str | None, Any]]:
        with engine.connect() as conn:
            rows = conn.execute(select(audit_log.c.record).order_by(audit_log.c.id)).scalars()
            return [(None, r) for r in rows]


class GraphEventsImporter(_LogImporter):
    """``users/<id>/graph_events.jsonl`` (and the signed-out ``users/anon/``) into
    ``graph_events``, each user's lines in order."""

    name = "graph_events"
    table = graph_events

    def claims(self) -> list[Claim]:
        return [user_file(app_graph_telemetry.EVENTS_FILE, "imported", self.name)]

    def _files(self, data_dir: Path) -> list[tuple[str | None, Path]]:
        out: list[tuple[str | None, Path]] = []
        users_dir = data_dir / "users"
        if not users_dir.is_dir():
            return out
        anon = users_dir / app_graph_telemetry.ANON / app_graph_telemetry.EVENTS_FILE
        if anon.is_file():
            out.append((None, anon))
        for child in sorted(users_dir.iterdir()):
            path = child / app_graph_telemetry.EVENTS_FILE
            if child.name != app_graph_telemetry.ANON and path.is_file():
                out.append((child.name, path))
        return out

    def run(self, data_dir: Path, engine: Engine, *, dry_run: bool) -> StoreReport:
        accounts = known_accounts(data_dir, engine, dry_run=dry_run)
        orphans = [
            owner
            for owner, _path in self._files(data_dir)
            if owner is not None and (not is_safe_user_id(owner) or owner not in accounts)
        ]
        if orphans:
            report = StoreReport(store=self.name)
            report.mismatches.extend(
                Mismatch(owner, "no account row; import users first") for owner in orphans
            )
            return report
        return super().run(data_dir, engine, dry_run=dry_run)

    def _parse(self, line: str) -> dict[str, Any] | None:
        return app_graph_telemetry.parse_line(line)

    def _row(self, owner: str | None, rec: dict[str, Any]) -> dict[str, Any]:
        return {"user_id": owner, "event": rec}

    def _stored(self, engine: Engine) -> list[tuple[str | None, Any]]:
        with engine.connect() as conn:
            rows = conn.execute(
                select(graph_events.c.user_id, graph_events.c.event).order_by(
                    graph_events.c.user_id.nulls_first(), graph_events.c.id
                )
            ).all()
        return [(r.user_id, r.event) for r in rows]


def table_importers() -> list[Any]:
    return [
        AccessPolicyImporter(),
        ReleasesImporter(),
        OutboxImporter(),
        MagicLinkImporter(),
        AuditImporter(),
        GraphEventsImporter(),
    ]

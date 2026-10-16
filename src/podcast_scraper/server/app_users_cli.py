"""Operator CLI for consumer-platform user administration (#1064).

Runs directly against the per-user files (no server needed)::

    python -m podcast_scraper.server.app_users_cli list   --data-dir <dir>
    python -m podcast_scraper.server.app_users_cli disable --data-dir <dir> <user_id>
    python -m podcast_scraper.server.app_users_cli enable  --data-dir <dir> <user_id>
    python -m podcast_scraper.server.app_users_cli delete  --data-dir <dir> <user_id>   # GDPR
    python -m podcast_scraper.server.app_users_cli export  --data-dir <dir> <user_id>
    python -m podcast_scraper.server.app_users_cli grant-mcp-all --data-dir <dir> [--apply]

Folding this under a top-level ``podcast app-users`` subcommand is a thin follow-up.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from podcast_scraper.server.app_account_deletion import delete_account
from podcast_scraper.server.app_user_store import (
    get_user,
    list_users,
    set_disabled,
    set_mcp_access,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="app-users", description="Consumer-platform user admin.")
    parser.add_argument(
        "--data-dir", required=True, type=Path, help="Per-user data dir (APP_DATA_DIR)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="List users.")
    for name in ("disable", "enable", "delete", "export"):
        node = sub.add_parser(name, help=f"{name} a user by id.")
        node.add_argument("user_id")
    grant = sub.add_parser(
        "grant-mcp-all",
        help="Give MCP access to every user without it, except deliberate admin revokes.",
    )
    grant.add_argument("--apply", action="store_true", help="Write; without it, a dry run.")
    return parser


def _revoked_by_admin(data_dir: Path) -> set[str]:
    """Users whose latest ``admin.user.mcp_access`` audit entry is a revoke.

    MCP is on by default since 2026-10-16, but profiles written before then hold ``false`` for
    everyone who was never granted it. The audit log (``audit.jsonl``, #1071) is the only record
    that tells a deliberate revoke apart from that old default, so it decides who is skipped.
    """
    latest: dict[str, bool] = {}
    path = data_dir / "audit.jsonl"
    if not path.exists():
        return set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("action") == "admin.user.mcp_access" and rec.get("user"):
            latest[str(rec["user"])] = bool(rec.get("mcp_access"))
    return {uid for uid, granted in latest.items() if not granted}


def _grant_mcp_all(data_dir: Path, apply: bool) -> int:
    revoked = _revoked_by_admin(data_dir)
    granted = skipped = already = 0
    for user in list_users(data_dir):
        if user.mcp_access:
            already += 1
            continue
        if user.user_id in revoked:
            skipped += 1
            print(f"skip    {user.user_id}  (revoked by an admin)")
            continue
        if apply:
            set_mcp_access(data_dir, user.user_id, True)
        granted += 1
        print(f"{'granted' if apply else 'would grant'} {user.user_id}")
    mode = "" if apply else " (dry run; pass --apply)"
    print(
        f"# {granted} granted, {already} already had it, {skipped} skipped{mode}", file=sys.stderr
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the user-admin command; return a process exit code."""
    args = _build_parser().parse_args(argv)
    data_dir: Path = args.data_dir

    if args.cmd == "list":
        users = list_users(data_dir)
        for user in users:
            flag = " [disabled]" if user.disabled else ""
            print(f"{user.user_id}\t{user.email}\t{user.name}{flag}")
        print(f"# {len(users)} user(s)", file=sys.stderr)
        return 0
    if args.cmd == "grant-mcp-all":
        return _grant_mcp_all(data_dir, args.apply)

    uid: str = args.user_id
    if args.cmd in ("disable", "enable"):
        ok = set_disabled(data_dir, uid, args.cmd == "disable")
        print(f"{args.cmd}d {uid}" if ok else f"unknown user {uid}")
        return 0 if ok else 1
    if args.cmd == "delete":
        # The full purge (#2273), not just the directory. No Apple token revocation from the CLI —
        # it has no provider config; the self-service and admin routes do that.
        record = get_user(data_dir, uid)
        if record is None:
            print(f"unknown user {uid}")
            return 1
        report = delete_account(data_dir, record)
        print(f"deleted {uid} ({report.outbox_envelopes} outbox envelope(s) removed)")
        return 0
    if args.cmd == "export":
        record = get_user(data_dir, uid)
        if record is None:
            print(f"unknown user {uid}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "user_id": record.user_id,
                    "email": record.email,
                    "name": record.name,
                    "provider": record.provider,
                    "subject": record.subject,
                    "disabled": record.disabled,
                },
                indent=2,
            )
        )
        return 0
    return 2  # unreachable (argparse requires a subcommand)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

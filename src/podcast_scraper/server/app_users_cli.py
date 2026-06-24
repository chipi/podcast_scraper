"""Operator CLI for consumer-platform user administration (#1064).

Runs directly against the per-user files (no server needed)::

    python -m podcast_scraper.server.app_users_cli list   --data-dir <dir>
    python -m podcast_scraper.server.app_users_cli disable --data-dir <dir> <user_id>
    python -m podcast_scraper.server.app_users_cli enable  --data-dir <dir> <user_id>
    python -m podcast_scraper.server.app_users_cli delete  --data-dir <dir> <user_id>   # GDPR
    python -m podcast_scraper.server.app_users_cli export  --data-dir <dir> <user_id>

Folding this under a top-level ``podcast app-users`` subcommand is a thin follow-up.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from podcast_scraper.server.app_user_store import (
    delete_user,
    get_user,
    list_users,
    set_disabled,
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
    return parser


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

    uid: str = args.user_id
    if args.cmd in ("disable", "enable"):
        ok = set_disabled(data_dir, uid, args.cmd == "disable")
        print(f"{args.cmd}d {uid}" if ok else f"unknown user {uid}")
        return 0 if ok else 1
    if args.cmd == "delete":
        ok = delete_user(data_dir, uid)
        print(f"deleted {uid}" if ok else f"unknown user {uid}")
        return 0 if ok else 1
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

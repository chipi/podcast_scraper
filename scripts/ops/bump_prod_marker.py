#!/usr/bin/env python3
"""Update the prod-state marker + pinned fixture after a successful prod deploy (#1176).

Invoked by ``.github/workflows/deploy-prod.yml`` on a green deploy. Writes:

- ``config/last_deployed_prod_version.json`` — code version, SHA, deploy date,
  and the migration ids in the registry at HEAD (== what just went live).
- ``tests/fixtures/upgrade/corpus_at_last_prod_release/upgrade_ledger.json`` —
  the ledger, listing every registered migration at its declared ``to_version``.
- ``tests/fixtures/upgrade/corpus_at_last_prod_release/corpus_manifest.json`` —
  ``produced_by.code_version`` = the deployed code version.

What this script does NOT touch: the fixture's on-disk artifact shapes
(``.gi.json``, ``.kg.json``, ``.metadata.json``). If a migration in this deploy
changed those shapes, the operator (or a follow-up commit) must update them
so ``tests/unit/upgrade/test_pinned_fixture_shape.py`` passes. That test is the
loud signal that reminds you.

Usage (from the repo root):

    python scripts/ops/bump_prod_marker.py \
        --code-version 2.7.1 \
        --sha 0123456789abcdef

The script is idempotent: running it twice with the same inputs is a no-op.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, doc: dict) -> None:
    """Write pretty-printed, trailing-newlined, key-sorted JSON to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _registry_snapshot() -> list[tuple[str, str, str]]:
    """(id, to_version, description) for every currently-registered migration.

    Imported lazily so the script fails gracefully with a clear error when run
    from an environment that cannot import the package.
    """
    try:
        from podcast_scraper.upgrade.registry import get_migrations
    except ImportError as exc:
        print(
            f"ERROR: could not import podcast_scraper.upgrade.registry: {exc}\n"
            "Run this script from the repo root with the project installed "
            "(pip install -e '.' or via .venv).",
            file=sys.stderr,
        )
        sys.exit(2)
    return [(m.id, m.to_version, m.description) for m in get_migrations()]


def _now_utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_marker(
    root: Path, *, code_version: str, sha: str, deployed_at: str, applied: list[str]
) -> None:
    doc = {
        "code_version": code_version,
        "sha": sha,
        "deployed_at": deployed_at,
        "applied_migrations": applied,
        "note": (
            "This marker tracks what is currently running on prod. Auto-updated by "
            "scripts/ops/bump_prod_marker.py (invoked by .github/workflows/deploy-prod.yml). "
            "See docs/guides/CORPUS_UPGRADE.md → 'Prod-state fixture maintenance' (#1176)."
        ),
    }
    _write_json(root / "config" / "last_deployed_prod_version.json", doc)


def _write_fixture_ledger(
    root: Path, registry: list[tuple[str, str, str]], deployed_at_iso: str
) -> None:
    """Build the fixture's ledger from the current registry snapshot."""
    fixture_dir = root / "tests" / "fixtures" / "upgrade" / "corpus_at_last_prod_release"
    applied_entries = [
        {"id": mid, "to_version": to_ver, "at": deployed_at_iso} for mid, to_ver, _desc in registry
    ]
    # ``version`` on the ledger tracks the last migration's to_version.
    version = registry[-1][1] if registry else ""
    doc = {"applied": applied_entries, "version": version}
    _write_json(fixture_dir / "upgrade_ledger.json", doc)


def _write_fixture_manifest(root: Path, code_version: str) -> None:
    fixture_dir = root / "tests" / "fixtures" / "upgrade" / "corpus_at_last_prod_release"
    manifest_path = fixture_dir / "corpus_manifest.json"
    manifest = _load_json(manifest_path)
    manifest.setdefault("produced_by", {})["code_version"] = code_version
    _write_json(manifest_path, manifest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--code-version", required=True, help="Deployed code version (e.g. 2.7.1 or 2.7.0-rc3)."
    )
    ap.add_argument("--sha", required=True, help="Full or short git SHA of the deployed commit.")
    ap.add_argument(
        "--deployed-at",
        default=_now_utc_date(),
        help="Deploy date YYYY-MM-DD UTC (default: today).",
    )
    ap.add_argument(
        "--ledger-at",
        default=_now_utc_iso(),
        help=(
            "Timestamp used for each ledger entry's ``at`` field " "(default: now UTC, RFC-3339)."
        ),
    )
    args = ap.parse_args(argv)

    root = _repo_root()
    registry = _registry_snapshot()
    applied = [mid for mid, _to, _desc in registry]

    _write_marker(
        root,
        code_version=args.code_version,
        sha=args.sha,
        deployed_at=args.deployed_at,
        applied=applied,
    )
    _write_fixture_ledger(root, registry, args.ledger_at)
    _write_fixture_manifest(root, args.code_version)

    print(
        f"OK: bumped prod marker → code_version={args.code_version}, sha={args.sha}, "
        f"applied_migrations={applied}"
    )
    print("NEXT: run `.venv/bin/python -m pytest tests/unit/upgrade/test_pinned_fixture_shape.py`")
    print("      to catch fixture on-disk-shape drift the marker+ledger bump cannot repair.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

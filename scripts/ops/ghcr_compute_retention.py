#!/usr/bin/env python3
"""Compute GHCR container version keep/delete sets (#802).

Uses ``gh api`` (GitHub REST). Dry-run by default; pass ``--apply`` to DELETE.

Keep set per image:
  - Latest 20 by ``created_at``
  - Any version tagged with ``v*`` (release)
  - Any version tagged ``main``
  - Versions referenced by the last 5 successful ``deploy-prod.yml`` runs
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Set, Tuple

OWNER = "chipi"
IMAGES = (
    "podcast-scraper-stack-api",
    "podcast-scraper-stack-viewer",
    "podcast-scraper-stack-pipeline-llm",
    "podcast-scraper-stack-pipeline-ml",
)
KEEP_RECENT = 20
DEPLOY_RUNS = 5
SHA_TAG_RE = re.compile(r"^sha-[a-f0-9]{7,40}$", re.I)
RELEASE_TAG_RE = re.compile(r"^v", re.I)


def _gh_json(args: List[str]) -> Any:
    proc = subprocess.run(
        ["gh", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "gh failed")
    return json.loads(proc.stdout or "null")


def _list_versions(image: str) -> List[Dict[str, Any]]:
    """Return every published version of *image* — empty list when the
    package has never been published (HTTP 404).

    The IMAGES tuple lists every container our deploy flow MAY push;
    some (e.g. ``podcast-scraper-stack-pipeline-ml``) aren't actually
    published yet because the corresponding deploy variant hasn't run.
    Treat "package not found" as "nothing to keep / nothing to delete"
    instead of failing the whole retention pass.
    """
    try:
        data = _gh_json(
            [
                "api",
                f"/users/{OWNER}/packages/container/{image}/versions",
                "--paginate",
            ]
        )
    except RuntimeError as exc:
        if "Package not found" in str(exc) or "404" in str(exc):
            print(
                f"# warn: package {image!r} not published yet; skipping",
                file=sys.stderr,
            )
            return []
        raise
    if not isinstance(data, list):
        return []
    return [v for v in data if isinstance(v, dict)]


def _tags_for_version(version: Dict[str, Any]) -> Set[str]:
    tags: Set[str] = set()
    meta = version.get("metadata") or {}
    container = meta.get("container") or {}
    for tag in container.get("tags") or []:
        if isinstance(tag, str) and tag.strip():
            tags.add(tag.strip())
    return tags


def _deploy_sha_tags() -> Set[str]:
    runs = _gh_json(
        [
            "run",
            "list",
            "--workflow",
            "deploy-prod.yml",
            "--status",
            "success",
            "--limit",
            str(DEPLOY_RUNS),
            "--json",
            "databaseId,headSha,displayTitle",
        ]
    )
    out: Set[str] = set()
    if not isinstance(runs, list):
        return out
    for run in runs:
        if not isinstance(run, dict):
            continue
        sha = str(run.get("headSha") or "")
        if len(sha) >= 7:
            out.add(f"sha-{sha[:7]}")
    return out


def _version_sort_key(version: Dict[str, Any]) -> datetime:
    raw = str(version.get("created_at") or "")
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def compute_keep_ids(
    versions: List[Dict[str, Any]], *, deploy_shas: Set[str]
) -> Tuple[Set[int], Dict[str, List[str]]]:
    keep: Set[int] = set()
    reasons: Dict[str, List[str]] = {}

    def mark(vid: int, reason: str) -> None:
        keep.add(vid)
        reasons.setdefault(reason, []).append(str(vid))

    sorted_versions = sorted(versions, key=_version_sort_key, reverse=True)
    for v in sorted_versions[:KEEP_RECENT]:
        vid = int(v["id"])
        mark(vid, f"recent_{KEEP_RECENT}")

    for v in versions:
        vid = int(v["id"])
        tags = _tags_for_version(v)
        if "main" in tags:
            mark(vid, "tag_main")
        if any(RELEASE_TAG_RE.match(t) for t in tags):
            mark(vid, "tag_release")
        if any(t in deploy_shas for t in tags if SHA_TAG_RE.match(t)):
            mark(vid, "deploy_prod_recent")

    return keep, reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="GHCR retention keep/delete planner (#802)")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete versions not in the keep set (default: dry-run only)",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    args = parser.parse_args()

    deploy_shas = _deploy_sha_tags()
    summary: Dict[str, Any] = {"deploy_sha_tags": sorted(deploy_shas), "images": {}}

    exit_code = 0
    for image in IMAGES:
        versions = _list_versions(image)
        keep_ids, reasons = compute_keep_ids(versions, deploy_shas=deploy_shas)
        delete_ids = [int(v["id"]) for v in versions if int(v["id"]) not in keep_ids]
        summary["images"][image] = {
            "total": len(versions),
            "keep": len(keep_ids),
            "delete": len(delete_ids),
            "keep_reasons": {k: len(v) for k, v in reasons.items()},
        }
        print(f"{image}: total={len(versions)} keep={len(keep_ids)} delete={len(delete_ids)}")
        if not args.apply:
            continue
        for vid in delete_ids:
            _gh_json(
                [
                    "api",
                    "-X",
                    "DELETE",
                    f"/users/{OWNER}/packages/container/{image}/versions/{vid}",
                ]
            )
            print(f"  deleted {image} version id={vid}")

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))

    if not args.apply:
        print("Dry-run only — re-run with --apply to delete the complement.")
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

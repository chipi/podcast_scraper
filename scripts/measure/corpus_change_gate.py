#!/usr/bin/env python3
"""Measure one migration on a copy of a snapshot clone and decide its class (the change gate).

    python scripts/measure/corpus_change_gate.py --migration 0031_named_voices_shown \\
        --clone <snapshot clone> --work <empty dir for the copy> \\
        --out docs/wip/changes/0031_named_voices_shown.gate.json

The clone is never written: it is copied to ``--work``, the scoreboard is taken on the clone, the
migration is applied to the copy alone (nothing else from the registry), its ``verify`` is read, the
scoreboard is taken again, and every file whose bytes changed is listed. The class is FULL when a
written file is a content file or changed a content key (``change_gate.classify_written``), or when
any gating scoreboard number moved (``change_gate.gating_moves``); otherwise LIGHT. The report is
what the landing check reads; a record that says light while this says full cannot land.

Bookkeeping the upgrade runner writes for every migration (``.podcast_scraper/``, the receipts
``*.jsonl`` at the corpus root) is not counted as the migration's output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from podcast_scraper.quality.change_gate import (  # noqa: E402
    classify_written,
    gating_moves,
    load_config,
)
from podcast_scraper.quality.corpus_scoreboard import score_corpus  # noqa: E402
from podcast_scraper.upgrade import outcome_hook  # noqa: E402
from podcast_scraper.upgrade.migration import MigrationContext  # noqa: E402
from podcast_scraper.upgrade.registry import get_migrations  # noqa: E402
from podcast_scraper.upgrade.state import FilesystemStateStore  # noqa: E402


def _hashes(root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in root.rglob("*"):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(root).as_posix()
        if rel.startswith(".podcast_scraper/") or ("/" not in rel and rel.endswith(".jsonl")):
            continue
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _verdicts_sha(migration: Any) -> Optional[str]:
    mod = importlib.import_module(type(migration).__module__)
    vf = getattr(mod, "verdicts_file", None)
    return (
        hashlib.sha256(vf().read_bytes()).hexdigest() if callable(vf) and vf().is_file() else None
    )


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def _read(path: Path) -> Optional[bytes]:
    return path.read_bytes() if path.is_file() else None


def _verify_all(
    mid: str, applied: Any, clone: Path, work: Path
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """``upgrade verify`` as prod runs it (runbook step 5): every migration in the clone's ledger
    plus this one, on the copy after the apply. A failing check is run again on the untouched
    clone, so a failure this migration caused (2026-10-27: 0035 left 0030's ad cut stale on 2
    episodes, seen only on prod) is told apart from one the snapshot already had. Report only;
    the exit code is unchanged."""
    out: List[Dict[str, Any]] = []
    regressions: List[str] = []
    for m in get_migrations():
        if m.id not in applied and m.id != mid:
            continue
        ok, msg = m.verify(MigrationContext(corpus_root=work))
        row: Dict[str, Any] = {"id": m.id, "ok": ok, "message": msg}
        if not ok and m.id != mid:
            before_ok, before_msg = m.verify(MigrationContext(corpus_root=clone))
            row["on_clone"] = {"ok": before_ok, "message": before_msg}
            if before_ok:
                regressions.append(m.id)
        out.append(row)
    return out, regressions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--migration", required=True, help="registered migration id")
    ap.add_argument("--clone", required=True, type=Path, help="snapshot clone (never written)")
    ap.add_argument(
        "--work", required=True, type=Path, help="where the copy is made (must not exist)"
    )
    ap.add_argument("--out", required=True, type=Path, help="gate report JSON")
    ap.add_argument("--snapshot", required=True, help="the snapshot id the clone was made from")
    args = ap.parse_args()

    config = load_config(REPO)
    migration = next((m for m in get_migrations() if m.id == args.migration), None)
    if migration is None:
        print(f"not registered: {args.migration}", file=sys.stderr)
        return 2
    if args.work.exists():
        print(f"--work must not exist: {args.work}", file=sys.stderr)
        return 2
    # The outcome records the migrations the clone held; its ledger must say so truthfully (a
    # migration applied directly, as this script applies, is not in the ledger by itself).
    applied = FilesystemStateStore(args.clone).applied_migration_ids()
    earlier = [m.id for m in get_migrations() if m.id < migration.id and m.id not in applied]
    if earlier:
        print(f"the clone's ledger lacks {earlier}: record them first", file=sys.stderr)
        return 2
    shutil.copytree(args.clone, args.work, symlinks=True)
    print(f"copied {args.clone} -> {args.work}", flush=True)
    before_hash = _hashes(args.work)
    before = score_corpus(args.clone)["total"]
    before_ep = outcome_hook.score_by_episode(args.work)
    try:
        plan = outcome_hook.episode_plan(migration, args.work)
    except outcome_hook.HookError as exc:
        plan = []
        print(f"no comparable plan: {exc}", file=sys.stderr)
    result = migration.apply(MigrationContext(corpus_root=args.work))
    ok, message = migration.verify(MigrationContext(corpus_root=args.work))
    verify_all, regressions = _verify_all(migration.id, applied, args.clone, args.work)
    after = score_corpus(args.work)["total"]
    after_ep = outcome_hook.score_by_episode(args.work)
    after_hash = _hashes(args.work)
    changed = sorted(
        rel
        for rel in set(before_hash) | set(after_hash)
        if before_hash.get(rel) != after_hash.get(rel)
    )
    written = [(rel, _read(args.clone / rel), _read(args.work / rel)) for rel in changed]
    file_class, reasons = classify_written(written, config)
    moves = gating_moves(before, after, config)
    computed = "full" if file_class == "full" or moves else "light"
    by_suffix: Dict[str, int] = {}
    for rel in changed:
        name = rel.split("/")[-1]
        parts = name.split(".")
        suffix = "." + ".".join(parts[-2:]) if len(parts) > 2 else "." + parts[-1]
        by_suffix[suffix] = by_suffix.get(suffix, 0) + 1
    report = {
        "migration": args.migration,
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "computed_class": computed,
        "why": (reasons[:20] + [f"gating number moved: {k} {v:+d}" for k, v in moves.items()])
        or ["no content file or key written, no gating number moved"],
        "gating_moves": moves,
        "scoreboard_before": before,
        "scoreboard_after": after,
        "files_changed": len(changed),
        "files_changed_by_suffix": dict(sorted(by_suffix.items())),
        "apply_message": result.message,
        "verify": {"ok": ok, "message": message},
        "verify_all": verify_all,
        "verify_regressions": regressions,
        "clone": str(args.clone),
        "code": {"sha": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain"))},
        "snapshot": args.snapshot,
        "applied_before": sorted(applied),
        "code_digest": outcome_hook.migration_digest(migration),
        "verdicts_sha256": _verdicts_sha(migration),
        "episode_deltas": outcome_hook.deltas(before_ep, after_ep, config["gating_metrics"]),
        "plan": plan,
        "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("computed_class", "gating_moves", "files_changed")}))
    failing = [v["id"] for v in verify_all if not v["ok"]]
    print(f"upgrade verify, all applied: {len(verify_all) - len(failing)} ok, failing {failing}")
    if regressions:
        print(f"VERIFY REGRESSIONS (ok on the clone, failing after this migration): {regressions}")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

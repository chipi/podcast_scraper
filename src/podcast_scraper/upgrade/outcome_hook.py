"""The migration outcome hook: a corpus migration applies on prod only as it was measured.

``docs/guides/CORPUS_CHANGE_METHOD.md`` D5 and D15, step 0: a migration of the change regime (id
0031 on) cannot apply without the outcome of its clone run shipped with the code, and prod's own
per-episode plan must equal the clone's, every difference listed with its cause. Criteria:
``docs/wip/changes/migration_outcome_hook.acceptance.md``.

For each regime migration ``upgrade run`` (not a dry run), under the corpus lock:

1. **static checks** (``check_static``): the shipped outcome ``upgrade/outcomes/<id>.json`` exists,
   has every field, carries a GO / GO WITH CONDITIONS verdict, was measured on clean code, and its
   code digest equals the installed code's (the migration module and every ``podcast_scraper``
   module it imports, plus its verdicts file); the migration is the first pending one and prod's
   applied set equals the clone's. Any failure: ``HookRefused``, nothing written.
2. **plan** (``plan_diff``): prod's per-episode plan (``dry_run_report``, rows with every counter
   and their input files' sha256) against the shipped one. Any difference: ``HookStopped``, with
   every difference written to ``.podcast_scraper/upgrade-outcomes/<id>.plan-diff.json``, the one
   file a stopped run writes. ``--accept-plan-diff <file>`` applies only when the file names this
   migration and outcome, gives a cause for every entry, and equals the recomputed diff exactly.
3. **around the apply** (``measure``): the scoreboard per episode before and after; with no plan
   difference every gating metric's change must equal the clone's, per episode; on accepted
   differing episodes no gating metric may rise. A failure is ``HookOutcomeFailed``: the apply STAYS
   and is in the ledger (no automatic undo), the undo is named, no later migration runs. The receipt
   ``.podcast_scraper/upgrade-outcomes/<id>.json`` is written on every path past the apply.

Migrations below the regime run exactly as before. There is no flag that skips the hook.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.resources
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

REGIME = "0031"
EXIT_REFUSED = 3
EXIT_STOPPED = 4
EXIT_OUTCOME = 5
GO = ("GO", "GO WITH CONDITIONS")
REQUIRED = (
    "migration",
    "snapshot",
    "applied_before",
    "code_digest",
    "code_dirty",
    "verdicts_sha256",
    "scoreboard_before",
    "scoreboard_after",
    "episode_deltas",
    "plan",
    "computed_class",
    "verdict",
    "conditions",
)
OUT_DIR = Path(".podcast_scraper") / "upgrade-outcomes"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # src/podcast_scraper


class HookError(Exception):
    """A hook decision; ``code`` is the CLI's exit code."""

    code = 1


class HookRefused(HookError):
    """Refused before any write (exit 3): outcome, code, verdict or run not the measured one."""

    code = EXIT_REFUSED


class HookStopped(HookError):
    """Stopped on a plan difference (exit 4): only the plan-diff file was written."""

    code = EXIT_STOPPED


class HookOutcomeFailed(HookError):
    """Applied, but the outcome is not the measured one or could not be measured (exit 5)."""

    code = EXIT_OUTCOME


def in_regime(migration_id: str) -> bool:
    """True for a migration the hook governs: id 0031 and later."""
    return migration_id[:4] >= REGIME


# ---------------------------------------------------------------- the shipped outcome


def outcome_path(migration_id: str) -> Path:
    """Where the installed package keeps the migration's shipped outcome (package data)."""
    return (
        Path(str(importlib.resources.files("podcast_scraper.upgrade")))
        / "outcomes"
        / (f"{migration_id}.json")
    )


def load_outcome(migration_id: str, path: Optional[Path] = None) -> Tuple[Dict[str, Any], str]:
    """``(outcome, its sha256)``; ``HookRefused`` naming what is wrong."""
    p = path or outcome_path(migration_id)
    if not p.is_file():
        raise HookRefused(f"{migration_id}: no shipped outcome {p.name} (method D5)")
    raw = p.read_bytes()
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise HookRefused(f"{migration_id}: outcome unreadable: {exc}") from exc
    if not isinstance(doc, dict):
        raise HookRefused(f"{migration_id}: outcome is not an object")
    missing = [k for k in REQUIRED if k not in doc]
    if missing:
        raise HookRefused(f"{migration_id}: outcome lacks {', '.join(missing)}")
    if doc["migration"] != migration_id:
        raise HookRefused(f"{migration_id}: outcome is for {doc['migration']!r}")
    if doc["verdict"] not in GO:
        raise HookRefused(f"{migration_id}: outcome verdict is {doc['verdict']!r}, not GO")
    if doc["code_dirty"]:
        raise HookRefused(f"{migration_id}: outcome was measured on uncommitted code")
    return doc, hashlib.sha256(raw).hexdigest()


def _imports(path: Path, package: str) -> Set[str]:
    """``podcast_scraper`` modules a source file imports (absolute and relative)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names if a.name.startswith("podcast_scraper")}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            if mod.startswith("podcast_scraper"):
                out.add(mod)
                out |= {f"{mod}.{a.name}" for a in node.names}
    return out


def _module_file(name: str) -> Optional[Path]:
    parts = name.split(".")[1:]
    if not parts:  # the package itself
        return PACKAGE_ROOT / "__init__.py"
    rel = Path(*parts)
    for cand in (PACKAGE_ROOT / rel.with_suffix(".py"), PACKAGE_ROOT / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def module_closure(module_name: str) -> Dict[str, Path]:
    """``{module: file}``: the module and every ``podcast_scraper`` module it reaches by import.

    Importing ``a.b.c`` runs ``a/__init__.py`` and ``a/b/__init__.py`` first, so every parent
    package is in the closure too, with what it imports."""
    seen: Dict[str, Path] = {}
    todo = [module_name]
    while todo:
        name = todo.pop()
        f = _module_file(name)
        if f is None or name in seen:
            continue
        seen[name] = f
        parts = name.split(".")
        todo.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        pkg = name if f.name == "__init__.py" else name.rsplit(".", 1)[0]
        todo.extend(_imports(f, pkg) - set(seen))
    return seen


def code_digest(module_name: str, extra: Optional[List[Path]] = None) -> str:
    """sha256 over the migration module and every ``podcast_scraper`` module it reaches by import,
    sorted by path, plus ``extra`` files (its verdicts file): the code the outcome was measured on.
    """
    h = hashlib.sha256()
    files = sorted(
        {p.resolve() for p in module_closure(module_name).values()}
        | {p.resolve() for p in extra or []}
    )
    for f in files:
        try:
            rel = str(f.relative_to(PACKAGE_ROOT.parent))
        except ValueError:
            rel = f.name
        h.update(rel.encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def migration_digest(migration: Any) -> str:
    """The migration's code digest as installed now, with the plan-level inputs it reads: its
    verdicts file and its frozen list (``list_file``), each when it has one."""
    mod = importlib.import_module(type(migration).__module__)
    extra = []
    for hook in ("verdicts_file", "list_file"):
        fn = getattr(mod, hook, None)
        if callable(fn) and fn().is_file():
            extra.append(fn())
    return code_digest(type(migration).__module__, extra)


# ---------------------------------------------------------------- plan


def episode_plan(migration: Any, root: Path) -> List[Dict[str, Any]]:
    """The migration's per-episode plan, normalised and sorted (``HookRefused`` without one)."""
    mod = importlib.import_module(type(migration).__module__)
    report: Optional[Callable[[Path], Any]] = getattr(mod, "dry_run_report", None)
    if report is None:
        raise HookRefused(f"{migration.id}: no dry_run_report: its plan cannot be compared")
    rows = []
    for r in report(Path(root)):
        rows.append(json.loads(json.dumps(r, sort_keys=True, default=str)))
    if any("inputs" not in r or "counts" not in r for r in rows):
        raise HookRefused(f"{migration.id}: plan rows lack 'counts' or 'inputs' (input sha256)")
    return sorted(rows, key=lambda r: str(r["meta"]))


def plan_diff(clone: List[Dict[str, Any]], prod: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every difference: an episode in one plan only, or a row that differs (counters, inputs)."""
    a = {str(r["meta"]): r for r in clone}
    b = {str(r["meta"]): r for r in prod}
    out: List[Dict[str, Any]] = []
    for meta in sorted(set(a) | set(b)):
        if meta not in b:
            out.append({"meta": meta, "kind": "clone_only"})
        elif meta not in a:
            out.append({"meta": meta, "kind": "prod_only"})
        elif a[meta] != b[meta]:
            keys = sorted(
                k for k in set(a[meta]) | set(b[meta]) if a[meta].get(k) != b[meta].get(k)
            )
            out.append({"meta": meta, "kind": "differs", "fields": keys})
    return out


def check_accept(
    accept: Dict[str, Any], migration_id: str, outcome_sha: str, diff: List[Dict[str, Any]]
) -> None:
    """The operator's accepted plan difference: for this migration, this outcome, every entry with
    a cause, and exactly the difference prod's plan produces now (``HookStopped`` otherwise)."""
    if accept.get("migration") != migration_id or accept.get("outcome_sha256") != outcome_sha:
        raise HookStopped(f"{migration_id}: the accept file is for another migration or outcome")
    if not str(accept.get("operator_go") or "").strip():
        raise HookStopped(f"{migration_id}: the accept file carries no operator go")
    entries = accept.get("entries") or []
    if any(not str(e.get("cause") or "").strip() for e in entries):
        raise HookStopped(f"{migration_id}: an accepted difference has no cause")

    def key(e: Dict[str, Any]) -> str:
        return json.dumps({k: e.get(k) for k in ("meta", "kind", "fields")}, sort_keys=True)

    if sorted(key(e) for e in entries) != sorted(key(e) for e in diff):
        raise HookStopped(f"{migration_id}: prod's plan difference is not the accepted one")


# ---------------------------------------------------------------- scoreboard


def score_by_episode(root: Path) -> Dict[str, Dict[str, int]]:
    """``{meta path relative to root: counts}`` over the served episodes."""
    from ..quality.corpus_scoreboard import score_episode
    from .corpus_selection import select_served_artifacts

    out: Dict[str, Dict[str, int]] = {}
    for meta in select_served_artifacts(Path(root), ".metadata.json")[0]:
        got = score_episode(meta)
        if got is not None:
            out[str(meta.relative_to(root))] = {k: int(v) for k, v in got["counts"].items()}
    return out


def deltas(
    before: Dict[str, Dict[str, int]], after: Dict[str, Dict[str, int]], metrics: List[str]
) -> Dict[str, Dict[str, int]]:
    """``{episode: {metric: after - before}}``, nonzero only."""
    out: Dict[str, Dict[str, int]] = {}
    for ep in sorted(set(before) | set(after)):
        d = {
            m: after.get(ep, {}).get(m, 0) - before.get(ep, {}).get(m, 0)
            for m in metrics
            if after.get(ep, {}).get(m, 0) != before.get(ep, {}).get(m, 0)
        }
        if d:
            out[ep] = d
    return out


def compare(
    prod: Dict[str, Dict[str, int]],
    clone: Dict[str, Dict[str, int]],
    differing: Set[str],
    metrics: List[str],
) -> List[str]:
    """Failures: a matched episode whose change is not the clone's, or a rise on a differing one."""
    fails = []
    for ep in sorted(set(prod) | set(clone)):
        p, c = prod.get(ep, {}), clone.get(ep, {})
        if ep in differing:
            fails += [f"{ep}: {m} rose by {p[m]}" for m in metrics if p.get(m, 0) > 0]
        else:
            fails += [
                f"{ep}: {m} changed {p.get(m, 0):+d}, the clone's {c.get(m, 0):+d}"
                for m in metrics
                if p.get(m, 0) != c.get(m, 0)
            ]
    return fails


def gating_metrics() -> List[str]:
    """The method's gating metrics (the change gate's config when present, else its own list)."""
    from ..quality.change_gate import CONFIG_PATH

    cfg = PACKAGE_ROOT.parents[1] / CONFIG_PATH
    if cfg.is_file():
        return list(json.loads(cfg.read_text(encoding="utf-8"))["gating_metrics"])
    return [
        "S1_hidden_unknown",
        "S1n_named_but_hidden",
        "S1c_inconsistent",
        "S2_hidden",
        "S3_principal_unidentified",
        "S10a_zero_visible",
        "S10b_zero_visible_panel",
        "S12_quotes_off_text",
        "A1_marker_hits",
    ]


# ---------------------------------------------------------------- the run


@dataclass
class Prepared:
    """What the static and plan checks established for one regime migration."""

    migration: Any
    outcome: Dict[str, Any]
    outcome_sha: str
    diff: List[Dict[str, Any]] = field(default_factory=list)
    accept: Optional[Dict[str, Any]] = None
    accept_sha: Optional[str] = None


def check_static(migration: Any, pending_ids: List[str], applied: Set[str]) -> Tuple[Dict, str]:
    """Outcome, verdict, clean code, code digest, first pending, applied set."""
    outcome, sha = load_outcome(migration.id)
    if pending_ids and pending_ids[0] != migration.id:
        raise HookRefused(f"{migration.id}: run {pending_ids[0]} first, in its own run")
    if set(outcome["applied_before"]) != set(applied):
        missing = sorted(set(outcome["applied_before"]) - set(applied))
        extra = sorted(set(applied) - set(outcome["applied_before"]))
        raise HookRefused(
            f"{migration.id}: prod's applied migrations differ from the clone's "
            f"(missing {missing}, extra {extra})"
        )
    digest = migration_digest(migration)
    if digest != outcome["code_digest"]:
        raise HookRefused(f"{migration.id}: the installed code is not the code that was measured")
    return outcome, sha


def prepare(
    migration: Any,
    root: Path,
    pending_ids: List[str],
    applied: Set[str],
    accept_path: Optional[Path] = None,
    write: bool = True,
) -> Prepared:
    """Static checks and the plan diff, before any snapshot or content write."""
    outcome, sha = check_static(migration, pending_ids, applied)
    prod = episode_plan(migration, root)
    diff = plan_diff(outcome["plan"], prod)
    prepared = Prepared(migration, outcome, sha, diff)
    if diff:
        if accept_path is None:
            if write:
                out = Path(root) / OUT_DIR / f"{migration.id}.plan-diff.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    json.dumps(
                        {"migration": migration.id, "outcome_sha256": sha, "entries": diff},
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            lines = "\n".join(f"  {e['kind']:10} {e['meta']} {e.get('fields', '')}" for e in diff)
            raise HookStopped(
                f"{migration.id}: prod's plan differs from the clone's in {len(diff)} episode(s); "
                f"each needs a cause (--accept-plan-diff):\n{lines}"
            )
        try:
            raw = Path(accept_path).read_bytes()
            accept = json.loads(raw)
        except (OSError, ValueError) as exc:
            raise HookStopped(
                f"{migration.id}: the accept file {accept_path} is unreadable ({exc})"
            ) from exc
        if not isinstance(accept, dict):
            raise HookStopped(f"{migration.id}: the accept file {accept_path} is not a JSON object")
        check_accept(accept, migration.id, sha, diff)
        prepared.accept = accept
        prepared.accept_sha = hashlib.sha256(raw).hexdigest()
    return prepared


def measure(
    prepared: Prepared,
    root: Path,
    apply: Callable[[], Any],
    snapshot: Optional[str],
) -> Any:
    """Score, apply, score, compare, write the receipt; ``HookOutcomeFailed`` on a mismatch.

    ``apply`` runs the migration AND records it in the ledger: an outcome failure leaves both."""
    mig = prepared.migration
    metrics = gating_metrics()
    try:
        before = score_by_episode(root)
    except Exception as exc:  # noqa: BLE001 - nothing applied yet: refuse
        raise HookRefused(f"{mig.id}: the before-scoreboard failed: {exc}") from exc
    result = apply()
    receipt: Dict[str, Any] = {
        "migration": mig.id,
        "outcome_sha256": prepared.outcome_sha,
        "code_digest": prepared.outcome["code_digest"],
        "snapshot": snapshot or "none",
        "plan_diff": prepared.diff,
        "accept_sha256": prepared.accept_sha,
        "accept": prepared.accept,
        "clone_deltas": prepared.outcome["episode_deltas"],
    }
    path = Path(root) / OUT_DIR / f"{mig.id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        after = score_by_episode(root)
    except Exception as exc:  # noqa: BLE001 - applied: say so, never pretend it passed
        receipt.update(verdict="NOT MEASURED", error=str(exc))
        path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        raise HookOutcomeFailed(
            f"{mig.id}: applied, but the after-scoreboard failed ({exc}); {undo_hint(mig)}"
        ) from exc
    prod = deltas(before, after, metrics)
    differing = {e["meta"] for e in prepared.diff}
    fails = compare(prod, prepared.outcome["episode_deltas"], differing, metrics)
    total_after = {m: sum(e.get(m, 0) for e in after.values()) for m in metrics}
    if total_after.get("S1c_inconsistent", 0):
        fails.append(f"S1c is {total_after['S1c_inconsistent']} after, not 0")
    receipt.update(
        verdict="FAIL" if fails else "PASS",
        prod_deltas=prod,
        totals_before={m: sum(e.get(m, 0) for e in before.values()) for m in metrics},
        totals_after=total_after,
        failures=fails,
    )
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if fails:
        raise HookOutcomeFailed(
            f"{mig.id}: applied, and the outcome is not the measured one ({len(fails)}): "
            f"{fails[:5]}; {undo_hint(mig)}"
        )
    return result


def _write_marks(migration: Any, root: Path) -> Tuple[int, bool]:
    """``(receipts file size, backup dir exists)``: what an apply that writes leaves behind."""
    receipts = getattr(migration, "_receipts_file", None) or getattr(
        importlib.import_module(type(migration).__module__), "RECEIPTS_FILE", None
    )
    tag = getattr(migration, "_backup_tag", None) or getattr(
        importlib.import_module(type(migration).__module__), "BACKUP_TAG", None
    )
    size = (
        (Path(root) / receipts).stat().st_size
        if receipts and (Path(root) / receipts).is_file()
        else -1
    )
    backups = (
        bool(tag) and (Path(root) / ".podcast_scraper" / "upgrade-backups" / str(tag)).is_dir()
    )
    return size, backups


def apply_empty(migration: Any, root: Path, apply: Callable[[], Any]) -> Any:
    """A regime migration whose plan here has no episode: it must write nothing (criteria R2)."""
    before = _write_marks(migration, root)
    result = apply()
    after = _write_marks(migration, root)
    written = (getattr(result, "details", None) or {}).get("episodes") or []
    if after != before or written:
        raise HookOutcomeFailed(
            f"{migration.id}: its plan was empty but the apply wrote "
            f"({len(written)} episode(s); receipts/backups {before} -> {after}); "
            f"{undo_hint(migration)}"
        )
    print(f"{migration.id}: plan empty: nothing to measure", file=sys.stderr)
    return result


def undo_hint(migration: Any) -> str:
    """The text printed after a failed outcome: the apply stays, and the exact undo command."""
    mod = type(migration).__module__
    return (
        f'the apply stays and is recorded; to undo: python -c "from pathlib import Path; '
        f"from {mod} import undo; print(undo(Path('<corpus root>')))\""
    )

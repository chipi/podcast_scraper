"""``make pipeline-check`` — resolve refs to checkouts, run a worker per side, compare, report.

    python -m pipeline_check.cli [--check NAME ...] [--base REF] [--candidate REF]
                                 [--locales "en,en-US"] [--feeds p01,p02] [--profile P]
                                 [--overrides '{"k": v}'] [--out DIR]

* ``--candidate`` defaults to the working tree this tool runs from; ``--base`` defaults to none
  (a single-ref health check). A ref is checked out into its own git worktree under
  ``.test_outputs/pipeline-check/worktrees/`` (kept for reuse); ``--candidate-src`` /
  ``--base-src`` take an existing ``src`` directory instead.
* ``--check`` picks checks from ``expectations.yaml`` (default: all). ``--locales`` / ``--feeds``
  replace a check's own lists (an ad-hoc run).

Exit code: 0 when every check passes (report-only checks always pass), 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import compare, report

TOOL_DIR = Path(__file__).resolve().parent
REPO = TOOL_DIR.parents[2]


def _load_expectations(path: Path) -> Dict[str, Any]:
    import yaml

    data: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


def resolve_src(ref: Optional[str], src: Optional[str], *, default_here: bool) -> Optional[Path]:
    """The ``src`` directory to run: an explicit path, a ref's worktree, or this checkout."""
    if src:
        return Path(src).resolve()
    if ref is None:
        return (REPO / "src") if default_here else None
    sha = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short=12", ref],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = REPO / ".test_outputs" / "pipeline-check" / "worktrees" / sha
    if not tree.exists():
        subprocess.run(
            ["git", "-C", str(REPO), "worktree", "add", "--detach", "-q", str(tree), sha],
            check=True,
        )
    return tree / "src"


def label(ref: Optional[str], src: Optional[Path]) -> Optional[str]:
    """``ref (sha)`` for a ref, so a stale local branch is visible in the report; else the path."""
    if src is None:
        return None
    if ref is None:
        return str(src)
    sha = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short=9", ref], capture_output=True, text=True
    ).stdout.strip()
    return f"{ref} ({sha})"


def run_worker(
    src: Path, corpus: Path, check: Dict[str, Any], args: argparse.Namespace, out: Path
) -> Dict[str, Any]:
    locales = (
        re.split(r"[ ,]+", args.locales.strip())
        if args.locales is not None
        else check.get("locales", ["en"])
    )
    locales = ["" if v == "-" else v for v in locales]
    feeds = args.feeds or ",".join(check.get("feeds", []))
    cmd = [
        sys.executable,
        "-m",
        "pipeline_check.worker",
        "--corpus",
        str(corpus),
        "--feeds",
        feeds,
        "--locales",
        json.dumps(locales),
        "--profile",
        args.profile or "",
        "--overrides",
        args.overrides,
        "--out",
        str(out),
    ]
    env = {**os.environ, "PYTHONPATH": f"{src}:{TOOL_DIR.parent}"}
    proc = subprocess.run(cmd, cwd=str(out.parent), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"worker failed for {src}:\n{proc.stderr[-3000:]}")
    print(proc.stdout.strip().splitlines()[-1])
    data: Dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    return data


def run_real_worker(
    src: Path, args: argparse.Namespace, run_dir: Path, cache: Path
) -> Dict[str, Any]:
    """One real pipeline run of *src* in its own process; logs to ``run_dir/pipeline.log``."""
    out = run_dir / "worker.json"
    cmd = [
        sys.executable,
        "-m",
        "pipeline_check.worker",
        "--real",
        "--rss",
        args.feed,
        "--max-episodes",
        str(args.max_episodes),
        "--run-dir",
        str(run_dir),
        "--transcript-cache",
        str(cache),
        "--profile",
        args.profile,
        "--overrides",
        args.overrides,
        "--out",
        str(out),
    ]
    env = {**os.environ, "PYTHONPATH": f"{src}:{TOOL_DIR.parent}"}
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "pipeline.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(run_dir), env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not out.exists():
        raise SystemExit(f"real worker failed for {src}; see {run_dir / 'pipeline.log'}")
    data: Dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    print(f"real run done: {run_dir.name} (pipeline exit {data['real']['exit_code']})")
    return data


def main_real(
    args: argparse.Namespace, spec: Dict[str, Any], candidate_src: Path, base_src: Optional[Path]
) -> int:
    """Base ``--base-runs`` times (the noise band), candidate once, one shared transcript cache."""
    from . import artifacts

    if not args.feed or not args.profile:
        raise SystemExit("REAL mode needs FEED=<rss url> and PROFILE=<profile>")
    root = args.out / "real"
    cache = root / "transcript-cache"
    real_spec = spec.get("real", {})
    bases = (
        [
            run_real_worker(base_src, args, root / f"base-{i + 1}", cache)
            for i in range(args.base_runs)
        ]
        if base_src
        else []
    )
    cand = run_real_worker(candidate_src, args, root / "candidate", cache)

    res = compare.CheckResult(
        name="real", episodes=len(cand["real"]["artifacts"]["episodes"]), variants=["real"]
    )
    target = real_spec.get("decisions_resolve_to")
    norm = cand.get("normalised", {})
    for ev in cand["events"]:
        res.decisions_per_area["Real run"] = res.decisions_per_area.get("Real run", 0) + 1
        why = compare.event_resolves(ev, target, norm) if target else None
        if why:
            res.findings.append(
                compare.Finding(
                    "hole", "Real run", f"{ev['site']} (called from {ev['caller']})", why
                )
            )
    llm = None
    if bases:
        diffs = artifacts.compare_deterministic(
            bases[0]["real"]["artifacts"],
            cand["real"]["artifacts"],
            spec.get("allowed_artifact_differences", {}),
            real_spec.get("llm_fields", []),
        )
        for d in diffs:
            res.findings.append(
                compare.Finding("base", "Real run artifacts", "base vs candidate", d)
            )
        res.stage_status["Real run"] = (
            f"{len(diffs)} unexpected artifact difference(s)"
            if diffs
            else "artifacts identical to base (allowed list excepted)"
        )
        llm = {
            "rows": artifacts.llm_rows(
                [b["real"]["artifacts"] for b in bases], cand["real"]["artifacts"]
            )
        }
    res.coverage = compare.coverage(cand)

    page = report.render(
        [res],
        base_label=label(args.base, base_src),
        candidate_label=label(args.candidate, candidate_src) or "",
        corpus=f"real feed {args.feed}",
        llm=llm,
    )
    (args.out / "report.md").write_text(page, encoding="utf-8")
    print(page)
    print(f"\nreport: {args.out / 'report.md'}")
    return 0 if res.passed else 1


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="append", default=None)
    ap.add_argument("--base")
    ap.add_argument("--candidate")
    ap.add_argument("--base-src")
    ap.add_argument("--candidate-src")
    ap.add_argument("--locales", default=None)
    ap.add_argument("--feeds", default="")
    ap.add_argument("--profile", default="")
    ap.add_argument("--overrides", default="{}")
    ap.add_argument("--expectations", type=Path, default=TOOL_DIR / "expectations.yaml")
    ap.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="fixture corpus both sides run on (default: defaults.corpus in this checkout)",
    )
    ap.add_argument("--out", type=Path, default=REPO / ".test_outputs" / "pipeline-check")
    ap.add_argument(
        "--real", action="store_true", help="real episodes through the full pipeline (DGX)"
    )
    ap.add_argument("--feed", default="", help="RSS URL for --real")
    ap.add_argument("--max-episodes", type=int, default=1)
    ap.add_argument(
        "--base-runs", type=int, default=2, help="base runs in --real (2 = a noise band)"
    )
    args = ap.parse_args(argv)

    spec = _load_expectations(args.expectations)
    corpus = (args.corpus or (REPO / spec["defaults"]["corpus"])).resolve()
    names = args.check or list(spec["checks"])
    candidate_src = resolve_src(args.candidate, args.candidate_src, default_here=True)
    base_src = resolve_src(args.base, args.base_src, default_here=False)
    assert candidate_src is not None
    args.out.mkdir(parents=True, exist_ok=True)
    if args.real:
        return main_real(args, spec, candidate_src, base_src)

    results: List[compare.CheckResult] = []
    for name in names:
        check = spec["checks"][name]
        cand = run_worker(candidate_src, corpus, check, args, args.out / f"{name}.candidate.json")
        base = (
            run_worker(base_src, corpus, check, args, args.out / f"{name}.base.json")
            if base_src
            else None
        )
        results.append(compare.evaluate(name, check, cand, base))

    page = report.render(
        results,
        base_label=label(args.base, base_src),
        candidate_label=label(args.candidate, candidate_src) or "",
        corpus=str(corpus),
    )
    (args.out / "report.md").write_text(page, encoding="utf-8")
    (args.out / "findings.json").write_text(
        json.dumps(
            {
                r.name: {"findings": [f.__dict__ for f in r.findings], "coverage": r.coverage}
                for r in results
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(page)
    print(f"\nreport: {args.out / 'report.md'}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Real-episode end-to-end validation for #644 single-feed corpus layout.

Runs the actual scraper against a real RSS feed three ways and diffs the
on-disk tree to prove that ``single_feed_uses_corpus_layout=True`` produces
the same shape as a multi-feed run.

Three runs, all with ``max_episodes=1``, ``transcribe_missing=False`` so we
exercise the scrape + directory-layout path without burning $ on cloud
transcription:

  1. Single feed, flag OFF — current default: ``<root>/run_*/``
  2. Single feed, flag ON  — expected: ``<root>/feeds/<slug>/run_*/``
  3. Multi feed (2 feeds)  — authoritative shape: ``<root>/feeds/<slug>/run_*/``

Pass gates:
  - Run 1 emits ``<root>/run_*/`` (no ``feeds/`` segment at top level)
  - Run 2 emits ``<root>/feeds/<slug>/run_*/`` exactly
  - Run 3 emits ``<root>/feeds/<slug1>/run_*/`` and ``<root>/feeds/<slug2>/run_*/``
  - Run 2's feed dir structure under ``feeds/<slug>/`` matches Run 3's exactly
    (same subdirs, same file categories)

Usage::

    .venv/bin/python scripts/validate/validate_layout_644.py

Exit code 0 on pass, 1 on fail.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = _REPO_ROOT / ".test_outputs" / "_validate_layout_644"

# Real, well-known podcast RSS feeds with short episodes. Both are stable and
# publicly available; used only for layout exercise (no content is evaluated).
FEED_A = "https://feeds.99percentinvisible.org/99percentinvisible"
FEED_B = "https://feeds.feedburner.com/planetmoneypodcast"


def _run_scrape(output_dir: Path, rss_urls: List[str], flag: bool) -> subprocess.CompletedProcess:
    """Invoke the CLI scraper with minimal work — one episode, no LLM, no transcription.

    ``single_feed_uses_corpus_layout`` has no CLI flag (RFC-follow-up noted),
    so we write a temp YAML config when flag=True.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # generate_summaries defaults to False; no --no flag exists. Skip it.
    args: List[str] = [
        str(_REPO_ROOT / ".venv" / "bin" / "python"),
        "-m",
        "podcast_scraper.cli",
        "--output-dir",
        str(output_dir),
        "--max-episodes",
        "1",
        "--no-transcribe-missing",
        "--no-generate-metadata",
        "--no-auto-speakers",
    ]

    # First URL goes as positional (populates args.rss → cfg.rss_url).
    # Extras use --rss (args.rss_extra → cfg.rss_urls via collect_feed_urls).
    args.append(rss_urls[0])
    for url in rss_urls[1:]:
        args.extend(["--rss", url])

    if flag:
        cfg_path = output_dir / "_validate_cfg.yaml"
        cfg_path.write_text("single_feed_uses_corpus_layout: true\n")
        args = args[:3] + ["--config", str(cfg_path)] + args[3:]

    return subprocess.run(args, capture_output=True, text=True, timeout=300)


def _top_level_dirs(root: Path) -> Set[str]:
    if not root.exists():
        return set()
    return {p.name for p in root.iterdir() if p.is_dir()}


def _feed_slugs_under(root: Path) -> List[str]:
    feeds_dir = root / "feeds"
    if not feeds_dir.exists():
        return []
    return sorted(p.name for p in feeds_dir.iterdir() if p.is_dir())


def _shape_under_feed(feed_dir: Path) -> Dict[str, List[str]]:
    """Return the set of subdirs and top-level files under a single feed dir,
    normalising run_* → run_<REDACTED> so runs with different timestamps
    compare equal."""
    out: Dict[str, List[str]] = {"subdirs": [], "files": []}
    if not feed_dir.exists():
        return out
    for p in feed_dir.iterdir():
        name = p.name
        if p.is_dir():
            if name.startswith("run_"):
                name = "run_<TS>"
            out["subdirs"].append(name)
        else:
            out["files"].append(name)
    out["subdirs"].sort()
    out["files"].sort()
    return out


def _describe_tree(root: Path, max_depth: int = 4) -> str:
    """Return a short description of a directory tree for human + test inspection."""
    lines: List[str] = []

    def walk(p: Path, depth: int) -> None:
        if depth > max_depth:
            return
        rel = p.relative_to(root) if p != root else Path(".")
        indent = "  " * depth
        lines.append(f"{indent}{rel}")
        if p.is_dir():
            for child in sorted(p.iterdir()):
                walk(child, depth + 1)

    walk(root, 0)
    return "\n".join(lines[:120])


def _check_singlefeed_flag_off(work_dir: Path) -> Tuple[bool, str]:
    """Expect: <root>/run_*/ at top level, NO feeds/."""
    top = _top_level_dirs(work_dir)
    has_run = any(n.startswith("run_") for n in top)
    has_feeds = "feeds" in top
    if has_run and not has_feeds:
        return True, "OK: top-level run_* dir, no feeds/"
    return False, f"FAIL: top-level dirs = {sorted(top)} (want run_*, not feeds/)"


def _check_singlefeed_flag_on(work_dir: Path) -> Tuple[bool, str]:
    """Expect: <root>/feeds/<slug>/run_*/ — no run_* at root."""
    top = _top_level_dirs(work_dir)
    has_top_run = any(n.startswith("run_") for n in top)
    has_feeds = "feeds" in top
    slugs = _feed_slugs_under(work_dir)
    if has_feeds and not has_top_run and len(slugs) == 1:
        feed_dir = work_dir / "feeds" / slugs[0]
        has_run_inside = any(p.name.startswith("run_") for p in feed_dir.iterdir())
        if has_run_inside:
            return True, f"OK: feeds/{slugs[0]}/run_* (no top-level run_*)"
        return False, f"FAIL: feeds/{slugs[0]}/ has no run_* subdir"
    return (
        False,
        f"FAIL: top={sorted(top)}, slugs={slugs} (want feeds/<one slug>/ with run_*)",
    )


def _check_multifeed(work_dir: Path) -> Tuple[bool, str]:
    """Expect: <root>/feeds/<slug1>/run_*/ and <root>/feeds/<slug2>/run_*/."""
    slugs = _feed_slugs_under(work_dir)
    if len(slugs) < 2:
        return False, f"FAIL: multi-feed expected ≥2 slugs, got {slugs}"
    for s in slugs:
        if not any(p.name.startswith("run_") for p in (work_dir / "feeds" / s).iterdir()):
            return False, f"FAIL: feeds/{s}/ has no run_*"
    return True, f"OK: {len(slugs)} feeds, each with run_*"


def _compare_feed_shapes(singlefeed_on_dir: Path, multifeed_dir: Path) -> Tuple[bool, str]:
    """Compare the per-feed structure between single-feed-with-flag and multi-feed."""
    sf_slugs = _feed_slugs_under(singlefeed_on_dir)
    mf_slugs = _feed_slugs_under(multifeed_dir)
    if not sf_slugs or not mf_slugs:
        return False, f"FAIL: missing slugs (single={sf_slugs}, multi={mf_slugs})"
    sf_shape = _shape_under_feed(singlefeed_on_dir / "feeds" / sf_slugs[0])
    mf_shape = _shape_under_feed(multifeed_dir / "feeds" / mf_slugs[0])
    if sf_shape == mf_shape:
        return True, f"OK: feed-dir shapes match ({sf_shape['subdirs']})"
    return (
        False,
        f"FAIL: shape mismatch\n  single: {sf_shape}\n  multi:  {mf_shape}",
    )


def main() -> int:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    report: Dict[str, object] = {}
    all_pass = True

    # --- Run 1: single feed, flag OFF ---
    print("\n[1/3] Single feed, flag OFF …", flush=True)
    out1 = WORK_DIR / "singlefeed_off"
    p1 = _run_scrape(out1, [FEED_A], flag=False)
    ok1, msg1 = _check_singlefeed_flag_off(out1)
    print(f"  exit={p1.returncode}  {msg1}")
    report["run1"] = {
        "exit": p1.returncode,
        "ok": ok1,
        "msg": msg1,
        "tree": _describe_tree(out1),
        "stderr": p1.stderr[-800:],
    }
    all_pass &= ok1 and p1.returncode == 0

    # --- Run 2: single feed, flag ON ---
    print("\n[2/3] Single feed, flag ON …", flush=True)
    out2 = WORK_DIR / "singlefeed_on"
    p2 = _run_scrape(out2, [FEED_A], flag=True)
    ok2, msg2 = _check_singlefeed_flag_on(out2)
    print(f"  exit={p2.returncode}  {msg2}")
    report["run2"] = {"exit": p2.returncode, "ok": ok2, "msg": msg2, "tree": _describe_tree(out2)}
    all_pass &= ok2 and p2.returncode == 0

    # --- Run 3: multi feed (authoritative shape) ---
    print("\n[3/3] Multi feed (2 feeds) …", flush=True)
    out3 = WORK_DIR / "multifeed"
    p3 = _run_scrape(out3, [FEED_A, FEED_B], flag=False)
    ok3, msg3 = _check_multifeed(out3)
    print(f"  exit={p3.returncode}  {msg3}")
    report["run3"] = {"exit": p3.returncode, "ok": ok3, "msg": msg3, "tree": _describe_tree(out3)}
    all_pass &= ok3 and p3.returncode == 0

    # --- Structural comparison: single-feed-on should equal multi-feed shape ---
    if ok2 and ok3:
        print("\n[compare] feed-dir shapes (single-flag-on vs multi-feed) …")
        ok4, msg4 = _compare_feed_shapes(out2, out3)
        print(f"  {msg4}")
        report["shape_compare"] = {"ok": ok4, "msg": msg4}
        all_pass &= ok4

    WORK_DIR.joinpath("report.json").write_text(json.dumps(report, indent=2, default=str))

    print()
    if all_pass:
        print("LAYOUT GATES: PASS")
    else:
        print("LAYOUT GATES: FAIL — see report.json")
        for run in ("run1", "run2", "run3"):
            if run in report and not report[run]["ok"]:  # type: ignore[index]
                rep = report[run]
                print(f"\n--- {run} stderr tail ---")
                # stderr was captured in subprocess but not stored; re-describe tree
                print(rep.get("tree", ""))  # type: ignore[union-attr]
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

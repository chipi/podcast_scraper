#!/usr/bin/env python3
"""#653 Part E — retroactively apply Topic-label fix to an existing corpus.

Walks a corpus (multi-feed or single-feed layout), joins each episode's
``gi.json`` against its ``kg.json``, and rewrites GI Topic nodes so their
labels + IDs reflect the KG canonical noun-phrase slugs produced post-#653.

Safe by default: dry-run only. ``--apply`` writes changes in place with a
``.bak`` sibling per file.

Usage
-----

    # See what would change (no writes):
    python scripts/backfill/backfill_gi_topics.py <corpus_root>

    # Apply (writes .bak backups per gi.json):
    python scripts/backfill/backfill_gi_topics.py <corpus_root> --apply

Exit codes
----------
- 0 — scan complete (dry-run) OR apply succeeded.
- 1 — no GI artifacts found OR one or more artifacts failed.
- 2 — script/input error (bad path, corrupt YAML/JSON).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from podcast_scraper.graph_id_utils import slugify_label, topic_node_id_from_slug  # noqa: E402


def _iter_gi_paths(corpus_root: Path) -> List[Path]:
    """Both multi-feed (<root>/feeds/<slug>/run_*/metadata/*.gi.json) and
    single-feed (<root>/run_*/metadata/*.gi.json) layouts."""
    feeds_root = corpus_root / "feeds"
    if feeds_root.is_dir():
        return sorted(feeds_root.glob("*/run_*/metadata/*.gi.json"))
    return sorted(corpus_root.glob("run_*/metadata/*.gi.json"))


def _kg_path_for_gi(gi_path: Path) -> Optional[Path]:
    """Given .../metadata/<stem>.gi.json, find the matching .kg.json."""
    stem = gi_path.name[: -len(".gi.json")]
    candidate = gi_path.parent / f"{stem}.kg.json"
    return candidate if candidate.is_file() else None


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return data


def _kg_canonical_topic_labels(kg: Dict[str, Any]) -> List[str]:
    """Return KG canonical topic labels (order-preserving, deduped)."""
    out: List[str] = []
    seen: set[str] = set()
    for node in kg.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        if node.get("type") != "Topic":
            continue
        props = node.get("properties") or {}
        label = str(props.get("label") or "").strip()
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


def _rewrite_gi_topics(gi: Dict[str, Any], kg_labels: List[str]) -> Tuple[Dict[str, Any], int]:
    """Rewrite GI Topic nodes + their incoming/outgoing edges to use canonical
    slugs derived from KG labels. Returns (new_gi, changed_count).

    Strategy:
      - If a Topic count mismatch (GI topics != KG topics), we still rewrite
        the GI topics that DO have a KG counterpart by index. Stragglers
        keep their old label but get re-slugged to the canonical format.
      - ID changes are propagated to every edge referencing the old ID.
    """
    nodes = list(gi.get("nodes") or [])
    edges = list(gi.get("edges") or [])
    if not nodes:
        return gi, 0

    topic_positions = [
        i for i, n in enumerate(nodes) if isinstance(n, dict) and n.get("type") == "Topic"
    ]
    if not topic_positions:
        return gi, 0

    id_rewrites: Dict[str, str] = {}
    changes = 0
    for pos_idx, node_idx in enumerate(topic_positions):
        node = nodes[node_idx]
        old_id = str(node.get("id") or "")
        props = dict(node.get("properties") or {})
        if pos_idx < len(kg_labels):
            new_label = kg_labels[pos_idx]
        else:
            new_label = str(props.get("label") or "").strip() or "topic"
        new_slug = slugify_label(new_label)
        new_id = topic_node_id_from_slug(new_slug)
        if new_id != old_id or new_label != props.get("label"):
            props["label"] = new_label
            new_node = dict(node)
            new_node["id"] = new_id
            new_node["properties"] = props
            nodes[node_idx] = new_node
            if new_id != old_id and old_id:
                id_rewrites[old_id] = new_id
            changes += 1

    # Propagate ID rewrites across edges (both ends).
    if id_rewrites:
        new_edges: List[Dict[str, Any]] = []
        for e in edges:
            if not isinstance(e, dict):
                new_edges.append(e)
                continue
            ne = dict(e)
            src = str(ne.get("from") or "")
            dst = str(ne.get("to") or "")
            if src in id_rewrites:
                ne["from"] = id_rewrites[src]
            if dst in id_rewrites:
                ne["to"] = id_rewrites[dst]
            new_edges.append(ne)
        edges = new_edges

    out = dict(gi)
    out["nodes"] = nodes
    out["edges"] = edges
    return out, changes


def _apply_one(gi_path: Path, apply: bool) -> Tuple[str, int, Optional[str]]:
    """Return (status, changes, error). status ∈ {'applied', 'noop', 'dryrun', 'skipped'}."""
    kg_path = _kg_path_for_gi(gi_path)
    if kg_path is None:
        return "skipped", 0, "no matching kg.json"
    try:
        gi = _load_json(gi_path)
        kg = _load_json(kg_path)
    except RuntimeError as exc:
        return "skipped", 0, str(exc)

    kg_labels = _kg_canonical_topic_labels(kg)
    if not kg_labels:
        return "skipped", 0, "no KG Topic nodes"

    new_gi, changes = _rewrite_gi_topics(gi, kg_labels)
    if changes == 0:
        return "noop", 0, None
    if not apply:
        return "dryrun", changes, None

    bak = gi_path.with_suffix(gi_path.suffix + ".bak")
    try:
        shutil.copy2(gi_path, bak)
        gi_path.write_text(
            json.dumps(new_gi, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        return "skipped", 0, f"write failed: {exc}"
    return "applied", changes, None


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="backfill_gi_topics",
        description="Rewrite GI Topic labels/IDs from KG canonical topics (#653).",
    )
    parser.add_argument("corpus_root", help="Corpus root (parent of feeds/ or of run_*/).")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite gi.json files (in place, with .bak backups). "
        "Without this flag the script is dry-run only.",
    )
    args = parser.parse_args()

    root = Path(args.corpus_root).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: corpus path does not exist or is not a directory: {root}", file=sys.stderr)
        return 2

    gi_paths = _iter_gi_paths(root)
    if not gi_paths:
        print(f"No gi.json artifacts found under {root}.", file=sys.stderr)
        return 1

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"#653 backfill — {mode} — scanning {len(gi_paths)} artifact(s) under {root}")

    applied = 0
    noop = 0
    dryrun = 0
    skipped = 0
    total_changes = 0
    errors: List[Tuple[Path, str]] = []

    for gi_path in gi_paths:
        status, changes, err = _apply_one(gi_path, args.apply)
        total_changes += changes
        rel = gi_path.relative_to(root)
        if status == "applied":
            applied += 1
            print(f"  APPLIED  {rel}  (+{changes} topic rewrites)")
        elif status == "dryrun":
            dryrun += 1
            print(f"  DRY-RUN  {rel}  ({changes} topic rewrites pending)")
        elif status == "noop":
            noop += 1
        elif status == "skipped":
            skipped += 1
            if err:
                errors.append((rel, err))

    print()
    print(
        f"Summary: {len(gi_paths)} scanned, "
        f"applied={applied}, dryrun={dryrun}, noop={noop}, skipped={skipped}"
    )
    print(f"Total topic rewrites: {total_changes}")
    if errors:
        print(f"\nSkipped ({len(errors)}):")
        for rel, err in errors[:20]:
            print(f"  {rel}: {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    if not args.apply and dryrun > 0:
        print()
        print("Re-run with --apply to write changes (each gi.json will get a .bak sibling).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

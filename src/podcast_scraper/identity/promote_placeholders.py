"""Promote an episode-scoped placeholder to the real person the graph already identifies (#1801).

`person:unresolved-brandon-{ep}` sitting in an episode that also contains `person:brandon-anderson`
is not an unsolved problem — the graph holds the answer. The placeholder exists only because the
2026-08-27 backfill ran with `heal=false`, which scopes everything rather than risk one wrong heal
across 678 episodes at once. That was the right call for a blanket run and it left 12 known cases
behind. This closes those.

WHY IT IS A SEPARATE OPERATION FROM m0007

m0007 scopes BARE ids, and `is_bare_person_id` returns False for anything already carrying
`unresolved-` (bare_name_scope.py:90) — that exclusion is what makes the migration idempotent. So
re-running m0007 at any heal setting cannot promote these: there is no bare id left to plan. The
work has to start from the placeholder instead.

WHAT MAKES IT SAFE ENOUGH TO DO NOW WHEN heal=false SAID NOT TO

Three things, and they did not hold when the blanket decision was made:

  * the candidates are NAMED. `unresolved-brandon-… -> brandon-anderson` is readable before it is
    applied; `heal=false` was chosen without knowing which cases it would cover.
  * `--mode plan` prints every promotion and writes nothing, so the whole set is reviewable.
  * the same one-candidate rule applies, over NODE-BACKED ids only (#1868) — an id existing solely
    as an edge endpoint or a quote's `speaker_id` is a dangling reference, not evidence.

Ambiguity still refuses. Two candidates means the graph does not know, and this operation will not
guess — that case is genuinely #1801's enricher problem.

IRREVERSIBLE. A promotion merges the placeholder's content onto a real person's global id. Wrong,
it attributes one person's words to another with no cheap undo. Hence plan-by-default, node-backed
candidates, and a refusal on any ambiguity.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .bare_name_scope import (
    person_node_ids_in,
    resolve_candidates,
    rewrite_ids,
    SCOPED_PREFIX,
)

_PERSON = "person:"


def _slug(person_id: str) -> str:
    return person_id.split(":", 1)[1] if ":" in person_id else person_id


def plan_promotions(gi_payload: dict, kg_payload: dict) -> Tuple[Dict[str, str], List[str]]:
    """``({placeholder_id: real_id}, [refused])`` for one episode. Pure; no I/O.

    A placeholder is promoted only when exactly ONE node-backed full name in the episode carries
    its token. Zero candidates is the ordinary case (nothing to promote); two or more is genuine
    ambiguity and is refused and reported, never guessed.
    """
    pool = person_node_ids_in(gi_payload) | person_node_ids_in(kg_payload)
    mapping: Dict[str, str] = {}
    refused: List[str] = []
    for pid in sorted(pool):
        slug = _slug(pid)
        if not slug.startswith(SCOPED_PREFIX):
            continue
        # `unresolved-{name}-{episode}` — the name is one token by construction, because only
        # single-token ids are ever scoped.
        name = slug[len(SCOPED_PREFIX) :].split("-")[0]
        candidates = resolve_candidates(f"{_PERSON}{name}", pool)
        if len(candidates) == 1:
            mapping[pid] = candidates[0]
        elif len(candidates) > 1:
            refused.append(f"{pid} -> {', '.join(candidates)}")
    return mapping, refused


def _iter_gi(root: Path):
    return sorted(root.rglob("*.gi.json"))


def _load(path: Path) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_atomic(path: Path, payload: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def run(root: Path, *, dry_run: bool) -> Dict[str, object]:
    """Walk the corpus; promote every unambiguous placeholder. Both layers or neither."""
    scanned = changed = promotions = unparsable = 0
    detail: List[str] = []
    refused_all: List[str] = []
    for gi_path in _iter_gi(root):
        scanned += 1
        kg_path = gi_path.with_name(gi_path.name[: -len(".gi.json")] + ".kg.json")
        gi = _load(gi_path)
        if gi is None:
            unparsable += 1
            continue
        kg = _load(kg_path) if kg_path.is_file() else None

        mapping, refused = plan_promotions(gi, kg or {})
        refused_all.extend(refused)
        if not mapping:
            continue

        new_gi, gi_changes = rewrite_ids(copy.deepcopy(gi), mapping)
        new_kg, kg_changes = rewrite_ids(copy.deepcopy(kg), mapping) if kg is not None else ({}, 0)
        if not gi_changes and not kg_changes:
            continue

        changed += 1
        promotions += len(mapping)
        feed = gi_path.parent.parent.name
        for old, new in sorted(mapping.items()):
            detail.append(f"{_slug(old)} -> {_slug(new)}  [{feed}]")
        if dry_run:
            continue
        # One map, both layers — a half-applied promotion would leave the two graphs disagreeing
        # about who this person is, which is the class of defect #1862 exists for.
        if gi_changes:
            _write_atomic(gi_path, new_gi)
        if kg_changes and kg is not None:
            _write_atomic(kg_path, new_kg)

    return {
        "scanned": scanned,
        "changed": changed,
        "promotions": promotions,
        "unparsable": unparsable,
        "detail": detail,
        "refused": sorted(set(refused_all)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Non-zero when the corpus is missing or the pass raises."""
    parser = argparse.ArgumentParser(
        prog="python -m podcast_scraper.identity.promote_placeholders",
        description="Promote unambiguous scoped placeholders to the real person (#1801).",
    )
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("plan", "apply"), default="plan")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root: Path = args.corpus_root
    if not root.is_dir():
        print(f"ERROR: corpus root does not exist: {root}", file=sys.stderr)
        return 1

    dry_run = args.mode == "plan"
    print(f"placeholder promotion — mode={args.mode} corpus={root}")
    if not dry_run:
        print("WRITING IN PLACE — a promotion merges content onto a REAL person's global id.")
    try:
        r = run(root, dry_run=dry_run)
    except Exception as exc:  # noqa: BLE001 — surface it, never a tidy zero
        print(f"ERROR: promotion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    verb = "would promote" if dry_run else "promoted"
    print(
        f"{verb} {r['promotions']} placeholder(s) across {r['changed']} episode(s); "
        f"{r['scanned']} scanned, {r['unparsable']} unparsable"
    )
    for line in r["detail"] if isinstance(r["detail"], list) else []:
        print(f"    {line}")
    refused = r["refused"]
    if isinstance(refused, list) and refused:
        print(f"REFUSED (ambiguous, left for #1801's enricher): {len(refused)}")
        for line in refused[:10]:
            print(f"    {line}")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())

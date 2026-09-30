"""Which episodes still need ``pipeline_stage=relabel_only``, as a work-list of episode ids.

WHY THIS IS COMMITTED. The 424-episode #2097 relabel batch was scoped by an ad-hoc
``/tmp/step3_all.py`` on the prod box. ``/tmp`` was cleared, the script went with it, and the
mop-up could no longer be scoped — the population was unknown, and guessing the criterion would
either re-label healthy episodes or silently leave damage. The criterion itself survived only
because the pieces it was built from ARE committed. This file closes that gap: a tool reused
between runs belongs in ``scripts/``, not in a temp directory.

NOTHING HERE IS A NEW CRITERION. Two borrowed pieces, each already the source of truth:

* **The serving view** — ``search.corpus_scope.dedupe_metadata_paths_newest_run_per_episode``,
  the same central membership rule m0007/m0009/m0010 select through. An episode with fourteen run
  dirs is judged on the copy the API actually serves; superseded copies are not the corpus.
* **The rules** — every universal predicate in :mod:`kg.speaker_coherence`, called individually so
  the report can say WHICH one fired. ``check_episode`` bundles exactly these and deliberately
  excludes ``check_roster_speakers_reach_the_graph``: a roster that never reached the graph is
  class A, which migration m0009 repairs without an LLM. This tool is about class B.

READ THE BREAKDOWN BEFORE DISPATCHING. ``relabel_only`` re-resolves speaker NAMES over a FROZEN
diarization and re-runs GI, which recomputes SPOKEN_BY. It repairs naming and attribution faults.
It cannot repair a diarization that merged every voice into one: there the names are downstream of
the real defect, and re-labelling faithfully reproduces the same wrong answer.

    speakers_actually_spoke    the roster names someone absent from the words  -> naming
    no_show_as_speaker         the roster holds a show/org, not a person       -> naming
    no_anonymous_speakers      SPEAKER_00 reached the graph                    -> naming
    roles_are_known            a role outside the known set                    -> naming
    spoken_by_targets_exist    SPOKEN_BY points at a node that is not there    -> graph
    one_quote_one_speaker      one quote attributed to two people              -> NOT VERIFIED
    not_collapsed_one_spkr     EVERY quote attributed to one person            -> attribution*

\\* ``not_collapsed_one_spkr`` LOOKS like a collapsed diarization and, on prod, was not. Measured
2026-09-29 on all 26 episodes that hit it: every one had 2-20 distinct ``speaker_label`` values in
its own ``*.segments.json`` (the diarization had separated the voices correctly) while every
attributed quote still landed on one person. So the fault was attribution, and relabel is the right
repair. The assumption that it meant re-diarization was stated confidently before this measurement
and was wrong; it would have sent 26 cheap repairs to the GPU path. Do not trust the rule name —
check the segments. If a future episode's segments DO hold a single voice, that one needs
re-diarization and relabel will not help.

``one_quote_one_speaker`` has not been tested against a relabel. Treat it as unknown until one is.

Read-only. Nothing here writes to the corpus; ``--worklist`` writes only the path you name.

Usage::

    python scripts/audit/relabel_worklist.py --corpus-dir /app/output
    python scripts/audit/relabel_worklist.py --corpus-dir /app/output \
        --worklist /app/output/relabel_repair_worklist.txt --only-relabel-fixable
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple


def _load(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    # A parseable artifact that is not an object would crash every `.get()` downstream.
    return data if isinstance(data, dict) else {}


def _rules() -> List[Tuple[str, Callable[..., List[str]]]]:
    """Each universal rule, named, so a violation can be attributed to one predicate.

    Kept in the same order as ``kg.speaker_coherence.check_episode`` so the union of these is
    exactly that function's result — if the two ever diverge, this list is the one that is wrong.
    """
    from podcast_scraper.kg import speaker_coherence as sc

    return [
        ("speakers_actually_spoke", lambda md, kg, gi: sc.check_speakers_actually_spoke(md, kg)),
        ("no_show_as_speaker", lambda md, kg, gi: sc.check_no_show_as_speaker(md, kg)),
        ("no_anonymous_speakers", lambda md, kg, gi: sc.check_no_anonymous_speakers(kg)),
        ("roles_are_known", lambda md, kg, gi: sc.check_roles_are_known(kg)),
        ("spoken_by_targets_exist", lambda md, kg, gi: sc.check_spoken_by_targets_exist(gi)),
        ("one_quote_one_speaker", lambda md, kg, gi: sc.check_one_quote_one_speaker(gi)),
        (
            "not_collapsed_one_spkr",
            lambda md, kg, gi: sc.check_not_collapsed_onto_one_speaker(md, gi),
        ),
        # #2198. NOT in RELABEL_FIXABLE: the cause is speaker DETECTION missing names the
        # transcript states, and a relabel reuses the frozen roster — untested as a repair.
        ("some_speaker_named", lambda md, kg, gi: sc.check_some_speaker_is_named(gi)),
    ]


#: Rules a relabel is known to repair. ``not_collapsed_one_spkr`` is IN — measured, see the module
#: docstring. ``one_quote_one_speaker`` is OUT because it is untested, not because it is known not
#: to work; ``spoken_by_targets_exist`` is a graph fault with no measured repair either.
RELABEL_FIXABLE = (
    "speakers_actually_spoke",
    "no_show_as_speaker",
    "no_anonymous_speakers",
    "roles_are_known",
    "not_collapsed_one_spkr",
)


def _served_kg_paths(root: Path) -> List[Path]:
    from podcast_scraper.search.corpus_scope import (
        dedupe_metadata_paths_newest_run_per_episode,
    )

    kg_paths = [p for p in sorted(root.rglob("*.kg.json")) if ".trash" not in p.parts]
    md_to_kg = {Path(str(p)[: -len(".kg.json")] + ".metadata.json"): p for p in kg_paths}
    kept = dedupe_metadata_paths_newest_run_per_episode(root, list(md_to_kg))
    return sorted(md_to_kg[m] for m in kept if m in md_to_kg)


def _roster(md: Mapping[str, Any]) -> List[str]:
    rows = (md.get("content") or {}).get("speakers") or []
    out = []
    for s in rows:
        out.append(str(s.get("name")) if isinstance(s, dict) else str(s))
    return out


def scan(root: Path) -> Dict[str, Dict[str, Any]]:
    """``episode_id -> {feed, rules, roster}`` for every SERVED episode with a violation."""
    rules = _rules()
    rows: Dict[str, Dict[str, Any]] = {}
    for kg_path in _served_kg_paths(root):
        stem = str(kg_path)[: -len(".kg.json")]
        md_path = Path(stem + ".metadata.json")
        if not md_path.is_file():
            continue
        md, kg, gi = _load(md_path), _load(kg_path), _load(Path(stem + ".gi.json"))
        episode_id = str((md.get("episode") or {}).get("episode_id") or "")
        if not episode_id:
            # No id means nothing to put in a work-list; report it rather than drop it silently.
            episode_id = f"<no episode_id: {kg_path.name}>"
        hit: List[str] = []
        counts: Dict[str, int] = {}
        for name, fn in rules:
            try:
                found = fn(md, kg, gi)
            except Exception as exc:  # a broken artifact must not hide the rest of the corpus
                found = [f"RULE-ERROR {name}: {exc}"]
            if found:
                hit.append(name)
                counts[name] = len(found)
        if hit:
            parts = kg_path.relative_to(root).parts
            rows[episode_id] = {
                "feed": parts[1] if len(parts) > 1 else "?",
                "rules": hit,
                "counts": counts,
                "roster": _roster(md),
            }
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-dir", required=True, type=Path)
    ap.add_argument(
        "--worklist",
        type=Path,
        default=None,
        help="Write the episode ids here, one per line (the --reprocess-episode-ids format).",
    )
    ap.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="RULE",
        help=(
            "Restrict the work-list to episodes hitting this rule; repeatable. "
            "Use --only-relabel-fixable for the rules a relabel is known to repair."
        ),
    )
    ap.add_argument(
        "--only-relabel-fixable",
        action="store_true",
        help=f"Shorthand for the rules a relabel is known to repair: {', '.join(RELABEL_FIXABLE)}",
    )
    ap.add_argument("--json", type=Path, default=None, help="Write the full classification here.")
    args = ap.parse_args(argv)

    rows = scan(args.corpus_dir)
    wanted = list(args.only or [])
    if args.only_relabel_fixable:
        wanted += list(RELABEL_FIXABLE)

    print(f"episodes violating : {len(rows)}")
    print()
    print(f"{'rule':<26}{'episodes':>10}{'messages':>10}")
    for name, _ in _rules():
        eps = [v for v in rows.values() if name in v["rules"]]
        if eps:
            msgs = sum(v["counts"].get(name, 0) for v in eps)
            fix = "" if name in RELABEL_FIXABLE else "   <- relabel repair NOT verified"
            print(f"{name:<26}{len(eps):>10}{msgs:>10}{fix}")
    print()
    print("rule combinations per episode:")
    combos = collections.Counter(tuple(v["rules"]) for v in rows.values())
    for combo, n in combos.most_common():
        print(f"  {n:>3}  {'+'.join(combo)}")
    print()
    print("by feed:")
    for feed, n in collections.Counter(v["feed"] for v in rows.values()).most_common():
        print(f"  {n:>3}  {feed}")

    selected = sorted(
        ep for ep, v in rows.items() if not wanted or any(r in wanted for r in v["rules"])
    )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, sort_keys=True), encoding="utf-8")
        print(f"\nclassification written: {args.json}")
    if args.worklist:
        args.worklist.write_text("".join(f"{e}\n" for e in selected), encoding="utf-8")
        scope = "+".join(wanted) if wanted else "ALL rules"
        print(f"work-list written: {args.worklist}  ({len(selected)} ids, scope={scope})")
    elif wanted:
        print(f"\nwould select {len(selected)} of {len(rows)} (scope={'+'.join(wanted)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

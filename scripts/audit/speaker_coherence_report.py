"""Report speaker/role coherence over a corpus, and what m0009 WOULD do to it.

The post-deploy runbook's verification steps had no runnable commands behind them (#2065 advisor
S5): "re-run the migration, expect 0/0" is blocked by the upgrade ledger, which excludes any
migration already recorded, and "re-run kg.speaker_coherence.check_corpus" named a function with no
CLI. Step 4 was a description, not a procedure. This is the procedure.

Two modes, because they answer different questions:

``--coherence`` (default)
    Every universal rule in :mod:`kg.speaker_coherence` over every episode. This is the
    corpus-state question: is what is on disk self-consistent right now.

``--migration-preview``
    Runs m0009's own ``apply()`` in dry-run, bypassing the ledger, and prints the result plus the
    two classes that need a human: ``suspect_demotions`` (the node had a voice, which is what an
    eponymous-show host looks like) and ``ambiguous_nodes`` (matched more than one roster entry).

``--roles`` adds a node-level role TRANSITION table. The violation count alone cannot see a
wrongful demotion — ``check_no_show_as_speaker`` scores one as a violation FIXED, because the
checks share the migration's own predicates. The transition table is the honest instrument: it
reports what changed per node, not how many rules stopped firing.

Read-only. Nothing here writes to the corpus.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _triples(root: Path, newest_run_only: bool = False) -> List[Tuple[str, Path, Path, Path]]:
    """``(label, metadata, kg, gi)`` for every episode with a kg artifact.

    DEFAULT IS UNDEDUPED, AND THAT IS DELIBERATE: m0009/m0010 glob ``rglob("*.kg.json")``
    the same way, so the default view is what the MIGRATION will actually read. Changing
    it would make this report stop describing the thing it exists to predict.

    ``newest_run_only=True`` restricts to the newest run per ``(feed_id, episode_id)`` —
    the same central membership rule the serving layer and ``_on_disk_guid_index`` use.
    That view answers a different and narrower question: *of the changes this migration
    would make, which land on a copy anyone can actually see?*

    BOTH ARE NEEDED, because a transition on a superseded copy is noise while the same
    transition on a served copy is a regression. Worked example (2026-09-22): m0009 would
    demote Krishna Rao guest -> mentioned on an Aug-18 copy whose roster wrongly names Sam
    Altman, while the served Aug-26 copy — repaired — correctly has him as guest. Reading
    only the undeduped view calls that a real-person demotion; reading only the deduped
    view hides that the migration will write it. The gate belongs on the deduped view.
    """
    kg_paths = sorted(root.rglob("*.kg.json"))
    if newest_run_only:
        from podcast_scraper.search.corpus_scope import (
            dedupe_metadata_paths_newest_run_per_episode,
        )

        # The dedupe rule keys on metadata paths; map kg -> metadata, filter, map back.
        md_to_kg = {Path(str(p)[: -len(".kg.json")] + ".metadata.json"): p for p in kg_paths}
        kept = dedupe_metadata_paths_newest_run_per_episode(root, list(md_to_kg))
        kg_paths = sorted(md_to_kg[m] for m in kept if m in md_to_kg)

    out = []
    for kg_path in kg_paths:
        stem = str(kg_path)[: -len(".kg.json")]
        out.append((kg_path.name, Path(stem + ".metadata.json"), kg_path, Path(stem + ".gi.json")))
    return out


def _coherence(root: Path, newest_run_only: bool = False) -> int:
    from podcast_scraper.kg.speaker_coherence import check_corpus

    episodes = []
    for label, md_path, kg_path, gi_path in _triples(root, newest_run_only):
        if not md_path.is_file():
            continue
        episodes.append((label, _load(md_path), _load(kg_path), _load(gi_path)))
    violations = check_corpus(episodes)
    kinds = collections.Counter(v.split(": ", 1)[-1].split("=")[0] for v in violations)
    print(f"episodes checked : {len(episodes)}")
    print(f"violations       : {len(violations)}")
    for kind, n in kinds.most_common(12):
        print(f"   {n:5}  {kind}")
    for v in violations[:40]:
        print(f"   - {v[:160]}")
    if len(violations) > 40:
        print(f"   … and {len(violations) - 40} more")
    return 0


def _migration_preview(root: Path, show_roles: bool, newest_run_only: bool = False) -> int:
    from podcast_scraper.upgrade.migration import MigrationContext
    from podcast_scraper.upgrade.migrations.m0009_backfill_speaker_roles import (
        BackfillSpeakerRolesMigration,
        demote_non_persons,
        promote_person_roles,
        roster_is_a_guess,
        roster_roles,
        voices_heard,
        voices_in_episode,
    )

    # apply() directly rather than `upgrade run`: the ledger excludes a migration already recorded,
    # so the runbook's "re-run it and expect 0/0" cannot be done through the CLI.
    result = BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=root, dry_run=True))
    print(result.message)
    for key, heading in (
        ("suspect_demotions", "SUSPECT demotions — the node had a voice (HAND-READ)"),
        ("ambiguous_nodes", "AMBIGUOUS — matched more than one roster entry (left untouched)"),
    ):
        rows = (result.details or {}).get(key) or []
        print(f"\n{heading}: {len(rows)}")
        for row in rows:
            print(f"   {row}")

    if not show_roles:
        return 0

    transitions: collections.Counter = collections.Counter()
    demotions: List[str] = []
    for _label, md_path, kg_path, gi_path in _triples(root, newest_run_only):
        if not md_path.is_file():
            continue
        kg, md, gi = _load(kg_path), _load(md_path), _load(gi_path)
        feed = str((md.get("feed") or {}).get("title") or "")
        voices = voices_in_episode(gi, kg)
        before = {
            n["id"]: (
                str((n.get("properties") or {}).get("name") or ""),
                str((n.get("properties") or {}).get("role") or "none"),
            )
            for n in kg.get("nodes", [])
            if str(n.get("type", "")).lower() == "person"
        }
        after = copy.deepcopy(kg)
        demote_non_persons(after, feed, voices)
        roles = roster_roles(md)
        # #2075, in step with m0009 `apply()`: a guessed roster writes no role in either direction,
        # so it is skipped here too rather than modelled as promote-only.
        if roles and not roster_is_a_guess(md, md_path):
            # `md_path` is NOT optional here. `voices_heard` returns None without it — by design,
            # it refuses to guess — and a None denominator switches OFF m0009's roster-denies
            # demotion. Omitting the path therefore produced a transition table that could not
            # show the one route the runbook sends the operator here to inspect: the destructive
            # one. The table would have read "0 demotions" whatever the migration was about to do.
            #
            # The provenance gate is repeated from `apply()` for the same reason: an instrument
            # that models the migration differently from the migration reports a run that is not
            # the one about to happen. These two lines must stay in step.
            heard = voices_heard(md, md_path)
            promote_person_roles(after, roles, voices_heard=heard, feed_title=feed)
        for node in after.get("nodes", []):
            if str(node.get("type", "")).lower() != "person":
                continue
            nid = node["id"]
            if nid not in before:
                continue
            name, old = before[nid]
            new = str((node.get("properties") or {}).get("role") or "none")
            if old == new:
                continue
            transitions[(old, new)] += 1
            if new == "mentioned" and old in ("host", "guest"):
                demotions.append(
                    f"{name} ({old}) — {feed}{'  [HAD A VOICE]' if nid in voices else ''}"
                )

    print("\nROLE TRANSITIONS (what the violation count cannot see)")
    for (old, new), n in sorted(transitions.items(), key=lambda kv: -kv[1]):
        print(f"   {old:10} -> {new:10} {n}")
    print(f"\nDEMOTIONS: {len(demotions)}")
    for row in demotions:
        print(f"   {row}")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument(
        "--newest-run-only",
        action="store_true",
        help="Restrict to the newest run per (feed_id, episode_id) — the SERVING view, the "
        "copies users actually see. DEFAULT IS OFF so the report keeps describing the "
        "MIGRATION view: m0007/m0009/m0010 glob every run copy, undeduped. A role "
        "transition on a superseded copy is noise; the same transition under this flag is "
        "a regression. "
        "SCOPES ONLY THIS SCRIPT'S OWN PASSES — DEMOTIONS and ROLE TRANSITIONS. The "
        "headline, SUSPECT and AMBIGUOUS sections come from m0009's own apply(dry_run=True) "
        "and are UNAFFECTED BY DESIGN, because they must describe what the migration will "
        "actually do across every copy. Measured on prod 2026-09-22: demotions read 205 "
        "unflagged vs 174 flagged — 205 is what m0009 does; do not quote 174 as its output.",
    )
    parser.add_argument(
        "--migration-preview",
        action="store_true",
        help="Dry-run m0009 (bypasses the upgrade ledger) instead of checking coherence.",
    )
    parser.add_argument(
        "--roles",
        action="store_true",
        help="With --migration-preview, add the node-level role transition table.",
    )
    args = parser.parse_args(argv)
    root = args.corpus_dir.expanduser().resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    if args.migration_preview:
        return _migration_preview(root, args.roles, args.newest_run_only)
    return _coherence(root, args.newest_run_only)


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())

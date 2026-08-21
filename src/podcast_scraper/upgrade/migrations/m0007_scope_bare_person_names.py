"""0007 — stop bare first names being global, followable people in an EXISTING corpus (#1685).

The pipeline stopped minting them (``workflow/metadata_generation`` runs
``identity.bare_name_scope`` before typed mentions), but a mint-time change only affects episodes
processed after it. Production's 1931 person entities were minted before, so without this step it
keeps its pooled tokens indefinitely — and worse, derived interests keep re-minting them into
user profiles from listening behaviour, with no click involved
(``server/app_user_corpus.derived_interest_tokens`` reads the same id space).

Measured on that corpus: 208 occurrences of 172 single-word person ids; 12 resolvable within
their own episode, 0 ambiguous, 196 orphan.

SHARES ITS RULE WITH THE PIPELINE, VERBATIM. Both call ``plan_bare_name_ids`` / ``rewrite_ids``.
Two implementations of "who is Sam in this episode" would drift, and the drift would be invisible
— the corpus would simply contain two different answers depending on when each episode was
processed.

PAIRED PER EPISODE, NOT PER FILE. Unlike 0006, the rule needs the episode's whole roster, which
is split across the ``.gi.json`` and ``.kg.json`` artifacts — the production measurement proved
the KG alone gives the wrong answer (``person:alex`` reads as an orphan while
``person:alex-mayassi`` sits in the GI layer). So the two files are read together, one id map is
computed from their union, and both are rewritten with it.

THE EPISODE ID COMES FROM THE ARTIFACT, NOT THE FILENAME. Scoped ids embed the episode, so if
this migration derived it differently from the pipeline the two would produce different ids for
the same episode — the exact drift the shared rule exists to prevent. Both layers carry an
``episode:{id}`` node; that is the key.

Idempotent: a migrated artifact has no bare person ids left, so the second run plans an empty map
and writes nothing. Unparsable files are recorded and skipped rather than failing the run
(mirrors 0003/0005/0006).
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ...identity.bare_name_scope import person_ids_in, plan_bare_name_ids, rewrite_ids
from ..migration import Migration, MigrationContext, MigrationResult


def _iter_gi_files(root: Path) -> Iterable[Path]:
    """All ``*.gi.json`` files under *root* (recursive). Stable order."""
    return sorted(root.rglob("*.gi.json"))


def _kg_sibling(gi_path: Path) -> Path:
    """The ``.kg.json`` beside a ``.gi.json`` — same stem, both written by the same episode."""
    return gi_path.with_name(gi_path.name[: -len(".gi.json")] + ".kg.json")


def _episode_id_of(*payloads: dict) -> str:
    """The ``episode:{id}`` node id shared by both layers, without its prefix.

    Falls back to an empty string, which ``scoped_person_id`` turns into ``unknown`` — a visible,
    greppable marker rather than a silent mis-scoping.
    """
    for payload in payloads:
        for node in (payload or {}).get("nodes") or []:
            if isinstance(node, dict):
                nid = node.get("id")
                if isinstance(nid, str) and nid.startswith("episode:"):
                    return nid.split(":", 1)[1]
    return ""


def _load(path: Path) -> Tuple[dict | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, exc.__class__.__name__


def _write_atomic(path: Path, payload: dict) -> None:
    """tmp + os.replace — a kill mid-write must not leave a truncated, unparsable artifact."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _plan_for_pair(
    gi_path: Path, heal: bool
) -> Tuple[Dict[str, str], dict | None, dict | None, str | None]:
    """``(id_map, gi_payload, kg_payload, error)`` for one episode's artifact pair."""
    gi_payload, err = _load(gi_path)
    if gi_payload is None:
        return {}, None, None, f"{gi_path}: {err}"
    kg_path = _kg_sibling(gi_path)
    kg_payload: dict | None = None
    if kg_path.is_file():
        kg_payload, kg_err = _load(kg_path)
        if kg_payload is None:
            return {}, gi_payload, None, f"{kg_path}: {kg_err}"

    roster = person_ids_in(gi_payload) | person_ids_in(kg_payload or {})
    episode_id = _episode_id_of(gi_payload, kg_payload or {})
    return plan_bare_name_ids(roster, episode_id, heal=heal), gi_payload, kg_payload, None


class ScopeBarePersonNamesMigration(Migration):
    """Episode-scope (or heal) bare person ids across an existing corpus."""

    id = "0007_scope_bare_person_names"
    to_version = "2.7.1"
    description = (
        "#1685: a single-token person id identifies someone within one episode and nobody "
        "globally — scope it per episode (or heal it to the full name when that episode "
        "contains exactly one candidate) so it stops being a global followable person"
    )

    #: Whether to heal a bare name that has exactly one full-name candidate in its episode.
    #:
    #: This is a CLASS CONSTANT and does NOT read ``cfg.bare_name_heal`` — a migration runs
    #: against a corpus directory, not a pipeline config, and there is no Config in scope. So if
    #: the pipeline flag is ever set False, the two disagree: new episodes scope everything while
    #: a backfill still heals. Stated here rather than left to be discovered, because the whole
    #: point of sharing `plan_bare_name_ids` verbatim is that the two cannot disagree about a
    #: verdict — this is the one input that is not shared.
    #:
    #: False scopes everything, including resolvable names — the strictly safer setting, because
    #: a wrong heal writes a real person's id onto someone else's content and cannot be undone,
    #: while a wrong scope can. Override on the instance if you want a conservative backfill.
    heal = True

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would rewrite, per verdict — pure read, no writes."""
        files = list(_iter_gi_files(ctx.corpus_root))
        if not files:
            return "no .gi.json files under corpus — nothing to migrate"
        episodes = healed = scoped = unparsable = 0
        for gi_path in files:
            id_map, _gi, _kg, err = _plan_for_pair(gi_path, self.heal)
            if err:
                unparsable += 1
                continue
            if not id_map:
                continue
            episodes += 1
            for old, new in id_map.items():
                if new.startswith("person:unresolved-"):
                    scoped += 1
                else:
                    healed += 1
        return (
            f"bare-name scoping plan: {len(files)} episodes scanned, {episodes} would change "
            f"({healed} healed to a full name, {scoped} episode-scoped), "
            f"{unparsable} unparsable (will be skipped)"
        )

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Rewrite each episode's ``.gi.json``/``.kg.json`` pair together, or neither.

        The pair shares one id map, so writing one layer without the other would leave the
        episode's two graphs disagreeing about who a person is — worse than not migrating.
        """
        files = list(_iter_gi_files(ctx.corpus_root))
        if not files:
            ctx.log(f"no .gi.json files under {ctx.corpus_root}")
            return MigrationResult(
                self.id,
                applied=True,
                dry_run=ctx.dry_run,
                message="no .gi.json files found",
                details={"episodes_scanned": 0},
            )

        changed: List[str] = []
        healed = scoped = unchanged = 0
        unparsable: List[str] = []

        for gi_path in files:
            id_map, gi_payload, kg_payload, err = _plan_for_pair(gi_path, self.heal)
            if err:
                unparsable.append(err)
                continue
            if not id_map or gi_payload is None:
                unchanged += 1
                continue

            for new in id_map.values():
                if new.startswith("person:unresolved-"):
                    scoped += 1
                else:
                    healed += 1

            new_gi, gi_changes = rewrite_ids(copy.deepcopy(gi_payload), id_map)
            new_kg, kg_changes = (
                rewrite_ids(copy.deepcopy(kg_payload), id_map)
                if kg_payload is not None
                else ({}, 0)
            )
            if not gi_changes and not kg_changes:
                unchanged += 1
                continue

            changed.append(str(gi_path.relative_to(ctx.corpus_root)))
            if ctx.dry_run:
                continue
            if gi_changes:
                _write_atomic(gi_path, new_gi)
            if kg_changes and kg_payload is not None:
                _write_atomic(_kg_sibling(gi_path), new_kg)

        message = (
            f"{'would rewrite' if ctx.dry_run else 'rewrote'} {len(changed)} episode(s): "
            f"{healed} healed, {scoped} scoped; {unchanged} already-current, "
            f"{len(unparsable)} unparsable"
        )
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=message,
            details={
                "episodes_scanned": len(files),
                "changed": len(changed),
                "healed": healed,
                "scoped": scoped,
                "unparsable": unparsable[:20],
            },
        )

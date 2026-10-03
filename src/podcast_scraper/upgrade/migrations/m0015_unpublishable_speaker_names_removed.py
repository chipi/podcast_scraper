"""0015 — a name today's publish gate refuses is nobody's voice (post-deploy C1, 2026-10-03).

The publish gate (``speaker_detectors.hosts.is_publishable_speaker_name``) now refuses role words,
interjections, committees, job titles, show/brand and region tails, count words and product
mononyms ("Host" x20, "OK" x3, "Thank", "Right", "GE", "House Select Committee", "Roblox CEO",
"Americas Online", "Norman Conquest", "Trivium China"; 45 of 4,638 published names on 2026-10-02).
New episodes are gated; this removes what is already on disk. No LLM, no reprocess. Only the
gate's EXPLICIT junk rules decide here (see ``refused``).

Each surface becomes what the pipeline writes for a voice it failed to name, exactly as m0012 does
for an organisation (the same ``_Episode`` rewrite, on all five surfaces or none): the roster entry
and ``detected_*`` mention go, segment ``speaker_label`` goes with ``voice_type: unknown``, the KG
host/guest Person node and its edges go, the GI Person / ``SPOKEN_BY`` / quote / insight
attribution goes (insights recomputed as unattributed), and the bridge identity row goes.

A NAME WITH A CLEAN FORM IS RENAMED, NOT REMOVED. "Your Host Luisa Leni" and "Celestin Ntawirema
CEO" fail the gate, but the person is there: m0017 renames them to the clean form, so this
migration leaves every name whose cleaned form is publishable alone (``rename_target``).

THE SET IS FROZEN at apply time from the served rosters and written to the receipts; ``verify``
judges against it. UNDO restores each file from ``.podcast_scraper/upgrade-backups/0015/`` while it
is still exactly what this migration wrote.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, read_receipts, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult
from .m0012_org_speakers_removed import _Episode as _OrgEpisode, _load

MIGRATION_ID = "0015_unpublishable_speaker_names_removed"
RECEIPTS_FILE = "unpublishable_speaker_names_removed.jsonl"
BACKUP_TAG = "0015"


#: Real people the gate refuses, kept by hand after reading the prod dry run (2026-10-03). Each
#: entry names who it is. Add here, never relax the gate for one cleanup.
KEEP: Dict[str, str] = {
    "RJ": "RJ Honicky, Latent Space co-host; the gate reads 2-3 capitals as an abbreviation (GE)",
}

#: Junk no gate rule catches safely (the ordinary-word rule would also remove real people),
#: removed by hand after reading every published host/guest on prod (2026-10-03, operator-approved).
REMOVE: Dict[str, str] = {
    "Before Gene": "ASR fragment: a sentence opener captured as a name",
    "As Colin": "ASR fragment: a sentence opener captured as a name",
    "Super Willing To Be": "ASR fragment: a clause captured as a name",
    "Moral": "a word, not a name",
    "Generation": "a word, not a name",
    "Pan-African": "an adjective, not a name",
}

#: Role words in FRONT of a person's name, as published by older rosters ("Your Host Luisa Leni").
_LEADING_ROLE_WORDS = frozenset({"your", "our", "host", "hosts", "co-host", "cohost", "presenter"})


def rename_target(name: str) -> Optional[str]:
    """The clean form a published name is renamed to (m0017), or ``None`` when it has none.

    Leading role words ("Your Host"), job titles, honorifics and possessive show prefixes go
    (``hosts._clean_stated_name``, the pipeline's own stated-name cleaner); the result must pass
    today's publish gate. Shared with m0017 so "removed" and "renamed" never overlap.
    """
    from ...speaker_detectors.hosts import _clean_stated_name, is_publishable_speaker_name

    tokens = name.split()
    while len(tokens) > 2 and tokens[0].lower().strip(",:") in _LEADING_ROLE_WORDS:
        tokens = tokens[1:]
    # "Ben Fritz's" (a possessive the transcript attached to the name).
    if len(tokens) >= 2 and tokens[-1].endswith(("'s", "’s")):
        tokens = tokens[:-1] + [tokens[-1][:-2]]
    clean = _clean_stated_name(" ".join(tokens))
    if clean and clean != name and is_publishable_speaker_name(clean):
        return clean
    return None


def refused(name: Any, feed_title: Optional[str] = None) -> bool:
    """An explicit junk rule of today's gate refuses this name, and it has no clean form.

    Two of the gate's checks are NOT used to remove what is published, because each also refuses
    real people (dry run on prod, 2026-10-03): the ordinary-English-word check on a multi-word name
    ("Ethan He", "Henry He", "Michael I. Jordan") and the show-name check ("Peter Attia" on The
    Peter Attia Drive, 43 episodes — the eponymous hosts m0014 restored). ``feed_title`` is kept in
    the signature for callers; it no longer decides anything.
    """
    from ...speaker_detectors.hosts import is_publishable_speaker_name

    del feed_title
    if not isinstance(name, str) or not name.strip() or name in KEEP:
        return False
    if name in REMOVE:
        return True
    if rename_target(name) is not None:
        return False
    return not is_publishable_speaker_name(name, require_person_shape=False)


def refused_names(root: Path) -> Dict[str, int]:
    """``{name: roster entries}`` for every served roster name the gate refuses (the frozen set)."""
    out: Dict[str, int] = {}
    for meta in select_served_artifacts(root, ".metadata.json")[0]:
        payload = _load(meta)
        if not isinstance(payload, dict):
            continue
        title = (payload.get("feed") or {}).get("title")
        for entry in (payload.get("content") or {}).get("speakers") or []:
            name = (entry or {}).get("name")
            if isinstance(name, str) and refused(name, title):
                out[name] = out.get(name, 0) + 1
    return out


class _Episode(_OrgEpisode):
    """m0012's five-surface rewrite, keyed on the frozen refused-name set (exact names)."""

    def _is_org(self, name: Any) -> bool:
        return isinstance(name, str) and name in self.orgs


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class UnpublishableSpeakerNamesRemovedMigration(Migration):
    """Remove published speaker names today's publish gate refuses, from every speaker surface."""

    id = MIGRATION_ID
    to_version = "2.7.9"
    description = (
        "post-deploy C1: a published speaker name today's publish gate refuses (role word, "
        "interjection, committee, job title, org/region tail) and that has no "
        "clean form to rename to is removed from roster, segments, KG, GI and bridge"
    )

    def _episodes(self, root: Path, names: Set[str]) -> Iterable[_Episode]:
        for meta in select_served_artifacts(root, ".metadata.json")[0]:
            ep = _Episode(meta, names)
            if ep.plan():
                yield ep

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would rewrite — pure read, no writes."""
        names = refused_names(ctx.corpus_root)
        if not names:
            return "no published speaker name is refused by today's gate — nothing to remove"
        totals: Dict[str, int] = {}
        episodes = 0
        for ep in self._episodes(ctx.corpus_root, set(names)):
            episodes += 1
            for k, v in ep.counts.items():
                totals[k] = totals.get(k, 0) + v
        return (
            f"refused speaker names plan: {len(names)} name(s) {sorted(names)}; {episodes} "
            "episode(s); " + ", ".join(f"{k}={v}" for k, v in sorted(totals.items()))
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """No served surface still carries a FROZEN refused name. ``(ok, message)``."""
        header, _rows = read_receipts(ctx.corpus_root, RECEIPTS_FILE)
        names = set(header.get("names") or {})
        if not names:
            return True, "no receipts — nothing was removed, nothing to verify"
        left = [
            f"{ep.meta.name}: {sorted(ep.counts)}" for ep in self._episodes(ctx.corpus_root, names)
        ]
        if left:
            return False, f"{len(left)} episode(s) still carry a refused name: {left[:5]}"
        return True, f"no served surface carries any of {len(names)} refused name(s)"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Remove the frozen refused names from every speaker surface; back up, receipt."""
        root = ctx.corpus_root
        names = refused_names(root)
        totals: Dict[str, int] = {}
        touched: List[str] = []
        receipts: List[dict] = []
        for ep in self._episodes(root, set(names)):
            touched.append(str(ep.meta.relative_to(root)))
            for k, v in ep.counts.items():
                totals[k] = totals.get(k, 0) + v
            if ctx.dry_run:
                continue
            for path in sorted(ep.changed):
                receipts.append(write_with_backup(root, BACKUP_TAG, path, ep.files[path]))
        if not ctx.dry_run:
            append_receipts(
                root,
                RECEIPTS_FILE,
                {"names": names, "kept": KEEP, "removed_by_hand": REMOVE},
                receipts,
            )
        verb = "would rewrite" if ctx.dry_run else "rewrote"
        message = (
            f"{verb} {len(touched)} episode(s) for {len(names)} refused name(s): "
            + ", ".join(f"{k}={v}" for k, v in sorted(totals.items()))
        )
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=message,
            details={
                "names": dict(sorted(names.items())),
                "episodes": len(touched),
                "totals": totals,
                "files_written": len(receipts),
                "episodes_sample": touched[:20],
            },
        )

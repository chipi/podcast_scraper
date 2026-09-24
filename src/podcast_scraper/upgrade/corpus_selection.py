"""Which artifact copies a migration may touch — the central corpus-membership rule, applied.

A corpus keeps every run. ``root.rglob("*.kg.json")`` therefore returns SUPERSEDED copies as well
as served ones, and the migrations were the only consumer of the corpus that did not reconcile
that: ``search.corpus_scope.dedupe_metadata_paths_newest_run_per_episode`` is documented there as
the "Central corpus-membership rule", and the serving layer (``corpus_metadata_index``) and the
repair route (``stages.scraping._on_disk_guid_index``) both honour it.

Measured on prod 2026-09-22: 2,312 files globbed against 2,002 after dedupe — 310 superseded
copies that are never served, never searched, and CANNOT be repaired, because ``relabel_only``
resolves through the newest-run index.

WHY THAT IS MORE THAN WASTED WORK. A superseded copy can carry a roster from before a repair. On
one of them m0009 demoted Krishna Rao — a real guest of "Anthropic's CFO on Managing Compute" —
from guest to mentioned, because that copy's roster wrongly names Sam Altman. His SERVED copy was
correct throughout. So a real person was demoted on disk, in a file nobody reads, and the
migration's own hand-read guard stayed silent about it (see ``m0009``'s pairing-vouch check).

Migrating only served copies makes that whole class unreachable: the copies with stale rosters are
no longer selected at all.

THE AUDIT SCRIPT MUST STAY IN STEP. ``scripts/audit/speaker_coherence_report.py`` deliberately
mirrored the migrations' undeduped selection, on the principle that an instrument which models the
migration differently reports a run that is not the one about to happen. It now mirrors this
instead; that is one change, not two independent ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

METADATA_SUFFIX = ".metadata.json"


def metadata_sibling_for(artifact: Path, suffix: str) -> Path:
    """The ``.metadata.json`` beside *artifact*, whose name ends with *suffix*."""
    return artifact.with_name(artifact.name[: -len(suffix)] + METADATA_SUFFIX)


def select_served_artifacts(root: Path, suffix: str) -> Tuple[List[Path], List[Path]]:
    """``(served, superseded)`` artifact paths under *root* matching *suffix*, stable order.

    Membership is decided on the artifact's metadata sibling, because that is what the central
    rule reads — it needs ``(feed_id, episode_id)``, which only the metadata carries.

    An artifact whose metadata sibling is MISSING counts as served. That is the safe direction and
    it is a deliberate choice: membership cannot be established for it, and excluding it would
    silently drop an episode from a migration on the strength of a missing file, which is a worse
    failure than migrating one extra copy. It is also exactly what happens today.
    """
    artifacts = sorted(root.rglob(f"*{suffix}"))
    if not artifacts:
        return [], []

    from ..search.corpus_scope import dedupe_metadata_paths_newest_run_per_episode

    by_metadata: dict[Path, Path] = {}
    orphans: List[Path] = []
    for art in artifacts:
        meta = metadata_sibling_for(art, suffix)
        if meta.is_file():
            by_metadata[meta] = art
        else:
            orphans.append(art)

    kept = set(dedupe_metadata_paths_newest_run_per_episode(root, sorted(by_metadata)))
    served = sorted([art for meta, art in by_metadata.items() if meta in kept] + orphans)
    superseded = sorted(art for meta, art in by_metadata.items() if meta not in kept)
    return served, superseded

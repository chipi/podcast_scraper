"""0035 — the voice identity rung on the served episodes that carry fingerprints (text only).

The pipeline's voice identity rung (``voice_identity_rung``, naming ladder doc v26 / v29) compares
an episode's voice fingerprints with the same feed's other episodes: a voice that matches a group
named by vote takes that name, and a name leaves a voice that is clearly not that person. Served
episodes already carry what it needs: the diarizer's fingerprints (``voice_embeddings`` in the
speaker-diagnostics sidecar, captured since #2339) and the feed's published names. So the rung can
run over what is served without the DGX: on prod, 413 of 2,939 served episodes (14%) carried
fingerprints on 2026-10-23, all written since 2026-10-12, and the share grows with every episode.

Measured (podcast-scraper-eval-data ``autoresearch/voice_embedding_match``, pre-registered val_v4,
10 feeds never used to shape it): wrong names 23 -> 17 (p = 0.031), correct names 70 -> 77
(p = 0.014).

THE LIST IS FROZEN AND REVIEWED, as 0034's. :func:`find_changes` replays the rung (the pipeline's
own ``voice_identity.decide`` on the stored fingerprints, threshold 0.7) over a corpus copy;
:func:`write_list` (0034's) freezes it in ``upgrade/lists/0035_voice_identity_names.json``. The
review judges a RENAME by its new name (``CORRECT`` / ``CORRECT_MISSPELLED`` applies it) and an
UNNAMING by the name it takes off (``WRONG_PERSON`` applies it); a listed voice without a verdict
fails ``verify``. Applied through 0029's voice-by-voice rewrite with 0031's derived surfaces; an
episode whose stored name moved on is refused untouched. Not in the registry until the list and its
verdicts exist. UNDO restores from ``.podcast_scraper/upgrade-backups/0035/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ...providers.ml.diarization import voice_identity as vi
from .. import reviewer_verdicts as rv
from ..corpus_selection import select_served_artifacts
from ..file_rewrite import undo_from_receipts
from . import m0029_naming_arc_voices_repaired as m29, m0034_talk_floor_names_removed as m34
from .m0025_one_person_one_entry import _load, _run_relative

MIGRATION_ID = "0035_voice_identity_names"
RECEIPTS_FILE = "voice_identity_names.jsonl"
BACKUP_TAG = "0035"
LIST_FILE = Path(__file__).resolve().parents[1] / "lists" / f"{MIGRATION_ID}.json"
#: The pipeline's defaults (``voice_identity_threshold``; ``apply_to_roster``'s ``min_talk``).
THRESHOLD = 0.7
MIN_TALK_S = 60.0


def episode_changes(
    meta_path: Path, threshold: float = THRESHOLD, min_talk_s: float = MIN_TALK_S
) -> Tuple[str, Dict[str, m29.Entry]]:
    """``(episode id, {voice: entry})``: what the rung changes on one served episode."""
    meta = _load(meta_path)
    if not isinstance(meta, dict):
        return "", {}
    eid = str((meta.get("episode") or {}).get("episode_id") or "")
    rel = _run_relative(
        (meta.get("content") or {}).get("transcript_file_path"), meta_path.parent.parent
    )
    if not eid or not rel.endswith(".txt"):
        return eid, {}
    loaded = _load(meta_path.parent.parent / rel.replace(".txt", ".speakers.diagnostics.json"))
    diag: Dict[str, Any] = loaded if isinstance(loaded, dict) else {}
    block = diag.get("voice_embeddings") or {}
    vecs = block.get("by_voice") or {}
    if not vecs:
        return eid, {}
    model = block.get("model_name") or None
    feed_dir = meta_path.parent.parent.parent
    groups = vi.build_groups(vi.feed_history(feed_dir, eid, model), threshold, min_talk_s)
    if not groups:
        return eid, {}
    served = m34.served_by_voice(m34._segments(meta_path, rel))
    talk = {
        str(v.get("voice")): float(v.get("talk_time_s") or 0.0) for v in diag.get("voices") or []
    }
    voices = {v: (name, role, talk.get(v, 0.0), vecs.get(v)) for v, (name, role) in served.items()}
    out: Dict[str, m29.Entry] = {}
    for voice, (new_name, new_role) in vi.decide(voices, groups, threshold).items():
        old_name, old_role = served.get(voice, (None, None))
        if new_name:
            group = next((g for g in groups if g.display == new_name), None)
            sim = (
                vi._cos(vi._unit(vecs[voice]), group.centroid) if group and vecs.get(voice) else 0.0
            )
            why = f"matches the feed's {new_name!r} fingerprint group at {sim:.2f}"
            out[voice] = (old_name, old_role, new_name, new_role or old_role, why)
        else:
            why = (
                f"clearly not {old_name!r}: the feed's fingerprint group for that name is elsewhere"
            )
            out[voice] = (old_name, old_role, None, None, why)
    return eid, out


def find_changes(root: Path) -> Dict[rv.Key, m29.Entry]:
    """The rung over every served episode of ``root`` that carries fingerprints (read only)."""
    out: Dict[rv.Key, m29.Entry] = {}
    for meta_path in select_served_artifacts(Path(root), ".metadata.json")[0]:
        eid, changes = episode_changes(meta_path)
        for voice, entry in changes.items():
            out[(eid, voice)] = entry
    return out


def load_list(path: Path = LIST_FILE) -> Dict[rv.Key, m29.Entry]:
    """The frozen list; empty when it has not been made yet."""
    return m34.load_list(path)


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class VoiceIdentityNamesMigration(m34.TalkFloorNamesRemovedMigration):
    """The voice identity rung's changes on served episodes, as listed and reviewed."""

    id = MIGRATION_ID
    to_version = "2.7.29"
    description = (
        "the voice identity rung on served episodes with fingerprints: a voice that matches the "
        "feed's voted fingerprint group takes its name, a name leaves a voice clearly not that "
        "person; frozen, reviewed list through 0029's rewrite with 0031's derived surfaces"
    )
    _receipts_file = RECEIPTS_FILE
    _backup_tag = BACKUP_TAG
    _plan_label = "voice identity"

    def listed(self) -> Dict[rv.Key, m29.Entry]:
        """This migration's frozen list (``upgrade/lists/0035_voice_identity_names.json``)."""
        return load_list()

    def review_names(self) -> Dict[rv.Key, str]:
        """``{(episode id, voice): name judged}``: a rename's new name, an unnaming's old one."""
        out: Dict[rv.Key, str] = {}
        for key, e in self.listed().items():
            name = e[2] or e[0]
            if name:
                out[key] = str(name)
        return out

    def _reviewed(self) -> Tuple[Dict[rv.Key, m29.Entry], List[rv.Key]]:
        """A rename applies on ``CORRECT`` / ``CORRECT_MISSPELLED``, an unnaming on
        ``WRONG_PERSON``; ``(entries to apply, listed entries without a verdict)``."""
        verdicts = self._verdicts()
        apply: Dict[rv.Key, m29.Entry] = {}
        missing: List[rv.Key] = []
        for key, entry in self.listed().items():
            v = verdicts.get(*key)
            if v is None:
                missing.append(key)
            elif entry[2] and v.verdict in ("CORRECT", "CORRECT_MISSPELLED"):
                apply[key] = entry
            elif not entry[2] and v.verdict == "WRONG_PERSON":
                apply[key] = entry
        return apply, missing


def verdicts_file() -> Path:
    """The verdicts file this migration reads (a plan-level input, hashed into the code digest)."""
    return rv.verdicts_path(MIGRATION_ID)


def dry_run_report(root: Path) -> Iterable[Dict[str, Any]]:
    """Read-only: one row per episode the migration would write or refuses."""
    mig = VoiceIdentityNamesMigration()
    mig._apply, _missing = mig._reviewed()
    write, refused, _totals = mig._scan(Path(root))
    for ep in write + refused:
        yield {
            "meta": str(ep.meta.relative_to(root)),
            "voices": dict(ep.voice_names),
            "refused": ep.refused,
        }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``--find <corpus copy> --out <list.json>``: replay the rung and freeze the list."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--find", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=LIST_FILE)
    a = ap.parse_args(argv)
    changes = find_changes(a.find)
    m34.write_list(a.out, changes, MIGRATION_ID)
    renames = sum(1 for e in changes.values() if e[2])
    print(f"{len(changes)} voices ({renames} named, {len(changes) - renames} unnamed) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

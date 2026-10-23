"""0034 — the talk floor on the episodes already served: a name off a voice that barely speaks.

New episodes get the talk floor in the pipeline (``speaker_name_min_talk_share`` 0.05, naming ladder
doc v26). Served episodes were written before it, and 88% of two-voice episodes were named by older
code that the pipeline will not re-run (podcast-scraper-eval-data NAMING-RETHINK). The floor needs
only what is on disk: a voice served with a name and holding less than ``FLOOR`` (5%) of the
episode's talk (its segments' seconds) loses the name. No LLM, no DGX, no reprocess.

Measured on served names (10-17 snapshot, four labelled segments): 92 names under the floor, 84 of
them bad (81 on ad / promo / clip voices, 4 of those host-read ads; 3 spurious), 1 wrong, 6 right
and 1 role error lost. The 7 right names lost are short host intros / outros and fragments, none
an ad read. On the labelled episodes :func:`find_changes` reproduces that measurement exactly;
every difference is a labelled run the corpus no longer serves, or a voice the labels do not
score.

The introduction-cue rule measured beside it (T3) is NOT here: on the runs the corpus serves its
name extraction put phrases such as "York Times" on voices and fired 11 times not right of 23
(operator, 2026-10-23: floor only).

THE LIST IS FROZEN AND REVIEWED. :func:`find_changes` runs the floor over a corpus copy and
:func:`write_list` freezes the result (``upgrade/lists/0034_talk_floor_names_removed.json``). The
migration applies that list and nothing else, through 0029's voice-by-voice rewrite with 0031's
derived surfaces (transcript prefixes and offsets, turns, diagnostics, context, ad maps, metadata
counts, GI, KG, bridge), and only the entries a fresh reviewer confirms (method step 8;
``upgrade/verdicts/0034_talk_floor_names_removed.tsv``). An entry refuses its episode, untouched,
when the stored name or role moved on since the list was made.

Not in the registry until the list and its verdicts exist: an empty list with no verdicts file
would fail ``verify`` on every upgrade run. UNDO restores from
``.podcast_scraper/upgrade-backups/0034/`` while each file is still as written.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .. import reviewer_verdicts as rv
from ..corpus_selection import select_served_artifacts
from ..file_rewrite import undo_from_receipts
from ..migration import MigrationContext, MigrationResult
from . import m0029_naming_arc_voices_repaired as m29
from .voice_rewrite_derived import DerivedSurfacesEpisode
from .m0025_one_person_one_entry import _load, _run_relative

MIGRATION_ID = "0034_talk_floor_names_removed"
RECEIPTS_FILE = "talk_floor_names_removed.jsonl"
BACKUP_TAG = "0034"
LIST_FILE = Path(__file__).resolve().parents[1] / "lists" / f"{MIGRATION_ID}.json"
#: The pipeline's floor on prod_dgx_full (``speaker_name_min_talk_share``).
FLOOR = 0.05
_LABEL = re.compile(r"^SPEAKER(_\d+)?$")


def _segments(meta: Path, rel: str) -> List[dict]:
    payload = _load(meta.parent.parent / rel.replace(".txt", ".segments.json"))
    rows = payload if isinstance(payload, list) else (payload or {}).get("segments")
    return [r for r in rows or [] if isinstance(r, dict)]


def talk_by_voice(rows: Sequence[dict]) -> Dict[str, float]:
    """Seconds each voice speaks, from its segment rows."""
    talk: Dict[str, float] = {}
    for r in rows:
        if r.get("speaker"):
            secs = max(0.0, float(r.get("end") or 0) - float(r.get("start") or 0))
            talk[str(r["speaker"])] = talk.get(str(r["speaker"]), 0.0) + secs
    return talk


def served_by_voice(rows: Sequence[dict]) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """``{voice: (its served name or None, its segment role)}``; a raw id is no name."""
    out: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for r in rows:
        v = r.get("speaker")
        if v and str(v) not in out:
            label = r.get("speaker_label") or None
            name = None if not label or _LABEL.match(str(label)) else str(label)
            out[str(v)] = (name, r.get("speaker_role") or None)
    return out


def floor_changes(rows: Sequence[dict], floor: float = FLOOR) -> Dict[str, m29.Entry]:
    """``{voice: entry}``: each named voice under ``floor`` of the episode's talk loses its name."""
    talk = talk_by_voice(rows)
    total = sum(talk.values()) or 1.0
    out: Dict[str, m29.Entry] = {}
    for voice, (name, role) in served_by_voice(rows).items():
        share = talk.get(voice, 0.0) / total
        if name and share < floor:
            out[voice] = (name, role, None, None, f"talk share {share:.1%} < {floor:.0%}")
    return out


def find_changes(root: Path) -> Dict[rv.Key, m29.Entry]:
    """The floor over every served episode of ``root`` (read only)."""
    out: Dict[rv.Key, m29.Entry] = {}
    for meta_path in select_served_artifacts(Path(root), ".metadata.json")[0]:
        meta = _load(meta_path)
        if not isinstance(meta, dict):
            continue
        eid = str((meta.get("episode") or {}).get("episode_id") or "")
        rel = _run_relative(
            (meta.get("content") or {}).get("transcript_file_path"), meta_path.parent.parent
        )
        if not eid or not rel.endswith(".txt"):
            continue
        for voice, entry in floor_changes(_segments(meta_path, rel)).items():
            out[(eid, voice)] = entry
    return out


def write_list(
    path: Path, changes: Mapping[rv.Key, m29.Entry], migration_id: str = MIGRATION_ID
) -> None:
    """Freeze ``changes`` as a migration's list (sorted, one row per voice)."""
    rows = [[eid, voice, *entry] for (eid, voice), entry in sorted(changes.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"migration": migration_id, "voices": rows}, indent=1) + "\n")


def load_list(path: Path = LIST_FILE) -> Dict[rv.Key, m29.Entry]:
    """The frozen list; empty when it has not been made yet."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[rv.Key, m29.Entry] = {}
    for r in data.get("voices") or []:
        out[(str(r[0]), str(r[1]))] = (r[2], r[3], r[4], r[5], str(r[6]))
    return out


class _Episode(DerivedSurfacesEpisode):
    """0029's voice-by-voice rewrite (unnaming included) with 0031's derived surfaces carried,
    deciding by the reviewed part of this migration's list (``voices``)."""


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class TalkFloorNamesRemovedMigration(m29.NamingArcVoicesRepairedMigration):
    """The talk floor on served episodes, as listed and reviewed.

    A fresh reviewer reads each listed voice against the name the floor would take off, with the
    verdicts' own meaning: ``WRONG_PERSON`` (the name is not this voice's) takes it off;
    ``CORRECT`` / ``CORRECT_MISSPELLED`` (it is, e.g. a host's short intro) keeps it. A listed
    voice without a verdict fails ``verify``, and ``apply`` writes nothing without a verdicts file.
    """

    id = MIGRATION_ID
    to_version = "2.7.28"
    description = (
        "the talk floor on served episodes: a name off a voice holding under 5% of the episode's "
        "talk, from a frozen, reviewed list, through 0029's voice-by-voice rewrite with 0031's "
        "derived surfaces"
    )
    _episode_class = _Episode
    _receipts_file = RECEIPTS_FILE
    _backup_tag = BACKUP_TAG
    _plan_label = "talk floor"
    _verdicts_dir: Optional[Path] = None

    def listed(self) -> Dict[rv.Key, m29.Entry]:
        """This migration's frozen list (``upgrade/lists/0034_talk_floor_names_removed.json``)."""
        return load_list()

    def review_names(self) -> Dict[rv.Key, str]:
        """``{(episode id, voice): the name the floor takes off}``, what the review judges."""
        return {k: str(e[0]) for k, e in self.listed().items() if e[0]}

    def _verdicts(self) -> rv.Verdicts:
        return rv.load(self.id, self.review_names(), self._verdicts_dir)

    def _reviewed(self) -> Tuple[Dict[rv.Key, m29.Entry], List[rv.Key]]:
        """``(entries to apply, listed entries without a verdict)``."""
        verdicts = self._verdicts()
        apply: Dict[rv.Key, m29.Entry] = {}
        missing: List[rv.Key] = []
        for key, entry in self.listed().items():
            v = verdicts.get(*key)
            if v is None:
                missing.append(key)
            elif v.verdict == "WRONG_PERSON":
                apply[key] = entry
        return apply, missing

    def _episode_kwargs(self) -> Dict[str, Any]:
        """Each episode decides by the reviewed entries only."""
        return {"voices": self._apply}

    _apply: Dict[rv.Key, m29.Entry] = {}

    def _receipt_header(self) -> Dict[str, Any]:
        """The receipts header records the verdicts the run applied."""
        return {**super()._receipt_header(), "verdicts_sha256": self._verdicts().sha256}

    def plan(self, ctx: MigrationContext) -> str:
        """Count what apply() would rewrite and what it refuses; pure read, no writes."""
        self._apply, _missing = self._reviewed()
        return super().plan(ctx)

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Pass when every listed voice has a verdict and none is left to write or refused."""
        try:
            self._apply, missing = self._reviewed()
        except rv.VerdictError as exc:
            return False, f"verdicts file unusable: {exc}"
        if missing:
            return False, f"{len(missing)} listed voice(s) without a verdict: {missing[:5]}"
        write, refused, _totals = self._scan(ctx.corpus_root)
        if write or refused:
            names = [ep.meta.name for ep in write + refused]
            return (
                False,
                f"{len(write)} episode(s) still to write, {len(refused)} refused: {names[:5]}",
            )
        return True, "every voice the review confirmed is off the floor's name"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Rewrite every episode with a reviewed voice still on its old name, with backups."""
        self._apply, _missing = self._reviewed()
        return super().apply(ctx)


def verdicts_file() -> Path:
    """The verdicts file this migration reads (a plan-level input, hashed into the code digest)."""
    return rv.verdicts_path(MIGRATION_ID)


def dry_run_report(root: Path) -> Iterable[Dict[str, Any]]:
    """Read-only: one row per episode the migration would write or refuses."""
    mig = TalkFloorNamesRemovedMigration()
    mig._apply, _missing = mig._reviewed()
    write, refused, _totals = mig._scan(Path(root))
    for ep in write + refused:
        yield {
            "meta": str(ep.meta.relative_to(root)),
            "voices": dict(ep.voice_names),
            "refused": ep.refused,
            "counts": dict(sorted(ep.counts.items())),
        }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``--find <corpus copy> --out <list.json>``: run the floor and freeze the list."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--find", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=LIST_FILE)
    a = ap.parse_args(argv)
    changes = find_changes(a.find)
    write_list(a.out, changes)
    episodes = len({k[0] for k in changes})
    print(f"{len(changes)} voices in {episodes} episodes under the {FLOOR:.0%} floor -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

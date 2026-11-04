"""0043 — a name on a voice that only reads an ad, a promo or a supporter shout-out comes off.

Today's roster code types such a voice ``commercial`` and names nobody (ad signatures, #2123; the
self-introduced promo rule, #2364). Stored episodes were named by older code, and 0034's review
kept these names because each voice introduces itself as the named person (operator's third
ruling, 2026-10-26). The operator reversed that for the listed voices on 2026-11-04. Record:
``docs/wip/changes/0043_non_participant_names_removed.md``.

THE LIST IS FROZEN AND REVIEWED, as 0034's:
``upgrade/lists/0043_non_participant_names_removed.json`` holds unnamings only, each with its class
and deciding segment in the evidence, and a fresh reviewer answers one question per voice: does it
only read an ad, a promo or a shout-out in this episode (``WRONG_PERSON`` / ``non_participant``
takes the name off; ``CORRECT`` / ``participant`` keeps it). A listed voice without a verdict fails
``verify``; no verdicts file, nothing written.

Applied through 0029's voice-by-voice rewrite with 0031's derived surfaces (roster, segment labels,
transcript prefixes and offsets, turns, diagnostics, context, ad maps, metadata counts, GI, KG,
bridge). On top of it, what the pipeline writes for such a voice: its segment rows (raw and
ad-free) and the GI nodes the rewrite hides are typed ``commercial``, not ``unknown``; the
diagnostics voice and summary census count it ``commercial``; and a name that leaves the episode
leaves the pre-1.2.0 ``detected_hosts`` / ``detected_guests`` too (those fields are the old roster's
own projection, and m0016 read them as a statement of the name). An episode whose stored name or
role moved on is refused untouched. Not in the registry until the list and its verdicts exist.
UNDO restores from ``.podcast_scraper/upgrade-backups/0043/``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ...speaker_detectors.entity_kind_votes import kind_key
from .. import reviewer_verdicts as rv
from ..file_rewrite import undo_from_receipts
from . import m0029_naming_arc_voices_repaired as m29, m0034_talk_floor_names_removed as m34
from .m0025_one_person_one_entry import _load

MIGRATION_ID = "0043_non_participant_names_removed"
RECEIPTS_FILE = "non_participant_names_removed.jsonl"
BACKUP_TAG = "0043"
LIST_FILE = Path(__file__).resolve().parents[1] / "lists" / f"{MIGRATION_ID}.json"
COMMERCIAL = "commercial"
#: An entry's evidence: its class (record section 3) and the voice's deciding segment.
_EVIDENCE = re.compile(r"^class [ABCE]: .*\bsegment \d+\b")
#: 0043's review question is "does this voice only read an ad, a promo or a shout-out here?", not
#: "is the name this voice's own": the readers are their own people, so the verdicts carry a basis
#: of their own.
BASES = {"WRONG_PERSON": ("non_participant",), "CORRECT": ("participant",)}


class ListError(ValueError):
    """The frozen list holds an entry this migration refuses to apply."""


def load_list(path: Path = LIST_FILE) -> Dict[rv.Key, m29.Entry]:
    """The frozen list; empty when it has not been made yet. Every entry is an unnaming and names
    its class and deciding segment."""
    out = m34.load_list(path)
    for key, (old, _old_role, new, _new_role, why) in out.items():
        if not old or new:
            raise ListError(f"{key}: 0043 only takes a name off; old name required, no new name")
        if not _EVIDENCE.search(why or ""):
            raise ListError(f"{key}: the entry names no class and deciding segment ({why!r})")
    return out


def _rows(payload: Any) -> List[dict]:
    rows = payload if isinstance(payload, list) else (payload or {}).get("segments")
    return [r for r in rows or [] if isinstance(r, dict)]


class _Episode(m34._Episode):
    """0034's unnaming rewrite, then the pipeline's ``commercial`` type on the unnamed voices."""

    def plan(self) -> bool:
        gi_before = self._gi_voice_types()
        planned = super().plan()
        if planned:
            gone = {v for v, name in self.voice_names.items() if name is None}
            self._type_segments(gone)
            self._type_gi(gi_before)
            self._type_diagnostics(gone)
            self._drop_left_names()
        return planned

    def _gi_path(self) -> Path:
        return self.meta.with_name(self.meta.name[: -len(".metadata.json")] + ".gi.json")

    def _gi_voice_types(self) -> Dict[str, Any]:
        gi = _load(self._gi_path())
        return {
            str(n.get("id")): (n.get("properties") or {}).get("speaker_voice_type")
            for n in (gi or {}).get("nodes") or []
            if isinstance(n, dict)
        }

    def _type_segments(self, gone: Set[str]) -> None:
        for path, payload in self.json_files.items():
            if not path.name.endswith(".segments.json"):
                continue
            for r in _rows(payload):
                if str(r.get("speaker")) in gone and r.get("voice_type") == "unknown":
                    r["voice_type"] = COMMERCIAL
                    self._bump("segments_typed_commercial")

    def _type_gi(self, before: Dict[str, Any]) -> None:
        gi = self.json_files.get(self._gi_path())
        for node in (gi or {}).get("nodes") or []:
            props = node.get("properties") if isinstance(node, dict) else None
            if (
                isinstance(props, dict)
                and node.get("type") in ("Quote", "Insight")
                and props.get("speaker_voice_type") == "unknown"
                and before.get(str(node.get("id"))) != "unknown"
            ):
                props["speaker_voice_type"] = COMMERCIAL
                self._bump("gi_nodes_typed_commercial")

    def _type_diagnostics(self, gone: Set[str]) -> None:
        for path, diag in self.json_files.items():
            if not path.name.endswith(".speakers.diagnostics.json") or not isinstance(diag, dict):
                continue
            raw = diag.get("summary")
            summary: Dict[str, Any] = raw if isinstance(raw, dict) else {}
            for v in diag.get("voices") or []:
                if not isinstance(v, dict) or str(v.get("voice")) not in gone:
                    continue
                old = v.get("voice_type")
                if old == COMMERCIAL:
                    continue
                v["voice_type"] = COMMERCIAL
                self._bump("diagnostics_voices_typed_commercial")
                _move_census(summary, old, float(v.get("talk_time_s") or 0.0))

    def _drop_left_names(self) -> None:
        meta = self.json_files.get(self.meta)
        content = (meta or {}).get("content") if isinstance(meta, dict) else None
        if not isinstance(content, dict):
            return
        removed = {kind_key(str(e[0])) for e in self._map_entries() if e[0]}
        still = {
            kind_key(str((s or {}).get("name") or ""))
            for s in content.get("speakers") or []
            if isinstance(s, dict)
        }
        left = removed - still
        for field in ("detected_hosts", "detected_guests"):
            values = content.get(field)
            if not isinstance(values, list):
                continue
            kept = [x for x in values if not (isinstance(x, str) and kind_key(x) in left)]
            if len(kept) != len(values):
                content[field] = kept
                self._bump(f"{field}_removed", len(values) - len(kept))

    def _map_entries(self) -> Iterable[m29.Entry]:
        meta = _load(self.meta) or {}
        episode_id = str((meta.get("episode") or {}).get("episode_id") or "")
        return [e for (eid, voice), e in self._map.items() if eid == episode_id]


def _move_census(summary: Dict[str, Any], old: Optional[str], talk_s: float) -> None:
    """Move one voice from ``old`` to ``commercial`` in the diagnostics summary's census."""
    by_type = summary.get("by_voice_type")
    if isinstance(by_type, dict):
        if old and isinstance(by_type.get(old), int) and by_type[old] > 0:
            by_type[old] -= 1
        by_type[COMMERCIAL] = int(by_type.get(COMMERCIAL) or 0) + 1
    census = summary.get("voice_census")
    if not isinstance(census, dict):
        return
    total = sum(
        float((c or {}).get("talk_s") or 0.0) for c in census.values() if isinstance(c, dict)
    )
    for kind, sign in ((old, -1), (COMMERCIAL, 1)):
        if not kind:
            continue
        c = census.setdefault(kind, {"count": 0, "talk_s": 0.0, "talk_share": 0.0})
        if not isinstance(c, dict):
            continue
        c["count"] = max(0, int(c.get("count") or 0) + sign)
        c["talk_s"] = round(max(0.0, float(c.get("talk_s") or 0.0) + sign * talk_s), 1)
    if total > 0:
        for c in census.values():
            if isinstance(c, dict):
                c["talk_share"] = round(float(c.get("talk_s") or 0.0) / total, 4)


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class NonParticipantNamesRemovedMigration(m34.TalkFloorNamesRemovedMigration):
    """Names off voices that only read an ad, a promo or a shout-out, as listed and reviewed."""

    id = MIGRATION_ID
    to_version = "2.7.37"
    description = (
        "a name off a served voice that only reads an ad, a promo or a supporter shout-out "
        "(today's roster types it commercial): frozen, reviewed list through 0029's rewrite with "
        "0031's derived surfaces, the voice typed commercial"
    )
    _episode_class = _Episode
    _receipts_file = RECEIPTS_FILE
    _backup_tag = BACKUP_TAG
    _plan_label = "non-participant names"
    _verified = "every voice the review confirmed as a non-participant is unnamed and commercial"

    def listed(self) -> Dict[rv.Key, m29.Entry]:
        """This migration's frozen list (``upgrade/lists/<id>.json``)."""
        return load_list()

    def _verdicts(self) -> rv.Verdicts:
        """The frozen list's verdicts, with their basis columns and 0043's own bases."""
        return rv.load(
            self.id,
            self.review_names(),
            self._verdicts_dir,
            listed=self.listed(),
            require_basis=True,
            bases=BASES,
        )


def verdicts_file() -> Path:
    """The verdicts file this migration reads (a plan-level input, hashed into the code digest)."""
    return rv.verdicts_path(MIGRATION_ID)


def list_file() -> Path:
    """The frozen list this migration applies (a plan-level input, hashed into the code digest)."""
    return LIST_FILE


def dry_run_report(root: Path) -> Iterable[Dict[str, Any]]:
    """Read-only: one row per episode it would write or refuses (0034's ``plan_rows``)."""
    return m34.plan_rows(root, NonParticipantNamesRemovedMigration())

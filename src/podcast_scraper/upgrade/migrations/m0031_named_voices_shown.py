"""0031 — the voices 0029 named, shown the way a fresh run shows a named person (#2380).

0029 put its names on with m0025's voice-by-voice rewrite, which until 1bc9d18d0 kept the hiding
an unnamed voice carries: the voice's segments stayed typed ``unknown``, its quotes kept
``speaker_voice_type``, and every insight whose first quote it speaks stayed ``surfaceable: false``
and routed to connect. Measured on the 2026-10-16 snapshot: 1,779 insights named but hidden in the
73 episodes 0029 wrote; their pre-0029 backups have none. Two more of 0029's episodes were
reprocessed after it ran (2026-10-13, 2026-10-14) and came back with the voice unnamed again, so
0029's own verify lists them.

0029 is in every applied ledger and never runs again, so this migration carries its list
(``NAMING_ARC_VOICES``) once more, through the fixed rewrite. Per listed voice that 0029 names:

- stored as the replay read it: named now, exactly as 0029 would name it today;
- stored with the new name, its segments still typed: the name re-given through the same
  rewrite, which changes no label and lifts the hiding;
- stored with the new name and untyped: done (a second run writes nothing);
- anything else: the episode is refused, untouched and listed.

ORDER. This is the part of 0029 that went missing, so it takes the number right after 0030. The
naming re-derivation that held this number until 2026-10-18 is now 0032 (held, #2380); it never
ran on prod, and the hold was moved with it.

REVIEW (revision B, 2026-10-18). The step-8 reviewer's verdicts are an input
(``upgrade/reviewer_verdicts.py``, ``upgrade/verdicts/0031_named_voices_shown.tsv``): an entry read
as WRONG_PERSON is UNNAMED through m0025's own unnaming path (0029 put a wrong name on that voice),
an entry with no verdict is refused as ``unreviewed``, and CORRECT / CORRECT_MISSPELLED entries are
applied as listed. The first review found one wrong person in 53 voices (a guest stored as
the host); no automatic rule separates that case yet, so the reviewer's read decides until a rule is
measured on gold (S7).

``verify`` checks the OUTCOME, not only the labels: every insight whose first quote a shown voice
speaks is visible, an unnamed voice carries the name nowhere, and no entry is unreviewed. UNDO
restores from
``.podcast_scraper/upgrade-backups/0031/`` while each file is still what this migration wrote.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ...gi.pipeline import _segment_char_spans
from ...identity.slugify import person_id
from .. import reviewer_verdicts as rv
from ..corpus_selection import select_served_artifacts
from ..file_rewrite import undo_from_receipts
from ..migration import MigrationContext
from . import m0029_naming_arc_voices_repaired as m29
from .m0025_one_person_one_entry import _load, _run_relative
from .voice_rewrite_derived import admap_matches, DerivedSurfacesEpisode

MIGRATION_ID = "0031_named_voices_shown"
RECEIPTS_FILE = "named_voices_shown.jsonl"
BACKUP_TAG = "0031"
HIDING = ("unknown", "unidentified")


def _rows(payload: Any) -> List[dict]:
    rows = payload if isinstance(payload, list) else (payload or {}).get("segments")
    return [r for r in rows or [] if isinstance(r, dict)]


def _typed_voices(meta: Path, rel: str) -> Set[str]:
    """Voices with any segment carrying a ``voice_type`` (a named person carries none)."""
    payload = _load(meta.parent.parent / rel.replace(".txt", ".segments.json"))
    return {str(r["speaker"]) for r in _rows(payload) if r.get("speaker") and r.get("voice_type")}


def _names(entries: Dict[rv.Key, m29.Entry]) -> Dict[rv.Key, str]:
    """``{(episode id, voice): listed new name}`` for the entries this migration applies."""
    return {k: str(e[2]) for k, e in entries.items() if e[2]}


class _Episode(DerivedSurfacesEpisode):
    """0029's decision through the fixed rewrite, decided by the review's verdicts; the ad maps
    and metadata counts are carried with it (``voice_rewrite_derived``)."""

    def __init__(
        self,
        meta: Path,
        root: Path,
        voices: Optional[Dict] = None,
        verdicts: Optional[rv.Verdicts] = None,
    ) -> None:
        super().__init__(meta, root, voices)
        self._verdicts = verdicts

    def decide(self) -> bool:
        meta = _load(self.meta)
        if not isinstance(meta, dict):
            return False
        episode_id = str((meta.get("episode") or {}).get("episode_id") or "")
        mine = {v: e for (eid, v), e in self._map.items() if eid == episode_id and e[2]}
        if not mine:
            return False
        rel = _run_relative(
            (meta.get("content") or {}).get("transcript_file_path"), self.meta.parent.parent
        )
        if not rel.endswith(".txt"):
            self.refused = "no transcript"
            return True
        self._load_seg_voices(rel)
        stored = m29._stored_by_voice(self.meta, rel)
        typed = _typed_voices(self.meta, rel)
        for voice, (old_name, old_role, new_name, new_role, _why) in sorted(mine.items()):
            verdict = self._verdicts.get(episode_id, voice) if self._verdicts else None
            name, role = stored.get(voice, (None, None))
            name = None if not name or m29._DIARIZATION_LABEL.match(name) else name
            if verdict is not None and verdict.verdict == "WRONG_PERSON":
                if name is None and old_name is None:
                    continue  # never named, or already unnamed: nothing to take off
                if name is not None and name == old_name:
                    continue  # already back to the listed old name: done
                if name != new_name:
                    self.refused = f"{voice}: stored {name!r}, the review judged {new_name!r}"
                    return True
                # The review read the listed name as someone else's: the voice goes back to what
                # it was before 0029 named it, the listed old name and role (None: unnamed).
                self._bump("unnamed_by_review")
                self.voice_names[voice] = old_name
                self.voice_roles_restored[voice] = old_role
                continue
            if (name, role) == (new_name, new_role) and voice not in typed:
                continue  # done: named and shown, nothing for this migration to change
            if (name, role) != (new_name, new_role) and (name, role) != (old_name, old_role):
                self.refused = (
                    f"{voice}: stored {name!r}/{role!r}, the replay read {old_name!r}/{old_role!r}"
                )
                return True
            if verdict is None:
                self._bump("unreviewed")
                self.refused = f"{voice}: unreviewed (no verdict for {new_name!r})"
                return True
            if (name, role) == (new_name, new_role):
                self._bump("voices_shown")
            self.voice_names[voice] = new_name
            if new_role:
                self.voice_roles[voice] = new_role
        return bool(self.voice_names)


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


def _voice_at(spans: Optional[list], char_start: Any) -> Optional[str]:
    if not spans or not isinstance(char_start, int):
        return None
    for a, b, seg in spans:
        if a <= char_start < b:
            return str(seg.get("speaker") or "") or None
    return None


def _written_metas(root: Path) -> Set[str]:
    """Metadata files this migration's receipts say it wrote (every run)."""
    path = root / RECEIPTS_FILE
    if not path.is_file():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        rel = str((json.loads(line) if line.strip() else {}).get("relpath") or "")
        if rel.endswith(".metadata.json"):
            out.add(str((root / rel).resolve()))
    return out


def _serves_nothing(root: Path) -> Optional[str]:
    """Why a root that serves no episode must not pass, or ``None`` for an empty corpus.

    A wrong root must not pass vacuously, and a fresh install (a manifest, no episodes) must pass,
    as every migration's verify does there. Metadata on disk that none is served from means the
    selection read nothing (a broken manifest, a selection bug), not an empty corpus."""
    if not isinstance(_load(root / "corpus_manifest.json"), dict):
        return "has no corpus manifest (absent or not a JSON object)"
    if next((root / "feeds").rglob("*.metadata.json"), None) is not None:
        return "has metadata on disk, none served"
    return None


class _Served:
    """One served listed episode as verify reads it: segments, transcripts, GI, diagnostics."""

    def __init__(self, meta_path: Path, rel: str) -> None:
        self.meta_path = meta_path
        self.run = meta_path.parent.parent
        self.rel = rel
        self.texts: Dict[str, Optional[list]] = {}
        self.segs: Dict[str, List[dict]] = {}
        for sfx, seg_sfx in ((".txt", ".segments.json"), (".adfree.txt", ".adfree.segments.json")):
            tp = self.run / rel.replace(".txt", sfx)
            segs = _rows(_load(self.run / rel.replace(".txt", seg_sfx)))
            self.segs[tp.name] = segs
            body = tp.read_text(encoding="utf-8") if tp.is_file() else ""
            self.texts[tp.name] = _segment_char_spans(body, segs) if body and segs else None
        gi = _load(Path(str(meta_path)[: -len(".metadata.json")] + ".gi.json")) or {}
        self.nodes = {str(n.get("id")): n for n in gi.get("nodes") or [] if isinstance(n, dict)}
        self.first: Dict[str, str] = {}
        for e in gi.get("edges") or []:
            if isinstance(e, dict) and e.get("type") == "SUPPORTED_BY":
                self.first.setdefault(str(e.get("from")), str(e.get("to")))
        self.diag = _load(self.run / rel.replace(".txt", ".speakers.diagnostics.json")) or {}

    def voice_of(self, quote_id: str) -> Optional[str]:
        props = (self.nodes.get(quote_id) or {}).get("properties") or {}
        ref = Path(str(props.get("transcript_ref") or "")).name
        return _voice_at(self.texts.get(ref), props.get("char_start"))

    def diag_voice(self, voice: str) -> List[dict]:
        return [
            d
            for d in self.diag.get("voices") or []
            if isinstance(d, dict) and d.get("voice") == voice
        ]

    def shown_problems(self, voice: str, where: str) -> Tuple[List[str], int]:
        """Insights whose FIRST quote the voice speaks and that are still hidden (and how many)."""
        problems, checked = [], 0
        for iid, n in self.nodes.items():
            if n.get("type") != "Insight" or self.voice_of(self.first.get(iid, "")) != voice:
                continue
            checked += 1
            props = n.get("properties") or {}
            if props.get("surfaceable") is False and props.get("speaker_voice_type") in HIDING:
                problems.append(
                    f"{where}: insight {iid} still hidden ({props['speaker_voice_type']})"
                )
        return problems, checked

    def unnamed_problems(self, voice: str, entry: m29.Entry, where: str) -> List[str]:
        """The wrong name left anywhere on the voice, or a role other than the listed old one."""
        name, want_role = str(entry[2]), entry[1]  # the listed old role (None: no role)
        problems = []
        for f, segs in self.segs.items():
            if any(r.get("speaker") == voice and r.get("speaker_role") != want_role for r in segs):
                problems.append(f"{where}: {f} gives the voice a role other than {want_role!r}")
        for dv in self.diag_voice(voice):
            if dv.get("role") != (want_role or "unknown"):
                problems.append(f"{where}: diagnostics give the voice role {dv.get('role')!r}")
            if dv.get("resolved_name") == name:
                problems.append(f"{where}: diagnostics still resolve the voice to {name!r}")
        for f, segs in self.segs.items():
            if any(r.get("speaker") == voice and r.get("speaker_label") == name for r in segs):
                problems.append(f"{where}: {f} still labels the voice {name!r}")
            others = any(r.get("speaker") != voice and r.get("speaker_label") == name for r in segs)
            tp = self.run / self.rel.replace(".txt", ".adfree.txt" if "adfree" in f else ".txt")
            if (
                not others
                and tp.is_file()
                and f"\n{name}:" in "\n" + tp.read_text(encoding="utf-8")
            ):
                problems.append(f"{where}: {tp.name} still prefixes lines with {name!r}")
        pid = person_id(name)
        for qid, n in self.nodes.items():
            props = n.get("properties") or {}
            if (
                n.get("type") == "Quote"
                and self.voice_of(qid) == voice
                and props.get("speaker_id") == pid
            ):
                problems.append(f"{where}: quote {qid} still {pid}")
            if (
                n.get("type") == "Insight"
                and self.voice_of(self.first.get(qid, "")) == voice
                and props.get("speaker") == name
            ):
                problems.append(f"{where}: insight {qid} still spoken by {name!r}")
        return problems


def _outcome(root: Path, verdicts: rv.Verdicts) -> Tuple[List[str], Dict[str, int]]:
    """What verify must see on the served episodes: ``(problems, counts)``.

    An entry the migration would still change shows up as written or refused in the plan; this
    checks the state left behind. Shown or named entries: no insight whose FIRST supporting quote
    the voice speaks (the pipeline's rule, ``gi.pipeline._apply_voice_flags``) is hidden as unknown
    or unidentified. Unnamed-by-review entries: the wrong name on none of the voice's segments,
    quotes, insights or diagnostics (or transcript lines when no other voice carries it), and the
    listed old role. Unreadable metadata fails; a corpus serving no listed episode passes.
    """
    problems: List[str] = []
    counts = {
        "insights_checked": 0,
        "unnamed_checked": 0,
        "unreviewed_unchanged": 0,
        "admaps_checked": 0,
        "served": 0,
        "served_listed": 0,
        "metadata_unreadable": 0,
    }
    entries = {k: e for k, e in m29.NAMING_ARC_VOICES.items() if e[2]}
    written = _written_metas(root)
    for meta_path in select_served_artifacts(root, ".metadata.json")[0]:
        counts["served"] += 1
        meta = _load(meta_path)
        if not isinstance(meta, dict):
            # Never "not listed": an unreadable record could be one of the list's (it fails).
            counts["metadata_unreadable"] += 1
            problems.append(f"{meta_path.name}: metadata unreadable")
            continue
        eid = str((meta.get("episode") or {}).get("episode_id") or "")
        mine = {v: e for (k, v), e in entries.items() if k == eid}
        if not mine:
            continue
        counts["served_listed"] += 1
        rel = _run_relative(
            (meta.get("content") or {}).get("transcript_file_path"), meta_path.parent.parent
        )
        if not rel.endswith(".txt"):
            continue
        ep = _Served(meta_path, rel)
        matches = admap_matches(ep.run / rel)
        if matches is not None:
            counts["admaps_checked"] += 1
            if not matches:
                problems.append(f"{meta_path.name}: the ad map does not match its transcript")
        for voice, e in sorted(mine.items()):
            v = verdicts.get(eid, voice)
            where = f"{meta_path.name} {voice}"
            if v is None and str(meta_path.resolve()) in written:
                problems.append(f"{where}: written by this migration, and has no verdict")
                continue
            if v is not None and v.verdict == "WRONG_PERSON":
                counts["unnamed_checked"] += 1
                problems += ep.unnamed_problems(voice, e, where)
                continue
            if v is None:
                # An entry this migration would change is refused while unreviewed (the plan and
                # verify say so); one it leaves alone is checked like any shown voice.
                counts["unreviewed_unchanged"] += 1
            found, checked = ep.shown_problems(voice, where)
            counts["insights_checked"] += checked
            problems += found
    return problems, counts


class NamedVoicesShownMigration(m29.NamingArcVoicesRepairedMigration):
    """0029's voices through the fixed rewrite, decided by the step-8 review's verdicts."""

    id = MIGRATION_ID
    to_version = "2.7.25"
    description = (
        "0029's voices through the fixed voice-by-voice rewrite (#2380), decided by the step-8 "
        "review: named where a reprocess unnamed them again, shown where still hidden under their "
        "name, unnamed where the review read the name as another person's"
    )
    _episode_class = _Episode
    _receipts_file = RECEIPTS_FILE
    _backup_tag = BACKUP_TAG
    _verdicts_dir: Optional[Path] = None

    def review_names(self) -> Dict[rv.Key, str]:
        """``{(episode id, voice): listed name}`` a review of this migration judges."""
        return _names(m29.NAMING_ARC_VOICES)

    def _verdicts(self) -> rv.Verdicts:
        return rv.load(MIGRATION_ID, self.review_names(), self._verdicts_dir)

    def _episode_kwargs(self) -> Dict[str, Any]:
        return {"verdicts": self._loaded}

    def _scan(self, root: Path) -> Tuple[List[m29._Episode], List[m29._Episode], Dict[str, int]]:
        self._loaded = self._verdicts()
        return super()._scan(root)

    def plan(self, ctx: MigrationContext) -> str:
        """Counts, plus the verdicts file it decided by (its sha256)."""
        text = super().plan(ctx)
        return f"{text}; verdicts {self._loaded.sha256[:12]} ({len(self._loaded.rows)} rows)"

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Pass when nothing is left to write or refused AND the outcome holds (``_outcome``)."""
        try:
            ok, message = super().verify(ctx)
        except rv.VerdictError as exc:
            return False, f"verdicts file unusable: {exc}"
        problems, counts = _outcome(ctx.corpus_root, self._loaded)
        tally = ", ".join(f"{k}={v}" for k, v in counts.items())
        if not ok or problems:
            parts = [message] if not ok else []
            if problems:
                parts.append(f"{len(problems)} outcome problem(s): {problems[:5]}")
            return False, f"{'; '.join(parts)}; {tally}"
        if counts["served"] == 0:
            why = _serves_nothing(Path(ctx.corpus_root))
            if why:
                return False, f"{ctx.corpus_root} serves no episode and {why}; {tally}"
        if counts["served_listed"] == 0:
            # A corpus that serves none of the list's episodes (a fixture, a fresh corpus): the
            # migration has nothing here to have done. Any served listed episode is checked below.
            empty = "an empty corpus: " if counts["served"] == 0 else ""
            return True, f"{empty}no listed episode is served here; {tally}"
        if counts["insights_checked"] + counts["unnamed_checked"] == 0:
            return False, f"verify checked nothing; {tally}"
        return True, f"every voice 0029 names carries its name and no hiding, as reviewed; {tally}"

    def _receipt_header(self) -> Dict[str, Any]:
        return {**super()._receipt_header(), "verdicts_sha256": self._loaded.sha256}

    def apply(self, ctx: MigrationContext) -> Any:
        """0029's apply, with the verdicts file's sha256 in the result and the receipt header."""
        result = super().apply(ctx)
        result.details["verdicts_sha256"] = self._loaded.sha256
        return result


def verdicts_file() -> Path:
    """The verdicts file this migration reads (a plan-level input, hashed into the code digest)."""
    return rv.verdicts_path(MIGRATION_ID)


def dry_run_report(root: Path) -> Iterable[Dict[str, Any]]:
    """Read-only: one row per episode the migration would write or refuses (the outcome hook's
    plan contract): every counter, and the sha256 of every file its decision read or looked for
    (``read_recorder``; ``"absent"`` for a file looked for and missing)."""
    from ..read_recorder import recording

    root = Path(root)
    mig = NamedVoicesShownMigration()
    mig._loaded = mig._verdicts()
    for meta in select_served_artifacts(root, ".metadata.json")[0]:
        ep = _Episode(meta, root, verdicts=mig._loaded)
        with recording(root) as rec:
            planned = ep.plan()
        if not planned and not ep.refused:
            continue
        if ep.refused:
            ep._bump("refused")  # as ``_scan`` counts it, so rows sum to apply's totals
        yield {
            "meta": str(meta.relative_to(root)),
            "voices": dict(ep.voice_names),
            "refused": ep.refused,
            "counts": dict(sorted(ep.counts.items())),
            "inputs": rec.inputs(),
            "verdicts_sha256": mig._loaded.sha256,
        }

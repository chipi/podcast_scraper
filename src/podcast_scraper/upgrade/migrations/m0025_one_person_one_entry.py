"""0025 — one person is one entry on an episode, whatever their title or spelling (c05cc0273).

The roster compared a leading title as the given name, and the record de-duplicated its unplaced
people with an exact-surname check only. So one person was published twice: a voice under the
title or spelling it said, and the feed's or the show notes' spelling beside it. Prod 2026-10-08:
22 served episodes ("Professor Hannah Frye" placed + "Hannah Fry" unplaced, "Traci Alloway" guest
+ "Tracy Alloway" host, "Bernard Leong" + "Bernard Leung", "Ali Ghodsi" twice, ...). The pipeline
no longer does this; this repairs what is on disk, with the pipeline's own predicates.

Two steps per episode, decided from its stored record and its feed's stated hosts (the speaker
diagnostics' ``tried.known_hosts``):

1. A PLACED name takes the stated host's spelling, and the host role, where the roster now snaps
   it (``roster._snap_near_identical_host``; for a voice the record calls a host, also
   ``_canonicalize_to_known_host``) — and only where this fix is the reason it did not before:
   the name carries a leading title, or the record also lists that host unplaced. Every DeepMind
   episode's "Professor Hannah Fry" becomes the feed's "Hannah Fry" (the person id does not move);
   Odd Lots' "Traci Alloway" guest becomes Tracy Alloway, host.
2. An UNPLACED entry the record already lists as a kept person
   (``_same_person`` or ``_same_person_on_one_episode``, as ``_unplaced_speakers`` now decides) is
   dropped; two placed entries that end with one name become one entry with both voices.

3. Per VOICE, for what a name map cannot express:
   * a SWAP: a voice's own self-introduction of a stated host, while the introduction reader had
     put that host's name on another voice. The self-introduced voice takes the host; the other
     takes the name the LLM gave it if the record lists that person unplaced, else none (DeepMind
     0014: Hannah Fry back on her own voice, Paige Bailey on the guest's);
   * a FORCED name (arithmetic) that is the same person as a name another voice holds by evidence
     is removed from its voice (The Long Run: "Andy Ratcliffe" forced onto the co-guest Yung Lie);
   * voices OLDER code named wrongly where today's roster, already fixed, names them right
     (``REPLAYED_VOICES``: two ad reads, three guests stored as the host, a co-host stored
     unnamed; each read in the replay and in the transcript);
   * two placed voices left with two spellings of one person go to the pipeline's own
     ``roster._one_name_per_person``, fed the stored labels, roles, talk and turn alternations; a
     pair it keeps apart (they converse) stays listed in ``left``.

An OLDER record (no ``voices`` / ``placed`` fields; half the corpus on 2026-10-08) is read through
its segments: each entry is placed on the voices its label is on, as the record builder reads it.

A rewrite reaches every surface that carries the name, all or none per episode: metadata
(``content.speakers``, ``detected_*``), ``.segments.json`` / ``.adfree.segments.json`` labels and
``speaker_role``, ``.kg.json`` / ``.gi.json`` / ``.bridge.json`` names and ids (m0017's rewrite,
ids move with the name; per voice, each GI quote is re-credited by the segment it sits in) plus the
Person node ``label``, the KG Person role, the speaker diagnostics, the context digest's speaker
fields, the three transcripts (re-rendered from the new segments with every GI quote and ad-free
offset carried across — m0023's alignment, so lines that become one speaker's merge — or m0023's
in-place rename where the file is not a pure render), and each variant's ``turns.json``, rebuilt by
the pipeline's own builder (or the episode is refused).

REFUSED, untouched and listed: a rename onto a name another voice holds in a different role that is
not a swap, one stated host claimed by two voices, a transcript neither m0023 path can rewrite, or a
quote whose voice cannot be told.

TWO MORE SPELLINGS the roster now reaches, so the record does too:

* a voice that PRESENTS the show (the roster's own presenter evidence, run on the stored segments)
  and introduces itself with a respelling of the one stated host no other voice holds, and whom
  the episode does not name as a participant, is that host: Empire's "Hello and welcome to Empire
  with me, Anita Arnond", published as a guest beside an unplaced "Anita Anand", becomes Anita
  Anand, host. "I'm Kevin Ross" on Kevin Roose's show never presents it, so it stays a guest.
* a host-pool name the episode ALSO names as a participant is a polluted pool, not a host: the
  voice takes the stated spelling and keeps its own role (The a16z Show: "Lucas Kaiser" -> Lukasz
  Kaiser, guest); and any placed name that is a respelling of ONE stated participant takes its
  spelling (``roster._stated_participant_spelling``: "Charming Lai" -> Chiamin Lai). A name the
  episode already states is never re-spelt.

NOT DONE: a voice an OLDER record names differently from what today's roster would, where no
one-person rule is at stake (a guest's voice carrying the host's name, a host left unnamed). That
is a re-derivation of the voice, not a repair of one person's entries.

``verify`` fails while any served episode still has something this migration would write OR
refuses, so a refusal never reads as done. UNDO restores from ``.podcast_scraper/upgrade-backups/
0025/`` while each file is still exactly what this migration wrote.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ...builders.context_digest_builder import build_context_digest
from ...identity.bare_name_scope import is_scoped_person_id
from ...identity.slugify import canonical_person_name, person_id
from ...providers.ml.diarization import roster as R
from ...providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from ...providers.ml.diarization.formatting import format_diarized_screenplay_with_offsets
from ...workflow.turns_artifact import build_turns_document, TURNS_SUFFIX
from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, dump_json, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult
from .m0017_speaker_names_canonicalised import _Episode as _RenameEpisode, _person_nodes
from .m0023_transcript_speaker_prefixes_resynced import (
    _Episode as _TranscriptEpisode,
    _write_text_with_backup,
    Refused,
)

MIGRATION_ID = "0025_one_person_one_entry"
RECEIPTS_FILE = "one_person_one_entry.jsonl"
BACKUP_TAG = "0025"
_CONTEXT_FIELDS = (("basic", "hosts"), ("basic", "guests"), (None, "people"))


def _load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def same_person(a: str, b: str) -> bool:
    """One person on one episode's record — the predicate ``_unplaced_speakers`` now uses."""
    return a.lower() == b.lower() or R._same_person(a, b) or R._same_person_on_one_episode(a, b)


def _has_title(name: str) -> bool:
    toks = name.split()
    return len(toks) > 2 and toks[0].lower().strip(".,") in R.HONORIFIC_TITLES


def host_spelling(
    name: str, role: Optional[str], hosts: List[str], taken: Set[str]
) -> Optional[str]:
    """The stated host *name* is, by the roster's own host rules, or ``None``."""
    snapped = R._snap_near_identical_host(name, hosts, frozenset(taken))
    if snapped != name:
        return snapped
    if role == "host":
        canon = R._canonicalize_to_known_host(name, hosts)
        if canon != name:
            return canon
    return None


def _unclaimed_host(
    name: str, hosts: List[str], taken: Set[str], participants_lower: Set[str]
) -> Optional[str]:
    """The one stated host a PRESENTING voice's name respells, unclaimed and not a participant
    (``roster._presenter_takes_unclaimed_host_spelling``)."""
    if name.lower() in {h.lower() for h in hosts}:
        return None
    matches = [
        h
        for h in hosts
        if h.lower() not in participants_lower
        and R._same_person_on_one_episode(name, h)
        and not any(R._same_person_on_one_episode(o, h) for o in taken)
    ]
    return matches[0] if len(matches) == 1 else None


def _stated_participant(
    name: str, hosts: List[str], taken: Set[str], participants_lower: Set[str]
) -> Optional[str]:
    """A stated spelling the roster's recovery pass now reaches: a host-pool name the episode
    ALSO names as a participant (``roster._recover_stated_names``, ``participants``). The role is
    the voice's own; only the spelling moves."""
    refs = [h for h in hosts if h.lower() in participants_lower]
    if not refs or name.lower() in {r.lower() for r in refs}:
        return None
    canon = R._canonicalize_to_stated_name(name, refs)
    if canon == name or canon in taken:
        return None
    return canon


def _segment_spans(text_path: Path, payload: Any) -> Optional[List[Tuple[int, int, str]]]:
    """``(char_start, char_end, voice)`` of every segment in *text_path*, or ``None``.

    The ad-free sidecar stores its offsets; the raw one does not, so its text is re-rendered with
    the pipeline's formatter and used only when that render IS the file.
    """
    rows = payload if isinstance(payload, list) else (payload or {}).get("segments")
    rows = [r for r in rows or [] if isinstance(r, dict) and r.get("speaker")]
    if not rows or not text_path.is_file():
        return None
    if all(isinstance(r.get("char_start"), int) for r in rows):
        return [(int(r["char_start"]), int(r["char_end"]), str(r["speaker"])) for r in rows]
    text, emitted = format_diarized_screenplay_with_offsets(rows)
    if text != text_path.read_text(encoding="utf-8"):
        return None
    ordered = [
        r
        for r in sorted(rows, key=lambda r: float(r.get("start") or 0.0))
        if (r.get("text") or "").strip()
    ]
    if len(ordered) != len(emitted):
        return None
    return [
        (int(e["char_start"]), int(e["char_end"]), str(r["speaker"]))
        for r, e in zip(ordered, emitted)
    ]


def _quote_voices(
    gi: Any,
    run: Path,
    rel: str,
    segs: Dict[Path, Any],
    voice_of_label: Dict[str, str],
    shared_labels: Set[str],
) -> Optional[Dict[str, str]]:
    """``{quote id: voice}`` for every quote, by WHERE it sits: the segment whose span holds its
    first character. A label is no key when two voices carry it (a forced twin of the guest), so
    the line-prefix fallback — used only where no span map exists — refuses such a label
    (``None``)."""
    spans: Dict[str, Optional[List[Tuple[int, int, str]]]] = {}
    for sfx, seg_sfx in ((".txt", ".segments.json"), (".adfree.txt", ".adfree.segments.json")):
        text_path = run / rel.replace(".txt", sfx)
        spans[text_path.name] = _segment_spans(
            text_path, segs.get(run / rel.replace(".txt", seg_sfx))
        )
    out: Dict[str, str] = {}
    texts: Dict[str, str] = {}
    tdir = (run / rel).parent
    for node in (gi or {}).get("nodes") or []:
        props = (node.get("properties") or {}) if isinstance(node, dict) else {}
        if node.get("type") != "Quote" or not isinstance(props.get("char_start"), int):
            continue
        ref = Path(str(props.get("transcript_ref") or "")).name
        cs = int(props["char_start"])
        span_map = spans.get(ref)
        if span_map is not None:
            hit = next((v for a, b, v in span_map if a <= cs < b), None)
            if hit is None:
                # A quote can start on the separator before its first word.
                hit = next((v for a, b, v in span_map if cs < a), None)
            if hit is not None:
                out[str(node["id"])] = hit
            continue
        if ref not in texts:
            path = tdir / ref
            texts[ref] = path.read_text(encoding="utf-8") if path.is_file() else ""
        line = texts[ref][:cs].rsplit("\n", 1)[-1]
        label = line.split(": ", 1)[0] if ": " in line else None
        if label in shared_labels:
            return None
        if label in voice_of_label:
            out[str(node["id"])] = voice_of_label[label]
    return out


#: Voices OLDER code named wrongly whose stored name today's roster replaces — every one of them a
#: defect already fixed going forward, so this is a one-time cleanup of old data, closed at these
#: entries. ``(episode id, voice): (name today's roster gives, its role, evidence)``; a ``None``
#: name leaves the voice unnamed. Each was read in the 2026-10-08 roster replay (today's code vs
#: the stored record) AND against the voice's own transcript lines. Voices where today's roster is
#: itself wrong (Hard Fork, In Our Time's interviewer) are deliberately absent: those need a code
#: fix, not a data one.
REPLAYED_VOICES: Dict[Tuple[str, str], Tuple[Optional[str], Optional[str], str]] = {
    ("a4f76752-381a-11f1-a32d-2b6d4bd14e72", "SPEAKER_05"): (
        None,
        None,
        "The Gray Area 'Is starting a family becoming impossible?': an ad read ('Support for the "
        "show comes from Talk High'), forced the host's spoken 'Anna-Louise Sussman' (2b7aab46e)",
    ),
    ("718445c2-4032-448e-9b6f-4d8b71d947cb", "SPEAKER_00"): (
        None,
        None,
        "Made In Africa 'Sustainable Elegance': an EY ad read (gold label v050: ad), forced "
        "'SCHOLA GATOBU' (2b7aab46e)",
    ),
    ("a040ea0b-b59d-4b7a-a380-c5473ef78107", "SPEAKER_01"): (
        None,
        None,
        "Talk Eastern Europe 'How Europe Can Stay Competitive': an EESC guest ('to follow on from "
        "what Sandra has been saying'), stored as the host Alexandra Karppi",
    ),
    ("13ef2838-f5e8-4985-acd5-f5445d847000", "SPEAKER_00"): (
        None,
        None,
        "Analyse 'We Never Left the Industrial Age': the guest ('war correspondent with CNN', "
        "'speechwriter for Gavin Newsom'), stored as the host Bernard Leong",
    ),
    ("5e78d393-0409-45d6-940e-843e8008bf26", "SPEAKER_01"): (
        None,
        None,
        "Analyse 'If AI Models Have No Moat': the guest Benedict Evans ('Benedict, welcome back'), "
        "stored as the host Bernard Leong",
    ),
    ("c9a0b23a-bca8-11f1-963d-9ba4b3b6b413", "SPEAKER_01"): (
        "William Dalrymple",
        "host",
        "Empire '400. Stalin': the co-host introducing the guest ('we've got you someone who's "
        "actually sat in the room with Vladimir Putin'), stored unnamed",
    ),
    ("bcecac98-430a-4361-9737-bfebe7ca9ad5", "SPEAKER_00"): (
        None,
        None,
        "DeepMind 'From deepfakes to DNA': the guest answering Hannah Fry's watermarking "
        "questions, stored as 'Hannah Fry'",
    ),
    ("e743621b-19a7-49c7-907e-07dd965ebd30", "SPEAKER_05"): (
        None,
        None,
        "DeepMind 'AI for Science': John Jumper and others answering ('John?', the AlphaFold "
        "release), stored as 'Sir Paul Nurse'",
    ),
}


def _run_relative(value: Any, run: Path) -> str:
    """The transcript path relative to the episode's RUN directory.

    Some metadata stores it absolute ("/app/output/feeds/<feed>/<run>/transcripts/x.txt"). Joined
    to the run as-is, an absolute path ignores the corpus root, so every file it names is the
    one under the path written at ingest, not the one under the corpus being upgraded (a copy, a
    restore). The part after the run directory's name is the same file in this corpus. ``""``
    when the path is absolute and does not pass through this run.
    """
    rel = str(value or "")
    path = Path(rel)
    if not path.is_absolute():
        return rel
    parts = path.parts
    if run.name not in parts:
        return ""
    at = len(parts) - 1 - parts[::-1].index(run.name)
    return str(Path(*parts[at + 1 :])) if at + 1 < len(parts) else ""


class _RunRenameEpisode(_RenameEpisode):
    """m0017's five-surface episode, with its segment files found under THIS run."""

    def __init__(self, meta: Path, rel: str) -> None:
        super().__init__(meta)
        self._rel = rel

    def load(self) -> bool:
        meta = _load(self.meta)
        if not isinstance(meta, dict):
            return False
        self.files[self.meta] = meta
        run = self.meta.parent.parent
        paths = [
            run / self._rel.replace(".txt", sfx)
            for sfx in (".segments.json", ".adfree.segments.json")
            if self._rel.endswith(".txt")
        ]
        base = str(self.meta)[: -len(".metadata.json")]
        paths += [Path(base + sfx) for sfx in (".kg.json", ".gi.json", ".bridge.json")]
        for path in paths:
            payload = _load(path)
            if isinstance(payload, (dict, list)):
                self.files[path] = payload
        self._before = {p: json.dumps(v, sort_keys=True) for p, v in self.files.items()}
        return True


class _Transcripts(_TranscriptEpisode):
    """m0023's in-place prefix rename, driven by this episode's rename map."""

    def __init__(self, meta: Path, root: Optional[Path], renames: Dict[str, str]) -> None:
        super().__init__(meta, root)
        self._renames = renames

    def _old_names(self, run: Path, rel: str) -> Dict[str, str]:
        return dict(self._renames)


class _Episode:
    """One episode's decision and rewrite, in memory; nothing touches disk here."""

    def __init__(self, meta: Path, root: Path) -> None:
        self.meta = meta
        self.root = root
        self.json_files: Dict[Path, Any] = {}
        self.text_files: Dict[Path, str] = {}
        self.counts: Dict[str, int] = {}
        self.refused: Optional[str] = None
        self.left: List[Tuple[str, str]] = []
        self.renames: Dict[str, str] = {}
        self.promote: Set[str] = set()
        self.dropped: List[str] = []
        #: Per-VOICE renames (``None`` unnames the voice) for the two shapes a name map cannot
        #: express: a swap between two voices, and a forced name that is a placed person's twin.
        self.voice_names: Dict[str, Optional[str]] = {}
        self.voice_roles: Dict[str, str] = {}
        self._old_label_cache: Set[str] = set()
        self._shared_labels: Set[str] = set()
        #: ``{segment label: voices}`` — where an OLDER record's entries are placed: they carry no
        #: ``voices`` / ``placed`` fields (1,106 of 2,329 served records on 2026-10-08), and the
        #: record builder reads every such entry as placed on the voices its label is on.
        self._seg_voices: Dict[str, List[str]] = {}

    def _bump(self, key: str, n: int = 1) -> None:
        if n:
            self.counts[key] = self.counts.get(key, 0) + n

    def _apply_replayed(self, meta: dict, rel: str) -> None:
        """``REPLAYED_VOICES`` for this episode — each only while its label still differs, so a
        second run finds nothing to do."""
        episode_id = str((meta.get("episode") or {}).get("episode_id") or "")
        voices = self._all_voices(rel)
        label_of = {v: lab for lab, vs in self._seg_voices.items() for v in vs}
        for (eid, voice), (name, role, _why) in REPLAYED_VOICES.items():
            if eid != episode_id or voice not in voices or label_of.get(voice) == name:
                continue
            self.voice_names[voice] = name
            if name and role:
                self.voice_roles[voice] = role

    def _live(self, entry: Any) -> List[str]:
        """The entry's voices that keep a name (not unnamed by ``REPLAYED_VOICES``)."""
        return [
            v
            for v in self._voices(entry)
            if not (v in self.voice_names and self.voice_names[v] is None)
        ]

    def _all_voices(self, rel: str) -> Set[str]:
        """Every diarized voice the episode's segments carry."""
        payload = _load(self.meta.parent.parent / rel.replace(".txt", ".segments.json"))
        rows = payload if isinstance(payload, list) else (payload or {}).get("segments")
        return {str(r["speaker"]) for r in rows or [] if isinstance(r, dict) and r.get("speaker")}

    def _voices(self, entry: Any) -> List[str]:
        """The voices a record entry is placed on: its own list, or for an older record's entry
        (no ``voices`` field, not marked unplaced) the voices whose segment label is its name."""
        if not isinstance(entry, dict):
            return []
        if "voices" in entry:
            return [str(v) for v in entry.get("voices") or []]
        if entry.get("placed") is False:
            return []
        return list(self._seg_voices.get(str(entry.get("name")), []))

    # ---- the decision -------------------------------------------------------------------------

    def decide(self) -> bool:
        """Fill ``renames`` / ``dropped`` / ``left`` / ``refused`` from the stored record."""
        meta = _load(self.meta)
        if not isinstance(meta, dict):
            return False
        content = meta.get("content") or {}
        speakers = [s for s in content.get("speakers") or [] if isinstance(s, dict)]
        rel = _run_relative(content.get("transcript_file_path"), self.meta.parent.parent)
        diag = (
            _load(self.meta.parent.parent / rel.replace(".txt", ".speakers.diagnostics.json"))
            if rel.endswith(".txt")
            else None
        )
        self._load_seg_voices(rel)
        tried = (diag or {}).get("tried") or {}
        hosts = [str(h) for h in tried.get("known_hosts") or [] if h]
        participants = [
            str(n)
            for n in list(tried.get("metadata_named") or [])
            + list(tried.get("detected_guests") or [])
            if n
        ]
        self._apply_replayed(meta, rel)
        # An entry whose every voice is being unnamed is, from here on, nobody's voice.
        placed = [s for s in speakers if self._live(s) and isinstance(s.get("name"), str)]
        unplaced_lower = {
            str(s.get("name")).lower()
            for s in speakers
            if not self._live(s) and isinstance(s.get("name"), str)
        }
        role_of = {str(s["name"]): s.get("role") for s in placed}
        claimed: Dict[str, str] = {}
        for s in placed:
            name = str(s["name"])
            if name in hosts:
                continue
            taken = {str(o["name"]) for o in placed if o is not s}
            target, as_host = self._target_for(
                s, taken, hosts, participants, unplaced_lower, meta, rel, placed
            )
            if target is None:
                continue
            # The record publishes the canonical spelling (#2130): "Peter Attia, MD" is stated,
            # "Peter Attia" is published, and a rename that only restores the credential is none.
            target = canonical_person_name(target) or target
            if target == name:
                continue
            if target in role_of and role_of[target] != s.get("role") and role_of[target]:
                if self._swap(s, target, placed, speakers, diag):
                    continue
                self.refused = f"{name!r} would take {target!r}, held in another role"
                return True
            if claimed.setdefault(target, name) != name:
                self.refused = f"{target!r} would be claimed by two voices"
                return True
            self.renames[name] = target
            if as_host and s.get("role") != "host":
                self.promote.add(target)
        self._dedupe(speakers)
        if self.left and not self.refused:
            self._unify_left(meta, rel, diag, hosts, participants, placed)
        return bool(self.renames or self.dropped or self.left or self.voice_names)

    def _unify_left(
        self,
        meta: dict,
        rel: str,
        diag: Any,
        hosts: List[str],
        participants: List[str],
        placed: List[dict],
    ) -> None:
        """Two placed voices, two spellings of one person: the pipeline's own
        ``roster._one_name_per_person`` decides, fed what the stored record holds — each voice's
        label and role from the segments, talk and turn alternations from their timings, the
        stated names from the diagnostics. Its answer becomes per-voice renames; a pair it keeps
        apart (two people who converse) stays listed in ``left``."""
        payload = _load(self.meta.parent.parent / rel.replace(".txt", ".segments.json"))
        rows = payload if isinstance(payload, list) else (payload or {}).get("segments")
        rows = [r for r in rows or [] if isinstance(r, dict) and r.get("speaker")]
        if not rows:
            return
        src = {
            str(v.get("voice")): str(v.get("source") or "raw")
            for v in (diag or {}).get("voices") or []
            if isinstance(v, dict)
        }
        by_voice: Dict[str, Any] = {}
        talk: Dict[str, float] = {}
        for r in rows:
            v = str(r["speaker"])
            talk[v] = talk.get(v, 0.0) + float(r.get("end") or 0.0) - float(r.get("start") or 0.0)
            label = r.get("speaker_label")
            if v not in by_voice:
                by_voice[v] = R.SpeakerRole(
                    name=str(label or v),
                    role=str(r.get("speaker_role") or "guest"),
                    named=bool(label),
                    source=src.get(v, "raw"),
                )
        dz = DiarizationResult(
            segments=[
                DiarizationSegment(
                    float(r.get("start") or 0.0), float(r.get("end") or 0.0), str(r["speaker"])
                )
                for r in rows
            ],
            num_speakers=len(by_voice),
        )
        episode = meta.get("episode") or {}
        text = " ".join(x for x in (episode.get("title"), episode.get("description")) if x)
        presenters = self._presenters(meta, rel, placed, participants)
        unified = R._one_name_per_person(
            dict(by_voice),
            talk,
            list(dict.fromkeys(hosts + participants)),
            hosts,
            episode_text=text or None,
            evidence_hosts={v for v, r in by_voice.items() if r.role == "host" and v in presenters},
            alternations=R._alternations(dz),
        )
        for v, role in unified.items():
            before = by_voice[v]
            if role.named and (role.name != before.name or role.role != before.role):
                self.voice_names[v] = role.name
                self.voice_roles[v] = role.role
        resolved = [
            (a, b)
            for a, b in self.left
            if len(
                {
                    unified[v].name
                    for v in self._seg_voices.get(a, []) + self._seg_voices.get(b, [])
                    if v in unified
                }
            )
            == 1
        ]
        self.left = [pair for pair in self.left if pair not in resolved]
        self._bump("left_unified", len(resolved))

    def _load_seg_voices(self, rel: str) -> None:
        if not rel.endswith(".txt"):
            return
        payload = _load(self.meta.parent.parent / rel.replace(".txt", ".segments.json"))
        rows = payload if isinstance(payload, list) else (payload or {}).get("segments")
        for r in rows or []:
            if isinstance(r, dict) and r.get("speaker") and r.get("speaker_label"):
                have = self._seg_voices.setdefault(str(r["speaker_label"]), [])
                if str(r["speaker"]) not in have:
                    have.append(str(r["speaker"]))

    def _target_for(
        self,
        s: dict,
        taken: Set[str],
        hosts: List[str],
        participants: List[str],
        unplaced_lower: Set[str],
        meta: dict,
        rel: str,
        placed: List[dict],
    ) -> Tuple[Optional[str], bool]:
        """``(the spelling this placed name now takes, whether as the host)``, or ``(None, _)``."""
        name = str(s["name"])
        participants_lower = {p.lower() for p in participants}
        target = host_spelling(name, s.get("role"), hosts, taken)
        # Only what this fix changes: a title, or the host listed beside the voice.
        if target is not None and not (_has_title(name) or target.lower() in unplaced_lower):
            target = None
        if target is not None:
            return target, True
        if (self._voices(s) or [""])[0] in self._presenters(meta, rel, placed, participants):
            target = _unclaimed_host(name, hosts, taken, participants_lower)
            if target is not None:
                return target, True
        # `_recover_stated_names` never re-spells a name the episode already states.
        if name.lower() in participants_lower | {h.lower() for h in hosts}:
            return None, False
        target = _stated_participant(name, hosts, taken, participants_lower)
        if target is None:
            spelt = R._stated_participant_spelling(name, participants, {t.lower() for t in taken})
            target = spelt if spelt != name else None
        return target, False

    def _dedupe(self, speakers: List[dict]) -> None:
        kept: List[str] = []
        kept_entry: Dict[str, dict] = {}
        for s in speakers:
            raw = s.get("name")
            if not isinstance(raw, str):
                continue
            voices = self._voices(s)
            if voices and all(v in self.voice_names for v in voices):
                final = self.voice_names[voices[0]]
                if final is None:
                    continue
            else:
                final = self.renames.get(raw, raw)
            match = next((k for k in kept if same_person(final, k)), None)
            if match is None:
                kept.append(final)
                kept_entry[final] = s
            elif not self._voices(s):
                self.dropped.append(raw)
            elif final != match:
                if not self._unname_forced_twin(s, kept_entry[match]):
                    self.left.append((match, final))

    def _swap(
        self, s: dict, target: str, placed: List[dict], speakers: List[dict], diag: Any
    ) -> bool:
        """DeepMind 0014: a voice's own "I'm Professor Hannah Fry" is the stated host, while the
        introduction reader had put "Hannah Fry" on the GUEST's voice. The self-introduction of a
        stated host outranks the reader; the guest's voice then takes the name the LLM gave it, if
        that is a stated person the record lists unplaced, else it is left unnamed."""
        holder = next((o for o in placed if o.get("name") == target), None)
        own, theirs = list(self._voices(s) or []), list(self._voices(holder or {}) or [])
        if (
            holder is None
            or s.get("source") != "self_intro"
            or holder.get("source") != "introduced"
            or len(own) != 1
            or len(theirs) != 1
        ):
            return False
        inputs = (((diag or {}).get("decision_trace") or {}).get("inputs")) or {}
        llm_name = (inputs.get("llm_voice_names") or {}).get(theirs[0])
        unplaced = {
            str(o["name"]): o
            for o in speakers
            if not self._voices(o) and isinstance(o.get("name"), str)
        }
        replacement: Optional[str] = None
        if (
            isinstance(llm_name, str)
            and llm_name in unplaced
            and not any(same_person(llm_name, str(o["name"])) for o in placed)
        ):
            replacement = llm_name
        self.voice_names[own[0]], self.voice_roles[own[0]] = target, "host"
        self.voice_names[theirs[0]] = replacement
        if replacement is not None:
            self.voice_roles[theirs[0]] = str(unplaced[replacement].get("role") or "guest")
            if self.voice_roles[theirs[0]] == "host":
                self.voice_roles[theirs[0]] = "guest"
        return True

    def _unname_forced_twin(self, s: dict, other: dict) -> bool:
        """The Long Run: the host's spoken "Andy Ratcliffe" was FORCED onto the co-guest's voice
        while the LLM had placed the stated Andy Rachleff on his own. A name placed by arithmetic
        that is a person another voice holds by evidence is nobody's name on this voice."""
        pairs = [(s, other), (other, s)]
        for forced, evidence in pairs:
            voices = list(self._voices(forced) or [])
            if (
                forced.get("source") == "forced"
                and evidence.get("source") not in (None, "forced")
                and self._voices(evidence)
                and voices
            ):
                for v in voices:
                    self.voice_names[str(v)] = None
                return True
        return False

    # ---- the rewrite --------------------------------------------------------------------------

    def plan(self) -> bool:
        """Decide, then build every rewritten file in memory. False when nothing is written."""
        if not self.decide() or self.refused:
            return False
        meta = _load(self.meta) or {}
        rel = _run_relative(
            (meta.get("content") or {}).get("transcript_file_path"), self.meta.parent.parent
        )
        if self.voice_names:
            return self._plan_by_voice(meta, rel)
        if not (self.renames or self.dropped):
            return False
        transcripts = _Transcripts(self.meta, self.root, self.renames)
        if self.renames and rel.endswith(".txt"):
            try:
                transcripts._in_place(self.meta.parent.parent, rel)
            except Refused as exc:
                self.refused = f"transcripts: {exc}"
                return False
        ids = self._id_map()
        ep = _RunRenameEpisode(self.meta, rel)
        if not ep.load():
            return False
        # The transcript step moved offsets in the GI and ad-free segments; rename on top of those.
        for path, payload in transcripts.json_files.items():
            ep.files[path] = payload
        ep.rewrite(self.renames, ids)
        self._node_labels(ep)
        promoted = self._hosts_promoted(meta)
        self._record(ep.files[self.meta], promoted)
        self._segment_roles(ep, promoted)
        self._kg_roles(ep, promoted, ids)
        for k, v in ep.counts.items():
            self._bump(k, v)
        for path in ep.changed:
            self.json_files[path] = ep.files[path]
        for path, text in transcripts.text_files.items():
            self.text_files[path] = text
        for k, v in transcripts.counts.items():
            self._bump(k, v)
        if self.renames and rel.endswith(".txt"):
            try:
                self._turns(ep.files, ep.changed, rel)
            except Refused as exc:
                self.json_files, self.text_files = {}, {}
                self.refused = f"turns: {exc}"
                return False
        self._diagnostics(rel, promoted)
        self._context(ep.files)
        return bool(self.json_files or self.text_files)

    def _plan_by_voice(self, meta: dict, rel: str) -> bool:
        """Rewrite every surface VOICE BY VOICE (``voice_names``), for a swap or an unnaming.

        A name map cannot say "SPEAKER_04's Hannah Fry becomes Paige Bailey while SPEAKER_02's
        Professor Hannah Fry becomes Hannah Fry": both spellings mint ``person:hannah-fry``, and
        the GI quotes carry only that id. So each quote's voice is read off the transcript line it
        sits on (its ``<label>: `` prefix, before any rename), and every surface is rewritten from
        ``{voice: new name}``.
        """
        if not rel.endswith(".txt"):
            self.refused = "no transcript to place quotes by"
            return False
        # A name rename in the same episode rides along, voice by voice.
        for e in (meta.get("content") or {}).get("speakers") or []:
            name = (e or {}).get("name")
            if name in self.renames:
                for v in self._voices(e) or []:
                    if str(v) not in self.voice_names:
                        self.voice_names[str(v)] = self.renames[name]
                        target = self.renames[name]
                        self.voice_roles[str(v)] = (
                            "host" if target in self.promote else str(e.get("role") or "guest")
                        )
        run = self.meta.parent.parent
        base = Path(str(self.meta)[: -len(".metadata.json")])
        seg_paths = [
            run / rel.replace(".txt", sfx) for sfx in (".segments.json", ".adfree.segments.json")
        ]
        segs = {p: _load(p) for p in seg_paths if p.is_file()}
        old_label = self._labels_by_voice(segs)
        if old_label is None:
            return False
        voice_of_label = {lab: v for v, lab in old_label.items()}
        new_label = {old_label[v]: (n if n else v) for v, n in self.voice_names.items()}
        # Labels a rewritten voice shares with a voice left as it is (a forced ad twin).
        others = {
            str(r.get("speaker_label"))
            for payload in segs.values()
            for r in (
                (payload if isinstance(payload, list) else (payload or {}).get("segments")) or []
            )
            if isinstance(r, dict)
            and r.get("speaker_label")
            and str(r.get("speaker")) not in self.voice_names
        }
        shared = set(new_label) & others
        # 1. Which voice each quote is, by where it sits, BEFORE anything moves.
        gi_path = Path(str(base) + ".gi.json")
        gi = _load(gi_path)
        quote_voice = _quote_voices(gi, run, rel, segs, voice_of_label, shared)
        if quote_voice is None:
            self.refused = "a quote's voice cannot be told from a label two voices carry"
            return False
        quote_voice = {q: v for q, v in quote_voice.items() if v in self.voice_names}
        self._shared_labels = shared
        files: Dict[Path, Any] = {self.meta: meta}
        changed: Set[Path] = {self.meta}
        # 2. Segments, voice by voice.
        for p, payload in segs.items():
            self._segments_by_voice(payload)
            files[p] = payload
            changed.add(p)
        # 3. Transcripts re-rendered from those segments, every offset carried across (m0023's
        # alignment): a rename can make two consecutive lines one speaker's, and the pipeline's
        # formatter writes those as ONE line. Where the old file is not a pure render, m0023's
        # in-place rename, under its own guard.
        transcripts = _Transcripts(self.meta, self.root, new_label)
        try:
            transcripts._plan(run, rel, gi=gi, segments=segs)
        except Refused:
            if shared:
                self.refused = "transcripts: not a render, and a rewritten label is shared"
                return False
            transcripts = _Transcripts(self.meta, self.root, new_label)
            try:
                transcripts._in_place(run, rel, gi)
            except Refused as exc:
                self.refused = f"transcripts: {exc}"
                return False
            for p in segs:
                if p in transcripts.json_files:
                    moved = transcripts.json_files[p]
                    self._segments_by_voice(moved)
                    files[p] = moved
        # 4. GI quotes, SPOKEN_BY edges, Person nodes, insights.
        new_ids: Dict[str, Optional[str]] = {
            v: (person_id(n) if n else None) for v, n in self.voice_names.items()
        }
        if isinstance(gi, dict):
            self._gi_by_voice(gi, quote_voice, new_ids, new_label)
            files[gi_path] = gi
            changed.add(gi_path)
        # 5. KG roles and nodes, bridge rows.
        self._graph_people(base, files, changed, new_ids)
        # 6. The record, the diagnostics, the turns, the context.
        self._record_by_voice(meta)
        self._diagnostics_by_voice(rel, files, changed)
        for key, count in transcripts.counts.items():
            self._bump(key, count)
        for p, text in transcripts.text_files.items():
            self.text_files[p] = text
        try:
            self._turns(files, changed, rel)
        except Refused as exc:
            self.json_files, self.text_files = {}, {}
            self.refused = f"turns: {exc}"
            return False
        for p in changed:
            self.json_files[p] = files[p]
        self._context(files)
        return True

    def _labels_by_voice(self, segs: Dict[Path, Any]) -> Optional[Dict[str, str]]:
        """``{voice: its one label}`` for every voice rewritten, or ``None`` (refused)."""
        label_of: Dict[str, Set[str]] = {}
        for payload in segs.values():
            rows = payload if isinstance(payload, list) else (payload or {}).get("segments")
            for r in rows or []:
                if isinstance(r, dict) and r.get("speaker"):
                    lab = r.get("speaker_label") or r["speaker"]
                    label_of.setdefault(str(r["speaker"]), set()).add(str(lab))
        old_label: Dict[str, str] = {}
        for v in self.voice_names:
            labs = label_of.get(v) or set()
            if len(labs) != 1:
                self.refused = f"{v} carries {len(labs)} labels"
                return None
            old_label[v] = next(iter(labs))
        if len(set(old_label.values())) != len(old_label):
            self.refused = "two voices share one label"
            return None
        self._old_label_cache = set(old_label.values())
        return old_label

    def _segments_by_voice(self, payload: Any) -> None:
        rows = payload if isinstance(payload, list) else (payload or {}).get("segments")
        for r in rows or []:
            v = str(r.get("speaker")) if isinstance(r, dict) else None
            if v not in self.voice_names:
                continue
            name = self.voice_names[v]
            if name:
                r["speaker_label"] = name
                r["speaker_role"] = self.voice_roles.get(v, r.get("speaker_role"))
            else:
                r.pop("speaker_label", None)
                r["voice_type"] = "unknown"
            self._bump("segments_relabelled")

    def _record_by_voice(self, meta: dict) -> None:
        content = meta.get("content") or {}
        out = []
        for e in content.get("speakers") or []:
            voices = self._voices(e or {})
            gone = [v for v in voices if v in self.voice_names and self.voice_names[v] is None]
            if gone and len(gone) < len(voices) and "voices" in e:
                e["voices"] = [v for v in e["voices"] if str(v) not in gone]
                self._bump("voices_unnamed_from_entry", len(gone))
                voices = self._voices(e)
            if voices and all(v in self.voice_names for v in voices):
                name = self.voice_names[voices[0]]
                if name is None:
                    self._bump("roster_entries_unnamed")
                    continue
                e["name"], e["role"] = name, self.voice_roles.get(voices[0], e.get("role"))
            out.append(e)
        # A voice that GAINS a name (stored unnamed): the unplaced entry for that person takes it,
        # or a new entry is added — in the record's own format (older records carry no voices).
        new_format = any("voices" in e for e in out if isinstance(e, dict))
        for v, name in self.voice_names.items():
            if not name or any(v in self._voices(e) for e in out if isinstance(e, dict)):
                continue
            role = self.voice_roles.get(v, "guest")
            twin = next(
                (
                    e
                    for e in out
                    if isinstance(e, dict)
                    and not self._voices(e)
                    and isinstance(e.get("name"), str)
                    and same_person(name, str(e["name"]))
                ),
                None,
            )
            entry = twin if twin is not None else {"id": "", "name": name, "role": role}
            entry.update(name=name, role=role)
            if new_format:
                entry.update(placed=True, voices=[v], source=entry.get("source") or "roster")
                if twin is not None:
                    entry["source"] = "roster"
            elif twin is not None:
                entry.pop("placed", None)
            if twin is None:
                out.append(entry)
            self._bump("voices_named")
        content["speakers"] = out
        self._record(meta, set())

    def _gi_by_voice(
        self,
        gi: dict,
        quote_voice: Dict[str, str],
        new_ids: Dict[str, Optional[str]],
        new_label: Dict[str, str],
    ) -> None:
        from ...gi.pipeline import _apply_route_and_tag

        quote_to: Dict[str, Optional[str]] = {
            q: new_ids[v] for q, v in quote_voice.items() if v in new_ids
        }
        for node in gi.get("nodes") or []:
            props = node.get("properties") if isinstance(node, dict) else None
            if not isinstance(props, dict):
                continue
            nid = str(node.get("id"))
            if node.get("type") == "Quote" and nid in quote_to:
                props["speaker_id"] = quote_to[nid]
                props.pop("speaker_name", None)
                if quote_to[nid] is None:
                    props["speaker_voice_type"] = "unknown"
                self._bump("quotes_reattributed")
            elif (
                node.get("type") == "Insight"
                and props.get("speaker") in new_label
                and props.get("speaker") not in self._shared_labels
            ):
                old = props["speaker"]
                if new_label[old] in self.voice_names:  # unnamed: a raw voice id
                    props.pop("speaker", None)
                    props["speaker_voice_type"] = "unknown"
                    props["surfaceable"] = False
                    if isinstance(props.get("tier"), int):
                        _apply_route_and_tag(props, props["tier"])
                else:
                    props["speaker"] = new_label[old]
                self._bump("insights_reattributed")
        edges = [
            e
            for e in gi.get("edges") or []
            if not (e.get("type") == "SPOKEN_BY" and str(e.get("from")) in quote_to)
        ]
        edges += [{"type": "SPOKEN_BY", "from": q, "to": to} for q, to in quote_to.items() if to]
        gi["edges"] = edges
        used = {e.get("to") for e in edges if e.get("type") == "SPOKEN_BY"}
        used |= {
            (n.get("properties") or {}).get("speaker_id")
            for n in gi.get("nodes") or []
            if n.get("type") == "Quote"
        }
        nodes = gi.get("nodes") or []
        have = {n.get("id") for n in nodes if n.get("type") == "Person"}
        names = {pid: n for n in self.voice_names.values() if n for pid in [person_id(n)]}
        for pid, name in names.items():
            if pid not in have:
                nodes.append({"id": pid, "type": "Person", "properties": {"name": name}})
                self._bump("gi_persons_added")
        gone_old = {
            person_id(lab)
            for lab in new_label
            if lab not in self.voice_names and lab not in self.voice_names.values()
        }
        gi["nodes"] = [
            n
            for n in nodes
            if not (n.get("type") == "Person" and n.get("id") in gone_old and n["id"] not in used)
        ]

    def _graph_people(
        self, base: Path, files: Dict[Path, Any], changed: Set[Path], new_ids: Dict
    ) -> None:
        kg_path, br_path = Path(str(base) + ".kg.json"), Path(str(base) + ".bridge.json")
        kg, bridge = _load(kg_path), _load(br_path)
        roles = {
            person_id(n): self.voice_roles.get(v, "guest") for v, n in self.voice_names.items() if n
        }
        gi = files.get(Path(str(base) + ".gi.json")) or {}
        in_gi = {n.get("id") for n in gi.get("nodes") or [] if n.get("type") == "Person"}
        if isinstance(kg, dict):
            nodes = kg.get("nodes") or []
            have = {n.get("id"): n for n in nodes if isinstance(n, dict)}
            for pid, role in roles.items():
                name = next(n for n in self.voice_names.values() if n and person_id(n) == pid)
                if pid in have:
                    props = have[pid].setdefault("properties", {})
                    if props.get("role") != role and props.get("role") in (
                        None,
                        "mentioned",
                        "guest",
                        "host",
                    ):
                        if not (props.get("role") == "host" and role == "guest"):
                            props["role"] = role
                            self._bump("kg_roles_set")
                else:
                    nodes.append(
                        {
                            "id": pid,
                            "type": "Person",
                            "properties": {"name": name, "label": name, "role": role},
                        }
                    )
                    self._bump("kg_persons_added")
            # A speaking node for a name now on NO voice (the unnamed forced twin) goes; a name
            # still on another voice keeps its node.
            on_voice = set(roles) | self._labels_on_voices(files)
            stale = {
                n.get("id")
                for n in nodes
                if n.get("type") == "Person"
                and (n.get("properties") or {}).get("role") in ("host", "guest")
                and n.get("id") not in on_voice
                and (n.get("properties") or {}).get("name")
                in {lab for lab in self._old_labels() if lab}
            }
            if stale:
                nodes = [n for n in nodes if n.get("id") not in stale]
                kg["edges"] = [
                    e
                    for e in kg.get("edges") or []
                    if e.get("from") not in stale and e.get("to") not in stale
                ]
                self._bump("kg_persons_removed", len(stale))
            kg["nodes"] = nodes
            files[kg_path] = kg
            changed.add(kg_path)
        if isinstance(bridge, dict):
            ids = bridge.get("identities") or []
            kg_ids = {
                n.get("id") for n in (kg or {}).get("nodes") or [] if n.get("type") == "Person"
            }
            out = []
            for ident in ids:
                ident_id = ident.get("id") if isinstance(ident, dict) else None
                if isinstance(ident_id, str) and ident_id.startswith("person:"):
                    if ident_id not in kg_ids and ident_id not in in_gi:
                        self._bump("bridge_identities_removed")
                        continue
                    src = ident.setdefault("sources", {})
                    if isinstance(src, dict):
                        src["gi"], src["kg"] = ident_id in in_gi, ident_id in kg_ids
                out.append(ident)
            have_ids = {i.get("id") for i in out if isinstance(i, dict)}
            for pid in roles:
                if pid not in have_ids:
                    name = next(n for n in self.voice_names.values() if n and person_id(n) == pid)
                    out.append(
                        {
                            "id": pid,
                            "type": "person",
                            "display_name": name,
                            "aliases": [],
                            "sources": {"gi": pid in in_gi, "kg": pid in kg_ids},
                        }
                    )
            bridge["identities"] = out
            files[br_path] = bridge
            changed.add(br_path)

    @staticmethod
    def _labels_on_voices(files: Dict[Path, Any]) -> Set[str]:
        """Person ids of every label any voice carries after the rewrite."""
        out: Set[str] = set()
        for path, payload in files.items():
            if not path.name.endswith(".segments.json"):
                continue
            rows = payload if isinstance(payload, list) else (payload or {}).get("segments")
            for r in rows or []:
                if isinstance(r, dict) and r.get("speaker_label"):
                    try:
                        out.add(person_id(str(r["speaker_label"])))
                    except ValueError:
                        continue
        return out

    def _old_labels(self) -> Set[str]:
        """The labels the rewritten voices carried before (set by ``_plan_by_voice``)."""
        return set(self._old_label_cache)

    def _diagnostics_by_voice(self, rel: str, files: Dict[Path, Any], changed: Set[Path]) -> None:
        path = self.meta.parent.parent / rel.replace(".txt", ".speakers.diagnostics.json")
        diag = _load(path)
        if not isinstance(diag, dict):
            return
        for v in diag.get("voices") or []:
            if not isinstance(v, dict) or v.get("voice") not in self.voice_names:
                continue
            name = self.voice_names[v["voice"]]
            if name:
                v.update(resolved_name=name, named=True, role=self.voice_roles.get(v["voice"]))
            else:
                v.update(resolved_name=v["voice"], named=False, source="raw")
            self._bump("diagnostics_voices_renamed")
        files[path] = diag
        changed.add(path)

    def _node_labels(self, ep: _RenameEpisode) -> None:
        """A Person node's ``label`` is its display name too; m0017 renames ``name`` only."""
        for path, payload in ep.files.items():
            if not path.name.endswith((".kg.json", ".gi.json")):
                continue
            for node in _person_nodes(payload):
                props = node.get("properties")
                if isinstance(props, dict) and props.get("label") in self.renames:
                    props["label"] = self.renames[props["label"]]
                    self._bump("node_labels_rewritten")

    def _turns(self, files: Dict[Path, Any], changed: Set[Path], rel: str) -> None:
        """Rebuild each variant's ``turns.json`` (RFC-123) from the rewritten text and segments.

        Its rows hold the speaker label AND character offsets into the transcript, so a rename
        that changes a prefix's length makes the old file wrong twice. The pipeline's own builder
        is used, which returns nothing unless the text is exactly the render of the segments:
        then the episode is refused rather than left with turns that index the old text.
        """
        run = self.meta.parent.parent
        for suffix in (".txt", ".adfree.txt"):
            text_path = run / rel.replace(".txt", suffix)
            turns_path = Path(str(text_path)[: -len(".txt")] + TURNS_SUFFIX)
            old = _load(turns_path)
            if not isinstance(old, dict):
                continue
            seg_path = Path(str(text_path)[: -len(".txt")] + ".segments.json")
            segs = files.get(seg_path)
            rows = segs if isinstance(segs, list) else (segs or {}).get("segments")
            text = self.text_files.get(text_path)
            if text is None:
                text = text_path.read_text(encoding="utf-8") if text_path.is_file() else ""
            raw_source = old.get("source")
            source: Dict[str, Any] = raw_source if isinstance(raw_source, dict) else {}
            sha = source.get("segments_sha256")
            if seg_path in changed:
                sha = hashlib.sha256(dump_json(segs).encode("utf-8")).hexdigest()
            doc = build_turns_document(
                text,
                [r for r in rows or [] if isinstance(r, dict)],
                rel_transcript_path=str(source.get("transcript_ref") or ""),
                episode_slug=old.get("episode_slug"),
                language=old.get("language"),
                segments_sha256=sha,
            )
            if doc is None:
                raise Refused(f"{turns_path.name} no longer rebuilds from its text")
            # The turn COUNT may move either way: a rename can merge two consecutive lines into one
            # speaker's, an unnamed voice can split one. The builder's own guard — it builds only
            # from text that is exactly the render of the segments — is the check.
            if doc != old:
                # The pipeline's own serialisation (turns_artifact.write_turns_artifact).
                self.text_files[turns_path] = json.dumps(doc, indent=0, allow_nan=False)
                self._bump("turns_rebuilt")

    def _id_map(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for old, new in self.renames.items():
            try:
                a, b = person_id(old), person_id(new)
            except ValueError:
                continue
            if a != b and not is_scoped_person_id(a):
                out[a] = b
        return out

    def _hosts_promoted(self, meta: dict) -> Set[str]:
        """New names that become the host where the record had the voice as a guest."""
        return set(self.promote)

    def _presenters(
        self, meta: dict, rel: str, placed: List[dict], participants: List[str]
    ) -> Set[str]:
        """Voices that present the show on their own words — the roster's own evidence function,
        fed the stored segments (``roster._presenter_voices_by_evidence``). Computed once."""
        if hasattr(self, "_presenter_cache"):
            return self._presenter_cache
        self._presenter_cache: Set[str] = set()
        if not rel.endswith(".txt"):
            return self._presenter_cache
        payload = _load(self.meta.parent.parent / rel.replace(".txt", ".segments.json"))
        rows = payload if isinstance(payload, list) else (payload or {}).get("segments")
        rows = [r for r in rows or [] if isinstance(r, dict) and r.get("speaker")]
        if not rows:
            return self._presenter_cache
        texts: Dict[str, List[str]] = {}
        for r in rows:
            texts.setdefault(str(r["speaker"]), []).append(str(r.get("text") or ""))
        dz = DiarizationResult(
            segments=[
                DiarizationSegment(
                    float(r.get("start") or 0.0), float(r.get("end") or 0.0), str(r["speaker"])
                )
                for r in rows
            ],
            num_speakers=len(texts),
        )
        branded, introducers = R._presenter_voices_by_evidence(
            dz,
            {v: " ".join(t) for v, t in texts.items()},
            set(),
            (meta.get("feed") or {}).get("title"),
            participants,
            {str(v): str(s["name"]) for s in placed for v in self._voices(s) or []},
        )
        self._presenter_cache = branded | introducers
        return self._presenter_cache

    def _record(self, meta: dict, promoted: Set[str]) -> None:
        """``content.speakers``: renamed (by m0017 already), merged, de-duplicated, renumbered."""
        content = meta.get("content") or {}
        entries = [s for s in content.get("speakers") or [] if isinstance(s, dict)]
        out: List[dict] = []
        for s in entries:
            name = s.get("name")
            if not isinstance(name, str):
                out.append(s)
                continue
            if name in promoted and self._voices(s):
                s["role"] = "host"
                self._bump("roles_promoted")
            twin = next(
                (
                    o
                    for o in out
                    if isinstance(o.get("name"), str) and same_person(name, str(o["name"]))
                ),
                None,
            )
            if twin is None:
                out.append(s)
            elif not self._voices(s):
                self._bump("unplaced_dropped")
            elif self._voices(twin) and twin.get("name") == name:
                if "voices" in twin and "voices" in s:
                    twin["voices"] = list(dict.fromkeys(list(twin["voices"]) + list(s["voices"])))
                self._bump("placed_merged")
            else:
                out.append(s)  # two placed voices, two names: left to the roster
        hosts = [s for s in out if self._voices(s) and s.get("role") == "host"]
        guests = [s for s in out if self._voices(s) and s.get("role") != "host"]
        unplaced = [s for s in out if not self._voices(s)]
        for group, stem in ((hosts, "host"), (guests, "guest")):
            for i, s in enumerate(group):
                s["id"] = stem if len(group) == 1 else f"{stem}_{i + 1}"
        for i, s in enumerate(unplaced):
            s["id"] = f"unplaced_{i + 1}"
        content["speakers"] = hosts + guests + unplaced

    def _voices_of(self, meta: dict, names: Set[str]) -> Set[str]:
        return {
            str(v)
            for s in (meta.get("content") or {}).get("speakers") or []
            if isinstance(s, dict) and s.get("name") in names
            for v in self._voices(s) or []
        }

    def _segment_roles(self, ep: _RenameEpisode, promoted: Set[str]) -> None:
        if not promoted:
            return
        for path, payload in ep.files.items():
            if not path.name.endswith(".segments.json"):
                continue
            rows = payload if isinstance(payload, list) else (payload or {}).get("segments")
            for row in rows or []:
                if (
                    isinstance(row, dict)
                    and row.get("speaker_label") in promoted
                    and row.get("speaker_role") not in (None, "host")
                ):
                    row["speaker_role"] = "host"
                    self._bump("segment_roles_promoted")

    def _kg_roles(self, ep: _RenameEpisode, promoted: Set[str], ids: Dict[str, str]) -> None:
        if not promoted:
            return
        wanted = set()
        for name in promoted:
            try:
                wanted.add(person_id(name))
            except ValueError:
                continue
        for path, payload in ep.files.items():
            if not path.name.endswith(".kg.json"):
                continue
            for node in _person_nodes(payload):
                props = node.get("properties")
                if (
                    node["id"] in wanted
                    and isinstance(props, dict)
                    and props.get("role") == "guest"
                ):
                    props["role"] = "host"
                    self._bump("kg_roles_promoted")

    def _diagnostics(self, rel: str, promoted: Set[str]) -> None:
        if not rel.endswith(".txt") or not self.renames:
            return
        path = self.meta.parent.parent / rel.replace(".txt", ".speakers.diagnostics.json")
        diag = _load(path)
        if not isinstance(diag, dict):
            return
        hit = 0
        for v in diag.get("voices") or []:
            if isinstance(v, dict) and v.get("named") and v.get("resolved_name") in self.renames:
                v["resolved_name"] = self.renames[v["resolved_name"]]
                if v["resolved_name"] in promoted:
                    v["role"] = "host"
                hit += 1
        if hit:
            self.json_files[path] = diag
            self._bump("diagnostics_voices_renamed", hit)

    def _context(self, files: Dict[Path, Any]) -> None:
        base = str(self.meta)[: -len(".metadata.json")]
        path = Path(base + ".context.json")
        ctx = _load(path)
        if not isinstance(ctx, dict):
            return
        rebuilt = build_context_digest(
            str(ctx.get("episode_id") or ""),
            gi_artifact=files.get(Path(base + ".gi.json")),
            kg_artifact=files.get(Path(base + ".kg.json")),
            metadata=files[self.meta],
        )
        hit = False
        for parent, key in _CONTEXT_FIELDS:
            old = ctx.get(parent) if parent else ctx
            new = rebuilt.get(parent) if parent else rebuilt
            if isinstance(old, dict) and isinstance(new, dict) and key in new:
                if old.get(key) != new[key]:
                    old[key] = new[key]
                    hit = True
        if hit:
            self.json_files[path] = ctx
            self._bump("context_rebuilt")


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class OnePersonOneEntryMigration(Migration):
    """One person, one entry: titled and respelt names take the stated host's; duplicates go."""

    id = MIGRATION_ID
    to_version = "2.7.19"
    description = (
        "one person is one entry per episode: a titled or respelt voice name takes the stated "
        "host's spelling and role, and a duplicate unplaced entry is dropped, across metadata, "
        "segments, KG, GI, bridge, diagnostics, context and transcripts (offsets shifted)"
    )

    def _scan(self, root: Path) -> Tuple[List[_Episode], List[_Episode], Dict[str, int]]:
        """``(episodes to write, episodes refused or left, totals)``."""
        write: List[_Episode] = []
        other: List[_Episode] = []
        totals: Dict[str, int] = {}
        for meta in select_served_artifacts(root, ".metadata.json")[0]:
            ep = _Episode(meta, root)
            if ep.plan():
                write.append(ep)
            if ep.refused or ep.left:
                other.append(ep)
            if ep.refused:
                ep._bump("refused")
            for k, v in ep.counts.items():
                totals[k] = totals.get(k, 0) + v
        return write, other, totals

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would rewrite — pure read, no writes."""
        write, other, totals = self._scan(ctx.corpus_root)
        return (
            f"one person one entry plan: {len(write)} episode(s), {len(other)} refused/left; "
            + (", ".join(f"{k}={v}" for k, v in sorted(totals.items())))
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Nothing left to write and nothing refused. Two placed voices left are reported only."""
        write, other, _totals = self._scan(ctx.corpus_root)
        refused = [ep for ep in other if ep.refused]
        if write or refused:
            names = [ep.meta.name for ep in write + refused]
            return False, (
                f"{len(write)} episode(s) still list one person twice, {len(refused)} refused: "
                f"{names[:5]}"
            )
        left = [ep for ep in other if ep.left]
        return True, (
            "no served episode lists one person twice"
            + (f" (except {len(left)} with two placed voices left to the roster)" if left else "")
        )

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Write every rewritten file of every episode; back up each, receipt it."""
        root = ctx.corpus_root
        write, other, totals = self._scan(root)
        receipts: List[dict] = []
        if not ctx.dry_run:
            for ep in write:
                for path, payload in sorted(ep.json_files.items()):
                    receipts.append(write_with_backup(root, BACKUP_TAG, path, payload))
                for path, text in sorted(ep.text_files.items()):
                    receipts.append(_write_text_with_backup(root, path, text, BACKUP_TAG))
            append_receipts(root, RECEIPTS_FILE, {"migration": MIGRATION_ID}, receipts)
        verb = "would rewrite" if ctx.dry_run else "rewrote"
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=f"{verb} {len(write)} episode(s); "
            + ", ".join(f"{k}={v}" for k, v in sorted(totals.items())),
            details={
                "episodes": [
                    {
                        "meta": str(ep.meta.relative_to(root)),
                        "renames": ep.renames,
                        "dropped": ep.dropped,
                    }
                    for ep in write
                ],
                "refused": [
                    {"meta": str(ep.meta.relative_to(root)), "why": ep.refused}
                    for ep in other
                    if ep.refused
                ],
                "left": [
                    {"meta": str(ep.meta.relative_to(root)), "pairs": ep.left}
                    for ep in other
                    if ep.left and not ep.refused
                ],
                "totals": totals,
                "files_written": len(receipts),
            },
        )


def dry_run_report(root: Path) -> Iterable[Dict[str, Any]]:
    """Read-only: one row per episode the migration would write, refuse or leave."""
    m = OnePersonOneEntryMigration()
    write, other, _totals = m._scan(Path(root))
    for ep in write + [o for o in other if o not in write]:
        yield {
            "meta": str(ep.meta.relative_to(root)),
            "renames": ep.renames,
            "dropped": ep.dropped,
            "left": ep.left,
            "refused": ep.refused,
            "files": sorted(
                str(p.relative_to(root)) for p in list(ep.json_files) + list(ep.text_files)
            ),
            "counts": ep.counts,
        }

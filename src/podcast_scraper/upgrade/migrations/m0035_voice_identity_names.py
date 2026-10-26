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

Two differences from the rung as it runs in the pipeline, both so the migration publishes what
the pipeline would:

- **The talk floor.** The pipeline floors AFTER the rung, so a voice under the floor (5%) ends
  unnamed whatever the rung says; a host-read ad carries the host's fingerprint. Such voices are
  not listed: they are 0034's, and a review here could only answer "it is the host's voice".
- **The source.** A renamed voice's diagnostics ``source`` is ``voice_identity``, as the pipeline
  records the rung's own names, so a name this migration applies never votes in the feed bank.

THE LIST IS FROZEN AND REVIEWED, as 0034's. :func:`find_changes` replays the rung (the pipeline's
own ``voice_identity.decide`` on the stored fingerprints, threshold 0.7) over a corpus copy;
:func:`write_list` (0034's) freezes it in ``upgrade/lists/0035_voice_identity_names.json``. The
review judges a RENAME by its new name (``CORRECT`` / ``CORRECT_MISSPELLED`` applies it) and an
UNNAMING by the name it takes off (``WRONG_PERSON`` applies it); a listed voice without a verdict
fails ``verify``. The verdicts header binds the whole list (every field of every entry), and the
list file is part of the migration's digest. Applied through 0029's voice-by-voice rewrite with
0031's derived surfaces; an episode whose stored name moved on is refused untouched. Not in the
registry until the list and its verdicts exist. UNDO restores from
``.podcast_scraper/upgrade-backups/0035/``.
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
#: Voices the rung lists that this migration must not change, each with the operator's reason.
EXCLUDED: Dict[rv.Key, str] = {
    ("Buzzsprout-20005821", "SPEAKER_00"): (
        "operator 2026-10-26: the right name, but the role the feed's group gives (guest) is wrong "
        "for a voice that introduces itself as the stand-in host; a review cannot correct a role"
    ),
}
#: 0035's bases: a RENAME may be WRONG_PERSON on ``undecidable`` (the name is not added for lack
#: of evidence); an UNNAMING may not (a name comes off only when the text shows someone else).
BASES = {**rv.BASES, "WRONG_PERSON": rv.BASES["WRONG_PERSON"] + (rv.UNDECIDABLE,)}


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
    rows = m34._segments(meta_path, rel)
    served = m34.served_by_voice(rows)
    seconds = m34.talk_by_voice(rows)
    total = sum(seconds.values()) or 1.0
    talk = {
        str(v.get("voice")): float(v.get("talk_time_s") or 0.0) for v in diag.get("voices") or []
    }
    voices = {v: (name, role, talk.get(v, 0.0), vecs.get(v)) for v, (name, role) in served.items()}
    out: Dict[str, m29.Entry] = {}
    for voice, (new_name, new_role) in vi.decide(voices, groups, threshold).items():
        # The pipeline runs the talk floor AFTER the rung, so a name the rung gives a voice under
        # the floor comes off again: a host-read ad carries the host's fingerprint. Listing it
        # here would put back, behind a review that can only answer "it is the host's voice",
        # the name 0034 takes off. Voices under the floor are 0034's.
        if seconds.get(voice, 0.0) / total < m34.FLOOR:
            continue
        old_name, old_role = served.get(voice, (None, None))
        vec = vi._unit(vecs[voice]) if vecs.get(voice) else None
        if new_name:
            # The group ``decide`` matched: the voice's most similar one.
            sim = max(vi._cos(vec, g.centroid) for g in groups) if vec else 0.0
            why = f"matches the feed's {new_name!r} fingerprint group at {sim:.2f}"
            out[voice] = (old_name, old_role, new_name, new_role or old_role, why)
        else:
            # The group ``decide`` compared: the LAST group of the name taken off.
            same = [g for g in groups if g.person == vi._key(old_name)]
            sim = vi._cos(vec, same[-1].centroid) if vec and same else 0.0
            why = (
                f"clearly not {old_name!r}: similarity {sim:.2f} to the feed's fingerprint group "
                "for that name"
            )
            out[voice] = (old_name, old_role, None, None, why)
    return eid, out


def find_changes(root: Path) -> Dict[rv.Key, m29.Entry]:
    """The rung over every served episode of ``root`` that carries fingerprints (read only)."""
    out: Dict[rv.Key, m29.Entry] = {}
    for meta_path in select_served_artifacts(Path(root), ".metadata.json")[0]:
        eid, changes = episode_changes(meta_path)
        for voice, entry in changes.items():
            if (eid, voice) not in EXCLUDED:
                out[(eid, voice)] = entry
    return out


def load_list(path: Path = LIST_FILE) -> Dict[rv.Key, m29.Entry]:
    """The frozen list; empty when it has not been made yet."""
    return m34.load_list(path)


class _Episode(m34._Episode):
    """0034's rewrite, with a renamed voice's diagnostics source ``voice_identity``.

    The rung's own names never vote in the feed bank (``voice_identity._NON_VOTING_SOURCES``), so
    a match cannot confirm itself episode after episode. A name this migration applies is the
    rung's name (a review only confirms it), so it is recorded as the pipeline records it.
    """

    RENAMED_SOURCE = vi.SOURCE
    #: A respelling ("Jon Smith" -> the group's "John Smith") is the rung's name too.
    RESPELLING_TAKES_RENAMED_SOURCE = True


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
    _episode_class = _Episode
    _receipts_file = RECEIPTS_FILE
    _backup_tag = BACKUP_TAG
    _plan_label = "voice identity"
    _verified = (
        "every rename the review confirmed carries the fingerprint group's name, and every "
        "unnaming the review confirmed is off its voice"
    )

    def listed(self) -> Dict[rv.Key, m29.Entry]:
        """This migration's frozen list (``upgrade/lists/0035_voice_identity_names.json``)."""
        return load_list()

    def _verdicts(self) -> rv.Verdicts:
        """The frozen list's verdicts, with their basis columns and 0035's basis set."""
        verdicts = rv.load(
            self.id,
            self.review_names(),
            self._verdicts_dir,
            listed=self.listed(),
            require_basis=True,
            bases=BASES,
        )
        for key, entry in self.listed().items():
            v = verdicts.get(*key)
            if v is None or v.basis != rv.UNDECIDABLE:
                continue
            if not entry[2] and v.verdict == "WRONG_PERSON":
                raise rv.VerdictError(
                    f"{key}: an unnaming applies only when the text shows someone else, "
                    "not on 'undecidable'"
                )
            if entry[2] and v.verdict in ("CORRECT", "CORRECT_MISSPELLED"):
                raise rv.VerdictError(
                    f"{key}: a rename applies only on positive evidence in the text, "
                    "not on 'undecidable'"
                )
        return verdicts

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


def list_file() -> Path:
    """The frozen list this migration applies (a plan-level input, hashed into the code digest)."""
    return LIST_FILE


def dry_run_report(root: Path) -> Iterable[Dict[str, Any]]:
    """Read-only: one row per episode the migration would write or refuses, with its counters
    and input sha256s (0034's ``plan_rows``)."""
    return m34.plan_rows(root, VoiceIdentityNamesMigration())


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

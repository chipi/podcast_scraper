"""0009 — restore ``host`` / ``guest`` on Person nodes from the diarization roster (#2062).

WHAT WENT WRONG. ``generate_episode_metadata`` handed the graph builder its ``detected_hosts`` /
``detected_guests`` PARAMETERS — the pre-diarization hint, read from the feed and the show notes
before a second of audio was processed. For a network feed that hint is routinely empty: the host
is named by the transcript self-intro, not the RSS author. Diarization resolved the real roster
moments later into ``content.speakers``, and the graph call never looked at it.

MEASURED ON PRODUCTION (330-episode feed-stratified sample, 2026-09-13):

    Person nodes by role      mentioned 89.5%   host 9.9%   guest 0.6%
    roster named a guest      176 of 263 episodes (66.9%)
    that guest in kg.json     4.5%  -> 93.2% of roster-named guests were lost

So every human on an episode rendered as a contributor, and the guest — the person most listeners
open the episode for — was never shown as the guest.

WHAT THIS MIGRATION DOES. The roster's answer was never lost: it is on disk in each episode's
``content.speakers``, with the authoritative per-voice role. This reads it back and promotes the
matching Person node in ``.kg.json``. No GPU, no LLM, no re-transcription — the data is already
there, it was simply not carried across one function call.

WHAT IT DELIBERATELY DOES NOT DO, because that part matters:

  * **It demotes a speaker role the roster contradicts, and only then.** Host and guest are
    SPEAKING roles. 39.5% of the ``host`` nodes in the sample — and 40% of the ``guest`` nodes —
    name someone who never spoke in that episode: a co-host who sat the episode out ("Sarah Guo" on
    an episode where Elad Gil interviews Glenn Fogel) or the show's own name as a person ("The
    China-Global South Project"). Those came from the same pre-diarization hint, so leaving them
    while promoting the real speakers would leave the episode claiming two hosts, one of whom was
    never there. A node the roster contradicts becomes ``mentioned`` — still in the graph, no
    longer claiming a microphone.

    A SPELLING VARIANT IS NOT A STRANGER, and an earlier version of this text wrongly listed one as
    a reason to demote. "Bernard Leong" against a roster that heard "Bernard Leung", or "Alexandra
    Karppi" against "Alexander Carpi", means the roster misheard the NAME — not that the human was
    absent — and demoting them replaces a wrong spelling with a wrong ROLE, so the host of the
    episode stops being its host. Measured on 287 production artifacts, exact matching did that to
    5 of its 70 demotions. Matching therefore uses ``kg.speaker_coherence.same_person`` in both
    directions: a variant-named node is promoted rather than reported missing, and never demoted.
    After the change the same 287 artifacts give 381 promotions (up from 336, the variants
    recovered), 65 demotions, and 0 real speakers stripped.

    Demotion happens ONLY for an episode that HAS a roster: with no roster there is no evidence to
    contradict anything, and the node is left untouched.
  * **It never ADDS a Person node.** 14.6% of roster speakers have no slug match in their episode's
    graph, and the sample shows why: they are near-miss NAME VARIANTS, not absent people —
    ``bernt børnich`` vs ``bernt bornich``, ``dr. alexander douglas`` vs ``alexander douglas``,
    ``professor zhang`` vs ``professor zhang chuan hong``. Inserting a node for those would put the
    same human in the graph twice, which is worse than the mislabelling being fixed. They are
    reported in ``details`` and need a re-enrichment (or #2056's variant resolver), not a guess.

EXPECTED REACH, measured on the same sample: 333 of 515 roster speakers (64.7%) are promoted —
167 hosts and 166 guests currently sitting as ``mentioned``. 20.8% are already correct and are
skipped; the remaining 14.6% are the variant cases above.

Idempotent: a second run finds every promoted node already ``host``/``guest`` and writes nothing.
Unparsable files are recorded and skipped rather than failing the run (mirrors 0003/0005/0006/0007).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ...identity.slugify import person_id
from ...kg.speaker_coherence import same_person
from ..migration import Migration, MigrationContext, MigrationResult

#: Roles this pass may WRITE. Anything else on a node means someone who knew more got there first.
_SPEAKER_ROLES = frozenset({"host", "guest"})

#: Roles this pass may OVERWRITE. A node already carrying a speaker role is never touched.
_PROMOTABLE = frozenset({"", "mentioned"})


def _iter_kg_files(root: Path) -> Iterable[Path]:
    """All ``*.kg.json`` files under *root* (recursive). Stable order."""
    return sorted(root.rglob("*.kg.json"))


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


def _metadata_sibling(kg_path: Path) -> Path:
    """``…/X.kg.json`` -> ``…/X.metadata.json`` (the two are written side by side)."""
    return kg_path.with_name(kg_path.name[: -len(".kg.json")] + ".metadata.json")


def roster_roles(metadata_payload: dict) -> Dict[str, str]:
    """``{person_id: role}`` for every named voice the diarization roster placed.

    Keyed by ``person_id`` — the same slug rule the graph itself uses — so an accent, an
    apostrophe, a double space or a case difference between the roster and the node cannot cause a
    miss. Only ``host`` / ``guest`` are returned; a roster entry with any other role carries no
    information this pass is allowed to write.
    """
    content = metadata_payload.get("content") or {}
    speakers = content.get("speakers") or []
    out: Dict[str, str] = {}
    for entry in speakers:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        role = str(entry.get("role") or "").strip().lower()
        if not name or role not in _SPEAKER_ROLES:
            continue
        pid = person_id(name)
        # First voice wins, matching the roster builder's own first-appearance precedence.
        out.setdefault(pid, role)
    return out


def _fuzzy_roster_hit(name: str, roles: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """``(roster_key, role)`` for the roster entry that plausibly names the same human, else
    ``(None, None)``.

    The roster keys are ``person_id`` slugs, so the name is recovered from the slug to compare
    against — good enough for a spelling comparison, and it keeps this migration reading the same
    ``content.speakers`` the rest of the pass reads.
    """
    if not name.strip():
        return None, None
    for key in roles:
        candidate = key.split(":", 1)[-1].replace("-", " ")
        if same_person(name, candidate):
            return key, roles[key]
    return None, None


def promote_person_roles(kg_payload: dict, roles: Dict[str, str]) -> Tuple[int, int, List[str]]:
    """Align Person roles in *kg_payload* with the roster. ``(promoted, demoted, unmatched)``.

    Mutates *kg_payload* in place. A node is matched on its ``id`` first (already a ``person:``
    slug) and on the slug of its display name second, because older artifacts predate the id rule.

    Two directions, both driven by the same fact — the roster knows who spoke:

    * a node the roster names, currently ``mentioned`` or roleless, is PROMOTED to its roster role;
    * a node claiming ``host``/``guest`` that the roster does NOT name is DEMOTED to ``mentioned``,
      because it did not speak in this episode.

    *roles* must be non-empty; an episode with no roster carries no evidence and must not reach
    here, or every speaker in it would be demoted.
    """
    if not roles:
        raise ValueError("promote_person_roles requires a non-empty roster; see the docstring")
    matched: set = set()
    promoted = 0
    demoted = 0
    for node in kg_payload.get("nodes") or []:
        if node.get("type") != "Person":
            continue
        props = node.setdefault("properties", {})
        nid = str(node.get("id") or "")
        name = str(props.get("name") or "")
        key: Optional[str] = nid if nid in roles else person_id(name)
        role: Optional[str] = roles.get(key) if key else None
        if role is None:
            # No exact hit. Before concluding this person is not on the roster, allow for the
            # roster having MISHEARD the name (#2062). Measured on 287 production artifacts: of 70
            # demotions, 5 stripped a real speaker — "Bernard Leong" against a roster that heard
            # "Bernard Leung", "Alexandra Karppi" against "Alexander Carpi". A variant means we
            # misheard the NAME, not that the human was absent, and demoting them replaces a wrong
            # spelling with a wrong ROLE: the host of the episode stops being its host.
            #
            # `same_person` is the coherence guard's predicate, so "is this the same person" has
            # one answer in the codebase rather than two that drift.
            key, role = _fuzzy_roster_hit(name, roles)
        current = str(props.get("role") or "").strip().lower()
        if role is None:
            # Genuinely not on the roster. If it claims to have spoken, the roster contradicts it.
            if current in _SPEAKER_ROLES:
                props["role"] = "mentioned"
                demoted += 1
            continue
        if key:
            matched.add(key)
        if current not in _PROMOTABLE:
            continue  # already carries a speaker role the roster agrees with
        props["role"] = role
        promoted += 1
    return promoted, demoted, sorted(set(roles) - matched)


class BackfillSpeakerRolesMigration(Migration):
    """Promote mislabelled Person nodes to the roster's host/guest role (#2062)."""

    id = "0009_backfill_speaker_roles"
    to_version = "2.7.3"
    description = (
        "#2062: restore host/guest on Person nodes from each episode's own content.speakers. The "
        "graph was built from the PRE-DIARIZATION hint, so 93.2% of roster-named guests reached "
        "kg.json as 'mentioned' and every human rendered as a contributor. Promotes a node the "
        "roster names, and DEMOTES to 'mentioned' a node claiming host/guest that the roster "
        "contradicts (39.5% of host nodes named someone who never spoke). Spelling variants are "
        "matched, not demoted. Never adds a node: a roster name with no node behind it needs a "
        "re-enrichment, and inserting one would duplicate the person"
    )

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        files = list(_iter_kg_files(ctx.corpus_root))
        changed: List[str] = []
        promoted_total = 0
        demoted_total = 0
        no_roster = 0
        already_correct = 0
        unmatched: List[str] = []
        unparsable: List[str] = []

        for path in files:
            payload, err = _load(path)
            if payload is None:
                unparsable.append(f"{path.name}: {err}")
                continue
            meta_path = _metadata_sibling(path)
            if not meta_path.is_file():
                no_roster += 1
                continue
            meta_payload, meta_err = _load(meta_path)
            if meta_payload is None:
                unparsable.append(f"{meta_path.name}: {meta_err}")
                continue
            roles = roster_roles(meta_payload)
            if not roles:
                no_roster += 1
                continue
            promoted, demoted, missing = promote_person_roles(payload, roles)
            unmatched.extend(f"{path.name}: {pid}" for pid in missing)
            if not promoted and not demoted:
                already_correct += 1
                continue
            promoted_total += promoted
            demoted_total += demoted
            changed.append(str(path.relative_to(ctx.corpus_root)))
            if not ctx.dry_run:
                _write_atomic(path, payload)

        message = (
            f"{'would promote' if ctx.dry_run else 'promoted'} {promoted_total} Person node(s) to "
            f"their roster role and {'would demote' if ctx.dry_run else 'demoted'} "
            f"{demoted_total} that the roster says never spoke, across {len(changed)} "
            f"artifact(s); {already_correct} already "
            f"correct, {no_roster} with no roster on disk, {len(unmatched)} roster name(s) with no "
            f"matching node (name variants — these need a re-enrich, not this migration), "
            f"{len(unparsable)} unparsable"
        )
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=message,
            details={
                "artifacts_scanned": len(files),
                "changed": len(changed),
                "persons_promoted": promoted_total,
                "persons_demoted": demoted_total,
                "already_correct": already_correct,
                "no_roster": no_roster,
                "unmatched_roster_names": len(unmatched),
                "unmatched_sample": unmatched[:20],
                "unparsable": unparsable[:20],
            },
        )

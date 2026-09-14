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

  * **It demotes only what it can prove did not speak, by two independent routes.** Host and
    guest are SPEAKING roles, and on production a large share of them name someone who was never
    in the room — a co-host who sat the episode out ("Sarah Guo" on an episode where Elad Gil
    interviews Glenn Fogel), or the show's own name as a person ("Africa Tech Summit"). Those came
    from the same pre-diarization hint, so promoting the real speakers without touching these
    would leave the episode claiming two hosts, one of whom was never there.

    The two routes answer different questions, and keeping them apart is what makes the pass safe:

    1. **The node is not a human.** A show name or a role word never held a microphone in any
       episode. That needs no roster and runs even where the roster is unusable — which is exactly
       where these sit, because a roster of nothing but show names is what :func:`roster_roles`
       throws away. See :func:`demote_non_persons`.
    2. **The roster accounts for every voice and this person is not among them.** Absence is
       evidence only when the roster explains everyone who spoke. Counting ``content.speakers``
       does NOT establish that: an entry the guards discard still padded the count, and an entry
       can name a voice with something that is not a person at all. On "How AWS S3 is built"
       diarization heard 2 voices and named ``['Gergely Orosz', 'Developer Survey']``; reading
       that as a full account demoted the actual guest, Mai-Lan Tomsen Bukovec, out of the seat
       the survey was occupying. So the denominator is the roster entries THIS GRAPH CAN PLACE —
       an entry matching no Person node here is an unresolved name, not a witness.

    A SPELLING VARIANT IS NOT A STRANGER, and an earlier version of this text wrongly listed one as
    a reason to demote. "Bernard Leong" against a roster that heard "Bernard Leung", or "Alexandra
    Karppi" against "Alexander Carpi", means the roster misheard the NAME — not that the human was
    absent — and demoting them replaces a wrong spelling with a wrong ROLE, so the host of the
    episode stops being its host. Measured on 287 production artifacts, exact matching did that to
    5 of its 70 demotions. Matching therefore uses ``kg.speaker_coherence.same_person`` in both
    directions: a variant-named node is promoted rather than reported missing, and never demoted.

    MEASURED ON THE 287-ARTIFACT PRODUCTION STAGING COPY, this pass gives **380 promotions and 21
    demotions**, and every one of the 21 was read individually: 17 are a show name in the host
    seat, 3 are a co-host or publisher org who demonstrably did not appear on that episode
    (checked against the episode description), and 1 is a co-host the roster confirms was absent.
    **None is a real speaker.** Person roles across the sample move
    ``mentioned 89.5% / host 9.9% / guest 0.6%`` to ``mentioned 68.1% / host 19.7% / guest 12.2%``.

    What it still cannot reach: an ORG in the host seat whose name is not the show's
    ("Mercatus Center at George Mason University" on *Conversations with Tyler*) survives, because
    no predicate here can tell it from a person and its episode's roster cannot speak to it.
    Those need a re-enrichment, not a migration.
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

from ...graph_id_utils import is_bare_speaker_label
from ...identity.slugify import person_id
from ...kg.speaker_coherence import same_person
from ...speaker_detectors.hosts import names_the_show
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
    feed_title = str((metadata_payload.get("feed") or {}).get("title") or "")
    out: Dict[str, str] = {}
    for entry in speakers:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        role = str(entry.get("role") or "").strip().lower()
        if not name or role not in _SPEAKER_ROLES:
            continue
        # THE ROSTER ON DISK CAN ITSELF BE WRONG (#2064/#2059). These artifacts were written before
        # the show-name and role-word fixes, so `content.speakers` still says `host="Africa Tech
        # Summit"` or `host="Host"`. Reading it as ground truth does not merely fail to repair those
        # episodes — it PROMOTES the bad name into a `host` role, entrenching the defect in the one
        # pass that is irreversible on production. Measured on the 330-episode sample: 25 entries
        # (13 show names, 12 role-word placeholders) would have been promoted.
        #
        # Same two predicates the pipeline uses, so the migration and the pipeline cannot disagree
        # about what counts as a person.
        if is_bare_speaker_label(name):
            continue
        if feed_title and names_the_show(name, feed_title):
            continue
        pid = person_id(name)
        # First voice wins, matching the roster builder's own first-appearance precedence.
        out.setdefault(pid, role)
    return out


def voices_heard(metadata_payload: dict) -> Optional[int]:
    """How many distinct voices diarization HEARD on this episode, or ``None`` when unknowable.

    This is the denominator for "does the roster account for everyone who spoke". It is NOT the
    same question as "did diarization name every voice": ``content.speakers`` can name a voice
    with something no one can resolve to a human — see
    :func:`promote_person_roles`, which compares this count against the roster entries the
    episode graph can actually place.

    Unknowable counts as no evidence. 6.6% of production episodes carry no
    ``diarization_num_speakers``, and when diarization names nobody the pipeline substitutes the
    PRE-DIARIZATION HINT into ``content.speakers`` wholesale (``metadata_generation``:
    ``speakers = diarized_speakers or _build_speakers_from_detected_names(...)``), which is
    indistinguishable on disk from a real roster. With no count there is nothing to reconcile, so
    the roster may not deny anyone.
    """
    content = metadata_payload.get("content") or {}
    heard = content.get("diarization_num_speakers")
    if isinstance(heard, bool) or not isinstance(heard, int) or heard <= 0:
        return None
    return heard


def _is_not_a_person(name: str, feed_title: str) -> bool:
    """True when this NODE cannot hold a speaking role, whatever any roster says.

    A show name and a role word are not humans, so they never held a microphone. That is a fact
    about the node, not evidence about who spoke, which is why it is separate from the roster
    reconciliation below and needs no roster at all.
    """
    if not name.strip():
        return False
    if is_bare_speaker_label(name):
        return True
    return bool(feed_title and names_the_show(name, feed_title))


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


def demote_non_persons(kg_payload: dict, feed_title: str = "") -> int:
    """Take every non-human out of a speaking role in *kg_payload*; return how many.

    Needs no roster: that the show's own name, or the word "Host", never spoke is true of the node
    itself. Split out because the episodes most likely to carry one are precisely the episodes
    whose roster :func:`roster_roles` throws away entirely, which the roster-driven pass skips.
    """
    demoted = 0
    for node in kg_payload.get("nodes") or []:
        if node.get("type") != "Person":
            continue
        props = node.setdefault("properties", {})
        current = str(props.get("role") or "").strip().lower()
        if current in _SPEAKER_ROLES and _is_not_a_person(str(props.get("name") or ""), feed_title):
            props["role"] = "mentioned"
            demoted += 1
    return demoted


def promote_person_roles(
    kg_payload: dict,
    roles: Dict[str, str],
    *,
    voices_heard: Optional[int] = None,
    feed_title: str = "",
) -> Tuple[int, int, List[str]]:
    """Align Person roles in *kg_payload* with the roster. ``(promoted, demoted, unmatched)``.

    Mutates *kg_payload* in place. A node is matched on its ``id`` first (already a ``person:``
    slug), on the slug of its display name second (older artifacts predate the id rule), and on
    :func:`~podcast_scraper.kg.speaker_coherence.same_person` third, so a roster that MISHEARD a
    name promotes the human rather than reporting them missing.

    Three directions, and only the first two are driven by the roster:

    * a node the roster names, currently ``mentioned`` or roleless, is PROMOTED to its roster role;
    * a node claiming ``host``/``guest`` that the roster does NOT name is DEMOTED to ``mentioned``
      — but ONLY when the roster accounts for every voice heard (see below);
    * a node that is not a human at all — the show's own name, a role word — is demoted whatever
      the roster says, because it never spoke in any episode.

    WHEN MAY SILENCE DENY SOMEONE. Absence from the roster is evidence of not speaking only if the
    roster explains every voice diarization heard. Counting ``content.speakers`` does not settle
    that, and production shows two ways it comes apart: an entry the guards in
    :func:`roster_roles` discard still padded the count, and an entry can NAME a voice with
    something that is not a person — "How AWS S3 is built" heard 2 voices and named
    ``['Gergely Orosz', 'Developer Survey']``, then demoted the actual guest, Mai-Lan Tomsen
    Bukovec, out of the seat the survey was occupying.

    So the denominator is the roster entries THIS GRAPH CAN PLACE: an entry that resolves to no
    Person node here is an unresolved name, not a witness. The pass already refuses to insert a
    node for such an entry; it equally refuses to count it. Over 287 production artifacts this
    takes 13 demotions to 11, removing the only wrong one and one org the roster could not speak
    to; promotions are unaffected, because promotion adds information and is safe whatever the
    roster omits.

    *roles* must be non-empty; an episode with no roster carries no evidence and must not reach
    here, or every speaker in it would be demoted.
    """
    if not roles:
        raise ValueError("promote_person_roles requires a non-empty roster; see the docstring")

    persons = [n for n in (kg_payload.get("nodes") or []) if n.get("type") == "Person"]

    # PASS 1 — which roster entries does this episode's graph actually account for?
    resolved: Dict[int, Tuple[Optional[str], Optional[str]]] = {}
    matched: set = set()
    for idx, node in enumerate(persons):
        props = node.setdefault("properties", {})
        nid = str(node.get("id") or "")
        name = str(props.get("name") or "")
        key: Optional[str] = nid if nid in roles else person_id(name)
        role: Optional[str] = roles.get(key) if key else None
        if role is None:
            key, role = _fuzzy_roster_hit(name, roles)
        resolved[idx] = (key, role)
        if role is not None and key:
            matched.add(key)

    roster_accounts_for_every_voice = voices_heard is not None and voices_heard <= len(matched)

    # PASS 2 — apply.
    promoted = 0
    demoted = 0
    for idx, node in enumerate(persons):
        props = node["properties"]
        name = str(props.get("name") or "")
        key, role = resolved[idx]
        current = str(props.get("role") or "").strip().lower()
        if current in _SPEAKER_ROLES and _is_not_a_person(name, feed_title):
            props["role"] = "mentioned"
            demoted += 1
            continue
        if role is None:
            if roster_accounts_for_every_voice and current in _SPEAKER_ROLES:
                props["role"] = "mentioned"
                demoted += 1
            continue
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
        "contradicts (19.4% of host nodes on a complete roster). Spelling variants are "
        "matched, not demoted. Never adds a node: a roster name with no node behind it needs a "
        "re-enrichment, and inserting one would duplicate the person"
    )

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        files = list(_iter_kg_files(ctx.corpus_root))
        changed: List[str] = []
        promoted_total = 0
        demoted_absent = 0
        demoted_non_person = 0
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
            feed_title = str((meta_payload.get("feed") or {}).get("title") or "")

            # Route 1 — the node is not a human. Needs no roster, so it runs on EVERY episode,
            # including the ones whose roster `roster_roles` discards entirely (usually because
            # every entry in it WAS a show name). Running it first also means the count below is
            # cleanly split: whatever `promote_person_roles` demotes afterwards is the roster
            # route alone, and the operator can see which rule moved what before trusting either.
            non_person = demote_non_persons(payload, feed_title)

            # Route 2 — the roster accounts for every voice and this person is not among them.
            roles = roster_roles(meta_payload)
            promoted = demoted = 0
            if roles:
                promoted, demoted, missing = promote_person_roles(
                    payload,
                    roles,
                    voices_heard=voices_heard(meta_payload),
                    feed_title=feed_title,
                )
                unmatched.extend(f"{path.name}: {pid}" for pid in missing)
            else:
                no_roster += 1

            if not promoted and not demoted and not non_person:
                if roles:
                    already_correct += 1
                continue
            promoted_total += promoted
            demoted_absent += demoted
            demoted_non_person += non_person
            changed.append(str(path.relative_to(ctx.corpus_root)))
            if not ctx.dry_run:
                _write_atomic(path, payload)

        message = (
            f"{'would promote' if ctx.dry_run else 'promoted'} {promoted_total} Person node(s) to "
            f"their roster role and {'would demote' if ctx.dry_run else 'demoted'} "
            f"{demoted_absent + demoted_non_person} from a speaking role "
            f"({demoted_non_person} that are not a person at all, {demoted_absent} the roster "
            f"accounts for every voice without naming), across {len(changed)} "
            f"artifact(s); {already_correct} already "
            f"correct, {no_roster} with no usable roster on disk, {len(unmatched)} roster name(s) "
            f"with no matching node (name variants — these need a re-enrich, not this migration), "
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
                "persons_demoted": demoted_absent + demoted_non_person,
                "persons_demoted_not_a_person": demoted_non_person,
                "persons_demoted_roster_denies": demoted_absent,
                "already_correct": already_correct,
                "no_roster": no_roster,
                "unmatched_roster_names": len(unmatched),
                "unmatched_sample": unmatched[:20],
                "unparsable": unparsable[:20],
            },
        )

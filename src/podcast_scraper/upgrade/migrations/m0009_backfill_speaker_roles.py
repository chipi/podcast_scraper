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
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ...graph_id_utils import is_bare_speaker_label
from ...identity.slugify import person_id as _person_id
from ...kg.speaker_coherence import same_person
from ...speaker_detectors.hosts import names_the_show
from ..migration import Migration, MigrationContext, MigrationResult
from ..role_ledger import append_ledger, file_sha, new_run_id, read_ledger, RoleChange

#: Roles this pass may WRITE. Anything else on a node means someone who knew more got there first.
_SPEAKER_ROLES = frozenset({"host", "guest"})

#: Edge types whose target is a person the roster HEARD speak. A show does not speak.
_SPOKEN_BY = "SPOKEN_BY"

logger = logging.getLogger(__name__)


def person_id(name: str) -> Optional[str]:
    """``identity.slugify.person_id``, or ``None`` when the name cannot be slugged.

    ``slugify.person_id`` RAISES when NFKD -> ASCII leaves nothing — every entirely non-Latin name
    does this (``person_id('张川红')``, ``person_id('Владимир Путин')``). This pass walks the whole
    corpus and ``apply()`` catches only OSError/JSONDecodeError, so one such name aborted the run
    mid-corpus with no resume point: the ledger is never written, and every re-run dies on the same
    file. Production carries Round Table China, China Plus, ChinaTalk and The Naked Pravda.

    An unsluggable name is not a migration failure, it is a name this pass cannot key on. Skip it
    and keep going — the node keeps whatever role it already had, which is the safe direction.
    """
    try:
        return _person_id(name)
    except Exception:  # noqa: BLE001 — an unkeyable name must not cost the whole corpus its run
        return None


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


def _gi_sibling(kg_path: Path) -> Path:
    """``…/X.kg.json`` -> ``…/X.gi.json`` (written side by side with the other two)."""
    return kg_path.with_name(kg_path.name[: -len(".kg.json")] + ".gi.json")


def voices_in_episode(gi_payload: dict, kg_payload: dict) -> Set[str]:
    """KG person ids the roster HEARD SPEAK — read from the GI layer, matched into the KG layer.

    ``SPOKEN_BY`` DOES NOT EXIST IN ``kg.json``. Measured over 287 production artifacts: the KG
    layer carries only ``HAS_EPISODE`` / ``MENTIONS`` / ``HOSTS`` / ``GUESTS_ON``, while 259 of the
    sibling ``gi.json`` files carry ``SPOKEN_BY``. An earlier version of this guard read the KG
    payload's own edges, so it returned an empty set on every real artifact and never fired — the
    unit test passed only because it hand-built a KG containing an edge type that shape never has.

    The two layers also mint DIFFERENT ids for one human (#2056: ``person:aaron-levy`` from the
    roster in GI, ``person:aaron-levie`` from the extractor in KG), so an exact id match is not
    enough. Exact first, then ``same_person`` against the GI speaker's display name — the same
    predicate the rest of this migration uses, so "is this the same human" keeps one answer.
    """
    spoken_ids: Set[str] = set()
    for edge in gi_payload.get("edges") or []:
        if not isinstance(edge, dict) or str(edge.get("type") or "") != _SPOKEN_BY:
            continue
        for key in ("to", "from"):
            target = str(edge.get(key) or "")
            if target.startswith("person:"):
                spoken_ids.add(target)
    if not spoken_ids:
        return set()
    gi_names = {
        str(n.get("id") or ""): str((n.get("properties") or {}).get("name") or "")
        for n in (gi_payload.get("nodes") or [])
        if isinstance(n, dict) and str(n.get("type") or "").lower() == "person"
    }
    spoken_names = [gi_names.get(i, "") for i in spoken_ids]
    out: Set[str] = set()
    for node in kg_payload.get("nodes") or []:
        if not isinstance(node, dict) or node.get("type") != "Person":
            continue
        nid = str(node.get("id") or "")
        if nid in spoken_ids:
            out.add(nid)
            continue
        nm = str((node.get("properties") or {}).get("name") or "")
        if nm and any(sn and same_person(nm, sn) for sn in spoken_names):
            out.add(nid)
    return out


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
        if pid is None:
            continue
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


#: Returned by :func:`_fuzzy_roster_hit` when a node plausibly names MORE THAN ONE roster entry.
#: Distinct from "no match": no match is evidence the person did not speak, ambiguity is evidence
#: of nothing at all, and the two must not be treated the same.
AMBIGUOUS = "<ambiguous>"


def _fuzzy_roster_hit(name: str, roles: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """``(roster_key, role)`` for the roster entry that plausibly names the same human.

    ``(None, None)`` when nothing matches, and ``(AMBIGUOUS, None)`` when more than one entry does.

    The roster keys are ``person_id`` slugs, so the name is recovered from the slug to compare
    against — good enough for a spelling comparison, and it keeps this migration reading the same
    ``content.speakers`` the rest of the pass reads.

    AMBIGUITY IS NOT A MATCH. This used to return the FIRST hit, and ``same_person`` treats a token
    subset as one human (``same_person('John', 'John Smith')`` is True), so a bare-mononym node was
    promoted to whichever John the dict yielded first — a role landed on the wrong person while the
    real full-name node kept its own. That is #2056's duplicate symptom manufactured BY the repair,
    irreversible, and invisible to the coherence checks because they share this predicate.
    """
    if not name.strip():
        return None, None
    hits = [key for key in roles if same_person(name, key.split(":", 1)[-1].replace("-", " "))]
    if not hits:
        return None, None
    if len(hits) > 1:
        return AMBIGUOUS, None
    return hits[0], roles[hits[0]]


def demote_non_persons(
    kg_payload: dict,
    feed_title: str = "",
    voices: Optional[Set[str]] = None,
    suspects: Optional[List[str]] = None,
    changes: Optional[List[Tuple[int, str, str, Optional[str], str, str]]] = None,
) -> int:
    """Take every non-human out of a speaking role in *kg_payload*; return how many.

    Needs no roster: that the show's own name, or the word "Host", never spoke is true of the node
    itself. Split out because the episodes most likely to carry one are precisely the episodes
    whose roster :func:`roster_roles` throws away entirely, which the roster-driven pass skips.

    THE KNOWN FALSE POSITIVE, and why it is not guarded here. ``names_the_show`` matches a
    multi-token PREFIX of the feed title, so it also classifies a host whose name LEADS their own
    show as the show — ``names_the_show('Lex Fridman', 'Lex Fridman Podcast')`` is True, likewise
    Rich Roll, Ezra Klein and Dan Carlin. The title cannot settle it: 'Lex Fridman Podcast' and
    'Latent Space: The AI Engineer Podcast' are structurally identical, and 'Latent.Space' really
    IS the show.

    "A show does not speak" LOOKS like the missing evidence and is not. ``SPOKEN_BY`` inherits the
    roster's mistakes — it is the roster that put the show in the host seat in the first place
    (#2064), so the show then has quotes attributed to it. Measured: 'Machine Learning Street' has
    a SPOKEN_BY edge on 4 of its 6 episodes. Sparing every node with a voice therefore rescues 4
    real show names that this pass exists to demote, to protect a host shape that does not occur
    in the corpus at all (all 8 title/name hits across the 55 production feeds are genuine show
    names). Trading measured damage for hypothetical damage is the wrong direction.

    So the prefix rule stands, and the risk is REPORTED instead: see ``suspect_demotions`` in the
    result details, which lists every demotion whose node also carried a voice. That is the class
    to hand-read before running on a corpus containing an eponymous show.
    """
    demoted = 0
    voices = voices or set()
    suspects = suspects if suspects is not None else []
    changes = changes if changes is not None else []
    for idx, node in enumerate(kg_payload.get("nodes") or []):
        if node.get("type") != "Person":
            continue
        props = node.setdefault("properties", {})
        current = str(props.get("role") or "").strip().lower()
        if current not in _SPEAKER_ROLES:
            continue
        name = str(props.get("name") or "")
        if is_bare_speaker_label(name):
            # A role word is not a human even if something attributed a quote to it.
            changes.append(
                (
                    idx,
                    str(node.get("id") or ""),
                    name,
                    props.get("role"),
                    "mentioned",
                    "not_a_person",
                )
            )
            props["role"] = "mentioned"
            demoted += 1
            continue
        if not (feed_title and names_the_show(name, feed_title)):
            continue
        if str(node.get("id") or "") in voices:
            # Demoted anyway — see the docstring — but recorded, because this is exactly the
            # shape an eponymous-show host would take.
            suspects.append(f"{name} (host of {feed_title!r}, had a voice)")
        changes.append(
            (idx, str(node.get("id") or ""), name, props.get("role"), "mentioned", "not_a_person")
        )
        props["role"] = "mentioned"
        demoted += 1
    return demoted


def promote_person_roles(
    kg_payload: dict,
    roles: Dict[str, str],
    *,
    voices_heard: Optional[int] = None,
    feed_title: str = "",
    ambiguous: Optional[List[str]] = None,
    changes: Optional[List[Tuple[int, str, str, Optional[str], str, str]]] = None,
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

    ambiguous_out = ambiguous if ambiguous is not None else []
    changes_out = changes if changes is not None else []
    _all_nodes = kg_payload.get("nodes") or []
    persons = [n for n in _all_nodes if n.get("type") == "Person"]
    # Position in `nodes`, so same-id duplicates in one file stay individually addressable on undo.
    node_index = [i for i, n in enumerate(_all_nodes) if n.get("type") == "Person"]

    # PASS 1 — which roster entries does this episode's graph actually account for?
    resolved: Dict[int, Tuple[Optional[str], Optional[str]]] = {}
    matched: set = set()
    for idx, node in enumerate(persons):
        props = node.setdefault("properties", {})
        nid = str(node.get("id") or "")
        name = str(props.get("name") or "")
        key: Optional[str] = nid if nid in roles else person_id(name)  # None when unsluggable
        role: Optional[str] = roles.get(key) if key else None
        if role is None:
            key, role = _fuzzy_roster_hit(name, roles)
        resolved[idx] = (key, role)
        if key == AMBIGUOUS:
            # Neither promote nor demote, and do NOT consume a roster key: leaving it unmatched
            # would let route (b) read the gap as "nobody spoke" and demote someone real.
            ambiguous_out.append(f"{name!r} matches more than one roster entry: {sorted(roles)}")
        elif role is not None and key:
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
        if key == AMBIGUOUS:
            continue  # evidence of nothing — see `_fuzzy_roster_hit`
        if current in _SPEAKER_ROLES and _is_not_a_person(name, feed_title):
            changes_out.append(
                (
                    node_index[idx],
                    str(node.get("id") or ""),
                    name,
                    props.get("role"),
                    "mentioned",
                    "not_a_person",
                )
            )
            props["role"] = "mentioned"
            demoted += 1
            continue
        if role is None:
            if roster_accounts_for_every_voice and current in _SPEAKER_ROLES:
                changes_out.append(
                    (
                        node_index[idx],
                        str(node.get("id") or ""),
                        name,
                        props.get("role"),
                        "mentioned",
                        "roster_denies",
                    )
                )
                props["role"] = "mentioned"
                demoted += 1
            continue
        if current not in _PROMOTABLE:
            continue  # already carries a speaker role the roster agrees with
        changes_out.append(
            (node_index[idx], str(node.get("id") or ""), name, props.get("role"), role, "promote")
        )
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

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Is this migration's effect actually PRESENT on the corpus right now?

        The base class default is "no verification defined", which made `make upgrade-verify` a
        no-op for 0009 — the one hook designed to answer this question answered "I do not check".
        That matters most immediately after a rollback: the roles are back where they started and
        nothing else in the system can tell.

        Checks the ledger's own rows against the artifacts: every recorded node should still hold
        its `role_after`. Samples rather than reads the corpus twice — the ledger already names
        exactly the nodes this migration touched, which is a far smaller set than the corpus.

        No ledger is NOT a failure: a corpus migrated before the ledger existed, or one where the
        migration legitimately changed nothing, has nothing to check. Saying so beats inventing a
        verdict.
        """
        try:
            rows = read_ledger(ctx.corpus_root)
        except ValueError as exc:
            return False, f"role ledger is unreadable: {exc}"
        if not rows:
            return True, "no ledger to verify against (corpus predates it, or nothing changed)"
        present = 0
        for row in rows:
            payload, _err = _load(Path(ctx.corpus_root) / row.episode)
            if payload is None:
                continue
            for node in payload.get("nodes") or []:
                if not isinstance(node, dict) or str(node.get("id") or "") != row.node_id:
                    continue
                if str((node.get("properties") or {}).get("role") or "") == row.role_after:
                    present += 1
                break
        ok = present == len(rows)
        return ok, f"{present} of {len(rows)} recorded role(s) still present"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Reconcile every episode's Person roles with its roster; report both demotion routes.

        Per episode: strip non-humans out of speaking roles (needs no roster, so it runs even
        where the roster is unusable), then, if the roster is usable, promote and demote against
        it. Episodes with no readable metadata sibling are counted, not failed.
        """
        files = list(_iter_kg_files(ctx.corpus_root))
        changed: List[str] = []
        promoted_total = 0
        demoted_absent = 0
        demoted_non_person = 0
        no_roster = 0
        already_correct = 0
        unmatched: List[str] = []
        suspect_demotions: List[str] = []
        ambiguous_nodes: List[str] = []
        run_id = new_run_id()
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
            # SPOKEN_BY lives in the GI layer, not the KG layer — see `voices_in_episode`.
            gi_payload, _gi_err = _load(_gi_sibling(path))
            voices = voices_in_episode(gi_payload or {}, payload)

            # Route 1 — the node is not a human. Needs no roster, so it runs on EVERY episode,
            # including the ones whose roster `roster_roles` discards entirely (usually because
            # every entry in it WAS a show name). Running it first also means the count below is
            # cleanly split: whatever `promote_person_roles` demotes afterwards is the roster
            # route alone, and the operator can see which rule moved what before trusting either.
            episode_changes: List[Tuple[int, str, str, Optional[str], str, str]] = []
            non_person = demote_non_persons(
                payload, feed_title, voices, suspect_demotions, episode_changes
            )

            # Route 2 — the roster accounts for every voice and this person is not among them.
            roles = roster_roles(meta_payload)
            promoted = demoted = 0
            if roles:
                promoted, demoted, missing = promote_person_roles(
                    payload,
                    roles,
                    voices_heard=voices_heard(meta_payload),
                    feed_title=feed_title,
                    ambiguous=ambiguous_nodes,
                    changes=episode_changes,
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
            _relpath = str(path.relative_to(ctx.corpus_root))
            changed.append(_relpath)
            if not ctx.dry_run:
                _write_atomic(path, payload)
                # WRITE-AHEAD, per episode. The first version wrote one ledger at the very end, so
                # a crash on episode 2 left episode 1 rewritten with NOTHING recording it — and a
                # later undo then reported a complete rollback having missed it entirely. The sha
                # is taken AFTER the write because it is what undo compares against to decide
                # whether anything else has since claimed the file.
                _sha = file_sha(path)
                append_ledger(
                    ctx.corpus_root,
                    [
                        RoleChange(
                            episode=_relpath,
                            node_id=nid,
                            name=nm,
                            role_before=before,
                            role_after=after,
                            route=route,
                            feed_title=feed_title,
                            file_sha_after=_sha,
                            node_index=nidx,
                            run_id=run_id,
                        )
                        for nidx, nid, nm, before, after, route in episode_changes
                    ],
                )

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
            + (
                f"; {len(suspect_demotions)} demotion(s) HAND-READ REQUIRED (the node had a "
                f"voice — an eponymous-show host would look like this)"
                if suspect_demotions
                else ""
            )
            + (
                f"; {len(ambiguous_nodes)} node(s) matched MORE THAN ONE roster entry and were "
                f"left untouched"
                if ambiguous_nodes
                else ""
            )
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
                "suspect_demotions": suspect_demotions,
                "ambiguous_nodes": ambiguous_nodes,
                "already_correct": already_correct,
                "no_roster": no_roster,
                "unmatched_roster_names": len(unmatched),
                "unmatched_sample": unmatched[:20],
                "unparsable": unparsable[:20],
            },
        )

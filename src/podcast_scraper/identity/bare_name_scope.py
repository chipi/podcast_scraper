"""Stop a bare first name becoming a global, followable person (#1685).

THE PROBLEM. `entity_node_id()` mints `person:{slugify(name)}` from whatever string the
extractor emitted. When three shows each say "Jensen" with no surname, three identical slugs
collide on ONE global node. Nobody decided to merge them; the merge is the absence of a
decision. And those ids are FOLLOWABLE — `POST /interests/{token}` accepts `person:` tokens, and
derived interests mint them straight from `entities_from_kg`, so a pooled token can enter a
user's profile with no click at all.

Measured on production (678 episodes, 14 feeds): 208 occurrences of 172 single-word person ids.
Hand-checked examples of what they actually are:

    person:jensen  HOLLOW  no insights, topics, quotes or shows; the corpus itself annotates it
                           "likely an unnamed/low-signal speaker", while `person:jensen-huang`
                           holds the real content
    person:sam     POOLED  carries a stated position about *Samuel Moyn* (mandatory retirement),
                           spans 2 feeds, while `person:sam-altman` exists separately
    person:alex    LOCAL   five positions from ONE episode, with `person:alex-mayassi` a
                           co-speaker in that same episode — and a different Alex in the other feed

THE RULE is the one production already trusts for insight mentions
(`gi/relational_edges.py::_resolve_span_to_index`, #1076 chunk 4-A): resolve a bare token against
the episode's own people by TOKEN SUBSET, and refuse when more than one candidate qualifies.

    exactly one candidate  ->  heal: use the full id (`alex` -> `person:alex-mayassi`)
    two or more            ->  scope: refuse to guess (the Donald/Eric shape)
    none                   ->  scope: nothing to resolve to

PRECEDENT for scoping: `graph_id_utils._scoped_speaker_person_id` already does exactly this for
the other class of under-specified name — a bare diarization label — "so anonymous voices don't
merge across episodes (#1b)". A bare first name is under-specified the same way and was never
covered. Scoped ids stay recognisable to `is_unresolved_speaker_placeholder`, which is consulted
in twelve modules including `entities_from_kg` itself, so they never surface as entity cards, as
follow targets, or as derived interests.

WHY SCOPE RATHER THAN DROP THE NODE. The bare ids already carry attribution — stated positions,
quotes, SPOKEN_BY edges. Dropping the node orphans that content. Scoping keeps it attached and
stays REVERSIBLE: a later canonicalisation pass can rewrite a scoped id upward, whereas a
never-minted record is gone.

WHY THIS IS A SEPARATE PASS, NOT A CHANGE TO `entity_node_id`. That function is pure and
per-name; this rule needs the episode's whole roster, which only exists after both layers have
extracted. There are also three mint families (`graph_id_utils.entity_node_id`,
`graph_id_utils.person_node_id`, `identity/slugify.person_id`), and the GI speaker path uses the
third — the very layer the production measurement proved matters. One pass over the finished
payloads covers all three, and is the SAME code the backfill migration runs, so the pipeline and
the migration cannot drift into disagreeing about who "Sam" is.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

logger = logging.getLogger(__name__)

#: Prefix marking an episode-local, unresolved person. Chosen to be filename-safe (the PKM export
#: derives file stems from ids: `person:jane` -> `person_jane`) and to be impossible to confuse
#: with a real slug.
SCOPED_PREFIX = "unresolved-"

_PERSON = "person:"

#: Edge keys that can hold a `person:` id. Used by BOTH :func:`person_ids_in` (which builds the
#: plan) and :func:`rewrite_ids` (which applies it). A key in one and not the other is precisely
#: the asymmetry that produced #1862 / #1868.
_EDGE_ENDPOINT_KEYS = ("source", "target", "from", "to")


def _slug_of(person_id: str) -> str:
    return person_id.split(":", 1)[1] if ":" in person_id else person_id


def is_bare_person_id(person_id: str) -> bool:
    """True when *person_id* is a single-token person slug — `person:jensen`, not `person:a-b`.

    Already-scoped ids are excluded so the pass is idempotent: running it twice must not scope a
    scoped id again. That matters because the backfill migration and the pipeline may both touch
    the same artifact.

    The explicit prefix checks are belt-and-braces TODAY — both `unresolved-` and `speaker-`
    contain a hyphen, so the single-token rule below already rejects them, and deleting the
    checks fails no test. They stay because they are what keeps idempotence true if the prefix
    ever changes to something hyphen-free, which the single-token rule would not survive.
    """
    if not person_id.startswith(_PERSON):
        return False
    slug = _slug_of(person_id)
    if not slug or slug.startswith(SCOPED_PREFIX) or slug.startswith("speaker-"):
        return False
    return "-" not in slug


def scoped_person_id(person_id: str, episode_id: str) -> str:
    """`person:jensen` + episode -> `person:unresolved-jensen-{episode}`.

    Keeps the NAME in the id on purpose: an operator reading a graph dump can still see who the
    reference was, which a pure opaque handle would destroy. Keeps the EPISODE in it because that
    is the scope in which the name means one person.
    """
    slug = _slug_of(person_id)
    ep = re.sub(r"[^a-z0-9]+", "-", str(episode_id).lower()).strip("-") or "unknown"
    return f"{_PERSON}{SCOPED_PREFIX}{slug}-{ep}"


def is_scoped_person_id(person_id: str) -> bool:
    """True when *person_id* is one of OUR placeholders — `person:unresolved-alex-ep42`.

    Distinct from :func:`is_bare_person_id`, which answers "should this be scoped?". This one
    answers "is this already a placeholder?", and the two are not complements: a real full name
    is neither.
    """
    return person_id.startswith(_PERSON) and _slug_of(person_id).startswith(SCOPED_PREFIX)


def resolve_candidates(bare_id: str, episode_person_ids: Iterable[str]) -> List[str]:
    """Full-name ids in this episode whose token set is a SUPERSET of the bare token.

    Token subset, not prefix, so a surname-only reference resolves too (`musk` is a token of
    `elon-musk`). Prefix matching would catch the first-name half and silently miss the other.

    PLACEHOLDERS ARE EXCLUDED, and that exclusion is the whole point of this function rather
    than an edge case. `person:unresolved-dario-ep-42` tokenises to
    ``{unresolved, dario, ep, 42}``, which is a superset of `dario` — so without this it
    qualifies as a "full name", and a placeholder becomes a resolution TARGET. Two ways that
    goes wrong, both observed in production data (2026-08-26 audit):

      * it REFUSES a correct heal. With `dario-amodei` alone there is one candidate and `dario`
        resolves to the real person. Add the placeholder and there are two, so the rule declines
        to guess and scopes instead — the real person is discarded by the presence of a
        placeholder that means nothing.
      * it heals ACROSS episodes. A placeholder minted in another episode still tokenises to
        contain `dario`, so `person:dario` here can be rewritten to
        `person:unresolved-dario-some-other-episode` — importing another episode's scope, which
        is precisely what episode-scoping exists to prevent.

    A placeholder is by construction a person we could NOT identify. Resolving a name to one
    resolves it to nothing while looking like a resolution.
    """
    bare = _slug_of(bare_id)
    return sorted(
        pid
        for pid in {str(p) for p in episode_person_ids}
        if pid.startswith(_PERSON)
        and not is_scoped_person_id(pid)
        and _slug_of(pid) != bare
        and bare in _slug_of(pid).split("-")
    )


def plan_bare_name_ids(
    episode_person_ids: Iterable[str],
    episode_id: str,
    *,
    heal: bool = True,
    candidate_ids: Iterable[str],
) -> Dict[str, str]:
    """``{old_id: new_id}`` for every bare person id in this episode. Pure; no I/O.

    Shared verbatim by the pipeline pass and the backfill migration — one implementation, so the
    two cannot produce different verdicts for the same episode.

    ``candidate_ids`` — REQUIRED — is what a bare name may be healed INTO; see
    :func:`person_node_ids_in`. The roster and the candidate pool answer different questions and
    the second must stay narrower than the first.

    ``heal=False`` scopes EVERYTHING, including the resolvable ones. That is the strictly safer
    setting and it exists because the two branches carry asymmetric risk: a wrong scoping is
    cheap and reversible, while a wrong heal writes a real person's id onto someone else's
    content — building a new pooled node on a high-traffic id, which is worse than the problem
    being fixed. The measurement can prove a bare name is RESOLVABLE; it cannot prove the
    resolution is CORRECT.
    """
    ids = {str(p) for p in episode_person_ids}
    # The pool a bare name may be healed INTO. REQUIRED, with no default: a default would be the
    # exact behaviour this parameter exists to remove, and a caller that forgot it would be
    # silently unprotected. This module's history is three attempts at one asymmetry — the next
    # one does not get to be built into the signature.
    pool = {str(p) for p in candidate_ids}
    mapping: Dict[str, str] = {}
    for bare in sorted(i for i in ids if is_bare_person_id(i)):
        candidates = resolve_candidates(bare, pool)
        if heal and len(candidates) == 1:
            mapping[bare] = candidates[0]
        else:
            mapping[bare] = scoped_person_id(bare, episode_id)
    return mapping


def rewrite_ids(payload: Mapping, id_map: Mapping[str, str]) -> Tuple[dict, int]:
    """Apply *id_map* to a GI/KG artifact in a copy. Returns ``(new_payload, changes)``.

    MERGES nodes rather than relabelling them. Healing `person:sam` in an episode that already
    contains `person:sam-altman` would otherwise produce two nodes sharing one id — a corrupt
    graph, and the trap that makes a naive rewrite unsafe. Edge endpoints and `speaker_id`
    properties are rewritten too, following the pattern `migrations/gil_kg_identity_migrations`
    established for the `speaker:` -> `person:` rename.
    """
    if not id_map or not isinstance(payload, Mapping):
        return (dict(payload) if isinstance(payload, Mapping) else {}), 0

    out = dict(payload)
    changes = 0

    nodes = payload.get("nodes")
    if isinstance(nodes, list):
        merged: Dict[str, dict] = {}
        order: List[str] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            nid = node.get("id")
            new_id = id_map.get(nid, nid) if isinstance(nid, str) else nid
            if isinstance(new_id, str) and new_id != nid:
                changes += 1
            node = {**node, "id": new_id}

            # Quote nodes carry `properties.speaker_id`, which is where ALL 23 production
            # coexistence cases lived. Rewriting the node id without this leaves the quote
            # attributed to an id that no longer exists — the same dangling reference one level
            # down. Precedent: `migrations/gil_kg_identity_migrations.py:49-57` (speaker: ->
            # person:).
            #
            # BEFORE the merge branch, deliberately. Placing it after meant a duplicate node took
            # `continue` and folded its properties into the survivor with the speaker_id NEVER
            # rewritten — the bare id landing on the survivor untouched, breaking both
            # single-pass completeness and idempotence. That is the same read/write asymmetry
            # this whole change exists to remove, reintroduced through the merge path.
            sid = _speaker_id_of(node)
            if sid is not None and sid in id_map:
                node = {
                    **node,
                    "properties": {**(node.get("properties") or {}), "speaker_id": id_map[sid]},
                }
                changes += 1

            key = str(new_id)
            if key in merged:
                # Same id from two source nodes: keep the first, fold in any properties the
                # duplicate carries that the survivor lacks. Never emit two nodes with one id.
                # Both sides have already had their speaker_id rewritten above, so whichever
                # `setdefault` keeps is a rewritten value.
                existing_props = dict(merged[key].get("properties") or {})
                for k, v in (node.get("properties") or {}).items():
                    existing_props.setdefault(k, v)
                merged[key] = {**merged[key], "properties": existing_props}
                continue
            merged[key] = node
            order.append(key)
        out["nodes"] = [merged[k] for k in order]

    edges = payload.get("edges")
    if isinstance(edges, list):
        new_edges = []
        for edge in edges:
            if not isinstance(edge, dict):
                new_edges.append(edge)
                continue
            e = dict(edge)
            for end in _EDGE_ENDPOINT_KEYS:
                val = e.get(end)
                if isinstance(val, str) and val in id_map:
                    e[end] = id_map[val]
                    changes += 1
            props = e.get("properties")
            if isinstance(props, dict) and isinstance(props.get("speaker_id"), str):
                sid = props["speaker_id"]
                if sid in id_map:
                    e = {**e, "properties": {**props, "speaker_id": id_map[sid]}}
                    changes += 1
            new_edges.append(e)
        out["edges"] = new_edges

    return out, changes


def surface_name_of(payload: Mapping, person_id: str) -> str:
    """The `name` property of *person_id* in *payload* — the name as SPOKEN.

    Scoping only rewrites the id, so a scoped node keeps "Alex" in `name`. That is what a future
    enricher needs to work from: the slug is lossy (case, punctuation, diacritics), the surface
    name is not.
    """
    for node in (payload.get("nodes") if isinstance(payload, Mapping) else None) or []:
        if isinstance(node, dict) and node.get("id") == person_id:
            props = node.get("properties")
            if isinstance(props, dict):
                name = props.get("name") or props.get("label")
                if isinstance(name, str) and name.strip():
                    return name.strip()
    return _slug_of(person_id)


def unresolved_persons_in_episode(gi_payload: Mapping, kg_payload: Mapping) -> List[Dict[str, Any]]:
    """Every episode-scoped person in one episode, with what a resolver would need (#1685).

    THE FOUNDATION FOR A FUTURE ENRICHER. Scoping stops a bare name becoming a global followable
    person, but it does not answer WHO the person was — that needs episode context (transcript,
    other entities, outside knowledge) and is a separate, harder job. This is the work-list that
    job would start from, so the enricher calls a function instead of parsing id strings.

    Every field is DERIVED from the artifacts rather than stored on the node, for two reasons.
    The artifact schemas pin `additionalProperties: False` on person nodes and allow only
    `aliases / description / label / name / role`, so storing a `resolution_status` would mean
    extending two schemas and bumping both `schema_version` values — a contract change that
    should wait until it has a reader. And derived state cannot drift from the artifact it
    describes, which is the same reason #1686 derives its retry count from the run dirs instead
    of keeping a counter.

    `reason` distinguishes the two jobs an enricher faces, and they are not equally hard:

        ambiguous   the episode names two or more people who could be this reference, and they
                    are listed. Pick one — a bounded, checkable decision.
        no_candidate nobody in the episode has a matching surname. The answer is not in the
                    graph at all; the enricher must read the transcript or look outward.
    """
    roster = person_ids_in(gi_payload) | person_ids_in(kg_payload)
    # Candidates must come from the RESOLVED people only. A scoped id contains the bare token
    # inside itself (`person:unresolved-jensen-ep-1` has `jensen` among its tokens), so leaving
    # scoped ids in the pool makes every orphan match itself and report as `ambiguous` — which
    # is precisely what the first version of this did.
    resolved_roster = {p for p in roster if not _slug_of(p).startswith(SCOPED_PREFIX)}
    out: List[Dict[str, Any]] = []
    for pid in sorted(roster):
        slug = _slug_of(pid)
        if not slug.startswith(SCOPED_PREFIX):
            continue
        # `unresolved-{name}-{episode}` — recover the name by stripping the prefix, then the
        # episode suffix. The episode is re-read from the artifact rather than parsed out, so a
        # name containing a hyphen cannot confuse the split.
        body = slug[len(SCOPED_PREFIX) :]
        surface = surface_name_of(gi_payload, pid) or surface_name_of(kg_payload, pid)
        candidates = resolve_candidates(f"{_PERSON}{body.split('-')[0]}", resolved_roster)
        out.append(
            {
                "id": pid,
                "surface_name": surface,
                # >1 is genuinely ambiguous. Exactly ONE candidate here means the id was
                # scoped despite being resolvable — heal was off, or the migration and the
                # pipeline saw different rosters — so it is `resolvable`, not `ambiguous`.
                # Calling it ambiguous would send the enricher looking for a choice that does
                # not exist, which is the same class of error as the self-matching candidate
                # bug this function already had once.
                "reason": (
                    "ambiguous"
                    if len(candidates) > 1
                    else ("resolvable" if candidates else "no_candidate")
                ),
                "candidates": candidates,
            }
        )
    return out


def _speaker_id_of(container: Mapping) -> Optional[str]:
    """A `person:` id in this node's or edge's ``properties.speaker_id``, if any.

    Deliberately NOT filtered by node ``type``, though the precedent
    ``migrations/gil_kg_identity_migrations.py:49-57`` filters on ``type == "Quote"``. Two
    reasons. `capability_audit`'s ``node_speaker`` bucket does not filter either, and a measure
    that looks in more places than the fix is exactly how this bug survived a first attempt. And
    a rewrite is plan-gated: it substitutes only ids already in the map, and only to their
    planned targets. Being permissive about WHERE therefore adds substitution LOCATIONS, never
    new old->new pairs — so there is no node type where rewriting a planned id to its planned
    target could misattribute anything.
    """
    props = container.get("properties") if isinstance(container, Mapping) else None
    if not isinstance(props, dict):
        return None
    sid = props.get("speaker_id")
    return sid if isinstance(sid, str) and sid.startswith(_PERSON) else None


def person_node_ids_in(payload: Mapping) -> Set[str]:
    """Only `person:` ids that have a NODE — the set an id may legitimately resolve TO.

    Split out from :func:`person_ids_in` because the two answer different questions and
    conflating them is dangerous. The roster ("what must be scoped") should be wide: anything the
    rewriter can write. The CANDIDATE pool ("what may a bare name be healed into") must be narrow:
    an id with no node is a dangling reference, the least-validated string in the artifact, and
    healing is the one branch that writes a REAL person's id onto content with no cheap undo.

    `rewrite_bridges_m0007._graph_person_ids` already applies exactly this rule, for a REVERSIBLE
    substitution — "the set a bridge id is allowed to point at". The irreversible branch should
    not accept weaker evidence than the reversible one.
    """
    ids: Set[str] = set()
    nodes = payload.get("nodes") if isinstance(payload, Mapping) else None
    if not isinstance(nodes, list):
        return ids
    for node in nodes:
        if isinstance(node, dict):
            nid = node.get("id")
            if isinstance(nid, str) and nid.startswith(_PERSON):
                ids.add(nid)
    return ids


def person_ids_in(payload: Mapping) -> Set[str]:
    """Every `person:` id in a GI or KG artifact — everywhere :func:`rewrite_ids` can write.

    MUST STAY EXACTLY AS WIDE AS THE REWRITER. This builds the roster the scoping pass plans
    from; `rewrite_ids` applies the resulting map. A roster that is narrower leaves ids it cannot
    see unplanned and unrewritten, while the same person's id somewhere it CAN see gets scoped —
    and the artifact then holds both forms and contradicts itself about who that person is.

    Measured, not theorised. Reading only ``nodes[].id`` (the original) took production from 6
    such episodes to 23 during the #1685 backfill on 2026-08-27, and the follow-up measurement
    showed ALL 23 were quote nodes' ``properties.speaker_id`` — a location in the GI schema
    (``$defs/quote_node``) that neither this function nor `rewrite_ids` reached. See #1868.

    Four places, mirroring `rewrite_ids` exactly: node ids, node ``properties.speaker_id``, edge
    endpoints, edge ``properties.speaker_id``.
    """
    ids: Set[str] = set(person_node_ids_in(payload))
    if not isinstance(payload, Mapping):
        return ids

    for node in payload.get("nodes") or []:
        if isinstance(node, dict):
            sid = _speaker_id_of(node)
            if sid:
                ids.add(sid)

    for edge in payload.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        for end in _EDGE_ENDPOINT_KEYS:
            val = edge.get(end)
            if isinstance(val, str) and val.startswith(_PERSON):
                ids.add(val)
        sid = _speaker_id_of(edge)
        if sid:
            ids.add(sid)
    return ids

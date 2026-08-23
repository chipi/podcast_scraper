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
from typing import Any, Dict, Iterable, List, Mapping, Set, Tuple

logger = logging.getLogger(__name__)

#: Prefix marking an episode-local, unresolved person. Chosen to be filename-safe (the PKM export
#: derives file stems from ids: `person:jane` -> `person_jane`) and to be impossible to confuse
#: with a real slug.
SCOPED_PREFIX = "unresolved-"

_PERSON = "person:"


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


def resolve_candidates(bare_id: str, episode_person_ids: Iterable[str]) -> List[str]:
    """Full-name ids in this episode whose token set is a SUPERSET of the bare token.

    Token subset, not prefix, so a surname-only reference resolves too (`musk` is a token of
    `elon-musk`). Prefix matching would catch the first-name half and silently miss the other.
    """
    bare = _slug_of(bare_id)
    return sorted(
        pid
        for pid in {str(p) for p in episode_person_ids}
        if pid.startswith(_PERSON) and _slug_of(pid) != bare and bare in _slug_of(pid).split("-")
    )


def plan_bare_name_ids(
    episode_person_ids: Iterable[str],
    episode_id: str,
    *,
    heal: bool = True,
) -> Dict[str, str]:
    """``{old_id: new_id}`` for every bare person id in this episode. Pure; no I/O.

    Shared verbatim by the pipeline pass and the backfill migration — one implementation, so the
    two cannot produce different verdicts for the same episode.

    ``heal=False`` scopes EVERYTHING, including the resolvable ones. That is the strictly safer
    setting and it exists because the two branches carry asymmetric risk: a wrong scoping is
    cheap and reversible, while a wrong heal writes a real person's id onto someone else's
    content — building a new pooled node on a high-traffic id, which is worse than the problem
    being fixed. The measurement can prove a bare name is RESOLVABLE; it cannot prove the
    resolution is CORRECT.
    """
    ids = {str(p) for p in episode_person_ids}
    mapping: Dict[str, str] = {}
    for bare in sorted(i for i in ids if is_bare_person_id(i)):
        candidates = resolve_candidates(bare, ids)
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
            key = str(new_id)
            if key in merged:
                # Same id from two source nodes: keep the first, fold in any properties the
                # duplicate carries that the survivor lacks. Never emit two nodes with one id.
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
            for end in ("source", "target", "from", "to"):
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


def person_ids_in(payload: Mapping) -> Set[str]:
    """Every `person:` node id in a GI or KG artifact."""
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

"""One human, one id, within one episode (#2056).

THE DEFECT, measured on production episode ``substack:post:189936942`` ("Every Agent Needs a Box
— Aaron Levie, Box", Latent Space)::

    person:aaron-levy    "Aaron Levy"    <- 65 x SPOKEN_BY        (diarization roster)
    person:aaron-levie   "Aaron Levie"   <-  2 x MENTIONS_PERSON  (entity extraction)

Two ids for one human in one episode, from two minting paths that never compare notes. The roster
heard the name spoken and ASR wrote "Levy"; the entity extractor read "Levie" off the text.
Neither layer is wrong about what it saw — they simply never met.

WHY THE EXISTING RESOLVER DOES NOT COVER THIS. ``kg.entity_clusters._are_xep_variants`` is the
CROSS-episode test. It runs over ``collect_entity_candidates``, which aggregates the whole corpus,
and is gated by ``same_show_required``. Nothing asks it about two ids inside a single episode.
That is why #2056 could correctly report "the resolver already matches every reported pair" while
duplicates kept reaching the product: the matcher was never the problem, nothing invoked it here.

THE ASYMMETRY THAT DRIVES EVERY RULE BELOW. A false SPLIT is cosmetic — two nodes for one person,
visible clutter. A false MERGE is corrupt data: one person's statements attributed to another,
propagating into quotes, roles and search, with no cheap undo. **Every rule here fails toward
leaving two nodes.** Specifically:

  * **Two ids that BOTH carry ``SPOKEN_BY`` are never merged.** The roster heard two distinct
    voices and named them; collapsing those reassigns real quotes to the wrong human. This is the
    single worst outcome the pass can produce and it is refused unconditionally, even when the
    names are near-identical.
  * **Episode-scoped placeholders are left alone.** ``person:unresolved-<name>-<episode>`` belongs
    to :mod:`identity.bare_name_scope`, which heals them on its own evidence. Two passes rewriting
    the same ids on different rules is how layers drift into disagreeing about who a person is.
  * **The plan never chains.** ``A -> B`` together with ``B -> C`` would leave a dangling id after
    one rewrite; a merged id is never itself a merge target.

MEASURED REACH: **21 of 287 production artifacts (7.3%)** carry such a duplicate, plus the
reported production case above, which is not in that sample. Every one was read individually and
every one is a genuine variant of one human — ``Bernard Leong``/``Bernard Leung``,
``Andrej``/``Andrei Karpathy``, ``Stewart``/``Stuart Brand``, ``Teresa``/``Theresa Bejan``,
``Steve Brusatte``/``Steve Broussatti``.

(It was 24 before the one-token rule landed in :mod:`kg.entity_clusters`. The three it costs here
include ``Aravind Srinivas``/``Arvind Surivas``, which drifts in both tokens — the same trade
recorded there: a false split is visible clutter, a false merge reassigns one human's statements
to another. Re-measure this number when that rule changes; do not copy it forward.)

This is deliberately NOT the general duplicate-person problem. That one needs matcher PRECISION
work — over the same 287 artifacts the cross-episode matcher produces ``Albert Einstein`` ==
``Robert Jensen``, ``Alex Bregman`` == ``Lex Friedman`` and ``Charles I`` == ``Charles II`` — and
is tracked separately. What makes THIS pass safe is not a better matcher but a piece of evidence
the corpus-wide pass does not have: who actually spoke in this episode.

The returned map is applied with :func:`identity.bare_name_scope.rewrite_ids`, which merges nodes
rather than relabelling them and rewrites edge endpoints and quote ``speaker_id`` properties.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Set

from ..kg.entity_clusters import _are_xep_variants
from ..kg.filters import _clean_entity_name
from .bare_name_scope import is_scoped_person_id

#: Edges whose target is a person the roster HEARD. Direct evidence of presence, and the reason
#: this side wins a merge: a mention is someone talking ABOUT a person, a quote is the person.
_SPEAKER_EDGES = frozenset({"SPOKEN_BY"})


def _person_names(payload: Mapping[str, Any]) -> Dict[str, str]:
    """``{person_id: display name}`` for every Person node in *payload*."""
    out: Dict[str, str] = {}
    if not isinstance(payload, Mapping):
        return out
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return out
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        if str(node.get("type") or "").strip().lower() != "person":
            continue
        pid = str(node.get("id") or "")
        props = node.get("properties")
        name = str((props or {}).get("name") or "") if isinstance(props, Mapping) else ""
        if pid and name:
            out.setdefault(pid, name)
    return out


def _speaker_ids(payload: Mapping[str, Any]) -> Set[str]:
    """Person ids the roster HEARD in this payload — the targets of a ``SPOKEN_BY`` edge.

    ``SPOKEN_BY`` points AT the person (``quote -> person``, verified against production
    artifacts), so the endpoint of interest is ``to``; ``from`` is read as a fallback so an
    inverted writer cannot silently turn this pass into a no-op.
    """
    spoke: Set[str] = set()
    if not isinstance(payload, Mapping):
        return spoke
    edges = payload.get("edges")
    if not isinstance(edges, list):
        return spoke
    for edge in edges:
        if not isinstance(edge, Mapping):
            continue
        if str(edge.get("type") or "") not in _SPEAKER_EDGES:
            continue
        for key in ("to", "from"):
            target = str(edge.get(key) or "")
            if target.startswith("person:"):
                spoke.add(target)
    return spoke


def _mergeable(pid: str) -> bool:
    """Ids this pass is allowed to touch at all."""
    if not pid.startswith("person:"):
        return False
    # Episode-scoped placeholders belong to `bare_name_scope` — see the module docstring.
    return not is_scoped_person_id(pid)


def plan_intra_episode_merges(
    gi_payload: Mapping[str, Any], kg_payload: Mapping[str, Any]
) -> Dict[str, str]:
    """``{loser_id: winner_id}`` for duplicate people inside ONE episode.

    Both payloads are read together because they are written together and a duplicate routinely
    straddles them (the production case has the speaker in GI and the mention in KG).

    A merge requires **exactly one** of the two ids to carry ``SPOKEN_BY``, and that id wins.

    That single rule is the whole justification for this pass. Its claim to be safer than the
    cross-episode matcher rests entirely on having evidence the matcher does not: the roster heard
    this person speak in THIS episode. Remove that evidence and nothing is left but the name
    similarity — which is measurably not good enough. Over 287 production artifacts the same
    matcher accepts ``Albert Einstein`` == ``Robert Jensen``, ``Alex Bregman`` ==
    ``Lex Friedman``, ``Charles I`` == ``Charles II`` and ``Albert Einstein`` ==
    ``Bert Vogelstein``. Merging on that alone would corrupt data to tidy a display.

    So two ids that were only ever MENTIONED are left as two nodes even when the names look
    identical. That is a deliberate false split: it is visible, harmless, and recoverable, and
    fixing it properly needs matcher precision work, tracked separately.

    Returns an empty map when there is nothing safe to do, which is the common case.
    """
    names: Dict[str, str] = {}
    for payload in (gi_payload, kg_payload):
        for pid, name in _person_names(payload).items():
            names.setdefault(pid, name)

    spoke: Set[str] = set()
    for payload in (gi_payload, kg_payload):
        spoke |= _speaker_ids(payload)

    candidates = sorted(pid for pid in names if _mergeable(pid))
    plan: Dict[str, str] = {}
    merged_away: Set[str] = set()

    for i, a in enumerate(candidates):
        if a in merged_away:
            continue
        for b in candidates[i + 1 :]:
            if b in merged_away or a in merged_away:
                continue
            name_a, name_b = names.get(a, ""), names.get(b, "")
            if not name_a or not name_b or name_a == name_b:
                # A nameless side is not evidence of anything — skip it. IDENTICAL names DO fall
                # through and merge (when exactly one of them spoke): two ids spelling one human
                # the same way is the clearest case this pass has, not one to be cautious about.
                # An earlier comment here claimed the opposite; the code always did this.
                # Two EMPTY names reach `_are_xep_variants("", "")`, which is False — verified,
                # since the guard above lets that pair through.
                if name_a != name_b:
                    continue
            if not _are_xep_variants(name_a, name_b, "person"):
                continue
            a_spoke, b_spoke = a in spoke, b in spoke
            if a_spoke and b_spoke:
                # TWO VOICES. The roster heard both and named both. Never merge — see the module
                # docstring; this is the outcome strictly worse than the duplicate.
                continue
            if not a_spoke and not b_spoke:
                # NO EPISODE-LOCAL EVIDENCE. Both were only mentioned, so all we have is the
                # name similarity this pass exists precisely not to trust on its own.
                continue
            winner, loser = (a, b) if a_spoke else (b, a)
            # Never chain: a winner that is already a loser elsewhere, or a loser already
            # recorded, would leave a dangling id after one rewrite pass.
            if loser in plan or winner in plan:
                continue
            plan[loser] = winner
            merged_away.add(loser)

    return plan


def _episode_prose(payload: Mapping[str, Any]) -> str:
    """Title + description off the Episode node — the human-written text for this episode."""
    if not isinstance(payload, Mapping):
        return ""
    parts = []
    nodes = payload.get("nodes")
    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            if str(node.get("type") or "").strip().lower() != "episode":
                continue
            props = node.get("properties")
            if isinstance(props, Mapping):
                parts.append(str(props.get("title") or ""))
                parts.append(str(props.get("description") or ""))
    return " ".join(p for p in parts if p)


def plan_display_names(
    gi_payload: Mapping[str, Any],
    kg_payload: Mapping[str, Any],
    id_plan: Mapping[str, str],
    *,
    episode_text: str = "",
) -> Dict[str, str]:
    """``{surviving_id: display name}`` — the ONE name the survivor shows in every layer.

    A name is emitted for **every** merge whose two sides spell the name differently, not only
    for the ones the prose settles. That is not tidiness; it is the whole correctness property,
    and an earlier version of this function got it wrong in a way the tests could not see.

    WHY EMITTING NOTHING IS NOT "LEAVE IT ALONE". :func:`bare_name_scope.rewrite_ids` keeps the
    properties of whichever node comes FIRST in that payload's node list — not the winner's. GI
    and KG order their nodes independently (``kg/pipeline`` appends the roster's speakers AFTER
    the extracted entities, so in KG the merged-away node is usually first). So "emit nothing"
    does not mean the survivor keeps one name: it means each artifact keeps a different one.

    Measured over the production sample, before this was fixed: of 11 real intra-episode merges,
    **8 left kg.json and gi.json displaying different names for the same person id** — the id was
    united and the label was not. ``person:stewart-brand`` read ``Stuart Brand`` on the card
    (``app_kg_index`` reads KG) and ``Stewart Brand`` on its quotes, while the episode title says
    Stewart. The merge that was supposed to end one-human-two-names produced one-id-two-names.

    WHICH SPELLING WINS IS DECIDED BY THE FEED, not by which layer produced it. An earlier version
    assumed the speaker side was systematically worse — reasoning that its name came through ASR —
    and generalised that from two examples. Measuring over 287 production artifacts showed it is
    wrong more often than right: the speaker side holds the CORRECT spelling for
    ``Andrej Karpathy``, ``Stewart Brand``, ``Teresa Bejan`` and ``Steve Brusatte``, and the wrong
    one for ``Bernard Leong``. Neither side is reliably better, so provenance alone yields no rule.

    The episode's own title and description are human-written — not transcribed, not generated —
    which makes them authoritative for spelling where neither ASR nor an LLM is. So whichever of
    the two spellings appears in that prose wins, in EITHER direction.

    THE TIE-BREAK, when the prose names both spellings or neither: the winner's name — the side
    the roster actually heard speak. It is not better evidence, and it is not claimed to be; it is
    a *stated, order-independent* choice, which is the property that matters here. The cost is
    recorded: it displays ``CELESTIN NTAWIREMA`` where the mention side had ``Celestine
    Ntawirema``. That is the only all-caps person name in the corpus (10 nodes of 13,642, one
    human), so it is a single ugly label, not a class — and casing is not this function's job.
    """
    if not id_plan:
        return {}
    names: Dict[str, str] = {}
    for payload in (gi_payload, kg_payload):
        for pid, name in _person_names(payload).items():
            names.setdefault(pid, name)

    # An empty prose string is NOT a reason to return early: the tie-break below still has to run,
    # or the survivor's label goes back to being whatever node happened to be first.
    prose = _clean_entity_name(
        " ".join(
            p for p in (episode_text, _episode_prose(gi_payload), _episode_prose(kg_payload)) if p
        )
    )

    out: Dict[str, str] = {}
    for loser, winner in id_plan.items():
        loser_name, winner_name = names.get(loser, ""), names.get(winner, "")
        if loser_name == winner_name:
            continue  # nothing to decide; both layers already agree
        if not winner_name:
            # A survivor with no name of its own takes the only name there is.
            if loser_name:
                out[winner] = loser_name
            continue
        if not loser_name:
            out[winner] = winner_name
            continue
        # `bool(name) and ...` matters: `_clean_entity_name("")` is "", and "" is `in` every
        # string — an empty name would otherwise read as PRESENT in the prose.
        loser_in = bool(prose) and _clean_entity_name(loser_name) in prose
        winner_in = bool(prose) and _clean_entity_name(winner_name) in prose
        out[winner] = loser_name if (loser_in and not winner_in) else winner_name
    return out


def apply_display_names(payload: Mapping[str, Any], renames: Mapping[str, str]) -> Any:
    """Return a copy of *payload* with Person node names replaced per *renames*.

    Applied AFTER ``rewrite_ids``, so the ids it keys on are the surviving ones.
    """
    if not renames or not isinstance(payload, Mapping):
        return dict(payload) if isinstance(payload, Mapping) else {}
    out = dict(payload)
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return out
    new_nodes = []
    for node in nodes:
        if isinstance(node, Mapping) and str(node.get("id") or "") in renames:
            props = dict(node.get("properties") or {})
            props["name"] = renames[str(node.get("id"))]
            if "label" in props:
                props["label"] = renames[str(node.get("id"))]
            node = {**node, "properties": props}
        new_nodes.append(node)
    out["nodes"] = new_nodes
    return out


def merge_targets_in(plan: Mapping[str, str]) -> Iterable[str]:
    """The surviving ids in *plan* — useful for logging which node absorbed which."""
    return sorted(set(plan.values()))

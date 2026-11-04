"""The person name a surface shows: one spelling per joined person, from the ledger (RFC-131 S6a).

A joined person shows their group's ``display_name``, which the ledger records with its evidence
(ADR-166 decision 1). Anyone else shows the name their own episode gives them. With no valid ledger
there are no joins (ADR-166), so every name is the episode's own; a ledger without spellings (one
built before S6a) changes nothing either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

_NS = "identity_person_display"

#: Group spelling rules that choose none (records s6g, s6h): the card shows the canonical's name.
UNDECIDED = frozenset({"tie", "plurality", "publisher_tie"})


class PersonDisplay:
    """``shown(person_id, own_name)``: the spelling to show for one person id."""

    def __init__(self, spelling_of: Dict[str, str]) -> None:
        self._spelling_of = spelling_of

    def shown(self, person_id: Optional[str], own_name: Optional[str]) -> Optional[str]:
        """The group spelling of ``person_id``, or ``own_name`` when the id has none."""
        if not own_name or not isinstance(person_id, str):
            return own_name
        return self._spelling_of.get(person_id.strip(), own_name)

    def name_of(self, person_id: Optional[str]) -> Optional[str]:
        """The group spelling of a joined id, or None (RFC-131 S6b)."""
        if not isinstance(person_id, str):
            return None
        return self._spelling_of.get(person_id.strip())

    def __len__(self) -> int:
        return len(self._spelling_of)


def _build(root: Path) -> PersonDisplay:
    from .ledger import read_ledger

    ledger, _state = read_ledger(root)
    return display_from_ledger(ledger)


def display_from_ledger(ledger: Optional[Dict[str, Any]]) -> PersonDisplay:
    """The view a ledger gives (None: no joins, every episode's own name)."""
    spelling_of: Dict[str, str] = {}
    if ledger is not None:
        for cluster in (ledger.get("clusters_payload") or {}).get("clusters") or []:
            shown = cluster.get("display_name")
            canonical = str(cluster.get("canonical_id") or "")
            if not canonical.startswith("person:"):
                continue
            if not shown:
                # A tie (record s6g): the card shows the canonical's own name, and each member
                # keeps its episode's spelling.
                if (cluster.get("display_evidence") or {}).get("rule") in UNDECIDED and cluster.get(
                    "canonical_name"
                ):
                    spelling_of[canonical] = str(cluster["canonical_name"])
                continue
            for member in cluster.get("members") or []:
                if member.get("id"):
                    spelling_of[str(member["id"])] = str(shown)
            spelling_of[canonical] = str(shown)
    return PersonDisplay(spelling_of)


def person_display(root: Path | str) -> PersonDisplay:
    """The view for corpus ``root``, cached by the corpus token. The token includes the ledger file,
    so a rebuilt ledger is picked up without a restart."""
    from ..perf_cache import corpus_mtime, get_or_compute

    root = Path(root)
    view: PersonDisplay = get_or_compute(
        _NS, str(root.resolve()), corpus_mtime(root), lambda: _build(root)
    )
    return view


#: Corpus enrichers whose rows name two people: ``{enricher_id: list key}`` (RFC-131 S6d).
_PERSON_PAIR_LISTS = {"guest_coappearance": "pairs", "topic_consensus": "consensus"}
PERSON_PAIR_ENRICHERS = frozenset(_PERSON_PAIR_LISTS)


def spell_enrichment_people(enricher_id: str, data: Any, people: Optional[PersonDisplay]) -> Any:
    """An enrichment envelope's ``data`` with each named person spelled through ``people``.

    Covers the pair rows' ``person_a_name`` / ``person_b_name`` and a co-appearance community's
    ``community_label`` (its anchor member's name). Returns a copy when anything changes, so a
    cached envelope is never mutated; ``data`` itself otherwise.
    """
    list_key = _PERSON_PAIR_LISTS.get(enricher_id)
    if people is None or not len(people) or list_key is None or not isinstance(data, dict):
        return data
    rows = data.get(list_key)
    if not isinstance(rows, list):
        return data
    own_name: Dict[str, str] = {}
    spelled_rows = []
    for row in rows:
        if not isinstance(row, dict):
            spelled_rows.append(row)
            continue
        new = dict(row)
        for side in ("a", "b"):
            pid, name = row.get(f"person_{side}_id"), row.get(f"person_{side}_name")
            if isinstance(pid, str) and isinstance(name, str):
                own_name.setdefault(pid, name)
                new[f"person_{side}_name"] = people.shown(pid, name)
        spelled_rows.append(new)
    out = dict(data)
    out[list_key] = spelled_rows
    communities = data.get("communities")
    if isinstance(communities, list):
        out["communities"] = [_spelled_community(c, own_name, people) for c in communities]
    return out


def _spelled_community(community: Any, own_name: Dict[str, str], people: PersonDisplay) -> Any:
    """The community's label spelled through its anchor, a member whose own name it is. Several
    members can carry the name (one person's ids): it is spelled when they all spell it alike."""
    if not isinstance(community, dict) or not isinstance(community.get("community_label"), str):
        return community
    label = community["community_label"]
    anchors = [m for m in community.get("member_ids") or [] if own_name.get(m) == label]
    spellings = {people.shown(m, label) for m in anchors}
    if len(spellings) != 1:
        return community
    return {**community, "community_label": spellings.pop()}

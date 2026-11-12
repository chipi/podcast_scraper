"""The person name a surface shows, through the display seam (E04 split line, decision 3).

``person_display(root)`` is what every surface reads: a ``PersonDisplay`` whose ``shown(id, own)``
gives the spelling to show for one person id. Which spelling that is comes from the display
builder for the corpus. The built-in builder is the identity ledger's (``ledger_display``, RFC-131
S6a): a joined person shows their group's ``display_name``, anyone else the name their own episode
gives them. ``set_display_builder`` replaces it; ``plain_display`` is the builder with no joins,
where every name is the episode's own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

_NS = "identity_person_display"


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


#: Builds the view for one corpus root.
DisplayBuilder = Callable[[Path], PersonDisplay]

_builder: Optional[DisplayBuilder] = None


def plain_display(root: Path) -> PersonDisplay:
    """The builder with no joins: every surface shows each episode's own name."""
    return PersonDisplay({})


def set_display_builder(builder: Optional[DisplayBuilder]) -> None:
    """Replace the display builder (``None``: back to the built-in ledger builder).

    Set once at startup, before any request: views already cached under another builder are not
    served under this one (the builder is part of the cache key).
    """
    global _builder
    _builder = builder


def _current_builder() -> DisplayBuilder:
    """``set_display_builder``'s, else the read side's owner (the identity ledger's while it is
    built in, E04 S3), else ``plain_display``."""
    if _builder is not None:
        return _builder
    from .. import read_hooks

    owner = read_hooks.person_display_builder()
    return owner if owner is not None else plain_display


def person_display(root: Path | str) -> PersonDisplay:
    """The view for corpus ``root``, cached by the corpus token. The token includes the ledger file,
    so a rebuilt ledger is picked up without a restart."""
    from ..perf_cache import corpus_mtime, get_or_compute

    root = Path(root)
    builder = _current_builder()
    view: PersonDisplay = get_or_compute(
        _NS, (str(root.resolve()), builder), corpus_mtime(root), lambda: builder(root)
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

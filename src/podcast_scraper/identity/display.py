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


class PersonDisplay:
    """``shown(person_id, own_name)``: the spelling to show for one person id."""

    def __init__(self, spelling_of: Dict[str, str]) -> None:
        self._spelling_of = spelling_of

    def shown(self, person_id: Optional[str], own_name: Optional[str]) -> Optional[str]:
        if not own_name or not isinstance(person_id, str):
            return own_name
        return self._spelling_of.get(person_id.strip(), own_name)

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
            if not shown or not canonical.startswith("person:"):
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

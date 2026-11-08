"""The person display the identity ledger gives (RFC-131 S6a): the built-in display builder.

A joined person shows their group's ``display_name``, which the ledger records with its evidence
(ADR-166 decision 1). Anyone else shows the name their own episode gives them. With no valid ledger
there are no joins (ADR-166), so every name is the episode's own; a ledger without spellings (one
built before S6a) changes nothing either.

This is the ledger side of the display seam (``identity.display``): it leaves public with the
ledger in the core release (E04 split line, decision 3), and Common registers it from there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .display import PersonDisplay

#: Group spelling rules that choose none (records s6g, s6h): the card shows the canonical's name.
UNDECIDED = frozenset({"tie", "plurality", "publisher_tie"})


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


def build_from_ledger(root: Path) -> PersonDisplay:
    """The display builder for corpus ``root``: its ledger's view."""
    from .ledger import read_ledger

    ledger, _state = read_ledger(root)
    return display_from_ledger(ledger)

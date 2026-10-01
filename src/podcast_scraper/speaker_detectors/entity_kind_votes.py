"""What the corpus's own KG extraction says a name IS — person or organisation (#2220, #2219).

KG extraction labels every entity it emits Person / Organization / Object, in every episode, and
the speaker path never read that label. So a name the corpus overwhelmingly calls an organisation
could still become a speaker candidate, because no string rule recognises it. Measured on prod
(2,002 served episodes, 2026-10-01), organisations today's rules kept as host/guest:

    name                    Organization  Person  roster host/guest entries
    The Brazilian Report              23       0   40
    Americas Online                   24       0   31
    Apple                             40       0    1
    Supreme Court                     27       0    1
    World Bank                        25       0    2
    Carnegie India                    10       2    8

WHO VOTES. Only extraction nodes, i.e. ``role == "mentioned"``. A ``host`` / ``guest`` Person was
put there by the roster — it is the thing being judged, and letting it vote would let a wrong roster
confirm itself (``Andreessen Horowitz`` is Person/host ×53 by that route and Organization ×54 by
extraction).

ONE DIRECTION ONLY. A strong Person vote is NOT used to overrule a rule that rejects a name:
extraction calls show names people too — ``Turkey Book`` Person ×10, ``Machine Learning Street``
Person ×31 — so votes cannot tell them from ``Peter Attia`` (Person ×39). Below the threshold the
existing rules decide, unchanged.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

from ..identity.slugify import canonical_person_name

logger = logging.getLogger(__name__)

#: At least this many Organization votes — "Jonas" with one stray vote must not lose his voice.
ORG_MIN_VOTES = 3
#: And at least this many times the Person votes ("Carnegie India" 10 : 2 qualifies).
ORG_TO_PERSON_RATIO = 4

_VOTING_ROLE = "mentioned"
_ORG = "Organization"
_PERSON = "Person"


def kind_key(name: str) -> str:
    """The key a name votes and is judged under: canonical spelling, case-folded."""
    text = " ".join(str(name or "").split())
    return (canonical_person_name(text) or text).casefold()


@dataclass(frozen=True)
class KindVotes:
    """``{kind_key: (organization_votes, person_votes)}`` over a corpus's served KGs."""

    counts: Mapping[str, Tuple[int, int]] = field(default_factory=dict)

    def calls_organisation(self, name: str) -> bool:
        """True when extraction calls *name* an organisation decisively enough to act on."""
        org, person = self.counts.get(kind_key(name), (0, 0))
        return org >= ORG_MIN_VOTES and org >= ORG_TO_PERSON_RATIO * person


def votes_from_kg_payloads(payloads: Iterable[Mapping]) -> KindVotes:
    """Count extraction votes over KG artifacts (pure; no I/O)."""
    org: Dict[str, int] = {}
    person: Dict[str, int] = {}
    for payload in payloads:
        for node in (payload or {}).get("nodes") or []:
            if not isinstance(node, dict):
                continue
            node_type = node.get("type")
            if node_type not in (_ORG, _PERSON):
                continue
            props = node.get("properties") or {}
            if props.get("role") != _VOTING_ROLE:
                continue
            name = props.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            key = kind_key(name)
            bucket = org if node_type == _ORG else person
            bucket[key] = bucket.get(key, 0) + 1
    keys = set(org) | set(person)
    return KindVotes({k: (org.get(k, 0), person.get(k, 0)) for k in keys})


@lru_cache(maxsize=4)
def corpus_kind_votes(corpus_root: str) -> KindVotes:
    """Votes over the SERVED ``.kg.json`` copies under *corpus_root*; empty when there are none.

    Cached per process: one pipeline run judges many names against the same corpus, and a KG
    written during the run changes a name's tally by at most one vote.
    """
    from ..upgrade.corpus_selection import select_served_artifacts

    root = Path(corpus_root)
    if not root.is_dir():
        return KindVotes()
    served, _superseded = select_served_artifacts(root, ".kg.json")

    def _payloads() -> Iterable[Mapping]:
        for path in served:
            try:
                yield json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

    votes = votes_from_kg_payloads(_payloads())
    logger.info(
        "entity kind votes: %d name(s) from %d served KG file(s) under %s",
        len(votes.counts),
        len(served),
        root,
    )
    return votes


def votes_for_output_dir(output_dir: Optional[str]) -> Optional[KindVotes]:
    """Votes for the corpus a run/feed directory belongs to (``<corpus>/feeds/<slug>/...``).

    For writers that only know where they write, not the run config. ``None`` when the path is not
    inside a corpus layout — no evidence, no change.
    """
    if not output_dir:
        return None
    path = Path(output_dir).resolve()
    for parent in (path, *path.parents):
        if parent.parent.name == "feeds" and parent.parent.parent.name:
            try:
                return corpus_kind_votes(str(parent.parent.parent))
            except Exception as exc:  # noqa: BLE001 — unreadable votes are no votes
                logger.warning("entity kind votes unavailable (%s: %s)", type(exc).__name__, exc)
                return None
    return None


def votes_for_cfg(cfg: object) -> Optional[KindVotes]:
    """The corpus votes for a run's config, or ``None`` outside a corpus (no evidence)."""
    from ..workflow import run_index

    root = run_index.corpus_root_from_cfg(cfg)
    if not root:
        return None
    try:
        return corpus_kind_votes(str(Path(root).resolve()))
    except Exception as exc:  # noqa: BLE001 — a vote that cannot be read is no vote, never a crash
        logger.warning("entity kind votes unavailable (%s: %s)", type(exc).__name__, exc)
        return None

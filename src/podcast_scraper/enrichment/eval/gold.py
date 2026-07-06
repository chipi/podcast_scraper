"""Generic enricher gold — one block per enricher, keyed by ``enricher_id``.

The gold counterpart to ``manifest.writes``: where ``writes`` keys an
enricher's *output* file, ``expected_enrichment[<enricher_id>]`` keys its
*gold*. There is deliberately **no per-enricher field name** anywhere — a new
enricher's gold is authored under its own id, never by adding a bespoke
``expected_velocity`` / ``expected_perspectives`` key to a fixture schema.

Gold lives in two conventional places, both under the same key:

* **Episode-scope** enrichers: inside each episode's ground-truth JSON —
  ``ground_truth["expected_enrichment"][enricher_id]``.
* **Corpus-scope** enrichers: inside a single corpus-level gold JSON — the same
  ``{"expected_enrichment": {enricher_id: {...}}}`` shape.

This module only *reads/normalizes* gold; it never authors values.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

# The one key under which every enricher's gold is authored. Mirrors the role
# ``manifest.writes`` plays for outputs — the single generic slot, keyed by id.
EXPECTED_ENRICHMENT_KEY = "expected_enrichment"


def gold_for(source: Mapping[str, Any], enricher_id: str) -> dict[str, Any] | None:
    """Return one enricher's gold block from a single ground-truth / gold doc.

    Accepts either a full ground-truth doc (with an ``expected_enrichment``
    sub-dict) or an already-unwrapped ``expected_enrichment`` mapping — so
    callers don't have to know which level they hold. Returns ``None`` when the
    enricher has no authored gold (honest-absent, distinct from empty ``{}``).
    """
    block = source.get(EXPECTED_ENRICHMENT_KEY)
    if isinstance(block, Mapping):
        candidate = block.get(enricher_id)
        return dict(candidate) if isinstance(candidate, Mapping) else None
    # Already unwrapped: the mapping IS the expected_enrichment block.
    candidate = source.get(enricher_id)
    return dict(candidate) if isinstance(candidate, Mapping) else None


def collect_episode_gold(
    ground_truths: Iterable[Mapping[str, Any]],
    enricher_id: str,
    *,
    id_key: str = "episode_id",
) -> dict[str, dict[str, Any]]:
    """Map ``episode_id → gold block`` across per-episode ground-truth docs.

    For episode-scope enrichers whose gold is authored per episode. Episodes
    with no authored gold for this enricher are omitted. ``id_key`` names the
    field holding each doc's episode id (v3 ground-truth uses ``episode_id``).
    """
    out: dict[str, dict[str, Any]] = {}
    for gt in ground_truths:
        g = gold_for(gt, enricher_id)
        if g is None:
            continue
        eid = str(gt.get(id_key) or "")
        if eid:
            out[eid] = g
    return out


def all_gold_enricher_ids(source: Mapping[str, Any]) -> list[str]:
    """List enricher ids that have an authored gold block in ``source``.

    Drives coverage assertions ("every registered scorer has gold") without
    enumerating enricher names — read the ids present, don't hardcode them.
    """
    block = source.get(EXPECTED_ENRICHMENT_KEY)
    if isinstance(block, Mapping):
        return sorted(str(k) for k in block.keys())
    return []


__all__ = [
    "EXPECTED_ENRICHMENT_KEY",
    "all_gold_enricher_ids",
    "collect_episode_gold",
    "gold_for",
]

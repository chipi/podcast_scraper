"""Reconcile the model we REQUESTED against the model a provider actually SERVED, from a run's logs.

This automates the check that only a human reading the provider dashboard caught in the sonnet-4-6
incident: for a finished run, did every call use the model the arm/profile asked for, or did a
provider quietly serve something else?

It reads the JSON log lines a run already emits — ``llm_cost`` events (which now carry both
``model`` and ``served_model``, Guard 3) and ``llm_model_substitution`` events (the loud mismatch
flag, Guard 2) — and returns one row per (provider, requested, served) disagreement with a call
count. An empty result means the run is clean. Wire it into a post-run gate to fail a run whose
models drifted, so the next arm never lands on the scoreboard mislabeled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..providers.known_models import verify_served_model


@dataclass(frozen=True)
class ModelMismatch:
    """One (provider, requested→served) disagreement seen in a run's cost/telemetry stream."""

    provider: str
    requested_model: str
    served_model: str
    call_count: int
    stages: tuple

    def as_dict(self) -> Dict[str, Any]:
        """This mismatch as a plain dict (for the API / MCP surface)."""
        return {
            "provider": self.provider,
            "requested_model": self.requested_model,
            "served_model": self.served_model,
            "call_count": self.call_count,
            "stages": list(self.stages),
        }


def reconcile_events(events: Iterable[Dict[str, Any]]) -> List[ModelMismatch]:
    """Return the model mismatches in a stream of parsed telemetry events (empty == clean).

    Considers ``llm_cost`` events (compares ``served_model`` vs ``model`` via the same
    normalisation the live guard uses) and any ``llm_model_substitution`` events. Aggregated by
    (provider, requested, served).
    """
    agg: Dict[tuple, Dict[str, Any]] = {}

    def _record(provider: str, requested: str, served: str, stage: Optional[str]) -> None:
        key = (provider, requested, served)
        slot = agg.setdefault(key, {"count": 0, "stages": set()})
        slot["count"] += 1
        if stage:
            slot["stages"].add(stage)

    for ev in events:
        if not isinstance(ev, dict):
            continue
        etype = ev.get("event_type")
        if etype == "llm_model_substitution":
            _record(
                str(ev.get("provider", "")),
                str(ev.get("requested_model", "")),
                str(ev.get("served_model", "")),
                ev.get("stage"),
            )
        elif etype == "llm_cost":
            provider = str(ev.get("provider", ""))
            requested = str(ev.get("model", ""))
            served = ev.get("served_model")
            if not served:
                continue
            if verify_served_model(provider, requested, str(served)) is not None:
                _record(provider, requested, str(served), ev.get("stage"))

    return sorted(
        (
            ModelMismatch(
                provider=p,
                requested_model=req,
                served_model=srv,
                call_count=slot["count"],
                stages=tuple(sorted(slot["stages"])),
            )
            for (p, req, srv), slot in agg.items()
        ),
        key=lambda m: (-m.call_count, m.provider, m.requested_model),
    )


def _iter_json_lines(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        # Log lines are often "<timestamp> LEVEL logger: {json}"; grab the JSON object if present.
        brace = line.find("{")
        if brace == -1:
            continue
        try:
            obj = json.loads(line[brace:])
        except ValueError:
            continue
        if isinstance(obj, dict):
            yield obj


def reconcile_run_log(path: str | Path) -> List[ModelMismatch]:
    """Reconcile a run's ``run.log`` (or any file of JSON-bearing log lines)."""
    p = Path(path)
    if not p.is_file():
        return []
    return reconcile_events(_iter_json_lines(p))

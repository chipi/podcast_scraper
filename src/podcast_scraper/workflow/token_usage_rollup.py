"""Slice recorded token/cost telemetry any way — per request / operation / episode / run / model.

The counting-integrity layer over the ``llm_cost`` event stream. Each event is one LLM request/
response (Path A summarization/transcription + Path B gi/cleaning are disjoint, so one call → one
event). This rollup:

* **de-dups by ``request_id``** — a doubly-written log line for the same provider request counts
  ONCE, so a flushed-twice log or a re-read file can never inflate totals. Events with no request_id
  (a provider that returns no id) each count once (nothing to collapse them onto).
* **aggregates by any dimension set** — ``group_by=("provider", "model")``, ``("episode_id",)``,
  ``("operation",)``, ``("run_id", "model")`` … so the operator can attribute spend to a model, a
  function, an episode, or a run and compare.

Totals carry the full token breakdown (input / output / cached / cache-write) AND the summed derived
cost, so nothing is lost between "how many tokens" and "how much did it cost". No over/under count:
the numbers are exactly the recorded events, de-duplicated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class UsageTotals:
    """Summed token counts + derived cost for one slice of the telemetry."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    estimated_cost_usd: float = 0.0
    # Calls whose response tripped a response-shape guardrail (paid-but-rejected spend, ADR-100).
    guardrail_calls: int = 0

    def add(self, ev: Dict[str, Any]) -> None:
        """Fold one ``llm_cost`` event's tokens + cost into this slice's running totals."""
        self.calls += 1
        self.input_tokens += int(ev.get("prompt_tokens") or 0)
        self.output_tokens += int(ev.get("completion_tokens") or 0)
        self.cached_input_tokens += int(ev.get("cached_input_tokens") or 0)
        self.cache_write_tokens += int(ev.get("cache_write_tokens") or 0)
        self.estimated_cost_usd = round(
            self.estimated_cost_usd + float(ev.get("estimated_cost_usd") or 0.0), 6
        )
        if ev.get("triggered_guardrail"):
            self.guardrail_calls += 1

    def as_dict(self) -> Dict[str, Any]:
        """These totals as a plain dict (JSON-serialisable for the API / MCP)."""
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "guardrail_calls": self.guardrail_calls,
        }


@dataclass
class RollupResult:
    """A grouped rollup: the grand total plus per-group totals keyed by the group_by values."""

    group_by: Tuple[str, ...]
    total: UsageTotals = field(default_factory=UsageTotals)
    groups: Dict[Tuple[Any, ...], UsageTotals] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """The grouped rollup as a plain dict: grand total + per-group rows sorted by cost."""
        return {
            "group_by": list(self.group_by),
            "total": self.total.as_dict(),
            "groups": [
                {**dict(zip(self.group_by, key)), **totals.as_dict()}
                for key, totals in sorted(
                    self.groups.items(),
                    key=lambda kv: kv[1].estimated_cost_usd,
                    reverse=True,
                )
            ],
        }


def _llm_cost_events(events: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for ev in events:
        if isinstance(ev, dict) and ev.get("event_type") == "llm_cost":
            yield ev


def rollup_events(
    events: Iterable[Dict[str, Any]],
    group_by: Tuple[str, ...] = ("provider", "model"),
) -> RollupResult:
    """Aggregate ``llm_cost`` events by ``group_by`` dimensions, de-duplicated by ``request_id``.

    Group keys read missing dimensions as the string ``"(none)"`` so a slice never silently drops
    events whose dimension is null (e.g. per-episode rollup of a call made outside any episode).
    """
    result = RollupResult(group_by=tuple(group_by))
    seen_request_ids: set = set()
    for ev in _llm_cost_events(events):
        rid = ev.get("request_id")
        if rid:
            if rid in seen_request_ids:
                continue  # a doubly-logged line for the same provider request — count once
            seen_request_ids.add(rid)
        result.total.add(ev)
        key = tuple(str(ev.get(dim) if ev.get(dim) is not None else "(none)") for dim in group_by)
        result.groups.setdefault(key, UsageTotals()).add(ev)
    return result


def _iter_json_lines(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        brace = line.find("{")
        if brace == -1:
            continue
        try:
            obj = json.loads(line[brace:])
        except ValueError:
            continue
        if isinstance(obj, dict):
            yield obj


def rollup_run_log(
    path: str | Path,
    group_by: Tuple[str, ...] = ("provider", "model"),
) -> RollupResult:
    """Roll up a run's ``run.log`` (or any file of JSON-bearing log lines) by ``group_by``."""
    p = Path(path)
    if not p.is_file():
        return RollupResult(group_by=tuple(group_by))
    return rollup_events(_iter_json_lines(p), group_by=group_by)


def slice_dimensions() -> List[str]:
    """The dimensions a caller may group by (for API/MCP surfacing)."""
    return [
        "provider",
        "model",
        "served_model",
        "operation",
        "stage",
        "episode_id",
        "run_id",
        "feed_id",
    ]

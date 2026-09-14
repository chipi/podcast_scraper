"""Enrichment events must reach the canonical telemetry stream, not only run.jsonl (#2071).

Measured on prod 2026-09-14: a 24h VictoriaLogs window held ZERO enrichment lines. Enrichment
appended to ``enrichments/run.jsonl`` and stopped there, while the pipeline emitted via
``emit_event(sink="log")`` -> stdout -> Alloy -> VictoriaLogs. So enrichment was observable on
demand through ``/api/enrichment/*`` but invisible to alerting, and a failing enricher left no
trail off-box: org_web's NameError existed solely in run.jsonl while the stack showed a bare 202.

These lock BOTH halves — the file consumers keep working, and the event is also shipped.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from podcast_scraper.enrichment.executor import EnrichmentExecutor


def _append(tmp_path: Path, payload: dict) -> Path:
    """Drive the single chokepoint every emission site funnels through."""
    target = tmp_path / "run.jsonl"
    EnrichmentExecutor._safe_append_event(object.__new__(EnrichmentExecutor), target, payload)
    return target


_PAYLOAD = {
    "event_type": "enrichment.enricher.completed",
    "ts": "2026-09-14T11:24:00Z",
    "status": "failed",
    "run_id": "r-1",
    "enricher_id": "org_web",
    "error": "name 'qid' is not defined",
    "error_class": "NameError",
}


def test_event_is_still_appended_to_run_jsonl(tmp_path: Path) -> None:
    """The existing consumers (/api/enrichment/*, run-summary) must be untouched."""
    target = _append(tmp_path, dict(_PAYLOAD))
    rows = [json.loads(line) for line in target.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["enricher_id"] == "org_web"
    assert rows[0]["error_class"] == "NameError"


def test_event_is_also_shipped_to_the_telemetry_stream(tmp_path: Path, caplog) -> None:
    """...and the same event reaches stdout, which is what Alloy ships to VictoriaLogs."""
    with caplog.at_level(logging.INFO, logger="podcast_scraper.events"):
        _append(tmp_path, dict(_PAYLOAD))

    emitted = [
        json.loads(r.getMessage())
        for r in caplog.records
        if r.name == "podcast_scraper.events" and r.getMessage().startswith("{")
    ]
    assert emitted, "no canonical event emitted — enrichment is invisible off-box again"
    ev = emitted[0]
    assert ev["event_type"] == "enrichment.enricher.completed"
    # The failure detail is the whole point: it must be debuggable WITHOUT box access.
    assert ev["error_class"] == "NameError"
    assert ev["enricher_id"] == "org_web"
    # Same timestamp as the file row, so the two can be correlated.
    assert ev["ts"] == _PAYLOAD["ts"]


def test_emission_failure_never_breaks_a_run(tmp_path: Path, monkeypatch) -> None:
    """Telemetry is best-effort by contract — a broken sink must not fail enrichment."""
    import podcast_scraper.enrichment.executor as executor_mod

    def _boom(*_a, **_k):
        raise RuntimeError("sink down")

    monkeypatch.setattr(executor_mod, "emit_event", _boom)
    # Must not raise.
    _append(tmp_path, dict(_PAYLOAD))

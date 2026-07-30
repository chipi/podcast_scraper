"""advisor #10: the pipeline_stage Grafana dashboard must be valid JSON and use ONE query dialect
(LogQL, like the house llm-cost dashboard) — no VictoriaLogs-only `field:*` syntax that would make
half the panels dead on a Loki datasource."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_DASH = Path("config/grafana/dashboards/common/grafana-dashboard-pipeline-stage.json")


@pytest.mark.unit
def test_dashboard_is_valid_json_with_panels():
    data = json.loads(_DASH.read_text())
    assert data["uid"] == "podcast-scraper-pipeline-stage"
    assert len(data["panels"]) >= 6


@pytest.mark.unit
def test_no_logsql_only_pipe_field_syntax():
    # `| quality_flags:*` / `| stage="asr"` LogsQL pipe-filters are invalid LogQL. After the fix all
    # panels use LogQL (metric queries + `|=` line filters).
    text = _DASH.read_text()
    assert "quality_flags:*" not in text
    for panel in json.loads(text)["panels"]:
        for target in panel.get("targets", []):
            expr = target["expr"]
            # every query selects the event stream and (if it drills) uses a `|=` line filter
            assert "pipeline_stage" in expr

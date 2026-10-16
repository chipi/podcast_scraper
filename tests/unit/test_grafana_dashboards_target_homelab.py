"""Every dashboard ``make obs-sync`` uploads must be readable by the homelab Grafana.

``scripts/obs/grafana_sync.py`` uploads every file under ``config/grafana/dashboards/<tenant>/``.
The homelab Grafana has VictoriaLogs (LogsQL), VictoriaMetrics (PromQL) and VictoriaTraces — no
Loki. Seven dashboards in ``common/`` were still Grafana Cloud exports (Loki LogQL, ``${DS_LOKI}``
/ ``${DS_PROMETHEUS}`` import placeholders) and rendered nothing until 2026-10-14; this keeps a new
export from slipping in the same way.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
DASHBOARDS = sorted((ROOT / "config" / "grafana" / "dashboards").rglob("*.json"))
# umami-pg: the homelab's read-only Umami datasource (agentic-ai-homelab, provisioning/datasources/
# umami-postgres.yml), used by podcast-player/beta-analytics-umami.json.
HOMELAB_UIDS = {"victorialogs", "victoriametrics", "victoriatraces-jaeger", "umami-pg"}
_LOGQL = re.compile(r"\|\s*json\b|\|\s*unwrap\b|\b(count|sum|rate)_over_time\(\s*\{")


def _walk(node: Any) -> Iterator[dict]:
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_there_are_dashboards_to_check() -> None:
    assert len(DASHBOARDS) >= 8


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_a_dashboard_targets_the_homelab_datasources(path: Path) -> None:
    raw = path.read_text(encoding="utf-8")
    assert "${DS_" not in raw, "an import placeholder Grafana's API upload never resolves"
    for node in _walk(json.loads(raw)):
        ds = node.get("datasource")
        if isinstance(ds, dict) and ds.get("uid") and not str(ds["uid"]).startswith("$"):
            assert ds["uid"] in HOMELAB_UIDS, f"unknown datasource uid {ds['uid']!r}"
        expr = node.get("expr")
        if isinstance(expr, str):
            assert not _LOGQL.search(
                expr
            ), f"Loki LogQL, which VictoriaLogs cannot run: {expr[:80]}"

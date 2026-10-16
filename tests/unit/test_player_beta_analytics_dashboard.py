"""The Umami beta-analytics dashboard's event registry is the player's own.

Its "Event health" panel lists every registered event and flags names that never fired (the first
suspect after a release) or that the registry does not have. That only means something while the
dashboard's list equals ``EVENT_NAMES`` in ``web/learning-player/src/services/analytics.ts``: a new
event added there and not here would show as "off-registry" for ever, and a removed one as
"never fired".
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = (
    ROOT / "config" / "grafana" / "dashboards" / "podcast-player" / "beta-analytics-umami.json"
)
ANALYTICS = ROOT / "web" / "learning-player" / "src" / "services" / "analytics.ts"


def _registry_in_code() -> list[str]:
    block = re.search(
        r"export const EVENT_NAMES\s*=\s*\[(.*?)\]\s*as const", ANALYTICS.read_text(), re.S
    )
    assert block, "EVENT_NAMES not found in analytics.ts"
    return re.findall(r"'([a-z_]+)'", block.group(1))


def _registry_in_dashboard() -> list[str]:
    for panel in json.loads(DASHBOARD.read_text())["panels"]:
        for target in panel.get("targets", []):
            sql = target.get("rawSql", "")
            m = re.search(r"reg\(name\) AS \(VALUES (.*?)\),\s*sess AS", sql, re.S)
            if m:
                return re.findall(r"\('([a-z_]+)'\)", m.group(1))
    raise AssertionError("no event-health panel with a reg(name) VALUES list")


def test_the_dashboard_registry_is_analytics_ts_event_names() -> None:
    code, dash = _registry_in_code(), _registry_in_dashboard()
    assert len(code) >= 40
    assert sorted(dash) == sorted(code), {
        "missing_in_dashboard": sorted(set(code) - set(dash)),
        "not_in_code": sorted(set(dash) - set(code)),
    }


def test_every_umami_panel_reads_umami_pg_and_filters_by_site() -> None:
    for panel in json.loads(DASHBOARD.read_text())["panels"]:
        for target in panel.get("targets", []):
            if "rawSql" in target:
                assert target["datasource"]["uid"] == "umami-pg", panel["title"]
                assert "website_id = '$site'" in target["rawSql"], panel["title"]

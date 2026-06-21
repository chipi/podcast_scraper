"""Current Grafana alerts (Grafana managed Alertmanager v2 API, bearer auth)."""

from __future__ import annotations

from .._http import get_json
from ..config import TargetConfig
from ..result import err, ok

_SOURCE = "grafana.alerts"


def recent_alerts(target: TargetConfig, limit: int = 20) -> dict:
    """Active/recent Grafana alerts (alertname, severity, state, summary)."""
    if not target.grafana_url or not target.grafana_token:
        return err(
            _SOURCE, "grafana not configured (grafana_url + grafana_token)", configured=False
        )
    url = f"{target.grafana_url.rstrip('/')}/api/alertmanager/grafana/api/v2/alerts"
    headers = {"Authorization": f"Bearer {target.grafana_token}"}
    try:
        data = get_json(url, headers=headers, timeout=target.timeout)
    except Exception as exc:  # noqa: BLE001
        return err(_SOURCE, f"GET {url} failed: {exc}")
    raw = data if isinstance(data, list) else []
    alerts = [
        {
            "alertname": (alert.get("labels") or {}).get("alertname"),
            "severity": (alert.get("labels") or {}).get("severity"),
            "state": (alert.get("status") or {}).get("state"),
            "startsAt": alert.get("startsAt"),
            "summary": (alert.get("annotations") or {}).get("summary"),
        }
        for alert in raw[: max(limit, 0)]
    ]
    firing = sum(1 for alert in alerts if alert["state"] == "active")
    return ok(_SOURCE, {"count": len(alerts), "firing": firing, "alerts": alerts})

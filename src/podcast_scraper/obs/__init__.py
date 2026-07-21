"""Vendor-neutral observability emission for podcast_scraper.

The app emits observability data in two open formats and depends on nothing else:

- **Metrics** — Prometheus text exposition (``/metrics``), already pluggable.
- **Events/logs** — canonical JSONL via :func:`podcast_scraper.obs.events.emit_event`.

Shipping is a SEPARATE, swappable layer (a collection agent tails stdout + the
corpus JSONL files and forwards to a backend). Grafana/Loki/VictoriaLogs is the
reference sink, not a coupling — swap the shipper by config, never touch this code.
See ADR-119.
"""

from .events import emit_event, EVENT_SCHEMA

__all__ = ["emit_event", "EVENT_SCHEMA"]

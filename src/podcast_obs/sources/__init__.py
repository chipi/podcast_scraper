"""Probe sources — one module per backing system.

Each function takes a :class:`podcast_obs.config.TargetConfig` and returns a
:func:`podcast_obs.result.ok`/:func:`~podcast_obs.result.err` envelope.

- :mod:`podcast_obs.sources.prod_api` — the deploy's own ``/api`` (health/version/runs).
  Credential-free: needs only ``api_base``, so it works against a local stack first.
- (later slices) ``github`` (deploys), ``grafana`` (cost/alerts), ``sentry`` (errors).
"""

from __future__ import annotations

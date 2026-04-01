# `metrics/` (local only)

This directory holds **dashboard inputs** used for local preview and scripts (`latest-*.json`,
`history-*.jsonl`, `index.html`, optional `dashboard-data.json`). These files are **not tracked in
git**; fresh clones start empty here.

**How to populate**

1. **CI + nightly snapshots from GitHub:** `make fetch-ci-metrics` and `make fetch-nightly-metrics`
   (see [CI & metrics](../docs/ci/METRICS.md)).
2. **Full local preview:** `make build-metrics-dashboard-preview` then `make serve-metrics-dashboard`.

**Canonical public data:** published under GitHub Pages (`metrics/` on the live site), not this folder
on `main`.

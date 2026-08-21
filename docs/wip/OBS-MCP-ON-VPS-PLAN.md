# Observability MCP on the VPS — implementation plan

**Status:** in progress · **Branch:** `feat/obs-mcp-vps` (off `main`) · **Date:** 2026-08-20
**Goal:** run **two** MCP servers on the prod VPS — the existing **content** MCP (Close Listening)
and a new **observability** MCP (`podcast_obs`) — both built in the stack-test publish family,
both deployed to prod, both agent-connectable. Agents connect to the obs MCP to see prod health.

## Corrected current state (verified on prod, 2026-08-20)

- The "MCP on the VPS" is **`player-mcp-1`** = the **stack-api** image in `cli mcp` mode
  (`python -m podcast_scraper.cli mcp --transport http --port 8009`), exposed at
  `mcp.closelistening.app` (RFC-112, in-process bearer auth vs `/internal/mcp/verify`).
- **`podcast-obs` runs nowhere** — no container, image not even pulled. The homelab-deploy idea
  is dropped. So this is about standing obs up **on the VPS** the same way.
- Prod VPS is **amd64** (`uname -m` → x86_64; Hetzner cx43).

## Decisions (operator, 2026-08-20)

- **Subdomain:** `ops.closelistening.app` (A + AAAA added on Cloudflare).
- **Auth:** **Option A** — add in-process bearer-verify to `podcast_obs`, gated to the
  **admin** row, against the **same** `/internal/mcp/verify` seam as the content MCP.
- **New tokens:** deal with them when wired. Grafana SA token → mintable locally on the homelab
  Grafana (operator delegated to me). GitHub Actions:read PAT → operator provides later.

## Backends + tokens (verified via the Alloy layer + homelab tailnet nodes)

The apps OTEL-push; the shipping + backends live in the Alloy stack + homelab. Homelab nodes are
on the tailnet and the VPS already reaches them.

| obs source | homelab endpoint | token | status |
| --- | --- | --- | --- |
| metrics (VictoriaMetrics) | `vm` node | none (tailnet) | reachable |
| logs (VictoriaLogs) | `homelab:9428` (Loki read) | none (tailnet) | reachable |
| errors (GlitchTip) | from DSN origin | `SENTRY_AUTH_TOKEN` | **already a GH secret** |
| alerts (Grafana) | `homelab:3000` (self-hosted) | `PODCAST_OBS_GRAFANA_TOKEN` | **net-new** — mint SA token on homelab Grafana |
| deploys (GitHub) | api.github.com | `PODCAST_OBS_GH_TOKEN` (Actions:read) | net-new, optional (operator) |
| prod API / traces / umami | on the box | reuse | present |

Every source degrades to "not configured" if its token is absent, so obs ships with the
tokenless/existing sources first and gains Grafana/GitHub incrementally. Grafana Cloud is retired
(ADR-117/119); obs reads self-hosted homelab backends over tailscale — no code change (GlitchTip
speaks the Sentry API). Committed config uses MagicDNS short names (`homelab`, `vm`) — never the
tailnet suffix (deny-list gate).

## Work breakdown

**Chunk 1 — build family (CI, safe, DONE this branch)**

- `stack-test.yml`: `podcast-obs` added as the **5th amd64 publish leg** (context `.`,
  `-f docker/observability/Dockerfile`, cache scope `publish-obs`), with the obs smoke
  (`--help` / `summary`) the standalone workflow used to run; `verify-manifests` covers it.
- Retire `.github/workflows/obs-image.yml`.
- `python-app.yml`: add `docker/observability/**` to the trigger paths so obs Dockerfile
  changes still drive the build chain.

**Chunk 2 — prod compose service (safe, not live until deployed)**

- Add an `obs` service (mirror the `mcp` service): `serve --transport http --host 0.0.0.0
  --port 8848`, loopback publish `127.0.0.1:8848`, healthcheck, `PODCAST_OBS_CONFIG` + env
  reusing the box's telemetry vars + the read tokens (staged, added incrementally).

**Chunk 3 — obs auth (Option A, code + tests)**

- Add in-process bearer-verify to `podcast_obs`'s MCP server: verify tokens vs
  `/internal/mcp/verify`, require `role: admin`, 401 otherwise. Public discovery
  (`/.well-known/oauth-protected-resource`) stays un-authed. Unit tests.

**Chunk 4 — config**

- Ship a homelab-pointed `observability.yaml` prod target (api_base, grafana `homelab:3000`,
  loki `homelab:9428`, VictoriaMetrics, GlitchTip via DSN + `SENTRY_AUTH_TOKEN`).

**Chunk 5 — prod deploy + edge (GATED on operator go, per-instance)**

- `obs.caddy` vhost mirroring `mcp.caddy` (expose only `/mcp` + `/.well-known/*` →
  `127.0.0.1:8848`, default-deny, never `/metrics`).
- Revamp the deploy job to bring the `obs` service up + drop `obs.caddy` + restart caddy,
  staging obs secrets via the tmpfs `/dev/shm` pattern. The obs reads are all on the `homelab`
  host (Grafana `homelab:3000`, VictoriaLogs `homelab:9428`), so the single `homelab`
  extra_host already resolves them — no per-node extra_hosts needed.
- Mint the homelab Grafana SA token; wire `SENTRY_AUTH_TOKEN`.
- Register the `ops.closelistening.app` remote MCP with claude.ai.
- **deploy-all fold-in (operator, 2026-08-20):** `deploy-all-prod.yml` currently orchestrates
  three deploys (player, operator, api/pipeline). Fold **individual** deploy paths for the
  observability MCP and the content MCP into it so each is independently deployable AND the
  default "deploy all" brings the whole ecosystem up together. Since both MCPs are services in
  `docker-compose.player-public.yml`, an individual deploy is a scoped `docker compose up -d
  <service>` — expose that as a target/input, and have deploy-all call it alongside the rest.

**Chunk 6 — verify**

- Agent connects to both MCPs; obs `summary` / `health` returns live prod data.

## Not covered / follow-ups

- `docs/guides/OBS_MCP_HOMELAB_DEPLOY.md` is now stale (homelab deploy retired) → rewrite for VPS.
- GitHub Actions:read PAT (deploy-history source) stays parked until the operator provides it.
- Whether the content MCP's verify seam already carries a `role`/admin distinction, or needs one
  added, is confirmed during Chunk 3.

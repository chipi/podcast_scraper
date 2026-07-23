# Handover — homelab-side observability work (o11y reorg, 2026-07-23)

**From:** podcast_scraper infra session (VPS/player launch + observability).
**To:** the agent working on **`agentic-ai-homelab`** + **`homelab-home`**.
**Why you're getting this:** I made changes **live on the homelab** (Grafana,
GlitchTip, Umami, the landing page) but couldn't commit two repos — one needs a
GPG-signed commit I can't produce over SSH, the other is your repo. Plus there are
two follow-ups that belong on your side (they touch `base.alloy` / the cAdvisor
compose). This note is everything you need to finish + persist it.

Homelab = the Mac mini (`homelab`, tailnet `100.87.33.61`). Grafana/VictoriaMetrics/
VictoriaLogs/VictoriaTraces/GlitchTip/Umami all run there. The VPS box
(`prod-podcast`) runs the apps + a node Alloy that ships to the homelab.

---

## 1. NEEDS COMMIT — `agentic-ai-homelab` (staged, GPG commit failed)

Path: `infra/observability/backend/grafana/`. I ran `git add` on these; the commit
died on `krgpg: No such file or directory` (your signing setup isn't in my SSH
session). **Please review + commit** (suggested message at the bottom).

**What changed — the Grafana dashboards are now file-provisioned by folder:**

- `provisioning/dashboards/dashboards.yml` — switched the file provider to
  `foldersFromFilesStructure: true` with `folder: ""` (was `folder: Homelab`,
  `foldersFromFilesStructure: false`). Now **each subdir under
  `grafana/dashboards/` is a Grafana folder.**
- `grafana/dashboards/` reorganised into subdirs:
  - `Homelab/` — the 8 existing boards (host, DGX, GPU, cadvisor, logs,
    orrery-refresh, bugfix-fleet, node-exporter-full). **Moved from the dir root**
    into this subdir (root files would land in "General" with the empty `folder`).
  - `Production Infra/` — `host-overview.json`, `containers.json`,
    `edge-security.json` (generic VPS host/containers/edge — nothing podcast).
  - `Podcast Operator/` — `overview.json` (operator API + viewer).
  - `Podcast Player/` — `overview.json` (consumer player; **source of truth is
    `podcast_scraper-infra/infra/observability/grafana/`** — this is a synced copy
    because Grafana provisioning must read local files).

**Board uids were renamed generic** (no more `vps-podcast-*`): `prod-infra-host-overview`,
`prod-infra-containers`, `prod-infra-edge-security`, `podcast-operator-overview`,
`podcast-player-overview`. Home-page + the podcast repo already reference the new uids.

**Gotcha I hit (so you don't):** Grafana will **not re-home an existing provisioned
dashboard** into a new folder, and won't API-delete a provisioned one (400). To move
them I had to pull the JSON files out, restart Grafana (dashboards drop), then restore
the files (recreated in the right folder). Renaming a uid in-file is the clean path —
new uid = fresh dashboard in the correct subdir folder, old one auto-removed. This is
already done; just commit the result.

## 2. NEEDS COMMIT — `homelab-home`

`~/homelab-home/gen.sh` gained a **third column, "Production · prod-podcast"** (next
to Mac mini + DGX): CPU/mem/disk/load/uptime/container-count from raw
`node_*{instance="prod-podcast"}` + `container_last_seen`, linking to the new
Grafana boards. `www/index.html` was regenerated (`bash gen.sh`). Commit `gen.sh`
(and the regenerated `www/index.html` if you track it).

## 3. Live-only state (FYI — no repo action, but know it exists)

- **GlitchTip**: new project **`player`** (id=5), org `homelab`. DSN is the dashless
  key `5edf9f1c…@telemetry.closelistening.app/5` (in GH secret `PROD_SENTRY_DSN_PLAYER`,
  env `prod`). Created via `manage.py` shell in `glitchtip-web-1`.
- **Umami**: new website **`player`** (`cd384a3e-fd25-476e-affa-08985b73d4da`,
  domain `closelistening.app`), owned by the `admin` user. Inserted into `umami-db`.
- **VPS box** `/opt/vps-observability/config.d/podcast.alloy` — new drop-in labelling
  the operator containers `app=podcast` (they were never collected). Tracked in
  `podcast_scraper-infra` + re-dropped by `deploy-prod` going forward. `player.alloy`
  is the same pattern for the player.

## 4. OPEN follow-ups for you (touch your repos)

- **cAdvisor container names — GH #1272 (option B).** On **Docker 29 (VPS + homelab)**
  cAdvisor emits only cgroup `id`, no container `name` (containerd image store breaks
  its RW-layer lookup — `image/overlayfs/layerdb` doesn't exist). Tested v0.49.1,
  v0.52.1, and `--containerd` — none fix it. So the **Containers** boards (Production
  Infra **and** homelab's `containers-cadvisor`) are empty. Fix = **replace cAdvisor's
  container role with a docker-API-based exporter**, fleet-wide. This edits the
  observability compose in `agentic-ai-homelab` (`cadvisor` service, VPS
  `/opt/vps-observability/docker-compose.yml`) → your call.
- **Player API metrics.** `base.alloy` scrapes `job=api` but the keep-filter is
  `/compose-api-1` only, so **player-api-1 isn't scraped** → the Podcast Player board
  has no API metrics (logs work). Extend the keep-filter (or add a player scrape).
  `base.alloy` lives in your repo.
- **#41 GlitchTip stack assessment** — quotas/retention/DB sizing/celery/upgrade
  posture (6.2.2), exporters, projects hygiene (incl. the new empty `player` project).

## 5. Quick reference

- Grafana admin: `admin` / `MWirwalB1Xb4Tbk5` (from `homelab-home/www/index.html`).
- Folders now: `Alerts, Homelab, Orrery, Podcast Operator, Podcast Player,
  Production Infra` (no dupes — verified).
- Re-provision after editing dashboard files: `docker restart grafana` (config
  changes) or wait 30s (file changes, `updateIntervalSeconds: 30`).
- Suggested commit (agentic-ai-homelab):
  `feat(grafana): folder-structure provisioning — Production Infra / Podcast Operator / Podcast Player`

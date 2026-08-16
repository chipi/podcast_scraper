# INCIDENT-2026-08-05 — prod root disk exhausted by unpruned Docker images

| Field | Value |
| --- | --- |
| Date | 2026-08-05 |
| Duration | ~16:00z–16:32z UTC active response (~30 min); underlying disk fill ran ~7 days |
| Severity | SEV-2 (public surfaces down ~30 min; core pipeline data intact) |
| Affected services | prod-podcast VPS: SSH, the public operator viewer + player app, prod-podcast:443 edge, prod telemetry to Grafana. Core API/pipeline data untouched. |
| Author(s) | operator, agent (Claude Code) |
| Status | final |
| Last updated | 2026-08-05 |

## Summary

The prod VPS root filesystem filled to 98.7% (2.1 GB free of 150 GB) because
every deploy pulls a new `podcast-scraper-stack-api` image and old ones were
never pruned — 68 images / 138.8 GB, only 10 in use (127.9 GB reclaimable). At
~1% free the box could no longer fork/write, taking down sshd; recovery was then
blocked in sequence by a Docker↔tailscale iptables clobber, tmpfs secrets wiped
by a reboot, and a stale Tailscale OAuth credential left behind by the prior
day's key rotation. Resolved by pruning images (reclaimed ~128 GB) and
re-deploying the public stacks pinned to the running version.

## Impact

- **Customer-facing**: yes — the public operator viewer and player learning app
  were down for ~30 min (nginx crash-looping after the reboot until the pinned
  redeploy). The core API/pipeline stayed on the running image throughout.
- **Data lost or corrupted**: no. Only unused Docker images were removed
  (re-pullable). Corpus and Postgres volumes were never touched.
- **Time to detect (TTD)**: ~days. The disk declined ~5 GB/day for a week with no
  effective alert; it was noticed incidentally by an agent reading observability
  at ~1% free.
- **Time to resolve (TTR)**: ~30 min from detection to full restore, but only
  after clearing four stacked secondary failures.
- **Time on incident response**: ~1 operator-hour + agent, wall-clock ~40 min.

---

## Phase 1: Facts (timeline)

| Time (UTC) | Event | Source |
| --- | --- | --- |
| ~7 days prior | Root `/` free = 36.7 GB, declining ~5 GB/day as deploys accumulate images | VictoriaMetrics `node_filesystem_avail_bytes{instance="prod-podcast"}` |
| 2026-08-04 | Tailscale OAuth rotation performed; infra/terraform + DR-drill path updated, runtime `TS_OAUTH_*` secret NOT updated (left at 2026-07-25) | operator; `gh secret list` |
| ~1 day prior | `/` free = 5.4 GB | VM offset query |
| ~16:00z | Agent flags via observability: `/` at 1.3% free (2.1 GB); sshd unreachable | observability review |
| ~16:0xz | Verified against VM (1.3% free, `predict_linear` → empty in ~3–5 h). SSH to prod `:22` = connection refused (sshd could not operate on the full disk) | agent |
| ~16:0xz | Hetzner Cloud console opened; box is SSH-key-only (no console password), so operator **rebooted** to regain a shell | operator |
| post-reboot | Reboot released held-deleted files: `/` reads 94.6% at boot (from 98.7%). All app containers auto-restart, but inbound tailnet TCP (`:22`, `:443`) all **refused** while pings still answer | agent, `df`, `ss` |
| ~16:1xz | Diagnosed: dockerd churn during the crisis clobbered tailscale's `INPUT` accept rule for `tailscale0`. `iptables -I INPUT -i tailscale0 -j ACCEPT` restores inbound TCP; SSH regained | agent |
| ~16:1xz | `docker image prune -af` + `builder prune` (operator, console) → `/` = 15% used, 124 GB free. Root cause confirmed: Images 68 total / 10 active / 127.9 GB reclaimable | `docker system df` |
| ~16:2xz | Found the public `operator` + `player` stacks crash-looping: their own `api` services never came back — the app secrets live in tmpfs `/dev/shm/*-secrets/` (RAM-only by design) and were **wiped by the reboot** | agent, container logs |
| ~16:2xz | In-place restore (pinned to running `sha-4e8b1a7`, no upgrade) via `deploy-operator.yml`/`deploy-player.yml` **fails**: runner cannot join the tailnet — `oauth2: cannot fetch token: 401 Unauthorized` | GH Actions run 31023660438 |
| ~16:2xz | Root-caused to the stale runtime `TS_OAUTH_*` secret (unchanged by the 08-04 rotation). New OAuth client provided; `TS_OAUTH_CLIENT_ID`/`TS_OAUTH_SECRET` updated | operator + agent, `gh secret set` |
| 16:27→16:30z | Pinned deploys re-run and **succeed**; `operator-api`, `operator-viewer`, `player-api`, `player-learning-app` all healthy; public surface serves 200 via `prod-podcast.<TAILNET>.ts.net` | GH runs 31025348685 / 31025351273 |
| ~16:31z | Telemetry shipper (`alloy` in `vps-observability`) found down (only cadvisor was up); restarted → remote-writing to VM again | agent |
| ~16:32z | Full recovery: 11/11 containers healthy, disk 120 GB free, public API 200, telemetry recovering | agent |

---

## Phase 2: Analysis

### Root cause

**Unbounded Docker image accumulation on a deploy-heavy box.** Each deploy pulls a
new `ghcr.io/chipi/podcast-scraper-stack-api:sha-…` image; nothing prunes the
previous ones. Over continuous deploys this reached **68 images / 138.8 GB, only
10 referenced by a container** — 127.9 GB of dead layers on a 150 GB root disk,
driving it to 98.7% full. At ~1% free the kernel/systemd could no longer write
runtime state and sshd fell over.

### Contributing factors

- **No image pruning** in the deploy path or as a scheduled job.
- **Docker↔tailscale iptables clash**: dockerd churn/pressure during the crisis
  reset iptables and dropped tailscale's `INPUT -i tailscale0 -j ACCEPT` rule, so
  every inbound tailnet TCP port (SSH, the `:443` edge) was refused while the
  control-plane (ping) still worked — this masked the real state and cost the most
  recovery time.
- **Stale runtime Tailscale OAuth**: the 08-04 rotation updated the
  infra/terraform (`TS_INFRA_OAUTH_*`) and the DR-drill was tested, but the
  runtime `TS_OAUTH_*` secret (used by every deploy and DR-drill *runtime* job)
  was left at its 2026-07-25 value → 401 → blocked the recovery deploy.
- **tmpfs secrets, no boot re-stage**: app secrets are delivered to
  `/dev/shm/*-secrets/` (RAM-only, never on disk — a deliberate security choice),
  so a reboot wipes them and the app cannot self-recover without a redeploy.
- **Key-only SSH, no console password**: the only route to a shell during the
  outage was a full reboot.

### Why detection took as long as it did

The disk fell ~5 GB/day for a week — a slow, monotonic signal that was present in
VictoriaMetrics but not backed by an effective, routed alert. It was caught
incidentally by an agent reading dashboards at ~1% free, i.e. hours from
hard-failure, not days earlier when it was cheap to fix.

### Why recovery took as long as it did

Four secondary failures stacked in front of the obvious fix (prune images), each
needing separate diagnosis: (1) sshd down from the full disk; (2) after the
reboot, all inbound tailnet TCP refused (the iptables clobber); (3) the reboot
wiped the app's tmpfs secrets so the public stacks couldn't restart; (4) the
pinned redeploy was blocked by the previous day's stale OAuth. Any one alone is
minutes; stacked, they turned a one-command cleanup into a ~30-minute chase.

### Counterfactuals (what didn't break that could have)

- The reboot released held-deleted files (98.7% → 94.6%), buying just enough slack
  to keep the box alive and reachable via console.
- Only images were purged — corpus and Postgres volumes were never at risk; **no
  data loss**.
- The operator had a valid replacement OAuth client secret on hand; had it needed
  recreating from scratch mid-incident, recovery would have been longer.
- Pinning the redeploy to `sha-4e8b1a7` avoided an unintended prod upgrade during
  an incident.

---

## Phase 3: Improvement plan

> Tracking left as TBD — to be filed as local tasks / issues per the operator's
> follow-up process, not opened unilaterally.

### Prevention (would have stopped this happening)

| Item | Tracking | Owner | Target |
| --- | --- | --- | --- |
| Prune images in the deploy path (`docker image prune -af` post-deploy) or a daily systemd timer | TBD | operator/agent | next |
| Make the `tailscale0` `INPUT` accept rule persistent (survive dockerd churn + reboot) | TBD | operator/agent | next |

### Detection (would have surfaced the problem sooner)

| Item | Tracking | Owner | Target |
| --- | --- | --- | --- |
| Disk-space alert `node_filesystem_avail_bytes / size < 15%` on prod, routed to a **real** destination (not a placeholder contact point) | TBD | operator/agent | next |

### Mitigation (would have reduced impact / recovery time)

| Item | Tracking | Owner | Target |
| --- | --- | --- | --- |
| Boot-time secret re-stage, or document that a reboot requires a redeploy (app cannot self-recover from tmpfs wipe) | TBD | operator | soon |
| Break-glass console credential so a shell doesn't require a reboot on the key-only box | TBD | operator | soon |

### Process (would have changed how we respond)

| Item | Tracking | Owner | Target |
| --- | --- | --- | --- |
| Tailscale OAuth rotation runbook: rotate + update **both** `TS_OAUTH_*` (runtime/deploy/drill) and `TS_INFRA_OAUTH_*` (terraform), and test the **deploy** path, not only the DR drill | TBD | operator | soon |

---

## What went well

- Metrics gave an accurate, quantified early picture: exact % free, ~5 GB/day
  fill rate, and a `predict_linear` time-to-full — no guesswork on severity.
- The `docker image prune -af` reclaimed ~128 GB cleanly with zero data loss.
- Pinning the redeploy to the running `sha-4e8b1a7` restored the exact
  pre-incident version — recovery, not an accidental release.
- The operator's hypothesis ("could this be yesterday's tailscale rotation?")
  was correct and cut straight to the OAuth root cause.

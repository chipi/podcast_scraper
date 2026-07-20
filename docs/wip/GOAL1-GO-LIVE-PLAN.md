# Goal-1 go-live plan — orrery on the shared common edge

**Goal:** take orrery live on the minimal shared public edge (ADR-114), with the
box hardened, observed, and Cloudflare-ready — **without losing any data**.

**Owner key:** 🧑 You (operator-only: infra apply, DNS, accounts, secrets) ·
🤖 Me (I prepare/run/verify what's safe from here) · 🤝 Both (I drive, you
authorize + we verify together).

---

## Where we are now

Everything for Goal-1 is **built and on `production`** but **not yet applied to
the running box**. Cloud-init runs *once* at first boot — the prod box booted
long before this config existed, so the edge/hardening/o11y/CF prep lives in the
repo but isn't live yet. Go-live = converge the running box to that config, then
open the door.

| Built (in repo) | Live on the box? |
|---|---|
| Caddy edge + `(hardened)` snippet + catch-all (ADR-114) | ❌ not applied |
| Host hardening: SSH key-only, fail2ban (sshd + caddy jails), metadata-egress guard | ❌ not applied |
| Firewall 80/443 rules (TF) | ❌ not applied |
| o11y common plane → **self-hosted DGX stack** (VictoriaMetrics/Logs/Traces + Grafana), not Grafana Cloud | ✅ **collector live** — box ships metrics+logs+security-logs (`up{instance=prod-podcast}` present) |
| Cloudflare prep: real-IP trust, CF-safe fail2ban, `cloudflare_origin_lock` (ADR-118) | ❌ not applied |

---

## Decisions I need from you before we touch prod

1. **Apply path** — how the built config reaches the running box:
   - **A. Imperative-once (recommended).** I write an idempotent `apply-edge.sh`
     that converges the *live* box to the cloud-init state (install Caddy, drop
     the Caddyfile, fail2ban jails, hardening drop-ins, Alloy). **No rebuild, no
     data-loss risk.** Future rebuilds still get it via cloud-init automatically.
   - **B. Rebuild.** Destroy + recreate so cloud-init runs fresh. **Only safe if
     the corpus is on the delete-protected volume AND a verified restore exists**
     (see precondition 2). Heavier; real data-loss risk if precondition 2 fails.
   - **C. Blue-green.** New box, migrate, cut over DNS. Most work; best for a
     clean cutover but overkill for adding an edge to a healthy box.

   → **My recommendation: A.** We're *adding an edge* to a healthy box, not
   changing its identity. B/C only earn their cost if we also want a fresh OS.

2. **Corpus location (the data-loss lever).** `volume_size_gb` isn't in the repo
   (set in CI/secret; default `0` = corpus on the **boot disk**). **Confirm which
   it is.** On the boot disk, *any* rebuild (path B) destroys the corpus unless
   backed-up-and-restored first. On the delete-protected volume, it survives.
   Path A sidesteps this entirely.

3. **Sequencing with your parallel work.** We re-rebase `production` onto `main`
   at the point you pick, before the firewall opens. Say when.

---

## The plan

### Phase 0 — Safety net (before anything touches prod)

| # | Step | Owner | Rollback / guard |
|---|---|---|---|
| 0.1 | Fresh corpus backup — run `backup-corpus-prod.yml` | 🤝 | n/a (additive) |
| 0.2 | **Verified restore** — run `verify-backup-restore.yml`, confirm green | 🤝 | this IS the guard |
| 0.3 | Confirm corpus location (precondition 2) + delete/rebuild protection still ON | 🤝 | — |
| 0.4 | (Belt-and-suspenders) Hetzner snapshot of the box | 🧑 | instant restore point |

**Gate:** do not proceed until 0.2 is green and 0.3 is answered.

### Phase 1 — Converge the box (Option A), firewall still CLOSED

| # | Step | Owner | Rollback |
|---|---|---|---|
| 1.1 | ✅ **DONE** — `scripts/ops/apply-edge.sh` (idempotent, `--dry-run`, mirrors cloud-init edge/hardening/o11y; firewall + corpus explicitly out of scope). Folds in 3.3. | 🤖 | — |
| 1.2 | Run `sudo apply-edge.sh --dry-run` then a real run on the box over tailnet (Caddy+hardening+Alloy+jails) | 🤝 | `systemctl stop caddy`; rm config |
| 1.3 | Verify **over tailnet** with `sudo scripts/ops/verify-edge.sh` (read-only): `caddy validate` ok, engine up, both fail2ban jails active, metadata DROP present, Alloy shipping-or-staged | 🤝 | still tailnet-only — no public exposure yet |

At the end of Phase 1 the edge is running but the **firewall is still closed** →
the box is still tailnet-only. Nothing is public. This is the safe rehearsal.

### Phase 2 — Observability live ✅ (self-hosted, superseded Grafana Cloud)

Migrated off Grafana Cloud to the **self-hosted DGX stack** (ADR-119): the
prod-podcast Alloy collector ships host metrics + pipeline/corpus logs + sshd/
fail2ban/caddy security-logs to VictoriaMetrics/VictoriaLogs; Grafana has the
`VPS — Podcast` dashboards; alert rules + generic (add-later) Slack/email/GlitchTip
channels are provisioned as code (homelab `backend/grafana/provisioning/alerting/`).

| # | Step | Owner | Status |
|---|---|---|---|
| 2.1 | Self-hosted backend (VM/VL/VT + Grafana) on the DGX, tailnet-only | 🤖 | ✅ live |
| 2.2 | prod-podcast Alloy collector → self-hosted; dashboards provisioned | 🤖 | ✅ live |
| 2.3 | Confirm host metrics + security logs arrive; alert rules armed | 🤝 | ✅ verified (rules health=ok) |
| 2.4 | Wire a real alert channel (Slack webhook) + prove one end-to-end fire | 🧑 | ⏳ channel scaffolded, not yet wired |

### Phase 3 — Pre-public gate (hard gate)

| # | Step | Owner |
|---|---|---|
| 3.1 | Walk the [THREAT_MODEL pre-public gate](../security/THREAT_MODEL.md#pre-public-gate--run-before-any-new-public-vhost) checklist — **pre-walked** in [GOAL1-PHASE3-PREP.md](GOAL1-PHASE3-PREP.md) (substrate items done; open: orrery cap_drop / digest-pin / rollback rehearsal) | 🤝 |
| 3.2 | Sign off: orrery clears it (static, no `docker.sock`, no keys) | 🧑 |
| 3.3 | ✅ **Folded into `apply-edge.sh`** (1.1) — the metadata-egress guard (script + unit + `enable --now` + live `iptables -C DOCKER-USER` re-assert) is applied by `apply-edge.sh §3` and asserted by `verify-edge.sh §3`. Nothing separate to run; converges when 1.2 runs. (review 2026-07-17 H8 / T-07) | 🤝 |
| 3.4 | **Complete the ADR-115 secrets cutover** (T-08) — 6 LLM keys still plaintext container env; steps in [GOAL1-PHASE3-PREP.md](GOAL1-PHASE3-PREP.md) §3.4 (age key → `prod.enc.yaml` → decrypt on box → add `-f docker-compose.secrets.yml` → drop keys from `.env`). Blocks public exposure of the **api** (not orrery). (review M26 / #1162) | 🧑 |

**No firewall opens until 3.2 is signed off.**

### Phase 4 — Open the firewall (the exposure moment)

| # | Step | Owner | Rollback |
|---|---|---|---|
| 4.1 | ✅ **Static pre-review done** — see [PHASE4-FIREWALL-PREREVIEW](GOAL1-PHASE4-FIREWALL-PREREVIEW.md): the `:80`/`:443` rules are an in-place `~` update of `hcloud_firewall.main`; `hcloud_server.prod` is protected by `lifecycle { ignore_changes = [user_data, ssh_keys] }` + a stable `firewall_ids` id ref, so it must show **no change**. Still confirm the live `tofu plan` matches before 4.2; **abort if the server shows `-/+`**. | 🤖 | abort if replace |
| 4.2 | `tofu apply` the 80/443 rules | 🧑 | remove rules + apply (<2 min) |
| 4.3 | Confirm :443 reachable from outside; box otherwise still locked down | 🤝 | 4.2 rollback |

### Phase 5 — Orrery onboarding (coordinated #1158 ↔ orrery#381)

| # | Step | Owner | Rollback |
|---|---|---|---|
| 5.1 | Deploy orrery container on `:8090` | 🧑/orrery | stop container |
| 5.2 | `scp orrery.caddy` → `/etc/caddy/sites/`; reload caddy (narrow sudoers) | 🤝 | `rm` vhost + reload (frees port, nothing shared changes) |
| 5.3 | DNS A/AAAA for orrery's domain → box public IP | 🧑 | revert DNS |
| 5.4 | ACME issues the cert; verify `https://<orrery-domain>` loads | 🤝 | — |

### Phase 6 — Cloudflare (optional, after Phase 5 green — ADR-118 rollout)

Follow **ADR-118 §Operator rollout** verbatim: orange-cloud DNS → Full(strict) →
verify through CF (real `client_ip` in logs) → **only then** flip
`cloudflare_origin_lock=true` + apply. Rollback: grey-cloud, or lock=false.

### Phase 7 — Post-live

- Re-rebase `production` onto `main`.
- ✅ Alert thresholds tuned against live baselines (target-down excludes GPU
  mode-swap services; disk 10%/5% tiers; 5xx 0.01/s; ssh+fail2ban re-homed to
  LogsQL). Remaining: wire the real Slack/email/GlitchTip channel values (2.4).
- **T-12**: ramp HSTS to `max-age=31536000; includeSubDomains; preload` once
  HTTPS is proven on the live vhost; verify orrery's CSP.
- Watch o11y for a few days; capture learnings for Goals 2 (player) / 3 (operator UI).

### Phase 8 — Audio archive transition (#1199)

Move audio to the Storage Box; **gated on DR-drill green + corpus backup verified**.
Touches only the podcast_scraper tenant's storage, not the shared edge. Full
sequence + data-at-risk analysis + new-command runbook:
[GOAL1-AUDIO-ARCHIVE-ROLLOUT.md](GOAL1-AUDIO-ARCHIVE-ROLLOUT.md). Short form:
provision Storage Box (TF, plan-reviewed) → verify rclone → backfill + integrity
check → flip `audio_storage_backend=remote` → e2e (`archive pull` / reprocess-prod)
→ decide archive backup policy → prune local.

---

## Data-loss guards (invariants for the whole run)

- **Backup + verified restore before prod is touched** (Phase 0.1–0.2).
- **Delete/rebuild protection stays ON** — never flipped without a per-instance ask.
- **Option A avoids rebuild entirely** — the corpus is never at risk.
- **SSH is tailnet-only** — no edge/firewall misstep can lock us out of the box.
- Every public-facing phase (4, 5, 6) has a **<5-min rollback**.

## Status update (2026-07-20)

- **Order confirmed: orrery goes public first** (pilot tenant); the podcast **api**
  follows and additionally needs the 3.4 secrets cutover before it can be exposed.
- **DR-drill blocker cleared** — `verify-backup-restore` is green (run 2026-07-19).
  The earlier SIGSEGV pause no longer applies; Phase 0.2 can pass.
- **Observability done on self-hosted rails** (Phase 2 above) — replaced the
  Grafana Cloud plan; box is shipping and alert rules verify healthy.
- **Still the gate:** Phase 0 (operator-run backup + verified restore + corpus-
  location confirm + snapshot) before anything touches the edge. Then Phase 1
  (`apply-edge.sh` on the box, tailnet, firewall closed).

## Prep status (2026-07-18)

Done (on `production`, unpushed): ✅ 1.1 `apply-edge.sh` + 1.3 `verify-edge.sh`
(real-run-tested in a systemd container — found+fixed 2 bugs); ✅ 4.1 firewall
pre-review; ✅ 3.1 gate pre-walk + 3.4 secrets steps
([GOAL1-PHASE3-PREP.md](GOAL1-PHASE3-PREP.md)); ✅ Phase 8 audio-archive rollout
([GOAL1-AUDIO-ARCHIVE-ROLLOUT.md](GOAL1-AUDIO-ARCHIVE-ROLLOUT.md)); ✅ Caddyfile
validated+fmt, TF `validate` green, obs-sync (3 T-11 rules) parses.

### Phase 0 — one-paste command sequence (operator runs; I can't trigger prod infra)

```bash
# 0.1 fresh corpus backup
gh workflow run backup-corpus-prod.yml
# 0.2 verified restore (THE guard — must be green before proceeding)
gh workflow run verify-backup-restore.yml
gh run watch "$(gh run list --workflow verify-backup-restore.yml -L1 --json databaseId --jq '.[0].databaseId')"
# 0.3 confirm corpus location: volume vs boot disk (data-loss lever)
#     check TF var volume_size_gb (>0 = delete-protected volume; 0 = boot disk)
# 0.4 (belt-and-suspenders) take a Hetzner snapshot of the box in the console
```

**Gate:** do not start Phase 1 until 0.2 is green and 0.3 is answered.

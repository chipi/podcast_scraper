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
| o11y common plane: Alloy metrics+logs, T-11 rules, `make obs-sync` (ADR-117) | ❌ not applied |
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
| 1.1 | I write idempotent `apply-edge.sh` (mirrors cloud-init runcmd; safe re-run) | 🤖 | — |
| 1.2 | Run it on the box over tailnet (Caddy+hardening+Alloy+jails) | 🤝 | `systemctl stop caddy`; rm config |
| 1.3 | Verify **over tailnet**: `caddy validate` ok, engine up, fail2ban jails active, Alloy shipping | 🤝 | still tailnet-only — no public exposure yet |

At the end of Phase 1 the edge is running but the **firewall is still closed** →
the box is still tailnet-only. Nothing is public. This is the safe rehearsal.

### Phase 2 — Observability live

| # | Step | Owner |
|---|---|---|
| 2.1 | Set Grafana Cloud creds (metrics + Loki) as TF vars / GH secrets | 🧑 |
| 2.2 | `make obs-sync --apply` (dashboards + T-11 rules) | 🤝 |
| 2.3 | Confirm host metrics + security logs arrive; T-11 alerts armed | 🤝 |

### Phase 3 — Pre-public gate (hard gate)

| # | Step | Owner |
|---|---|---|
| 3.1 | Walk the [THREAT_MODEL pre-public gate](../security/THREAT_MODEL.md#pre-public-gate--run-before-any-new-public-vhost) checklist | 🤝 |
| 3.2 | Sign off: orrery clears it (static, no `docker.sock`, no keys) | 🧑 |

**No firewall opens until 3.2 is signed off.**

### Phase 4 — Open the firewall (the exposure moment)

| # | Step | Owner | Rollback |
|---|---|---|---|
| 4.1 | I pre-review `tofu plan` — MUST be in-place `~`, **not** a server `-/+` replace | 🤖 | abort if replace |
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
- Tune T-11 alert thresholds on real traffic; wire contact points.
- **T-12**: ramp HSTS to `max-age=31536000; includeSubDomains; preload` once
  HTTPS is proven on the live vhost; verify orrery's CSP.
- Watch o11y for a few days; capture learnings for Goals 2 (player) / 3 (operator UI).

---

## Data-loss guards (invariants for the whole run)

- **Backup + verified restore before prod is touched** (Phase 0.1–0.2).
- **Delete/rebuild protection stays ON** — never flipped without a per-instance ask.
- **Option A avoids rebuild entirely** — the corpus is never at risk.
- **SSH is tailnet-only** — no edge/firewall misstep can lock us out of the box.
- Every public-facing phase (4, 5, 6) has a **<5-min rollback**.

## My immediate to-dos (once you pick the apply path)

1. Write `apply-edge.sh` (Phase 1.1) — idempotent, mirrors cloud-init, dry-run mode.
2. Pre-stage the `tofu plan` review checklist for Phase 4.1.
3. Draft the Phase-0 backup/verify command sequence so it's one paste.

# Infra security hardening — sequenced execution plan

- **Status**: Active (execution plan; tracked in [#1160](https://github.com/chipi/podcast_scraper/issues/1160))
- **Date**: 2026-07-08
- **SSOT**: [`docs/security/THREAT_MODEL.md`](../security/THREAT_MODEL.md) — the T-01…T-12
  register this plan executes against. Flip a threat's status to `mitigated` there
  as each item lands here.
- **Decision context**: [ADR-114](../adr/ADR-114-shared-multi-tenant-public-edge-caddy.md)
  (shared public edge). **Harden the box before we widen its exposure.**

## Sequencing rule

Infra-first, from the substrate up: **Docker/orchestration → Host → Edge → App**.
The crown jewels (LLM keys, corpus, host root) sit in the Docker/host layers and are
independent of the shared edge — so they land first. Edge items land *with* the Caddy
engine; the player-API item gates the player going public.

Each phase must be green (verified) before the next public-exposure step proceeds. No
new public vhost ships until the [pre-public gate](../security/THREAT_MODEL.md#pre-public-gate-run-before-any-new-public-vhost) passes.

---

## Phase 1 — Docker / orchestration hardening (do first) 🔴

Current state: containers have **zero** hardening (no `cap_drop`/`security_opt`/
`read_only`/non-root anywhere); `api` mounts `docker.sock` RW + holds all 6 provider keys.

- [ ] **T-01a** Keep the operator `api` service **tailnet-private forever** — never a
      public Caddy vhost. Document as a hard constraint (it holds `docker.sock` + keys
      *by design*, for job-spawn).
- [ ] **T-01b** Public consumer surfaces (podcast player API, public gi/kg read) run as a
      **separate least-privilege service**: no `docker.sock`, no write-scope provider keys
      (D6 = no request-time LLM, so none needed). Confirms `SERVER-SIDE-GAP-ANALYSIS.md`.
- [ ] **T-01c** For job-mode Docker access, replace the raw RW `docker.sock` mount with a
      **`docker-socket-proxy`** allowlisting only the compose calls the api makes.
      *Verify:* api can still spawn `pipeline-llm`; proxy denies `container delete`/`exec`.
- [ ] **T-04** Add container hardening to every service in `compose/*.prod*.yml`:
      `cap_drop: [ALL]`, `security_opt: [no-new-privileges:true]`, non-root `user:`,
      `read_only: true` + explicit `tmpfs` where writable dirs are needed.
      *Verify:* stack-test green; each service boots + passes health.
- [ ] **T-08** Secret blast radius: least-key per stage (only the keys a stage calls),
      per-env keys (prod ≠ pre-prod), **provider billing caps + spend alerts** on every
      key, rotation runbook cross-check. *Verify:* remove unused keys from a stage's env,
      confirm it still runs.
- [ ] **T-09** Pin GHCR images by **digest** (or cosign-verify at deploy) instead of the
      mutable `:main`/`:sha` tag in `deploy.sh`. Pin the Caddy apt version.
      *Verify:* deploy resolves the pinned digest; a tag repoint does not change what runs.

## Phase 2 — Host hardening 🟠

Landed in `infra/cloud-init/prod.user-data`. **⚠ cloud-init is first-boot-only and
`hcloud_server` ignores `user_data` drift → these do NOT touch the live VPS.** They
apply on the next rebuild; the **live box needs the same steps applied imperatively
once** (tailscale-serve precedent). Cloud-init YAML validated structurally.

- [x] **SSH hygiene** — `sshd_config.d/99-hardening.conf`: `PermitRootLogin
      prohibit-password` (keeps root **key** emergency access; denies password),
      `PasswordAuthentication no`, `KbdInteractiveAuthentication no`. runcmd
      `sshd -t && systemctl reload ssh`.
- [x] **fail2ban** — added to `packages`; `jail.d/sshd.local` (systemd backend,
      maxretry 5 / findtime 10m / bantime 1h); `systemctl enable --now fail2ban`.
      Caddy access-log jail deferred until the edge lands.
- [x] **T-07 (metadata slice)** — `block-metadata-egress.service` (oneshot,
      `After=docker`) inserts a `DOCKER-USER -d 169.254.169.254 -j DROP` rule
      (idempotent, reboot-safe), killing SSRF→cloud-creds from any container.
      *Live verify (post-apply):* container `curl 169.254.169.254` times out.
      Caddy `deploy` writable-sites-dir constraint + vhost review deferred to the edge.
- [ ] **T-10 (full egress allowlist)** — DEFERRED (design). A blanket egress allowlist
      on api/pipeline risks breaking GHCR / provider-API / PyPI / HF pulls; needs a
      scoped destination list first. Metadata block (above) is the high-value slice done.
- [ ] **T-11** Security detection/alerting via Grafana/Alloy + Sentry — DEFERRED
      (observability config, not cloud-init): alerts on auth-fail spikes, 4xx/5xx
      anomalies, new listeners, fail2ban bans, egress anomalies.

## Phase 3 — Edge hardening (lands with the Caddy engine) 🟠

Engine built in `infra/cloud-init/Caddyfile` + `prod.user-data` + firewall in TF
(#1158 / ADR-114). `caddy validate` green. **Applies on rebuild; live box gets the
engine via imperative-once, and the firewall 80/443 rule needs a gated `tofu apply`
(verify in-place `~`, no server replace).**

- [x] **T-02** `admin off` in the base Caddyfile. *Verify (post-apply):* `curl localhost:2019` fails.
- [x] **T-03** On-demand TLS OFF (default) + explicit `:443` catch-all (`tls internal` +
      `abort`) for unknown Host. Validated via `caddy validate`.
- [ ] **T-05** Evaluate **Cloudflare free** as an additive WAF/DDoS/rate-limit front (D4,
      still open) — zero origin change (ADR-114 designed for it).
- [x] **T-12 (partial)** Shared `(hardened)` snippet: HSTS `max-age=86400` + compression +
      access log; tenants `import hardened`. Ramp HSTS to 1y + preload after the first live
      vhost; confirm per-app CSP; Caddy TLS defaults are strong.
- [ ] **fail2ban Caddy jail** — deferred until there's an authed surface to protect (lands
      with the consumer API / T-06); a jail on a static reverse-proxy log has little to bite.

## Phase 4 — Platform / player app (gates player going public) 🟠

- [ ] **T-06** The consumer/player API requires **authN/authZ**, tight **CORS** allowlist,
      **per-token rate limiting**, and input validation before it is exposed. Scope the
      auth model (tokens vs. OAuth) with the player EPIC work. *Verify:* unauth request is
      401; cross-origin from a non-allowlisted origin is blocked.

---

## Definition of done

- Every T-01…T-12 row in `THREAT_MODEL.md` is `mitigated` or explicitly `accepted`
  (with rationale in the revision log).
- The pre-public gate checklist passes for the first tenant that goes public.
- No regression to the "controls already in place" list in the threat model.

---

## Appendix A — Phase 1 engineering detail (grounded 2026-07-08)

Grounded in the actual code, not the abstract list. All Phase-1 container changes land
in `compose/docker-compose.prod.yml` (the prod/pre-prod overlay) so the base
`stack.yml` stays portable for dev/CI. **Verification gate for the caps/read-only set:
`make stack-test-up` boot + `/api/health` green** — caps can break container boot, so
that boot test is mandatory before the caps commit.

### T-01 reframe — what actually mitigates it

The api shells out to the **`docker compose` CLI** (`server/pipeline_docker_factory.py:253`,
`asyncio.create_subprocess_exec(["docker","compose",…])`), so it needs container
**create** rights — the single most dangerous Docker capability. Therefore:

- **Primary control (T-01a/b):** the operator `api` (holds `docker.sock` + keys **by
  design**, for job-spawn) **stays tailnet-private forever**. Public surfaces are a
  **separate service with no `docker.sock` and no write-scope keys** (D6 = no
  request-time LLM → none needed). This is the real fix and it's architectural.
- **T-01c (docker-socket-proxy): downgraded to optional / low-ROI.** Because
  `docker compose run` inherently needs container-create, a proxy barely shrinks blast
  radius, and it is a **new dependency** (`tecnativa/docker-socket-proxy` image) →
  needs explicit dep approval. Recommend **accept** the RW socket on the *private* api
  and rely on T-01a, unless we later split job-spawn to a dedicated broker.

### Per-service container hardening matrix (T-04)

`no-new-privileges:true` is universally safe (blocks privilege *escalation* only; none
of these need it) → applied to all services immediately. Caps + `read_only` are
per-service and boot-gated.

| Service | Base | `cap_drop`/`cap_add` | `read_only` | Notes |
| --- | --- | --- | --- | --- |
| `viewer` | nginx:1.27-alpine, listens `:80` | drop ALL; add `NET_BIND_SERVICE, CHOWN, SETUID, SETGID, DAC_OVERRIDE` | **yes** + tmpfs `/var/cache/nginx`, `/var/run`, `/tmp`, **`/etc/nginx/conf.d`** | ✅ boot-verified. conf.d tmpfs required — the image renders its config from a template at runtime |
| `api` | non-root `podcast`; entrypoint starts root → chowns `/app/output`, mirrors `docker.sock` GID, drops via `su-exec` | drop ALL; add `CHOWN, DAC_OVERRIDE, FOWNER, SETUID, SETGID` | **no** | ✅ boot-verified (`/api/health` 200). writes HF_HOME, LanceDB locks, output; keeps RW `docker.sock` (T-01a) |
| `pipeline-llm` | non-root `podcast`, one-shot | drop ALL; add `CHOWN, DAC_OVERRIDE, FOWNER, SETUID, SETGID` | **no** | parity with `api` (shared entrypoint pattern); exercised on next pipeline run |
| `grafana-agent` | grafana/agent, reads `docker.sock` **RO** | **no `cap_drop`** — `no-new-privileges` only | no | ⚠️ `cap_drop: ALL` boot-FAILED (`exec … operation not permitted`; binary has file-caps). Left at no-new-privileges |

Boot-verified 2026-07-08 against the local `:latest` images: two-service `api`+`viewer`
compose under the exact caps → `/` and `/api/health` both 200; grafana-agent `cap_drop`
tested and rejected. `su-exec` drop + `no-new-privileges` are compatible (drop is
root→lower, not a setuid escalation).

### T-09 image pinning

`deploy.sh`/overlay pull `ghcr.io/chipi/podcast-scraper-stack-*:${PODCAST_IMAGE_TAG:-main}`
(mutable). Move prod to **digest pins** (`…@sha256:…`) resolved from the Stack-test-green
build, or cosign-verify at deploy. Pin the Caddy apt version when the edge lands.

### T-08 secret least-key / billing

- **Least-key (code):** the api env injects all 6 provider keys into every spawned
  pipeline; scope to only the provider a profile uses. Verify by removing an unused
  key and re-running.
- **Billing caps + alerts (external, provider consoles):** set per-key spend caps +
  alerts — not a repo change, tracked here so it isn't forgotten.

### Execution order within Phase 1

1. `no-new-privileges` on all prod services (safe, now). ✅ this batch
2. Caps + `read_only` matrix above → **boot-gate with `make stack-test-up`** → commit.
3. T-09 digest pinning.
4. T-08 least-key (code) + billing caps (operator, consoles).
5. T-01c decision: accept RW socket on private api (recommended) vs. socket-proxy dep.
